#!/usr/bin/env python3
"""A股V2 Portfolio Backtest — 优化版"""
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
t0 = time.time()

# 加载数据
h = pd.read_parquet('data/a_hist_10y.parquet')
h = h.rename(columns={'Code':'sym','Date':'date','O':'open','H':'high','L':'low','C':'close','V':'volume'})
h['date'] = pd.to_datetime(h['date'].astype(str), format='%Y%m%d')
m = pd.read_parquet('data/moneyflow_core.parquet')
m['sym'] = m['ts_code'].str.replace(r'\.\w+$', '', regex=True)
m['date'] = pd.to_datetime(m['trade_date'].astype(str), format='%Y%m%d')
for c in ['sm','md','lg','elg']:
    m[f'{c}_net'] = m[f'buy_{c}_amount'] - m[f'sell_{c}_amount']
m['total_net'] = m['net_mf_amount']
m = m[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']].drop_duplicates(['sym','date'])
df = pd.merge(h, m, on=['sym','date']).sort_values(['sym','date']).reset_index(drop=True)
df = df[df['close'] > 0]
df['circ_mv'] = df['volume'] * df['close']
df['turnover_rate'] = 0
print(f'Data: {len(df):,} ({time.time()-t0:.0f}s)')

# 特征工程
syms = df['sym'].values; close = df['close'].values
high_arr = df['high'].values; low_arr = df['low'].values; vol = df['volume'].values
tr_rate = df['turnover_rate'].values; cmv = df['circ_mv'].values; n = len(df)
sc = np.where(syms[1:] != syms[:-1])[0] + 1
starts = np.concatenate([[0], sc]); ends = np.concatenate([sc, [n]])

feat_names = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20','atr_pct','vol_r',
              'rsi14','macd','macd_sig','macd_hist','log_circ_mv','turnover_20']
mf_cols = ['sm_net','md_net','lg_net','elg_net','total_net']
for c in mf_cols:
    feat_names += [f'{c}_5', f'{c}_20']

F = {k: np.full(n, np.nan) for k in feat_names}
mf_d = {c: df[c].values for c in mf_cols}

for idx in range(len(starts)):
    s, e = starts[idx], ends[idx]
    if e - s < 30: continue
    c_ = close[s:e]; h_ = high_arr[s:e]; l_ = low_arr[s:e]; v = vol[s:e]; mv = cmv[s:e]
    F['r1'][s:e] = np.concatenate([[np.nan], np.diff(c_) / c_[:-1]])
    for lag in [5, 10, 20]:
        a = np.full(e - s, np.nan); a[lag:] = c_[lag:] / c_[:-lag] - 1; F[f'r{lag}'][s:e] = a
    for w in [5, 10, 20]:
        ma = pd.Series(c_).rolling(w).mean().values; F[f'd{w}'][s:e] = (c_ - ma) / (ma + 1e-10)
    ret = np.concatenate([[np.nan], np.diff(c_) / (c_[:-1] + 1e-10)])
    F['vol5'][s:e] = pd.Series(ret).rolling(5).std().values
    F['vol20'][s:e] = pd.Series(ret).rolling(20).std().values
    F['atr_pct'][s:e] = pd.Series(h_ - l_).rolling(14).mean().values / (c_ + 1e-10)
    F['vol_r'][s:e] = pd.Series(v).rolling(5).mean().values / (pd.Series(v).rolling(20).mean().values + 1)
    delta = np.concatenate([[0], np.diff(c_)]); g, l = np.maximum(delta, 0), np.maximum(-delta, 0)
    F['rsi14'][s:e] = 100 - (100 / (1 + pd.Series(g).rolling(14).mean().values / (pd.Series(l).rolling(14).mean().values + 1e-10)))
    e12 = pd.Series(c_).ewm(span=12).mean().values; e26 = pd.Series(c_).ewm(span=26).mean().values
    F['macd'][s:e] = e12 - e26; F['macd_sig'][s:e] = pd.Series(e12 - e26).ewm(span=9).mean().values
    F['macd_hist'][s:e] = F['macd'][s:e] - F['macd_sig'][s:e]
    for col in mf_cols:
        vals = mf_d[col][s:e]
        F[f'{col}_5'][s:e] = pd.Series(vals).rolling(5).sum().values
        F[f'{col}_20'][s:e] = pd.Series(vals).rolling(20).sum().values
    F['log_circ_mv'][s:e] = np.log1p(np.where(np.isnan(mv), v * c_, mv))
    F['turnover_20'][s:e] = np.zeros(e - s)

for k, a in F.items():
    df[k] = a
df['fwd20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)
df = df.dropna(subset=feat_names + ['fwd20'])
print(f'Features: {len(df):,} ({time.time()-t0:.0f}s)')

# 保存特征parquet（后续复用）
df.to_parquet('data/cn/features_v2.parquet', index=False)
print(f'Saved features parquet')

# Portfolio Backtest
print('\nPortfolio Backtest...')
trade_dates = sorted(df['date'].unique())
rebal_dates = trade_dates[::20]
rebal_dates = [d for d in rebal_dates if pd.Timestamp('2017-01-01') <= d <= max(trade_dates) - pd.Timedelta(days=45)]

all_port_rets = []
all_bench_rets = []
ics_list = []

for i in range(len(rebal_dates) - 1):
    rebal = rebal_dates[i]
    next_rebal = rebal_dates[i + 1]
    
    # 训练：最近2年数据（减少训练量）
    train_start = rebal - pd.DateOffset(years=2)
    train = df[(df['date'] >= train_start) & (df['date'] <= rebal)]
    if len(train) < 5000: continue
    
    X_tr = train[feat_names].values
    y_tr = train['fwd20'].values
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    
    # 选股
    current = df[df['date'] == rebal].dropna(subset=feat_names).copy()
    if len(current) < 20: continue
    current['score'] = model.predict(current[feat_names].values)
    current = current[current['close'] > 3].sort_values('score', ascending=False)
    top15 = current.head(15)['sym'].tolist()
    
    # 持有期收益
    hold = df[(df['date'] > rebal) & (df['date'] <= next_rebal)]
    top15_hold = hold[hold['sym'].isin(top15)].sort_values(['sym', 'date'])
    top15_hold['daily_ret'] = top15_hold.groupby('sym')['close'].pct_change()
    port_rets = top15_hold.groupby('date')['daily_ret'].mean().dropna()
    
    # 基准
    all_hold = hold.sort_values(['sym', 'date'])
    all_hold['daily_ret'] = all_hold.groupby('sym')['close'].pct_change()
    bench_rets = all_hold.groupby('date')['daily_ret'].mean().dropna()
    
    # IC
    test = df[(df['date'] >= rebal) & (df['date'] <= next_rebal)]
    if len(test) > 100:
        test_pred = test.dropna(subset=feat_names + ['fwd20']).copy()
        if len(test_pred) > 50:
            test_pred['pred'] = model.predict(test_pred[feat_names].values)
            for d in test_pred['date'].unique():
                dd = test_pred[test_pred['date'] == d]
                if len(dd) > 20:
                    ics_list.append(np.corrcoef(dd['fwd20'], dd['pred'])[0, 1])
    
    all_port_rets.extend([(d, r) for d, r in port_rets.items()])
    all_bench_rets.extend([(d, r) for d, r in bench_rets.items()])
    
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/{len(rebal_dates)-1} periods done ({time.time()-t0:.0f}s)')

# 计算指标
port = pd.DataFrame(all_port_rets, columns=['date', 'ret']).drop_duplicates('date').sort_values('date')
bench = pd.DataFrame(all_bench_rets, columns=['date', 'ret']).drop_duplicates('date').sort_values('date')

port['cum'] = (1 + port['ret']).cumprod()
bench['cum'] = (1 + bench['ret']).cumprod()

years = (port['date'].max() - port['date'].min()).days / 365.25
cagr = port['cum'].iloc[-1] ** (1 / years) - 1
ann_vol = port['ret'].std() * np.sqrt(252)
rf = 0.02
sharpe = (cagr - rf) / ann_vol
downside = port['ret'][port['ret'] < 0].std() * np.sqrt(252)
sortino = (cagr - rf) / downside if downside > 0 else 0
cum_max = port['cum'].cummax()
dd = port['cum'] / cum_max - 1
max_dd = dd.min()
calmar = cagr / abs(max_dd) if max_dd != 0 else 0

byears = (bench['date'].max() - bench['date'].min()).days / 365.25
bench_cagr = bench['cum'].iloc[-1] ** (1 / byears) - 1
bench_vol = bench['ret'].std() * np.sqrt(252)
bench_sharpe = (bench_cagr - rf) / bench_vol
bench_cum_max = bench['cum'].cummax()
bench_dd = (bench['cum'] / bench_cum_max - 1).min()

alpha = cagr - bench_cagr

print(f'\n{"="*55}')
print(f'A股模型V2 Portfolio Backtest')
print(f'{"="*55}')
print(f'回测: {port["date"].min().date()} ~ {port["date"].max().date()} ({years:.1f}年)')
print(f'调仓: 每20交易日 | 持仓: Top15等权')
print(f'')
print(f'{"指标":<18} {"模型组合":<14} {"全市场基准":<14}')
print(f'{"-"*46}')
print(f'{"年化收益":<18} {cagr*100:>12.2f}% {bench_cagr*100:>12.2f}%')
print(f'{"年化波动率":<18} {ann_vol*100:>12.2f}% {bench_vol*100:>12.2f}%')
print(f'{"Sharpe":<18} {sharpe:>12.2f} {bench_sharpe:>12.2f}')
print(f'{"Sortino":<18} {sortino:>12.2f}')
print(f'{"最大回撤":<18} {max_dd*100:>12.2f}% {bench_dd*100:>12.2f}%')
print(f'{"Calmar":<18} {calmar:>12.2f}')
print(f'{"Alpha":<18} {alpha*100:>12.2f}%')
print(f'{"总收益":<18} {(port["cum"].iloc[-1]-1)*100:>12.2f}% {(bench["cum"].iloc[-1]-1)*100:>12.2f}%')

ic_mean = np.nanmean(ics_list) if ics_list else 0
print(f'\nIC: {ic_mean*100:.2f}% ({len(ics_list)} dates)')
print(f'Time: {time.time()-t0:.0f}s')
