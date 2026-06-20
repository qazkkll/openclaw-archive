#!/usr/bin/env python3
"""A股V2 — 最新信号生成（复用训练好的模型）"""
import pandas as pd, numpy as np, xgboost as xgb, json, time, os, warnings
import tushare as ts
warnings.filterwarnings('ignore')

ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()
t0 = time.time()

prod = xgb.XGBRegressor()
prod.load_model('models/cn/a_stock_xgb_v2.json')

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

new_dfs = []
for d in ['20260617','20260618']:
    try:
        dd = pro.daily(trade_date=d)
        mm = pro.moneyflow(trade_date=d)
        bb = pro.daily_basic(trade_date=d, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv')
        if len(dd) > 0 and len(mm) > 0:
            dd['date'] = pd.to_datetime(dd['trade_date'])
            dd['sym'] = dd['ts_code'].str.replace(r'\.\w+$', '', regex=True)
            mm['date'] = pd.to_datetime(mm['trade_date'])
            mm['sym'] = mm['ts_code'].str.replace(r'\.\w+$', '', regex=True)
            for c in ['sm','md','lg','elg']:
                mm[f'{c}_net'] = mm[f'buy_{c}_amount'] - mm[f'sell_{c}_amount']
            mm['total_net'] = mm['net_mf_amount']
            bb['date'] = pd.to_datetime(bb['trade_date'])
            bb['sym'] = bb['ts_code'].str.replace(r'\.\w+$', '', regex=True)
            nd = dd[['sym','date','open','high','low','close','vol']].rename(columns={'vol':'volume'})
            nd = nd.merge(mm[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']], on=['sym','date'])
            nd = nd.merge(bb[['sym','date','turnover_rate','circ_mv','total_mv']], on=['sym','date'], how='left')
            new_dfs.append(nd)
    except Exception as e:
        print(f'{d} error: {e}')

df = pd.merge(h, m, on=['sym','date'])
if new_dfs:
    df = pd.concat([df] + new_dfs, ignore_index=True)
df = df.sort_values(['sym','date']).reset_index(drop=True)
df = df[df['close'] > 0]
df['circ_mv'] = df['circ_mv'].fillna(df['volume'] * df['close'])
df['turnover_rate'] = df['turnover_rate'].fillna(0)
print(f'Data: {len(df):,} rows, {time.time()-t0:.0f}s')

syms = df['sym'].values
close = df['close'].values
high = df['high'].values
low = df['low'].values
vol = df['volume'].values
tr_rate = df['turnover_rate'].values
cmv = df['circ_mv'].values
n = len(df)
sc = np.where(syms[1:] != syms[:-1])[0] + 1
starts = np.concatenate([[0], sc])
ends = np.concatenate([sc, [n]])

feat_names = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20',
              'atr_pct','vol_r','rsi14','macd','macd_sig','macd_hist','log_circ_mv','turnover_20']
mf_cols = ['sm_net','md_net','lg_net','elg_net','total_net']
for c in mf_cols:
    feat_names += [f'{c}_5', f'{c}_20']

F = {k: np.full(n, np.nan) for k in feat_names}
mf_d = {c: df[c].values for c in mf_cols}

print(f'Features: {len(starts)} stocks...')
for idx in range(len(starts)):
    s, e = starts[idx], ends[idx]
    if e - s < 30: continue
    c_ = close[s:e]; h_ = high[s:e]; l_ = low[s:e]; v = vol[s:e]
    tr = tr_rate[s:e]; mv = cmv[s:e]
    F['r1'][s:e] = np.concatenate([[np.nan], np.diff(c_) / c_[:-1]])
    for lag in [5, 10, 20]:
        a = np.full(e - s, np.nan)
        a[lag:] = c_[lag:] / c_[:-lag] - 1
        F[f'r{lag}'][s:e] = a
    for w in [5, 10, 20]:
        ma = pd.Series(c_).rolling(w).mean().values
        F[f'd{w}'][s:e] = (c_ - ma) / (ma + 1e-10)
    ret = np.concatenate([[np.nan], np.diff(c_) / (c_[:-1] + 1e-10)])
    F['vol5'][s:e] = pd.Series(ret).rolling(5).std().values
    F['vol20'][s:e] = pd.Series(ret).rolling(20).std().values
    F['atr_pct'][s:e] = pd.Series(h_ - l_).rolling(14).mean().values / (c_ + 1e-10)
    F['vol_r'][s:e] = pd.Series(v).rolling(5).mean().values / (pd.Series(v).rolling(20).mean().values + 1)
    delta = np.concatenate([[0], np.diff(c_)])
    g, l = np.maximum(delta, 0), np.maximum(-delta, 0)
    F['rsi14'][s:e] = 100 - (100 / (1 + pd.Series(g).rolling(14).mean().values / (pd.Series(l).rolling(14).mean().values + 1e-10)))
    e12 = pd.Series(c_).ewm(span=12).mean().values
    e26 = pd.Series(c_).ewm(span=26).mean().values
    F['macd'][s:e] = e12 - e26
    F['macd_sig'][s:e] = pd.Series(e12 - e26).ewm(span=9).mean().values
    F['macd_hist'][s:e] = F['macd'][s:e] - F['macd_sig'][s:e]
    for col in mf_cols:
        vals = mf_d[col][s:e]
        F[f'{col}_5'][s:e] = pd.Series(vals).rolling(5).sum().values
        F[f'{col}_20'][s:e] = pd.Series(vals).rolling(20).sum().values
    F['log_circ_mv'][s:e] = np.log1p(np.where(np.isnan(mv), v * c_, mv))
    F['turnover_20'][s:e] = pd.Series(np.where(np.isnan(tr), 0, tr)).rolling(20).mean().values

for k, a in F.items():
    df[k] = a
print(f'Features done: {time.time()-t0:.0f}s')

latest = df['date'].max()
ldf = df[df['date'] == latest].dropna(subset=feat_names).copy()
print(f'Signal: {latest.date()}, {len(ldf)} stocks')

ldf['score'] = prod.predict(ldf[feat_names].values)
ldf = ldf.sort_values('score', ascending=False)

si = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
ind_map = dict(zip(si['ts_code'], si['industry']))
ldf['industry'] = ldf['sym'].map(lambda x: ind_map.get(f'{x}.SZ', ind_map.get(f'{x}.SH', '?')))

top15 = ldf.head(15).copy()
top15['rank'] = range(1, 16)
top15['expected_ret'] = top15['score'] * 100

print(f'\n  A股V2 Top 15 ({latest.date()})')
print(f'  {"#":<3} {"股票":<8} {"行业":<8} {"价格":>7} {"预期":>6} {"5d":>6} {"20d":>6}')
print(f'  {"-"*50}')
for _, r in top15.iterrows():
    print(f'  {r["rank"]:<3} {r["sym"]:<8} {r["industry"]:<8} {r["close"]:>7.2f} {r["expected_ret"]:>5.1f}% {r["r5"]*100:>5.1f}% {r["r20"]*100:>5.1f}%')

bot10 = ldf.tail(10).copy()
print(f'\n  Bottom 10:')
for _, r in bot10.iterrows():
    print(f'    {r["sym"]:<8} {r["industry"]:<8} {r["close"]:>7.2f} {r["score"]*100:>5.1f}%')

signal = {
    'date': str(latest.date()),
    'model': 'a_stock_xgb_v2',
    'wf': {'ic': 0.0809, 'rank_ic': 0.0817, 'icir': 0.996, 'ls': 0.0262},
    'top15': top15[['sym','close','score','expected_ret','industry']].to_dict('records'),
}
os.makedirs('research', exist_ok=True)
with open('research/v2_signal.json', 'w') as f:
    json.dump(signal, f, indent=2, default=str)
ldf[['sym','close','score','industry']].to_parquet('research/v2_all_scores.parquet', index=False)
print(f'\nSaved: research/v2_signal.json ({time.time()-t0:.0f}s)')
