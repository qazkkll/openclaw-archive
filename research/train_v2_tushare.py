#!/usr/bin/env python3
"""
A股模型V2 — 按天拉取tushare数据
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import tushare as ts
import json, time, os, warnings
warnings.filterwarnings('ignore')

TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
OUT = '/home/hermes/.hermes/openclaw-archive/research'
MODEL_DIR = '/home/hermes/.hermes/openclaw-archive/models/cn'
os.makedirs(MODEL_DIR, exist_ok=True)

ts.set_token(TOKEN)
pro = ts.pro_api()

print("=" * 60)
print("A股模型V2 — tushare按天拉取")
print("=" * 60)

t0 = time.time()

# ===== 1. 交易日历 =====
print("\n[1] 获取交易日历...")
cal = pro.trade_cal(exchange='SSE', start_date='20230101', end_date='20260620', is_open='1')
trade_dates = sorted(cal['cal_date'].tolist())
print(f"  {len(trade_dates)} 个交易日 ({trade_dates[0]}~{trade_dates[-1]})")

# ===== 2. 按天拉取 =====
print(f"\n[2] 按天拉取日线+资金流+市值 ({len(trade_dates)}天)...")

all_daily, all_mf, all_basic = [], [], []
errors = 0

for i, d in enumerate(trade_dates):
    try:
        # 日线
        df_d = pro.daily(trade_date=d)
        if len(df_d) > 0:
            all_daily.append(df_d)
        
        # 资金流
        df_m = pro.moneyflow(trade_date=d)
        if len(df_m) > 0:
            all_mf.append(df_m)
        
        # 市值+换手率
        df_b = pro.daily_basic(trade_date=d, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv')
        if len(df_b) > 0:
            all_basic.append(df_b)
        
        if (i+1) % 50 == 0:
            print(f"    {i+1}/{len(trade_dates)} ({d}) daily={len(all_daily[-1])} mf={len(all_mf[-1])} basic={len(all_basic[-1])}")
        
        time.sleep(0.15)  # 限流
    except Exception as e:
        errors += 1
        if errors < 5:
            print(f"    {d} 错误: {e}")
        time.sleep(0.5)

daily = pd.concat(all_daily, ignore_index=True)
mf = pd.concat(all_mf, ignore_index=True)
basic = pd.concat(all_basic, ignore_index=True)
print(f"  日线: {len(daily):,} 行")
print(f"  资金流: {len(mf):,} 行")
print(f"  市值: {len(basic):,} 行")
print(f"  错误: {errors} 天")
print(f"  耗时: {time.time()-t0:.0f}s")

# ===== 3. 数据对齐 =====
print("\n[3] 数据对齐...")

daily['date'] = pd.to_datetime(daily['trade_date'])
daily['sym'] = daily['ts_code'].str.replace(r'\.\w+$', '', regex=True)

mf['date'] = pd.to_datetime(mf['trade_date'])
mf['sym'] = mf['ts_code'].str.replace(r'\.\w+$', '', regex=True)
mf['sm_net'] = mf['buy_sm_amount'] - mf['sell_sm_amount']
mf['md_net'] = mf['buy_md_amount'] - mf['sell_md_amount']
mf['lg_net'] = mf['buy_lg_amount'] - mf['sell_lg_amount']
mf['elg_net'] = mf['buy_elg_amount'] - mf['sell_elg_amount']
mf['total_net'] = mf['net_mf_amount']

basic['date'] = pd.to_datetime(basic['trade_date'])
basic['sym'] = basic['ts_code'].str.replace(r'\.\w+$', '', regex=True)

df = daily[['sym','date','open','high','low','close','vol']].rename(columns={'vol':'volume'})
df = df.merge(mf[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']], on=['sym','date'])
df = df.merge(basic[['sym','date','turnover_rate','circ_mv','total_mv']], on=['sym','date'], how='left')
df = df.sort_values(['sym','date']).reset_index(drop=True)
df = df[df['close'] > 0]
print(f"  合并: {len(df):,} 行, {df['sym'].nunique()} 股")

# 行业
stock_info = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
industry_map = dict(zip(stock_info['ts_code'], stock_info['industry']))

# ===== 4. 特征工程 =====
print("\n[4] 特征工程...")

syms = df['sym'].values
close = df['close'].values
high = df['high'].values
low = df['low'].values
vol = df['volume'].values
turnover = df['turnover_rate'].values
circ_mv = df['circ_mv'].values
n = len(df)

sym_change = np.where(syms[1:] != syms[:-1])[0] + 1
starts = np.concatenate([[0], sym_change])
ends = np.concatenate([sym_change, [n]])

feat_names = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20',
              'atr_pct','vol_r','rsi14','macd','macd_sig','macd_hist',
              'log_circ_mv','turnover_20']
mf_cols = ['sm_net','md_net','lg_net','elg_net','total_net']
for c in mf_cols:
    feat_names += [f'{c}_5', f'{c}_20']

features = {name: np.full(n, np.nan) for name in feat_names}
mf_data = {c: df[c].values for c in mf_cols}

for idx in range(len(starts)):
    s, e = starts[idx], ends[idx]
    if e - s < 30: continue
    
    c = close[s:e]
    h_ = high[s:e]
    l_ = low[s:e]
    v = vol[s:e]
    tr = turnover[s:e]
    mv = circ_mv[s:e]
    
    features['r1'][s:e] = np.concatenate([[np.nan], np.diff(c)/c[:-1]])
    for lag in [5,10,20]:
        arr = np.full(e-s, np.nan)
        arr[lag:] = c[lag:]/c[:-lag]-1
        features[f'r{lag}'][s:e] = arr
    
    for w in [5,10,20]:
        ma = pd.Series(c).rolling(w).mean().values
        features[f'd{w}'][s:e] = (c-ma)/(ma+1e-10)
    
    ret = np.concatenate([[np.nan], np.diff(c)/(c[:-1]+1e-10)])
    features['vol5'][s:e] = pd.Series(ret).rolling(5).std().values
    features['vol20'][s:e] = pd.Series(ret).rolling(20).std().values
    features['atr_pct'][s:e] = pd.Series(h_-l_).rolling(14).mean().values / (c+1e-10)
    features['vol_r'][s:e] = pd.Series(v).rolling(5).mean().values / (pd.Series(v).rolling(20).mean().values + 1)
    
    delta = np.concatenate([[0], np.diff(c)])
    g, l = np.maximum(delta,0), np.maximum(-delta,0)
    features['rsi14'][s:e] = 100-(100/(1+pd.Series(g).rolling(14).mean().values/(pd.Series(l).rolling(14).mean().values+1e-10)))
    
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    features['macd'][s:e] = ema12-ema26
    features['macd_sig'][s:e] = pd.Series(features['macd'][s:e]).ewm(span=9).mean().values
    features['macd_hist'][s:e] = features['macd'][s:e]-features['macd_sig'][s:e]
    
    for col in mf_cols:
        vals = mf_data[col][s:e]
        features[f'{col}_5'][s:e] = pd.Series(vals).rolling(5).sum().values
        features[f'{col}_20'][s:e] = pd.Series(vals).rolling(20).sum().values
    
    features['log_circ_mv'][s:e] = np.log1p(mv)
    features['turnover_20'][s:e] = pd.Series(tr).rolling(20).mean().values

for name, arr in features.items():
    df[name] = arr

df['fwd20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20)/x-1)
df = df.dropna(subset=feat_names + ['fwd20'])
print(f"  有效: {len(df):,} 行, {df['sym'].nunique()} 股, {len(feat_names)} 特征")

# ===== 5. Walk-Forward =====
print("\n[5] Walk-Forward验证...")
min_d, max_d = df['date'].min(), df['date'].max()
splits = []
c = min_d
while True:
    te = c + pd.DateOffset(years=1, months=6)
    vs = te + pd.Timedelta(days=1)
    ve = vs + pd.DateOffset(months=6)
    if ve > max_d:
        ve = max_d
        if vs < max_d: splits.append((c, te, vs, ve))
        break
    splits.append((c, te, vs, ve))
    c += pd.DateOffset(years=1)
print(f"  {len(splits)} folds")

wf = []
for i, (ts_, te, vs, ve) in enumerate(splits):
    tr = df[(df['date']>=ts_)&(df['date']<=te)]
    tt = df[(df['date']>=vs)&(df['date']<=ve)]
    if len(tr)<5000 or len(tt)<2000: continue
    
    X_tr, y_tr = tr[feat_names].values, tr['fwd20'].values
    X_te = tt[feat_names].values
    
    t1 = time.time()
    m = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    
    tc = tt[['date','fwd20']].copy()
    tc['pred'] = pred
    ics, rics = [], []
    for d in tc['date'].unique():
        dd = tc[tc['date']==d]
        if len(dd)<20: continue
        ics.append(np.corrcoef(dd['fwd20'], dd['pred'])[0,1])
        rics.append(spearmanr(dd['fwd20'], dd['pred'])[0])
    
    ic, ric = np.nanmean(ics), np.nanmean(rics)
    ic_s = np.nanstd(ics)
    
    tc['pct'] = tc.groupby('date')['pred'].rank(pct=True)
    top = tc[tc['pct']>=0.9]['fwd20'].mean()
    bot = tc[tc['pct']<=0.1]['fwd20'].mean()
    ls = top-bot
    
    wf.append({'fold':i+1,'ic':ic,'rank_ic':ric,'icir':ic/(ic_s+1e-10),'ls':ls,'t':time.time()-t1})
    print(f"  F{i+1}: IC={ic:.4f} RIC={ric:.4f} LS={ls*100:.2f}%")

print(f"\n  汇总:")
for k in ['ic','rank_ic','icir','ls']:
    vals = [r[k] for r in wf if not np.isnan(r[k])]
    if vals:
        print(f"    {k:<12} {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}%")

# ===== 6. 生产模型 + 信号 =====
print("\n[6] 训练生产模型...")
X_all = df[feat_names].values
y_all = df['fwd20'].values
prod = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
prod.fit(X_all, y_all)
prod.save_model(f'{MODEL_DIR}/a_stock_xgb_v2.json')
print(f"  保存: {MODEL_DIR}/a_stock_xgb_v2.json")

imp = prod.feature_importances_
fi = sorted(zip(feat_names, imp), key=lambda x: -x[1])
print(f"\n  特征重要性:")
for fn, fv in fi:
    bar = '█' * int(fv/max(imp)*20)
    print(f"    {fn:<18} {fv:.4f} {bar}")

# 信号
print(f"\n[7] 生成信号...")
latest = df['date'].max()
latest_df = df[df['date'] == latest].copy()
print(f"  {latest.date()}: {len(latest_df)} 只")

X_latest = latest_df[feat_names].values
latest_df['score'] = prod.predict(X_latest)
latest_df = latest_df.sort_values('score', ascending=False)
latest_df['industry'] = latest_df['sym'].map(lambda x: industry_map.get(f'{x}.SZ', industry_map.get(f'{x}.SH', '?')))

top15 = latest_df.head(15).copy()
top15['rank'] = range(1, len(top15)+1)
top15['expected_ret'] = top15['score'] * 100
top15['circ_mv_yi'] = top15['circ_mv'] / 10000

print(f"\n  🎯 A股V2 Top 15 ({latest.date()})")
print(f"  {'#':<3} {'股票':<8} {'行业':<8} {'价格':>7} {'预期':>6} {'5d':>6} {'20d':>6} {'换手':>6} {'市值亿':>7}")
print(f"  {'-'*62}")
for _, r in top15.iterrows():
    print(f"  {r['rank']:<3} {r['sym']:<8} {r['industry']:<8} {r['close']:>7.2f} {r['expected_ret']:>5.1f}% {r['r5']*100:>5.1f}% {r['r20']*100:>5.1f}% {r['turnover_rate']:>5.1f}% {r['circ_mv_yi']:>6.0f}")

# 保存
signal = {
    'date': str(latest.date()),
    'model': 'a_stock_xgb_v2',
    'features': feat_names,
    'wf_ic': float(np.mean([r['ic'] for r in wf])) if wf else 0,
    'wf_ls': float(np.mean([r['ls'] for r in wf])) if wf else 0,
    'top15': top15[['sym','close','score','expected_ret','industry']].to_dict('records'),
}
with open(f'{OUT}/v2_signal.json', 'w') as f:
    json.dump(signal, f, indent=2, default=str)

# 全市场评分
all_scores = latest_df[['sym','close','score','industry']].copy()
all_scores['pct_rank'] = all_scores['score'].rank(pct=True)
all_scores.to_parquet(f'{OUT}/v2_all_scores.parquet', index=False)

print(f"\n  信号: {OUT}/v2_signal.json")
print(f"  全市场: {OUT}/v2_all_scores.parquet")
print(f"\n总耗时: {time.time()-t0:.0f}s")
