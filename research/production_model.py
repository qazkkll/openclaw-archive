#!/usr/bin/env python3
"""
最终验证 + 今日信号生成
1. 用2016-2026全量数据训练生产模型
2. Walk-Forward 7段验证
3. 生成今日Top15推荐信号
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data'
OUT = '/home/hermes/.hermes/openclaw-archive/research'
MODEL_OUT = '/home/hermes/.hermes/openclaw-archive/models/cn'

print("=" * 60)
print("A股生产模型 + 今日信号")
print("=" * 60)

# ===== 加载全量数据 =====
print("\n[1] 加载全量数据...")
t0 = time.time()

h = pd.read_parquet(f'{DATA}/a_hist_10y.parquet')
h = h.rename(columns={'Code':'sym','Date':'date','O':'open','H':'high','L':'low','C':'close','V':'volume'})
h['date'] = pd.to_datetime(h['date'].astype(str), format='%Y%m%d')

m = pd.read_parquet(f'{DATA}/moneyflow_core.parquet')
m['sym'] = m['ts_code'].str.replace(r'\.\w+$', '', regex=True)
m['date'] = pd.to_datetime(m['trade_date'].astype(str), format='%Y%m%d')
for c in ['sm','md','lg','elg']:
    m[f'{c}_net'] = m[f'buy_{c}_amount'] - m[f'sell_{c}_amount']
m['total_net'] = m['net_mf_amount']
m = m[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']].drop_duplicates(['sym','date'])

df = pd.merge(h, m, on=['sym','date'])
df = df.sort_values(['sym','date']).reset_index(drop=True)
df = df[df['close']>0]
print(f"  全量: {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== 特征工程 =====
print("\n[2] 特征工程...")

syms = df['sym'].values
close = df['close'].values
high = df['high'].values
low = df['low'].values
vol = df['volume'].values
n = len(df)

sym_change = np.where(syms[1:] != syms[:-1])[0] + 1
starts = np.concatenate([[0], sym_change])
ends = np.concatenate([sym_change, [n]])

features = {
    'r1': np.full(n, np.nan), 'r5': np.full(n, np.nan),
    'r10': np.full(n, np.nan), 'r20': np.full(n, np.nan),
    'd5': np.full(n, np.nan), 'd10': np.full(n, np.nan), 'd20': np.full(n, np.nan),
    'vol5': np.full(n, np.nan), 'vol20': np.full(n, np.nan),
    'atr_pct': np.full(n, np.nan), 'vol_r': np.full(n, np.nan),
    'rsi14': np.full(n, np.nan),
    'macd': np.full(n, np.nan), 'macd_sig': np.full(n, np.nan), 'macd_hist': np.full(n, np.nan),
}

mf_cols = ['sm_net','md_net','lg_net','elg_net','total_net']
mf_data = {c: df[c].values for c in mf_cols}
for c in mf_cols:
    features[f'{c}_5'] = np.full(n, np.nan)
    features[f'{c}_20'] = np.full(n, np.nan)

features['log_amt_20'] = np.full(n, np.nan)

for idx in range(len(starts)):
    s, e = starts[idx], ends[idx]
    if e - s < 30: continue
    
    c = close[s:e]
    h_ = high[s:e]
    l_ = low[s:e]
    v = vol[s:e]
    
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
    
    atr = pd.Series(h_-l_).rolling(14).mean().values
    features['atr_pct'][s:e] = atr/(c+1e-10)
    
    vm5 = pd.Series(v).rolling(5).mean().values
    vm20 = pd.Series(v).rolling(20).mean().values
    features['vol_r'][s:e] = vm5/(vm20+1)
    
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
    
    features['log_amt_20'][s:e] = np.log1p(pd.Series(v*c).rolling(20).mean().values)

for name, arr in features.items():
    df[name] = arr

df['fwd20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20)/x-1)

feat_list = list(features.keys())
df = df.dropna(subset=feat_list+['fwd20'])
print(f"  有效: {len(df):,} 行 ({time.time()-t0:.0f}s)")

# ===== Walk-Forward 7段 =====
print("\n[3] Walk-Forward 7段...")

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
    c += pd.DateOffset(years=1, months=6)
print(f"  {len(splits)} folds")

wf_results = []
for i, (ts, te, vs, ve) in enumerate(splits):
    tr = df[(df['date']>=ts)&(df['date']<=te)]
    tt = df[(df['date']>=vs)&(df['date']<=ve)]
    if len(tr)<30000 or len(tt)<5000: continue
    
    X_tr, y_tr = tr[feat_list].values, tr['fwd20'].values
    X_te = tt[feat_list].values
    
    t1 = time.time()
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    
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
    
    top_d = tc[tc['pct']>=0.9].groupby('date')['fwd20'].mean()
    if len(top_d)>5:
        cu = (1+top_d).cumprod()
        dd_v = (cu/cu.cummax()-1).min()
        ann = cu.iloc[-1]**(252/max(len(cu),1))-1
    else:
        ann, dd_v = 0, 0
    
    wf_results.append({
        'fold':i+1, 'train':f"{ts.date()}~{te.date()}", 'test':f"{vs.date()}~{ve.date()}",
        'ic':ic, 'rank_ic':ric, 'icir':ic/(ic_s+1e-10), 'ls':ls, 'ann':ann, 'dd':dd_v,
        'train_n':len(tr), 'test_n':len(tt), 'time':time.time()-t1,
    })
    print(f"  F{i+1}: IC={ic:.4f} RIC={ric:.4f} LS={ls*100:.2f}% DD={dd_v*100:.1f}%")

# 汇总
print(f"\n{'='*60}")
print("Walk-Forward 汇总")
print(f"{'='*60}")
for k in ['ic','rank_ic','icir','ls','ann','dd']:
    vals = [r[k] for r in wf_results if not np.isnan(r[k])]
    if vals:
        print(f"  {k:<12} 均值={np.mean(vals)*100:.2f}%  std={np.std(vals)*100:.2f}%  min={np.min(vals)*100:.2f}%  max={np.max(vals)*100:.2f}%")

# ===== 生产模型 =====
print("\n[4] 训练生产模型...")
latest_date = df['date'].max()
print(f"  最新数据日期: {latest_date.date()}")

# 用最近3年数据训练生产模型
train_cutoff = latest_date - pd.DateOffset(years=3)
prod_df = df[df['date'] >= train_cutoff].copy()
print(f"  训练数据: {len(prod_df):,} 行, {prod_df['date'].min().date()}~{prod_df['date'].max().date()}")

X_prod = prod_df[feat_list].values
y_prod = prod_df['fwd20'].values

prod_model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
prod_model.fit(X_prod, y_prod)

# 保存模型
import os
os.makedirs(MODEL_OUT, exist_ok=True)
prod_model.save_model(f'{MODEL_OUT}/a_stock_xgb_v1.json')
print(f"  模型保存: {MODEL_OUT}/a_stock_xgb_v1.json")

# 特征重要性
imp = prod_model.feature_importances_
fi = sorted(zip(feat_list, imp), key=lambda x: -x[1])
print(f"\n  特征重要性:")
for fn, fv in fi:
    bar = '█' * int(fv/max(imp)*25)
    print(f"    {fn:<18} {fv:.4f} {bar}")

# ===== 今日信号 =====
print(f"\n[5] 今日信号生成...")
# 获取最新一天的数据
latest = df[df['date'] == latest_date].copy()
print(f"  最新日期: {latest_date.date()}, {len(latest)} 只股票")

if len(latest) > 0:
    X_latest = latest[feat_list].values
    latest['score'] = prod_model.predict(X_latest)
    
    # 排名
    latest = latest.sort_values('score', ascending=False)
    
    # Top 15
    top15 = latest.head(15)[['sym','close','score','r5','r20','atr_pct']].copy()
    top15['rank'] = range(1, 16)
    top15['expected_ret'] = top15['score'] * 100
    
    print(f"\n  {'='*50}")
    print(f"  🎯 A股模型 Top 15 推荐 ({latest_date.date()})")
    print(f"  {'='*50}")
    print(f"  {'#':<4} {'股票':<8} {'价格':>8} {'模型分':>8} {'预期收益':>8} {'5日动量':>8} {'20日动量':>8}")
    print(f"  {'-'*50}")
    for _, r in top15.iterrows():
        print(f"  {r['rank']:<4} {r['sym']:<8} {r['close']:>8.2f} {r['score']:>8.4f} {r['expected_ret']:>7.2f}% {r['r5']*100:>7.2f}% {r['r20']*100:>7.2f}%")
    
    # Bottom 10 (做空信号)
    bot10 = latest.tail(10)[['sym','close','score','r5','r20']].copy()
    print(f"\n  ⚠️ Bottom 10 (避免):")
    for _, r in bot10.iterrows():
        print(f"    {r['sym']:<8} {r['close']:>8.2f} {r['score']*100:>7.2f}%")
    
    # 保存信号
    signal = {
        'date': str(latest_date.date()),
        'top15': top15.to_dict('records'),
        'bottom10': bot10.to_dict('records'),
        'model_version': 'a_stock_xgb_v1',
        'features': feat_list,
    }
    with open(f'{OUT}/latest_signal.json', 'w') as f:
        json.dump(signal, f, indent=2, default=str)
    print(f"\n  信号保存: {OUT}/latest_signal.json")

# 保存完整结果
output = {
    'walk_forward': wf_results,
    'summary': {k: float(np.mean([r[k] for r in wf_results if not np.isnan(r[k])])) for k in ['ic','rank_ic','icir','ls','ann','dd']},
    'feature_importance': [{'feature':f,'importance':float(v)} for f,v in fi],
    'model_info': {
        'version': 'a_stock_xgb_v1',
        'features': len(feat_list),
        'train_samples': len(prod_df),
        'train_period': f"{prod_df['date'].min().date()}~{prod_df['date'].max().date()}",
    }
}
with open(f'{OUT}/production_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n总耗时: {time.time()-t0:.0f}s")
