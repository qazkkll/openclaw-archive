#!/usr/bin/env python3
"""
A股模型研究：预计算特征 + 3轮实验
1. 预计算所有特征（只用2020-2026数据，约4M行）
2. 实验1: tech_only vs tech+mf vs tech+mf+size
3. 输出对比结果
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data'
OUT = '/home/hermes/.hermes/openclaw-archive/research'

print("=" * 60)
print("A股模型：特征实验")
print("=" * 60)

# ===== 加载 =====
print("\n[1] 加载...")
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

# 只用2020-2026
df = df[df['date'] >= '2020-01-01'].copy()
print(f"  2020+: {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== 特征工程（向量化，避免groupby lambda）=====
print("\n[2] 特征工程...")

# 按sym排序，用numpy加速
syms = df['sym'].values
dates = df['date'].values
close = df['close'].values
high = df['high'].values
low = df['low'].values
vol = df['volume'].values

# 找到每只股票的边界
sym_change = np.where(syms[1:] != syms[:-1])[0] + 1
starts = np.concatenate([[0], sym_change])
ends = np.concatenate([sym_change, [len(df)]])

n = len(df)
# 预分配
r1 = np.full(n, np.nan)
r5 = np.full(n, np.nan)
r10 = np.full(n, np.nan)
r20 = np.full(n, np.nan)
d5 = np.full(n, np.nan)
d10 = np.full(n, np.nan)
d20 = np.full(n, np.nan)
vol5 = np.full(n, np.nan)
vol20 = np.full(n, np.nan)
atr_pct = np.full(n, np.nan)
vol_r = np.full(n, np.nan)
rsi14 = np.full(n, np.nan)
macd_arr = np.full(n, np.nan)
macd_sig = np.full(n, np.nan)
macd_hist = np.full(n, np.nan)

mf_cols = ['sm_net','md_net','lg_net','elg_net','total_net']
mf_arrays = {c: df[c].values for c in mf_cols}
mf_5 = {c: np.full(n, np.nan) for c in mf_cols}
mf_20 = {c: np.full(n, np.nan) for c in mf_cols}

log_amt_20 = np.full(n, np.nan)

print(f"  计算 {len(starts)} 只股票的特征...")
for idx in range(len(starts)):
    s, e = starts[idx], ends[idx]
    if e - s < 30:
        continue
    
    c = close[s:e]
    h_ = high[s:e]
    l_ = low[s:e]
    v = vol[s:e]
    
    # 动量
    r1[s:e] = np.diff(c, prepend=c[0]) / (np.roll(c, 1) + 1e-10)
    r1[s] = np.nan
    for lag, arr in [(5, r5), (10, r10), (20, r20)]:
        if e-s > lag:
            arr[s+lag:e] = c[lag:] / c[:-lag] - 1
    
    # 均线偏离
    for w, arr in [(5, d5), (10, d10), (20, d20)]:
        ma = pd.Series(c).rolling(w).mean().values
        arr[s:e] = (c - ma) / (ma + 1e-10)
    
    # 波动率
    ret = np.diff(c) / (c[:-1] + 1e-10)
    ret = np.concatenate([[np.nan], ret])
    v5 = pd.Series(ret).rolling(5).std().values
    v20 = pd.Series(ret).rolling(20).std().values
    vol5[s:e] = v5
    vol20[s:e] = v20
    
    # ATR
    atr = pd.Series(h_ - l_).rolling(14).mean().values
    atr_pct[s:e] = atr / (c + 1e-10)
    
    # 成交量比
    vm5 = pd.Series(v).rolling(5).mean().values
    vm20 = pd.Series(v).rolling(20).mean().values
    vol_r[s:e] = vm5 / (vm20 + 1)
    
    # RSI
    delta = np.diff(c, prepend=c[0])
    delta[0] = 0
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    ag = pd.Series(gain).rolling(14).mean().values
    al = pd.Series(loss).rolling(14).mean().values
    rsi14[s:e] = 100 - (100 / (1 + ag / (al + 1e-10)))
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    macd_arr[s:e] = ema12 - ema26
    macd_sig[s:e] = pd.Series(macd_arr[s:e]).ewm(span=9).mean().values
    macd_hist[s:e] = macd_arr[s:e] - macd_sig[s:e]
    
    # 资金流
    for col in mf_cols:
        vals = mf_arrays[col][s:e]
        mf_5[col][s:e] = pd.Series(vals).rolling(5).sum().values
        mf_20[col][s:e] = pd.Series(vals).rolling(20).sum().values
    
    # 市值代理
    amt = v * c
    log_amt_20[s:e] = np.log1p(pd.Series(amt).rolling(20).mean().values)

# 写入df
for name, arr in [
    ('r1',r1),('r5',r5),('r10',r10),('r20',r20),
    ('d5',d5),('d10',d10),('d20',d20),
    ('vol5',vol5),('vol20',vol20),('atr_pct',atr_pct),('vol_r',vol_r),
    ('rsi14',rsi14),('macd',macd_arr),('macd_sig',macd_sig),('macd_hist',macd_hist),
    ('log_amt_20',log_amt_20),
]:
    df[name] = arr

for col in mf_cols:
    df[f'{col}_5'] = mf_5[col]
    df[f'{col}_20'] = mf_20[col]

# 标签
df['fwd20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20)/x-1)

print(f"  特征完成: {time.time()-t0:.0f}s")

# 特征集
tech = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20','atr_pct','vol_r','rsi14','macd','macd_sig','macd_hist']
mf_feat = [f'{c}_5' for c in mf_cols] + [f'{c}_20' for c in mf_cols]
sz = ['log_amt_20']

all_feats = tech + mf_feat + sz
df = df.dropna(subset=all_feats + ['fwd20'])
print(f"  有效: {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== Walk-Forward =====
print("\n[3] Walk-Forward...")

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
    c += pd.DateOffset(years=1)  # 更密集的fold
print(f"  {len(splits)} folds")

experiments = {
    'tech_only': tech,
    'tech+mf': tech + mf_feat,
    'tech+mf+size': tech + mf_feat + sz,
}

all_res = {}

for name, feats in experiments.items():
    avail = [f for f in feats if f in df.columns]
    print(f"\n  === {name} ({len(avail)} feat) ===")
    res = []
    
    for i, (ts, te, vs, ve) in enumerate(splits):
        tr = df[(df['date']>=ts)&(df['date']<=te)]
        tt = df[(df['date']>=vs)&(df['date']<=ve)]
        if len(tr)<30000 or len(tt)<5000: continue
        
        X_tr, y_tr = tr[avail].values, tr['fwd20'].values
        X_te = tt[avail].values
        
        t1 = time.time()
        m = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        
        tc = tt[['date','fwd20']].copy()
        tc['pred'] = p
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
        ls = top - bot
        
        res.append({'ic':ic,'icir':ic/(ic_s+1e-10),'rank_ic':ric,'ls':ls,'t':time.time()-t1})
        print(f"    F{i+1}: IC={ic:.4f} RIC={ric:.4f} LS={ls*100:.2f}%")
    
    s = {}
    for k in ['ic','icir','rank_ic','ls','t']:
        vs = [r[k] for r in res if not np.isnan(r[k])]
        if vs: s[k]=float(np.mean(vs)); s[k+'_std']=float(np.std(vs))
    all_res[name] = {'summary':s, 'nfeat':len(avail)}
    print(f"  → IC={s.get('ic',0)*100:.2f}% LS={s.get('ls',0)*100:.2f}%")

# 特征重要性（最后的model）
print(f"\n特征重要性 TOP 10:")
imp = m.feature_importances_
fi = sorted(zip(avail, imp), key=lambda x: -x[1])
for fn, fv in fi[:10]:
    print(f"  {fn:<18} {fv:.4f}")

# 对比
print(f"\n{'='*60}")
print("特征组合对比")
print(f"{'='*60}")
print(f"{'实验':<16} {'IC':>8} {'RankIC':>8} {'ICIR':>8} {'多空':>8} {'#feat'}")
print("-"*55)
for n, d in all_res.items():
    s = d['summary']
    print(f"{n:<16} {s.get('ic',0)*100:>7.2f}% {s.get('rank_ic',0)*100:>7.2f}% {s.get('icir',0)*100:>7.1f}% {s.get('ls',0)*100:>7.2f}% {d['nfeat']:>5}")

best = max(all_res.items(), key=lambda x: x[1]['summary'].get('ls',0))
print(f"\n最优: {best[0]}")

with open(f'{OUT}/round2_results.json', 'w') as f:
    json.dump(all_res, f, indent=2, default=str)
print(f"保存: {OUT}/round2_results.json ({time.time()-t0:.0f}s)")
