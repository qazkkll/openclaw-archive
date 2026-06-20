#!/usr/bin/env python3
"""A股V2 — tushare全量（按天拉取，保存parquet增量缓存）"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
import tushare as ts
import json, time, os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)  # 行缓冲

ts.set_token('ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db')
pro = ts.pro_api()

CACHE = '/home/hermes/.hermes/openclaw-archive/data/cn'
OUT = '/home/hermes/.hermes/openclaw-archive/research'
MODEL_DIR = '/home/hermes/.hermes/openclaw-archive/models/cn'
for d in [CACHE, OUT, MODEL_DIR]: os.makedirs(d, exist_ok=True)

print("=" * 60)
print("A股V2 — tushare全量")
print("=" * 60)
t0 = time.time()

# ===== 1. 检查缓存 =====
print("\n[1] 检查缓存...")
cached_dates = set()
for f in ['daily.parquet', 'moneyflow.parquet', 'basic.parquet']:
    p = f'{CACHE}/{f}'
    if os.path.exists(p):
        df = pd.read_parquet(p)
        if 'trade_date' in df.columns:
            dates = df['trade_date'].unique()
            cached_dates.update(dates)
            print(f"  {f}: {len(df):,} 行, 最新 {max(dates)}")
        else:
            print(f"  {f}: {len(df):,} 行")

# 交易日历
cal = pro.trade_cal(exchange='SSE', start_date='20230101', end_date='20260620', is_open='1')
all_dates = sorted(cal['cal_date'].tolist())
new_dates = [d for d in all_dates if d not in cached_dates]
print(f"  总交易日: {len(all_dates)}, 已缓存: {len(cached_dates)}, 需拉取: {len(new_dates)}")

# ===== 2. 拉取新数据 =====
if new_dates:
    print(f"\n[2] 拉取 {len(new_dates)} 天...")
    all_d, all_m, all_b = [], [], []
    errs = 0
    for i, d in enumerate(new_dates):
        try:
            dd = pro.daily(trade_date=d)
            if len(dd) > 0: all_d.append(dd)
            mm = pro.moneyflow(trade_date=d)
            if len(mm) > 0: all_m.append(mm)
            bb = pro.daily_basic(trade_date=d, fields='ts_code,trade_date,turnover_rate,circ_mv,total_mv')
            if len(bb) > 0: all_b.append(bb)
            time.sleep(0.2)
            if (i+1) % 20 == 0: print(f"    {i+1}/{len(new_dates)}")
        except Exception as e:
            errs += 1
            time.sleep(1)
    
    # 合并缓存
    for name, data, fname in [
        ('日线', all_d, 'daily.parquet'),
        ('资金流', all_m, 'moneyflow.parquet'),
        ('市值', all_b, 'basic.parquet'),
    ]:
        if data:
            new = pd.concat(data, ignore_index=True)
            cache_path = f'{CACHE}/{fname}'
            if os.path.exists(cache_path):
                old = pd.read_parquet(cache_path)
                combined = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=['ts_code','trade_date'])
            else:
                combined = new
            combined.to_parquet(cache_path, index=False)
            print(f"  {name}: +{len(new):,} → 总计 {len(combined):,}")
    
    print(f"  耗时: {time.time()-t0:.0f}s")
else:
    print("\n[2] 无需拉取")

# ===== 3. 加载+合并 =====
print("\n[3] 加载数据...")
daily = pd.read_parquet(f'{CACHE}/daily.parquet')
mf = pd.read_parquet(f'{CACHE}/moneyflow.parquet')
basic = pd.read_parquet(f'{CACHE}/basic.parquet')

daily['date'] = pd.to_datetime(daily['trade_date'])
daily['sym'] = daily['ts_code'].str.replace(r'\.\w+$', '', regex=True)
daily = daily.rename(columns={'vol':'volume'})

mf['date'] = pd.to_datetime(mf['trade_date'])
mf['sym'] = mf['ts_code'].str.replace(r'\.\w+$', '', regex=True)
mf['sm_net'] = mf['buy_sm_amount'] - mf['sell_sm_amount']
mf['md_net'] = mf['buy_md_amount'] - mf['sell_md_amount']
mf['lg_net'] = mf['buy_lg_amount'] - mf['sell_lg_amount']
mf['elg_net'] = mf['buy_elg_amount'] - mf['sell_elg_amount']
mf['total_net'] = mf['net_mf_amount']

basic['date'] = pd.to_datetime(basic['trade_date'])
basic['sym'] = basic['ts_code'].str.replace(r'\.\w+$', '', regex=True)

df = daily[['sym','date','open','high','low','close','volume']].merge(
    mf[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']], on=['sym','date'])
df = df.merge(basic[['sym','date','turnover_rate','circ_mv','total_mv']], on=['sym','date'], how='left')
df = df.sort_values(['sym','date']).reset_index(drop=True)
df = df[df['close'] > 0]
print(f"  {len(df):,} 行, {df['sym'].nunique()} 股, {df['date'].min().date()}~{df['date'].max().date()}")

# 行业
si = pro.stock_basic(exchange='', list_status='L', fields='ts_code,industry')
ind_map = dict(zip(si['ts_code'], si['industry']))

# ===== 4. 特征 =====
print("\n[4] 特征工程...")

syms = df['sym'].values
close = df['close'].values
high = df['high'].values
low = df['low'].values
vol = df['volume'].values
turnover = df['turnover_rate'].values if 'turnover_rate' in df.columns else np.full(len(df), np.nan)
circ_mv = df['circ_mv'].values if 'circ_mv' in df.columns else np.full(len(df), np.nan)
n = len(df)

sym_change = np.where(syms[1:] != syms[:-1])[0] + 1
starts = np.concatenate([[0], sym_change])
ends = np.concatenate([sym_change, [n]])

feat_names = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20',
              'atr_pct','vol_r','rsi14','macd','macd_sig','macd_hist',
              'log_circ_mv','turnover_20']
mf_cols = ['sm_net','md_net','lg_net','elg_net','total_net']
for c in mf_cols: feat_names += [f'{c}_5', f'{c}_20']

features = {name: np.full(n, np.nan) for name in feat_names}
mf_data = {c: df[c].values for c in mf_cols}

for idx in range(len(starts)):
    s, e = starts[idx], ends[idx]
    if e - s < 30: continue
    c = close[s:e]; h_ = high[s:e]; l_ = low[s:e]; v = vol[s:e]
    tr = turnover[s:e]; mv = circ_mv[s:e]
    
    features['r1'][s:e] = np.concatenate([[np.nan], np.diff(c)/c[:-1]])
    for lag in [5,10,20]:
        a = np.full(e-s, np.nan); a[lag:] = c[lag:]/c[:-lag]-1
        features[f'r{lag}'][s:e] = a
    for w in [5,10,20]:
        features[f'd{w}'][s:e] = (c-pd.Series(c).rolling(w).mean().values)/(pd.Series(c).rolling(w).mean().values+1e-10)
    ret = np.concatenate([[np.nan], np.diff(c)/(c[:-1]+1e-10)])
    features['vol5'][s:e] = pd.Series(ret).rolling(5).std().values
    features['vol20'][s:e] = pd.Series(ret).rolling(20).std().values
    features['atr_pct'][s:e] = pd.Series(h_-l_).rolling(14).mean().values/(c+1e-10)
    features['vol_r'][s:e] = pd.Series(v).rolling(5).mean().values/(pd.Series(v).rolling(20).mean().values+1)
    delta = np.concatenate([[0], np.diff(c)])
    g, l = np.maximum(delta,0), np.maximum(-delta,0)
    features['rsi14'][s:e] = 100-(100/(1+pd.Series(g).rolling(14).mean().values/(pd.Series(l).rolling(14).mean().values+1e-10)))
    e12 = pd.Series(c).ewm(span=12).mean().values; e26 = pd.Series(c).ewm(span=26).mean().values
    features['macd'][s:e] = e12-e26
    features['macd_sig'][s:e] = pd.Series(e12-e26).ewm(span=9).mean().values
    features['macd_hist'][s:e] = features['macd'][s:e]-features['macd_sig'][s:e]
    for col in mf_cols:
        vals = mf_data[col][s:e]
        features[f'{col}_5'][s:e] = pd.Series(vals).rolling(5).sum().values
        features[f'{col}_20'][s:e] = pd.Series(vals).rolling(20).sum().values
    features['log_circ_mv'][s:e] = np.log1p(mv) if not np.all(np.isnan(mv)) else np.log1p(v*c)
    features['turnover_20'][s:e] = pd.Series(tr).rolling(20).mean().values if not np.all(np.isnan(tr)) else pd.Series(v).pct_change().rolling(20).mean().values

for name, arr in features.items(): df[name] = arr
df['fwd20'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20)/x-1)
df = df.dropna(subset=feat_names + ['fwd20'])
print(f"  有效: {len(df):,} 行, {df['sym'].nunique()} 股")

# ===== 5. Walk-Forward =====
print("\n[5] Walk-Forward...")
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
    tc = tt[['date','fwd20']].copy(); tc['pred'] = pred
    ics, rics = [], []
    for d in tc['date'].unique():
        dd = tc[tc['date']==d]
        if len(dd)<20: continue
        ics.append(np.corrcoef(dd['fwd20'], dd['pred'])[0,1])
        rics.append(spearmanr(dd['fwd20'], dd['pred'])[0])
    ic, ric = np.nanmean(ics), np.nanmean(rics)
    ic_s = np.nanstd(ics)
    tc['pct'] = tc.groupby('date')['pred'].rank(pct=True)
    top = tc[tc['pct']>=0.9]['fwd20'].mean(); bot = tc[tc['pct']<=0.1]['fwd20'].mean()
    ls = top-bot
    wf.append({'fold':i+1,'ic':ic,'rank_ic':ric,'icir':ic/(ic_s+1e-10),'ls':ls})
    print(f"  F{i+1}: IC={ic:.4f} RIC={ric:.4f} LS={ls*100:.2f}%")

print(f"\n  汇总:")
for k in ['ic','rank_ic','icir','ls']:
    vals = [r[k] for r in wf if not np.isnan(r[k])]
    if vals: print(f"    {k:<12} {np.mean(vals)*100:.2f}% ± {np.std(vals)*100:.2f}%")

# ===== 6. 生产模型+信号 =====
print("\n[6] 生产模型...")
X_all = df[feat_names].values; y_all = df['fwd20'].values
prod = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42, verbosity=0)
prod.fit(X_all, y_all)
prod.save_model(f'{MODEL_DIR}/a_stock_xgb_v2.json')
print(f"  保存: {MODEL_DIR}/a_stock_xgb_v2.json")

imp = prod.feature_importances_
fi = sorted(zip(feat_names, imp), key=lambda x: -x[1])
print(f"\n  特征重要性:")
for fn, fv in fi: print(f"    {fn:<18} {fv:.4f}")

print(f"\n[7] 信号...")
latest = df['date'].max()
ldf = df[df['date']==latest].copy()
print(f"  {latest.date()}: {len(ldf)} 只")

if len(ldf) >= 15:
    ldf['score'] = prod.predict(ldf[feat_names].values)
    ldf = ldf.sort_values('score', ascending=False)
    ldf['industry'] = ldf['sym'].map(lambda x: ind_map.get(f'{x}.SZ', ind_map.get(f'{x}.SH', '?')))
    
    top15 = ldf.head(15).copy()
    top15['rank'] = range(1, 16)
    top15['expected_ret'] = top15['score'] * 100
    
    print(f"\n  🎯 A股V2 Top 15 ({latest.date()})")
    print(f"  {'#':<3} {'股票':<8} {'行业':<8} {'价格':>7} {'预期':>6} {'5d':>6} {'20d':>6}")
    print(f"  {'-'*50}")
    for _, r in top15.iterrows():
        print(f"  {r['rank']:<3} {r['sym']:<8} {r['industry']:<8} {r['close']:>7.2f} {r['expected_ret']:>5.1f}% {r['r5']*100:>5.1f}% {r['r20']*100:>5.1f}%")
    
    signal = {
        'date': str(latest.date()),
        'model': 'a_stock_xgb_v2',
        'wf_summary': {k: float(np.mean([r[k] for r in wf])) for k in ['ic','rank_ic','icir','ls']},
        'top15': top15[['sym','close','score','expected_ret','industry']].to_dict('records'),
    }
    with open(f'{OUT}/v2_signal.json', 'w') as f:
        json.dump(signal, f, indent=2, default=str)
    
    ldf[['sym','close','score','industry']].to_parquet(f'{OUT}/v2_all_scores.parquet', index=False)
    print(f"\n  信号: {OUT}/v2_signal.json")

print(f"\n总耗时: {time.time()-t0:.0f}s")
