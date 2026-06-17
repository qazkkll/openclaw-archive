"""
绿箭V7.5 深度特征挖掘 - Part 1 (数据加载+案例收集)
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import defaultdict

t0 = time.time()
print('加载数据...', flush=True)

df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
may = df[df['date'].astype(str).str[:7] == '2026-05'].copy()
may_dates = sorted(may['date'].astype(str).str[:10].unique())
may_date_set = set(may_dates)

with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet', 'r', encoding='utf-8', errors='replace') as f:
    hist = json.load(f)

with open('_green_arrow_cache.json', 'r') as f:
    cache = json.load(f)
all_day_scores = cache['scores']

print('构建价格DB...', flush=True)
df_sym_dates = df[['sym','date']].drop_duplicates().copy()
df_sym_dates.loc[:, 'datestr'] = df_sym_dates['date'].astype(str).str[:10]
all_sym_dates = df_sym_dates.groupby('sym')['datestr'].apply(lambda x: sorted(x.values)).to_dict()

price_db_full = {}
for sym in set(may['sym'].unique()):
    if sym not in hist: continue
    h = hist[sym]
    c, hi, lo = h.get('c',[]), h.get('h',[]), h.get('l',[])
    if not (c and hi and lo): continue
    fdates = all_sym_dates.get(sym, [])
    if not fdates: continue
    n_h = len(c); n_f = len(fdates)
    if n_f > n_h: continue
    offset = n_h - n_f
    pmap = {}
    for j, d in enumerate(fdates):
        idx = offset + j
        if idx < n_h:
            pmap[d] = {'close':float(c[idx]),'high':float(hi[idx]),'low':float(lo[idx]),'open':float(c[idx])}
    if pmap:
        price_db_full[sym] = pmap

report_path = '/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_report.json'
if os.path.exists(report_path):
    with open(report_path, 'r') as f: report = json.load(f)
    FEATS = report['features']
else:
    FEATS = [c for c in df.columns if c not in ['sym','date','label','fwd_5d_ret']]

print(f'特征列: {len(FEATS)}', flush=True)
print(f'价格DB: {len(price_db_full)}只股票', flush=True)

# ==== 收集案例 ====
print('\n收集案例...', flush=True)

syms_with_score = {}
for d in may_dates:
    for r in all_day_scores.get(d, []):
        if r['score'] >= 85:
            syms_with_score.setdefault(r['sym'], {})[d] = r['score']

# 买入记录
bought_set = set()
holdings = {}
for date in may_dates:
    candidates = all_day_scores.get(date, [])[:5]
    for r in candidates:
        if r['score'] >= 85:
            key = f'{r["sym"]}_{date}'
            if key not in holdings:
                dp = price_db_full.get(r['sym'], {}).get(date)
                if dp and dp['close'] > 0:
                    holdings[key] = True
                    bought_set.add(key)

# 收集爆涨/不涨案例
all_surge_cases = []
all_surge_missed = []
all_flat_cases = []

count = 0
for sym, date_scores in syms_with_score.items():
    pmap = price_db_full.get(sym, {})
    for date, score in date_scores.items():
        if date not in may_date_set: continue
        buy_idx = may_dates.index(date)
        dp = pmap.get(date)
        if not dp or dp['close'] <= 0: continue
        
        buy_price = dp['close']
        peak_ret = 0
        for d in may_dates[buy_idx:]:
            dp2 = pmap.get(d)
            if dp2 and dp2['high'] > buy_price:
                ret = (dp2['high'] - buy_price) / buy_price
                if ret > peak_ret: peak_ret = ret
        
        key = f'{sym}_{date}'
        is_bought = key in bought_set
        
        if peak_ret >= 0.30:
            case = {'sym': sym, 'date': date, 'score': score, 'peak_ret': peak_ret,
                    'buy_price': buy_price, 'bought': is_bought, 'feat_seq': []}
            for off in range(-5, 6):
                idx = buy_idx + off
                d = may_dates[idx] if 0 <= idx < len(may_dates) else None
                if d is None: continue
                dp3 = pmap.get(d)
                row = may[(may['sym']==sym) & (may['date'].astype(str).str[:10]==d)]
                entry = {'date': d, 'offset': off}
                if dp3 is not None:
                    entry.update({'close':dp3['close'],'high':dp3['high'],'low':dp3['low']})
                if len(row) > 0:
                    r = row.iloc[0]
                    for col in FEATS:
                        try:
                            v = r[col]
                            if pd.notna(v) and np.isfinite(v):
                                entry[col] = float(v)
                        except: pass
                case['feat_seq'].append(entry)
            
            if is_bought:
                all_surge_cases.append(case)
            elif len(all_surge_missed) < 200:
                all_surge_missed.append(case)
        
        elif is_bought and peak_ret < 0.05 and len(all_flat_cases) < 200:
            case = {'sym': sym, 'date': date, 'score': score, 'peak_ret': peak_ret,
                    'buy_price': buy_price, 'bought': True, 'feat_seq': []}
            for off in range(-5, 6):
                idx = buy_idx + off
                d = may_dates[idx] if 0 <= idx < len(may_dates) else None
                if d is None: continue
                dp3 = pmap.get(d)
                row = may[(may['sym']==sym) & (may['date'].astype(str).str[:10]==d)]
                entry = {'date': d, 'offset': off}
                if dp3 is not None:
                    entry.update({'close':dp3['close'],'high':dp3['high'],'low':dp3['low']})
                if len(row) > 0:
                    r = row.iloc[0]
                    for col in FEATS:
                        try:
                            v = r[col]
                            if pd.notna(v) and np.isfinite(v):
                                entry[col] = float(v)
                        except: pass
                case['feat_seq'].append(entry)
            all_flat_cases.append(case)
        
        count += 1
        if count % 5000 == 0:
            print(f'  扫描中: {count}条...', flush=True)

print(f'爆涨(买入): {len(all_surge_cases)}', flush=True)
print(f'爆涨(遗漏): {len(all_surge_missed)}', flush=True)
print(f'不涨(买入): {len(all_flat_cases)}', flush=True)

# 保存中间结果
results = {
    'surge_bought': [c for c in all_surge_cases],
    'surge_missed': [c for c in all_surge_missed],
    'flat_bought': [c for c in all_flat_cases],
    'FEATS': FEATS,
    'may_dates': may_dates
}

# 精简存储（去掉大体积特征）
for k in ['surge_bought','surge_missed','flat_bought']:
    for c in results[k]:
        if 'feat_seq' in c:
            del c['feat_seq']

with open('_deep_analysis_cache.json', 'w') as f:
    json.dump(results, f, default=str)
print('缓存保存完成', flush=True)

# 保存价格DB子集（仅爆涨票）
surge_syms = set(c['sym'] for c in all_surge_cases + all_surge_missed + all_flat_cases)
price_subset = {}
for sym in surge_syms:
    if sym in price_db_full:
        price_subset[sym] = price_db_full[sym]
with open('_price_subset.json', 'w') as f:
    json.dump(price_subset, f, default=str)

print(f'价格子集: {len(price_subset)}只股票', flush=True)
print(f'Part 1完成, 耗时: {time.time()-t0:.1f}s', flush=True)
