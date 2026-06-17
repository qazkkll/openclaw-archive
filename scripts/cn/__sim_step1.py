"""
Step 1: 加载数据，构建价格映射，评分整个5月，结果缓存到JSON
优化版：groupby一次性提取日期
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import xgboost as xgb

t0 = time.time()
print('=== Step 1: 加载数据 ===', flush=True)

# 1. 模型
print('加载模型...', flush=True)
model = xgb.Booster()
model.load_model('/home/hermes/.hermes/openclaw-project/data/models/us_v7_5.json')

# 2. 特征数据
print('加载特征数据...', flush=True)
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
may = df[df['date'].astype(str).str[:7] == '2026-05'].copy()
may_dates = sorted(may['date'].astype(str).str[:10].unique())
print(f'5月: {len(may_dates)}天, {len(may)}行', flush=True)
print(f'日期: {may_dates[0]} ~ {may_dates[-1]}', flush=True)

# 3. 历史K线
print('加载历史K线...', flush=True)
with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet', 'r', encoding='utf-8', errors='replace') as f:
    hist = json.load(f)

# 4. 价格映射 - groupby一次性提取
print('提取每只股票的日期序列...', flush=True)
df_sym_dates = df[['sym','date']].drop_duplicates().copy()
df_sym_dates.loc[:, 'datestr'] = df_sym_dates['date'].astype(str).str[:10]
all_feat_dates = df_sym_dates.groupby('sym')['datestr'].apply(lambda x: sorted(x.values)).to_dict()
print(f'提取完成: {len(all_feat_dates)}只股票', flush=True)

print('构建价格映射...', flush=True)
syms = set(may['sym'].unique())
price_db = {}
missing = 0

for sym in syms:
    if sym not in hist:
        missing += 1
        continue
    h = hist[sym]
    c = h.get('c',[]); hi = h.get('h',[]); lo = h.get('l',[])
    if not (c and hi and lo):
        missing += 1
        continue
    feat_dates = all_feat_dates.get(sym)
    if not feat_dates:
        missing += 1
        continue
    n_feat = len(feat_dates)
    n_hist = len(c)
    if n_feat > n_hist:
        missing += 1
        continue
    offset = n_hist - n_feat
    pmap = {}
    ok = True
    for j, d in enumerate(feat_dates):
        idx = offset + j
        if idx < n_hist:
            pmap[d] = {'close': float(c[idx]), 'high': float(hi[idx]), 'low': float(lo[idx])}
        else:
            ok = False
            break
    if ok and pmap:
        price_db[sym] = pmap
    else:
        missing += 1

print(f'有价格: {len(price_db)}/{len(syms)}, 缺失: {missing}', flush=True)
print(f'耗时: {time.time()-t0:.1f}s', flush=True)

# 5. 特征列
report_path = '/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_report.json'
if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        report = json.load(f)
    FEATS = report['features']
else:
    FEATS = [c for c in df.columns if c not in ['sym','date','label','fwd_5d_ret']]
print(f'特征列: {len(FEATS)}', flush=True)

# 6. 逐日评分
print('逐日评分...', flush=True)
all_day_scores = {}
for date in may_dates:
    day_data = may[may['date'].astype(str).str[:10]==date].copy()
    day_data = day_data.dropna(subset=FEATS)
    if len(day_data)==0:
        all_day_scores[date] = []
        print(f'  {date}: 0只（无有效特征）', flush=True)
        continue

    X = day_data[FEATS].values.astype(float)
    dmat = xgb.DMatrix(X, feature_names=FEATS)
    raw_margin = model.predict(dmat, output_margin=True)

    mean_m = np.mean(raw_margin)
    std_m = np.std(raw_margin)
    z = (raw_margin - mean_m) / max(std_m, 0.001)
    scores = 100 / (1 + np.exp(-z * 1.5))
    scores = np.clip(scores, 0, 100)

    records = [{'sym': s, 'score': round(float(sc), 1)} for s, sc in zip(day_data['sym'].values, scores)]
    records.sort(key=lambda x: -x['score'])
    all_day_scores[date] = records

    n85 = sum(1 for r in records if r['score']>=85)
    n70 = sum(1 for r in records if r['score']>=70)
    print(f'  {date}: {len(records)}只, >=85: {n85}, >=70: {n70}', flush=True)

print(f'评分完成，总耗时: {time.time()-t0:.1f}s', flush=True)
print('Step 1 完成。接下来运行 Step 2 进行交易模拟。', flush=True)
