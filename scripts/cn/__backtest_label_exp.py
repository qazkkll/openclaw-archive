"""
标签实验 — 5月逐日回溯对比
对比: L20 / L30 / L50 / L-s20 的彩票捕捉率
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from collections import defaultdict

t0 = time.time()
print('标签实验 5月回溯对比')
print('='*50)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

versions = ['L20', 'L30', 'L50', 'L-s20']
models = {}
reports = {}

for ver in versions:
    m = xgb.Booster()
    m.load_model(f'{MD}/us_v7_5_{ver.lower()}.json')
    r = json.load(open(f'{MD}/us_v7_5_{ver.lower()}_report.json'))
    models[ver] = m
    reports[ver] = r
    print(f'{ver}: {len(r["features"])} feat, best_iter={r["best_iteration"]}, val_auc={r["val_auc"]:.4f}')

# 加载数据
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

with open('_deep_analysis_cache.json') as f:
    cache = json.load(f)
surge_all = cache['surge_bought'] + cache['surge_missed']
surge_by_date = defaultdict(list)
for c in surge_all:
    surge_by_date[c['date']].append(c)
may_dates = cache['may_dates']

# 彩票特征生成
def gen_feats(day_df):
    d = day_df.copy()
    d['close_log'] = np.log1p(d['ma5'].clip(lower=0.01))
    d['close_x_vol'] = d['ma5'] * d['vol_ratio']
    d['plus_di_x_low_vol'] = d['plus_di'] * (1/(1+d['vol_ratio']))
    d['adx_x_rsi'] = d['adx'] * d['rsi14']
    d['bb_x_vol'] = d['bb_width'] * d['vol_ratio']
    d['rsi_x_kdj'] = d['rsi14'] * (d['k'] + d['d']) / 100
    d['low_price'] = (d['ma5'] < 3.0).astype(float)
    return d

# 逐日评分
print('\n逐日评分...')
day_results = []

for d in may_dates:
    day = d[:10]
    today = df[df['date_str'] == day].copy()
    if len(today) == 0:
        continue
    
    lot = gen_feats(today)
    lot = lot[(lot['ma5']>=1)&(lot['ma5']<=10)].reset_index(drop=True)
    if len(lot) < 10:
        continue
    
    day_surges = surge_by_date.get(d, [])
    surge_syms = set(c['sym'] for c in day_surges if c['buy_price'] < 10)
    
    row = {'date': day, 'n_surge': len(surge_syms)}
    
    for ver in versions:
        feats = reports[ver]['features']
        avail = [c for c in feats if c in lot.columns]
        clean = lot.dropna(subset=avail).reset_index(drop=True)
        if len(clean) == 0:
            row[ver] = {'top5': 0, 'top10': 0, 'captured': []}
            continue
        X = clean[avail].values.astype(np.float32)
        prob = models[ver].predict(xgb.DMatrix(X, feature_names=avail))
        t5 = set(clean.iloc[i]['sym'] for i in np.argsort(-prob)[:5])
        t10 = set(clean.iloc[i]['sym'] for i in np.argsort(-prob)[:10])
        row[ver] = {
            'top5': len(surge_syms & t5),
            'top10': len(surge_syms & t10),
            'captured': list(surge_syms & t5),
        }
    
    day_results.append(row)
    
    if len(day_results) % 5 == 0:
        print(f'  {day}: {len(surge_syms)}爆涨 | '
              + ' | '.join(f'{ver}:t5={row[ver]["top5"]}' for ver in versions))

# ===== 汇总 =====
print('\n' + '='*50)
print('汇总（5月 $1-10 爆涨捕捉）')
print('='*50)

total_surge = sum(r['n_surge'] for r in day_results)
print(f'总爆涨事件: {total_surge}笔\n')

print(f'{"模型":>8s}   {"Top5捕获":>10s}   {"Top5率":>8s}   {"Top10捕获":>10s}   {"Top10率":>8s}')
print('-'*55)

for ver in versions:
    t5 = sum(r[ver]['top5'] for r in day_results)
    t10 = sum(r[ver]['top10'] for r in day_results)
    print(f'  {ver:>6s}   {t5:>10d}   {t5/total_surge*100:>7.1f}%   {t10:>10d}   {t10/total_surge*100:>7.1f}%')

print()
# 两两对比
for v1, v2 in [('L20','L30'), ('L20','L50'), ('L30','L50'), ('L-s20','L30')]:
    t5_1 = sum(r[v1]['top5'] for r in day_results)
    t5_2 = sum(r[v2]['top5'] for r in day_results)
    t10_1 = sum(r[v1]['top10'] for r in day_results)
    t10_2 = sum(r[v2]['top10'] for r in day_results)
    print(f'  {v1} vs {v2}: Top5 {t5_1-t5_2:+d} | Top10 {t10_1-t10_2:+d}')

# 每日明细
print('\n逐日明细:')
print(f'{"日期":>10s}  {"爆涨":>4s}  ', end='')
for ver in versions:
    print(f'  {ver+"_t5":>10s}   {"top5明细":>20s}', end='')
print()
for r in day_results:
    print(f'{r["date"]:>10s}  {r["n_surge"]:>4d}  ', end='')
    for ver in versions:
        s = f'{r[ver]["top5"]}/{r[ver]["top10"]}'
        c = ','.join(r[ver]['captured'])
        print(f'  {s:>10s}   {c!s:>20s}', end='')
    print()

print(f'\n⏱️ 耗时: {time.time()-t0:.1f}s')
