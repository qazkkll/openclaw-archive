"""
L50 + 评分后筛选 — 提升真实命中率
方案1: 硬过滤(RSI<55, vol<2, plus_di>15)
方案2: 软过滤(只排除明显趋势股RSI>65/vol>3.5, 仍取top5)
对比原版L50原始top5
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

print('L50 + 评分后筛选 — 提升真实命中率')
print('='*55)

model = xgb.Booster()
model.load_model(f'{MD}/us_v7_5_l50.json')
with open(f'{MD}/us_v7_5_l50_report.json') as f:
    report = json.load(f)
FEATS = report['features']

def gen_lottery_feats(df):
    d = df.copy()
    d['close_log'] = np.log1p(d['ma5'].clip(lower=0.01))
    d['close_x_vol'] = d['ma5'] * d['vol_ratio']
    d['plus_di_x_low_vol'] = d['plus_di'] * (1 / (1 + d['vol_ratio']))
    d['adx_x_rsi'] = d['adx'] * d['rsi14']
    d['bb_x_vol'] = d['bb_width'] * d['vol_ratio']
    d['rsi_x_kdj'] = d['rsi14'] * (d['k'] + d['d']) / 100
    d['low_price'] = (d['ma5'] < 3.0).astype(float)
    return d

df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]
all_dates = sorted(df['date_str'].unique())

test_dates = [
    '2024-04-01', '2024-06-03', '2024-09-03', '2024-12-02',
    '2025-01-02', '2025-03-03', '2026-04-01'
]

def get_day_score(date):
    day = df[df['date_str'] == date].copy()
    day = gen_lottery_feats(day)
    pool = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=FEATS).reset_index(drop=True)
    if len(pool) < 5:
        return []
    X = pool[FEATS].values.astype(np.float32)
    prob = model.predict(xgb.DMatrix(X, feature_names=FEATS))
    results = []
    for i in range(len(pool)):
        results.append({
            'sym': pool.iloc[i]['sym'],
            'score': float(prob[i] * 100),
            'price': float(pool.iloc[i]['ma5']),
            'fwd_5d_ret': float(pool.iloc[i].get('fwd_5d_ret', 0)),
            'rsi14': float(pool.iloc[i].get('rsi14', 50)),
            'vol_ratio': float(pool.iloc[i].get('vol_ratio', 0)),
            'plus_di': float(pool.iloc[i].get('plus_di', 0)),
        })
    results.sort(key=lambda x: -x['score'])
    return results

def find_nearest(date):
    for d in reversed(sorted([d for d in all_dates if d <= date])):
        day = df[df['date_str'] == d].copy()
        day = gen_lottery_feats(day)
        pool = day[(day['ma5'] >= 1.0) & (day['ma5'] <= 10.0)].dropna(subset=FEATS)
        if len(pool) >= 20:
            return d
    return None

# ===== 方案1: 硬过滤 =====
print('\n========== 方案1: 硬过滤 ==========')
print('条件: score>60, RSI<55, vol_ratio<2, plus_di>15')

h1_hit, h1_pick = 0, 0
for td in test_dates:
    cd = find_nearest(td)
    if not cd: continue
    res = get_day_score(cd)
    if not res: continue
    fil = [r for r in res if r['score']>60 and r['rsi14']<55 and r['vol_ratio']<2.0 and r['plus_di']>15]
    takes = fil[:5] if len(fil)>=5 else fil[:max(1,len(fil))]
    hits = sum(1 for r in takes if r['fwd_5d_ret']>0.50)
    h1_hit += hits; h1_pick += len(takes)
    print(f'  {cd}: {len(fil)}只过筛 -> top{len(takes)} -> {hits}/{len(takes)} 命中(>50%)')
    if hits>0:
        for r in takes:
            if r['fwd_5d_ret']>0.50:
                print(f'    + {r["sym"]} ${r["price"]:.2f} -> +{r["fwd_5d_ret"]*100:.0f}%')

if h1_pick>0:
    print(f'\n  总命中率: {h1_hit}/{h1_pick} = {h1_hit/h1_pick*100:.1f}%')

# ===== 方案2: 软过滤 =====
print('\n\n========== 方案2: 软过滤 ==========')
print('条件: 排除RSI>65或vol>3.5(明显趋势股), 其余按score取top5')
print('如果过筛不够5只, 从次优线补足')

h2_hit, h2_pick = 0, 0
for td in test_dates:
    cd = find_nearest(td)
    if not cd: continue
    res = get_day_score(cd)
    if not res: continue
    fil = [r for r in res if r['rsi14']<=65 and r['vol_ratio']<=3.5]
    fil.sort(key=lambda x: -x['score'])
    takes = fil[:5]
    hits = sum(1 for r in takes if r['fwd_5d_ret']>0.50)
    h2_hit += hits; h2_pick += len(takes)
    picks_str = ', '.join(f'{r["sym"]}(${r["price"]:.2f})' for r in takes)
    print(f'  {cd}: -> top5 -> {hits}/{len(takes)} 命中')
    if hits>0:
        for r in takes:
            if r['fwd_5d_ret']>0.50:
                print(f'    + {r["sym"]} ${r["price"]:.2f} -> +{r["fwd_5d_ret"]*100:.0f}%')
    print(f'    选票: {picks_str}')

if h2_pick>0:
    print(f'\n  总命中率: {h2_hit}/{h2_pick} = {h2_hit/h2_pick*100:.1f}%')

# ===== 对比原版 =====
print('\n\n========== 原版L50(无筛选) ==========')
h0_hit, h0_pick = 0, 0
for td in test_dates:
    cd = find_nearest(td)
    if not cd: continue
    res = get_day_score(cd)
    if not res: continue
    takes = res[:5]
    hits = sum(1 for r in takes if r['fwd_5d_ret']>0.50)
    h0_hit += hits; h0_pick += len(takes)

if h0_pick>0:
    print(f'  总命中率: {h0_hit}/{h0_pick} = {h0_hit/h0_pick*100:.1f}%')

# ===== 汇总 =====
print('\n\n========== 汇总对比 ==========')
print(f'  原版L50 top5:         {h0_hit}/{h0_pick} = {h0_hit/h0_pick*100:.1f}%' if h0_pick>0 else '')
print(f'  方案1(硬过滤):        {h1_hit}/{h1_pick} = {h1_hit/h1_pick*100:.1f}%' if h1_pick>0 else '')
print(f'  方案2(软过滤):        {h2_hit}/{h2_pick} = {h2_hit/h2_pick*100:.1f}%' if h2_pick>0 else '')
