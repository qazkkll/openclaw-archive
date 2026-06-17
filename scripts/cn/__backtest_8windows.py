"""
跨周期验证 — 8个市场窗口
旧模型 vs L20 vs L30 vs L50
选: 2018-12(熊底), 2020-03(崩盘反弹), 2021-01(小盘牛市顶),
     2022-09(极度低迷), 2023-01(反弹), 2024-06(正常), 
     2024-12(小盘牛), 2026-04(近期)
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from collections import defaultdict

t0 = time.time()
print('跨周期验证: 8个窗口 x 4个模型')
print('='*55)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# 加载模型
old_model = xgb.Booster()
old_model.load_model(f'{MD}/us_v7_5.json')
old_report = json.load(open(f'{MD}/us_v7_5_report.json'))

models = {}
reports = {}
for ver in ['L20', 'L30', 'L50']:
    m = xgb.Booster()
    m.load_model(f'{MD}/us_v7_5_{ver.lower()}.json')
    r = json.load(open(f'{MD}/us_v7_5_{ver.lower()}_report.json'))
    models[ver] = m
    reports[ver] = r

# 加载数据
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

# 预计算特征
lt = df.copy()
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1/(1+lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)

OLD_FEATS = old_report['features']
LOT_FEATS = reports['L50']['features']

# 测试窗口
windows = [
    ('2018-12', '2018-12-01', '2018-12-31', '熊市底部'),
    ('2020-03', '2020-03-01', '2020-03-31', 'COVID崩盘反弹'),
    ('2021-01', '2021-01-01', '2021-01-31', '小盘牛市顶'),
    ('2022-09', '2022-09-01', '2022-09-30', '极度低迷'),
    ('2023-01', '2023-01-01', '2023-01-31', '反弹期'),
    ('2024-09', '2024-09-01', '2024-09-30', '正常/震荡'),
    ('2024-12', '2024-12-01', '2024-12-31', '小盘牛市'),
    ('2026-04', '2026-04-01', '2026-04-30', '近期'),
]

def score_day_old(day_df):
    """旧模型 - $1-10中取top5"""
    clean = day_df.dropna(subset=OLD_FEATS).reset_index(drop=True)
    if len(clean) < 5: return set()
    X = clean[OLD_FEATS].values.astype(np.float32)
    mo = old_model.predict(xgb.DMatrix(X, feature_names=OLD_FEATS), output_margin=True)
    zo = (mo - np.mean(mo)) / max(np.std(mo), 0.001)
    sc = 100/(1+np.exp(-zo*1.5))
    return set(clean.iloc[i]['sym'] for i in np.argsort(-sc)[:5])

def score_day(day_df, ver):
    """彩票模型"""
    feats = reports[ver]['features']
    clean = day_df.dropna(subset=feats).reset_index(drop=True)
    if len(clean) < 5: return set()
    X = clean[feats].values.astype(np.float32)
    prob = models[ver].predict(xgb.DMatrix(X, feature_names=feats))
    return set(clean.iloc[i]['sym'] for i in np.argsort(-prob)[:5])

all_results = {}

for name, start, end, desc in windows:
    print(f'\n{"─"*55}')
    print(f'{name} ({desc}): {start} ~ {end}')
    print('─'*55)
    
    month = lt[(lt['date_str'] >= start) & (lt['date_str'] <= end)].copy()
    month_lt10 = month[month['ma5'] < 10].reset_index(drop=True)
    dates = sorted(month_lt10['date_str'].unique())
    
    # 计算未来爆发标签
    month_lt10['w_s50'] = (month_lt10['fwd_5d_ret'] > 0.50).astype(int)
    month_lt10['w_s30'] = (month_lt10['fwd_5d_ret'] > 0.30).astype(int)
    month_lt10['w_s20'] = (month_lt10['fwd_5d_ret'] > 0.20).astype(int)
    
    surge_by_date = {}
    for d in dates:
        day = month_lt10[month_lt10['date_str'] == d]
        surge_by_date[d] = {
            's50': set(day[day['w_s50']==1]['sym'].values),
            's30': set(day[day['w_s30']==1]['sym'].values),
            's20': set(day[day['w_s20']==1]['sym'].values),
        }
    
    totals = {ver: {'s50':0,'s30':0,'s20':0} for ver in ['OLD','L20','L30','L50']}
    
    for d in dates:
        day = month_lt10[month_lt10['date_str'] == d]
        surge = surge_by_date[d]
        
        old_t5 = score_day_old(day)
        totals['OLD']['s50'] += len(surge['s50'] & old_t5)
        totals['OLD']['s30'] += len(surge['s30'] & old_t5)
        totals['OLD']['s20'] += len(surge['s20'] & old_t5)
        
        for ver in ['L20','L30','L50']:
            t5 = score_day(day, ver)
            totals[ver]['s50'] += len(surge['s50'] & t5)
            totals[ver]['s30'] += len(surge['s30'] & t5)
            totals[ver]['s20'] += len(surge['s20'] & t5)
    
    total_s50 = sum(len(v['s50']) for v in surge_by_date.values())
    total_s30 = sum(len(v['s30']) for v in surge_by_date.values())
    
    print(f'  总爆涨: >50%={total_s50}  >30%={total_s30}')
    print(f'  {"模型":>6s}  |  >50%捕获  |  >30%捕获')
    print(f'  {"-"*30}')
    for ver in ['OLD','L20','L30','L50']:
        t = totals[ver]
        print(f'  {ver:>6s}  |  {t["s50"]:>3d}/{total_s50:<3d} ({t["s50"]/max(total_s50,1)*100:>4.1f}%) | {t["s30"]:>3d}/{total_s30:<3d} ({t["s30"]/max(total_s30,1)*100:>4.1f}%)')
    
    all_results[name] = {
        'totals': totals,
        'events': {'s50': total_s50, 's30': total_s30},
        'desc': desc,
    }

# ===== 最终汇总 =====
print('\n\n' + '='*60)
print('最终跨周期汇总 — Old vs L20 vs L30 vs L50')
print(f'{"窗口":>10s}  {"描述":>16s}  {"Old t5":>8s}  {"L20 t5":>8s}  {"L30 t5":>8s}  {"L50 t5":>8s}')
print('='*70)

for name in [w[0] for w in windows]:
    r = all_results[name]
    t = r['totals']
    e = r['events']['s50']
    print(f'{name:>10s}  {r["desc"]:>16s}  {t["OLD"]["s50"]:>3d}/{e:<3d}  {t["L20"]["s50"]:>3d}/{e:<3d}  {t["L30"]["s50"]:>3d}/{e:<3d}  {t["L50"]["s50"]:>3d}/{e:<3d}')

# 总计
print()
print('总计:')
t_old = sum(r['totals']['OLD']['s50'] for r in all_results.values())
t_l20 = sum(r['totals']['L20']['s50'] for r in all_results.values())
t_l30 = sum(r['totals']['L30']['s50'] for r in all_results.values())
t_l50 = sum(r['totals']['L50']['s50'] for r in all_results.values())
t_all = sum(r['events']['s50'] for r in all_results.values())
print(f'  总>50%事件: {t_all}')
print(f'  旧模型捕获: {t_old} ({t_old/t_all*100:.1f}%)')
print(f'  L20捕获:    {t_l20} ({t_l20/t_all*100:.1f}%)')
print(f'  L30捕获:    {t_l30} ({t_l30/t_all*100:.1f}%)')
print(f'  L50捕获:    {t_l50} ({t_l50/t_all*100:.1f}%)')

print(f'\n⏱️ 耗时: {time.time()-t0:.1f}s')
