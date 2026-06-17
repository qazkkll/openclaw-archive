"""
跨月验证：L20 / L30 / L50 在4个不同月份的彩票捕捉率对比
选: 2024-06, 2024-09, 2024-12, 2025-03 (不同的市场窗口)
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb
from collections import defaultdict

t0 = time.time()
print('跨月验证: 旧模型 vs L20 vs L30 vs L50')
print('='*50)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'

# 旧模型
old_model = xgb.Booster()
old_model.load_model(f'{MD}/us_v7_5.json')
old_report = json.load(open(f'{MD}/us_v7_5_report.json'))
OLD_FEATS = old_report['features']

# 彩票模型
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
lt = df.copy()
lt['close_log'] = np.log1p(lt['ma5'].clip(lower=0.01))
lt['close_x_vol'] = lt['ma5'] * lt['vol_ratio']
lt['plus_di_x_low_vol'] = lt['plus_di'] * (1/(1+lt['vol_ratio']))
lt['adx_x_rsi'] = lt['adx'] * lt['rsi14']
lt['bb_x_vol'] = lt['bb_width'] * lt['vol_ratio']
lt['rsi_x_kdj'] = lt['rsi14'] * (lt['k'] + lt['d']) / 100
lt['low_price'] = (lt['ma5'] < 3.0).astype(float)
ALL_LO_FEATS = reports['L20']['features']

# 测试月份
test_months = ['2024-06', '2024-09', '2024-12', '2025-03']

results = {}
for ym in test_months:
    print(f'\n{"="*50}')
    print(f'{ym}:')
    print('='*50)
    
    month = lt[lt['date_str'].str[:7] == ym].copy()
    month_lt10 = month[month['ma5'] < 10].reset_index(drop=True)
    print(f'  $1-10行数: {len(month_lt10)}')
    
    # 按日期分组
    dates = sorted(month_lt10['date_str'].unique())
    
    # 提前计算每天哪些票"会爆涨"（>50%）
    month_lt10['will_surge50'] = (month_lt10['fwd_5d_ret'] > 0.50).astype(int)
    month_lt10['will_surge30'] = (month_lt10['fwd_5d_ret'] > 0.30).astype(int)
    month_lt10['will_surge20'] = (month_lt10['fwd_5d_ret'] > 0.20).astype(int)
    
    surge_by_date = {}
    for d in dates:
        day = month_lt10[month_lt10['date_str'] == d]
        surge_by_date[d] = {
            's50': set(day[day['will_surge50']==1]['sym'].values),
            's30': set(day[day['will_surge30']==1]['sym'].values),
            's20': set(day[day['will_surge20']==1]['sym'].values),
        }
    
    # 逐日评分 (旧模型 + 彩票模型)
    totals = {ver: {'s50_t5':0,'s50_t10':0,'s30_t5':0,'s30_t10':0,'s20_t5':0,'s20_t10':0} 
              for ver in ['OLD'] + [f'L{x}' for x in [20,30,50]]}
    
    for d in dates:
        day = month_lt10[month_lt10['date_str'] == d]
        if len(day) < 5:
            continue
        surge = surge_by_date[d]
        
        # ---- 旧模型 ----
        clean_old = day.dropna(subset=OLD_FEATS).reset_index(drop=True)
        if len(clean_old) > 5:
            X_old = clean_old[OLD_FEATS].values.astype(np.float32)
            mo = old_model.predict(xgb.DMatrix(X_old, feature_names=OLD_FEATS), output_margin=True)
            zo = (mo - np.mean(mo)) / max(np.std(mo), 0.001)
            sc = 100/(1+np.exp(-zo*1.5))
            t5 = set(clean_old.iloc[i]['sym'] for i in np.argsort(-sc)[:5])
            t10 = set(clean_old.iloc[i]['sym'] for i in np.argsort(-sc)[:10])
            for lbl, sk in [('s50','s50'),('s30','s30'),('s20','s20')]:
                totals['OLD'][f'{sk}_t5'] += len(surge[lbl] & t5)
                totals['OLD'][f'{sk}_t10'] += len(surge[lbl] & t10)
        
        # ---- 彩票模型 ----
        for ver in ['L20', 'L30', 'L50']:
            feats = reports[ver]['features']
            clean = day.dropna(subset=feats).reset_index(drop=True)
            if len(clean) < 5: continue
            X = clean[feats].values.astype(np.float32)
            prob = models[ver].predict(xgb.DMatrix(X, feature_names=feats))
            t5 = set(clean.iloc[i]['sym'] for i in np.argsort(-prob)[:5])
            t10 = set(clean.iloc[i]['sym'] for i in np.argsort(-prob)[:10])
            for lbl, sk in [('s50','s50'),('s30','s30'),('s20','s20')]:
                totals[ver][f'{sk}_t5'] += len(surge[lbl] & t5)
                totals[ver][f'{sk}_t10'] += len(surge[lbl] & t10)
    
    # 输出
    print(f'\n{"模型":>8s} | 标签50%: Top5/Top10 | 标签30%: Top5/Top10 | 标签20%: Top5/Top10')
    print('-'*70)
    for ver in ['OLD', 'L20', 'L30', 'L50']:
        t = totals[ver]
        s50_t5, s50_t10 = t['s50_t5'], t['s50_t10']
        s30_t5, s30_t10 = t['s30_t5'], t['s30_t10']
        s20_t5, s20_t10 = t['s20_t5'], t['s20_t10']
        print(f'  {ver:>6s} |  {s50_t5:>2d}/{s50_t10:<3d}          |  {s30_t5:>2d}/{s30_t10:<3d}          |  {s20_t5:>2d}/{s20_t10:<3d}')
    
    # 总爆涨事件
    total_s50 = sum(len(s['s50']) for s in surge_by_date.values())
    total_s30 = sum(len(s['s30']) for s in surge_by_date.values())
    total_s20 = sum(len(s['s20']) for s in surge_by_date.values())
    print(f'\n  总爆涨事件: >50%={total_s50}  >30%={total_s30}  >20%={total_s20}')
    
    results[ym] = {'totals': totals, 'events': {'s50':total_s50,'s30':total_s30,'s20':total_s20}}

# ===== 最终汇总 =====
print('\n\n' + '='*60)
print('跨月汇总: 旧模型 vs 彩票模型 对比')
print(f'{"月份":>10s} | {"旧 Top5":>8s} | {"L20 T5":>6s} | {"L30 T5":>6s} | {"L50 T5":>6s} | {"标":>10s}')
print('-'*60)

for ym in test_months:
    t = results[ym]['totals']
    e = results[ym]['events']
    print(f'{ym:>10s} | {t["OLD"]["s50_t5"]:>3d}/{e["s50"]:<3d} | {t["L20"]["s50_t5"]:>3d}/{e["s50"]:<3d} | {t["L30"]["s50_t5"]:>3d}/{e["s50"]:<3d} | {t["L50"]["s50_t5"]:>3d}/{e["s50"]:<3d} | >50%')
    print(f'{"":>10s} | {t["OLD"]["s30_t5"]:>3d}/{e["s30"]:<3d} | {t["L20"]["s30_t5"]:>3d}/{e["s30"]:<3d} | {t["L30"]["s30_t5"]:>3d}/{e["s30"]:<3d} | {t["L50"]["s30_t5"]:>3d}/{e["s30"]:<3d} | >30%')
    print(f'{"":>10s} | {t["OLD"]["s20_t5"]:>3d}/{e["s20"]:<3d} | {t["L20"]["s20_t5"]:>3d}/{e["s20"]:<3d} | {t["L30"]["s20_t5"]:>3d}/{e["s20"]:<3d} | {t["L50"]["s20_t5"]:>3d}/{e["s20"]:<3d} | >20%')
    print()

print(f'\n⏱️ 耗时: {time.time()-t0:.1f}s')
