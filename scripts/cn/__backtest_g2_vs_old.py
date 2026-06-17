"""
G2 vs 旧模型 — 5月逐日回溯对比
目标：比较两个模型在捕捉爆涨彩票股上的实际表现
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

t0 = time.time()
print('G2 vs 旧模型 5月逐日对比')
print('='*50)

MD = '/home/hermes/.hermes/openclaw-project/data/models'
ML = '/home/hermes/.hermes/openclaw-archive/scripts/system'
DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'

# ===== 加载两个模型 =====
old_model = xgb.Booster()
old_model.load_model(f'{MD}/us_v7_5.json')
old_report = json.load(open(f'{MD}/us_v7_5_report.json'))
OLD_FEATS = old_report['features']

g2_model = xgb.Booster()
g2_model.load_model(f'{MD}/us_v7_5_g2.json')
g2_report = json.load(open(f'{MD}/us_v7_5_g2_report.json'))
G2_FEATS = g2_report['features']

print(f'旧模型: {len(OLD_FEATS)} 特征, best_iter={old_report.get("best_iteration","N/A")}')
print(f'G2模型: {len(G2_FEATS)} 特征, best_iter={g2_report["best_iteration"]}')

# ===== 加载数据 =====
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

# ===== 加载5月爆涨记录 =====
with open('_deep_analysis_cache.json') as f:
    cache = json.load(f)
surge_bought = cache['surge_bought']
surge_missed = cache['surge_missed']
flat_bought = cache['flat_bought']
surge_all = surge_bought + surge_missed  # 所有5月爆涨案例

# 爆涨按日期分组
surge_by_date = defaultdict(list)
for c in surge_all:
    surge_by_date[c['date']].append(c)

may_dates = cache['may_dates']

# ===== 逐日评分 =====
print('\n逐日评分中...')
day_results = []

for di, d in enumerate(may_dates):
    day = d[:10]
    today_data = df[df['date_str'] == day].copy()
    if len(today_data) == 0:
        continue

    # ---- 旧模型评分 ----
    clean_old = today_data.dropna(subset=OLD_FEATS).reset_index(drop=True)
    if len(clean_old) == 0:
        continue
    X_old = clean_old[OLD_FEATS].values.astype(np.float32)
    dmat_old = xgb.DMatrix(X_old, feature_names=OLD_FEATS)
    margin_old = old_model.predict(dmat_old, output_margin=True)
    
    # sigmoid映射0-100
    z_old = (margin_old - np.mean(margin_old)) / max(np.std(margin_old), 0.001)
    scores_old = 100 / (1 + np.exp(-z_old * 1.5))
    
    old_ranked = sorted(
        [{'sym': clean_old.iloc[i]['sym'], 'score': float(scores_old[i]), 'margin': float(margin_old[i])}
         for i in range(len(clean_old))],
        key=lambda x: -x['score']
    )
    old_top5 = set(r['sym'] for r in old_ranked[:5])
    old_top10 = set(r['sym'] for r in old_ranked[:10])

    # ---- G2模型评分 ----
    # 生成新特征 (从ma5和其他已有特征)
    today_data_g2 = today_data.copy()
    today_data_g2['close_log'] = np.log1p(today_data_g2['ma5'].clip(lower=0.01))
    today_data_g2['close_x_vol_ratio'] = today_data_g2['ma5'] * today_data_g2['vol_ratio']
    today_data_g2['plus_di_x_low_vol'] = today_data_g2['plus_di'] * (1 / (1 + today_data_g2['vol_ratio']))
    today_data_g2['adx_x_rsi'] = today_data_g2['adx'] * today_data_g2['rsi14']
    
    clean_g2 = today_data_g2.dropna(subset=G2_FEATS).reset_index(drop=True)
    if len(clean_g2) == 0:
        continue
    X_g2 = clean_g2[G2_FEATS].values.astype(np.float32)
    dmat_g2 = xgb.DMatrix(X_g2, feature_names=G2_FEATS)
    margin_g2 = g2_model.predict(dmat_g2, output_margin=True)
    
    z_g2 = (margin_g2 - np.mean(margin_g2)) / max(np.std(margin_g2), 0.001)
    scores_g2 = 100 / (1 + np.exp(-z_g2 * 1.5))
    
    g2_ranked = sorted(
        [{'sym': clean_g2.iloc[i]['sym'], 'score': float(scores_g2[i]), 'margin': float(margin_g2[i])}
         for i in range(len(clean_g2))],
        key=lambda x: -x['score']
    )
    g2_top5 = set(r['sym'] for r in g2_ranked[:5])
    g2_top10 = set(r['sym'] for r in g2_ranked[:10])

    # ---- 当日爆涨票 ----
    day_surges = surge_by_date.get(d, [])
    surge_syms = set(c['sym'] for c in day_surges)
    surge_count = len(surge_syms)

    # ---- 捕捉统计 ----
    old_captured_5 = surge_syms & old_top5
    old_captured_10 = surge_syms & old_top10
    g2_captured_5 = surge_syms & g2_top5
    g2_captured_10 = surge_syms & g2_top10

    # 按评分排序（旧模型）看爆涨票的排名
    old_missed_ranks = []
    for c in day_surges:
        sym = c['sym']
        if sym in old_top5:
            old_missed_ranks.append(0)  # 捕捉到
        else:
            rank = next((i+1 for i, r in enumerate(old_ranked[:50]) if r['sym'] == sym), 99)
            old_missed_ranks.append(rank)

    avg_old_rank = np.mean([r for r in old_missed_ranks if r > 0]) if any(r>0 for r in old_missed_ranks) else 0

    # G2的遗漏排名
    g2_missed_ranks = []
    for c in day_surges:
        sym = c['sym']
        if sym in g2_top5:
            g2_missed_ranks.append(0)
        else:
            rank = next((i+1 for i, r in enumerate(g2_ranked[:50]) if r['sym'] == sym), 99)
            g2_missed_ranks.append(rank)

    avg_g2_rank = np.mean([r for r in g2_missed_ranks if r > 0]) if any(r>0 for r in g2_missed_ranks) else 0

    old_missed_count = sum(1 for r in old_missed_ranks if r > 0)
    g2_missed_count = sum(1 for r in g2_missed_ranks if r > 0)

    day_results.append({
        'date': d[:10],
        'n_surge': surge_count,
        'old_top5': len(old_captured_5),
        'old_top10': len(old_captured_10),
        'g2_top5': len(g2_captured_5),
        'g2_top10': len(g2_captured_10),
        'old_missed_avg_rank': round(avg_old_rank, 1),
        'g2_missed_avg_rank': round(avg_g2_rank, 1),
        'surge_syms': list(surge_syms),
        'old_captured_syms': list(old_captured_5),
        'g2_captured_syms': list(g2_captured_5),
    })
    
    # 进度
    if (di + 1) % 5 == 0 or di == len(may_dates) - 1:
        print(f'  {d[:10]} → 爆涨{surge_count} | 旧top5捕获{len(old_captured_5)} G2top5捕获{len(g2_captured_5)}')
print(f'评分完成: {len(day_results)}个交易日')

# ===== 汇总统计 =====
print('\n' + '='*50)
print('汇总对比')
print('='*50)

total_surge = sum(r['n_surge'] for r in day_results)
total_old5 = sum(r['old_top5'] for r in day_results)
total_old10 = sum(r['old_top10'] for r in day_results)
total_g25 = sum(r['g2_top5'] for r in day_results)
total_g210 = sum(r['g2_top10'] for r in day_results)

print(f'\n5月爆涨事件总数: {total_surge}笔')
print(f'\n{"捕获":>20s}   {"旧model Top5":>15s}   {"G2 Top5":>10s}   {"变化":>8s}')
print(f'  {"-"*55}')
print(f'  {"top5捕获(笔)":>20s}   {total_old5:>15d}   {total_g25:>10d}   {total_g25-total_old5:>+8d}')
print(f'  {"top5捕获率":>20s}   {total_old5/total_surge*100:>14.1f}%   {total_g25/total_surge*100:>9.1f}%   {(total_g25-total_old5)/total_surge*100:>+7.1f}%')
print(f'  {"top10捕获(笔)":>20s}   {total_old10:>15d}   {total_g210:>10d}   {total_g210-total_old10:>+8d}')
print(f'  {"top10捕获率":>20s}   {total_old10/total_surge*100:>14.1f}%   {total_g210/total_surge*100:>9.1f}%   {(total_g210-total_old10)/total_surge*100:>+7.1f}%')

# 遗漏排名对比
old_ranks = [r['old_missed_avg_rank'] for r in day_results if r['old_missed_avg_rank'] > 0]
g2_ranks = [r['g2_missed_avg_rank'] for r in day_results if r['g2_missed_avg_rank'] > 0]
print(f'\n没被top5捕获的爆涨票平均排名:')
print(f'  旧模型: {np.mean(old_ranks):.1f} (基于{len(old_ranks)}天)')
print(f'  G2模型: {np.mean(g2_ranks):.1f} (基于{len(g2_ranks)}天)')

# 具体漏网鱼对比
print('\n\n爆涨票被旧模型漏掉但G2模型捕获的案例:')
win_count = 0
for r in day_results:
    old_missed_only = set(r['surge_syms']) - set(r['old_captured_syms'])
    g2_captured_only = set(r['g2_captured_syms']) - set(r['old_captured_syms'])
    new_captures = old_missed_only & g2_captured_only
    if new_captures:
        win_count += len(new_captures)
        for sym in new_captures:
            print(f'  {r["date"]} {sym}: 旧模型遗漏 → G2捕获')

print(f'G2模型额外捕获的爆涨票数量: {win_count}')

# 反之: 旧模型捕获但G2遗漏的
print('\n\n爆涨票被旧模型捕获但G2模型遗漏的案例:')
lost_count = 0
for r in day_results:
    g2_missed_only = set(r['surge_syms']) - set(r['g2_captured_syms'])
    old_captured_only = set(r['old_captured_syms']) - set(r['g2_captured_syms'])
    lost_captures = g2_missed_only & old_captured_only
    if lost_captures:
        lost_count += len(lost_captures)
        for sym in lost_captures:
            print(f'  {r["date"]} {sym}: 旧模型捕获 → G2遗漏')

print(f'G2模型丢失的爆涨票数量: {lost_count}')
print(f'净改善: +{win_count - lost_count}')

# ===== 6月前三周对比 =====
print('\n\n' + '='*50)
print('6月1-12日前瞻对比')
print('='*50)

jun_dates = sorted(df[df['date_str'].str[:7] == '2026-06']['date_str'].unique())
print(f'6月可用数据: {jun_dates}')
for jd in jun_dates:
    jun_day = df[df['date_str'] == jd].copy()
    # 旧模型
    clean_old = jun_day.dropna(subset=OLD_FEATS).reset_index(drop=True)
    if len(clean_old) < 50: continue
    X_old = clean_old[OLD_FEATS].values.astype(np.float32)
    dmo = xgb.DMatrix(X_old, feature_names=OLD_FEATS)
    mo = old_model.predict(dmo, output_margin=True)
    zo = (mo - np.mean(mo)) / max(np.std(mo), 0.001)
    so = 100 / (1 + np.exp(-zo * 1.5))
    old_t5 = set(clean_old.iloc[i]['sym'] for i in np.argsort(-so)[:5])
    
    # G2
    jun_day_g2 = jun_day.copy()
    jun_day_g2['close_log'] = np.log1p(jun_day_g2['ma5'].clip(lower=0.01))
    jun_day_g2['close_x_vol_ratio'] = jun_day_g2['ma5'] * jun_day_g2['vol_ratio']
    jun_day_g2['plus_di_x_low_vol'] = jun_day_g2['plus_di'] * (1 / (1 + jun_day_g2['vol_ratio']))
    jun_day_g2['adx_x_rsi'] = jun_day_g2['adx'] * jun_day_g2['rsi14']

    clean_g2 = jun_day_g2.dropna(subset=G2_FEATS).reset_index(drop=True)
    if len(clean_g2) < 50: continue
    X_g2 = clean_g2[G2_FEATS].values.astype(np.float32)
    dmg = xgb.DMatrix(X_g2, feature_names=G2_FEATS)
    mg = g2_model.predict(dmg, output_margin=True)
    zg = (mg - np.mean(mg)) / max(np.std(mg), 0.001)
    sg = 100 / (1 + np.exp(-zg * 1.5))
    g2_t5 = set(clean_g2.iloc[i]['sym'] for i in np.argsort(-sg)[:5])

    # 重叠
    common = old_t5 & g2_t5
    old_only = old_t5 - g2_t5
    g2_only = g2_t5 - old_t5
    
    # 价格特征
    old_prices = []
    for sym in old_t5:
        row = clean_old[clean_old['sym'] == sym]
        if len(row) > 0:
            old_prices.append(float(row.iloc[0].get('ma5', 0)))
    g2_prices = []
    for sym in g2_t5:
        row = clean_g2[clean_g2['sym'] == sym]
        if len(row) > 0:
            g2_prices.append(float(row.iloc[0].get('ma5', 0)))

    print(f'\n{jd}:')
    print(f'  旧模型Top5: 均价${np.mean(old_prices):.1f} | {", ".join(sorted(old_t5))}')
    print(f'  G2模型Top5: 均价${np.mean(g2_prices):.1f} | {", ".join(sorted(g2_t5))}')
    print(f'  重叠: {len(common)}只 | 旧独有: {len(old_only)}只 | G2独有: {len(g2_only)}只')
    
    # G2独有的票中, 低于$5的数量
    g2_low_count = sum(1 for p in g2_prices if p < 5)
    print(f'  G2独有票 <$5: {g2_low_count}只')

print(f'\n✅ 对比完成 | 耗时: {time.time()-t0:.1f}s')
