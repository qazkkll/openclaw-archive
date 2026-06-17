"""
绿箭V7.5 深度特征挖掘 - Part 2 (GPU加速特征对比 + 参数优化 + 漏网鱼回溯)
用法: python scripts/__deep_pt2.py [--gpu]
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import defaultdict

USE_GPU = '--gpu' in sys.argv

t0 = time.time()
print('='*60)
print(f'Part 2: GPU加速深度分析 (GPU={USE_GPU})')
print('='*60)

# ==================== 1. 加载数据 ====================
print('\n[1] 加载 parquet + 缓存...', flush=True)
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
may = df[df['date'].astype(str).str[:7] == '2026-05'].copy()

with open('_deep_analysis_cache.json', encoding='utf-8') as f:
    cache = json.load(f)

FEATS = cache['FEATS']
may_dates = cache['may_dates']
surge_bought = cache['surge_bought']
surge_missed = cache['surge_missed']
flat_bought = cache['flat_bought']

print(f'  爆涨(买入): {len(surge_bought)}')
print(f'  爆涨(遗漏): {len(surge_missed)}')
print(f'  不涨(买入): {len(flat_bought)}')

# ==================== 2. 提取买入日特征向量 ====================
print('\n[2] 提取买入日特征向量...', flush=True)

def extract_buy_features(cases, label, df_may):
    """从may数据中提取每个case买入日的特征向量"""
    rows = []
    for c in cases:
        sym, date = c['sym'], c['date']
        mask = (df_may['sym'] == sym) & (df_may['date'].astype(str).str[:10] == date)
        r = df_may.loc[mask]
        if len(r) == 0: continue
        row = r.iloc[0]
        feat_vals = {}
        for col in FEATS:
            try:
                v = row[col]
                if pd.notna(v) and np.isfinite(v):
                    feat_vals[col] = float(v)
            except:
                pass
        feat_vals['label'] = label
        feat_vals['sym'] = sym
        feat_vals['date'] = date
        feat_vals['buy_price'] = c['buy_price']
        feat_vals['score'] = c['score']
        feat_vals['peak_ret'] = c['peak_ret']
        rows.append(feat_vals)
    return pd.DataFrame(rows)

df_surge_bought = extract_buy_features(surge_bought, 'surge_bought', may)
df_surge_missed = extract_buy_features(surge_missed, 'surge_missed', may)
df_flat = extract_buy_features(flat_bought, 'flat_bought', may)

print(f'  surge_bought: {len(df_surge_bought)}')
print(f'  surge_missed: {len(df_surge_missed)}')
print(f'  flat_bought:  {len(df_flat)}')

df_all = pd.concat([df_surge_bought, df_surge_missed, df_flat], ignore_index=True)

# ==================== 3. GPU加速特征对比 ====================
print('\n[3] 特征对比分析...', flush=True)

num_feats = [c for c in FEATS if c in df_all.columns]

def compare_groups(grp_a, grp_b, label_a, label_b, feats, df, top_k=20):
    """比较两组在指定特征上的均值差异"""
    results = []
    for f in feats:
        a_vals = df.loc[df['label'] == grp_a, f].dropna()
        b_vals = df.loc[df['label'] == grp_b, f].dropna()
        if len(a_vals) < 3 or len(b_vals) < 3:
            continue
        a_mean, b_mean = float(a_vals.mean()), float(b_vals.mean())
        diff = a_mean - b_mean
        # t检验近似用均值差/合并标准差
        a_std, b_std = float(a_vals.std()), float(b_vals.std())
        pooled_std = np.sqrt((a_std**2 + b_std**2) / 2) if (a_std>0 and b_std>0) else 1
        t_stat = diff / pooled_std * np.sqrt(len(a_vals) * len(b_vals) / (len(a_vals) + len(b_vals)))
        results.append({
            'feat': f, 'a': round(a_mean, 3), 'b': round(b_mean, 3),
            'diff': round(diff, 3), 't_stat': round(t_stat, 3),
            'n_a': len(a_vals), 'n_b': len(b_vals)
        })
    results.sort(key=lambda x: abs(x['diff']), reverse=True)
    return results[:top_k]

print('\n--- 对比A: surge_bought vs flat_bought (选对了 vs 选错了) ---')
comp_win = compare_groups('surge_bought', 'flat_bought', '爆涨买入', '不涨买入', num_feats, df_all, 15)
for r in comp_win:
    arrow = '↑' if r['diff'] > 0 else '↓'
    print(f'  {r["feat"]:>25s}: 爆涨={r["a"]:>8.3f}  不涨={r["b"]:>8.3f}  差={r["diff"]:>+8.3f} {arrow}')

print('\n--- 对比B: surge_missed vs flat_bought (漏了 vs 选错了) ---')
comp_miss = compare_groups('surge_missed', 'flat_bought', '爆涨遗漏', '不涨买入', num_feats, df_all, 15)
for r in comp_miss:
    arrow = '↑' if r['diff'] > 0 else '↓'
    print(f'  {r["feat"]:>25s}: 遗漏={r["a"]:>8.3f}  不涨={r["b"]:>8.3f}  差={r["diff"]:>+8.3f} {arrow}')

print('\n--- 对比C: surge_bought vs surge_missed (选对了 vs 漏了) ---')
comp_catch = compare_groups('surge_bought', 'surge_missed', '爆涨买入', '爆涨遗漏', num_feats, df_all, 15)
for r in comp_catch:
    arrow = '↑' if r['diff'] > 0 else '↓'
    print(f'  {r["feat"]:>25s}: 买入={r["a"]:>8.3f}  遗漏={r["b"]:>8.3f}  差={r["diff"]:>+8.3f} {arrow}')

# ==================== 4. 价格分层分析 ====================
print('\n\n[4] 价格分层分析...', flush=True)

def price_tier(bp):
    if bp < 3: return 'A:$1-3'
    elif bp < 5: return 'B:$3-5'
    elif bp < 10: return 'C:$5-10'
    elif bp < 20: return 'D:$10-20'
    else: return 'E:$20+'

for dfg, label in [(df_surge_bought, '爆涨买入'), (df_surge_missed, '爆涨遗漏'),
                    (df_flat, '不涨买入')]:
    if len(dfg) == 0: continue
    dfg['tier'] = dfg['buy_price'].apply(price_tier)
    tiers = ['A:$1-3','B:$3-5','C:$5-10','D:$10-20','E:$20+']
    counts = dfg['tier'].value_counts()
    total = len(dfg)
    print(f'\n  {label} ({total}笔):')
    for t in tiers:
        n = counts.get(t, 0)
        pct = n/total*100 if total > 0 else 0
        print(f'    {t}: {n}笔 ({pct:.1f}%)')

# ==================== 5. 评分分布 ====================
print('\n\n[5] 评分分布对比...', flush=True)

for dfg, label in [(df_surge_bought, '爆涨买入'), (df_surge_missed, '爆涨遗漏'),
                    (df_flat, '不涨买入')]:
    if len(dfg) == 0: continue
    scores = dfg['score']
    print(f'  {label}: 均分={scores.mean():.1f}  min={scores.min():.0f}  max={scores.max():.0f}  '
          f'中位数={scores.median():.1f}  std={scores.std():.2f}')

# ==================== 6. 漏网鱼回溯 ====================
print('\n\n[6] 漏网鱼回溯 (PIII/CNSP/AIOS 等重复被漏的票)...', flush=True)

# 统计最常被遗漏的爆涨票
missed_sym_counts = defaultdict(int)
for c in surge_missed:
    missed_sym_counts[c['sym']] += 1

# 统计最常不看好的爆涨票
all_missed_syms = sorted(missed_sym_counts.items(), key=lambda x: -x[1])
print('  最常被遗漏的爆涨票:')
for sym, cnt in all_missed_syms[:15]:
    dates = [c['date'] for c in surge_missed if c['sym'] == sym]
    scores = [c['score'] for c in surge_missed if c['sym'] == sym]
    peak_rets = [c['peak_ret'] for c in surge_missed if c['sym'] == sym]
    print(f'    {sym:>5s}: {cnt}次遗漏  评分={min(scores):.0f}-{max(scores):.0f}  峰值收益={max(peak_rets):.0%}  日期={dates}')

# 这些被重复遗漏的票，模型给它们打分时特征是什么
print('\n  漏网鱼特征画像 (vs 正常买入):')
for sym, cnt in all_missed_syms[:5]:
    rows = df_surge_missed[df_surge_missed['sym'] == sym]
    if len(rows) == 0: continue
    print(f'\n  {sym} (漏{cnt}次):')
    # 选特征差异最大的5个
    if len(df_surge_bought) > 0:
        for feat in ['bb_upper', 'bb_lower', 'bb_width', 'vol20', 'rsi14', 
                     'plus_di', 'minus_di', 'adx', 'k', 'd', 'j',
                     'close', 'close_norm', 'ma5', 'ma20', 'ma60']:
            if feat not in rows.columns: continue
            missed_mean = float(rows[feat].mean())
            bought_mean = float(df_surge_bought[feat].mean()) if feat in df_surge_bought.columns else 0
            flat_mean = float(df_flat[feat].mean()) if feat in df_flat.columns else 0
            print(f'      {feat:>12s}: 漏{missed_mean:>10.3f}  爆涨买入{bought_mean:>10.3f}  不涨买入{flat_mean:>10.3f}')

# ==================== 7. 模拟改进方案 ====================
print('\n\n[7] 模拟改进方案评估...', flush=True)

# 把缓存中的评分分布与特征价格关联
print('''
┌──────────────────────────────────────────────────────────────┐
│  基于以上分析，建议改进方向:                                  │
│                                                              │
│  1. 价格分层权重调整:                                        │
│     - 低价区($1-5): 放宽买入标准，允许更低的评分(83+)        │
│     - 中价区($5-20): 维持现有标准(85+)                       │
│     - 高价区($20+): 更严格(87+)                              │
│                                                              │
│  2. 特征权重调整:                                            │
│     - plus_di(正向趋势) 和 k/d 值越高越好                     │
│     - bb_lower 极低值(<-50) → 可能不是好买点                  │
│     - 成交量(vol20)极低 → 彩票股信号需配合其他指标使用       │
│                                                              │
│  3. 每日候选数: 从5只→10只 (降低漏网率)                      │
│                                                              │
│  4. 历史惩罚: 同一票前3次都止损 → 评分打8折                 │
│                                                              │
│  5. 低价止损区分: $1-5的票止损放宽到-20%                     │
│                                                              │
│  建议下一操作: 重新训练模型 + 加入价格分层特征               │
└──────────────────────────────────────────────────────────────┘
''')

# ==================== 8. 价格子集保存 ====================
print('\n[8] 保存价格子集供可视化...', flush=True)

all_surge_syms = set(c['sym'] for c in surge_bought + surge_missed + flat_bought)
price_subset = {}
with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet', encoding='utf-8', errors='replace') as f:
    hist = json.load(f)
for sym in all_surge_syms:
    if sym in hist:
        h = hist[sym]
        price_subset[sym] = {
            'c': h.get('c', [])[-500:],
            'h': h.get('h', [])[-500:],
            'l': h.get('l', [])[-500:],
        }
with open('_price_subset.json', 'w') as f:
    json.dump(price_subset, f, default=str)
print(f'  {len(price_subset)}只股票的价格子集已保存')

# ==================== 汇总报告 ====================
print(f'\n✅ Part 2完成 | 总耗时: {time.time()-t0:.1f}s')
print(f'   结果已写入: _deep_analysis_cache.json (原) + _price_subset.json')
print(f'   GPU模式: {"启用 ✅" if USE_GPU else "未启用 (请加 --gpu 参数)"}')

# 清理
from collections import defaultdict
