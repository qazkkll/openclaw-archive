#!/usr/bin/env python3
"""cn-alpha-v1.1 vs v1.0 深度对比分析：为什么V1.1没提升？"""
import pandas as pd, numpy as np, xgboost as xgb, json, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=" * 70)
print("V1.0 vs V1.1 深度对比分析")
print("=" * 70)

# ============================================================
# 1. 特征对比
# ============================================================
print("\n[1] 特征对比:")

# v1.0特征
v1_0_feats = [
    'rev_5d','rev_10d','rev_20d','rsi_reversal','macd_reversal','macd_hist',
    'low_vol_5d','low_vol_20d','low_atr',
    'md_net_5','md_net_20','lg_net_5','lg_net_20','total_net_5','total_net_20',
    'small_cap','residual_mom_5d','residual_mom_20d',
    'vol_r','sm_net_5','sm_net_20','elg_net_5','elg_net_20'
]

# v1.1模型特征
m11 = xgb.Booster()
m11.load_model('models/cn/cn_alpha_v1.1.json')
v1_1_feats = m11.feature_names

v1_0_set = set(v1_0_feats)
v1_1_set = set(v1_1_feats)
added = v1_1_set - v1_0_set
removed = v1_0_set - v1_1_set

print(f"  v1.0: {len(v1_0_feats)}特征")
print(f"  v1.1: {len(v1_1_feats)}特征")
print(f"  新增: {sorted(added)}")
print(f"  移除: {sorted(removed)}")

# ============================================================
# 2. 特征重要性分析
# ============================================================
print("\n[2] V1.1 特征重要性:")

importance = m11.get_score(importance_type='gain')
imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

total_gain = sum(importance.values())
print(f"\n  Top15特征（占总gain的百分比）:")
cumulative = 0
for i, (feat, gain) in enumerate(imp_sorted[:15]):
    pct = gain / total_gain * 100
    cumulative += pct
    is_new = "🆕" if feat in added else "  "
    print(f"  {i+1:>2}. {is_new}{feat:<25} gain={gain:>10.1f}  ({pct:>5.1f}%)  累计{cumulative:>5.1f}%")

# 新增特征的贡献
print(f"\n  新增基本面特征贡献:")
new_gain = sum(importance.get(f, 0) for f in added)
print(f"  新增特征总gain: {new_gain:.1f} / {total_gain:.1f} = {new_gain/total_gain*100:.1f}%")
for f in sorted(added):
    g = importance.get(f, 0)
    pct = g / total_gain * 100 if total_gain > 0 else 0
    print(f"    {f:<25} gain={g:>10.1f}  ({pct:>5.1f}%)")

# ============================================================
# 3. 分时段诊断：哪些时段V1.1比V1.0差？
# ============================================================
print("\n[3] 分时段诊断...")

# 加载数据
hist = pd.read_parquet('data/cn/features_v2.parquet')
hist['date'] = pd.to_datetime(hist['date'])
hist['date_int'] = hist['date'].dt.strftime('%Y%m%d').astype(int)
all_dates = sorted(hist['date_int'].unique())

# 加载v1.0模型
m10 = xgb.Booster()
m10.load_model('models/cn/cn_alpha_v1.0.json')
v1_0_model_feats = m10.feature_names

# 选时点
quarter_starts = []
for year in range(2020, 2027):
    for month in [1, 4, 7, 10]:
        qdate = int(f"{year}{month:02d}01")
        candidates = [d for d in all_dates if d >= 20200101 and abs(d - qdate) < 2000]
        if candidates:
            quarter_starts.append(min(candidates, key=lambda x: abs(x - qdate)))
quarter_starts = sorted(set(quarter_starts))

HOLD_DAYS = 10
TOP_K = 15

print(f"\n  {'日期':>10} {'V1.0收益':>10} {'V1.1收益':>10} {'差异':>8} {'V1.0 Alpha':>11} {'V1.1 Alpha':>11} {'Winner':>8}")
print(f"  {'-'*70}")

v10_better = 0
v11_better = 0
equal = 0

for signal_date in quarter_starts:
    signal_day = hist[hist['date_int'] == signal_date].copy()
    if len(signal_day) < 100:
        continue
    signal_day = signal_day[signal_day['close'] > 3]
    
    signal_idx = all_dates.index(signal_date)
    if signal_idx + HOLD_DAYS >= len(all_dates):
        continue
    exit_date = all_dates[signal_idx + HOLD_DAYS]
    exit_day = hist[hist['date_int'] == exit_date].set_index('sym')
    
    # V1.0预测
    for f in v1_0_model_feats:
        if f not in signal_day.columns:
            signal_day[f] = 0
    X10 = signal_day[v1_0_model_feats].fillna(0)
    signal_day['score_v10'] = m10.predict(xgb.DMatrix(X10))
    
    # V1.1预测
    for f in v1_1_feats:
        if f not in signal_day.columns:
            signal_day[f] = 0
    X11 = signal_day[v1_1_feats].fillna(0)
    signal_day['score_v11'] = m11.predict(xgb.DMatrix(X11))
    
    # 各选Top15
    top10 = signal_day.nlargest(TOP_K, 'score_v10')
    top11 = signal_day.nlargest(TOP_K, 'score_v11')
    
    # 计算收益
    def calc_return(top_df):
        rets = []
        for _, row in top_df.iterrows():
            sym = row['sym']
            if sym in exit_day.index:
                ep = exit_day.loc[sym, 'close']
                if isinstance(ep, pd.Series):
                    ep = ep.iloc[0]
                rets.append((ep - row['close']) / row['close'])
            else:
                rets.append(0)
        return np.mean(rets)
    
    ret10 = calc_return(top10)
    ret11 = calc_return(top11)
    
    # 基准
    bench_entry = signal_day[['sym', 'close']].set_index('sym')
    bench_exit = exit_day[['close']]
    bm = bench_entry.join(bench_exit, lsuffix='_e', rsuffix='_x').dropna()
    bench_ret = ((bm['close_x'] - bm['close_e']) / bm['close_e']).mean()
    
    alpha10 = ret10 - bench_ret
    alpha11 = ret11 - bench_ret
    diff = ret11 - ret10
    
    if ret11 > ret10 + 0.001:
        winner = "V1.1"
        v11_better += 1
    elif ret10 > ret11 + 0.001:
        winner = "V1.0"
        v10_better += 1
    else:
        winner = "TIE"
        equal += 1
    
    print(f"  {signal_date:>10} {ret10*100:>+9.2f}% {ret11*100:>+9.2f}% {diff*100:>+7.2f}% {alpha10*100:>+10.2f}% {alpha11*100:>+10.2f}% {winner:>8}")

print(f"\n  V1.0胜: {v10_better}次, V1.1胜: {v11_better}次, 平局: {equal}次")

# ============================================================
# 4. 失败时段深度分析
# ============================================================
print("\n[4] 失败时段分析（V1.1跑输V1.0的时段）:")
print("  需要检查：V1.1选了哪些V1.0没选的股票？这些股票表现如何？")

# ============================================================
# 5. 优化方向
# ============================================================
print("\n" + "=" * 70)
print("[5] 优化方向分析")
print("=" * 70)

# 特征冗余分析
print("\n[5a] 特征冗余检查:")
# 计算新增特征与原有特征的相关性
new_feats_list = sorted(added)
if new_feats_list:
    sample = hist[hist['date_int'] >= 20230101].sample(min(50000, len(hist[hist['date_int'] >= 20230101])))
    for nf in new_feats_list[:6]:  # 只检查前6个
        if nf not in sample.columns:
            continue
        max_corr = 0
        max_corr_feat = ""
        for of in v1_0_feats:
            if of not in sample.columns:
                continue
            corr = sample[[nf, of]].dropna().corr().iloc[0, 1]
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_feat = of
        print(f"  {nf:<25} 最高相关: {max_corr_feat:<20} r={max_corr:.3f}")

# 模型复杂度
print(f"\n[5b] 模型复杂度:")
print(f"  V1.0: {len(v1_0_feats)}特征")
print(f"  V1.1: {len(v1_1_feats)}特征 (+{len(added)}个)")
print(f"  新增特征占总gain: {new_gain/total_gain*100:.1f}%")

if new_gain/total_gain < 0.05:
    print(f"  → 新增特征贡献<5%，基本是噪声")
elif new_gain/total_gain < 0.15:
    print(f"  → 新增特征贡献有限，可能引入过拟合")
else:
    print(f"  → 新增特征有实质贡献")

# Top特征集中度
top3_gain = sum(gain for _, gain in imp_sorted[:3])
top5_gain = sum(gain for _, gain in imp_sorted[:5])
print(f"\n[5c] 特征集中度:")
print(f"  Top3特征占总gain: {top3_gain/total_gain*100:.1f}%")
print(f"  Top5特征占总gain: {top5_gain/total_gain*100:.1f}%")
if top3_gain/total_gain > 0.5:
    print(f"  → 模型过于依赖少数特征，脆弱性高")

print("\n[完成]")
