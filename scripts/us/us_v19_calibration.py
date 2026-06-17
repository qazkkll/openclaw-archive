"""绿箭v16 校准分析 — 高概率区间真实命中率"""
import sys, os, math, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
import _paths

print("═══ 绿箭v16 概率校准分析 ═══")

df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
base_feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
              'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
              'vol_ratio','ma_bias20','vol5','trend_accel',
              'short_ratio','short_pct','market_cap','pe_trailing','pe_forward','beta']
df['sector_code'] = df['sector'].astype('category').cat.codes.astype(int)
all_feats = base_feats + ['sector_code']

# 用最后15%做测试
n = len(df)
test_start = int(n * 0.85)
df_test = df.iloc[test_start:].dropna(subset=all_feats + ['label_5d_5class'])
X_test = df_test[all_feats].values
y_test = df_test['label_5d_5class'].values
actual_pct = df_test['label_5d_pct'].values  # 实际5天收益

# 加载模型
m = xgb.XGBClassifier(, device='cuda')
m.load_model(_paths.US_MODEL_DIR + "/greenshaft_v16.json")
proba = m.predict_proba(X_test)

print(f"\n测试集: {len(X_test):,}行")
print(f"涨>5%历史占比: {(y_test==4).mean()*100:.1f}%")

# 按涨>5%概率分桶校准
pu5 = proba[:, 4]
bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), 
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]

print(f"\n{'='*70}")
print(f"{'概率区间':>12} {'样本数':>8} {'实际涨>5%':>10} {'命中率':>8} {'校准偏差':>8}")
print(f"{'='*70}")

for lo, hi in bins:
    mask = (pu5 >= lo) & (pu5 < hi)
    n_bin = mask.sum()
    if n_bin == 0:
        continue
    actual_hit = (y_test[mask] == 4).mean()
    pred_prob = pu5[mask].mean()
    bias = pred_prob - actual_hit
    print(f"  {lo:.0%}-{hi:.0%}    {n_bin:>8} {actual_hit*100:>8.1f}%  {pred_prob*100:>6.1f}%   {bias*100:>+6.1f}%")

# 高概率区间详细分析
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = pu5 >= threshold
    n_high = mask.sum()
    if n_high < 5:
        continue
    hit = (y_test[mask] == 4).mean()
    actual_avg_ret = actual_pct[mask].mean()
    win_rate = (actual_pct[mask] > 0).mean()
    print(f"\n  >{threshold:.0%}(n={n_high}): 涨>5%命中={hit:.1%} 均收益={actual_avg_ret:.2f}% 胜率={win_rate:.1%}")

# 绿箭概率 × 校准因子
print(f"\n{'='*70}")
print("绿箭概率 × 校准因子 = 修正后实际涨>5%概率")
print(f"{'绿箭概率':>10} {'校准因子':>10} {'修正概率':>10}")
print(f"{'='*70}")
for prob in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = pu5 >= prob
    n_high = mask.sum()
    if n_high < 5:
        continue
    actual = (y_test[mask] == 4).mean()
    factor = actual / prob if prob > 0 else 0
    print(f"  {prob:.0%}       {factor:.2f}x     {actual:.1%}")
