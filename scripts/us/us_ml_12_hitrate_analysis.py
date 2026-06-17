#!/usr/bin/env python3
"""各种口径下，模型预测涨5%的命中率
口径A: 实际涨5%+ (严格)
口径B: 实际涨3%+ (宽松)
口径C: 实际涨0%+ (任何正收益)
口径D: 实际跌不到3% (不亏钱就算赢)
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
print("载入数据...")
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])

exclude = {'ticker','date','label','fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y_true = df['fwd_5d_ret'].values

n = len(X)
te, ve = int(n*0.7), int(n*0.85)

# 训练
spw = (te - (y_true[:te]>0.05).sum()) / (y_true[:te]>0.05).sum()
params = {'objective':'binary:logistic','eval_metric':'auc','tree_method':'hist',
          'device':'cuda','max_depth':6,'learning_rate':0.05,'subsample':0.8,
          'colsample_bytree':0.8,'scale_pos_weight':spw,'random_state':42}

dtrain = xgb.DMatrix(X[:te], (y_true[:te]>0.05).astype(int))
dval = xgb.DMatrix(X[te:ve], (y_true[te:ve]>0.05).astype(int))
model = xgb.train(params, dtrain, num_boost_round=400,
                  evals=[(dtrain,'train'),(dval,'val')],
                  early_stopping_rounds=15, verbose_eval=False)

dtest = xgb.DMatrix(X[ve:])
y_pred = model.predict(dtest)
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(model.predict(dval), (y_true[te:ve]>0.05).astype(int))
probs = ir.transform(y_pred)

y_true_test = y_true[ve:]

print(f"\n{'='*70}")
print(f"模型预测涨5%在不同口径下的实际命中率")
print(f"{'='*70}")
print(f"{'概率阈':>6} {'推票数':>8} {'涨5%+':>10} {'涨3%+':>10} {'任何涨':>10} {'不亏>3%':>10}")
print("-"*70)

for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55]:
    mask = probs > thr
    n_rec = int(mask.sum())
    actual = y_true_test[mask]
    
    p5 = (actual > 0.05).mean() * 100 if n_rec > 0 else 0
    p3 = (actual > 0.03).mean() * 100 if n_rec > 0 else 0
    pp = (actual > 0).mean() * 100 if n_rec > 0 else 0
    psafe = (actual > -0.03).mean() * 100 if n_rec > 0 else 0
    
    print(f"{thr:.2f} {n_rec:>8} {p5:>9.1f}% {p3:>9.1f}% {pp:>9.1f}% {psafe:>9.1f}%")

print(f"\n{'='*70}")
print(f"同口径，按不同阈值对比:")
print(f"{'阈值':>6} {'推票':>8} {'涨5%':>8} {'涨3%':>8} {'涨0%':>8} {'不亏':>8} {'胜率(3%+)':>10} {'单票期望':>10}")
print("-"*70)

for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    mask = probs > thr
    n_rec = int(mask.sum())
    actual = y_true_test[mask]
    
    p5 = (actual > 0.05).mean() if n_rec > 0 else 0
    p3 = (actual > 0.03).mean() if n_rec > 0 else 0
    pp = (actual > 0).mean() if n_rec > 0 else 0
    psafe = (actual > -0.03).mean() if n_rec > 0 else 0
    
    # 期望值计算: 假设5%涨幅平均赚5%，跌了亏2%，不亏>3%=赚3%
    # 简化: 涨5%赚5%, 涨3-5%赚3%, 涨0-3%赚1.5%, 跌0-3%亏1.5%, 跌>3%亏5%
    if n_rec > 0:
        e5 = ((actual > 0.05).mean() * 0.05)
        e3_5 = (((actual > 0.03) & (actual <= 0.05)).mean() * 0.03)
        e0_3 = (((actual > 0) & (actual <= 0.03)).mean() * 0.015)
        e_3_0 = (((actual > -0.03) & (actual <= 0)).mean() * -0.015)
        e_l3 = ((actual <= -0.03).mean() * -0.05)
        expected = e5 + e3_5 + e0_3 + e_3_0 + e_l3
    else:
        expected = 0
    
    print(f"{thr:.2f} {n_rec:>8} {p5*100:>6.1f}% {p3*100:>6.1f}% {pp*100:>6.1f}% {psafe*100:>6.1f}% {p3*100:>8.1f}% {expected*100:>+8.2f}%")

print(f"\n解释:")
print(f"  '涨5%+': 实际收益>5% (模型目标)")
print(f"  '涨3%+': 实际收益>3% (收益能覆盖成本)")
print(f"  '任何涨': 实际收益>0% (不亏)")
print(f"  '不亏>3%': 实际收益>-3% (小亏可接受)")
print(f"  '胜率(3%+)': 涨3%+算赢")
print(f"  '单票期望': 严格算的期望收益(涨5%赚5%，跌3%亏5%)")
