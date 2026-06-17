#!/usr/bin/env python3
"""
us_ml_11_threshold_sweep.py — 决策阈值精细扫描
固定模型(v5 5%二分类)，只调预测阈值看精度/召回/F1曲线
不重新训练，秒出
"""
import sys, os, json, time, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
print("us_ml_11: 决策阈值精细扫描")

df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
y = (df['fwd_5d_ret'] > 0.05).astype(int).values
print(f"正样本率: {y.mean()*100:.2f}%")

exclude = {'ticker','date','label','fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
del df

n = len(X)
te, ve = int(n*0.7), int(n*0.85)
X_train, X_val, X_test = X[:te], X[te:ve], X[ve:]
y_train, y_val, y_test = y[:te], y[te:ve], y[ve:]
del X

# 训练一次模型
print("训练模型...")
spw = (len(y_train)-y_train.sum())/y_train.sum()  # ~5.1
params = {'objective':'binary:logistic','eval_metric':['logloss','auc'],
          'tree_method':'hist','device':'cuda','max_depth':6,'learning_rate':0.05,
          'subsample':0.8,'colsample_bytree':0.8,'scale_pos_weight':spw,'random_state':42}

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
dtest = xgb.DMatrix(X_test, y_test)

model = xgb.train(params, dtrain, num_boost_round=400,
                  evals=[(dtrain,'train'),(dval,'val')],
                  early_stopping_rounds=15, verbose_eval=False)

y_pred = model.predict(dtest)
val_pred = model.predict(dval)

# 校准
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(val_pred, y_val)
y_calib = ir.transform(y_pred)

auc = roc_auc_score(y_test, y_calib)
print(f"AUC: {auc:.4f}")

# 阈值精细扫描
print(f"\n{'='*65}")
print(f"{'阈值':>6} {'精度':>10} {'召回':>8} {'F1':>8} {'推票数':>8} {'正确数':>8}")
print("-"*65)

best = {'f1':0}
for thr in np.arange(0.05, 0.96, 0.025):
    yb = (y_calib > thr).astype(int)
    p = precision_score(y_test, yb, zero_division=0)
    r = recall_score(y_test, yb, zero_division=0)
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    n_rec = int(yb.sum())
    n_correct = int((yb & y_test).sum())
    print(f"{thr:.3f} {p:>10.4f} {r:>8.4f} {f:>8.4f} {n_rec:>8} {n_correct:>8}")
    if f > best['f1']:
        best = {'f1':round(f,4), 'thr':round(thr,3), 'p':round(p,4), 'r':round(r,4),
                'n_rec':n_rec, 'n_correct':n_correct}

print(f"\n{'='*65}")
print(f"最佳F1: 阈值={best['thr']:.3f} 精度={best['p']:.4f} 召回={best['r']:.4f} F1={best['f1']:.4f}")
print(f"  推{n_rec}只, 正确{n_correct}只 ({best['n_correct']/best['n_rec']*100:.1f}%)")
print(f"\n高精度模式(>50%):")
for thr in [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]:
    yb = (y_calib > thr).astype(int)
    p = precision_score(y_test, yb, zero_division=0)
    r = recall_score(y_test, yb, zero_division=0)
    f = 2*p*r/(p+r) if (p+r)>0 else 0
    n_rec = int(yb.sum())
    n_correct = int((yb & y_test).sum())
    print(f"  thr={thr:.2f}: 精度={p:.4f} 召回={r:.4f} F1={f:.4f} 推{best['n_rec']}只中{best['n_correct']}只")
