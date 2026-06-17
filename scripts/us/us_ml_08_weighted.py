#!/usr/bin/env python3
"""
us_ml_08_weighted.py — 权重调整训练，提升召回
在v5特征上用不同的scale_pos_weight测试精度/召回trade-off
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models/us'
CKPT = f'{MODEL_DIR}/greenshaft_v5wt_ckpt.json'

print("us_ml_08: 权重调优召回")

df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
y = (df['fwd_5d_ret'] > 0.05).astype(int).values
pos_ratio = y.mean()
print(f"正样本率: {pos_ratio*100:.2f}%")

exclude = {'ticker', 'date', 'label', 'fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
X = df[feat_cols].values.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
del df

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
del X

base_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist', 'device': 'cuda',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42,
}

# 多种权重测试
weights = [1, 2, 3, 5, 7, 10, 15, 20]
results = {}
start = 0
if os.path.exists(CKPT):
    cp = json.load(open(CKPT))
    results = cp.get('results', {})
    start = cp.get('completed_to', 0)
    print(f"断点: {start}/{len(weights)}")

for i in range(start, len(weights)):
    w = weights[i]
    t0 = time.time()
    
    spw = (len(y_train) - y_train.sum()) / y_train.sum() / w
    
    params = {**base_params, 'scale_pos_weight': spw}
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    dtrain.set_weight(np.full(len(y_train), spw))
    
    model = xgb.train(params, dtrain, num_boost_round=400,
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=15, verbose_eval=False)
    
    dtest = xgb.DMatrix(X_test, y_test)
    y_pred = model.predict(dtest)
    
    # 校准
    val_raw = model.predict(dval)
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_raw, y_val)
    y_calib = ir.transform(y_pred)
    
    # 找最佳阈值（最大化F1）
    best_f1, best_t, best_p, best_r = 0, 0.5, 0, 0
    for th in np.linspace(0.1, 0.9, 17):
        yb = (y_calib > th).astype(int)
        p = precision_score(y_test, yb, zero_division=0)
        r = recall_score(y_test, yb, zero_division=0)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f > best_f1:
            best_f1, best_t, best_p, best_r = f, th, p, r
    
    auc = roc_auc_score(y_test, y_calib)
    
    key = f"w{w}_spw{spw:.1f}"
    results[key] = {
        'weight': w, 'spw': round(spw, 1),
        'auc': round(float(auc), 4),
        'best_thr': round(best_t, 2),
        'precision': round(float(best_p), 4),
        'recall': round(float(best_r), 4),
        'f1': round(float(best_f1), 4),
    }
    
    json.dump({'results': results, 'completed_to': i+1}, open(CKPT, 'w'))
    del model, dtrain, dval, dtest, ir
    
    print(f"  w={w:>2} spw={spw:>6.1f} | AUC={auc:.4f} | 精度={best_p:.4f} 召回={best_r:.4f} F1={best_f1:.4f} (阈值={best_t:.2f}) | {time.time()-t0:.0f}s", flush=True)

# 结果排序
print(f"\n{'='*60}")
print(f"权重扫描结果 (按F1排序):")
print(f"{'权重':>6} {'spw':>6} {'AUC':>6} {'阈值':>6} {'精度':>8} {'召回':>8} {'F1':>8}")
print("-"*60)
for key, res in sorted(results.items(), key=lambda x: -x[1]['f1']):
    print(f"{res['weight']:>6} {res['spw']:>6.1f} {res['auc']:.4f} {res['best_thr']:>6.2f} {res['precision']:>8.4f} {res['recall']:>8.4f} {res['f1']:>8.4f}")

# 最佳
best_key = max(results, key=lambda k: results[k]['f1'])
best = results[best_key]
print(f"\n最佳: {best_key}")
print(f"  AUC={best['auc']:.4f} 精度={best['precision']:.4f} 召回={best['recall']:.4f} F1={best['f1']:.4f} (阈值{best['best_thr']:.2f})")
