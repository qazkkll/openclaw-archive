#!/usr/bin/env python3
"""
us_ml_09_threshold_compare.py — 不同涨跌幅阈值对比
从1%到7%看AUC/精度/召回的trade-off，找出最佳阈值
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
CKPT = '/home/hermes/.hermes/openclaw-project/data/models/us/us_threshold_ckpt.json'
OUTPUT = '/home/hermes/.hermes/openclaw-project/data/models/us/threshold_compare.json'

print("us_ml_09: 阈值对比")
df = pd.read_parquet(INPUT)

exclude = {'ticker', 'date', 'label', 'fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]

thrs = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07]
results = {}
start = 0
if os.path.exists(CKPT):
    cp = json.load(open(CKPT))
    results = cp.get('results', {})
    start = cp.get('completed_to', 0)
    print(f"断点: {start}/{len(thrs)}")

for i in range(start, len(thrs)):
    thr = thrs[i]
    t0 = time.time()
    
    df_tmp = df.dropna(subset=['fwd_5d_ret'])
    y = (df_tmp['fwd_5d_ret'] > thr).astype(int).values
    X = df_tmp[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    n = len(X)
    train_end, val_end = int(n*0.7), int(n*0.85)
    
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist', 'device': 'cuda',
        'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'random_state': 42,
        'scale_pos_weight': (1-y.mean())/y.mean(),
    }
    
    dtrain = xgb.DMatrix(X[:train_end], y[:train_end])
    dval = xgb.DMatrix(X[train_end:val_end], y[train_end:val_end])
    dtest = xgb.DMatrix(X[val_end:], y[val_end:])
    
    model = xgb.train(xgb_params, dtrain, num_boost_round=400,
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=15, verbose_eval=False)
    
    y_pred = model.predict(dtest)
    
    # 校准
    val_pred = model.predict(dval)
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(val_pred, y[train_end:val_end])
    y_calib = ir.transform(y_pred)
    
    y_true = y[val_end:]
    auc = roc_auc_score(y_true, y_calib)
    
    # 寻找最佳F1阈值
    best = {'f1': 0, 'thr': 0.5, 'p': 0, 'r': 0}
    for th in np.linspace(0.05, 0.95, 19):
        yb = (y_calib > th).astype(int)
        p = precision_score(y_true, yb, zero_division=0)
        r = recall_score(y_true, yb, zero_division=0)
        f = 2*p*r/(p+r) if (p+r)>0 else 0
        if f > best['f1']:
            best = {'f1': round(float(f),4), 'thr': round(th,2),
                    'p': round(float(p),4), 'r': round(float(r),4)}
    
    # 校准偏差
    calib_bias = (y_calib.mean() - y.mean()) * 100
    
    ratio = y.mean() * 100
    print(f"  >{thr*100:>3.0f}% pos={ratio:>5.2f}% | AUC={auc:.4f} | F1={best['f1']:.4f} 精度={best['p']:.4f} 召回={best['r']:.4f} (thr={best['thr']:.2f}) 校准偏差={calib_bias:+.2f}% | {time.time()-t0:.0f}s")
    
    results[f'thr_{int(thr*100)}'] = {
        'threshold': thr, 'pos_pct': round(ratio, 2),
        'auc': round(float(auc), 4), 'f1': best['f1'],
        'precision': best['p'], 'recall': best['r'],
        'best_thr': best['thr'],
        'calib_bias_pct': round(calib_bias, 2),
    }
    
    json.dump({'results': results, 'completed_to': i+1}, open(CKPT, 'w'))
    del model, dtrain, dval, dtest, df_tmp, X, y

print(f"\n{'='*70}")
print(f"阈值对比结果 (按F1排序):")
print(f"{'阈值':>5} {'正样本':>8} {'AUC':>6} {'F1':>8} {'精度':>8} {'召回':>8} {'决策阈值':>8} {'校准偏差':>8}")
print("-"*70)
for key, res in sorted(results.items(), key=lambda x: -x[1]['f1']):
    print(f"{res['threshold']*100:>3.0f}% {res['pos_pct']:>7.1f}% {res['auc']:.4f} {res['f1']:>8.4f} {res['precision']:>8.4f} {res['recall']:>8.4f} {res['best_thr']:>8.2f} {res['calib_bias_pct']:>+7.2f}%")

json.dump(results, open(OUTPUT, 'w'), indent=2)
print(f"\n结果保存: {OUTPUT}")
if os.path.exists(CKPT):
    os.remove(CKPT)
