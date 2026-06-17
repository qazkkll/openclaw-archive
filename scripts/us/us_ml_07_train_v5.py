#!/usr/bin/env python3
"""
us_ml_07_train_v5.py — v5特征训练
5%二分类 + 参数扫描 + 全量训练 + 评估
GPU加速
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models/us'
MODEL_OUT = f'{MODEL_DIR}/greenshaft_v5_5pct.json'
PRED_OUT = f'{MODEL_DIR}/greenshaft_v5_prediction.json'
CKPT = f'{MODEL_DIR}/greenshaft_v5_ckpt.json'

print("us_ml_07: v5特征 - 5%二分类训练")

df = pd.read_parquet(INPUT)
df = df.dropna(subset=['fwd_5d_ret'])
print(f"  总行数: {len(df):,}")

y = (df['fwd_5d_ret'] > 0.05).astype(int).values
print(f"  正样本率: {y.mean()*100:.2f}%")

exclude = {'ticker', 'date', 'label', 'fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
print(f"  特征({len(feat_cols)}): {feat_cols}")

X = df[feat_cols].values.astype(np.float32)
# 清理inf/nan
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

del X, df
print(f"  训练: {len(X_train):,}, 验证: {len(X_val):,}, 测试: {len(X_test):,}")

# 参数扫描
param_grid = [
    {'md': 5, 'lr': 0.05, 'n': 400, 'ss': 0.8, 'cs': 0.8},
    {'md': 6, 'lr': 0.05, 'n': 400, 'ss': 0.8, 'cs': 0.8},
    {'md': 4, 'lr': 0.05, 'n': 400, 'ss': 0.8, 'cs': 0.8},
    {'md': 5, 'lr': 0.10, 'n': 300, 'ss': 0.8, 'cs': 0.8},
    {'md': 5, 'lr': 0.03, 'n': 600, 'ss': 0.8, 'cs': 0.8},
    {'md': 6, 'lr': 0.03, 'n': 600, 'ss': 0.7, 'cs': 0.7},
    {'md': 7, 'lr': 0.03, 'n': 500, 'ss': 0.8, 'cs': 0.7},
]

base_params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc'],
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': 42,
}

print("\n参数扫描...")
T0 = time.time()
all_results = {}
start = 0
if os.path.exists(CKPT):
    cp = json.load(open(CKPT))
    all_results = cp.get('results', {})
    start = cp.get('completed_to', 0)
    print(f"  断点: {start}/{len(param_grid)}")

for i in range(start, len(param_grid)):
    p = param_grid[i]
    t0 = time.time()
    
    params = {**base_params, 'max_depth': p['md'], 'learning_rate': p['lr'],
              'subsample': p['ss'], 'colsample_bytree': p['cs']}
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    
    model = xgb.train(params, dtrain, num_boost_round=p['n'],
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=15, verbose_eval=False)
    
    dtest = xgb.DMatrix(X_test, y_test)
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    y_bin = (y_pred > 0.5).astype(int)
    prec = precision_score(y_test, y_bin, zero_division=0)
    rec = recall_score(y_test, y_bin, zero_division=0)
    
    key = f"md{p['md']}_lr{p['lr']}_n{p['n']}_ss{p['ss']}"
    all_results[key] = {'auc': round(float(auc), 4), 'prec': round(float(prec), 4), 
                        'rec': round(float(rec), 4), 'time': round(time.time()-t0, 1)}
    
    json.dump({'results': all_results, 'completed_to': i+1}, open(CKPT, 'w'))
    del model, dtrain, dval, dtest, y_pred, y_bin
    
    sec = time.time() - t0
    print(f"  {key}: AUC={auc:.4f} 精度={prec:.4f} 召回={rec:.4f} {sec:.0f}s", flush=True)

print(f"\n{'='*50}")
print("参数扫描结果:")
for key, res in sorted(all_results.items(), key=lambda x: -x[1]['auc']):
    print(f"  {key}: AUC={res['auc']:.4f} 精度={res['prec']:.4f}")
best_key = max(all_results, key=lambda k: all_results[k]['auc'])
best_params = param_grid[list(all_results.keys()).index(best_key)]
print(f"最佳: {best_key}")

# 全量训练
print(f"\n全量训练 (best={best_key})...")
t0 = time.time()
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)

final_params = {**base_params,
    'max_depth': best_params['md'], 'learning_rate': best_params['lr'],
    'subsample': best_params['ss'], 'colsample_bytree': best_params['cs'],
}

model = xgb.train(final_params, dtrain, num_boost_round=best_params['n'],
                  evals=[(dtrain, 'train'), (dval, 'val')],
                  early_stopping_rounds=15, verbose_eval=False)

dtest = xgb.DMatrix(X_test, y_test)
y_pred_raw = model.predict(dtest)

auc = roc_auc_score(y_test, y_pred_raw)
print(f"  裸AUC: {auc:.4f}")

# 校准
ir = IsotonicRegression(out_of_bounds='clip')
val_raw = model.predict(dval)
ir.fit(val_raw, y_val)
y_calib = ir.transform(y_pred_raw)

calib_bias = (y_calib.mean() - y_test.mean()) * 100

# 校准后精度
y_calib_bin = (y_calib > 0.5).astype(int)
prec = precision_score(y_test, y_calib_bin, zero_division=0)
rec = recall_score(y_test, y_calib_bin, zero_division=0)
f1 = f1_score(y_test, y_calib_bin, zero_division=0)

print(f"  校准后偏差: {calib_bias:.2f}%")
print(f"  精度: {prec:.4f}  召回: {rec:.4f}  F1: {f1:.4f}")

# 分档校准
print(f"\n校准分档:")
bins = np.linspace(0, 1, 11)
for i in range(10):
    lo, hi = bins[i], bins[i+1]
    mask = (y_calib >= lo) & (y_calib < hi)
    if mask.sum() > 50:
        actual = y_test[mask].mean()
        pred = y_calib[mask].mean()
        diff = (pred - actual) * 100
        print(f"  [{lo:.1f}-{hi:.1f}): 预测{pred:.3f} 实际{actual:.3f} 偏差{diff:+.2f}% ({mask.sum():,})")

# 特征重要性
importances = model.get_score(importance_type='gain')
total = sum(importances.values()) or 1
imp = {feat_cols[int(k[1:])]: float(v/total*100) for k, v in importances.items()}
top10 = sorted(imp.items(), key=lambda x: -x[1])[:10]
print(f"\nTop 10特征:")
for name, pct_ in top10:
    print(f"  {name}: {pct_:.1f}%")

# 保存模型
model.save_model(MODEL_OUT)
calib_info = {'method': 'isotonic', 'bias_pct': round(calib_bias, 2)}
json.dump(calib_info, open(f'{MODEL_DIR}/greenshaft_v5_calib.json', 'w'))

# 最新推荐
print(f"\n生成推荐...")
df2 = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
latest = df2.groupby('ticker').last().reset_index()
X_latest = latest[feat_cols].values.astype(np.float32)
X_latest = np.nan_to_num(X_latest, nan=0.0, posinf=0.0, neginf=0.0)
dlatest = xgb.DMatrix(X_latest)
preds = model.predict(dlatest)
probs = ir.transform(preds)

df_out = latest[['ticker', 'date']].copy()
df_out['prob_gain_5pct'] = np.round(probs, 4)
df_out = df_out.sort_values('prob_gain_5pct', ascending=False)

print(f"\n{'='*60}")
print(f"📊 绿箭v5 (5%二分类) 推荐")
print(f"{'='*60}")
print(f"AUC={auc:.4f} 精度={prec:.4f} 校准偏差={calib_bias:.2f}%")
print(f"{'='*60}")
print(f"{'#':>3} {'代码':<8} {'日期':<12} {'涨>5%概率':>10}")
print("-"*40)
for i, (_, row) in enumerate(df_out.head(15).iterrows()):
    print(f"{i+1:>3} {row['ticker']:<8} {str(row['date'])[:10]:<12} {row['prob_gain_5pct']:.4f}")

# 保存推荐
top50 = df_out.head(50).to_dict('records')
pred_data = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'greenshaft_v5_5pct',
    'auc': round(float(auc), 4),
    'precision': round(float(prec), 4),
    'recall': round(float(rec), 4),
    'calib_bias_pct': round(calib_bias, 2),
    'top_50': [{k: (str(v) if isinstance(v, (pd.Timestamp, np.integer)) else float(v) if isinstance(v, np.floating) else v) for k, v in item.items() if k in ['ticker','date','prob_gain_5pct']} for item in top50],
}
json.dump(pred_data, open(PRED_OUT, 'w'), indent=2)

print(f"\n模型: {MODEL_OUT}")
print(f"推荐: {PRED_OUT}")
print(f"总耗时: {(time.time()-T0)/60:.1f}分钟")
