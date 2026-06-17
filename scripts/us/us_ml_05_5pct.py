#!/usr/bin/env python3
"""
us_ml_05_5pct.py — 5%阈值二分类训练
从v4特征数据改标签: fwd_5d_ret > 0.05 = 1, 否则 = 0
GPU加速 + 完整评估
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v4.parquet'
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models/us'
MODEL_OUT = f'{MODEL_DIR}/greenshaft_v5_5pct.json'
PRED_OUT = f'{MODEL_DIR}/greenshaft_v5_prediction.json'

print("us_ml_05: 5%阈值二分类训练")

df = pd.read_parquet(INPUT)
print(f"  总行数: {len(df):,}")

# 改标签: >5%上涨=1
df = df.dropna(subset=['fwd_5d_ret'])
y = (df['fwd_5d_ret'] > 0.05).astype(int).values
print(f"  标签分布: 0={y.sum()}, 1={(y==0).sum()}")
print(f"  正样本率: {y.mean()*100:.2f}%")

exclude = {'ticker', 'date', 'label', 'fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
print(f"  特征数: {len(feat_cols)}")

X = df[feat_cols].values.astype(np.float32)

# 时间序列切分
n = len(X)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
splits = [
    (0, train_end),
    (train_end, val_end),
    (val_end, n)
]

X_train, X_val, X_test = X[splits[0][0]:splits[0][1]], X[splits[1][0]:splits[1][1]], X[splits[2][0]:splits[2][1]]
y_train, y_val, y_test = y[splits[0][0]:splits[0][1]], y[splits[1][0]:splits[1][1]], y[splits[2][0]:splits[2][1]]

del X, y, df

pct = y_train.mean()*100
print(f"  训练: {len(X_train):,} (正{pct:.1f}%), 验证: {len(X_val):,}, 测试: {len(X_test):,}")

# 参数
params = {
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],
    'tree_method': 'hist', 'device': 'cuda',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42,
}

print("\n训练XGBoost...")
t0 = time.time()
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
dtest = xgb.DMatrix(X_test, y_test)

model = xgb.train(params, dtrain, num_boost_round=500,
                  evals=[(dtrain, 'train'), (dval, 'val')],
                  early_stopping_rounds=20, verbose_eval=False)
train_time = time.time() - t0

# 测试
y_pred_raw = model.predict(dtest)
y_pred_binary = (y_pred_raw > 0.5).astype(int)

# 指标
auc = roc_auc_score(y_test, y_pred_raw)
acc = (y_pred_binary == y_test).mean()
prec = precision_score(y_test, y_pred_binary)
rec = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f"\n  裸模型:")
print(f"  AUC:   {auc:.4f}")
print(f"  Acc:   {acc:.4f}")
print(f"  精度:  {prec:.4f}")
print(f"  召回:  {rec:.4f}")
print(f"  F1:    {f1:.4f}")

# 校准
print("\n校准...")
ir = IsotonicRegression(out_of_bounds='clip')
val_raw = model.predict(dval)
ir.fit(val_raw, y_val)
y_calib = ir.transform(y_pred_raw)

pred_gain = y_calib.mean()
actual_gain = y_test.mean()
calib_bias = (pred_gain - actual_gain) * 100

# 校准后重新算决策指标(0.5阈值)
y_calib_binary = (y_calib > 0.5).astype(int)
calib_acc = (y_calib_binary == y_test).mean()
calib_prec = precision_score(y_test, y_calib_binary)
calib_rec = recall_score(y_test, y_calib_binary)

print(f"  校准后偏差: {calib_bias:.2f}%")
print(f"  Acc:  {calib_acc:.4f}")
print(f"  精度: {calib_prec:.4f}")
print(f"  召回: {calib_rec:.4f}")

# 分档校准
print("\n校准分档:")
bins = np.linspace(0, 1, 11)
for i in range(10):
    lo, hi = bins[i], bins[i+1]
    mask = (y_calib >= lo) & (y_calib < hi)
    if mask.sum() > 0:
        actual = y_test[mask].mean()
        predicted = y_calib[mask].mean()
        diff = predicted - actual
        print(f"  [{lo:.1f}-{hi:.1f}): 预测{predicted:.3f}, 实际{actual:.3f}, 偏差{diff*100:+.2f}% ({mask.sum():,}样本)")

# 特征重要性
importances = model.get_score(importance_type='gain')
total = sum(importances.values()) or 1
imp_pct = {feat_cols[int(k[1:])]: float(v/total*100) for k, v in importances.items()}
top = sorted(imp_pct.items(), key=lambda x: -x[1])[:10]
print(f"\nTop 10特征:")
for name, pct in top:
    print(f"  {name}: {pct:.1f}%")

# 保存模型
model.save_model(MODEL_OUT)
print(f"\n模型保存: {MODEL_OUT}")

# 生成推荐
print("\n生成最新推荐...")
df2 = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
latest = df2.groupby('ticker').last().reset_index()
X_latest = latest[feat_cols].values.astype(np.float32)
dlatest = xgb.DMatrix(X_latest)
preds = model.predict(dlatest)
probs = ir.transform(preds)

top50 = latest.copy()
top50['prob_gain_5pct'] = np.round(probs, 4)
top50 = top50.sort_values('prob_gain_5pct', ascending=False)

# Only show >50% 
print(f"\n{'='*60}")
print(f"绿箭v5 (5%二分类) 推荐")
print(f"{'='*60}")
print(f"评测: AUC={auc:.4f}, Acc={calib_acc:.4f}, 精度={calib_prec:.4f}, 校准偏差={calib_bias:.2f}%")
print(f"{'='*60}")
print(f"{'#':>3} {'代码':<8} {'日期':<12} {'涨>5%概率':>10}")
print("-"*40)
top15 = top50.head(15)
for i, (_, row) in enumerate(top15.iterrows()):
    prob = row['prob_gain_5pct']
    print(f"{i+1:>3} {row['ticker']:<8} {str(row['date'])[:10]:<12} {prob:.4f}")

# 保存推荐
top50_list = top50.head(100).to_dict('records')
pred_data = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'threshold': '>5%',
    'model_type': 'binary_logistic',
    'auc': round(float(auc), 4),
    'accuracy': round(float(calib_acc), 4),
    'precision': round(float(calib_prec), 4),
    'calib_bias_pct': round(float(calib_bias), 2),
    'config': {'max_depth': 6, 'lr': 0.05, 'n_estimators': 500, 'subsample': 0.8, 'colsample': 0.8},
    'top_100': [{k: (str(v) if isinstance(v, (pd.Timestamp, np.integer)) else float(v) if isinstance(v, np.floating) else v) for k, v in item.items() if k in ['ticker','date','prob_gain_5pct']} for item in top50_list],
}
json.dump(pred_data, open(PRED_OUT, 'w'), indent=2)

print(f"\n{'='*60}")
print(f"参考: v19 AUC 0.64ish")
print(f"      v5  AUC {auc:.4f}")
print(f"      说人话: 模型排序能力 {'还不错' if auc > 0.65 else '一般' if auc > 0.6 else '不太行'}")
print(f"总耗时: {(time.time()-t0)/60:.1f}分钟")
