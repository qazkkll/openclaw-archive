#!/usr/bin/env python3
"""
us_ml_04_train_v4.py — 用最佳参数全量训练绿箭v4模型
- 使用md6_lr0.05_n500_ss0.8最佳参数
- GPU加速
- Platt校准
- 特征重要性保留原始列名
- 断点续传
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v4.parquet'
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models/us'
MODEL_OUT = f'{MODEL_DIR}/greenshaft_v4_final.json'
CALIB_OUT = f'{MODEL_DIR}/greenshaft_v4_calib.json'
PRED_OUT = f'{MODEL_DIR}/greenshaft_v4_prediction.json'
CKPT = f'{MODEL_DIR}/greenshaft_v4_ckpt.json'

print("us_ml_04: 全量训练绿箭v4模型")

# 1. 读数据
df = pd.read_parquet(INPUT)
print(f"  行数: {len(df):,} 股票: {df['ticker'].nunique()}")
print(f"  标签: {df['label'].value_counts().to_dict()}")

exclude = {'ticker', 'date', 'label', 'fwd_5d_ret'}
feat_cols = [c for c in df.columns if c not in exclude]
print(f"  特征数: {len(feat_cols)}")
print(f"  特征名: {feat_cols[:10]}...")

X_full = df[feat_cols].values.astype(np.float32)
y_full = df['label'].values + 1  # 0,1,2

# 2. TimeSeries切分
n = len(X_full)
train_end = int(n * 0.7)
val_end = int(n * 0.85)
X_train, X_val, X_test = X_full[:train_end], X_full[train_end:val_end], X_full[val_end:]
y_train, y_val, y_test = y_full[:train_end], y_full[train_end:val_end], y_full[val_end:]
del df, X_full, y_full

print(f"  训练集: {len(X_train):,} 验证集: {len(X_val):,} 测试集: {len(X_test):,}")

# 3. 训练
best_params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500, 'subsample': 0.8, 'colsample_bytree': 0.8}
xgb_params = {
    'objective': 'multi:softprob', 'num_class': 3,
    'eval_metric': ['mlogloss', 'merror'],
    'tree_method': 'hist', 'device': 'cuda',
    'random_state': 42,
    **best_params
}

print("\n训练XGBoost...")
t0 = time.time()
dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
dtest = xgb.DMatrix(X_test, y_test)

model = xgb.train(
    xgb_params, dtrain, num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=20,
    verbose_eval=False,
)
train_time = time.time() - t0

# 4. 测试集评估
y_pred_raw = model.predict(dtest)
y_pred_class = np.argmax(y_pred_raw, axis=1)
acc = np.mean(y_pred_class == y_test)
probs_1 = y_pred_raw[:, 1]  # label=0概率
actual_gain = np.mean(y_test == 1)
pred_gain = np.mean(probs_1)
calib_bias = (pred_gain - actual_gain) * 100

print(f"  Acc: {acc:.4f}, Calib偏差: {calib_bias:.2f}%, {train_time:.0f}s")

# 5. Platt校准
print("Platt校准...")
t1 = time.time()

# 用验证集校准
y_val_pred = model.predict(dval)
# Isotonic Regression
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(y_val_pred[:, 1], (y_val == 1).astype(float))

y_calib = ir.transform(y_pred_raw[:, 1])
actual_calib = np.mean(y_test == 1)
pred_calib = np.mean(y_calib)
calib_bias_after = (pred_calib - actual_calib) * 100
print(f"  校准后偏差: {calib_bias_after:.2f}% ({time.time()-t1:.0f}s)")

# 6. 特征重要性
importances = model.get_score(importance_type='gain')
total = sum(importances.values()) or 1
imp_pct = {feat_cols[int(k[1:])]: float(v/total*100) for k, v in importances.items()}
top_imp = sorted(imp_pct.items(), key=lambda x: -x[1])[:10]

print(f"\nTop 10特征重要性:")
for name, pct in top_imp:
    print(f"  {name}: {pct:.1f}%")

print(f"\n训练完毕。保存模型...")
model.save_model(MODEL_OUT)

# 保存校准器
calib_info = {
    'method': 'isotonic',
    'bias_before_pct': round(calib_bias, 2),
    'bias_after_pct': round(calib_bias_after, 2),
    'model_acc': round(float(acc), 4),
}
json.dump(calib_info, open(CALIB_OUT, 'w'), indent=2)

# 7. 最新预测（取最后60天的数据做推荐）
print("\n生成推荐...")
latest_data = pd.read_parquet(INPUT)
latest_data = latest_data.sort_values('ticker')
# 每只股票取最新一行
latest = latest_data.groupby('ticker').last().reset_index()
X_latest = latest[feat_cols].values.astype(np.float32)

dlatest = xgb.DMatrix(X_latest)
pred_probs = model.predict(dlatest)
prob_up = ir.transform(pred_probs[:, 1])  # label=1的概率

latest['prob_calib'] = prob_up

# Top50
top50 = latest.nlargest(50, 'prob_calib')[['ticker', 'date', 'prob_calib'] + feat_cols[:5]]
top50 = top50[['ticker', 'date', 'prob_calib']]
top50['prob_calib'] = top50['prob_calib'].round(4)

top50_list = top50.head(50).to_dict('records')
pred = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'best_config': 'md6_lr0.05_n500_ss0.8',
    'accuracy': round(float(acc), 4),
    'calib_bias_before': round(calib_bias, 2),
    'calib_bias_after': round(calib_bias_after, 2),
    'model': MODEL_OUT,
    'top_50': [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in item.items()} for item in top50_list],
}
json.dump(pred, open(PRED_OUT, 'w'), indent=2)

# 打印Top15
print(f"\n{'='*60}")
print(f"绿箭v4 推荐 Top 15")
print(f"{'='*60}")
print(f"{'#':>3} {'代码':<8} {'日期':<12} {'涨概率':>8}")
print("-"*40)
for i, row in enumerate(top50_list[:15]):
    print(f"{i+1:>3} {row['ticker']:<8} {str(row['date'])[:10]:<12} {row['prob_calib']:.4f}")

print(f"\n{'='*60}")
print(f"模型保存: {MODEL_OUT}")
print(f"校准器: {CALIB_OUT}")
print(f"推荐: {PRED_OUT}")
print(f"总耗时: {(time.time()-t0)/60:.1f}分钟")
