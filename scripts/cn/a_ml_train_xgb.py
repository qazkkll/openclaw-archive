"""ML第2步：训练XGBoost分类模型
读取 ml_training_data.parquet -> 训练XGBoost -> 回测对比A1

输入: /home/hermes/.hermes/openclaw-archive/data\ml_training_data.parquet
输出: data/models/xgb_v1.json / xgb_v1_metrics.json
运行: python scripts/ml_train_xgb.py
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

T0 = time.time()

print("[1/5] 加载训练数据...")
df = pd.read_parquet(os.path.join(DATA_OUT, "ml_training_data.parquet"))
print(f"  行数: {len(df)}, 列: {len(df.columns)}")

with open(os.path.join(DATA_OUT, "ml_feature_cols.json")) as f:
    feature_cols = json.load(f)

print("[1b] 生成高收益Label...")
closes = df["close"].values
close_next = df["close_next"].values
pct_change = (close_next - closes) / closes * 100
df["label_high"] = (pct_change > 2.0).astype(int)
df["label_normal"] = (close_next > closes).astype(int)

print(f"  高收益型(>2%): {df['label_high'].mean()*100:.1f}% 正样本")
print(f"  普通型(>=0%): {df['label_normal'].mean()*100:.1f}% 正样本")

df = df.dropna(subset=feature_cols + ["label_high", "label_normal"])
print(f"  去NaN后: {len(df)}行")

print("[2/5] 按时间分割...")
df["date_parsed"] = pd.to_datetime(df["trade_date"])

train_mask = df["date_parsed"] < "2023-01-01"
val_mask = (df["date_parsed"] >= "2023-01-01") & (df["date_parsed"] < "2025-01-01")
test_mask = df["date_parsed"] >= "2025-01-01"

X_train = df[train_mask][feature_cols]
y_train = df[train_mask]["label_high"]
X_val = df[val_mask][feature_cols]
y_val = df[val_mask]["label_high"]
X_test = df[test_mask][feature_cols]
y_test = df[test_mask]["label_high"]

print(f"  训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")

# 普通涨跌也准备做对比
y_train_norm = df[train_mask]["label_normal"]
y_val_norm = df[val_mask]["label_normal"]
y_test_norm = df[test_mask]["label_normal"]

print("[3/5] 训练XGBoost...")
t = time.time()

pos_ratio = (y_train == 1).sum() / len(y_train)
print(f"  正样本比例: {pos_ratio:.3%}")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(1-pos_ratio)/pos_ratio,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    device='cuda',
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
train_time = time.time() - t
print(f"  训练耗时: {train_time:.0f}秒")
best_iter = model.best_iteration if hasattr(model, "best_iteration") else model.get_booster().best_iteration
print(f"  最佳迭代: {best_iter}")

print("[4/5] 评估...")
results = {}
for name, X, y in [("训练", X_train, y_train), ("验证", X_val, y_val), ("测试", X_test, y_test)]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    auc = roc_auc_score(y, proba)
    print(f"  {name}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} auc={auc:.4f}")
    results[name] = {"acc": round(acc,4), "precision": round(prec,4), "recall": round(rec,4), "auc": round(auc,4)}

# 特征重要性
importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
print("\nTop 10 特征:")
for i, row in importance.head(10).iterrows():
    print(f"  {row['feature']:20s} {row['importance']:.4f}")

# 测试集上普通label的AUC
proba_test = model.predict_proba(X_test)[:, 1]
auc_norm = roc_auc_score(y_test_norm, proba_test)
print(f"\n  高收益模型在普通label(涨/跌)上的AUC: {auc_norm:.4f}")

# 同时训练一个普通涨跌模型做对比
print("\n--- 训练普通涨跌模型做对比 ---")
model2 = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0, device='cuda')
model2.fit(X_train, y_train_norm, eval_set=[(X_val, y_val_norm)], verbose=False)
proba2 = model2.predict_proba(X_test)[:, 1]
auc2 = roc_auc_score(y_test_norm, proba2)
print(f"  普通模型在普通label上的AUC: {auc2:.4f}")

# 但普通模型对高收益目标的AUC
proba2_high = model2.predict_proba(X_test)[:, 1]
auc2_high = roc_auc_score(y_test, proba2_high)
print(f"  普通模型在高收益label上的AUC: {auc2_high:.4f}")

print("\n[5/5] 保存模型...")
model.save_model(os.path.join(MODEL_DIR, "xgb_v1_high.json"))
model2.save_model(os.path.join(MODEL_DIR, "xgb_v1_normal.json"))

metrics = {
    "train_time_sec": round(train_time, 1),
    "best_iteration": int(best_iter),
    "label_type": "high_return (>2%)",
    "n_train": int(len(X_train)),
    "n_val": int(len(X_val)),
    "n_test": int(len(X_test)),
    "test_positive_rate": float(y_test.mean()),
    "feature_count": len(feature_cols),
    "auc_on_normal_label": round(auc_norm, 4),
    "normal_model_auc": round(auc2, 4),
    "normal_model_auc_on_high": round(auc2_high, 4),
}
metrics["results"] = results

with open(os.path.join(MODEL_DIR, "xgb_v1_high_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

TOTAL = time.time() - T0
print(f"\n✅ 总耗时: {TOTAL:.0f}秒 ({TOTAL/60:.1f}分钟)")
