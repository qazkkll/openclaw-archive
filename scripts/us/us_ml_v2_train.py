#!/usr/bin/env python3
"""
美股ML训练 — 从预计算parquet读取
用法: python3 scripts/us_ml_v2_train.py
"""

import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()

print("[1/2] 加载预计算特征...")
t = time.time()
df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
with open(f"{ML_DIR}/us_feature_cols.json") as f:
    feature_cols = json.load(f)
print(f"  数据: {len(df)}行, {len(df.columns)}列")
print(f"  特征: {len(feature_cols)}列: {feature_cols}")
print(f"  加载耗时: {time.time()-t:.0f}s")

# 只对label_pct有效的行训练
df_train = df.dropna(subset=['label_pct']).copy()
print(f"  有效训练行(有label): {len(df_train)}行")

# ─── 训练 ───
print("\n[2/2] 训练...")

X = df_train[feature_cols].values
y_bucket = df_train['label_bucket'].values
y_actual = df_train['label_pct'].values
syms = df_train['sym'].values
prices = df_train['price'].values

# 按时间分 80/20
split = int(len(df_train) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_bucket[:split], y_bucket[split:]
test_syms = syms[split:]
test_actual = y_actual[split:]
test_prices = prices[split:]

print(f"  训练: {len(X_train)}, 测试: {len(X_test)}")

# 类别权重
classes = np.array([0, 1, 2, 3])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
weight_dict = {i: w for i, w in enumerate(weights)}
sample_weight = np.array([weight_dict[yi] for yi in y_train])
print(f"  类别权重: {dict(zip(['大跌','小跌','小涨','大涨'],[f'{w:.2f}' for w in weights]))}")

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, num_class=4, device='cuda')

model.fit(X_train, y_train,
          sample_weight=sample_weight,
          eval_set=[(X_test, y_test)],
          verbose=50)
train_time = time.time() - t
print(f"  训练耗时: {time.time()-t:.0f}s")

# ─── 评估 ───
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n  测试集准确率: {acc:.3f}")
print(classification_report(y_test, y_pred,
      target_names=['大跌<-5%','小跌-5~0%','小涨0~5%','大涨>5%']))

# Top10%策略
proba_large_up = y_proba[:, 3]
proba_up = y_proba[:, 2] + y_proba[:, 3]

test_df = pd.DataFrame({
    'sym': test_syms, 'price': test_prices, 'actual_ret': test_actual,
    'proba_large_up': proba_large_up, 'proba_up': proba_up,
    'pred': y_pred,
})

# 按大涨概率排序
ranked = test_df.sort_values('proba_large_up', ascending=False)
top10 = ranked.head(int(len(ranked) * 0.1))
win_rate = (top10['actual_ret'] > 0).mean()
avg_ret = top10['actual_ret'].mean()
sharpe = avg_ret / top10['actual_ret'].std() * math.sqrt(252) if top10['actual_ret'].std() > 0 else 0
print(f"\n  Top10%策略模拟:")
print(f"    胜率(涨): {win_rate:.1%} ({len(top10)}样本)")
print(f"    平均收益: {avg_ret:.2f}%")
print(f"    夏普(年化): {sharpe:.3f}")

# ─── 今日预测 ───
print(f"\n─── 今日预测 ───")
latest = df.dropna(subset=feature_cols).drop_duplicates(subset='sym', keep='last')
print(f"  可评分股票: {len(latest)}只")

X_latest = latest[feature_cols].values
y_latest_proba = model.predict_proba(X_latest)

results = []
for i, (_, row) in enumerate(latest.iterrows()):
    probs = y_latest_proba[i].tolist()
    expected_ret = sum(probs[j] * v for j, v in enumerate([-7, -2.5, 2.5, 7]))
    results.append({
        'sym': row['sym'], 'price': float(row['price']),
        'prob_large_up': probs[3], 'prob_small_up': probs[2],
        'prob_small_down': probs[1], 'prob_large_down': probs[0],
        'prob_up': probs[2] + probs[3],
        'expected_ret': expected_ret,
    })

results.sort(key=lambda x: -x['expected_ret'])

print(f"\n{'═'*80}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%':>7} {'涨0~5%':>7} {'跌0~5%':>7} {'跌>5%':>7} {'总涨':>6} {'期望%':>6}")
print(f"{'─'*80}")
for i, r in enumerate(results[:30]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_large_up']*100:>6.1f}% {r['prob_small_up']*100:>6.1f}% "
          f"{r['prob_small_down']*100:>6.1f}% {r['prob_large_down']*100:>6.1f}% {r['prob_up']*100:>5.1f}% {r['expected_ret']:>6.2f}%")

# 保存
output = {
    'timestamp': now.isoformat(),
    'model_type': 'xgb_multiclass_v2',
    'feature_cols': feature_cols,
    'train_samples': int(len(X_train)),
    'test_samples': int(len(X_test)),
    'train_time_sec': round(train_time, 1),
    'test_accuracy': round(acc, 4),
    'test_sharpe_top10pct': round(sharpe, 4),
    'test_win_rate_top10pct': round(win_rate, 4),
    'test_avg_ret_top10pct': round(avg_ret, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
    'all_scores': results,
}

model.save_model(f"{MODEL_DIR}/us_xgb_v2.json")
with open(f"{MODEL_DIR}/us_xgb_v2_prediction.json", 'w') as f:
    json.dump(output, f, indent=2)

TOTAL = time.time() - T0
print(f"\n✅ 美股ML v2 完成! ({TOTAL:.0f}s)")
print(f"   模型: data/models/us_xgb_v2.json")
print(f"   预测: data/models/us_xgb_v2_prediction.json")
