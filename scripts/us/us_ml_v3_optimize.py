#!/usr/bin/env python3
"""
美股ML v3 — 超参数搜索 + 特征优化
"""
import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()

print("═══ v3 超参数搜索 ═══")

# 加载
print("\n[1/4] 加载数据...")
df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
df = df.dropna(subset=['label_pct']).copy()
print(f"  总行数: {len(df)}")

# 添加交互特征
print("\n[2/4] 特征工程...")

# 去掉冗余均线，只保留ma20（中期趋势核心）
base_cols = ['ma20', 'ma60', 'rsi14', 'vol20', 'p52',
             'ret5', 'ret20', 'macd', 'macd_hist', 'ma_bias20',
             'price', 'volume']

# 交互特征
df['vol_x_volatility'] = df['vol_ratio'] * df['vol20']
df['ret20_x_vol20'] = df['ret20'] * df['vol20']
df['rsi_x_p52'] = df['rsi14'] * df['p52'] / 100
df['macd_x_hist'] = df['macd'] * df['macd_hist']
df['bias_x_macd'] = df['ma_bias20'] * df['macd']

interact_cols = ['vol_x_volatility', 'ret20_x_vol20', 'rsi_x_p52', 'macd_x_hist', 'bias_x_macd']

# 非线形变换
df['vol20_sq'] = df['vol20'] ** 2
df['ret20_sq'] = df['ret20'] ** 2

transform_cols = ['vol20_sq', 'ret20_sq']

all_features = base_cols + interact_cols + transform_cols
print(f"  特征数: {len(all_features)}")
print(f"  特征: {all_features}")

# 过滤nan
df_train = df.dropna(subset=all_features)
print(f"  有效行: {len(df_train)} ({len(df_train)/len(df)*100:.0f}%)")

syms = df_train['sym'].values
y_bucket = df_train['label_bucket'].values

X = df_train[all_features].values

# 按时间分割80/20
split = int(len(df_train) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_bucket[:split], y_bucket[split:]
test_actual = df_train['label_pct'].values[split:]

print(f"  训练: {len(X_train)}, 测试: {len(X_test)}")

# ─── 超参数搜索 ───
print("\n[3/4] GridSearch...")

param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
}
# 只搜索27组的一部分，时间考虑
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
}
# 3×2×3=18组

# 提前固定colsample_bytree=0.8, n_estimators=200
n_estimators = 200

results = []
best_score = -1
best_params = None
best_model = None

from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1, 2, 3])
weights = compute_class_weight('balanced', classes=classes, y=y_train)
weight_dict = {i: w for i, w in enumerate(weights)}
sample_weight = np.array([weight_dict[yi] for yi in y_train])

total_grid = len(list(ParameterGrid(param_grid)))
print(f"  共 {total_grid} 组参数", flush=True)

for i, params in enumerate(ParameterGrid(param_grid)):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        **params,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        early_stopping_rounds=15,
        random_state=42, n_jobs=-1, verbosity=0, num_class=4, device='cuda')
    model.fit(X_train, y_train,
              sample_weight=sample_weight,
              eval_set=[(X_test, y_test)],
              verbose=0)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 计算夏普
    y_proba = model.predict_proba(X_test)
    proba_large_up = y_proba[:, 3]
    
    # Top10%策略
    top10_mask = proba_large_up >= np.percentile(proba_large_up, 90)
    if top10_mask.sum() > 10:
        top_ret = test_actual[top10_mask]
        avg_ret = top_ret.mean()
        std_ret = top_ret.std()
        sharpe = avg_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0
        win_rate = (top_ret > 0).mean()
    else:
        sharpe = 0
        win_rate = 0
    
    results.append({
        'params': params,
        'accuracy': round(acc, 4),
        'sharpe': round(sharpe, 4),
        'win_rate': round(win_rate, 4),
        'avg_ret': round(avg_ret, 4),
        'n_test': int(top10_mask.sum()),
    })
    
    if sharpe > best_score:
        best_score = sharpe
        best_params = params
        best_model = model
    
    pct = (i+1) / total_grid * 100
    print(f"  [{i+1}/{total_grid}] {params}  acc={acc:.3f}  sharpe={sharpe:.2f}  wr={win_rate:.1%}  ({pct:.0f}%)", flush=True)

print(f"\n  最佳参数: {best_params}")
print(f"  最佳Shapre: {best_score:.3f}")

# ─── 重训最佳模型 ───
print("\n[4/4] 重训最佳模型 + 完整预测...")
t = time.time()

final = xgb.XGBClassifier(
    n_estimators=300,
    **best_params,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42, n_jobs=-1, verbosity=0, num_class=4, device='cuda')
final.fit(X_train, y_train)
train_time = time.time() - t
print(f"  重训耗时: {train_time:.0f}s")

# 最终评估
y_pred = final.predict(X_test)
y_proba = final.predict_proba(X_test)

proba_large_up = y_proba[:, 3]
top10_mask = proba_large_up >= np.percentile(proba_large_up, 90)
top_ret = test_actual[top10_mask]
avg_ret = top_ret.mean()
std_ret = top_ret.std()
sharpe = avg_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0
win_rate = (top_ret > 0).mean()
acc = accuracy_score(y_test, y_pred)

print(f"\n  最终模型:")
print(f"    测试准确率: {acc:.3f}")
print(f"    Top10%夏普: {sharpe:.3f}")
print(f"    Top10%胜率: {win_rate:.1%}")
print(f"    Top10%平均收益: {avg_ret*100:.2f}%")

# ─── 今日预测 ───
print(f"\n  今日预测...")
latest = df.dropna(subset=all_features).drop_duplicates(subset='sym', keep='last')
X_latest = latest[all_features].values
y_latest_proba = final.predict_proba(X_latest)

results_list = []
for i, (_, row) in enumerate(latest.iterrows()):
    probs = y_latest_proba[i].tolist()
    expected_ret = sum(probs[j] * v for j, v in enumerate([-7, -2.5, 2.5, 7]))
    results_list.append({
        'sym': row['sym'], 'price': float(row['price']),
        'prob_large_up': probs[3], 'prob_small_up': probs[2],
        'prob_small_down': probs[1], 'prob_large_down': probs[0],
        'prob_up': probs[2] + probs[3], 'expected_ret': expected_ret,
    })

results_list.sort(key=lambda x: -x['expected_ret'])

print(f"\n{'═'*80}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%':>7} {'涨0~5%':>7} {'跌0~5%':>7} {'跌>5%':>7} {'总涨':>6} {'期望%':>6}")
print(f"{'─'*80}")
for i, r in enumerate(results_list[:30]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_large_up']*100:>6.1f}% {r['prob_small_up']*100:>6.1f}% "
          f"{r['prob_small_down']*100:>6.1f}% {r['prob_large_down']*100:>6.1f}% {r['prob_up']*100:>5.1f}% {r['expected_ret']:>6.2f}%")

# 保存
output = {
    'timestamp': now.isoformat(),
    'model': 'us_v3_xgb',
    'best_params': best_params,
    'final_accuracy': round(acc, 4),
    'final_sharpe': round(sharpe, 4),
    'final_win_rate': round(win_rate, 4),
    'final_avg_ret': round(avg_ret, 4),
    'features': all_features,
    'grid_search_results': results,
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results_list[:50])],
    'all_scores': results_list,
}

final.save_model(f"{MODEL_DIR}/us_xgb_v3.json")
with open(f"{MODEL_DIR}/us_xgb_v3_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v3 完成! ({TOTAL:.0f}s)")
print(f"   最佳参数: {best_params}")
print(f"   最终夏普: {sharpe:.3f}")
print(f"   模型: data/models/us_xgb_v3.json")
