#!/usr/bin/env python3
"""
美股ML v11 — 只用最近2年数据训练 + v9的配置
最新数据可能比旧数据更有价值
"""
import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# 使用统一路径
sys.path.insert(0, r'/home/hermes/.hermes/openclaw-archive\scripts')
import _paths

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
os.makedirs(_paths.MODEL_DIR, exist_ok=True)
T0 = time.time()

print("═══ v11 只用最近2年数据训练 ═══")

df = pd.read_parquet(_paths.US_ML_FEATS)
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']

df['vol20_ma20'] = df['vol20'] - (df['volume'] / df['volume'].rolling(20).mean())
df['vol20_ma20'] = df['vol20_ma20'].fillna(0)
df['vol20_ma60'] = df['vol20'] - (df['volume'] / df['volume'].rolling(60).mean())
df['vol20_ma60'] = df['vol20_ma60'].fillna(0)
df['ret60_vol20'] = df['ret60'] * df['vol20']

all_features = feature_cols + ['vol20_ma20', 'vol20_ma60', 'ret60_vol20']
df = df.dropna(subset=['label_pct'] + all_features).copy()

def bucket3(p):
    if p > 2: return 2
    if p > -2: return 1
    return 0
df['label_bucket'] = df['label_pct'].apply(bucket3)

X = df[all_features].values
y_b = df['label_bucket'].values
y_a = df['label_pct'].values

# 取后45% ≈ 最后2年 (1209399×0.45≈544k行)
TRAIN_RATIO = 0.45  # 只用最后45%的数据
N = len(df)
cut = int(N * (1 - TRAIN_RATIO))
X_recent, y_b_recent, y_a_recent = X[cut:], y_b[cut:], y_a[cut:]
print(f"  总数据: {N}行")
print(f"  只用后{TRAIN_RATIO*100:.0f}%: {len(X_recent)}行 (≈最后2年)")

# Walk-Forward — 只用近期数据内部分割
# 近期数据再分: 前75%训练, 后25%测试
wf_n = len(X_recent)
wf_split = int(wf_n * 0.75)

classes = np.array([0,1,2])
weights = compute_class_weight('balanced', classes=classes, y=y_b_recent)
wd = {i:w for i,w in enumerate(weights)}
print(f"  权重: {[f'{w:.2f}' for w in weights]}")

# 时间衰减
sw = np.array([wd[yi] for yi in y_b_recent])
decay = np.linspace(0.5, 1.0, wf_n)

train_X = X_recent[:wf_split]
train_y = y_b_recent[:wf_split]
test_X = X_recent[wf_split:]
test_y_b = y_b_recent[wf_split:]
test_actual = y_a_recent[wf_split:]
test_sw = sw[:wf_split] * decay[:wf_split]

model = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
model.fit(train_X, train_y,
          sample_weight=test_sw,
          eval_set=[(test_X, test_y_b)],
          verbose=50)

y_proba = model.predict_proba(test_X)
proba_up = y_proba[:, 2]
top10 = proba_up >= np.percentile(proba_up, 90)
r = test_actual[top10]
sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wr = (r > 0).mean()
acc = accuracy_score(test_y_b, model.predict(test_X))
print(f"\n  仅近期数据测试: acc={acc:.3f}, 夏普={sharpe:.3f}, 胜率={wr:.1%}")
print(f"  Top10%平均收益: {r.mean()*100:.2f}%")

# 对比：全量数据
print("\n[对比] 全量数据训练...")
train_end_all = int(N * 0.85)
sw_full = np.array([wd[yi] for yi in y_b[:train_end_all]])
decay_full = np.linspace(0.3, 1.0, train_end_all)
sw_full *= decay_full

model_all = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
model_all.fit(X[:train_end_all], y_b[:train_end_all],
              sample_weight=sw_full,
              eval_set=[(X[train_end_all:], y_b[train_end_all:])],
              verbose=0)

# 对比测试集：都用全量模型的后15%
y_proba_all = model_all.predict_proba(X[train_end_all:])
proba_up_all = y_proba_all[:, 2]
test_actual_all = y_a[train_end_all:]
top10_all = proba_up_all >= np.percentile(proba_up_all, 90)
r_all = test_actual_all[top10_all]
sharpe_all = r_all.mean() / r_all.std() * math.sqrt(252) if r_all.std() > 0 else 0
wr_all = (r_all > 0).mean()
acc_all = accuracy_score(y_b[train_end_all:], model_all.predict(X[train_end_all:]))
print(f"  全量测试: acc={acc_all:.3f}, 夏普={sharpe_all:.3f}, 胜率={wr_all:.1%}")

# 但公平对比：让全量模型也预测最近的15%
# 取全量模型后15%的最近一段 = test_X对应的行
# N * 0.85 到 N 的最近15%，和 wf_split 后的 test_X 比较
# test_X是从cut开始又取后25%，约等于全量总数据的后 (1-0.55)*0.25 ≈ 11%
# 近似用最近11%的数据做对比
recent_test_start = int(N * 0.89)
y_proba_all_recent = model_all.predict_proba(X[recent_test_start:])
proba_up_all_recent = y_proba_all_recent[:, 2]
test_actual_all_recent = y_a[recent_test_start:]
top10_all_recent = proba_up_all_recent >= np.percentile(proba_up_all_recent, 90)
r_all_recent = test_actual_all_recent[top10_all_recent]
sharpe_all_recent = r_all_recent.mean() / r_all_recent.std() * math.sqrt(252) if r_all_recent.std() > 0 else 0
print(f"  全量模型(最近11%): 夏普={sharpe_all_recent:.3f}")

print(f"\n  >>> 对比: 近期模型夏普={sharpe:.3f} vs 全量模型(近期)夏普={sharpe_all_recent:.3f}")
print(f"  >>> 如果近期模型胜,说明旧数据拖后腿")
print(f"  >>> 如果全量模型胜,说明旧数据还有价值")

# ─── 保存近期模型（全量数据训练一个最终版用于预测） ───
model_all.save_model(_paths.MODEL_DIR + "/us_xgb_v11.json")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_xgb_v11_near',
    'recent_only_sharpe': round(float(sharpe), 4),
    'recent_only_win_rate': round(float(wr), 4),
    'full_model_sharpe': round(float(sharpe_all), 4),
    'full_model_recent_sharpe': round(float(sharpe_all_recent), 4),
    'full_model_accuracy': round(float(acc_all), 4),
}
with open(_paths.MODEL_DIR + "/us_xgb_v11_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v11 完成! ({TOTAL:.0f}s)")
