#!/usr/bin/env python3
"""
美股ML v5 — LightGBM + Walk-Forward
"""
import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()

print("═══ v5 LightGBM + Walk-Forward ═══")

df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']

df = df.dropna(subset=['label_pct'] + feature_cols).copy()
print(f"  总行数: {len(df)}, 特征: {len(feature_cols)}")

X = df[feature_cols].values
y_bucket = df['label_bucket'].values
y_actual = df['label_pct'].values

# ─── Walk-Forward ───
print("\n[1/4] Walk-Forward (3窗口)...")

from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1, 2, 3])
weights = compute_class_weight('balanced', classes=classes, y=y_bucket)
weight_dict = {i: w for i, w in enumerate(weights)}

wf_windows = [
    ('WF1', 0.00, 0.60, 0.60, 0.75, 0.75, 0.85),
    ('WF2', 0.15, 0.65, 0.65, 0.80, 0.80, 0.90),
    ('WF3', 0.30, 0.70, 0.70, 0.85, 0.85, 1.00),
]

all_preds = []
all_actuals = []

for name, ts, te, vs, ve, tst, tste in wf_windows:
    n = len(df)
    X_tr = X[int(ts*n):int(te*n)]
    y_tr = y_bucket[int(ts*n):int(te*n)]
    X_va = X[int(vs*n):int(ve*n)]
    y_va = y_bucket[int(vs*n):int(ve*n)]
    X_te = X[int(tst*n):int(tste*n)]
    y_te_act = y_actual[int(tst*n):int(tste*n)]
    
    sw = np.array([weight_dict[yi] for yi in y_tr])
    
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        num_leaves=31, min_child_samples=20,
        reg_alpha=0.01, reg_lambda=0.01,
        objective='multiclass', num_class=4,
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    # LightGBM有class_weight参数，更简单
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])
    
    y_proba = model.predict_proba(X_te)
    proba_large_up = y_proba[:, 3]
    all_preds.extend(proba_large_up.tolist())
    all_actuals.extend(y_te_act.tolist())
    
    top10 = proba_large_up >= np.percentile(proba_large_up, 90)
    if top10.sum() > 5:
        r = y_te_act[top10]
        sp = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
        wr = (r > 0).mean()
        print(f"  {name}: 测{len(X_te)}行, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

# Walk-Forward整体
preds = np.array(all_preds)
actuals = np.array(all_actuals)
top10_mask = preds >= np.percentile(preds, 90)
top_ret = actuals[top10_mask]
wf_sharpe = top_ret.mean() / top_ret.std() * math.sqrt(252) if top_ret.std() > 0 else 0
wf_wr = (top_ret > 0).mean()
print(f"\n  Walk-Forward合成: 夏普={wf_sharpe:.3f}, 胜率={wf_wr:.1%}")

# ─── 训练全量模型 ───
print("\n[2/4] 训练全量模型...")
train_end = int(len(df) * 0.85)
sw = np.array([weight_dict[yi] for yi in y_bucket[:train_end]])

final = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    num_leaves=31, min_child_samples=20,
    reg_alpha=0.01, reg_lambda=0.01,
    objective='multiclass', num_class=4,
    random_state=42, n_jobs=-1, verbosity=-1,
    class_weight='balanced',
)
final.fit(X[:train_end], y_bucket[:train_end],
          eval_set=[(X[train_end:], y_bucket[train_end:])],
          callbacks=[lgb.early_stopping(15), lgb.log_evaluation(0)])

# 测试
y_proba = final.predict_proba(X[train_end:])
proba_up = y_proba[:, 3]
test_actual = y_actual[train_end:]
top10 = proba_up >= np.percentile(proba_up, 90)
r = test_actual[top10]
sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wr = (r > 0).mean()
acc = accuracy_score(y_bucket[train_end:], final.predict(X[train_end:]))
print(f"  测试集: acc={acc:.3f}, 夏普={sharpe:.3f}, 胜率={wr:.1%}")

# ─── 今日预测 ───
print("\n[3/4] 今日预测...")
latest = df.dropna(subset=feature_cols).drop_duplicates(subset='sym', keep='last')
X_latest = latest[feature_cols].values
y_latest_proba = final.predict_proba(X_latest)

results = []
for i, (_, row) in enumerate(latest.iterrows()):
    probs = y_latest_proba[i].tolist()
    expected_ret = sum(probs[j] * v for j, v in enumerate([-7, -2.5, 2.5, 7]))
    results.append({
        'sym': row['sym'], 'price': float(row['price']),
        'prob_large_up': probs[3], 'prob_small_up': probs[2],
        'prob_small_down': probs[1], 'prob_large_down': probs[0],
        'prob_up': probs[2] + probs[3], 'expected_ret': expected_ret,
    })
results.sort(key=lambda x: -x['expected_ret'])

print(f"\n{'═'*80}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>5%':>7} {'涨0~5%':>7} {'跌0~5%':>7} {'跌>5%':>7} {'总涨':>6} {'期望%':>6}")
print(f"{'─'*80}")
for i, r in enumerate(results[:30]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_large_up']*100:>6.1f}% "
          f"{r['prob_small_up']*100:>6.1f}% {r['prob_small_down']*100:>6.1f}% "
          f"{r['prob_large_down']*100:>6.1f}% {r['prob_up']*100:>5.1f}% {r['expected_ret']:>6.2f}%")

# ─── 保存 ───
print("\n[4/4] 保存...")
final.booster_.save_model(f"{MODEL_DIR}/us_lgb_v5.txt")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_lgb_v5',
    'wf_sharpe': round(wf_sharpe, 4),
    'wf_win_rate': round(wf_wr, 4),
    'final_sharpe': round(sharpe, 4),
    'final_win_rate': round(wr, 4),
    'final_accuracy': round(acc, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
    'all_scores': results,
}
with open(f"{MODEL_DIR}/us_lgb_v5_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v5 完成! ({TOTAL:.0f}s)")
print(f"  Walk-Forward夏普: {wf_sharpe:.3f}")
