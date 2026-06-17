#!/usr/bin/env python3
"""
美股ML v6 — XGB + n_estimators=500 + 3分类Label
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

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()

print("═══ v6 XGB+500树+3分类Label ═══")

df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']

df = df.dropna(subset=['label_pct'] + feature_cols).copy()

# ─── 3分类Label ───
# 涨>2% → 2, 涨跌-2%~+2% → 1, 跌>2% → 0
# 用略窄的窗口（比原5%敏感），目标是抓更多涨的
def bucket3(pct):
    if pct > 2: return 2   # 涨
    if pct > -2: return 1  # 平
    return 0                # 跌

df['label_bucket'] = df['label_pct'].apply(bucket3)

dist = df['label_bucket'].value_counts(normalize=True).sort_index()
print(f"\n  3分类分布:")
for i, name in [(0,'跌>2%'), (1,'平±2%'), (2,'涨>2%')]:
    print(f"    {name}: {dist.get(i,0)*100:.1f}%")

# ─── Walk-Forward ───
print("\n[1/4] Walk-Forward (3窗口)...")

X = df[feature_cols].values
y_bucket = df['label_bucket'].values
y_actual = df['label_pct'].values

classes = np.array([0, 1, 2])
weights = compute_class_weight('balanced', classes=classes, y=y_bucket)
weight_dict = {i: w for i, w in enumerate(weights)}
print(f"  权重: {[f'{w:.2f}' for w in weights]}")

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
    
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=20,
        random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
    model.fit(X_tr, y_tr,
              sample_weight=sw,
              eval_set=[(X_va, y_va)],
              verbose=0)
    
    y_proba = model.predict_proba(X_te)
    proba_up = y_proba[:, 2]  # 涨>2%的概率
    all_preds.extend(proba_up.tolist())
    all_actuals.extend(y_te_act.tolist())
    
    top10 = proba_up >= np.percentile(proba_up, 90)
    if top10.sum() > 5:
        r = y_te_act[top10]
        sp = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
        wr = (r > 0).mean()
        print(f"  {name}: 测{len(X_te)}行, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

# 合成
preds = np.array(all_preds)
actuals = np.array(all_actuals)
top10_mask = preds >= np.percentile(preds, 90)
top_ret = actuals[top10_mask]
wf_sharpe = top_ret.mean() / top_ret.std() * math.sqrt(252) if top_ret.std() > 0 else 0
wf_wr = (top_ret > 0).mean()
print(f"\n  Walk-Forward合成: 夏普={wf_sharpe:.3f}, 胜率={wf_wr:.1%}")

# ─── 全量 ───
print("\n[2/4] 训练全量模型...")
train_end = int(len(df) * 0.85)
sw = np.array([weight_dict[yi] for yi in y_bucket[:train_end]])

final = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
final.fit(X[:train_end], y_bucket[:train_end],
          sample_weight=sw,
          eval_set=[(X[train_end:], y_bucket[train_end:])],
          verbose=0)

y_proba = final.predict_proba(X[train_end:])
proba_up = y_proba[:, 2]
test_actual = y_actual[train_end:]
top10 = proba_up >= np.percentile(proba_up, 90)
r = test_actual[top10]
sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wr = (r > 0).mean()
acc = accuracy_score(y_bucket[train_end:], final.predict(X[train_end:]))
print(f"  测试: acc={acc:.3f}, 夏普={sharpe:.3f}, 胜率={wr:.1%}")

# ─── 今日预测 ───
print("\n[3/4] 今日预测...")
latest = df.dropna(subset=feature_cols).drop_duplicates(subset='sym', keep='last')
X_latest = latest[feature_cols].values
y_latest_proba = final.predict_proba(X_latest)

# 将3分类概率映射回期望收益估算
# 涨>2%, 平±2%, 跌>2%
results = []
for i, (_, row) in enumerate(latest.iterrows()):
    probs = y_latest_proba[i].tolist()
    expected_ret = sum(probs[j] * v for j, v in enumerate([-3, 0, 3]))
    results.append({
        'sym': row['sym'], 'price': float(row['price']),
        'prob_up': probs[2], 'prob_flat': probs[1], 'prob_down': probs[0],
        'duration': row['label_pct'] if 'label_pct' in row else 0,  # 占位
    })
results.sort(key=lambda x: -x['prob_up'])

print(f"\n{'═'*60}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>2%':>7} {'平±2%':>7} {'跌>2%':>7} {'总涨':>6}")
print(f"{'─'*60}")
for i, r in enumerate(results[:30]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up']*100:>6.1f}% "
          f"{r['prob_flat']*100:>6.1f}% {r['prob_down']*100:>6.1f}% {r['prob_up']*100:>5.1f}%")

# ─── 保存 ───
print("\n[4/4] 保存...")
final.save_model(f"{MODEL_DIR}/us_xgb_v6.json")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_xgb_v6',
    'label_type': '3class(>2%/±2%/>2%)',
    'wf_sharpe': round(wf_sharpe, 4), 'wf_win_rate': round(wf_wr, 4),
    'final_sharpe': round(sharpe, 4), 'final_win_rate': round(wr, 4),
    'final_accuracy': round(acc, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
    'all_scores': results,
}
with open(f"{MODEL_DIR}/us_xgb_v6_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v6 完成! ({TOTAL:.0f}s)")
print(f"  WF夏普: {wf_sharpe:.3f} | 测试夏普: {sharpe:.3f}")
