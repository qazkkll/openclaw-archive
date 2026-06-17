#!/usr/bin/env python3
"""
美股ML v10 — 不对称损失（涨权重更高）+ v9所有
"""
import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()

print("═══ v10 不对称损失 ═══")

df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']

# v9的新特征
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

print(f"  总行数: {len(df)}, 特征: {len(all_features)}")
dist = df['label_bucket'].value_counts(normalize=True).sort_index()
print(f"  分布: 跌>2%={dist.get(0,0)*100:.1f}%, 平={dist.get(1,0)*100:.1f}%, 涨>2%={dist.get(2,0)*100:.1f}%")

# ─── Walk-Forward ───
print("\n[1/3] Walk-Forward (不对称权重)...")

# 不对称权重：涨>2%的样本类权重加倍
# 标准类别平衡权重 + 涨类额外1.5倍
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0,1,2])
base_w = compute_class_weight('balanced', classes=classes, y=y_b)
# 不对称：涨的权重再乘1.5
base_w[2] *= 2.0  # 涨类权重翻倍
wd = {i:w for i,w in enumerate(base_w)}
print(f"  不对称权重: 跌={base_w[0]:.2f}, 平={base_w[1]:.2f}, 涨={base_w[2]:.2f}")

wf_windows = [
    ('WF1', 0.00, 0.60, 0.60, 0.75, 0.75, 0.85),
    ('WF2', 0.15, 0.65, 0.65, 0.80, 0.80, 0.90),
    ('WF3', 0.30, 0.70, 0.70, 0.85, 0.85, 1.00),
]
all_preds, all_actuals = [], []

for name, ts, te, vs, ve, tst, tste in wf_windows:
    n = len(df)
    tr_s, tr_e = int(ts*n), int(te*n)
    X_tr, y_tr = X[tr_s:tr_e], y_b[tr_s:tr_e]
    X_te = X[int(tst*n):int(tste*n)]
    y_te_act = y_a[int(tst*n):int(tste*n)]
    
    # 时间衰减 + 不对称权重
    sw = np.array([wd[yi] for yi in y_tr])
    decay = np.linspace(0.3, 1.0, len(sw))
    sw *= decay
    
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=20,
        random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
    model.fit(X_tr, y_tr,
              sample_weight=sw,
              eval_set=[(X[int(vs*n):int(ve*n)], y_b[int(vs*n):int(ve*n)])],
              verbose=0)
    
    y_proba = model.predict_proba(X_te)
    proba_up = y_proba[:, 2]
    all_preds.extend(proba_up.tolist())
    all_actuals.extend(y_te_act.tolist())
    
    top10 = proba_up >= np.percentile(proba_up, 90)
    if top10.sum() > 5:
        r = y_te_act[top10]
        sp = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
        wr = (r > 0).mean()
        print(f"  {name}: 测{len(X_te)}行, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

preds, acts = np.array(all_preds), np.array(all_actuals)
t10 = preds >= np.percentile(preds, 90)
r = acts[t10]
wf_sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wf_wr = (r > 0).mean()
print(f"\n  合成: 夏普={wf_sharpe:.3f}, 胜率={wf_wr:.1%}")

# ─── 全量模型 ───
print("\n[2/3] 训练全量模型...")
train_end = int(len(df) * 0.85)
sw_full = np.array([wd[yi] for yi in y_b[:train_end]])
decay = np.linspace(0.3, 1.0, train_end)
sw_full *= decay

final = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, num_class=3, device='cuda')
final.fit(X[:train_end], y_b[:train_end],
          sample_weight=sw_full,
          eval_set=[(X[train_end:], y_b[train_end:])],
          verbose=0)

y_proba = final.predict_proba(X[train_end:])
proba_up = y_proba[:, 2]
test_actual = y_a[train_end:]
top10 = proba_up >= np.percentile(proba_up, 90)
r = test_actual[top10]
sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wr = (r > 0).mean()
acc = accuracy_score(y_b[train_end:], final.predict(X[train_end:]))
print(f"  测试: acc={acc:.3f}, 夏普={sharpe:.3f}, 胜率={wr:.1%}")

# ─── 今日预测 ───
print("\n[3/3] 今日预测...")
latest = df.dropna(subset=all_features).drop_duplicates(subset='sym', keep='last')
X_latest = latest[all_features].values
y_proba_latest = final.predict_proba(X_latest)

results = []
for i, (_, row) in enumerate(latest.iterrows()):
    probs = y_proba_latest[i].tolist()
    results.append({
        'sym': row['sym'], 'price': float(row['price']),
        'prob_up': probs[2], 'prob_flat': probs[1], 'prob_down': probs[0],
    })
results.sort(key=lambda x: -x['prob_up'])

print(f"\n{'═'*60}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'涨>2%':>7} {'平±2%':>7} {'跌>2%':>7}")
print(f"{'─'*60}")
for i, r in enumerate(results[:30]):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_up']*100:>6.1f}% "
          f"{r['prob_flat']*100:>6.1f}% {r['prob_down']*100:>6.1f}%")

final.save_model(f"{MODEL_DIR}/us_xgb_v10.json")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_xgb_v10_asymmetric',
    'features': all_features,
    'wf_sharpe': round(wf_sharpe, 4), 'wf_win_rate': round(wf_wr, 4),
    'final_sharpe': round(sharpe, 4), 'final_win_rate': round(wr, 4),
    'final_accuracy': round(acc, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
    'all_scores': results,
}
with open(f"{MODEL_DIR}/us_xgb_v10_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v10 完成! ({TOTAL:.0f}s)")
