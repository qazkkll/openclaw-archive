#!/usr/bin/env python3
"""
美股ML v9 — 时间衰减权重 + vol_rank特征
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

print("═══ v9 时间衰减 + vol_rank ═══")

df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']
df = df.dropna(subset=['label_pct'] + feature_cols).copy()

# 加vol_rank特征：该样本的vol20在所有股票中的排名
# 按每只股票最后一天只有一条记录，我们用vol20的cross-section排名
# 近似：直接按行号窗口算vol20排名
# 简单点：vol20相对ma20的位置
df['vol20_ma20'] = df['vol20'] - (df['volume'] / df['volume'].rolling(20).mean())
df['vol20_ma20'] = df['vol20_ma20'].fillna(0)

# vol20相对ma60
df['vol20_ma60'] = df['vol20'] - (df['volume'] / df['volume'].rolling(60).mean())
df['vol20_ma60'] = df['vol20_ma60'].fillna(0)

# ret60_vol20: 60日动量 × 20日波动率
df['ret60_vol20'] = df['ret60'] * df['vol20']

new_feats = ['vol20_ma20', 'vol20_ma60', 'ret60_vol20']
all_features = feature_cols + new_feats
print(f"  特征: {len(all_features)}列 (新增: {new_feats})")

def bucket3(p):
    if p > 2: return 2
    if p > -2: return 1
    return 0
df['label_bucket'] = df['label_pct'].apply(bucket3)

X = df[all_features].values
y_b = df['label_bucket'].values
y_a = df['label_pct'].values

classes = np.array([0,1,2])
weights = compute_class_weight('balanced', classes=classes, y=y_b)
wd = {i:w for i,w in enumerate(weights)}

# ─── Walk-Forward ───
print("\n[1/3] Walk-Forward (3窗口, 含时间衰减)...")

wf_windows = [
    ('WF1', 0.00, 0.60, 0.60, 0.75, 0.75, 0.85),
    ('WF2', 0.15, 0.65, 0.65, 0.80, 0.80, 0.90),
    ('WF3', 0.30, 0.70, 0.70, 0.85, 0.85, 1.00),
]

all_preds, all_actuals = [], []

for name, ts, te, vs, ve, tst, tste in wf_windows:
    n = len(df)
    tr_start, tr_end = int(ts*n), int(te*n)
    va_start, va_end = int(vs*n), int(ve*n)
    te_start, te_end = int(tst*n), int(tste*n)
    
    X_tr = X[tr_start:tr_end]
    y_tr = y_b[tr_start:tr_end]
    X_va = X[va_start:va_end]
    y_va = y_b[va_start:va_end]
    X_te = X[te_start:te_end]
    y_te_act = y_a[te_start:te_end]
    
    # 基权重（类别平衡）
    sw = np.array([wd[yi] for yi in y_tr])
    
    # 时间衰减权重：最新数据权重 = 1.0，最旧数据权重 = 0.3
    decay = np.linspace(0.3, 1.0, len(sw))
    sw = sw * decay
    
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
    proba_up = y_proba[:, 2]
    all_preds.extend(proba_up.tolist())
    all_actuals.extend(y_te_act.tolist())
    
    top10 = proba_up >= np.percentile(proba_up, 90)
    if top10.sum() > 5:
        r = y_te_act[top10]
        sp = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
        wr = (r > 0).mean()
        print(f"  {name}: 测{len(X_te)}行, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

preds, act_s = np.array(all_preds), np.array(all_actuals)
t10 = preds >= np.percentile(preds, 90)
r = act_s[t10]
wf_sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wf_wr = (r > 0).mean()
print(f"\n  Walk-Forward合成(新特征+时间衰减): 夏普={wf_sharpe:.3f}, 胜率={wf_wr:.1%}")

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

# ─── 保存 ───
final.save_model(f"{MODEL_DIR}/us_xgb_v9.json")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_xgb_v9',
    'features': all_features,
    'wf_sharpe': round(wf_sharpe, 4), 'wf_win_rate': round(wf_wr, 4),
    'final_sharpe': round(sharpe, 4), 'final_win_rate': round(wr, 4),
    'final_accuracy': round(acc, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
    'all_scores': results,
}
with open(f"{MODEL_DIR}/us_xgb_v9_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v9 完成! ({TOTAL:.0f}s)")
