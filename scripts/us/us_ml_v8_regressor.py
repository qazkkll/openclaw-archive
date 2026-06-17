#!/usr/bin/env python3
"""
美股ML v8 — XGBRegressor (回归) + Walk-Forward
直接预测明日收益，按预测值排序选Top10%
"""
import sys, json, os, time, math, warnings
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ)
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
MODEL_DIR = os.path.join(WORKSPACE, "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
T0 = time.time()

print("═══ v8 XGBRegressor ═══")

df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']
df = df.dropna(subset=['label_pct'] + feature_cols).copy()

X = df[feature_cols].values
y = df['label_pct'].values

print(f"  总行数: {len(df)}, 特征: {len(feature_cols)}")
print(f"  标签: 连续值(pct_change), 中位数={np.median(y):.2f}%")

# ─── Walk-Forward ───
print("\n[1/3] Walk-Forward (3窗口)...")

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
    y_tr = y[int(ts*n):int(te*n)]
    X_va = X[int(vs*n):int(ve*n)]
    y_va = y[int(vs*n):int(ve*n)]
    X_te = X[int(tst*n):int(tste*n)]
    y_te = y[int(tst*n):int(tste*n)]
    
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mae', early_stopping_rounds=20,
        random_state=42, n_jobs=-1, verbosity=0, device='cuda')
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              verbose=0)
    
    y_pred = model.predict(X_te)
    all_preds.extend(y_pred.tolist())
    all_actuals.extend(y_te.tolist())
    
    # Top10%按预测值排序
    top10 = y_pred >= np.percentile(y_pred, 90)
    if top10.sum() > 5:
        r = y_te[top10]
        sp = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
        wr = (r > 0).mean()
        mae = np.abs(y_pred - y_te).mean()
        print(f"  {name}: 测{len(X_te)}行, MAE={mae:.2f}%, 夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)

# 合成
preds = np.array(all_preds)
actuals = np.array(all_actuals)
top10_mask = preds >= np.percentile(preds, 90)
top_ret = actuals[top10_mask]
wf_sharpe = top_ret.mean() / top_ret.std() * math.sqrt(252) if top_ret.std() > 0 else 0
wf_wr = (top_ret > 0).mean()
print(f"\n  Walk-Forward合成: 夏普={wf_sharpe:.3f}, 胜率={wf_wr:.1%}")

# ─── 全量模型 ───
print("\n[2/3] 训练全量模型...")
train_end = int(len(df) * 0.85)
final = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mae', early_stopping_rounds=20,
    random_state=42, n_jobs=-1, verbosity=0, device='cuda')
final.fit(X[:train_end], y[:train_end],
          eval_set=[(X[train_end:], y[train_end:])],
          verbose=0)

y_pred_test = final.predict(X[train_end:])
test_actual = y[train_end:]
top10 = y_pred_test >= np.percentile(y_pred_test, 90)
r = test_actual[top10]
sharpe = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
wr = (r > 0).mean()
mae = np.abs(y_pred_test - test_actual).mean()
print(f"  测试: MAE={mae:.2f}%, 夏普={sharpe:.3f}, 胜率={wr:.1%}")

# ─── 今日预测 ───
print("\n[3/3] 今日预测...")
latest = df.dropna(subset=feature_cols).drop_duplicates(subset='sym', keep='last')
X_latest = latest[feature_cols].values
y_pred_latest = final.predict(X_latest)

results = []
for i, (_, row) in enumerate(latest.iterrows()):
    results.append({
        'sym': row['sym'],
        'price': float(row['price']),
        'predicted_ret': float(y_pred_latest[i]),
        'predicted_ret_pct': round(y_pred_latest[i] * 100, 2),
    })
results.sort(key=lambda x: -x['predicted_ret'])

print(f"\n{'═'*65}")
print(f"{'排名':>3} {'代码':>8} {'价格':>8} {'预测收益':>8} {'预期%':>7}")
print(f"{'─'*65}")
# 截取预测收益>0.5%的
shown = [r for r in results if r['predicted_ret']*100 > 0.5]
if len(shown) > 30:
    shown = shown[:30]
for i, r in enumerate(shown):
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['predicted_ret_pct']:>7.2f}% {'→':>5}")

if len(results) > len(shown):
    print(f"  ... (还有{len(results)-len(shown)}只低于0.5%)")

# ─── 保存 ───
final.save_model(f"{MODEL_DIR}/us_xgb_v8_reg.json")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_xgb_v8_regressor',
    'wf_sharpe': round(wf_sharpe, 4), 'wf_win_rate': round(wf_wr, 4),
    'final_sharpe': round(sharpe, 4), 'final_win_rate': round(wr, 4),
    'final_mae': round(mae, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
}
with open(f"{MODEL_DIR}/us_xgb_v8_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v8 完成! ({TOTAL:.0f}s)")
