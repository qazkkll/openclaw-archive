#!/usr/bin/env python3
"""
美股ML v4 — Walk-Forward验证 + 回退v2特征基线
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

print("═══ v4 Walk-Forward验证 ═══")

# 加载
print("\n[1/5] 加载数据...")
df = pd.read_parquet(f"{ML_DIR}/us_ml_feats.parquet")
# 用回v2原始特征（去掉交互）
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']
print(f"  特征: {len(feature_cols)}列")
df = df.dropna(subset=['label_pct'] + feature_cols).copy()
print(f"  有效行: {len(df)}")

# 按sym排序，每个sym内部按时间（最早→最晚）排
# 由于us_ml_feats.parquet是按sym顺序填充的（每只股票最后750天），
# 实际上第一个sym出现最早数据，最后一个sym最新数据
# 我们需要用索引做时间分割
# 直接按行号分割：前60%训练，中25%验证，后15%测试
train_end = int(len(df) * 0.60)
val_end = int(len(df) * 0.85)

X = df[feature_cols].values
y_bucket = df['label_bucket'].values
y_actual = df['label_pct'].values

X_train = X[:train_end]
y_train_bucket = y_bucket[:train_end]
X_val = X[train_end:val_end]
y_val_bucket = y_bucket[train_end:val_end]
y_val_actual = y_actual[train_end:val_end]
X_test = X[val_end:]
y_test_bucket = y_bucket[val_end:]
y_test_actual = y_actual[val_end:]

print(f"\n  时间分割:")
print(f"    训练: {len(X_train)}行 (~{len(X_train)/len(X)*100:.0f}%)")
print(f"    验证: {len(X_val)}行 (~{len(X_val)/len(X)*100:.0f}%)")
print(f"    测试: {len(X_test)}行 (~{len(X_test)/len(X)*100:.0f}%)")

# 类别权重
classes = np.array([0, 1, 2, 3])
weights = compute_class_weight('balanced', classes=classes, y=y_train_bucket)
weight_dict = {i: w for i, w in enumerate(weights)}
sample_weight = np.array([weight_dict[yi] for yi in y_train_bucket])

# ─── 2. Walk-Forward: 滚动训练 ───
print("\n[2/5] Walk-Forward滚动训练 (3窗口)...")

# 三个窗口: 
# WF1: 训练0~60%, 验证60~75%, 测75~85%
# WF2: 训练15~65%, 验证65~80%, 测80~90%  
# WF3: 训练30~70%, 验证70~85%, 测85~100%

wf_windows = [
    {'name': 'WF1', 'train_start': 0.00, 'train_end': 0.60, 'val_start': 0.60, 'val_end': 0.75, 'test_start': 0.75, 'test_end': 0.85},
    {'name': 'WF2', 'train_start': 0.15, 'train_end': 0.65, 'val_start': 0.65, 'val_end': 0.80, 'test_start': 0.80, 'test_end': 0.90},
    {'name': 'WF3', 'train_start': 0.30, 'train_end': 0.70, 'val_start': 0.70, 'val_end': 0.85, 'test_start': 0.85, 'test_end': 1.00},
]

all_wf_preds = []  # 所有WF的预测累加
all_wf_actual = []

for wf in wf_windows:
    tsi = int(len(df) * wf['train_start'])
    tei = int(len(df) * wf['train_end'])
    vsi = int(len(df) * wf['val_start'])
    vei = int(len(df) * wf['val_end'])
    tsti = int(len(df) * wf['test_start'])
    tstei = int(len(df) * wf['test_end'])
    
    X_wf_train = X[tsi:tei]
    y_wf_train = y_bucket[tsi:tei]
    X_wf_val = X[vsi:vei]
    y_wf_val = y_bucket[vsi:vei]
    X_wf_test = X[tsti:tstei]
    y_wf_actual = y_actual[tsti:tstei]
    
    sw = np.array([weight_dict[yi] for yi in y_wf_train])
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', early_stopping_rounds=15,
        random_state=42, n_jobs=-1, verbosity=0, num_class=4, device='cuda')
    model.fit(X_wf_train, y_wf_train,
              sample_weight=sw,
              eval_set=[(X_wf_val, y_wf_val)],
              verbose=0)
    
    y_proba = model.predict_proba(X_wf_test)
    proba_large_up = y_proba[:, 3]
    
    all_wf_preds.extend(proba_large_up.tolist())
    all_wf_actual.extend(y_wf_actual.tolist())
    
    # 当前窗口评估
    top10_mask = proba_large_up >= np.percentile(proba_large_up, 90)
    if top10_mask.sum() > 5:
        top_ret = y_wf_actual[top10_mask]
        avg = top_ret.mean()
        std = top_ret.std()
        sp = avg / std * math.sqrt(252) if std > 0 else 0
        wr = (top_ret > 0).mean()
        print(f"  {wf['name']}: 测{len(X_wf_test)}行, Top10%夏普={sp:.3f}, 胜率={wr:.1%}", flush=True)
    else:
        print(f"  {wf['name']}: 数据不足", flush=True)

# Walk-Forward整体评估
wf_preds = np.array(all_wf_preds)
wf_actual = np.array(all_wf_actual)

wf_top10_mask = wf_preds >= np.percentile(wf_preds, 90)
wf_top_ret = wf_actual[wf_top10_mask]
wf_avg = wf_top_ret.mean()
wf_std = wf_top_ret.std()
wf_sharpe = wf_avg / wf_std * math.sqrt(252) if wf_std > 0 else 0
wf_wr = (wf_top_ret > 0).mean()

print(f"\n  Walk-Forward整体:")
print(f"    样本: {len(wf_preds)}行 (其中Top10%: {wf_top10_mask.sum()})")
print(f"    Top10%夏普: {wf_sharpe:.3f}")
print(f"    Top10%胜率: {wf_wr:.1%}")
print(f"    Top10%平均收益: {wf_avg*100:.2f}%")

# ─── 3. 训练最终模型（全量） ───
print("\n[3/5] 训练最终全量模型...")
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='mlogloss', early_stopping_rounds=15,
    random_state=42, n_jobs=-1, verbosity=0, num_class=4, device='cuda')
model.fit(X_train, y_train_bucket,
          sample_weight=sample_weight,
          eval_set=[(X_val, y_val_bucket)],
          verbose=0)

# 测试
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
acc = accuracy_score(y_test_bucket, y_pred)

proba_large_up = y_proba[:, 3]
top10_mask = proba_large_up >= np.percentile(proba_large_up, 90)
top_ret = y_test_actual[top10_mask]
avg_ret = top_ret.mean()
std_ret = top_ret.std()
sharpe = avg_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0
win_rate = (top_ret > 0).mean()

print(f"  全量模型测试集:")
print(f"    准确率: {acc:.3f}")
print(f"    Top10%夏普: {sharpe:.3f}")
print(f"    Top10%胜率: {win_rate:.1%}")

# ─── 4. 今日预测 ───
print("\n[4/5] 今日预测...")
latest = df.dropna(subset=feature_cols).drop_duplicates(subset='sym', keep='last')
X_latest = latest[feature_cols].values
y_proba_latest = model.predict_proba(X_latest)

results = []
for i, (_, row) in enumerate(latest.iterrows()):
    probs = y_proba_latest[i].tolist()
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
    print(f"{i+1:>3} {r['sym']:>8} {r['price']:>8.2f} {r['prob_large_up']*100:>6.1f}% {r['prob_small_up']*100:>6.1f}% "
          f"{r['prob_small_down']*100:>6.1f}% {r['prob_large_down']*100:>6.1f}% {r['prob_up']*100:>5.1f}% {r['expected_ret']:>6.2f}%")

# ─── 5. 保存 ───
print("\n[5/5] 保存...")
output = {
    'timestamp': now.isoformat(),
    'model': 'us_v4_xgb_walkforward',
    'features': feature_cols,
    'wf_sharpe': round(wf_sharpe, 4),
    'wf_win_rate': round(wf_wr, 4),
    'wf_avg_ret': round(float(wf_avg), 4),
    'final_sharpe': round(sharpe, 4),
    'final_win_rate': round(win_rate, 4),
    'final_accuracy': round(acc, 4),
    'predictions': [{'rank': i+1, **r} for i, r in enumerate(results[:50])],
    'all_scores': results,
}
model.save_model(f"{MODEL_DIR}/us_xgb_v4.json")
with open(f"{MODEL_DIR}/us_xgb_v4_prediction.json", 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

TOTAL = time.time() - T0
print(f"\n✅ v4 完成! ({TOTAL:.0f}s)")
