"""
绿箭v19 — 最终版（含Platt校准）
1. 训练原始XGBoost模型
2. 用Platt缩放校准概率
3. 保存校准模型+最终预测
4. 用于每天的"今日美股推荐"
"""
import sys, os, json, math, time, gc, pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 60)
print("绿箭v19 最终版（含Platt校准）")
print("=" * 60)

# === 1. 数据 ===
print("加载数据...")
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()

with open(_paths.ML_DIR + "/us_sector_etf.json") as f:
    etf_data = json.load(f)
s2e = {
    'Technology':'XLK','Financial Services':'XLF','Financial':'XLF',
    'Energy':'XLE','Healthcare':'XLV','Industrials':'XLI',
    'Consumer Defensive':'XLP','Consumer Cyclical':'XLY','Utilities':'XLU',
    'Basic Materials':'XLB','Materials':'XLB','Real Estate':'XLRE',
    'Communication Services':'XLC','Semiconductor':'SMH'
}
def get_er(s):
    e = s2e.get(s)
    return etf_data[e]['ret5'] if e and e in etf_data else etf_data['SPY']['ret5']
df['sector_etf_ret5'] = df['sector'].apply(get_er)
for k in ['SPY','QQQ','IWM']:
    df[f'{k.lower()}_ret5'] = etf_data[k]['ret5']
df['sc'] = df['sector'].astype('category').cat.codes.astype(int)

feats = [
    'price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
    'ret1','ret5','ret20','ret60','macd','macd_signal','macd_hist',
    'vol_ratio','ma_bias20','vol5','trend_accel',
    'short_ratio','short_pct','market_cap','sector_etf_ret5',
    'spy_ret5','qqq_ret5','iwm_ret5','sc'
]

df = df.dropna(subset=feats + ['label_5d_pct', 'label_5d_5class']).copy()
df = df.sort_values(['date','sym']).reset_index(drop=True)
dates = sorted(df['date'].unique())
print(f"数据: {len(df):,}行, {df['sym'].nunique()}只, {len(dates)}天")

# 70/30分割训练校准
split_idx = int(len(dates) * 0.7)
train_dates = dates[:split_idx]
calib_dates = dates[split_idx:]  # 后30%用于校准

train = df[df['date'].isin(train_dates)]
calib = df[df['date'].isin(calib_dates)]
print(f"训练集: {len(train):,}行")
print(f"校准集: {len(calib):,}行")

# === 2. 训练原始模型 ===
print("\n训练XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
model.fit(train[feats].values, train['label_5d_5class'].values)
print("  训练完成")

# === 3. Platt校准 ===
print("Platt校准...")
# 用校准集的原始概率训练Logistic回归
raw_probs_calib = model.predict_proba(calib[feats].values)[:, 4].reshape(-1, 1)
calib_binary = (calib['label_5d_pct'] > 5).astype(int).values  # 涨>5%为1

calibrator = LogisticRegression(C=1.0, solver='lbfgs')
calibrator.fit(raw_probs_calib, calib_binary)
print(f"  校准系数: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}")

# 用全部数据重新训练最终模型
print("\n训练最终模型（全部数据）...")
model_full = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
model_full.fit(df[feats].values, df['label_5d_5class'].values)
print("  完成")

# === 4. 校准验证 ===
print("\n校准验证:")
all_raw = model_full.predict_proba(df[feats].values)[:, 4].reshape(-1, 1)
all_calib = calibrator.predict_proba(all_raw)[:, 1]
all_actuals = df['label_5d_pct'].values

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"{'概率区间':>8} {'样本数':>8} {'预测中位':>10} {'涨>5%':>9} {'偏差':>7}")
print("-" * 44)
total_abs_bias = 0
n_bins = 0
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (all_calib * 100 >= lo) & (all_calib * 100 < hi)
    cnt = mask.sum()
    if cnt < 10:
        continue
    pred_avg = float(np.mean(all_calib[mask])) * 100
    actual_up5 = float((all_actuals[mask] > 5).mean()) * 100
    bias = pred_avg - actual_up5
    total_abs_bias += abs(bias)
    n_bins += 1
    print(f"  {lo:>3}-{hi:<3}%  {cnt:>8}  {pred_avg:>7.1f}%  {actual_up5:>7.1f}%  {bias:>+6.1f}%")
avg_bias = total_abs_bias / n_bins if n_bins else 0
print(f"\n  平均绝对偏差: {avg_bias:.1f}%  {'✅ 通过' if avg_bias < 5 else '⚠️ 偏大'}")

# === 5. 预测最后一天 ===
print("\n预测选股...")
last_date = dates[-1]
day_df = df[df['date'] == last_date].copy()
raw_preds = model_full.predict_proba(day_df[feats].values)[:, 4].reshape(-1, 1)
calib_preds = calibrator.predict_proba(raw_preds)[:, 1]
day_df['up5_calib'] = calib_preds
day_df['up5_raw'] = raw_preds.flatten()

print(f"\n{'='*70}")
print(f"绿箭v19 选股 — 截至 {last_date}")
print(f"{'='*70}")

configs = [
    (5, 10, 'd5_T10 (每5天调仓, Top10)'),
    (10, 15, 'd10_T15 ★甜点 (每10天调仓, Top15)'),
    (10, 20, 'd10_T20 (每10天调仓, Top20)'),
]

for interval_days, top_n, label in configs:
    picks = day_df.nlargest(top_n, 'up5_calib')
    avg_prob = picks['up5_calib'].mean()
    avg_raw = picks['up5_raw'].mean()
    avg_actual = picks['label_5d_pct'].mean()
    
    print(f"\n--- {label} ---")
    print(f"  Calib_prob={avg_prob:.1%}  Raw_prob={avg_raw:.1%}  Actual_5d_ret={avg_actual:+.2f}%")
    print(f"  {'#':>2} {'代码':>6} {'价格':>8} {'Calib':>8} {'Raw':>8} {'5d收益':>8}")
    print(f"  {'-'*44}")
    for i, (_, r) in enumerate(picks.iterrows()):
        print(f"  {i+1:>2} {r['sym']:>6} ${r['price']:>6.2f} {r['up5_calib']:>7.1%} {r['up5_raw']:>7.1%} {r['label_5d_pct']:>+7.2f}%")

# === 6. 保存 ===
print("\n保存模型...")
model_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_final.json")
model_full.save_model(model_path)
calib_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_calib.pkl")
with open(calib_path, 'wb') as f:
    pickle.dump(calibrator, f)

pred_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_final_prediction.json")
predictions = []
for i, (_, r) in enumerate(day_df.nlargest(50, 'up5_calib').iterrows()):
    predictions.append({
        'rank': i+1,
        'sym': r['sym'],
        'price': round(float(r['price']), 2),
        'up5_calib': round(float(r['up5_calib']), 4),
        'up5_raw': round(float(r['up5_raw']), 4),
        'actual_5d': round(float(r['label_5d_pct']), 2),
    })

pred_out = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'greenshaft_v19_final',
    'calibrated': True,
    'calib_coef_a': float(calibrator.coef_[0][0]),
    'calib_intercept_b': float(calibrator.intercept_[0]),
    'data_date': last_date,
    'calibration_bias_mean': round(avg_bias, 2),
    'configs': {
        'd5_T10': {'interval': 5, 'top_n': 10, 'desc': '每5天调仓 Top10 激进型'},
        'd10_T15': {'interval': 10, 'top_n': 15, 'desc': '★甜点 每10天调仓 Top15 平衡型'},
        'd10_T20': {'interval': 10, 'top_n': 20, 'desc': '每10天调仓 Top20 稳健型'},
    },
    'predictions': predictions,
}
with open(pred_path, 'w') as f:
    json.dump(pred_out, f, ensure_ascii=False, indent=2)
print(f"  模型: {model_path}")
print(f"  校准: {calib_path}")
print(f"  预测: {pred_path}")

# 特征重要性
print("\n特征重要性:")
imps = sorted(zip(feats, model_full.feature_importances_), key=lambda x: -x[1])
for f, imp in imps:
    print(f"  {f:>25} {imp:.4f}")

print(f"\n总耗时: {time.time() - T0:.0f}s")
print("=" * 60)
print("绿箭v19 最终版就绪")
print("=" * 60)
