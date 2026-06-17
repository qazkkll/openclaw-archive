"""
绿箭v19 — 正式版模型训练 + 生产选股脚本
重新训练模型（用全部可用数据），输出选股推荐

流程:
1. 加载特征数据
2. 用全部历史数据训练XGBoost（同回测参数）
3. 预测当天所有股票
4. 输出Top15/20推荐（按调仓间隔分级）
5. 缓存模型供每日使用
"""
import sys, os, json, math, time, gc
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 60)
print("绿箭v19 正式版 — 训练+选股")
print("=" * 60)

# === 1. 加载数据 ===
print("加载特征数据...")
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet")
df = df[(df['label_5d_pct'] >= -50) & (df['label_5d_pct'] <= 50)].copy()

# 补特征
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
print(f"日期: {dates[0]} ~ {dates[-1]}")

# === 2. 用全部数据训练最终模型 ===
print("\n训练最终模型（全部数据）...")
X = df[feats].values
y = df['label_5d_5class'].values

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5,
    eval_metric='mlogloss', verbosity=0, device='cuda'
)
model.fit(X, y)
print("  训练完成")

# === 3. 预测全部股票的最后一天 ===
print("\n预测选股...")
last_date = dates[-1]
day_df = df[df['date'] == last_date].copy()
print(f"  最后日期: {last_date}, {len(day_df)}只股票")

# 预测所有股票的up5概率
X_day = day_df[feats].values
probs = model.predict_proba(X_day)[:, 4]
day_df['up5_prob'] = probs

# 获取label_5d_pct（但实际上线时不可用，仅用于验证）
day_df['label_5d_pct'] = day_df['label_5d_pct'].values

# === 4. 输出多种参数 ===
intervals = [
    (5, 10, 'd5_T10'),
    (10, 15, 'd10_T15'),  # 🏆 甜点
    (10, 20, 'd10_T20'),
]

print(f"\n{'='*70}")
print(f"绿箭v19 选股推荐 — 截至 {last_date}")
print(f"{'='*70}")

for interval_days, top_n, label in intervals:
    picks = day_df.nlargest(top_n, 'up5_prob')
    avg_prob = picks['up5_prob'].mean()
    avg_actual = picks['label_5d_pct'].mean() if 'label_5d_pct' in picks else 0
    
    print(f"\n--- {label} (每{interval_days}d调仓, Top{top_n}) ---")
    print(f"  avg_prob={avg_prob:.1%}  ", end='')
    if 'label_5d_pct' in picks.columns:
        print(f"actual_5d={avg_actual:+.2f}%  ", end='')
    print(f"")
    print(f"  {'#':>2} {'代码':>8} {'名称':>20} {'价格':>8} {'涨>5%概率':>10} {'5d收益':>8}")
    print(f"  {'-'*60}")
    
    for i, (_, r) in enumerate(picks.iterrows()):
        name = r.get('sym', r.get('name', ''))
        pct_5d = r.get('label_5d_pct', 0)
        print(f"  {i+1:>2} {r['sym']:>8} {name:>20} ${r['price']:>6.2f} {r['up5_prob']:>9.1%} {pct_5d:>+7.2f}%")

# === 5. 保存模型 + 最新预测 ===
print("\n保存模型与预测...")
model_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_final.json")
model.save_model(model_path)
print(f"  模型: {model_path}")

pred_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_final_prediction.json")
predictions = []
for i, (_, r) in enumerate(day_df.nlargest(50, 'up5_prob').iterrows()):
    predictions.append({
        'rank': i+1,
        'sym': r['sym'],
        'price': round(float(r['price']), 2),
        'up5': float(r['up5_prob']),
        'actual_5d': round(float(r['label_5d_pct']), 2),
    })

pred_out = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'greenshaft_v19_final',
    'data_date': last_date,
    'features': feats,
    'predictions': predictions,
    'intervals': {
        'd5_T10': {'interval_days': 5, 'top_n': 10},
        'd10_T15': {'interval_days': 10, 'top_n': 15},
        'd10_T20': {'interval_days': 10, 'top_n': 20},
    }
}
with open(pred_path, 'w') as f:
    json.dump(pred_out, f, ensure_ascii=False, indent=2)
print(f"  预测: {pred_path}")

# === 6. 校准验证（最后一步） ===
print("\n后验校准验证...")
all_probs = model.predict_proba(df[feats].values)[:, 4]
all_actuals = df['label_5d_pct'].values

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"\n{'概率区间':>8} {'样本数':>8} {'预测中位':>10} {'涨>5%实际':>10} {'偏差':>8}")
print("-" * 46)
total_bias = 0
n_bins = 0
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (all_probs * 100 >= lo) & (all_probs * 100 < hi)
    cnt = mask.sum()
    if cnt < 10:
        continue
    pred_avg = float(np.mean(all_probs[mask])) * 100
    actual_up5 = float((all_actuals[mask] > 5).mean()) * 100
    bias = pred_avg - actual_up5
    total_bias += abs(bias)
    n_bins += 1
    print(f"  {lo:>3}-{hi:<3}%  {cnt:>8}  {pred_avg:>7.1f}%     {actual_up5:>7.1f}%     {bias:>+6.1f}%")
avg_bias = total_bias / n_bins if n_bins > 0 else 0
print(f"\n  平均绝对偏差: {avg_bias:.1f}%  ✅ {'通过' if avg_bias < 8 else '偏差过大'}")

print(f"\n总耗时: {time.time() - T0:.0f}s")
print("=" * 60)
print("绿箭v19 正式版就绪")
print("=" * 60)
