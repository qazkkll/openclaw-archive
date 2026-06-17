"""
绿箭v19 每日预测专用（仅推理，不训练）
读取预训练模型 + 最新特征 → 输出校准后Top30预测
"""
import sys, os, json, time, pickle, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("=" * 60)
print("绿箭v19 每日预测（推理模式）")
print("=" * 60)

# === 1. 加载特征数据 ===
print("加载特征数据...")
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
df = df.dropna(subset=feats).copy()
df = df.sort_values(['date','sym']).reset_index(drop=True)
dates = sorted(df['date'].unique())
print(f"数据: {len(df):,}行, {df['sym'].nunique()}只, {len(dates)}天")

# === 2. 加载预训练模型 ===
model_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_final.json")
calib_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_calib.pkl")
print(f"加载模型: {model_path}")
# Load booster directly to avoid sklearn wrapper issues with loaded models
booster = xgb.Booster()
booster.load_model(model_path)
print(f"  模型加载成功 (特征数: {len(feats)})")
calibrator = pickle.load(open(calib_path, 'rb'))
print(f"  校准系数: a={calibrator.coef_[0][0]:.4f}, b={calibrator.intercept_[0]:.4f}")

# === 3. 预测最新日期 ===
last_date = dates[-1]
day_df = df[df['date'] == last_date].copy()
print(f"最新日期: {last_date}, 候选股票: {len(day_df)}只")
# Predict using booster directly
dmat = xgb.DMatrix(day_df[feats].values, feature_names=feats)
raw_preds_all = booster.predict(dmat)  # shape (n, 5), softmax outputs for 5 classes
raw_preds = raw_preds_all[:, 4].reshape(-1, 1)  # class 4 = up>5%
calib_preds = calibrator.predict_proba(raw_preds)[:, 1]
day_df['up5_calib'] = calib_preds
day_df['up5_raw'] = raw_preds.flatten()

# === 4. 输出Top30 ===
print(f"\n{'='*70}")
print(f"绿箭v19 选股 — 截至 {last_date}")
print(f"{'='*70}")

configs = [
    (5, 10, 'd5_T10 (每5天调仓, Top10)'),
    (10, 15, 'd10_T15 ★甜点 (每10天调仓, Top15)'),
    (10, 20, 'd10_T20 (每10天调仓, Top20)'),
]

summary = {}
for interval_days, top_n, label in configs:
    picks = day_df.nlargest(top_n, 'up5_calib')
    avg_prob = picks['up5_calib'].mean()
    avg_raw = picks['up5_raw'].mean()
    
    print(f"\n--- {label} ---")
    print(f"  Calib_prob={avg_prob:.1%}  Raw_prob={avg_raw:.1%}")
    print(f"  {'#':>2} {'代码':>6} {'价格':>8} {'Calib':>8} {'Raw':>8}")
    print(f"  {'-'*38}")
    for i, (_, r) in enumerate(picks.iterrows()):
        print(f"  {i+1:>2} {r['sym']:>6} ${r['price']:>6.2f} {r['up5_calib']:>7.1%} {r['up5_raw']:>7.1%}")
    
    summary[label] = {
        'avg_calib_prob': round(float(avg_prob), 4),
        'avg_raw_prob': round(float(avg_raw), 4),
        'count': top_n
    }

# === 5. 保存预测结果（Top30） ===
pred_path = os.path.join(_paths.US_MODEL_DIR, "greenshaft_v19_prediction.json")
top30 = day_df.nlargest(30, 'up5_calib')
predictions = []
for i, (_, r) in enumerate(top30.iterrows()):
    predictions.append({
        'rank': i+1,
        'sym': r['sym'],
        'price': round(float(r['price']), 2),
        'up5_calib': round(float(r['up5_calib']), 4),
        'up5_raw': round(float(r['up5_raw']), 4),
        'sector': r.get('sector', ''),
    })

pred_out = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'model': 'greenshaft_v19_final',
    'calibrated': True,
    'data_date': last_date,
    'data_source': 'us_ml_feats_v3_dated.parquet',
    'summary': summary,
    'predictions': predictions,
}
with open(pred_path, 'w') as f:
    json.dump(pred_out, f, ensure_ascii=False, indent=2)
print(f"\n✅ 预测已保存: {pred_path} (Top30)")

print(f"\n总耗时: {time.time() - T0:.0f}s")
print("=" * 60)
print("完成")
print("=" * 60)
