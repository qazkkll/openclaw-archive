#!/usr/bin/env python3
"""
us_v75_daily_score.py — V7.5 极致版每日评分
使用极致参数 T7_H10_S20_R5
输出: 候选排名 (Top30) + 持仓信号 + 风控状态 + 调仓指令
依赖: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet (最新日特征)
       /home/hermes/.hermes/openclaw-project/scripts/system/us_filtered_syms.json (过滤名单)
"""
import sys, os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, xgboost as xgb

BASE = '/home/hermes/.hermes/openclaw-archive'; ML = f'{BASE}/ml'; MD = f'{BASE}/data/models'; VER = 'us_v7_5'
print('='*70); print(f'V7.5 极致版每日评分 {time.strftime("%Y-%m-%d %H:%M")}'); print('='*70)
T0 = time.time()

# ========== 1. 加载模型 ==========
model = xgb.Booster(); model.load_model(f'{MD}/{VER}.json'); model.set_param({'device':'cuda'})
cal = pickle.load(open(f'{MD}/{VER}_calibrator.pkl', 'rb'))
report = json.load(open(f'{MD}/{VER}_report.json'))
FEATS = report['features']
print(f'模型: {VER}, {len(FEATS)}特征')

# ========== 2. 特征数据 ==========
df = pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['date_str'] = df['date'].astype(str).str[:10]

# 特征清洗
feat_cols = [f for f in FEATS if f in df.columns]
for f in feat_cols:
    df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).clip(-1e6, 1e6)
df = df.replace([np.inf, -np.inf], 0)

latest_date = sorted(df['date_str'].unique())[-1]
latest = df[df['date_str'] == latest_date].copy()
print(f'评分日: {latest_date}, {len(latest)}只')

if len(latest) < 30:
    print('今日数据不足, 跳过'); sys.exit(0)

# ========== 3. 基本面过滤 ==========
FILTER_PATH = f'{ML}/us_filtered_syms.json'
if os.path.exists(FILTER_PATH):
    flist = json.load(open(FILTER_PATH))
    valid_syms = set(flist['syms'])
    before = len(latest)
    latest = latest[latest['sym'].isin(valid_syms)].copy()
    print(f'基本面过滤: {before}->{len(latest)}只')
else:
    print('警告: 无过滤名单, 用全部1602只')

# ========== 4. 收盘价 ==========
# 从特征数据直接取最新close（不做额外下载）
# feats里有ema12等指标，但没有列名close。需从开盘价索引取
open_idx, close_idx = pickle.load(open(f'{ML}/us_v75_close_idx_v4.pkl', 'rb'))
latest['close'] = latest['sym'].map(lambda s: close_idx.get(s, {}).get(latest_date, 0)).fillna(0)
# 如果close_idx里没有今日收盘，用open_idx里的开盘价代替
latest['close'] = latest['close'].replace(0, np.nan)
latest['close'] = latest['close'].fillna(
    latest['sym'].map(lambda s: open_idx.get(s, {}).get(latest_date, 0))).fillna(0)
print(f'有收盘价: {(latest["close"]>0).sum()}只')

# ========== 5. 评分 ==========
print(f'\n评分中...', flush=True)
X = np.nan_to_num(latest[feat_cols].values.astype(np.float32), nan=0)
raw = model.predict(xgb.DMatrix(X, feature_names=feat_cols))
calib = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
latest['prob_5pct'] = calib
latest = latest.sort_values('prob_5pct', ascending=False)

# Top50统计
top50 = latest.head(50)
avg_prob = top50['prob_5pct'].mean()
print(f'Top50平均概率: {avg_prob:.3f}')

# 市场热度
if avg_prob < 0.25:
    market_temp = '冷'
elif avg_prob < 0.33:
    market_temp = '温'
else:
    market_temp = '热'
print(f'市场热度: {market_temp}')

# ========== 6. 极致参数推荐 ==========
# T7_H10_S20_R5: 推Top7中概率>0.35的
threshold_buy = 0.35
threshold_watch = 0.32

recs = latest[latest['prob_5pct'] > threshold_buy].head(7)
watch = latest[(latest['prob_5pct'] >= threshold_watch) & (latest['prob_5pct'] <= threshold_buy)].head(15)

print(f'\n━━━ 【买入候选】概率>{threshold_buy} (T7) ━━━')
print(f'{"代码":>8s} {"概率":>8s} {"收盘价":>10s} {"信号":>12s}')
print('-' * 40)
for _, r in recs.iterrows():
    print(f'{r["sym"]:>8s} {r["prob_5pct"]:>7.1%} {r.get("close", 0):>10.2f} {"← 买入":>12s}')

print(f'\n━━━ 【关注】概率{threshold_watch}-{threshold_buy} ━━━')
print(f'{"代码":>8s} {"概率":>8s}')
print('-' * 20)
for _, r in watch.iterrows():
    print(f'{r["sym"]:>8s} {r["prob_5pct"]:>7.1%}')

print(f'\n━━━ 【策略参数】━━━')
print(f'  T=7(最多持7只)  H=10(持10天)  S=-20%(止损)  R=5(每5天重平衡)')

print(f'\n━━━ 【风控】━━━')
if market_temp == '热':
    print(f'  市场过热(Top50概率>{0.33:.2f})，仓位控制在60%以下')
elif market_temp == '冷':
    print(f'  市场偏冷(Top50概率<{0.25:.2f})，持有现金为主')
else:
    print(f'  市场温和，正常执行T7_H10_S20_R5')

# ========== 7. 输出JSON ==========
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'date': latest_date,
    'market_temp': market_temp,
    'top50_avg_prob': round(avg_prob, 4),
    'strategy': {'T': 7, 'H': 10, 'S': 20, 'R': 5},
    'buy_signals': [{'sym': r['sym'], 'prob': round(float(r['prob_5pct']), 4),
                     'close': float(r.get('close', 0))}
                    for _, r in recs.iterrows()],
    'watch_signals': [{'sym': r['sym'], 'prob': round(float(r['prob_5pct']), 4)}
                      for _, r in watch.iterrows()],
    'top30_stocks': [{'sym': r['sym'], 'prob': round(float(r['prob_5pct']), 4)}
                     for _, r in latest.head(30).iterrows()],
}
os.makedirs(f'{BASE}/data', exist_ok=True)
dst = f'{BASE}/data/scored_v75_{latest_date}.json'
json.dump(output, open(dst, 'w'), indent=2, ensure_ascii=False)
print(f'\n保存: {dst}')
print(f'耗时: {time.time()-T0:.1f}s')
