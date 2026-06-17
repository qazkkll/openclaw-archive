# -*- coding: utf-8 -*-
"""
蓝盾V3 — 不做方向预测，只做条件排序
把ML当作"在一堆相似的票里找出谁更好"的排序器
用pairwise ranking loss，而不是回归涨多少
"""
import warnings, json, os
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

print('加载数据...')
feat = pd.read_parquet('/home/hermes/.hermes/openclaw-project/data/us/sp500_feats.parquet')
feat = feat.sort_values(['Code','Date']).reset_index(drop=True)
feat['Date'] = pd.to_datetime(feat['Date'])

raw_dir = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'
all_rows = []
for f in sorted(os.listdir(raw_dir)):
    if not f.startswith('sp500_chunk_') or not f.endswith('.json'): continue
    raw = json.load(open(os.path.join(raw_dir, f)))
    for sym, bars in raw.items():
        for b in bars: b['Code'] = sym
        all_rows.extend(bars)
raw_df = pd.DataFrame(all_rows)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df['DollarVol'] = raw_df['C'] * raw_df['V']

feat = feat.merge(raw_df[['Code','Date','C','DollarVol']], on=['Code','Date'], how='left')
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())

# 扩展特征
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')

feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')
feat['vol_5d_norm'] = feat['vol_5d'] / (feat.groupby('Date')['vol_5d'].transform('mean') + 1e-8)
feat['rsi_50_pct'] = (feat['rsi_14'] - 50) / 50
feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)

# 增速特征（衡量速度变化）
feat['ma_cross_accel'] = feat['ma20_ma50_cross'].diff()
feat['dvol_accel'] = feat.groupby('Code')['dvol_ratio'].diff()
feat['rsi_strength'] = np.where(feat['rsi_14'] > 60, 1, np.where(feat['rsi_14'] > 40, 0, -1))
feat['vol_strength'] = np.where(feat['vol_5d_norm'] > 1.2, 1, np.where(feat['vol_5d_norm'] > 0.8, 0, -1))

# V5.5核心规则的特征化
feat['ma50_trend'] = feat['ma_50_ratio'] - 1.0  # 价格在MA50之上多少
feat['ma20_ma50_gap'] = feat['ma_20_ratio'] - feat['ma_50_ratio']

feat_cols = [
    # 动量
    'ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
    'ret_5d','ret_20d',
    # 均线
    'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
    'ma50_trend','ma20_ma50_gap','ma20_ma50_cross','ma_cross_accel',
    # 波动
    'vol_5d','vol_10d','vol_20d','vol_5d_norm','vol_ratio_5','vol_strength',
    # RSI
    'rsi_14','rsi_50_pct','rsi_strength',
    # 位置
    'price_pos_20','price_pos_50','price_pos_100',
    # MACD
    'macd','macd_sig','macd_hist',
    # ATR
    'atr_pct',
    # 相对
    'rel_ret_5d','rel_ret_10d',
    # 量
    'dvol_ratio','dvol_accel','dvol_ma5',
    # 大盘
    'market_ret',
]

# 筛选有效数据
valid = feat.dropna(subset=feat_cols + ['ret_f5']).copy()
# 只看大盘成交量足够的票
valid = valid[valid['dvol_ma5'] >= 5_000_000].copy()
print(f'有效样本(含成交量过滤): {len(valid)}')

dates = sorted(valid['Date'].unique())
split1 = int(len(dates) * 0.7)
split2 = int(len(dates) * 0.85)
train_dates = set(dates[:split1])
val_dates = set(dates[split1:split2])
test_dates = set(dates[split2:])

# 不用趋势初筛，让ML自己学是否应该买
train = valid[valid['Date'].isin(train_dates)].copy()
val = valid[valid['Date'].isin(val_dates)].copy()
test = valid[valid['Date'].isin(test_dates)].copy()

print(f'训练: {len(train)}, 验证: {len(val)}, 测试: {len(test)}')

# ========= 训练 =========
# 目标: 二分类 — 未来5天是否涨>2%（不是涨多少，是涨不涨）
# 2% threshold: 比0%更严格，减少噪音
train['target'] = (train['ret_f5'] > 0.02).astype(int)
val['target'] = (val['ret_f5'] > 0.02).astype(int)
test['target'] = (test['ret_f5'] > 0.02).astype(int)

print(f'\n涨>2%占比: 训练{train["target"].mean()*100:.1f}%, 验证{val["target"].mean()*100:.1f}%, 测试{test["target"].mean()*100:.1f}%')

# 类别平衡
pos = train[train['target']==1]
neg = train[train['target']==0]
print(f'训练中正样本{len(pos)}, 负样本{len(neg)}')

# sample_weight处理不平衡
scale_pos_weight = len(neg) / max(len(pos), 1)
print(f'scale_pos_weight: {scale_pos_weight:.1f}')

model = xgb.XGBClassifier(
    n_estimators=800, max_depth=5, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.4,
    reg_alpha=0.5, reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    early_stopping_rounds=100,
    random_state=42, n_jobs=-1
)

model.fit(
    train[feat_cols].values, train['target'].values,
    eval_set=[(train[feat_cols].values, train['target'].values),
              (val[feat_cols].values, val['target'].values)],
    verbose=200
)

# ========= 准确率分析 =========
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

for name, df, X in [('训练', train, train[feat_cols].values),
                     ('验证', val, val[feat_cols].values),
                     ('测试', test, test[feat_cols].values)]:
    pred_prob = model.predict_proba(X)[:, 1]
    pred_class = model.predict(X)
    auc = roc_auc_score(df['target'], pred_prob)
    prec = precision_score(df['target'], pred_class)
    rec = recall_score(df['target'], pred_class)
    f1 = f1_score(df['target'], pred_class)
    print(f'\n{name}: AUC={auc:.4f}, Precision={prec*100:.1f}%, Recall={rec*100:.1f}%, F1={f1:.3f}')
    
# ========= 回测 =========
print('\n=== 蓝盾V3 回测 ===')
test_df = test.copy()
test_df['prob'] = model.predict_proba(test[feat_cols].values)[:, 1]

all_trades = []
for d in sorted(test_df['Date'].unique()):
    day = test_df[test_df['Date'] == d]
    if len(day) < 10: continue
    top10 = day.nlargest(10, 'prob')
    win_rate = (top10['ret_f5'] > 0.02).mean()
    avg_ret = top10['ret_f5'].mean()
    all_trades.append({'date': str(d)[:10], 'avg_ret': avg_ret, 'win_rate': win_rate, 'top_prob': top10['prob'].mean()})

df = pd.DataFrame(all_trades)
geo = np.exp(np.log(1 + df['avg_ret']).mean()) - 1
ann = geo * 252 / 5
winrate = (df['avg_ret'] > 0).mean()
dd = (1 + df['avg_ret']).cumprod()
dd_max = (dd / dd.cummax() - 1).min()
sharpe = df['avg_ret'].mean() / max(df['avg_ret'].std(), 0.001) * np.sqrt(252/5)

print(f'')
print(f'V3: 年化(几何)={ann*100:.1f}%, 夏普={sharpe:.2f}, 回撤={dd_max*100:.1f}%')
print(f'每日Top10方向正确率: {winrate*100:.1f}%')
print(f'目标涨>2%实际命中率: {df["win_rate"].mean()*100:.1f}%')
print(f'平均ML概率: {df["top_prob"].mean()*100:.1f}%')

# ========= Top10每日信号质量 =========
print(f'\n--- Top10预测概率 vs 实际收益 ---')
df['pred_bucket'] = pd.qcut(df['top_prob'], 3, labels=['Low','Mid','High'])
print(df.groupby('pred_bucket')[['avg_ret','win_rate']].mean().to_string())

# ========= 保存 =========
model.save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v3.model')
meta = {
    'model': 'blueshield_v3',
    'strategy': 'binary_classifier_ret>2pct',
    'features': feat_cols,
    'n_features': len(feat_cols),
    'backtest': {
        'annual_return': float(ann),
        'sharpe': float(sharpe),
        'max_drawdown': float(dd_max),
        'win_rate': float(winrate)
    },
    'scale_pos_weight': float(scale_pos_weight),
    'date': '2026-06-11'
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v3_meta.json', 'w'), indent=2)
print(f'\n保存完成: blueshield_v3')
