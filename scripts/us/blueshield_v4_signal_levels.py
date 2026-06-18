#!/usr/bin/env python3
"""
蓝盾V4-Classifier 信号分级报告
70%出信号，80%+重点标注
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据 & 特征
# ============================================================
DATA = "/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet"
df = pd.read_parquet(DATA)
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'sym': 'code'})
df = df.sort_values(['code', 'date']).reset_index(drop=True)

sp500_tickers = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'IVV', 'XLK', 'XLF',
                 'XLV', 'XLE', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC', 'XLY'}
df = df[~df['code'].isin(sp500_tickers)].copy()

# 特征（复用全部58个）
features = {}

for w in [5, 10, 20, 60]:
    features[f'ret_{w}d'] = df.groupby('code')['close'].pct_change(w)
    features[f'vol_{w}d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())

for w in [14, 28]:
    def calc_rsi(x, window=w):
        delta = x.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)
    features[f'rsi_{w}'] = df.groupby('code')['close'].transform(calc_rsi)

ema12 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema26 = df.groupby('code')['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
macd_line = ema12 - ema26
signal_line = macd_line.groupby(df['code']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
features['macd'] = macd_line
features['macd_signal'] = signal_line
features['macd_hist'] = macd_line - signal_line

sma20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20).mean())
std20 = df.groupby('code')['close'].transform(lambda x: x.rolling(20).std())
bb_upper = sma20 + 2 * std20
bb_lower = sma20 - 2 * std20
features['bb_width'] = (bb_upper - bb_lower) / sma20
features['bb_pct'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)

for w in [5, 10, 20, 60]:
    ma = df.groupby('code')['close'].transform(lambda x: x.rolling(w).mean())
    features[f'bias_{w}d'] = (df['close'] - ma) / ma

for w in [5, 20]:
    features[f'vol_ratio_{w}d'] = df.groupby('code')['volume'].transform(
        lambda x: x / x.rolling(w).mean()
    )

features['high_low_range'] = (df['high'] - df['low']) / df['close']
features['close_open_range'] = (df['close'] - df['open']) / df['open']

for w in [5, 10, 20]:
    features[f'momentum_{w}d'] = df.groupby('code')['close'].pct_change(w)

for w in [10, 20]:
    features[f'trend_strength_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: x.rolling(w).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == w else np.nan, raw=True)
    )

for w in [20, 60]:
    features[f'price_position_{w}d'] = df.groupby('code')['close'].transform(
        lambda x: (x - x.rolling(w).min()) / (x.rolling(w).max() - x.rolling(w).min() + 1e-10)
    )

df['ret_5d'] = df.groupby('code')['close'].pct_change(5)
df['ret_20d'] = df.groupby('code')['close'].pct_change(20)
df['vol_ratio'] = df.groupby('code')['volume'].transform(lambda x: x / x.rolling(20).mean())
df['bias_20d'] = df.groupby('code')['close'].transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).mean())

for col in ['ret_5d', 'ret_20d', 'vol_ratio', 'bias_20d']:
    features[f'rank_{col}'] = df.groupby('date')[col].rank(pct=True)

df['daily_ret'] = df.groupby('code')['close'].pct_change()
for w in [5, 20]:
    df[f'ret_{w}d_raw'] = df.groupby('code')['close'].pct_change(w)
    market_avg = df.groupby('date')[f'ret_{w}d_raw'].transform('mean')
    market_std = df.groupby('date')[f'ret_{w}d_raw'].transform('std')
    features[f'zscore_ret_{w}d'] = (df[f'ret_{w}d_raw'] - market_avg) / (market_std + 1e-10)

for w in [5, 20, 60]:
    features[f'volatility_{w}d'] = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())

vol20 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(20).std())
vol60 = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(60).std())
features['vol_change'] = vol20 / (vol60 + 1e-10)

df['vol_20d'] = vol20
features['rank_volatility'] = df.groupby('date')['vol_20d'].rank(pct=True)

price_up = df.groupby('code')['close'].pct_change(5) > 0
vol_down = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) < df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['price_vol_diverge'] = (price_up & vol_down).astype(float)

price_down = df.groupby('code')['close'].pct_change(5) < 0
vol_up = df.groupby('code')['volume'].transform(lambda x: x.rolling(5).mean()) > df.groupby('code')['volume'].transform(lambda x: x.rolling(20).mean())
features['panic_signal'] = (price_down & vol_up).astype(float)

obv = df.groupby('code').apply(lambda x: (np.sign(x['close'].diff()) * x['volume']).cumsum()).reset_index(level=0, drop=True)
features['obv_slope'] = obv.groupby(df['code']).transform(lambda x: x.rolling(20).apply(
    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 20 else np.nan, raw=True
))

features['price_vol_corr'] = df.groupby('code').apply(
    lambda x: x['close'].pct_change().rolling(20).corr(x['volume'].pct_change())
).reset_index(level=0, drop=True)

feat_df = pd.DataFrame(features, index=df.index)
feat_df['close'] = df['close']
feat_df['volume'] = df['volume']
feat_df['code'] = df['code']
feat_df['date'] = df['date']

exclude = {'close', 'volume', 'code', 'date'}
all_feature_cols = [c for c in feat_df.columns if c not in exclude]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 2. 训练模型（蓝盾V4-Classifier配置）
# ============================================================
WINDOW = 5
POS_THRESH = 0.03

feat_df['fwd_ret'] = feat_df.groupby('code')['close'].pct_change(WINDOW).shift(-WINDOW)
valid = feat_df.dropna(subset=all_feature_cols + ['fwd_ret']).copy()
valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]

valid = valid.sort_values('date')
train_end = valid['date'].quantile(0.6)
val_end = valid['date'].quantile(0.8)

train = valid[valid['date'] <= train_end].copy()
val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)].copy()
test = valid[valid['date'] > val_end].copy()

X_train = train[all_feature_cols].values
X_val = val[all_feature_cols].values
X_test = test[all_feature_cols].values

y_train = (train['fwd_ret'].values >= POS_THRESH).astype(int)
y_val = (val['fwd_ret'].values >= POS_THRESH).astype(int)

n_pos = y_train.sum()
scale = (len(y_train) - n_pos) / n_pos

# 训练获取特征重要性
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

lgb_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale,
     'verbose': -1, 'seed': 42},
    train_data, num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

top30_features = importance.head(30)['feature'].tolist()

# 用Top30训练最终模型
X_train_30 = train[top30_features].values
X_val_30 = val[top30_features].values
X_test_30 = test[top30_features].values

train_data_30 = lgb.Dataset(X_train_30, label=y_train)
val_data_30 = lgb.Dataset(X_val_30, label=y_val, reference=train_data_30)

final_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale,
     'verbose': -1, 'seed': 42},
    train_data_30, num_boost_round=1000,
    valid_sets=[val_data_30],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# ============================================================
# 3. 信号分级分析（70%出信号，80%+重点标注）
# ============================================================
print("=" * 70)
print("蓝盾V4-Classifier 信号分级分析")
print("=" * 70)

test = test.copy()
test['prob'] = final_model.predict(X_test_30)
test['actual'] = test['fwd_ret'].values

# 70%门槛出信号
signals_70 = test[test['prob'] >= 0.70].copy()
signals_70 = signals_70.sort_values('prob', ascending=False)

# 信号分级
def get_level(prob):
    if prob >= 0.80:
        return '🟢🟢 强烈推荐'
    elif prob >= 0.70:
        return '🟢 重点关注'
    else:
        return '⚪ 观察'

signals_70['level'] = signals_70['prob'].apply(get_level)

print(f"\n70%门槛总信号: {len(signals_70)} 笔")
print(f"  🟢🟢 强烈推荐(≥80%): {(signals_70['prob'] >= 0.80).sum()} 笔")
print(f"  🟢 重点关注(70-80%): {((signals_70['prob'] >= 0.70) & (signals_70['prob'] < 0.80)).sum()} 笔")

# 各级别统计
for level in ['🟢🟢 强烈推荐', '🟢 重点关注']:
    level_signals = signals_70[signals_70['level'] == level]
    if len(level_signals) == 0:
        continue
    
    actual = level_signals['actual']
    win_0 = (actual > 0).mean()
    win_3 = (actual > 0.03).mean()
    
    winners = actual[actual > 0]
    losers = actual[actual <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    pl = avg_win / avg_loss if avg_loss > 0 else float('inf')
    
    print(f"\n  {level} ({len(level_signals)}笔):")
    print(f"    胜率: {win_0:.1%}  >3%胜率: {win_3:.1%}  盈亏比: {pl:.2f}")

# ============================================================
# 4. 验证排名稳定性
# ============================================================
print(f"\n{'=' * 70}")
print("排名稳定性验证")
print(f"{'=' * 70}")

# 80%门槛的排名
signals_80 = test[test['prob'] >= 0.80].copy()
signals_80 = signals_80.sort_values('prob', ascending=False)

print(f"\n80%门槛信号 ({len(signals_80)}笔):")
for i, (_, row) in enumerate(signals_80.iterrows()):
    print(f"  #{i+1} {row['code']:<8} 概率{row['prob']:.1%} 实际{row['actual']:+.2%}")

print(f"\n70%门槛信号 ({len(signals_70)}笔):")
for i, (_, row) in enumerate(signals_70.iterrows()):
    marker = " ★" if row['prob'] >= 0.80 else ""
    print(f"  #{i+1} {row['code']:<8} 概率{row['prob']:.1%} 实际{row['actual']:+.2%}{marker}")

# 验证：80%门槛的前3名在70%门槛中是否还是前3名？
print(f"\n排名稳定性:")
top3_80 = set(signals_80.head(3)['code'].values) if len(signals_80) >= 3 else set(signals_80['code'].values)
top3_70 = set(signals_70.head(3)['code'].values) if len(signals_70) >= 3 else set(signals_70['code'].values)

if top3_80 == top3_70:
    print(f"  ✅ 80%门槛前3名 = 70%门槛前3名（排名稳定）")
else:
    print(f"  ⚠️ 80%门槛前3名 ≠ 70%门槛前3名")
    print(f"     80%门槛: {top3_80}")
    print(f"     70%门槛: {top3_70}")

# ============================================================
# 5. 保存信号列表
# ============================================================
signal_list = []
for _, row in signals_70.iterrows():
    signal_list.append({
        'code': row['code'],
        'date': str(row['date'].date()),
        'probability': round(row['prob'], 4),
        'level': row['level'],
        'actual_return': round(row['actual'], 4),
    })

with open('/home/hermes/.hermes/openclaw-archive/analysis/v4_classifier_signals.json', 'w') as f:
    json.dump(signal_list, f, indent=2, ensure_ascii=False)

print(f"\n信号列表已保存到 analysis/v4_classifier_signals.json")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
