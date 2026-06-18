#!/usr/bin/env python3
"""
蓝盾V4-Classifier 验证分析
时间分布 + 连续亏损 + 月度收益 + 稳健性检验
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据 & 特征（复用）
# ============================================================
print("=" * 70)
print("蓝盾V4-Classifier 验证分析")
print("=" * 70)

DATA = "/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet"
df = pd.read_parquet(DATA)
df['date'] = pd.to_datetime(df['date'])
df = df.rename(columns={'sym': 'code'})
df = df.sort_values(['code', 'date']).reset_index(drop=True)

sp500_tickers = {'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'IVV', 'XLK', 'XLF',
                 'XLV', 'XLE', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC', 'XLY'}
df = df[~df['code'].isin(sp500_tickers)].copy()

print(f"数据: {len(df):,} 行, {df['code'].nunique()} 只股票")

# ============================================================
# 2. 特征计算
# ============================================================
print("\n计算特征...")
t_start = time.time()

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

# 截面特征
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
    vol = df.groupby('code')['close'].transform(lambda x: x.pct_change().rolling(w).std())
    features[f'volatility_{w}d'] = vol

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

print(f"特征数: {len(all_feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 3. 数据划分 & 模型训练
# ============================================================
WINDOW = 5
POS_THRESH = 0.03  # 蓝盾V4-Classifier配置
ENTRY_PROB = 0.80

feat_df['fwd_ret'] = feat_df.groupby('code')['close'].pct_change(WINDOW).shift(-WINDOW)
valid = feat_df.dropna(subset=all_feature_cols + ['fwd_ret']).copy()
valid = valid[valid['fwd_ret'].between(-0.5, 0.5)]

valid = valid.sort_values('date')
train_end = valid['date'].quantile(0.6)
val_end = valid['date'].quantile(0.8)

train = valid[valid['date'] <= train_end].copy()
val = valid[(valid['date'] > train_end) & (valid['date'] <= val_end)].copy()
test = valid[valid['date'] > val_end].copy()

print(f"\n训练: {len(train):,}  验证: {len(val):,}  测试: {len(test):,}")

X_train = train[all_feature_cols].values
X_val = val[all_feature_cols].values
X_test = test[all_feature_cols].values

y_train = (train['fwd_ret'].values >= POS_THRESH).astype(int)
y_val = (val['fwd_ret'].values >= POS_THRESH).astype(int)

n_pos = y_train.sum()
scale = (len(y_train) - n_pos) / n_pos

# 训练模型获取特征重要性
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

# 特征重要性 → Top30
importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

top30_features = importance.head(30)['feature'].tolist()

# 用Top30重新训练最终模型
X_train_top30 = train[top30_features].values
X_val_top30 = val[top30_features].values
X_test_top30 = test[top30_features].values

train_data_30 = lgb.Dataset(X_train_top30, label=y_train)
val_data_30 = lgb.Dataset(X_val_top30, label=y_val, reference=train_data_30)

final_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': scale,
     'verbose': -1, 'seed': 42},
    train_data_30, num_boost_round=1000,
    valid_sets=[val_data_30],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# 预测
test = test.copy()
test['prob'] = final_model.predict(X_test_top30)
test['actual'] = test['fwd_ret'].values

# 筛选信号
signals = test[test['prob'] >= ENTRY_PROB].copy()
signals = signals.sort_values('date')

print(f"\n蓝盾V4-Classifier信号: {len(signals)} 笔")
print(f"正类阈值: {POS_THRESH:.0%}, 入场阈值: {ENTRY_PROB:.0%}")
print(f"Top30特征: {top30_features[:10]}...")

# ============================================================
# 4. 总体统计
# ============================================================
print(f"\n{'=' * 70}")
print("总体统计")
print(f"{'=' * 70}")

actual = signals['actual']
win_0 = (actual > 0).mean()
win_3 = (actual > 0.03).mean()
win_target = (actual >= POS_THRESH).mean()

winners = actual[actual > 0]
losers = actual[actual <= 0]
avg_win = winners.mean() if len(winners) > 0 else 0
avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
pl_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
ev = win_0 * avg_win - (1 - win_0) * avg_loss

print(f"总信号数: {len(signals)}")
print(f"实际>0%: {win_0:.1%}  实际>3%: {win_3:.1%}  实际≥3%(达标): {win_target:.1%}")
print(f"平均赢: {avg_win:.2%}  平均亏: {avg_loss:.2%}")
print(f"盈亏比: {pl_ratio:.2f}")
print(f"期望值/笔: {ev:.4f} ({ev*100:.2f}%)")

# ============================================================
# 5. 时间分布分析
# ============================================================
print(f"\n{'=' * 70}")
print("时间分布分析")
print(f"{'=' * 70}")

signals['month'] = signals['date'].dt.to_period('M')
signals['year'] = signals['date'].dt.year
signals['weekday'] = signals['date'].dt.day_name()

# 月度信号分布
monthly = signals.groupby('month').agg(
    signals=('prob', 'count'),
    avg_prob=('prob', 'mean'),
    win_rate=('actual', lambda x: (x > 0).mean()),
    avg_ret=('actual', 'mean')
)

print(f"\n月度分布:")
print(f"{'月份':>10} | {'信号数':>6} | {'平均概率':>8} | {'胜率':>6} | {'平均收益':>8}")
print("-" * 55)
for month, row in monthly.iterrows():
    marker = "🟢" if row['win_rate'] >= 0.70 else ("🟡" if row['win_rate'] >= 0.60 else "🔴")
    print(f"{str(month):>10} | {row['signals']:>6.0f} | {row['avg_prob']:>7.1%} | {row['win_rate']:>5.1%} | {row['avg_ret']:>7.2%} {marker}")

# 年度统计
yearly = signals.groupby('year').agg(
    signals=('prob', 'count'),
    win_rate=('actual', lambda x: (x > 0).mean()),
    avg_ret=('actual', 'mean')
)

print(f"\n年度统计:")
for year, row in yearly.iterrows():
    print(f"  {year}: {row['signals']:.0f}个信号, 胜率{row['win_rate']:.1%}, 平均收益{row['avg_ret']:.2%}")

# 星期分布
weekday = signals.groupby('weekday').agg(
    signals=('prob', 'count'),
    win_rate=('actual', lambda x: (x > 0).mean())
)
print(f"\n星期分布:")
for day, row in weekday.iterrows():
    print(f"  {day}: {row['signals']:.0f}个信号, 胜率{row['win_rate']:.1%}")

# ============================================================
# 6. 连续亏损分析
# ============================================================
print(f"\n{'=' * 70}")
print("连续亏损分析")
print(f"{'=' * 70}")

signals_sorted = signals.sort_values('date')
signals_sorted['win'] = (signals_sorted['actual'] > 0).astype(int)

# 计算连续亏损
streak = 0
max_streak = 0
streaks = []
for i in range(len(signals_sorted)):
    if signals_sorted.iloc[i]['win'] == 0:
        streak += 1
    else:
        if streak > 0:
            streaks.append(streak)
        max_streak = max(max_streak, streak)
        streak = 0
if streak > 0:
    streaks.append(streak)
    max_streak = max(max_streak, streak)

print(f"最大连续亏损: {max_streak} 笔")
if streaks:
    print(f"平均连续亏损: {np.mean(streaks):.1f} 笔")
    print(f"连续亏损分布:")
    for s in sorted(set(streaks)):
        print(f"  连续{s}笔亏损: {streaks.count(s)} 次")

# 连续盈利
win_streak = 0
max_win_streak = 0
for i in range(len(signals_sorted)):
    if signals_sorted.iloc[i]['win'] == 1:
        win_streak += 1
        max_win_streak = max(max_win_streak, win_streak)
    else:
        win_streak = 0

print(f"\n最大连续盈利: {max_win_streak} 笔")

# ============================================================
# 7. 月度收益分布
# ============================================================
print(f"\n{'=' * 70}")
print("月度收益分布")
print(f"{'=' * 70}")

monthly_ret = signals.groupby('month').agg(
    total_ret=('actual', 'sum'),
    avg_ret=('actual', 'mean'),
    n_trades=('actual', 'count'),
    win_rate=('actual', lambda x: (x > 0).mean())
)

pos_months = (monthly_ret['win_rate'] >= 0.60).sum()
neg_months = (monthly_ret['win_rate'] < 0.50).sum()

print(f"\n月度胜率分布:")
print(f"  胜率≥60%的月份: {pos_months} ({pos_months/len(monthly_ret):.1%})")
print(f"  胜率<50%的月份: {neg_months} ({neg_months/len(monthly_ret):.1%})")

# 最佳和最差月份
best_month = monthly_ret['win_rate'].idxmax()
worst_month = monthly_ret['win_rate'].idxmin()
print(f"\n最佳月份: {best_month} (胜率{monthly_ret.loc[best_month, 'win_rate']:.1%}, {monthly_ret.loc[best_month, 'n_trades']:.0f}笔)")
print(f"最差月份: {worst_month} (胜率{monthly_ret.loc[worst_month, 'win_rate']:.1%}, {monthly_ret.loc[worst_month, 'n_trades']:.0f}笔)")

# ============================================================
# 8. Top30特征重要性
# ============================================================
print(f"\n{'=' * 70}")
print("Top30特征重要性")
print(f"{'=' * 70}")

for i, (_, row) in enumerate(importance.head(30).iterrows()):
    print(f"  {i+1:>2}. {row['feature']:<25} {row['importance']:>10.0f}")

# ============================================================
# 9. 保存配置
# ============================================================
config = {
    'model_name': '蓝盾V4-Classifier',
    'model_type': 'LightGBM',
    'positive_threshold': POS_THRESH,
    'entry_probability': ENTRY_PROB,
    'features': top30_features,
    'window': WINDOW,
    'metrics': {
        'win_rate_0': win_0,
        'win_rate_3': win_3,
        'win_rate_target': win_target,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': pl_ratio,
        'expected_value': ev,
        'total_signals': len(signals),
    }
}

import json
with open('/home/hermes/.hermes/openclaw-archive/models/us/v4_classifier_config.json', 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

# 保存模型
final_model.save_model('/home/hermes/.hermes/openclaw-archive/models/us/v4_classifier.model')

print(f"\n配置已保存到 models/us/v4_classifier_config.json")
print(f"模型已保存到 models/us/v4_classifier.model")

print("\n" + "=" * 70)
print("蓝盾V4-Classifier 验证完成！")
print("=" * 70)
