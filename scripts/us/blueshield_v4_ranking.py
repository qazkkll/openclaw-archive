#!/usr/bin/env python3
"""
蓝盾V4 回归+排名实验
监控信号价格分布，避免与绿箭重叠
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("蓝盾V4 回归+排名实验（监控价格分布）")
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
# 2. 特征计算（全部58个）
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

print(f"特征数: {len(all_feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 3. 回归模型训练
# ============================================================
WINDOW = 5

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
y_train = train['fwd_ret'].values
y_val = val['fwd_ret'].values

# 特征重要性 → Top30
train_data_temp = lgb.Dataset(X_train, label=(y_train >= 0.03).astype(int))
val_data_temp = lgb.Dataset(X_val, label=(y_val >= 0.03).astype(int), reference=train_data_temp)

temp_model = lgb.train(
    {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'scale_pos_weight': 10,
     'verbose': -1, 'seed': 42},
    train_data_temp, num_boost_round=1000,
    valid_sets=[val_data_temp],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

importance = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': temp_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

top30_features = importance.head(30)['feature'].tolist()

# 回归模型（Top30特征）
X_train_30 = train[top30_features].values
X_val_30 = val[top30_features].values
X_test_30 = test[top30_features].values

train_data = lgb.Dataset(X_train_30, label=y_train)
val_data = lgb.Dataset(X_val_30, label=y_val, reference=train_data)

reg_model = lgb.train(
    {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': 42},
    train_data, num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

print(f"回归模型最佳轮数: {reg_model.best_iteration}")

# ============================================================
# 4. 排名分析 & 价格分布监控
# ============================================================
print(f"\n{'=' * 70}")
print("排名分析 & 价格分布监控")
print(f"{'=' * 70}")

test = test.copy()
test['pred_ret'] = reg_model.predict(X_test_30)
test['actual'] = test['fwd_ret'].values

# 按预测收益排名
test = test.sort_values(['date', 'pred_ret'], ascending=[True, False])

# 测试不同持仓数量
for top_n in [5, 10, 20, 50]:
    print(f"\n--- Top {top_n} 策略 ---")
    
    # 每天选预测最高的top_n只
    daily_picks = test.groupby('date').head(top_n)
    
    # 统计
    avg_ret = daily_picks['actual'].mean()
    win_rate = (daily_picks['actual'] > 0).mean()
    
    # 价格分布
    avg_price = daily_picks['close'].mean()
    pct_below_10 = (daily_picks['close'] < 10).mean()
    pct_10_50 = ((daily_picks['close'] >= 10) & (daily_picks['close'] < 50)).mean()
    pct_50_100 = ((daily_picks['close'] >= 50) & (daily_picks['close'] < 100)).mean()
    pct_above_100 = (daily_picks['close'] >= 100).mean()
    
    # 与绿箭重叠风险
    green_arrow_zone = pct_below_10  # <$10是绿箭的地盘
    
    print(f"  平均收益: {avg_ret:.2%}  胜率: {win_rate:.1%}")
    print(f"  平均价格: ${avg_price:.2f}")
    print(f"  价格分布:")
    print(f"    <$10 (绿箭区): {pct_below_10:.1%} {'⚠️ 重叠风险!' if pct_below_10 > 0.3 else '✅'}")
    print(f"    $10-50: {pct_10_50:.1%}")
    print(f"    $50-100: {pct_50_100:.1%}")
    print(f"    >$100: {pct_above_100:.1%}")

# ============================================================
# 5. 月度表现
# ============================================================
print(f"\n{'=' * 70}")
print("月度表现（Top10策略）")
print(f"{'=' * 70}")

daily_picks_10 = test.groupby('date').head(10)
daily_picks_10['month'] = daily_picks_10['date'].dt.to_period('M')

monthly = daily_picks_10.groupby('month').agg(
    avg_ret=('actual', 'mean'),
    win_rate=('actual', lambda x: (x > 0).mean()),
    n_picks=('actual', 'count'),
    avg_price=('close', 'mean')
)

print(f"\n{'月份':>10} | {'平均收益':>8} | {'胜率':>6} | {'选股数':>6} | {'平均价格':>8}")
print("-" * 55)
for month, row in monthly.iterrows():
    marker = "🟢" if row['win_rate'] >= 0.60 else ("🟡" if row['win_rate'] >= 0.50 else "🔴")
    print(f"{str(month):>10} | {row['avg_ret']:>7.2%} | {row['win_rate']:>5.1%} | {row['n_picks']:>6.0f} | ${row['avg_price']:>7.2f} {marker}")

# ============================================================
# 6. 结论
# ============================================================
print(f"\n{'=' * 70}")
print("结论")
print(f"{'=' * 70}")

# Top10的详细统计
top10_daily = test.groupby('date').head(10)
avg_ret_top10 = top10_daily['actual'].mean()
win_rate_top10 = (top10_daily['actual'] > 0).mean()
avg_price_top10 = top10_daily['close'].mean()
pct_green_arrow = (top10_daily['close'] < 10).mean()

print(f"\nTop10策略统计:")
print(f"  日均收益: {avg_ret_top10:.2%}")
print(f"  胜率: {win_rate_top10:.1%}")
print(f"  平均价格: ${avg_price_top10:.2f}")
print(f"  绿箭重叠率: {pct_green_arrow:.1%}")

if pct_green_arrow > 0.3:
    print(f"\n⚠️ 警告: 信号过多集中在<$10股票，与绿箭重叠!")
    print(f"   建议: 过滤掉<$10股票，只选>$10的")
else:
    print(f"\n✅ 信号主要集中在>$10股票，与绿箭区分良好")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
