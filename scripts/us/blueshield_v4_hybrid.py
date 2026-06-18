#!/usr/bin/env python3
"""
蓝盾V4 混合方案：V3公式过滤 + ML排名
V3≥80分进入候选池 → ML预测排名 → 输出结果
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
print("蓝盾V4 混合方案：V3公式过滤 + ML排名")
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
# 2. V3公式评分（简化版，基于可用特征）
# ============================================================
print("\n计算V3公式评分...")

# V3公式：110分制，6维度
# 1. 趋势（30分）：价格在均线上方 + 均线多头排列
# 2. 动量（25分）：RSI适中 + 动量正
# 3. MACD（25分）：MACD金叉 + 柱状图正
# 4. 均线偏离（10分）：价格在合理范围内
# 5. RSI（10分）：RSI不超买不超卖
# 6. 52周位置（10分）：价格在52周高位附近

def calc_v3_score(df_group):
    """计算单只股票的V3公式评分"""
    scores = pd.Series(index=df_group.index, dtype=float)
    
    for idx in df_group.index:
        row = df_group.loc[idx]
        score = 0
        
        # 1. 趋势（30分）
        # MA5 > MA20 > MA60 = 多头排列
        ma5 = df_group.loc[:idx, 'close'].rolling(5).mean().iloc[-1] if len(df_group.loc[:idx]) >= 5 else np.nan
        ma20 = df_group.loc[:idx, 'close'].rolling(20).mean().iloc[-1] if len(df_group.loc[:idx]) >= 20 else np.nan
        ma60 = df_group.loc[:idx, 'close'].rolling(60).mean().iloc[-1] if len(df_group.loc[:idx]) >= 60 else np.nan
        
        if not any(pd.isna([ma5, ma20, ma60])):
            if row['close'] > ma5:
                score += 10
            if ma5 > ma20:
                score += 10
            if ma20 > ma60:
                score += 10
        
        # 2. 动量（25分）
        ret5 = df_group.loc[:idx, 'close'].pct_change(5).iloc[-1] if len(df_group.loc[:idx]) >= 5 else np.nan
        ret20 = df_group.loc[:idx, 'close'].pct_change(20).iloc[-1] if len(df_group.loc[:idx]) >= 20 else np.nan
        
        if not pd.isna(ret5) and ret5 > 0:
            score += 12.5
        if not pd.isna(ret20) and ret20 > 0:
            score += 12.5
        
        # 3. MACD（25分）
        if len(df_group.loc[:idx]) >= 26:
            ema12 = df_group.loc[:idx, 'close'].ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = df_group.loc[:idx, 'close'].ewm(span=26, adjust=False).mean().iloc[-1]
            macd = ema12 - ema26
            
            if macd > 0:
                score += 12.5
            if len(df_group.loc[:idx]) >= 35:
                macd_series = df_group.loc[:idx, 'close'].ewm(span=12, adjust=False).mean() - df_group.loc[:idx, 'close'].ewm(span=26, adjust=False).mean()
                signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1]
                if macd > signal:
                    score += 12.5
        
        # 4. 均线偏离（10分）
        if not pd.isna(ma20):
            bias = (row['close'] - ma20) / ma20
            if -0.05 < bias < 0.10:  # 合理范围
                score += 10
        
        # 5. RSI（10分）
        if len(df_group.loc[:idx]) >= 14:
            delta = df_group.loc[:idx, 'close'].diff()
            gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
            loss = (-delta).clip(lower=0).rolling(14).mean().iloc[-1]
            if loss > 0:
                rs = gain / loss
                rsi = 100 - 100 / (1 + rs)
                if 30 < rsi < 70:  # 不超买不超卖
                    score += 10
        
        # 6. 52周位置（10分）
        if len(df_group.loc[:idx]) >= 252:
            high_52w = df_group.loc[:idx, 'high'].rolling(252).max().iloc[-1]
            low_52w = df_group.loc[:idx, 'low'].rolling(252).min().iloc[-1]
            if high_52w > low_52w:
                pos_52w = (row['close'] - low_52w) / (high_52w - low_52w)
                if pos_52w > 0.7:  # 在52周高位附近
                    score += 10
        
        scores[idx] = score
    
    return scores

# 计算V3评分（这一步很慢，因为要逐行计算）
print("计算V3评分（需要几分钟）...")
t_start = time.time()

# 为了速度，用简化版V3（基于已有特征）
def calc_v3_fast(group):
    """快速V3评分"""
    scores = pd.Series(index=group.index, dtype=float)
    
    close = group['close']
    
    # MA
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    
    # 趋势分（30分）
    trend = ((close > ma5).astype(float) * 10 + 
             (ma5 > ma20).astype(float) * 10 + 
             (ma20 > ma60).astype(float) * 10)
    
    # 动量分（25分）
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    momentum = ((ret5 > 0).astype(float) * 12.5 + 
                (ret20 > 0).astype(float) * 12.5)
    
    # MACD分（25分）
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_score = ((macd > 0).astype(float) * 12.5 + 
                  (macd > signal).astype(float) * 12.5)
    
    # 均线偏离分（10分）
    bias = (close - ma20) / ma20
    bias_score = ((bias > -0.05) & (bias < 0.10)).astype(float) * 10
    
    # RSI分（10分）
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_score = ((rsi > 30) & (rsi < 70)).astype(float) * 10
    
    # 52周位置分（10分）
    high_52w = group['high'].rolling(252).max()
    low_52w = group['low'].rolling(252).min()
    pos_52w = (close - low_52w) / (high_52w - low_52w + 1e-10)
    pos_score = (pos_52w > 0.7).astype(float) * 10
    
    scores = trend + momentum + macd_score + bias_score + rsi_score + pos_score
    
    return scores

df['v3_score'] = df.groupby('code').apply(calc_v3_fast).reset_index(level=0, drop=True)

print(f"V3评分计算完成, 耗时: {time.time()-t_start:.1f}s")
print(f"V3评分分布: {df['v3_score'].describe()}")

# ============================================================
# 3. 特征计算（ML用）
# ============================================================
print("\n计算ML特征...")
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
feat_df['v3_score'] = df['v3_score']

exclude = {'close', 'volume', 'code', 'date'}
all_feature_cols = [c for c in feat_df.columns if c not in exclude]
feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

print(f"特征数: {len(all_feature_cols)}, 耗时: {time.time()-t_start:.1f}s")

# ============================================================
# 4. 数据划分 & ML训练
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

# Top30特征
top30_features = ['vol_60d', 'volatility_60d', 'vol_20d', 'high_low_range', 'vol_5d',
                  'volatility_20d', 'rsi_28', 'rank_volatility', 'vol_change', 'ret_60d',
                  'ret_5d', 'vol_10d', 'price_position_60d', 'rank_bias_20d', 'bias_5d',
                  'bias_10d', 'zscore_ret_20d', 'bias_60d', 'ret_20d', 'vol_ratio_5d',
                  'vol_ratio_20d', 'ret_10d', 'trend_strength_10d', 'trend_strength_20d',
                  'zscore_ret_5d', 'rank_ret_20d', 'rsi_14', 'macd_hist', 'bias_20d',
                  'close_open_range']

X_train = train[top30_features].values
X_val = val[top30_features].values
X_test = test[top30_features].values
y_train = train['fwd_ret'].values
y_val = val['fwd_ret'].values

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

reg_model = lgb.train(
    {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
     'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
     'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'seed': 42},
    train_data, num_boost_round=1000,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# ============================================================
# 5. 混合方案测试
# ============================================================
print(f"\n{'=' * 70}")
print("混合方案测试")
print(f"{'=' * 70}")

test = test.copy()
test['pred_ret'] = reg_model.predict(X_test)
test['actual'] = test['fwd_ret'].values

# 测试不同V3门槛
v3_thresholds = [60, 70, 80, 90]
top_n_list = [5, 10, 20]

for v3_thresh in v3_thresholds:
    print(f"\n--- V3≥{v3_thresh} 过滤 ---")
    
    # V3过滤
    filtered = test[test['v3_score'] >= v3_thresh].copy()
    
    if len(filtered) == 0:
        print(f"  无数据")
        continue
    
    # 按ML预测排名
    filtered = filtered.sort_values(['date', 'pred_ret'], ascending=[True, False])
    
    for top_n in top_n_list:
        daily_picks = filtered.groupby('date').head(top_n)
        
        if len(daily_picks) == 0:
            continue
        
        avg_ret = daily_picks['actual'].mean()
        win_rate = (daily_picks['actual'] > 0).mean()
        avg_price = daily_picks['close'].mean()
        pct_green = (daily_picks['close'] < 10).mean()
        
        print(f"  Top{top_n}: 收益{avg_ret:+.2%} 胜率{win_rate:.1%} 均价${avg_price:.0f} 绿箭{pct_green:.1%}")

# 对比：纯ML（无V3过滤）
print(f"\n--- 纯ML（无V3过滤）---")
test_sorted = test.sort_values(['date', 'pred_ret'], ascending=[True, False])

for top_n in top_n_list:
    daily_picks = test_sorted.groupby('date').head(top_n)
    avg_ret = daily_picks['actual'].mean()
    win_rate = (daily_picks['actual'] > 0).mean()
    avg_price = daily_picks['close'].mean()
    pct_green = (daily_picks['close'] < 10).mean()
    
    print(f"  Top{top_n}: 收益{avg_ret:+.2%} 胜率{win_rate:.1%} 均价${avg_price:.0f} 绿箭{pct_green:.1%}")

# ============================================================
# 6. 最优配置 & 输出样例
# ============================================================
print(f"\n{'=' * 70}")
print("最优配置 & 输出样例")
print(f"{'=' * 70}")

# 找最优：V3≥70 + Top10
best_filtered = test[test['v3_score'] >= 70].copy()
best_filtered = best_filtered.sort_values(['date', 'pred_ret'], ascending=[True, False])
best_picks = best_filtered.groupby('date').head(10)

print(f"\nV3≥70 + ML Top10 配置:")
print(f"  平均收益: {best_picks['actual'].mean():+.2%}")
print(f"  胜率: {(best_picks['actual'] > 0).mean():.1%}")
print(f"  平均价格: ${best_picks['close'].mean():.0f}")
print(f"  绿箭重叠: {(best_picks['close'] < 10).mean():.1%}")

# 输出最近几天的信号样例
print(f"\n最近信号样例:")
recent = test[test['date'] >= test['date'].max() - pd.Timedelta(days=10)]
recent = recent[recent['v3_score'] >= 70]
recent = recent.sort_values(['date', 'pred_ret'], ascending=[True, False])
recent_picks = recent.groupby('date').head(5)

print(f"\n{'日期':>12} | {'代码':>6} | {'V3分数':>6} | {'ML预测':>8} | {'实际收益':>8} | {'价格':>8}")
print("-" * 70)

for _, row in recent_picks.iterrows():
    marker = "🟢" if row['actual'] > 0 else "🔴"
    print(f"{str(row['date'].date()):>12} | {row['code']:>6} | {row['v3_score']:>5.0f} | {row['pred_ret']:>+7.2%} | {row['actual']:>+7.2%} | ${row['close']:>7.2f} {marker}")

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)
