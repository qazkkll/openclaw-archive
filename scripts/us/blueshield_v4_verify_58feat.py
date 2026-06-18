#!/usr/bin/env python3
"""
重新验证：58特征NeuralNet的真实表现
对比21特征和58特征的差异
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings('ignore')

print("=" * 70)
print("重新验证：58特征 vs 21特征")
print("=" * 70)
print()

# 加载数据
df = pd.read_parquet('data/us/us_hist_sp500_10y.parquet')
df = df.rename(columns={'sym': 'symbol'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

# 完整58特征
def build_full_features(group):
    g = group.copy()
    
    # 收益率 (10)
    for n in [1, 2, 3, 5, 10, 20, 60, 120, 250, 500]:
        g[f'ret_{n}d'] = g['close'].pct_change(n)
    
    # 动量 (8)
    for n in [5, 10, 20, 60, 120, 250]:
        g[f'mom_{n}'] = g['close'] / g['close'].shift(n) - 1
    g['mom_5d_20d'] = g['mom_5'] / g['mom_20'].replace(0, np.nan)
    g['mom_20d_60d'] = g['mom_20'] / g['mom_60'].replace(0, np.nan)
    
    # 波动率 (6)
    for n in [5, 10, 20, 60, 120, 250]:
        g[f'vol_{n}d'] = g['close'].pct_change().rolling(n).std() * np.sqrt(252)
    
    # RSI (2)
    delta = g['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for n in [14, 28]:
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g[f'rsi_{n}'] = 100 - (100 / (1 + rs))
    
    # MACD (3)
    ema12 = g['close'].ewm(span=12).mean()
    ema26 = g['close'].ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    
    # 布林带 (3)
    sma20 = g['close'].rolling(20).mean()
    std20 = g['close'].rolling(20).std()
    g['bb_upper'] = sma20 + 2 * std20
    g['bb_lower'] = sma20 - 2 * std20
    g['bb_width'] = (g['bb_upper'] - g['bb_lower']) / sma20
    g['bb_pos'] = (g['close'] - g['bb_lower']) / (g['bb_upper'] - g['bb_lower']).replace(0, np.nan)
    
    # 成交量 (4)
    vol_sma20 = g['volume'].rolling(20).mean()
    vol_sma60 = g['volume'].rolling(60).mean()
    g['vol_ratio_20d'] = g['volume'] / vol_sma20.replace(0, np.nan)
    g['vol_ratio_60d'] = g['volume'] / vol_sma60.replace(0, np.nan)
    g['vol_change'] = g['volume'].pct_change(5)
    g['vol_trend'] = vol_sma20 / vol_sma60.replace(0, np.nan)
    
    # 价格形态 (6)
    g['high_low_range'] = (g['high'] - g['low']) / g['close']
    g['close_to_high'] = (g['high'] - g['close']) / g['close']
    g['close_to_low'] = (g['close'] - g['low']) / g['close']
    g['gap'] = g['open'] / g['close'].shift(1) - 1
    g['body_size'] = abs(g['close'] - g['open']) / g['close']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    
    # 均线偏离 (6)
    for n in [5, 10, 20, 50, 120, 250]:
        sma = g['close'].rolling(n).mean()
        g[f'bias_{n}'] = (g['close'] - sma) / sma
    
    # 52周 (2)
    g['high_52w'] = g['high'].rolling(250).max()
    g['low_52w'] = g['low'].rolling(250).min()
    g['dist_52w_high'] = g['close'] / g['high_52w'] - 1
    g['dist_52w_low'] = g['close'] / g['low_52w'] - 1
    
    # 截面排名 (8)
    g['rank_ret_5d'] = g['ret_5d'].rank(pct=True)
    g['rank_ret_20d'] = g['ret_20d'].rank(pct=True)
    g['rank_vol_20d'] = g['vol_20d'].rank(pct=True)
    g['rank_vol_60d'] = g['vol_60d'].rank(pct=True)
    g['rank_rsi_14'] = g['rsi_14'].rank(pct=True)
    g['rank_bias_20'] = g['bias_20'].rank(pct=True)
    g['rank_vol_ratio'] = g['vol_ratio_20d'].rank(pct=True)
    g['rank_bb_pos'] = g['bb_pos'].rank(pct=True)
    
    return g

print("[1/3] 构建58维特征...")
t0 = time.time()
groups = []
for sym, grp in df.groupby('symbol'):
    groups.append(build_full_features(grp))
df = pd.concat(groups, ignore_index=True)
print(f"  完成 ({time.time()-t0:.1f}s)")

# 目标变量
df['target_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
df['target_binary'] = (df['target_5d'] > 0.03).astype(int)

# 清理
feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target_5d', 'target_binary']]
for c in feature_cols:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ['target_5d', 'target_binary'])

print(f"  特征数: {len(feature_cols)}")
print(f"  有效数据: {len(df):,} 行")
print(f"  正类比例: {df['target_binary'].mean():.1%}")

# Walk-Forward
df = df.sort_values('date').reset_index(drop=True)
train_end = pd.Timestamp('2021-12-31')
test_mask = df['date'] > pd.Timestamp('2023-12-31')

X_train = df.loc[df['date'] <= train_end, feature_cols].values
y_train = df.loc[df['date'] <= train_end, 'target_binary'].values
X_test = df.loc[test_mask, feature_cols].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print()
print("[2/3] 训练58特征NeuralNet...")
t0 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    early_stopping=True,
    random_state=42,
    verbose=False
)
mlp.fit(X_train_s, y_train)
print(f"  训练完成 ({time.time()-t0:.1f}s), {mlp.n_iter_} iterations")

test_probs = mlp.predict_proba(X_test_s)[:, 1]
test_df = df.loc[test_mask, ['date', 'symbol', 'close', 'target_5d']].copy()
test_df['prob'] = test_probs

# 详细分析
print()
print("[3/3] 58特征详细分析")
print()

for threshold in [0.70, 0.75, 0.80, 0.85, 0.90]:
    signals = test_df[test_df['prob'] >= threshold]
    if len(signals) == 0:
        continue
    
    win_3pct = (signals['target_5d'] > 0.03).mean()
    win_0pct = (signals['target_5d'] > 0).mean()
    avg_ret = signals['target_5d'].mean()
    big_loss = (signals['target_5d'] < -0.05).mean()
    huge_loss = (signals['target_5d'] < -0.10).mean()
    
    print(f"阈值 {threshold}:")
    print(f"  信号数: {len(signals):,}")
    print(f"  胜率(涨>3%): {win_3pct:.1%}")
    print(f"  简单胜率(涨>0%): {win_0pct:.1%}")
    print(f"  平均涨幅: {avg_ret:.2%}")
    print(f"  大亏比例(跌>5%): {big_loss:.1%}")
    print(f"  巨亏比例(跌>10%): {huge_loss:.1%}")
    print(f"  期望值: {win_3pct * avg_ret - (1-win_3pct) * abs(signals[signals['target_5d']<0]['target_5d'].mean()):.4f}")
    print()

# 按日期分组看组合表现
print("=" * 70)
print("组合层面表现（85%阈值）")
print("=" * 70)
print()

threshold = 0.85
signals = test_df[test_df['prob'] >= threshold]
daily_groups = signals.groupby('date').agg({
    'target_5d': ['mean', 'median', 'min', 'max', 'count']
}).reset_index()
daily_groups.columns = ['date', 'mean_ret', 'median_ret', 'min_ret', 'max_ret', 'n_stocks']

print(f"有信号的天数: {len(daily_groups)}")
print(f"平均每天信号数: {daily_groups['n_stocks'].mean():.1f}")
print()

# 组合收益
port_rets = daily_groups['mean_ret']
sharpe = port_rets.mean() / port_rets.std() * np.sqrt(252) if port_rets.std() > 0 else 0
print(f"组合每日平均收益: {port_rets.mean():.4%}")
print(f"组合每日收益标准差: {port_rets.std():.4%}")
print(f"组合夏普: {sharpe:.2f}")
print(f"组合胜率(涨>0%): {(port_rets > 0).mean():.1%}")
print()

# 关键：看收益分布
print("收益分布:")
for pct in [0, 5, 10, 20, 50, 90, 95, 99]:
    print(f"  {pct}%分位: {port_rets.quantile(pct/100):.2%}")

# 对比：如果只买最好的1只
print()
print("对比：每天只买最好的1只")
top1_rets = daily_groups['max_ret']
sharpe_top1 = top1_rets.mean() / top1_rets.std() * np.sqrt(252) if top1_rets.std() > 0 else 0
print(f"  平均收益: {top1_rets.mean():.4%}")
print(f"  标准差: {top1_rets.std():.4%}")
print(f"  夏普: {sharpe_top1:.2f}")
print(f"  胜率: {(top1_rets > 0).mean():.1%}")

# 对比：每天买最差的1只（反向指标）
print()
print("对比：每天买最差的1只（反向指标）")
bottom1_rets = daily_groups['min_ret']
sharpe_bottom1 = bottom1_rets.mean() / bottom1_rets.std() * np.sqrt(252) if bottom1_rets.std() > 0 else 0
print(f"  平均收益: {bottom1_rets.mean():.4%}")
print(f"  标准差: {bottom1_rets.std():.4%}")
print(f"  夏普: {sharpe_bottom1:.2f}")
print(f"  胜率: {(bottom1_rets > 0).mean():.1%}")
