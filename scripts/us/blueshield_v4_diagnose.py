#!/usr/bin/env python3
"""
诊断：为什么78.4%胜率只换来0.66夏普？
核心问题：模型比较的"胜率"和决策框架的"胜率"是两回事
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("=" * 70)
print("诊断：78.4%胜率 vs 0.66夏普 — 差距分析")
print("=" * 70)
print()

# 加载数据
df = pd.read_parquet('data/us/us_hist_sp500_10y.parquet')
df = df.rename(columns={'sym': 'symbol'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

# 特征工程（简化版，只保留关键特征）
def build_features(group):
    g = group.copy()
    
    # 收益率
    for n in [1, 3, 5, 10, 20, 60]:
        g[f'ret_{n}d'] = g['close'].pct_change(n)
    
    # 动量
    for n in [5, 20, 60]:
        g[f'mom_{n}'] = g['close'] / g['close'].shift(n) - 1
    
    # 波动率
    for n in [5, 20, 60]:
        g[f'vol_{n}d'] = g['close'].pct_change().rolling(n).std() * np.sqrt(252)
    
    # RSI
    delta = g['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    g['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = g['close'].ewm(span=12).mean()
    ema26 = g['close'].ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    
    # 布林带
    sma20 = g['close'].rolling(20).mean()
    std20 = g['close'].rolling(20).std()
    g['bb_width'] = (sma20 + 2*std20 - (sma20 - 2*std20)) / sma20
    g['bb_pos'] = (g['close'] - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
    
    # 成交量
    vol_sma20 = g['volume'].rolling(20).mean()
    g['vol_ratio'] = g['volume'] / vol_sma20.replace(0, np.nan)
    
    # 价格位置
    g['high_low_range'] = (g['high'] - g['low']) / g['close']
    g['bias_20'] = (g['close'] - sma20) / sma20
    
    return g

print("[1/4] 构建特征...")
groups = []
for sym, grp in df.groupby('symbol'):
    groups.append(build_features(grp))
df = pd.concat(groups, ignore_index=True)

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

# 训练模型
print()
print("[2/4] 训练NeuralNet...")
df = df.sort_values('date').reset_index(drop=True)
train_end = pd.Timestamp('2021-12-31')
test_mask = df['date'] > pd.Timestamp('2023-12-31')

X_train = df.loc[df['date'] <= train_end, feature_cols].values
y_train = df.loc[df['date'] <= train_end, 'target_binary'].values
X_test = df.loc[test_mask, feature_cols].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

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

test_probs = mlp.predict_proba(X_test_s)[:, 1]
test_df = df.loc[test_mask, ['date', 'symbol', 'close', 'target_5d']].copy()
test_df['prob'] = test_probs

print(f"  测试集: {len(test_df):,} 行")

# ============================================================
# 诊断1：模型比较 vs 决策框架的胜率定义差异
# ============================================================
print()
print("=" * 70)
print("诊断1：胜率定义差异")
print("=" * 70)
print()

# 模型比较的胜率：预测买入的股票，5天后涨>3%的比例
threshold = 0.85
signals = test_df[test_df['prob'] >= threshold]
win_rate_model = (signals['target_5d'] > 0.03).mean()
print(f"模型比较胜率（概率>{threshold}的样本中，5天后涨>3%的比例）:")
print(f"  信号数: {len(signals):,}")
print(f"  胜率: {win_rate_model:.1%}")
print(f"  平均涨幅: {signals['target_5d'].mean():.2%}")
print(f"  中位涨幅: {signals['target_5d'].median():.2%}")
print()

# 决策框架的胜率：组合每日收益>0的比例
# 模拟一个简单组合：等权持有所有信号股票，持有5天
print("决策框架胜率（组合每日收益>0的比例）:")
print("  这个胜率衡量的是整个组合的稳定性，不是单只股票的准确性")
print()

# ============================================================
# 诊断2：单笔交易 vs 组合表现
# ============================================================
print("=" * 70)
print("诊断2：单笔交易 vs 组合表现")
print("=" * 70)
print()

# 按日期分组，计算每日信号数
daily_signals = test_df[test_df['prob'] >= threshold].groupby('date').agg({
    'symbol': 'count',
    'target_5d': ['mean', 'median', 'min', 'max']
}).reset_index()
daily_signals.columns = ['date', 'n_signals', 'avg_ret', 'median_ret', 'min_ret', 'max_ret']

print(f"有信号的天数: {len(daily_signals)}")
print(f"平均每天信号数: {daily_signals['n_signals'].mean():.1f}")
print(f"信号数范围: {daily_signals['n_signals'].min()} - {daily_signals['n_signals'].max()}")
print()

# 关键：即使78%的股票涨了，如果跌的那几只跌很多，组合整体可能亏
print("关键分析：信号股票的收益分布")
print(f"  涨>3%的比例: {(signals['target_5d'] > 0.03).mean():.1%} (模型比较胜率)")
print(f"  涨>0%的比例: {(signals['target_5d'] > 0).mean():.1%} (简单胜率)")
print(f"  跌>5%的比例: {(signals['target_5d'] < -0.05).mean():.1%} (大亏比例)")
print(f"  跌>10%的比例: {(signals['target_5d'] < -0.10).mean():.1%} (巨亏比例)")
print()

# ============================================================
# 诊断3：组合层面的收益分布
# ============================================================
print("=" * 70)
print("诊断3：组合层面的收益分布")
print("=" * 70)
print()

# 模拟每日等权组合
daily_portfolio_returns = []
for date in daily_signals['date']:
    day_stocks = test_df[(test_df['date'] == date) & (test_df['prob'] >= threshold)]
    if len(day_stocks) > 0:
        # 等权组合的5天收益
        portfolio_ret = day_stocks['target_5d'].mean()
        daily_portfolio_returns.append({
            'date': date,
            'n_stocks': len(day_stocks),
            'portfolio_ret': portfolio_ret,
            'best_stock': day_stocks['target_5d'].max(),
            'worst_stock': day_stocks['target_5d'].min()
        })

port_df = pd.DataFrame(daily_portfolio_returns)

print(f"组合每日收益统计:")
print(f"  平均收益: {port_df['portfolio_ret'].mean():.2%}")
print(f"  中位收益: {port_df['portfolio_ret'].median():.2%}")
print(f"  收益>0%的比例: {(port_df['portfolio_ret'] > 0).mean():.1%} (组合胜率)")
print(f"  收益>3%的比例: {(port_df['portfolio_ret'] > 0.03).mean():.1%}")
print(f"  收益<-5%的比例: {(port_df['portfolio_ret'] < -0.05).mean():.1%}")
print()

print("对比:")
print(f"  模型比较胜率: {win_rate_model:.1%} (单只股票涨>3%)")
print(f"  组合胜率: {(port_df['portfolio_ret'] > 0).mean():.1%} (组合涨>0%)")
print(f"  差距: {win_rate_model - (port_df['portfolio_ret'] > 0).mean():.1%}")
print()

# ============================================================
# 诊断4：Sharpe的计算
# ============================================================
print("=" * 70)
print("诊断4：Sharpe为什么低？")
print("=" * 70)
print()

# 计算组合收益的波动率
daily_rets = port_df['portfolio_ret']
sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

print(f"组合收益统计:")
print(f"  平均收益: {daily_rets.mean():.4%} (每天)")
print(f"  收益标准差: {daily_rets.std():.4%} (每天)")
print(f"  夏普比率: {sharpe:.2f}")
print()

print("夏普低的原因:")
print(f"  1. 收益波动大: 标准差={daily_rets.std():.2%}, 是平均收益的 {daily_rets.std()/abs(daily_rets.mean()):.1f}倍")
print(f"  2. 负收益日: {(daily_rets < 0).mean():.1%} 的日子亏钱")
print(f"  3. 大亏日: {(daily_rets < -0.05).mean():.1%} 的日子亏>5%")
print()

# 关键发现
print("=" * 70)
print("关键发现")
print("=" * 70)
print()
print("1. 78.4%是单只股票的胜率，不是组合的胜率")
print("2. 组合胜率只有 ~50%（因为多只股票同时持有）")
print("3. 即使78%的股票涨了，如果22%的股票跌很多，组合整体可能亏")
print("4. Sharpe关注的是收益的稳定性，不是单次交易的胜率")
print()
print("解决方案:")
print("  1. 仓位加权（概率越高仓位越大）")
print("  2. 限制最大亏损（止损）")
print("  3. 减少同时持有的股票数（集中持仓）")
print("  4. 市场过滤（大盘下跌时不入场）")
