#!/usr/bin/env python3
"""
蓝盾 V4 决策框架设计 + 完整回测
=================================
模型：NeuralNet (MLP)
数据：S&P 500 10年日K线
目标：设计最佳决策方案，计算年化、最大回撤、平均回撤等完整指标

决策维度：
1. 入场条件：模型概率 > 阈值
2. 仓位管理：最多持几只？每只多少%？
3. 止损/止盈：固定止损 or 移动止损
4. 持仓周期：固定天数 or 动态退出
5. 调仓频率：每日 or 固定周期
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import time

warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据加载与特征工程
# ============================================================
print("=" * 70)
print("蓝盾 V4 决策框架设计")
print("=" * 70)
print()

print("[1/5] 加载数据...")
df = pd.read_parquet('data/us/us_hist_sp500_10y.parquet')
df = df.rename(columns={'sym': 'symbol'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
print(f"  原始数据: {len(df):,} 行, {df['symbol'].nunique()} 只股票")
print(f"  时间范围: {df['date'].min().date()} → {df['date'].max().date()}")

# ============================================================
# 2. 特征工程（58维）
# ============================================================
print()
print("[2/5] 构建58维特征...")

def build_features(group):
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

groups = []
for sym, grp in df.groupby('symbol'):
    groups.append(build_features(grp))
df = pd.concat(groups, ignore_index=True)

# 目标变量
df['target'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
df['target_binary'] = (df['target'] > 0.03).astype(int)

# 清理inf/NaN
feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target', 'target_binary']]
for c in feature_cols:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ['target', 'target_binary'])
print(f"  特征数: {len(feature_cols)}")
print(f"  有效数据: {len(df):,} 行")
print(f"  正类比例: {df['target_binary'].mean():.1%}")

# ============================================================
# 3. Walk-Forward 训练
# ============================================================
print()
print("[3/5] Walk-Forward 训练...")

df = df.sort_values('date').reset_index(drop=True)
train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

train_mask = df['date'] <= train_end
val_mask = (df['date'] > train_end) & (df['date'] <= val_end)
test_mask = df['date'] > val_end

X_train = df.loc[train_mask, feature_cols].values
y_train = df.loc[train_mask, 'target_binary'].values
X_val = df.loc[val_mask, feature_cols].values
X_test = df.loc[test_mask, feature_cols].values

print(f"  训练集: {X_train.shape[0]:,} 行 (→ {train_end.date()})")
print(f"  验证集: {X_val.shape[0]:,} 行")
print(f"  测试集: {X_test.shape[0]:,} 行 ({val_end.date()+pd.Timedelta(days=1)} → {df['date'].max().date()})")

# 标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# 训练NeuralNet
print()
print("  训练 NeuralNet...")
t0 = time.time()
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)
mlp.fit(X_train_s, y_train)
print(f"  训练完成 ({time.time()-t0:.1f}s), {mlp.n_iter_} iterations")

test_probs = mlp.predict_proba(X_test_s)[:, 1]
test_df = df.loc[test_mask, ['date', 'symbol', 'close', 'target']].copy()
test_df['prob'] = test_probs

print(f"  概率分布: min={test_probs.min():.3f}, median={np.median(test_probs):.3f}, max={test_probs.max():.3f}")
print(f"  >85%: {(test_probs > 0.85).sum()} 笔")
print(f"  >80%: {(test_probs > 0.80).sum()} 笔")

# ============================================================
# 4. 决策策略回测引擎
# ============================================================
print()
print("[4/5] 运行多决策策略回测...")
print()

class StrategyBacktester:
    def __init__(self, test_df):
        self.test_df = test_df.copy()
        self.test_df['date'] = pd.to_datetime(self.test_df['date'])
        
    def run(self, name, config):
        df = self.test_df.copy()
        dates = sorted(df['date'].unique())
        
        positions = {}
        cash = 1.0
        equity_curve = []
        
        threshold = config.get('threshold', 0.85)
        max_pos = config.get('max_positions', 10)
        stop_loss = config.get('stop_loss', None)
        take_profit = config.get('take_profit', None)
        hold_days = config.get('hold_days', 5)
        rebalance_freq = config.get('rebalance_freq', 1)
        
        for i, date in enumerate(dates):
            day_data = df[df['date'] == date]
            
            # 检查退出条件
            to_close = []
            for sym, pos in positions.items():
                sym_data = day_data[day_data['symbol'] == sym]
                if len(sym_data) == 0:
                    continue
                current_price = sym_data['close'].values[0]
                ret = current_price / pos['entry_price'] - 1
                days_held = i - pos['entry_idx']
                
                if stop_loss and ret <= stop_loss:
                    to_close.append(sym)
                elif take_profit and ret >= take_profit:
                    to_close.append(sym)
                elif hold_days and days_held >= hold_days:
                    to_close.append(sym)
            
            # 平仓
            for sym in to_close:
                pos = positions[sym]
                sym_data = day_data[day_data['symbol'] == sym]
                if len(sym_data) > 0:
                    exit_price = sym_data['close'].values[0]
                    pnl = exit_price / pos['entry_price'] - 1
                    cash += pos['size'] * (1 + pnl)
                del positions[sym]
            
            # 开仓
            if i % rebalance_freq == 0 and len(positions) < max_pos:
                signals = day_data[
                    (day_data['prob'] >= threshold) & 
                    (~day_data['symbol'].isin(positions.keys()))
                ].sort_values('prob', ascending=False)
                
                slots = max_pos - len(positions)
                for _, row in signals.head(slots).iterrows():
                    size = cash / (slots + 1)
                    if size > 0.01:
                        positions[row['symbol']] = {
                            'entry_price': row['close'],
                            'entry_idx': i,
                            'size': size
                        }
                        cash -= size
            
            # 组合价值
            position_value = 0
            for sym, pos in positions.items():
                sym_data = day_data[day_data['symbol'] == sym]
                if len(sym_data) > 0:
                    position_value += pos['size'] * (sym_data['close'].values[0] / pos['entry_price'])
            
            equity = cash + position_value
            equity_curve.append({
                'date': date,
                'equity': equity,
                'n_positions': len(positions)
            })
        
        eq = pd.DataFrame(equity_curve)
        eq['returns'] = eq['equity'].pct_change()
        
        # 指标计算
        total_days = (eq['date'].max() - eq['date'].min()).days
        total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
        annual_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
        
        rolling_max = eq['equity'].cummax()
        drawdown = eq['equity'] / rolling_max - 1
        max_drawdown = drawdown.min()
        
        # 平均回撤
        in_dd = False
        dd_start = 0
        dd_periods = []
        for i in range(len(eq)):
            if drawdown.iloc[i] < -0.001 and not in_dd:
                in_dd = True
                dd_start = i
            elif drawdown.iloc[i] >= 0 and in_dd:
                in_dd = False
                dd_periods.append(drawdown.iloc[dd_start:i].min())
        if in_dd:
            dd_periods.append(drawdown.iloc[dd_start:].min())
        
        avg_drawdown = np.mean(dd_periods) if dd_periods else 0
        avg_dd_depth = np.mean(dd_periods) if dd_periods else 0
        
        # 夏普
        daily_ret = eq['returns'].dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        
        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 胜率（每日）
        win_rate = (daily_ret > 0).mean() if len(daily_ret) > 0 else 0
        
        # 盈亏比
        wins = daily_ret[daily_ret > 0]
        losses = daily_ret[daily_ret < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_dd_depth,
            'sharpe': sharpe,
            'calmar': calmar,
            'win_rate': win_rate,
            'pl_ratio': pl_ratio,
            'avg_positions': eq['n_positions'].mean(),
            'n_days': len(eq)
        }

# 定义10种策略
strategies = {
    'A. 纯模型-激进': {'threshold': 0.85, 'max_positions': 20, 'hold_days': 5, 'rebalance_freq': 1},
    'B. 纯模型-保守': {'threshold': 0.90, 'max_positions': 10, 'hold_days': 5, 'rebalance_freq': 1},
    'C. 止损-10%': {'threshold': 0.85, 'max_positions': 15, 'hold_days': 10, 'stop_loss': -0.10, 'rebalance_freq': 1},
    'D. 止损-15%': {'threshold': 0.85, 'max_positions': 15, 'hold_days': 10, 'stop_loss': -0.15, 'rebalance_freq': 1},
    'E. 止损-10%+止盈20%': {'threshold': 0.85, 'max_positions': 15, 'hold_days': 15, 'stop_loss': -0.10, 'take_profit': 0.20, 'rebalance_freq': 1},
    'F. 每周调仓': {'threshold': 0.85, 'max_positions': 15, 'hold_days': 5, 'rebalance_freq': 5},
    'G. 持有3天': {'threshold': 0.85, 'max_positions': 20, 'hold_days': 3, 'rebalance_freq': 1},
    'H. 持有10天': {'threshold': 0.85, 'max_positions': 15, 'hold_days': 10, 'rebalance_freq': 1},
    'I. 高阈值+止损': {'threshold': 0.90, 'max_positions': 10, 'hold_days': 10, 'stop_loss': -0.10, 'rebalance_freq': 1},
    'J. 平衡型': {'threshold': 0.85, 'max_positions': 10, 'hold_days': 7, 'stop_loss': -0.10, 'take_profit': 0.30, 'rebalance_freq': 2}
}

bt = StrategyBacktester(test_df)
results = []

for name, config in strategies.items():
    print(f"  {name}...")
    r = bt.run(name, config)
    results.append(r)
    print(f"    年化 {r['annual_return']:+.1%} | 回撤 {r['max_drawdown']:.1%} | 夏普 {r['sharpe']:.2f}")

# ============================================================
# 5. 结果汇总
# ============================================================
print()
print("=" * 110)
print("蓝盾 V4 NeuralNet 决策框架 — 测试期回测结果")
print("=" * 110)
print(f"{'策略':<25} {'年化收益':>10} {'最大回撤':>10} {'平均回撤':>10} {'夏普':>7} {'Calmar':>7} {'日胜率':>7} {'盈亏比':>7} {'均持仓':>7}")
print("-" * 110)

results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
for r in results_sorted:
    print(f"{r['name']:<25} {r['annual_return']:>+9.1%} {r['max_drawdown']:>9.1%} {r['avg_drawdown']:>9.1%} {r['sharpe']:>7.2f} {r['calmar']:>7.2f} {r['win_rate']:>6.1%} {r['pl_ratio']:>7.2f} {r['avg_positions']:>6.1f}")

# 综合评分
for r in results:
    r['score'] = (
        max(r['sharpe'], 0) * 30 + 
        r['annual_return'] * 100 * 30 + 
        max(r['calmar'], 0) * 20 + 
        (1 + r['max_drawdown']) * 100 * 20
    )

best = max(results, key=lambda x: x['score'])

print()
print("=" * 110)
print()
print("🏆 推荐策略:", best['name'])
print()
print("┌─────────────────────────────────────────────────┐")
print(f"│  年化收益:     {best['annual_return']:>+8.1%}                        │")
print(f"│  最大回撤:     {best['max_drawdown']:>8.1%}                        │")
print(f"│  平均回撤:     {best['avg_drawdown']:>8.1%}                        │")
print(f"│  夏普比率:     {best['sharpe']:>8.2f}                        │")
print(f"│  Calmar比率:   {best['calmar']:>8.2f}                        │")
print(f"│  日胜率:       {best['win_rate']:>8.1%}                        │")
print(f"│  盈亏比:       {best['pl_ratio']:>8.2f}                        │")
print(f"│  平均持仓数:   {best['avg_positions']:>8.1f}                        │")
print("└─────────────────────────────────────────────────┘")

# 保存
output = {
    'model': 'NeuralNet (MLP)',
    'features': len(feature_cols),
    'test_period': f"{test_df['date'].min().date()} → {test_df['date'].max().date()}",
    'recommended': best['name'],
    'recommended_config': strategies[best['name']],
    'metrics': {k: v for k, v in best.items() if k != 'name'},
    'all_strategies': [{k: v for k, v in r.items()} for r in results_sorted]
}

with open('analysis/v4_decision_framework.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print()
print("结果已保存 → analysis/v4_decision_framework.json")
