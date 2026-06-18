#!/usr/bin/env python3
"""
蓝盾 V4 Top-N 排序策略回测
==========================
核心思路：不用分类（涨/跌），用排序（比别人好/差）
每天只买概率最高的N只股票
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

print("=" * 80)
print("蓝盾 V4 Top-N 排序策略")
print("=" * 80)
print()

# ============================================================
# 1. 数据 + 特征
# ============================================================
print("[1/4] 加载数据 + 构建58维特征...")
t0 = time.time()

df = pd.read_parquet('data/us/us_hist_sp500_10y.parquet')
df = df.rename(columns={'sym': 'symbol'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

def build_features(group):
    g = group.copy()
    for n in [1, 2, 3, 5, 10, 20, 60, 120, 250, 500]:
        g[f'ret_{n}d'] = g['close'].pct_change(n)
    for n in [5, 10, 20, 60, 120, 250]:
        g[f'mom_{n}'] = g['close'] / g['close'].shift(n) - 1
    g['mom_5d_20d'] = g['mom_5'] / g['mom_20'].replace(0, np.nan)
    g['mom_20d_60d'] = g['mom_20'] / g['mom_60'].replace(0, np.nan)
    for n in [5, 10, 20, 60, 120, 250]:
        g[f'vol_{n}d'] = g['close'].pct_change().rolling(n).std() * np.sqrt(252)
    delta = g['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for n in [14, 28]:
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g[f'rsi_{n}'] = 100 - (100 / (1 + rs))
    ema12 = g['close'].ewm(span=12).mean()
    ema26 = g['close'].ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    sma20 = g['close'].rolling(20).mean()
    std20 = g['close'].rolling(20).std()
    g['bb_upper'] = sma20 + 2 * std20
    g['bb_lower'] = sma20 - 2 * std20
    g['bb_width'] = (g['bb_upper'] - g['bb_lower']) / sma20
    g['bb_pos'] = (g['close'] - g['bb_lower']) / (g['bb_upper'] - g['bb_lower']).replace(0, np.nan)
    vol_sma20 = g['volume'].rolling(20).mean()
    vol_sma60 = g['volume'].rolling(60).mean()
    g['vol_ratio_20d'] = g['volume'] / vol_sma20.replace(0, np.nan)
    g['vol_ratio_60d'] = g['volume'] / vol_sma60.replace(0, np.nan)
    g['vol_change'] = g['volume'].pct_change(5)
    g['vol_trend'] = vol_sma20 / vol_sma60.replace(0, np.nan)
    g['high_low_range'] = (g['high'] - g['low']) / g['close']
    g['close_to_high'] = (g['high'] - g['close']) / g['close']
    g['close_to_low'] = (g['close'] - g['low']) / g['close']
    g['gap'] = g['open'] / g['close'].shift(1) - 1
    g['body_size'] = abs(g['close'] - g['open']) / g['close']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    for n in [5, 10, 20, 50, 120, 250]:
        sma = g['close'].rolling(n).mean()
        g[f'bias_{n}'] = (g['close'] - sma) / sma
    g['high_52w'] = g['high'].rolling(250).max()
    g['low_52w'] = g['low'].rolling(250).min()
    g['dist_52w_high'] = g['close'] / g['high_52w'] - 1
    g['dist_52w_low'] = g['close'] / g['low_52w'] - 1
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

df['target_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
df['target_binary'] = (df['target_5d'] > 0.03).astype(int)

feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target_5d', 'target_binary']]
for c in feature_cols:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ['target_5d', 'target_binary'])
df = df.sort_values('date').reset_index(drop=True)

print(f"  特征: {len(feature_cols)}, 数据: {len(df):,} 行 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. 训练模型
# ============================================================
print()
print("[2/4] 训练NeuralNet...")

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

X_train = df.loc[df['date'] <= train_end, feature_cols].values
y_train = df.loc[df['date'] <= train_end, 'target_binary'].values
X_test = df.loc[df['date'] > val_end, feature_cols].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

t0 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu', solver='adam', alpha=0.001,
    max_iter=500, early_stopping=True, random_state=42, verbose=False
)
mlp.fit(X_train_s, y_train)
print(f"  完成 ({time.time()-t0:.1f}s), {mlp.n_iter_} iterations")

test_probs = mlp.predict_proba(X_test_s)[:, 1]
test_df = df.loc[df['date'] > val_end, ['date', 'symbol', 'close', 'target_5d']].copy()
test_df['prob'] = test_probs

# ============================================================
# 3. Top-N 策略回测
# ============================================================
print()
print("[3/4] Top-N 排序策略回测...")

class TopNBacktester:
    def __init__(self, test_df):
        self.test_df = test_df.copy()
        self.test_df['date'] = pd.to_datetime(self.test_df['date'])
        
    def run(self, name, top_n, hold_days=5, stop_loss=None, take_profit=None):
        df = self.test_df.copy()
        dates = sorted(df['date'].unique())
        
        cash = 1.0
        positions = {}  # {symbol: {entry_price, entry_idx, size}}
        equity_curve = []
        
        for i, date in enumerate(dates):
            day_data = df[df['date'] == date]
            
            # 检查退出
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
                elif days_held >= hold_days:
                    to_close.append(sym)
            
            for sym in to_close:
                pos = positions[sym]
                sym_data = day_data[day_data['symbol'] == sym]
                if len(sym_data) > 0:
                    pnl = sym_data['close'].values[0] / pos['entry_price'] - 1
                    cash += pos['size'] * (1 + pnl)
                del positions[sym]
            
            # 开仓：每天买概率最高的 top_n 只（还没持有的）
            signals = day_data[
                (~day_data['symbol'].isin(positions.keys()))
            ].sort_values('prob', ascending=False).head(top_n)
            
            if len(signals) > 0:
                size_per_stock = cash / (len(signals) + 1)  # 留1份现金
                for _, row in signals.iterrows():
                    if size_per_stock > 0.01:
                        positions[row['symbol']] = {
                            'entry_price': row['close'],
                            'entry_idx': i,
                            'size': size_per_stock
                        }
                        cash -= size_per_stock
            
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
        
        return self._calc_metrics(equity_curve, name)
    
    def _calc_metrics(self, equity_curve, name):
        eq = pd.DataFrame(equity_curve)
        eq['returns'] = eq['equity'].pct_change()
        
        total_days = (eq['date'].max() - eq['date'].min()).days
        total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
        annual_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0
        
        rolling_max = eq['equity'].cummax()
        drawdown = eq['equity'] / rolling_max - 1
        max_drawdown = drawdown.min()
        
        # 平均回撤
        in_dd = False
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
        
        daily_ret = eq['returns'].dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (daily_ret > 0).mean() if len(daily_ret) > 0 else 0
        
        # 盈亏比
        wins = daily_ret[daily_ret > 0]
        losses = daily_ret[daily_ret < 0]
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
        pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Sortino (只用下行波动率)
        downside_ret = daily_ret[daily_ret < 0]
        downside_std = downside_ret.std() if len(downside_ret) > 0 else 1
        sortino = daily_ret.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'win_rate': win_rate,
            'pl_ratio': pl_ratio,
            'avg_positions': eq['n_positions'].mean(),
            'n_trades': len(eq)
        }

bt = TopNBacktester(test_df)

# 测试不同Top-N配置
configs = [
    ('Top-1', 1, 5, None, None),
    ('Top-1 + 止损10%', 1, 5, -0.10, None),
    ('Top-1 + 止损10%+止盈20%', 1, 5, -0.10, 0.20),
    ('Top-3', 3, 5, None, None),
    ('Top-3 + 止损10%', 3, 5, -0.10, None),
    ('Top-3 + 止损10%+止盈20%', 3, 5, -0.10, 0.20),
    ('Top-5', 5, 5, None, None),
    ('Top-5 + 止损10%', 5, 5, -0.10, None),
    ('Top-5 + 止损10%+止盈20%', 5, 5, -0.10, 0.20),
    ('Top-10', 10, 5, None, None),
    ('Top-10 + 止损10%', 10, 5, -0.10, None),
    ('Top-10 + 止损10%+止盈20%', 10, 5, -0.10, 0.20),
    ('Top-3 持有3天', 3, 3, None, None),
    ('Top-3 持有7天', 3, 7, None, None),
    ('Top-3 持有10天', 3, 10, None, None),
    ('Top-5 持有3天', 5, 3, None, None),
    ('Top-5 持有7天', 5, 7, None, None),
    ('Top-5 持有10天', 5, 10, None, None),
]

results = []
for name, top_n, hold_days, sl, tp in configs:
    print(f"  {name}...")
    r = bt.run(name, top_n, hold_days, sl, tp)
    results.append(r)

# ============================================================
# 4. 结果汇总
# ============================================================
print()
print("=" * 100)
print("蓝盾 V4 Top-N 排序策略 — 完整回测结果")
print("=" * 100)
print()

# 按夏普排序
results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)

print(f"{'策略':<28} {'年化':>8} {'最大回撤':>10} {'平均回撤':>10} {'夏普':>7} {'Sortino':>8} {'Calmar':>7} {'胜率':>6} {'盈亏比':>7}")
print("-" * 100)

for r in results_sorted:
    print(f"{r['name']:<28} {r['annual_return']:>+7.1%} {r['max_drawdown']:>9.1%} {r['avg_drawdown']:>9.1%} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['win_rate']:>5.1%} {r['pl_ratio']:>7.2f}")

# 综合评分
for r in results:
    r['score'] = (
        max(r['sharpe'], 0) * 30 + 
        max(r['sortino'], 0) * 20 +
        r['annual_return'] * 100 * 30 + 
        max(r['calmar'], 0) * 20
    )

best = max(results, key=lambda x: x['score'])

print()
print("=" * 100)
print()
print("🏆 综合最优:", best['name'])
print()
print("┌─────────────────────────────────────────────────┐")
print(f"│  年化收益:     {best['annual_return']:>+8.1%}                        │")
print(f"│  最大回撤:     {best['max_drawdown']:>8.1%}                        │")
print(f"│  平均回撤:     {best['avg_drawdown']:>8.1%}                        │")
print(f"│  夏普比率:     {best['sharpe']:>8.2f}                        │")
print(f"│  Sortino:      {best['sortino']:>8.2f}                        │")
print(f"│  Calmar:       {best['calmar']:>8.2f}                        │")
print(f"│  胜率:         {best['win_rate']:>8.1%}                        │")
print(f"│  盈亏比:       {best['pl_ratio']:>8.2f}                        │")
print("└─────────────────────────────────────────────────┘")

# 分组对比
print()
print("=" * 100)
print("分组对比：Top-N 的影响")
print("=" * 100)
print()

for n in [1, 3, 5, 10]:
    group = [r for r in results if f'Top-{n}' in r['name'] and '持有' not in r['name']]
    if group:
        avg_sharpe = np.mean([r['sharpe'] for r in group])
        avg_annual = np.mean([r['annual_return'] for r in group])
        best_in_group = max(group, key=lambda x: x['sharpe'])
        print(f"Top-{n}: 平均夏普 {avg_sharpe:.2f}, 平均年化 {avg_annual:.1%}, 最佳: {best_in_group['name']} (夏普 {best_in_group['sharpe']:.2f})")

print()
print("=" * 100)
print("持有期的影响")
print("=" * 100)
print()

for days in [3, 5, 7, 10]:
    group = [r for r in results if f'持有{days}天' in r['name'] or (days == 5 and '持有' not in r['name'] and '止损' not in r['name'] and '止盈' not in r['name'])]
    if group:
        avg_sharpe = np.mean([r['sharpe'] for r in group])
        print(f"持有{days}天: 平均夏普 {avg_sharpe:.2f}")

# 保存
output = {
    'model': 'NeuralNet (MLP, 58 features)',
    'test_period': f"{test_df['date'].min().date()} → {test_df['date'].max().date()}",
    'recommended': best['name'],
    'results': [{k: v for k, v in r.items()} for r in results_sorted]
}

with open('analysis/v4_topn_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print()
print("结果已保存 → analysis/v4_topn_results.json")
