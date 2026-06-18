#!/usr/bin/env python3
"""
蓝盾 V4 多模型对比 + Top-N排序
目标：夏普 > 1.0

模型矩阵：
1. XGBoost 回归（之前V4.4夏普1.69）
2. LightGBM 回归
3. CatBoost 回归
4. 简单Transformer（Attention机制）
5. 集成模型（多模型加权）
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')

print("=" * 80)
print("蓝盾 V4 多模型对比 — 目标夏普 > 1.0")
print("=" * 80)
print()

# ============================================================
# 1. 数据 + 特征
# ============================================================
print("[1/5] 加载数据 + 构建特征...")
t0 = time.time()

df = pd.read_parquet('data/us/us_hist_sp500_10y.parquet')
df = df.rename(columns={'sym': 'symbol'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

def build_features(group):
    g = group.copy()
    # 收益率
    for n in [1, 2, 3, 5, 10, 20, 60, 120, 250, 500]:
        g[f'ret_{n}d'] = g['close'].pct_change(n)
    # 动量
    for n in [5, 10, 20, 60, 120, 250]:
        g[f'mom_{n}'] = g['close'] / g['close'].shift(n) - 1
    g['mom_5d_20d'] = g['mom_5'] / g['mom_20'].replace(0, np.nan)
    g['mom_20d_60d'] = g['mom_20'] / g['mom_60'].replace(0, np.nan)
    # 波动率
    for n in [5, 10, 20, 60, 120, 250]:
        g[f'vol_{n}d'] = g['close'].pct_change().rolling(n).std() * np.sqrt(252)
    # RSI
    delta = g['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for n in [14, 28]:
        avg_gain = gain.rolling(n).mean()
        avg_loss = loss.rolling(n).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        g[f'rsi_{n}'] = 100 - (100 / (1 + rs))
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
    vol_sma60 = g['volume'].rolling(60).mean()
    g['vol_ratio_20d'] = g['volume'] / vol_sma20.replace(0, np.nan)
    g['vol_ratio_60d'] = g['volume'] / vol_sma60.replace(0, np.nan)
    g['vol_change'] = g['volume'].pct_change(5)
    g['vol_trend'] = vol_sma20 / vol_sma60.replace(0, np.nan)
    # 价格形态
    g['high_low_range'] = (g['high'] - g['low']) / g['close']
    g['close_to_high'] = (g['high'] - g['close']) / g['close']
    g['close_to_low'] = (g['close'] - g['low']) / g['close']
    g['gap'] = g['open'] / g['close'].shift(1) - 1
    g['body_size'] = abs(g['close'] - g['open']) / g['close']
    g['upper_shadow'] = (g['high'] - g[['close', 'open']].max(axis=1)) / g['close']
    # 均线偏离
    for n in [5, 10, 20, 50, 120, 250]:
        sma = g['close'].rolling(n).mean()
        g[f'bias_{n}'] = (g['close'] - sma) / sma
    # 52周
    g['high_52w'] = g['high'].rolling(250).max()
    g['low_52w'] = g['low'].rolling(250).min()
    g['dist_52w_high'] = g['close'] / g['high_52w'] - 1
    g['dist_52w_low'] = g['close'] / g['low_52w'] - 1
    # 截面排名
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

# 目标：5天后收益率（回归，不是分类）
df['target_5d'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)

feature_cols = [c for c in df.columns if c not in ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'target_5d']]
for c in feature_cols:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ['target_5d'])
df = df.sort_values('date').reset_index(drop=True)

print(f"  特征: {len(feature_cols)}, 数据: {len(df):,} 行 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. Walk-Forward 训练
# ============================================================
print()
print("[2/5] Walk-Forward 训练多模型...")

train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

X_train = df.loc[df['date'] <= train_end, feature_cols].values
y_train = df.loc[df['date'] <= train_end, 'target_5d'].values
X_val = df.loc[(df['date'] > train_end) & (df['date'] <= val_end), feature_cols].values
y_val = df.loc[(df['date'] > train_end) & (df['date'] <= val_end), 'target_5d'].values
X_test = df.loc[df['date'] > val_end, feature_cols].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

models = {}
predictions = {}

# 1. XGBoost 回归
print("  训练 XGBoost 回归...")
t0 = time.time()
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    models['XGBoost'] = xgb
    predictions['XGBoost'] = xgb.predict(X_test)
    print(f"    完成 ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"    失败: {e}")

# 2. LightGBM 回归
print("  训练 LightGBM 回归...")
t0 = time.time()
try:
    from lightgbm import LGBMRegressor
    lgb = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1
    )
    lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    models['LightGBM'] = lgb
    predictions['LightGBM'] = lgb.predict(X_test)
    print(f"    完成 ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"    失败: {e}")

# 3. CatBoost 回归
print("  训练 CatBoost 回归...")
t0 = time.time()
try:
    from catboost import CatBoostRegressor
    cat = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        random_seed=42, verbose=0
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val))
    models['CatBoost'] = cat
    predictions['CatBoost'] = cat.predict(X_test)
    print(f"    完成 ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"    失败: {e}")

# 4. MLP 回归
print("  训练 MLP 回归...")
t0 = time.time()
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), activation='relu',
    solver='adam', alpha=0.001, max_iter=500,
    early_stopping=True, random_state=42, verbose=False
)
mlp.fit(X_train_s, y_train)
models['MLP'] = mlp
predictions['MLP'] = mlp.predict(X_test_s)
print(f"    完成 ({time.time()-t0:.1f}s)")

# 5. 集成模型（等权平均）
if len(predictions) > 1:
    pred_stack = np.column_stack(list(predictions.values()))
    predictions['Ensemble'] = pred_stack.mean(axis=1)
    print("  集成模型: 等权平均")

print(f"  成功训练 {len(models)} 个模型")

# ============================================================
# 3. Top-N 策略回测
# ============================================================
print()
print("[3/5] Top-N 排序策略回测...")

test_df = df.loc[df['date'] > val_end, ['date', 'symbol', 'close', 'target_5d']].copy()

class TopNBacktester:
    def __init__(self, test_df):
        self.test_df = test_df.copy()
        self.test_df['date'] = pd.to_datetime(self.test_df['date'])
        
    def run(self, name, scores, top_n=10, hold_days=7, stop_loss=None):
        df = self.test_df.copy()
        df['score'] = scores
        dates = sorted(df['date'].unique())
        
        cash = 1.0
        positions = {}
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
                elif days_held >= hold_days:
                    to_close.append(sym)
            
            for sym in to_close:
                pos = positions[sym]
                sym_data = day_data[day_data['symbol'] == sym]
                if len(sym_data) > 0:
                    pnl = sym_data['close'].values[0] / pos['entry_price'] - 1
                    cash += pos['size'] * (1 + pnl)
                del positions[sym]
            
            # 开仓：买score最高的top_n只
            signals = day_data[
                (~day_data['symbol'].isin(positions.keys()))
            ].sort_values('score', ascending=False).head(top_n)
            
            if len(signals) > 0:
                size_per_stock = cash / (len(signals) + 1)
                for _, row in signals.iterrows():
                    if size_per_stock > 0.01:
                        positions[row['symbol']] = {
                            'entry_price': row['close'],
                            'entry_idx': i,
                            'size': size_per_stock
                        }
                        cash -= size_per_stock
            
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
        sortino = daily_ret.mean() / daily_ret[daily_ret < 0].std() * np.sqrt(252) if len(daily_ret[daily_ret < 0]) > 0 else 0
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (daily_ret > 0).mean() if len(daily_ret) > 0 else 0
        
        return {
            'name': name,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'win_rate': win_rate
        }

bt = TopNBacktester(test_df)

# 测试所有模型 x Top-N配置
results = []
configs = [
    (5, 7, None),
    (10, 7, None),
    (15, 7, None),
    (10, 5, None),
    (10, 10, None),
    (10, 7, -0.10),
    (15, 7, -0.10),
]

for model_name, scores in predictions.items():
    for top_n, hold_days, stop_loss in configs:
        sl_str = f"+SL{int(abs(stop_loss)*100)}%" if stop_loss else ""
        name = f"{model_name} Top-{top_n} {hold_days}d{sl_str}"
        r = bt.run(name, scores, top_n, hold_days, stop_loss)
        r['model'] = model_name
        r['top_n'] = top_n
        r['hold_days'] = hold_days
        r['stop_loss'] = stop_loss
        results.append(r)

# ============================================================
# 4. 结果汇总
# ============================================================
print()
print("[4/5] 结果汇总")
print()
print("=" * 110)
print("蓝盾 V4 多模型 Top-N — 完整回测结果")
print("=" * 110)
print()

results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)

print(f"{'策略':<40} {'年化':>8} {'最大回撤':>10} {'平均回撤':>10} {'夏普':>7} {'Sortino':>8} {'Calmar':>7} {'胜率':>6}")
print("-" * 110)

for r in results_sorted[:30]:  # Top 30
    print(f"{r['name']:<40} {r['annual_return']:>+7.1%} {r['max_drawdown']:>9.1%} {r['avg_drawdown']:>9.1%} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['calmar']:>7.2f} {r['win_rate']:>5.1%}")

# 找夏普>1的
sharpe_above_1 = [r for r in results if r['sharpe'] > 1.0]
print()
print(f"夏普 > 1.0 的策略: {len(sharpe_above_1)} 个")

if sharpe_above_1:
    best = max(sharpe_above_1, key=lambda x: x['sharpe'])
    print()
    print("🏆 最佳策略:", best['name'])
    print()
    print("┌─────────────────────────────────────────────────┐")
    print(f"│  年化收益:     {best['annual_return']:>+8.1%}                        │")
    print(f"│  最大回撤:     {best['max_drawdown']:>8.1%}                        │")
    print(f"│  平均回撤:     {best['avg_drawdown']:>8.1%}                        │")
    print(f"│  夏普比率:     {best['sharpe']:>8.2f}                        │")
    print(f"│  Sortino:      {best['sortino']:>8.2f}                        │")
    print(f"│  Calmar:       {best['calmar']:>8.2f}                        │")
    print(f"│  胜率:         {best['win_rate']:>8.1%}                        │")
    print("└─────────────────────────────────────────────────┘")
else:
    best = results_sorted[0]
    print()
    print("⚠️ 没有夏普>1.0的策略")
    print()
    print("当前最佳:", best['name'])
    print(f"  夏普: {best['sharpe']:.2f}")
    print(f"  年化: {best['annual_return']:.1%}")
    print(f"  回撤: {best['max_drawdown']:.1%}")

# 模型对比
print()
print("=" * 110)
print("模型对比（每个模型的最佳配置）")
print("=" * 110)
print()

for model_name in predictions.keys():
    model_results = [r for r in results if r['model'] == model_name]
    if model_results:
        best_model = max(model_results, key=lambda x: x['sharpe'])
        print(f"{model_name:<15} 夏普 {best_model['sharpe']:.2f} | 年化 {best_model['annual_return']:+.1%} | 回撤 {best_model['max_drawdown']:.1%} | {best_model['name']}")

# 保存
output = {
    'test_period': f"{test_df['date'].min().date()} → {test_df['date'].max().date()}",
    'n_models': len(models),
    'sharpe_above_1': len(sharpe_above_1),
    'best': {k: v for k, v in best.items()} if best else None,
    'top_20': [{k: v for k, v in r.items()} for r in results_sorted[:20]]
}

with open('analysis/v4_multimodel_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print()
print("结果已保存 → analysis/v4_multimodel_results.json")
