#!/usr/bin/env python3
"""
蓝盾V4 深度研究 — 模型层+决策层联合优化
目标：
  1. 模型层：降低平均回撤（选股质量）+ 提升夏普
  2. 决策层：动态仓位管理降低最大回撤
  
Andy要求：
  - 高波动可接受（配合高收益）
  - 最大回撤可通过减仓控制
  - 平均回撤高 = 模型退出信号差 = 根因
"""

import pandas as pd
import numpy as np
import json
import time
import warnings
from sklearn.metrics import roc_auc_score
from collections import defaultdict

warnings.filterwarnings('ignore')

DATA = '/home/hermes/.hermes/openclaw-archive/data/us/us_hist_sp500_10y.parquet'
OUTPUT_DIR = '/home/hermes/.hermes/openclaw-archive/analysis'

print("=" * 90)
print("蓝盾V4 深度研究 — 模型层+决策层联合优化")
print("=" * 90)
print()

# ============================================================
# 1. 数据加载 + 增强特征
# ============================================================
print("[1/6] 加载数据 + 构建增强特征...")
t0 = time.time()

df = pd.read_parquet(DATA)
df = df.rename(columns={'sym': 'code'})
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['code', 'date']).reset_index(drop=True)

# 排除ETF
etfs = {'SPY','QQQ','IWM','DIA','VOO','IVV','XLK','XLF','XLV','XLE','XLI',
        'XLP','XLU','XLRE','XLB','XLC','XLY','SPX'}
df = df[~df['code'].isin(etfs)].copy()

# 预计算市场特征：用全截面中位数收益/波动
print("  计算市场特征...")
df_market = df.groupby('date').agg(
    mkt_ret_1d=('close', lambda x: x.pct_change().median()),
    mkt_ret_5d=('close', lambda x: x.pct_change(5).median()),
    mkt_ret_20d=('close', lambda x: x.pct_change(20).median()),
    mkt_breadth=('close', lambda x: (x.pct_change(5) > 0).mean()),
    mkt_vol_20d=('close', lambda x: x.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(x) > 20 else np.nan),
    mkt_avg_vol=('volume', lambda x: x.mean()),
).reset_index()
df = df.merge(df_market, on='date', how='left')
print(f"  数据: {len(df):,}行, {df['code'].nunique()}只股票")

def build_enhanced_features(group):
    """增强特征集 — 加入截面排名和跨期特征"""
    g = group.copy()
    c = g['close']
    v = g['volume']
    h = g['high']
    l = g['low']
    o = g['open']
    
    # === 收益率 ===
    for n in [1, 2, 3, 5, 10, 20, 60]:
        g[f'ret_{n}d'] = c.pct_change(n)
    
    # === 波动率 ===
    for n in [5, 10, 20, 60]:
        g[f'vol_{n}d'] = c.pct_change().rolling(n).std() * np.sqrt(252)
    
    # === RSI ===
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    for n in [14, 28]:
        ag = gain.rolling(n).mean()
        al = loss.rolling(n).mean()
        rs = ag / al.replace(0, np.nan)
        g[f'rsi_{n}'] = 100 - 100 / (1 + rs)
    
    # === MACD ===
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    g['macd'] = ema12 - ema26
    g['macd_signal'] = g['macd'].ewm(span=9).mean()
    g['macd_hist'] = g['macd'] - g['macd_signal']
    
    # === 布林带 ===
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    g['bb_width'] = (4 * std20) / sma20.replace(0, np.nan)
    g['bb_pos'] = (c - (sma20 - 2 * std20)) / (4 * std20).replace(0, np.nan)
    
    # === 成交量 ===
    vol_sma20 = v.rolling(20).mean()
    vol_sma60 = v.rolling(60).mean()
    g['vol_ratio_20d'] = v / vol_sma20.replace(0, np.nan)
    g['vol_ratio_60d'] = v / vol_sma60.replace(0, np.nan)
    g['vol_trend'] = vol_sma20 / vol_sma60.replace(0, np.nan)
    
    # === 价格形态 ===
    g['high_low_range'] = (h - l) / c
    g['close_open_range'] = (c - o) / o.replace(0, np.nan)
    g['upper_shadow'] = (h - pd.concat([c, o], axis=1).max(axis=1)) / c
    g['lower_shadow'] = (pd.concat([c, o], axis=1).min(axis=1) - l) / c
    g['body_ratio'] = abs(c - o) / (h - l).replace(0, np.nan)
    
    # === 均线偏离 ===
    for n in [5, 10, 20, 50]:
        sma = c.rolling(n).mean()
        g[f'bias_{n}'] = (c - sma) / sma.replace(0, np.nan)
    
    # === 52周 (用250天但不强制dropna) ===
    g['high_52w'] = h.rolling(250).max()
    g['low_52w'] = l.rolling(250).min()
    g['dist_52w_high'] = c / g['high_52w'] - 1
    g['dist_52w_low'] = c / g['low_52w'] - 1
    
    # === 动量 ===
    for n in [5, 10, 20, 60]:
        g[f'mom_{n}'] = c / c.shift(n) - 1
    
    # === ATR ===
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    g['atr_14'] = tr.rolling(14).mean()
    g['atr_pct'] = g['atr_14'] / c
    
    # === 趋势强度 ===
    for n in [20, 60]:
        g[f'trend_{n}'] = c.rolling(n).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x) if len(x) == n else np.nan,
            raw=True
        )
    
    # === 收益率统计 ===
    ret5 = c.pct_change(5)
    ret20 = c.pct_change(20)
    g['ret_mean_20'] = c.pct_change().rolling(20).mean()
    g['ret_skew_20'] = c.pct_change().rolling(20).skew()
    g['ret_kurt_20'] = c.pct_change().rolling(20).kurt()
    
    # === 相对强度 ===
    g['ret_ratio_5_20'] = ret5 / ret20.replace(0, np.nan)
    g['ret_ratio_10_60'] = c.pct_change(10) / c.pct_change(60).replace(0, np.nan)
    
    return g

groups = []
for code, grp in df.groupby('code'):
    groups.append(build_enhanced_features(grp))
df = pd.concat(groups, ignore_index=True)

# 合并市场特征（已在df_market中计算，直接使用）
df['mkt_sma20'] = df.groupby('code')['mkt_ret_5d'].transform(lambda x: x.rolling(20).mean())
df['market_score'] = np.where(df['mkt_breadth'] > 0.5, 1.0, 0.7)
df['market_score'] = np.where(df['mkt_breadth'] > 0.6, 1.0, df['market_score'])

# 目标
df['target_5d'] = df.groupby('code')['close'].transform(lambda x: x.shift(-5) / x - 1)

# 特征列
exclude = {'date', 'code', 'open', 'high', 'low', 'close', 'volume', 'target_5d',
           'high_52w', 'low_52w', 'mkt_avg_vol'}
feature_cols = [c for c in df.columns if c not in exclude]
df = df.replace([np.inf, -np.inf], np.nan)

# 核心特征（无长窗口依赖）用于dropna
core_drop = [c for c in feature_cols if not c.startswith('dist_52w') 
             and not c.startswith('trend_') and not c.startswith('ret_skew')
             and not c.startswith('ret_kurt') and not c.startswith('ret_ratio')]
df = df.dropna(subset=core_drop + ['target_5d'])
# 其余特征允许NaN（tree model天然处理）
df = df.sort_values('date').reset_index(drop=True)

print(f"  特征: {len(feature_cols)}维, 数据: {len(df):,}行 ({time.time()-t0:.1f}s)")

# ============================================================
# 2. 数据划分
# ============================================================
train_end = pd.Timestamp('2021-12-31')
val_end = pd.Timestamp('2023-12-31')

train_mask = df['date'] <= train_end
val_mask = (df['date'] > train_end) & (df['date'] <= val_end)
test_mask = df['date'] > val_end

X_train = df.loc[train_mask, feature_cols].values
X_val = df.loc[val_mask, feature_cols].values
X_test = df.loc[test_mask, feature_cols].values

y_train = df.loc[train_mask, 'target_5d'].values
y_val = df.loc[val_mask, 'target_5d'].values
y_test = df.loc[test_mask, 'target_5d'].values

print(f"  训练: {len(X_train):,} | 验证: {len(X_val):,} | 测试: {len(X_test):,}")

# ============================================================
# 3. 多模型训练（回归，预测5日收益率）
# ============================================================
print(f"\n[2/6] 训练多模型...")
t0 = time.time()

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

models = {}

# XGBoost
print("  XGBoost...")
xgb_m = XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=10,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
models['XGBoost'] = xgb_m

# LightGBM
print("  LightGBM...")
lgb_m = LGBMRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
    min_child_samples=20,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
models['LightGBM'] = lgb_m

# CatBoost
print("  CatBoost...")
cat_m = CatBoostRegressor(
    iterations=800, depth=6, learning_rate=0.03,
    l2_leaf_reg=3, random_seed=42, verbose=0
)
cat_m.fit(X_train, y_train, eval_set=(X_val, y_val))
models['CatBoost'] = cat_m

print(f"  训练完成 ({time.time()-t0:.1f}s)")

# 预测
predictions = {}
for name, m in models.items():
    predictions[name] = m.predict(X_test)

# ============================================================
# 4. Top-N回测框架 + 退出分析
# ============================================================
print(f"\n[3/6] Top-N回测 + 退出质量分析...")
print("  （重点关注：平均回撤、每笔持仓的最大回撤、持有期分布）")

test_df = df.loc[test_mask, ['date', 'code', 'close', 'target_5d']].copy()
test_df = test_df.reset_index(drop=True)

class DeepBacktester:
    """深度回测：追踪每笔交易的完整生命周期"""
    
    def __init__(self, test_df):
        self.test_df = test_df.copy()
        self.test_df['date'] = pd.to_datetime(self.test_df['date'])
        self.dates = sorted(self.test_df['date'].unique())
        self.date_idx = {d: i for i, d in enumerate(self.dates)}
        
        # 构建每日价格快查
        self.daily = {}
        for d in self.dates:
            day = self.test_df[self.test_df['date'] == d].set_index('code')
            self.daily[d] = day
    
    def run(self, scores, top_n=15, hold_days=7, stop_loss=None, 
            trailing_stop=None, market_timing=None):
        """
        market_timing: Series indexed by date, 0~1, 表示仓位比例
        """
        n = len(self.dates)
        equity = 1.0
        equity_curve = []
        all_trades = []
        positions = {}
        
        for i, date in enumerate(self.dates):
            day = self.daily.get(date, pd.DataFrame())
            if len(day) == 0:
                equity_curve.append({'date': date, 'equity': equity, 'n_pos': 0})
                continue
            
            # === 检查退出 ===
            to_close = []
            for sym, pos in positions.items():
                if sym not in day.index:
                    continue
                price = day.loc[sym, 'close'] if sym in day.index else pos['last_price']
                ret = price / pos['entry_price'] - 1
                days_held = i - pos['entry_idx']
                
                # 更新最高价（trailing stop）
                if trailing_stop and price > pos.get('peak_price', pos['entry_price']):
                    pos['peak_price'] = price
                    pos['peak_ret'] = price / pos['entry_price'] - 1
                
                close_it = False
                reason = ''
                
                if stop_loss and ret <= stop_loss:
                    close_it = True
                    reason = 'stop_loss'
                elif trailing_stop and pos.get('peak_ret', 0) > 0.05:
                    # 从最高点回落超过trailing_stop
                    if (pos['peak_price'] - price) / pos['peak_price'] >= trailing_stop:
                        close_it = True
                        reason = 'trailing_stop'
                elif days_held >= hold_days:
                    close_it = True
                    reason = 'hold_expire'
                
                if close_it:
                    pnl = price / pos['entry_price'] - 1
                    equity += pos['size'] * pnl
                    pos['exit_price'] = price
                    pos['exit_date'] = date
                    pos['pnl'] = pnl
                    pos['days_held'] = days_held
                    pos['exit_reason'] = reason
                    all_trades.append(pos.copy())
                    to_close.append(sym)
            
            for sym in to_close:
                del positions[sym]
            
            # === 开仓 ===
            available = [s for s in day.index if s not in positions]
            if len(available) > 0 and top_n > 0:
                # 为available股票匹配分数
                score_map = {}
                for sym in available:
                    # 从scores中查找（按date+code匹配）
                    mask = (test_df['date'] == date) & (test_df['code'] == sym)
                    idx = mask.values.nonzero()[0]
                    if len(idx) > 0:
                        score_map[sym] = scores[idx[0]]
                
                if score_map:
                    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
                    new_syms = [s for s, _ in ranked[:top_n]]
                    
                    # 市场择时调整仓位
                    timing_factor = 1.0
                    if market_timing is not None and date in market_timing.index:
                        timing_factor = float(market_timing.loc[date])
                        timing_factor = max(0.1, min(1.0, timing_factor))
                    
                    available_cash = equity * timing_factor
                    n_new = len(new_syms)
                    if n_new > 0 and available_cash > 0.01:
                        size_per = available_cash / n_new
                        for sym in new_syms:
                            if sym in day.index and size_per > 0.005:
                                positions[sym] = {
                                    'code': sym,
                                    'entry_price': day.loc[sym, 'close'],
                                    'entry_date': date,
                                    'entry_idx': i,
                                    'last_price': day.loc[sym, 'close'],
                                    'peak_price': day.loc[sym, 'close'],
                                    'peak_ret': 0,
                                    'size': size_per,
                                }
                                equity -= size_per
            
            # 计算持仓市值
            pos_value = 0
            for sym, pos in positions.items():
                if sym in day.index:
                    price = day.loc[sym, 'close']
                    pos['last_price'] = price
                    pos_value += pos['size'] * (price / pos['entry_price'])
            
            equity = equity  # cash portion stays same (减去size_per已扣除)
            # 重新计算equity: cash + positions
            cash = 1.0
            for sym, pos in positions.items():
                cash -= pos['size']
            equity = cash + pos_value
            
            equity_curve.append({'date': date, 'equity': equity, 'n_pos': len(positions)})
        
        return equity_curve, all_trades
    
    def calc_metrics(self, equity_curve, all_trades, name=""):
        """全面指标计算"""
        eq = pd.DataFrame(equity_curve)
        eq['returns'] = eq['equity'].pct_change()
        
        total_days = (eq['date'].max() - eq['date'].min()).days
        total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
        annual_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1
        
        rolling_max = eq['equity'].cummax()
        dd = eq['equity'] / rolling_max - 1
        max_dd = dd.min()
        
        # 平均回撤 = 每次回撤期间的平均最大深度
        in_dd = False
        dd_periods = []
        for i in range(len(dd)):
            if dd.iloc[i] < -0.001 and not in_dd:
                in_dd = True
                dd_start = i
            elif dd.iloc[i] >= -0.001 and in_dd:
                in_dd = False
                dd_periods.append(dd.iloc[dd_start:i].min())
        if in_dd:
            dd_periods.append(dd.iloc[dd_start:].min())
        avg_dd = np.mean(dd_periods) if dd_periods else 0
        
        # 日度指标
        daily_ret = eq['returns'].dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
        sortino = daily_ret.mean() / daily_ret[daily_ret < 0].std() * np.sqrt(252) if (daily_ret < 0).sum() > 0 else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # 交易层面指标
        if all_trades:
            trade_pnls = [t['pnl'] for t in all_trades]
            win_trades = [p for p in trade_pnls if p > 0]
            lose_trades = [p for p in trade_pnls if p <= 0]
            avg_win = np.mean(win_trades) if win_trades else 0
            avg_loss = abs(np.mean(lose_trades)) if lose_trades else 0.001
            win_rate = len(win_trades) / len(trade_pnls) if trade_pnls else 0
            avg_hold = np.mean([t['days_held'] for t in all_trades])
            
            # 每笔交易的最大回撤（持仓期间价格从最高点的跌幅）
            trade_drawdowns = []
            # 简化：用pnl估算（实际持仓回撤需要tick数据）
            avg_trade_pnl = np.mean(trade_pnls)
            
            # 退出原因分布
            exit_reasons = defaultdict(int)
            for t in all_trades:
                exit_reasons[t.get('exit_reason', 'unknown')] += 1
        else:
            win_rate = avg_win = avg_loss = avg_hold = 0
            exit_reasons = {}
            trade_pnls = []
        
        return {
            'name': name,
            'annual_return': annual_return,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'avg_dd_count': len(dd_periods),
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'win_rate': win_rate,
            'n_trades': len(all_trades) if all_trades else 0,
            'avg_win': avg_win if all_trades else 0,
            'avg_loss': avg_loss if all_trades else 0,
            'avg_hold_days': avg_hold if all_trades else 0,
            'exit_reasons': dict(exit_reasons),
            'daily_returns': daily_ret.tolist() if len(daily_ret) > 0 else [],
        }

bt = DeepBacktester(test_df)

# ============================================================
# 5. 实验矩阵
# ============================================================
print(f"\n[4/6] 实验矩阵（模型×策略）...")

# 策略1: 基线 — XGB Top-15, 7天
# 策略2: LGB Top-15, 7天  
# 策略3: CatBoost Top-15, 7天
# 策略4: 等权集成 Top-15, 7天
# 策略5: 集成 Top-10 vs Top-15 vs Top-20
# 策略6: 加止损/Trailing Stop
# 策略7: 市场择时

# 市场择时信号: 用截面市场特征
mkt_data = df.loc[test_mask, ['date', 'mkt_breadth', 'mkt_ret_5d']].drop_duplicates(subset='date').sort_values('date')
mkt_data['mkt_sma20'] = mkt_data['mkt_ret_5d'].rolling(20).mean()
mkt_data['market_score'] = np.where(mkt_data['mkt_breadth'] > 0.5, 1.0, 0.6)
mkt_data['market_score'] = np.where(mkt_data['mkt_breadth'] > 0.6, 1.0, mkt_data['market_score'])
spy_timing = mkt_data.set_index('date')['market_score']

# 实验配置
experiments = []

# 基线实验
for model_name, pred in predictions.items():
    experiments.append({
        'name': f'{model_name} Top-15 7d',
        'model': model_name,
        'scores': pred,
        'top_n': 15,
        'hold_days': 7,
        'stop_loss': None,
        'trailing_stop': None,
        'timing': None,
    })

# 集成实验
ens_avg = np.column_stack(list(predictions.values())).mean(axis=1)
ens_weighted = predictions['XGBoost'] * 0.4 + predictions['LightGBM'] * 0.3 + predictions['CatBoost'] * 0.3

for name, scores in [('等权集成', ens_avg), ('加权集成', ens_weighted)]:
    for top_n in [10, 15, 20]:
        experiments.append({
            'name': f'{name} Top-{top_n} 7d',
            'model': name,
            'scores': scores,
            'top_n': top_n,
            'hold_days': 7,
            'stop_loss': None,
            'trailing_stop': None,
            'timing': None,
        })

# 止损实验
for name, scores in [('加权集成', ens_weighted)]:
    experiments.append({
        'name': '加权集成 Top-15 7d SL-8%',
        'scores': scores,
        'top_n': 15, 'hold_days': 7,
        'stop_loss': -0.08, 'trailing_stop': None, 'timing': None,
    })
    experiments.append({
        'name': '加权集成 Top-15 7d TS-5%',
        'scores': scores,
        'top_n': 15, 'hold_days': 7,
        'stop_loss': -0.10, 'trailing_stop': 0.05, 'timing': None,
    })
    experiments.append({
        'name': '加权集成 Top-15 5d',
        'scores': scores,
        'top_n': 15, 'hold_days': 5,
        'stop_loss': None, 'trailing_stop': None, 'timing': None,
    })

# 市场择时实验
for name, scores in [('加权集成', ens_weighted), ('等权集成', ens_avg)]:
    experiments.append({
        'name': f'{name} Top-15 7d + 市场择时',
        'scores': scores,
        'top_n': 15, 'hold_days': 7,
        'stop_loss': None, 'trailing_stop': None,
        'timing': spy_timing,
    })
    experiments.append({
        'name': f'{name} Top-15 7d TS-5% + 市场择时',
        'scores': scores,
        'top_n': 15, 'hold_days': 7,
        'stop_loss': -0.10, 'trailing_stop': 0.05,
        'timing': spy_timing,
    })

# 不同持有期
for hold in [3, 5, 10, 14]:
    experiments.append({
        'name': f'加权集成 Top-15 {hold}d',
        'scores': ens_weighted,
        'top_n': 15, 'hold_days': hold,
        'stop_loss': None, 'trailing_stop': None, 'timing': None,
    })

print(f"  共 {len(experiments)} 个实验配置")

# ============================================================
# 6. 运行所有实验
# ============================================================
print(f"\n[5/6] 运行 {len(experiments)} 个实验...")
results = []

for i, exp in enumerate(experiments):
    t1 = time.time()
    eq, trades = bt.run(
        scores=exp['scores'],
        top_n=exp['top_n'],
        hold_days=exp['hold_days'],
        stop_loss=exp.get('stop_loss'),
        trailing_stop=exp.get('trailing_stop'),
        market_timing=exp.get('timing'),
    )
    metrics = bt.calc_metrics(eq, trades, name=exp['name'])
    metrics['config'] = exp
    results.append(metrics)
    elapsed = time.time() - t1
    print(f"  [{i+1}/{len(experiments)}] {exp['name']:<40} 夏普{metrics['sharpe']:.2f} 年化{metrics['annual_return']:+.1%} 最大DD{metrics['max_drawdown']:.1%} 平均DD{metrics['avg_drawdown']:.1%} ({elapsed:.1f}s)")

# ============================================================
# 7. 结果分析
# ============================================================
print(f"\n[6/6] 结果深度分析")
print("=" * 120)

# 按夏普排序
results_by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'策略':<45} {'年化':>7} {'最大DD':>8} {'平均DD':>8} {'夏普':>6} {'Sortino':>7} {'胜率':>6} {'交易数':>6} {'平均持仓':>7} {'退出原因'}")
print("-" * 130)
for r in results_by_sharpe:
    exits = r['exit_reasons']
    exit_str = ', '.join([f"{k}:{v}" for k, v in exits.items()]) if exits else '-'
    if len(exit_str) > 30:
        exit_str = exit_str[:27] + '...'
    print(f"{r['name']:<45} {r['annual_return']:>+6.1%} {r['max_drawdown']:>7.1%} {r['avg_drawdown']:>7.1%} {r['sharpe']:>6.2f} {r['sortino']:>7.2f} {r['win_rate']:>5.1%} {r['n_trades']:>6} {r['avg_hold_days']:>6.1f} {exit_str}")

# === 核心分析 ===
print("\n" + "=" * 120)
print("关键分析")
print("=" * 120)

# 1. 模型层：平均回撤对比
print("\n📊 模型层分析 — 平均回撤（选股质量）")
print("-" * 80)
single_model_results = [r for r in results if 'Top-15 7d' in r['name'] and '集成' not in r['name'] and '市场' not in r['name']]
for r in sorted(single_model_results, key=lambda x: x['avg_drawdown'], reverse=True):
    quality = "🟢" if r['avg_drawdown'] > -0.03 else "🟡" if r['avg_drawdown'] > -0.05 else "🔴"
    print(f"  {quality} {r['name']:<30} 平均DD: {r['avg_drawdown']:.2%} | 最大DD: {r['max_drawdown']:.1%} | 夏普: {r['sharpe']:.2f}")

# 2. 集成效果
print("\n📊 集成效果分析")
print("-" * 80)
for r in [r for r in results if 'Top-15 7d' in r['name']]:
    print(f"  {r['name']:<40} 夏普: {r['sharpe']:.2f} | 平均DD: {r['avg_drawdown']:.2%}")

# 3. 决策层：止损/Trailing效果
print("\n📊 决策层分析 — 止损/Trailing效果")
print("-" * 80)
decision_results = [r for r in results if '加权集成' in r['name'] and 'Top-15' in r['name']]
for r in sorted(decision_results, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:<45} 夏普: {r['sharpe']:.2f} | 最大DD: {r['max_drawdown']:.1%} | 平均DD: {r['avg_drawdown']:.1%}")

# 4. 市场择时效果
print("\n📊 市场择时效果")
print("-" * 80)
timing_results = [r for r in results if '市场择时' in r['name']]
for r in sorted(timing_results, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:<45} 夏普: {r['sharpe']:.2f} | 最大DD: {r['max_drawdown']:.1%} | 平均DD: {r['avg_drawdown']:.1%}")

# 5. 持有期分析
print("\n📊 持有期分析")
print("-" * 80)
hold_results = [r for r in results if '加权集成 Top-15' in r['name'] and '市场' not in r['name'] and 'SL' not in r['name'] and 'TS' not in r['name']]
for r in sorted(hold_results, key=lambda x: x['sharpe'], reverse=True):
    print(f"  {r['name']:<45} 夏普: {r['sharpe']:.2f} | 平均DD: {r['avg_drawdown']:.1%} | 胜率: {r['win_rate']:.1%}")

# === 最优方案推荐 ===
print("\n" + "=" * 120)
print("🏆 最优方案推荐（按夏普×收益/回撤效率）")
print("=" * 120)

# 综合评分：夏普 * (1 + 平均回撤/最大回撤) — 平均回撤越小越好
for r in results:
    r['efficiency'] = r['sharpe'] * (1 + r['avg_drawdown'])  # 平均DD越小(负)，效率越低

results_by_eff = sorted(results, key=lambda x: x['efficiency'], reverse=True)
for i, r in enumerate(results_by_eff[:10]):
    print(f"\n  #{i+1} {r['name']}")
    print(f"      年化: {r['annual_return']:+.1%} | 夏普: {r['sharpe']:.2f} | 最大DD: {r['max_drawdown']:.1%} | 平均DD: {r['avg_drawdown']:.2%}")
    print(f"      胜率: {r['win_rate']:.1%} | 盈亏比: {r['avg_win']/max(r['avg_loss'],0.001):.2f} | 交易数: {r['n_trades']}")
    exits = r['exit_reasons']
    if exits:
        print(f"      退出: {exits}")

# 保存结果
output = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'total_experiments': len(experiments),
    'top_10_by_sharpe': [
        {
            'name': r['name'],
            'annual_return': r['annual_return'],
            'max_drawdown': r['max_drawdown'],
            'avg_drawdown': r['avg_drawdown'],
            'sharpe': r['sharpe'],
            'sortino': r['sortino'],
            'win_rate': r['win_rate'],
            'n_trades': r['n_trades'],
            'exit_reasons': r['exit_reasons'],
        }
        for r in results_by_sharpe[:10]
    ],
    'top_10_by_efficiency': [
        {
            'name': r['name'],
            'annual_return': r['annual_return'],
            'max_drawdown': r['max_drawdown'],
            'avg_drawdown': r['avg_drawdown'],
            'sharpe': r['sharpe'],
            'efficiency': r['efficiency'],
        }
        for r in results_by_eff[:10]
    ],
}

with open(f'{OUTPUT_DIR}/v4_deep_research_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存 → analysis/v4_deep_research_results.json")
print(f"\n总耗时: {time.time()-t0:.1f}s")
print("=" * 90)
