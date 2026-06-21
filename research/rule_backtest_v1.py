#!/usr/bin/env python3
"""
A股规则型策略回测 — 完整框架 v1
CEO决策：先修复回测BUG，再优化策略
目标：DD < -15%, 年化 > 10%
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

WORKSPACE = os.path.expanduser('~/.hermes/openclaw-archive')
DATA_DIR = os.path.join(WORKSPACE, 'data')

print("="*60)
print("A股规则型策略回测 v1 — CEO自主开发")
print("="*60)

# === 1. 加载数据 ===
print("\n[1] 加载数据...")
t0 = time.time()

# OHLCV
df_ohlcv = pd.read_parquet(os.path.join(DATA_DIR, 'a_hist_10y.parquet'))
df_ohlcv = df_ohlcv.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df_ohlcv['date'] = df_ohlcv['date'].astype(int)
print(f"  OHLCV: {len(df_ohlcv):,}行, {df_ohlcv['sym'].nunique()}只")

# 资金流
df_mf = pd.read_parquet(os.path.join(DATA_DIR, 'cn/moneyflow_core.parquet'))
df_mf['sym'] = df_mf['ts_code'].str[:6]
df_mf['date'] = df_mf['trade_date'].astype(int)
# 资金流净额（大单+超大单）
for col in ['sm', 'md', 'lg', 'elg']:
    df_mf[f'{col}_net'] = df_mf[f'buy_{col}_amount'] - df_mf[f'sell_{col}_amount']
df_mf['total_net'] = df_mf['net_mf_amount']
df_mf = df_mf[['sym', 'date', 'sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']].copy()
print(f"  资金流: {len(df_mf):,}行, {df_mf['sym'].nunique()}只")

# 合并
df = df_ohlcv.merge(df_mf, on=['sym', 'date'], how='left')
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  合并后: {len(df):,}行, {df['sym'].nunique()}只")
print(f"  耗时: {time.time()-t0:.0f}秒")

# === 2. 过滤 ===
print("\n[2] 过滤...")
# 过滤ST和退市
df = df[~df['sym'].str.startswith(('688',))].copy()  # 暂时排除科创板（波动大）
# 过滤价格
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
# 过滤成交量
df = df[df['volume'] > 0].copy()
print(f"  过滤后: {len(df):,}行, {df['sym'].nunique()}只")

# === 3. 特征计算（向量化） ===
print("\n[3] 计算特征...")
t0 = time.time()

# 按股票排序
df = df.sort_values(['sym', 'date'])

# 价格变化
df['ret1'] = df.groupby('sym')['close'].pct_change(1)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret60'] = df.groupby('sym')['close'].pct_change(60)

# 均线
df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())

# 偏离均线
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']

# 波动率
df['vol5'] = df.groupby('sym')['ret1'].transform(lambda x: x.rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['ret1'].transform(lambda x: x.rolling(20, min_periods=5).std())

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = df.groupby('sym')['close'].transform(lambda x: gain.loc[x.index].rolling(14, min_periods=1).mean())
avg_loss = df.groupby('sym')['close'].transform(lambda x: loss.loc[x.index].rolling(14, min_periods=1).mean())
# 简化RSI
df['rsi_14'] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

# 资金流聚合
for col in ['sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())

# 资金流动量
df['flow_mom'] = df['lg_net_5d'] - df.groupby('sym')['lg_net_5d'].shift(20) / 4

# 标签（前向收益）
for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

# 市场指标（截面）
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['market_ma60'] = df.groupby('date')['ma60_bias'].transform('mean')
df['market_ret20'] = df.groupby('date')['ret20'].transform('mean')

# 板块标记
df['board'] = '主板'
df.loc[df['sym'].str.startswith('30'), 'board'] = '创业板'
# 科创板已被过滤

print(f"  特征计算完成, 耗时: {time.time()-t0:.0f}秒")
print(f"  有效行（fwd_5d非NaN）: {df['fwd_5d'].notna().sum():,}")

# === 4. 回测函数 ===
def backtest_strategy(df, strategy_name, config):
    """
    规则型策略回测
    config: {
        'entry_rules': {'feature': (min, max), ...},
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None or -0.08,
        'market_filter': None or 'cautious',
        'rebal_freq': 5,
    }
    """
    # 提取参数
    entry_rules = config.get('entry_rules', {})
    top_n = config.get('top_n', 15)
    hold_days = config.get('hold_days', 5)
    stop_loss = config.get('stop_loss', None)
    market_filter = config.get('market_filter', None)
    rebal_freq = config.get('rebal_freq', hold_days)
    cost = config.get('cost', 0.003)  # 双边0.3%
    
    # 获取所有交易日
    all_dates = sorted(df['date'].unique())
    
    # 市场状态判断
    if market_filter:
        market_state = {}
        for d in all_dates:
            day_data = df[df['date'] == d]
            if len(day_data) == 0:
                continue
            avg_bias = day_data['ma60_bias'].mean()
            avg_ret20 = day_data['ret20'].mean()
            br = day_data['breadth'].mean()
            
            # Bull: 均线向上 + 动量正 + 宽度>50%
            bull = avg_bias > 0 and avg_ret20 > 0 and br > 0.5
            # Cautious: 部分满足
            cautious = (avg_bias > 0 or avg_ret20 > 0) and br > 0.3
            
            if bull:
                market_state[d] = 'bull'
            elif cautious:
                market_state[d] = 'cautious'
            else:
                market_state[d] = 'bear'
    
    # 模拟交易
    trades = []
    equity = 100000.0  # 10万起始资金
    equity_curve = [(all_dates[0], equity)]
    positions = []  # [{sym, entry_price, entry_date, entry_idx}]
    
    # 调仓日
    rebal_dates = all_dates[::rebal_freq]
    
    for i, d in enumerate(all_dates):
        # 市场过滤
        if market_filter:
            state = market_state.get(d, 'bear')
            if state == 'bear':
                # 清仓
                for pos in positions[:]:
                    day_data = df[(df['date'] == d) & (df['sym'] == pos['sym'])]
                    if len(day_data) > 0:
                        exit_price = day_data.iloc[0]['close']
                        ret = (exit_price - pos['entry_price']) / pos['entry_price']
                        trades.append({
                            'sym': pos['sym'],
                            'entry_date': pos['entry_date'],
                            'exit_date': d,
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'return': ret - cost,
                            'days_held': i - pos['entry_idx'],
                            'exit_reason': 'market_filter'
                        })
                positions = []
                continue
        
        # 止损检查
        if stop_loss:
            for pos in positions[:]:
                day_data = df[(df['date'] == d) & (df['sym'] == pos['sym'])]
                if len(day_data) > 0:
                    current_price = day_data.iloc[0]['close']
                    ret = (current_price - pos['entry_price']) / pos['entry_price']
                    if ret < stop_loss:
                        trades.append({
                            'sym': pos['sym'],
                            'entry_date': pos['entry_date'],
                            'exit_date': d,
                            'entry_price': pos['entry_price'],
                            'exit_price': current_price,
                            'return': stop_loss - cost,
                            'days_held': i - pos['entry_idx'],
                            'exit_reason': 'stop_loss'
                        })
                        positions.remove(pos)
        
        # 持有期到期
        for pos in positions[:]:
            if i - pos['entry_idx'] >= hold_days:
                day_data = df[(df['date'] == d) & (df['sym'] == pos['sym'])]
                if len(day_data) > 0:
                    exit_price = day_data.iloc[0]['close']
                    ret = (exit_price - pos['entry_price']) / pos['entry_price']
                    trades.append({
                        'sym': pos['sym'],
                        'entry_date': pos['entry_date'],
                        'exit_date': d,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'return': ret - cost,
                        'days_held': i - pos['entry_idx'],
                        'exit_reason': 'hold_expire'
                    })
                positions.remove(pos)
        
        # 调仓日选股
        if d in rebal_dates and len(positions) < top_n:
            day_data = df[df['date'] == d].copy()
            
            # 应用入场规则
            for feature, (min_val, max_val) in entry_rules.items():
                if feature in day_data.columns:
                    if min_val is not None:
                        day_data = day_data[day_data[feature] >= min_val]
                    if max_val is not None:
                        day_data = day_data[day_data[feature] <= max_val]
            
            # 排除已有持仓
            held_syms = {p['sym'] for p in positions}
            day_data = day_data[~day_data['sym'].isin(held_syms)]
            
            # 综合评分：反转+资金流+低波动
            day_data['score'] = 0.0
            if 'ret20' in day_data.columns:
                # 反转：跌幅越大越好（A股特性）
                day_data['score'] += (-day_data['ret20'].fillna(0)).clip(-0.5, 0.5) * 2
            if 'total_net_5d' in day_data.columns:
                # 资金流入
                rk = day_data['total_net_5d'].rank(pct=True)
                day_data['score'] += rk.fillna(0.5) * 2
            if 'vol20' in day_data.columns:
                # 低波动
                rk = day_data['vol20'].rank(pct=True, ascending=True)
                day_data['score'] += (1 - rk.fillna(0.5)) * 1
            if 'rsi_14' in day_data.columns:
                # 超卖
                day_data['score'] += ((day_data['rsi_14'] < 40) * 1.0)
            
            # 价格>3
            day_data = day_data[day_data['close'] >= 3]
            
            # 选Top N
            picks = day_data.nlargest(top_n, 'score')
            
            # 买入
            for _, row in picks.iterrows():
                if len(positions) >= top_n:
                    break
                positions.append({
                    'sym': row['sym'],
                    'entry_price': row['close'],
                    'entry_date': d,
                    'entry_idx': i
                })
        
        # 更新权益
        total_value = 0
        for pos in positions:
            day_data = df[(df['date'] == d) & (df['sym'] == pos['sym'])]
            if len(day_data) > 0:
                current_price = day_data.iloc[0]['close']
                ret = (current_price - pos['entry_price']) / pos['entry_price']
                total_value += (pos['entry_price'] * (1 + ret)) / pos['entry_price']
            else:
                total_value += 1  # 持有原价
        
        if positions:
            avg_ret = (total_value / len(positions)) - 1
            current_equity = equity * (1 + avg_ret * len(positions) / top_n)
        else:
            current_equity = equity
        
        equity_curve.append((d, current_equity))
    
    # 清仓剩余持仓
    if positions:
        last_date = all_dates[-1]
        for pos in positions:
            day_data = df[(df['date'] == last_date) & (df['sym'] == pos['sym'])]
            if len(day_data) > 0:
                exit_price = day_data.iloc[0]['close']
                ret = (exit_price - pos['entry_price']) / pos['entry_price']
                trades.append({
                    'sym': pos['sym'],
                    'entry_date': pos['entry_date'],
                    'exit_date': last_date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'return': ret - cost,
                    'days_held': 0,
                    'exit_reason': 'end_of_test'
                })
    
    return trades, equity_curve

# === 5. 计算回测指标 ===
def calc_metrics(trades, equity_curve, strategy_name):
    if not trades:
        return {'strategy': strategy_name, 'trades': 0}
    
    rets = np.array([t['return'] for t in trades])
    
    # 基础指标
    n_trades = len(trades)
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # 权益曲线
    eq = np.array([e[1] for e in equity_curve])
    dates = [e[0] for e in equity_curve]
    
    # 最大回撤
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max()
    
    # 年化收益
    if len(eq) > 1:
        total_days = (dates[-1] - dates[0])
        years = total_days / 365.25
        total_return = eq[-1] / eq[0] - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    else:
        cagr = 0
    
    # Sharpe（简化）
    if len(rets) > 1:
        # 每笔交易的年化收益
        avg_hold = np.mean([t['days_held'] for t in trades]) if trades else 5
        trades_per_year = 252 / max(avg_hold, 1)
        ann_ret = rets.mean() * trades_per_year
        ann_std = rets.std() * np.sqrt(trades_per_year)
        sharpe = ann_ret / ann_std if ann_std > 0 else 0
    else:
        sharpe = 0
    
    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0:
        downside_std = downside.std()
        sortino = ann_ret / (downside_std * np.sqrt(trades_per_year)) if downside_std > 0 else 0
    else:
        sortino = 0
    
    # 胜率
    avg_win_rate = win_rate
    
    # 收益分布
    pct_5 = np.percentile(rets, 5)
    pct_25 = np.percentile(rets, 25)
    pct_75 = np.percentile(rets, 75)
    pct_95 = np.percentile(rets, 95)
    
    # 按退出原因分组
    exit_reasons = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(t['return'])
    
    return {
        'strategy': strategy_name,
        'trades': n_trades,
        'win_rate': round(float(win_rate), 4),
        'avg_win': round(float(avg_win), 4),
        'avg_loss': round(float(avg_loss), 4),
        'pl_ratio': round(float(pl_ratio), 4),
        'avg_return': round(float(rets.mean()), 4),
        'median_return': round(float(np.median(rets)), 4),
        'cagr': round(float(cagr), 4),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_dd': round(float(max_dd), 4),
        'pct_5': round(float(pct_5), 4),
        'pct_25': round(float(pct_25), 4),
        'pct_75': round(float(pct_75), 4),
        'pct_95': round(float(pct_95), 4),
        'exit_reasons': {k: {'count': len(v), 'avg_ret': round(float(np.mean(v)), 4)} for k, v in exit_reasons.items()}
    }

# === 6. 实验矩阵 ===
print("\n[4] 开始回测实验...")
print("="*60)

strategies = {
    'A_baseline': {
        'entry_rules': {},
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None,
        'market_filter': None,
        'rebal_freq': 5,
    },
    'B_reversal_only': {
        'entry_rules': {'ret20': (None, -0.05)},  # 20日跌幅>5%
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None,
        'market_filter': None,
        'rebal_freq': 5,
    },
    'C_flow_only': {
        'entry_rules': {'total_net_5d': (0, None)},  # 5日资金净流入>0
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None,
        'market_filter': None,
        'rebal_freq': 5,
    },
    'D_combined': {
        'entry_rules': {'ret20': (None, -0.03), 'total_net_5d': (0, None)},
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None,
        'market_filter': None,
        'rebal_freq': 5,
    },
    'E_combined_oversold': {
        'entry_rules': {'ret20': (None, -0.05), 'rsi_14': (None, 40)},
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None,
        'market_filter': None,
        'rebal_freq': 5,
    },
    'F_full_rule': {
        'entry_rules': {'ret20': (None, -0.03), 'rsi_14': (None, 50)},
        'top_n': 15,
        'hold_days': 5,
        'stop_loss': None,
        'market_filter': None,
        'rebal_freq': 5,
    },
}

# 测试期：2020-2026（包含牛熊）
test_start = 20200101
test_end = 20260616

df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
print(f"测试期: {test_start} ~ {test_end}, {len(df_test):,}行")

results = []
all_trades = {}

for name, config in strategies.items():
    print(f"\n--- {name} ---")
    t0 = time.time()
    trades, equity_curve = backtest_strategy(df_test, name, config)
    metrics = calc_metrics(trades, equity_curve, name)
    results.append(metrics)
    all_trades[name] = trades
    
    print(f"  交易: {metrics['trades']}笔")
    print(f"  胜率: {metrics['win_rate']:.1%}")
    print(f"  平均收益: {metrics['avg_return']:.2%}")
    print(f"  年化: {metrics['cagr']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']:.1%}")
    print(f"  耗时: {time.time()-t0:.0f}秒")

# === 7. 优化实验：止损+市场过滤 ===
print("\n" + "="*60)
print("[5] 优化实验：止损+市场过滤")
print("="*60)

# 用最佳基础策略（D_combined）做优化
base_config = {
    'entry_rules': {'ret20': (None, -0.03), 'total_net_5d': (0, None)},
    'top_n': 15,
    'hold_days': 5,
    'rebal_freq': 5,
}

optimizations = {
    'G_sl5pct': {**base_config, 'stop_loss': -0.05},
    'H_sl8pct': {**base_config, 'stop_loss': -0.08},
    'I_sl10pct': {**base_config, 'stop_loss': -0.10},
    'J_market_filter': {**base_config, 'market_filter': True},
    'K_hold10d': {**base_config, 'hold_days': 10, 'rebal_freq': 10},
    'L_hold20d': {**base_config, 'hold_days': 20, 'rebal_freq': 20},
    'M_sl8_mf_hold10': {**base_config, 'stop_loss': -0.08, 'market_filter': True, 'hold_days': 10, 'rebal_freq': 10},
    'N_full_opt': {**base_config, 'stop_loss': -0.08, 'market_filter': True, 'hold_days': 10, 'rebal_freq': 10},
}

for name, config in optimizations.items():
    print(f"\n--- {name} ---")
    t0 = time.time()
    trades, equity_curve = backtest_strategy(df_test, name, config)
    metrics = calc_metrics(trades, equity_curve, name)
    results.append(metrics)
    all_trades[name] = trades
    
    print(f"  交易: {metrics['trades']}笔")
    print(f"  胜率: {metrics['win_rate']:.1%}")
    print(f"  平均收益: {metrics['avg_return']:.2%}")
    print(f"  年化: {metrics['cagr']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  最大回撤: {metrics['max_dd']:.1%}")
    print(f"  耗时: {time.time()-t0:.0f}秒")

# === 8. 汇总结果 ===
print("\n" + "="*60)
print("[6] 回测结果汇总")
print("="*60)

# 按Sharpe排序
results_sorted = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)

print(f"\n{'策略':<20} {'交易':>6} {'胜率':>6} {'均收':>8} {'年化':>8} {'Sharpe':>8} {'DD':>8}")
print("-"*70)
for r in results_sorted:
    print(f"{r['strategy']:<20} {r['trades']:>6} {r['win_rate']:>5.1%} {r['avg_return']:>7.2%} {r['cagr']:>7.1%} {r['sharpe']:>7.2f} {r['max_dd']:>7.1%}")

# === 9. 最优方案分析 ===
print("\n" + "="*60)
print("[7] 最优方案详细分析")
print("="*60)

best = results_sorted[0]
print(f"\n最优策略: {best['strategy']}")
print(f"  交易笔数: {best['trades']}")
print(f"  胜率: {best['win_rate']:.1%}")
print(f"  平均收益: {best['avg_return']:.2%}")
print(f"  年化收益: {best['cagr']:.1%}")
print(f"  Sharpe: {best['sharpe']:.2f}")
print(f"  Sortino: {best['sortino']:.2f}")
print(f"  最大回撤: {best['max_dd']:.1%}")
print(f"  盈亏比: {best['pl_ratio']:.2f}")
print(f"  收益分布: 5%={best['pct_5']:.2%}, 25%={best['pct_25']:.2%}, 75%={best['pct_75']:.2%}, 95%={best['pct_95']:.2%}")
print(f"  退出原因: {best['exit_reasons']}")

# === 10. 保存结果 ===
output_file = os.path.join(WORKSPACE, 'research', 'backtest_v1_results.json')
with open(output_file, 'w') as f:
    json.dump({
        'test_period': f'{test_start} ~ {test_end}',
        'data_rows': len(df_test),
        'results': results_sorted,
    }, f, indent=2, ensure_ascii=False, default=str)

print(f"\n结果已保存到: {output_file}")

# 保存详细交易记录
trades_file = os.path.join(WORKSPACE, 'research', 'backtest_v1_trades.json')
with open(trades_file, 'w') as f:
    json.dump({k: v[:100] for k, v in all_trades.items()}, f, indent=2, ensure_ascii=False, default=str)

print(f"交易记录已保存到: {trades_file}")
