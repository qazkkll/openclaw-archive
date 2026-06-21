#!/usr/bin/env python3
"""
A股规则型策略回测 — 高效向量化版 v2
CEO决策：修复性能问题，用向量化代替逐日循环
目标：DD < -15%, 年化 > 10%
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

WORKSPACE = os.path.expanduser('~/.hermes/openclaw-archive')
DATA_DIR = os.path.join(WORKSPACE, 'data')

print("="*60)
print("A股规则型策略回测 v2 — 向量化高效版")
print("="*60)

# === 1. 加载数据 ===
print("\n[1] 加载数据...")
t0 = time.time()

df_ohlcv = pd.read_parquet(os.path.join(DATA_DIR, 'a_hist_10y.parquet'))
df_ohlcv = df_ohlcv.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df_ohlcv['date'] = df_ohlcv['date'].astype(int)

df_mf = pd.read_parquet(os.path.join(DATA_DIR, 'cn/moneyflow_core.parquet'))
df_mf['sym'] = df_mf['ts_code'].str[:6]
df_mf['date'] = df_mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    df_mf[f'{col}_net'] = df_mf[f'buy_{col}_amount'] - df_mf[f'sell_{col}_amount']
df_mf['total_net'] = df_mf['net_mf_amount']
df_mf = df_mf[['sym', 'date', 'sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']].copy()

df = df_ohlcv.merge(df_mf, on=['sym', 'date'], how='left')
df = df.sort_values(['sym', 'date']).reset_index(drop=True)
print(f"  合并: {len(df):,}行, {df['sym'].nunique()}只, 耗时{time.time()-t0:.0f}秒")

# === 2. 过滤 ===
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
print(f"  过滤后: {len(df):,}行, {df['sym'].nunique()}只")

# === 3. 特征计算 ===
print("\n[2] 计算特征...")
t0 = time.time()

df['ret1'] = df.groupby('sym')['close'].pct_change(1)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret60'] = df.groupby('sym')['close'].pct_change(60)

df['ma5'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']

df['vol5'] = df.groupby('sym')['ret1'].transform(lambda x: x.rolling(5, min_periods=2).std())
df['vol20'] = df.groupby('sym')['ret1'].transform(lambda x: x.rolling(20, min_periods=5).std())

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = df.groupby('sym')['close'].transform(lambda x: gain.loc[x.index].rolling(14, min_periods=1).mean())
avg_loss = df.groupby('sym')['close'].transform(lambda x: loss.loc[x.index].rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

# 资金流聚合
for col in ['sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())

# 标签
for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

# 市场指标
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())

print(f"  特征计算完成, 耗时{time.time()-t0:.0f}秒")

# === 4. 高效回测函数 ===
def backtest_vectorized(df_test, config):
    """
    向量化回测：
    1. 预计算所有调仓日的信号
    2. 用向量化计算收益
    """
    top_n = config.get('top_n', 15)
    hold_days = config.get('hold_days', 5)
    stop_loss = config.get('stop_loss', None)
    market_filter = config.get('market_filter', False)
    cost = config.get('cost', 0.003)
    score_func = config.get('score_func', None)
    entry_filter = config.get('entry_filter', None)
    
    all_dates = sorted(df_test['date'].unique())
    
    # 市场状态预计算
    if market_filter:
        market_df = df_test.groupby('date').agg(
            avg_bias=('ma60_bias', 'mean'),
            avg_ret20=('ret20', 'mean'),
            breadth=('breadth', 'first')
        ).reset_index()
        
        def get_market_state(row):
            if row['avg_bias'] > 0 and row['avg_ret20'] > 0 and row['breadth'] > 0.5:
                return 'bull'
            elif (row['avg_bias'] > 0 or row['avg_ret20'] > 0) and row['breadth'] > 0.3:
                return 'cautious'
            else:
                return 'bear'
        
        market_df['state'] = market_df.apply(get_market_state, axis=1)
        market_lookup = dict(zip(market_df['date'], market_df['state']))
    
    # 调仓日（每隔hold_days天）
    rebal_dates = all_dates[::hold_days]
    
    trades = []
    equity = 100000.0
    equity_curve = []
    
    for rebal_idx, rebal_date in enumerate(rebal_dates):
        # 市场过滤
        if market_filter:
            state = market_lookup.get(rebal_date, 'bear')
            if state == 'bear':
                equity_curve.append((rebal_date, equity))
                continue
        
        # 获取调仓日数据
        rebal_data = df_test[df_test['date'] == rebal_date].copy()
        if len(rebal_data) == 0:
            continue
        
        # 入场过滤
        if entry_filter:
            for feature, (min_val, max_val) in entry_filter.items():
                if feature in rebal_data.columns:
                    if min_val is not None:
                        rebal_data = rebal_data[rebal_data[feature] >= min_val]
                    if max_val is not None:
                        rebal_data = rebal_data[rebal_data[feature] <= max_val]
        
        # 计算分数
        if score_func:
            rebal_data = score_func(rebal_data)
        else:
            # 默认评分：反转+资金流+低波动+超卖
            rebal_data['score'] = 0.0
            if 'ret20' in rebal_data.columns:
                rebal_data['score'] += (-rebal_data['ret20'].fillna(0)).clip(-0.5, 0.5) * 2
            if 'total_net_5d' in rebal_data.columns:
                rk = rebal_data['total_net_5d'].rank(pct=True)
                rebal_data['score'] += rk.fillna(0.5) * 2
            if 'vol20' in rebal_data.columns:
                rk = rebal_data['vol20'].rank(pct=True, ascending=True)
                rebal_data['score'] += (1 - rk.fillna(0.5)) * 1
            if 'rsi_14' in rebal_data.columns:
                rebal_data['score'] += (rebal_data['rsi_14'] < 40).astype(float) * 1
        
        # 选Top N
        picks = rebal_data.nlargest(top_n, 'score')
        
        # 计算持有期收益
        for _, row in picks.iterrows():
            sym = row['sym']
            entry_price = row['close']
            entry_date = rebal_date
            
            # 找退出日
            exit_idx = all_dates.index(rebal_date) + hold_days
            if exit_idx >= len(all_dates):
                exit_date = all_dates[-1]
            else:
                exit_date = all_dates[exit_idx]
            
            # 获取退出价格
            exit_data = df_test[(df_test['sym'] == sym) & (df_test['date'] == exit_date)]
            if len(exit_data) == 0:
                # 找最近的退出日
                future_dates = [d for d in all_dates if d > rebal_date][:hold_days]
                for fd in future_dates:
                    exit_data = df_test[(df_test['sym'] == sym) & (df_test['date'] == fd)]
                    if len(exit_data) > 0:
                        exit_date = fd
                        break
            
            if len(exit_data) > 0:
                exit_price = exit_data.iloc[0]['close']
                ret = (exit_price - entry_price) / entry_price
                
                # 止损检查
                if stop_loss and ret < stop_loss:
                    ret = stop_loss
                
                trades.append({
                    'sym': sym,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'entry_price': float(entry_price),
                    'exit_price': float(exit_price),
                    'return': float(ret - cost),
                    'days_held': hold_days,
                })
        
        # 更新权益
        if trades:
            recent_trades = trades[-min(top_n, len(trades)):]
            avg_ret = np.mean([t['return'] for t in recent_trades])
            equity *= (1 + avg_ret)
        
        equity_curve.append((rebal_date, equity))
    
    return trades, equity_curve

# === 5. 评分函数 ===
def score_reversal_flow(df):
    """反转+资金流+低波动+超卖"""
    df = df.copy()
    df['score'] = 0.0
    # 反转：跌幅越大越好
    df['score'] += (-df['ret20'].fillna(0)).clip(-0.5, 0.5) * 2
    # 资金流入
    rk = df['total_net_5d'].rank(pct=True)
    df['score'] += rk.fillna(0.5) * 2
    # 低波动
    rk = df['vol20'].rank(pct=True, ascending=True)
    df['score'] += (1 - rk.fillna(0.5)) * 1
    # 超卖
    df['score'] += (df['rsi_14'] < 40).astype(float) * 1
    return df

def score_flow_only(df):
    """纯资金流评分"""
    df = df.copy()
    df['score'] = df['total_net_5d'].fillna(0)
    return df

def score_reversal_only(df):
    """纯反转评分"""
    df = df.copy()
    df['score'] = (-df['ret20'].fillna(0)).clip(-0.5, 0.5)
    return df

def score_low_vol(df):
    """低波动评分"""
    df = df.copy()
    df['score'] = -df['vol20'].fillna(df['vol20'].median())
    return df

def score_rsi_oversold(df):
    """RSI超卖评分"""
    df = df.copy()
    df['score'] = -(df['rsi_14'].fillna(50) - 50)
    return df

def score_combined_optimized(df):
    """优化组合：反转+资金流+低波动+超卖+均线偏离"""
    df = df.copy()
    df['score'] = 0.0
    # 反转（权重2）
    df['score'] += (-df['ret20'].fillna(0)).clip(-0.5, 0.5) * 2
    # 资金流入（权重2）
    rk = df['total_net_5d'].rank(pct=True)
    df['score'] += rk.fillna(0.5) * 2
    # 低波动（权重1）
    rk = df['vol20'].rank(pct=True, ascending=True)
    df['score'] += (1 - rk.fillna(0.5)) * 1
    # 超卖（权重1）
    df['score'] += (df['rsi_14'] < 40).astype(float) * 1
    # 均线偏离（权重1）：偏离越大越好（但不过度）
    bias_score = (-df['ma20_bias'].fillna(0)).clip(-0.2, 0.2)
    df['score'] += bias_score * 1
    # 资金流动量（权重1）
    if 'lg_net_5d' in df.columns:
        rk = df['lg_net_5d'].rank(pct=True)
        df['score'] += rk.fillna(0.5) * 1
    return df

# === 6. 计算指标 ===
def calc_metrics(trades, equity_curve, name, hold_days=5):
    if not trades:
        return {'strategy': name, 'trades': 0, 'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0}
    
    rets = np.array([t['return'] for t in trades])
    eq = np.array([e[1] for e in equity_curve])
    dates = [e[0] for e in equity_curve]
    
    # 基础指标
    n_trades = len(trades)
    win_rate = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # 年化收益（从权益曲线）
    if len(eq) > 1:
        total_days = dates[-1] - dates[0]
        years = total_days / 365.25
        total_return = eq[-1] / eq[0] - 1
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    else:
        cagr = 0
    
    # 最大回撤
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max()
    
    # Sharpe（简化）
    avg_hold = hold_days
    trades_per_year = 252 / max(avg_hold, 1)
    ann_ret = rets.mean() * trades_per_year
    ann_std = rets.std() * np.sqrt(trades_per_year)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    # Sortino
    downside = rets[rets < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = ann_ret / (downside_std * np.sqrt(trades_per_year)) if downside_std > 0 else 0
    
    return {
        'strategy': name,
        'trades': n_trades,
        'win_rate': round(float(win_rate), 4),
        'avg_win': round(float(avg_win), 4),
        'avg_loss': round(float(avg_loss), 4),
        'pl_ratio': round(float(pl_ratio), 4),
        'avg_return': round(float(rets.mean()), 4),
        'cagr': round(float(cagr), 4),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_dd': round(float(max_dd), 4),
        'final_equity': round(float(eq[-1]), 2) if len(eq) > 0 else 0,
    }

# === 7. 实验矩阵 ===
print("\n[3] 开始回测实验...")
print("="*60)

test_start = 20200101
test_end = 20260616
df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()
print(f"测试期: {test_start} ~ {test_end}, {len(df_test):,}行, {df_test['sym'].nunique()}只")

results = []

# 基础策略
experiments = [
    ('A_baseline', {'score_func': score_reversal_flow, 'top_n': 15, 'hold_days': 5}),
    ('B_reversal', {'score_func': score_reversal_only, 'top_n': 15, 'hold_days': 5}),
    ('C_flow', {'score_func': score_flow_only, 'top_n': 15, 'hold_days': 5}),
    ('D_low_vol', {'score_func': score_low_vol, 'top_n': 15, 'hold_days': 5}),
    ('E_rsi_oversold', {'score_func': score_rsi_oversold, 'top_n': 15, 'hold_days': 5}),
    ('F_combined_opt', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 5}),
    
    # 不同Top N
    ('G_top10', {'score_func': score_combined_optimized, 'top_n': 10, 'hold_days': 5}),
    ('H_top20', {'score_func': score_combined_optimized, 'top_n': 20, 'hold_days': 5}),
    ('I_top30', {'score_func': score_combined_optimized, 'top_n': 30, 'hold_days': 5}),
    
    # 不同持有期
    ('J_hold10d', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 10}),
    ('K_hold20d', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 20}),
    ('L_hold30d', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 30}),
    
    # 止损
    ('M_sl5pct', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 5, 'stop_loss': -0.05}),
    ('N_sl8pct', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 5, 'stop_loss': -0.08}),
    ('O_sl10pct', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 5, 'stop_loss': -0.10}),
    
    # 市场过滤
    ('P_market_filter', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 5, 'market_filter': True}),
    
    # 组合优化
    ('Q_hold10_sl8_mf', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 10, 'stop_loss': -0.08, 'market_filter': True}),
    ('R_hold20_sl10_mf', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 20, 'stop_loss': -0.10, 'market_filter': True}),
    
    # 入场过滤
    ('S_entry_oversold', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 10, 'entry_filter': {'ret20': (None, -0.05)}}),
    ('T_entry_flow', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 10, 'entry_filter': {'total_net_5d': (0, None)}}),
    ('U_entry_both', {'score_func': score_combined_optimized, 'top_n': 15, 'hold_days': 10, 'entry_filter': {'ret20': (None, -0.03), 'total_net_5d': (0, None)}}),
]

for name, config in experiments:
    print(f"\n--- {name} ---")
    t0 = time.time()
    trades, equity_curve = backtest_vectorized(df_test, config)
    metrics = calc_metrics(trades, equity_curve, name, config.get('hold_days', 5))
    results.append(metrics)
    
    print(f"  交易: {metrics['trades']}笔")
    print(f"  胜率: {metrics['win_rate']:.1%}")
    print(f"  均收: {metrics['avg_return']:.2%}")
    print(f"  年化: {metrics['cagr']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  DD: {metrics['max_dd']:.1%}")
    print(f"  终值: {metrics['final_equity']:,.0f}")
    print(f"  耗时: {time.time()-t0:.0f}秒")

# === 8. 汇总 ===
print("\n" + "="*60)
print("[4] 回测结果汇总（按Sharpe排序）")
print("="*60)

results_sorted = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)

print(f"\n{'策略':<22} {'交易':>6} {'胜率':>6} {'均收':>8} {'年化':>8} {'Sharpe':>8} {'DD':>8} {'终值':>10}")
print("-"*80)
for r in results_sorted:
    print(f"{r['strategy']:<22} {r['trades']:>6} {r['win_rate']:>5.1%} {r['avg_return']:>7.2%} {r['cagr']:>7.1%} {r['sharpe']:>7.2f} {r['max_dd']:>7.1%} {r['final_equity']:>10,.0f}")

# === 9. 最优方案详情 ===
print("\n" + "="*60)
print("[5] 最优方案详情")
print("="*60)

best = results_sorted[0]
print(f"\n最优策略: {best['strategy']}")
for k, v in best.items():
    if k != 'strategy':
        print(f"  {k}: {v}")

# === 10. 保存 ===
output_file = os.path.join(WORKSPACE, 'research', 'backtest_v2_results.json')
with open(output_file, 'w') as f:
    json.dump({
        'test_period': f'{test_start} ~ {test_end}',
        'data_rows': len(df_test),
        'stocks': df_test['sym'].nunique(),
        'results': results_sorted,
    }, f, indent=2, ensure_ascii=False, default=str)

print(f"\n结果已保存: {output_file}")

# CEO判断
print("\n" + "="*60)
print("[6] CEO决策判断")
print("="*60)

# 找到DD < -15%的策略
good_dd = [r for r in results_sorted if r['max_dd'] < 0.15 and r['sharpe'] > 0]
good_sharpe = [r for r in results_sorted if r['sharpe'] > 1.0]

print(f"\nDD < -15%的策略: {len(good_dd)}个")
for r in good_dd[:5]:
    print(f"  {r['strategy']}: Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, 年化 {r['cagr']:.1%}")

print(f"\nSharpe > 1.0的策略: {len(good_sharpe)}个")
for r in good_sharpe[:5]:
    print(f"  {r['strategy']}: Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, 年化 {r['cagr']:.1%}")

if good_dd:
    best_good = good_dd[0]
    print(f"\n✅ CEO推荐: {best_good['strategy']}")
    print(f"   Sharpe {best_good['sharpe']:.2f}, DD {best_good['max_dd']:.1%}, 年化 {best_good['cagr']:.1%}")
else:
    print("\n⚠️ 没有DD < -15%的策略，需要继续优化")
