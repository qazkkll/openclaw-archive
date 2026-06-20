#!/usr/bin/env python3
"""
纯规则型策略优化 — CEO决策（修正版）
目标：DD降到-15%以下，年化保持10%+
"""
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

def log(msg):
    print(msg, flush=True)
    with open('research/rule_optimize_log.txt', 'a') as f:
        f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")

open('research/rule_optimize_log.txt', 'w').close()
log("=" * 60)
log("纯规则型策略优化（修正版）")
log("=" * 60)

# ============================================================
# 1. 加载数据并采样
# ============================================================
log("\n[1] 加载数据...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
df = df.sort_values(['sym', 'date_int'])

# 采样：每3天取一天
all_dates = sorted(df['date_int'].unique())
sample_dates = all_dates[::3]
df = df[df['date_int'].isin(sample_dates)].copy()

log(f"  采样后: {len(df):,}行, {df['sym'].nunique()}只股票, {len(sample_dates)}天")

# ============================================================
# 2. 计算特征
# ============================================================
log("\n[2] 计算特征...")

# 计算派生特征
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']

# 资金流5日聚合
if 'total_net_5d' not in df.columns:
    for col in ['sm_net', 'md_net', 'lg_net', 'elg_net', 'total_net']:
        df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 60日动量
df['mom_60d'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change(60))

# 波动率分位数
df['vol_20d_pct'] = df.groupby('date_int')['vol20'].rank(pct=True)

# 计算未来收益（用于回测）
for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

log(f"  特征计算完成")

# ============================================================
# 3. 市场状态判断
# ============================================================
log("\n[3] 计算市场状态...")

# 简化市场状态：用市场平均20日收益判断
market_avg_r20 = df.groupby('date_int')['r20'].mean()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()

def get_market_state(date_int):
    r20 = market_avg_r20.get(date_int, 0)
    ma60 = market_ma60.get(date_int, 0)
    ma120 = market_ma120.get(date_int, 0)
    
    ma_bull = ma60 > ma120
    mom_pos = r20 > 0
    
    if not ma_bull and not mom_pos:
        return 'bear'
    elif not ma_bull or not mom_pos:
        return 'cautious'
    else:
        return 'bull'

market_state_map = {d: get_market_state(d) for d in sample_dates}

log(f"  市场状态分布:")
states = pd.Series(market_state_map).value_counts()
for state, count in states.items():
    log(f"    {state}: {count}天 ({count/len(states)*100:.1f}%)")

# ============================================================
# 4. 规则型策略定义（向量化）
# ============================================================
log("\n[4] 定义规则型策略...")

def rule_based_score_vectorized(df, params):
    """向量化计算规则型得分"""
    score = pd.Series(0, index=df.index)
    
    # 1. 偏离均线（越负越好，说明超卖）
    score += (df['ma20_bias'] < params['ma20_bias_threshold']).astype(int) * params['ma20_bias_weight']
    
    # 2. RSI超卖
    score += (df['rsi14'] < params['rsi_threshold']).astype(int) * params['rsi_weight']
    
    # 3. 资金净流入
    score += (df['total_net_5d'] > params['flow_threshold']).astype(int) * params['flow_weight']
    
    # 4. 低波动（波动率分位数<阈值）
    score += (df['vol_20d_pct'] < params['vol_threshold']).astype(int) * params['vol_weight']
    
    # 5. 长期趋势向上
    score += (df['mom_60d'] > params['mom_threshold']).astype(int) * params['mom_weight']
    
    return score

# ============================================================
# 5. 回测函数（修正版 - 正确的组合收益计算）
# ============================================================
def backtest_strategy(df, params, market_filter=True, stop_loss=None, 
                      top_n=15, hold_days=5, cost=0.0015):
    """
    回测规则型策略（修正版）
    正确计算组合收益：每日组合收益 = 持仓股票收益的等权平均
    """
    df = df.copy()
    
    # 向量化计算得分
    df['score'] = rule_based_score_vectorized(df, params)
    
    # 过滤条件
    df = df[
        (df['close'] >= 3) &  # 价格>3
        (df['close'] <= 1000) &  # 排除异常高价
        (~df['sym'].str.contains('ST|退市', na=False))  # 排除ST
    ]
    
    # 获取所有交易日
    trade_dates = sorted(df['date_int'].unique())
    
    # 每hold_days天调仓一次
    rebal_dates = trade_dates[::hold_days]
    
    # 存储每日组合收益
    daily_portfolio_returns = []
    
    for i, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else trade_dates[-1]
        
        # 市场过滤
        if market_filter:
            market_state = market_state_map.get(rebal_date, 'bull')
            if market_state == 'bear':
                # 熊市空仓，收益为0
                continue
            elif market_state in ['cautious', 'weak']:
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        # 选股日数据
        day_data = df[df['date_int'] == rebal_date].copy()
        if len(day_data) < 100:
            continue
        
        # 选股：得分最高的top_n只
        selected = day_data.nlargest(top_n, 'score')
        selected_syms = selected['sym'].tolist()
        
        # 持有期数据
        hold_period = df[
            (df['date_int'] >= rebal_date) & 
            (df['date_int'] <= next_rebal) &
            (df['sym'].isin(selected_syms))
        ].copy()
        
        if len(hold_period) == 0:
            continue
        
        # 计算每日组合收益
        hold_dates = sorted(hold_period['date_int'].unique())
        
        for j, h_date in enumerate(hold_dates[1:], 1):  # 从第二天开始
            prev_date = hold_dates[j-1]
            
            # 获取当天和前一天的持仓数据
            today_data = hold_period[hold_period['date_int'] == h_date]
            prev_data = hold_period[hold_period['date_int'] == prev_date]
            
            # 计算每只股票的收益
            stock_returns = []
            for sym in selected_syms:
                sym_today = today_data[today_data['sym'] == sym]
                sym_prev = prev_data[prev_data['sym'] == sym]
                
                if len(sym_today) > 0 and len(sym_prev) > 0:
                    close_today = sym_today['close'].values[0]
                    close_prev = sym_prev['close'].values[0]
                    
                    if close_prev > 0:
                        ret = close_today / close_prev - 1
                        
                        # 检查止损
                        if stop_loss is not None:
                            entry_price = selected[selected['sym'] == sym]['close'].values[0]
                            cum_ret = close_today / entry_price - 1
                            if cum_ret <= stop_loss:
                                ret = stop_loss  # 止损时收益为止损比例
                        
                        stock_returns.append(ret)
            
            # 组合收益 = 持仓股票收益的等权平均
            if stock_returns:
                port_ret = np.mean(stock_returns) * position_pct
                daily_portfolio_returns.append(port_ret)
    
    if len(daily_portfolio_returns) == 0:
        return {
            'annual_return': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'n_days': 0,
        }
    
    # 计算指标
    returns = np.array(daily_portfolio_returns)
    
    # 年化收益（假设每年252个交易日）
    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Sharpe ratio
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # 最大回撤
    cum_returns = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # 胜率
    win_rate = (returns > 0).mean()
    
    return {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_days': len(returns),
    }

# ============================================================
# 6. 实验矩阵
# ============================================================
log("\n[5] 开始实验矩阵...")

# 基础参数
base_params = {
    'ma20_bias_threshold': -0.05,
    'ma20_bias_weight': 2,
    'rsi_threshold': 30,
    'rsi_weight': 2,
    'flow_threshold': 0,
    'flow_weight': 1,
    'vol_threshold': 0.3,
    'vol_weight': 1,
    'mom_threshold': 0,
    'mom_weight': 1,
}

# 实验配置
experiments = [
    {
        'name': 'A_baseline',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': None,
        'hold_days': 5,
    },
    {
        'name': 'B_market_filter',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': None,
        'hold_days': 5,
    },
    {
        'name': 'C_stop_loss_5pct',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': -0.05,
        'hold_days': 5,
    },
    {
        'name': 'D_stop_loss_8pct',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    {
        'name': 'E_stop_loss_10pct',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': -0.10,
        'hold_days': 5,
    },
    {
        'name': 'F_filter_sl8pct',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    {
        'name': 'G_relaxed_threshold',
        'params': {
            **base_params,
            'ma20_bias_threshold': -0.03,
            'rsi_threshold': 40,
            'vol_threshold': 0.4,
        },
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    {
        'name': 'H_flow_heavy',
        'params': {
            **base_params,
            'flow_weight': 3,
            'ma20_bias_weight': 1,
        },
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    {
        'name': 'I_hold_10d',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 10,
    },
    {
        'name': 'J_hold_20d',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 20,
    },
]

# 运行实验
results = []
for exp in experiments:
    log(f"\n  运行 {exp['name']}...")
    t0 = time.time()
    
    result = backtest_strategy(
        df, 
        exp['params'],
        market_filter=exp['market_filter'],
        stop_loss=exp['stop_loss'],
        hold_days=exp['hold_days']
    )
    
    result['name'] = exp['name']
    result['market_filter'] = exp['market_filter']
    result['stop_loss'] = exp['stop_loss']
    result['hold_days'] = exp['hold_days']
    result['time'] = time.time() - t0
    
    results.append(result)
    
    log(f"    年化: {result['annual_return']*100:.1f}%")
    log(f"    Sharpe: {result['sharpe']:.2f}")
    log(f"    最大回撤: {result['max_drawdown']*100:.1f}%")
    log(f"    胜率: {result['win_rate']*100:.1f}%")
    log(f"    交易天数: {result['n_days']}")
    log(f"    耗时: {result['time']:.1f}s")

# ============================================================
# 7. 结果分析
# ============================================================
log("\n" + "=" * 60)
log("[6] 结果分析")
log("=" * 60)

# 按Sharpe排序
results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)

log("\n按Sharpe排序:")
for r in results_sorted:
    tag = "⭐" if r['sharpe'] > 1.0 else ("✅" if r['sharpe'] > 0.5 else "⚠️" if r['sharpe'] > 0 else "❌")
    log(f"  {tag} {r['name']:<25} Sharpe={r['sharpe']:.2f} 年化={r['annual_return']*100:.1f}% DD={r['max_drawdown']*100:.1f}% 胜率={r['win_rate']*100:.1f}%")

# 找出最优方案
best = results_sorted[0]
log(f"\n🏆 最优方案: {best['name']}")
log(f"  Sharpe: {best['sharpe']:.2f}")
log(f"  年化收益: {best['annual_return']*100:.1f}%")
log(f"  最大回撤: {best['max_drawdown']*100:.1f}%")
log(f"  胜率: {best['win_rate']*100:.1f}%")
log(f"  市场过滤: {best['market_filter']}")
log(f"  止损: {best['stop_loss']}")
log(f"  持有天数: {best['hold_days']}")

# 检查是否达到目标
target_sharpe = 1.0
target_dd = -0.15

log(f"\n目标检查:")
log(f"  Sharpe > {target_sharpe}: {'✅ 达标' if best['sharpe'] > target_sharpe else '❌ 未达标'}")
log(f"  DD > {target_dd*100}%: {'✅ 达标' if best['max_drawdown'] > target_dd else '❌ 未达标'}")

# ============================================================
# 8. 保存结果
# ============================================================
log("\n[7] 保存结果...")

output = {
    'timestamp': datetime.now().isoformat(),
    'experiments': results,
    'best': best,
    'target_sharpe': target_sharpe,
    'target_dd': target_dd,
}

with open('research/rule_optimize_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

log("  结果已保存到 research/rule_optimize_results.json")

# 更新daily archive
log("\n[8] 更新daily archive...")
with open('/home/hermes/.hermes/memory-archive/daily/2026-06-21.md', 'a') as f:
    f.write(f"\n\n## 纯规则型策略优化结果（修正版）({datetime.now().strftime('%H:%M')})\n\n")
    f.write("### 实验结果（按Sharpe排序）\n\n")
    f.write("| 策略 | Sharpe | 年化 | DD | 胜率 | 市场过滤 | 止损 | 持有天数 |\n")
    f.write("|------|--------|------|-----|------|----------|------|----------|\n")
    for r in results_sorted:
        f.write(f"| {r['name']} | {r['sharpe']:.2f} | {r['annual_return']*100:.1f}% | {r['max_drawdown']*100:.1f}% | {r['win_rate']*100:.1f}% | {r['market_filter']} | {r['stop_loss']} | {r['hold_days']} |\n")
    
    f.write(f"\n### 最优方案\n")
    f.write(f"- **策略**: {best['name']}\n")
    f.write(f"- **Sharpe**: {best['sharpe']:.2f}\n")
    f.write(f"- **年化收益**: {best['annual_return']*100:.1f}%\n")
    f.write(f"- **最大回撤**: {best['max_drawdown']*100:.1f}%\n")
    f.write(f"- **胜率**: {best['win_rate']*100:.1f}%\n")
    f.write(f"- **市场过滤**: {best['market_filter']}\n")
    f.write(f"- **止损**: {best['stop_loss']}\n")
    f.write(f"- **持有天数**: {best['hold_days']}\n")
    
    if best['sharpe'] > target_sharpe and best['max_drawdown'] > target_dd:
        f.write(f"\n✅ **达到目标**: Sharpe>{target_sharpe}, DD>{target_dd*100}%\n")
    else:
        f.write(f"\n❌ **未达到目标**: 需要继续优化\n")

log("  daily archive已更新")

log("\n" + "=" * 60)
log("CEO决策：纯规则型策略优化完成（修正版）")
log("=" * 60)
