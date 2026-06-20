#!/usr/bin/env python3
"""
纯规则型策略优化 — CEO决策
目标：DD降到-15%以下，年化保持10%+
优化方向：
1. 止损规则对比（-5%/-8%/-10%）
2. 市场过滤器（熊市减仓/空仓）
3. 因子权重和阈值调整
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

# 清空日志
open('research/rule_optimize_log.txt', 'w').close()
log("=" * 60)
log("纯规则型策略优化 — CEO决策")
log("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
log("\n[1] 加载数据...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)
df = df.sort_values(['sym', 'date_int'])

log(f"  原始数据: {len(df):,}行, {df['sym'].nunique()}只股票")

# ============================================================
# 2. 计算所需特征
# ============================================================
log("\n[2] 计算特征...")

# 基础特征（已在数据中）
# r5, r10, r20, rsi14, vol20, ma20, close, etc.

# 计算派生特征
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']

# 资金流5日聚合
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

# 计算市场宽度（涨跌家数比）
market_stats = df.groupby('date_int').agg(
    n_stocks=('sym', 'count'),
    n_up=('r5', lambda x: (x > 0).sum()),
    avg_r20=('r20', 'mean'),
    avg_vol=('vol20', 'mean')
).reset_index()

market_stats['breadth'] = market_stats['n_up'] / market_stats['n_stocks']
market_stats['market_ma60'] = market_stats['avg_r20'].rolling(60, min_periods=1).mean()
market_stats['market_ma120'] = market_stats['avg_r20'].rolling(120, min_periods=1).mean()

# 判断市场状态
def get_market_state(row):
    ma_bull = row['market_ma60'] > row['market_ma120']
    mom_pos = row['avg_r20'] > 0
    breadth = row['breadth'] > 0.5
    
    if not ma_bull and not mom_pos:
        return 'bear'
    elif not ma_bull or not mom_pos:
        return 'cautious'
    elif not breadth:
        return 'weak'
    else:
        return 'bull'

market_stats['market_state'] = market_stats.apply(get_market_state, axis=1)
market_state_map = dict(zip(market_stats['date_int'], market_stats['market_state']))

log(f"  市场状态分布:")
for state, count in market_stats['market_state'].value_counts().items():
    log(f"    {state}: {count}天 ({count/len(market_stats)*100:.1f}%)")

# ============================================================
# 4. 规则型策略定义
# ============================================================
log("\n[4] 定义规则型策略...")

def rule_based_score(row, params):
    """
    计算规则型得分
    params: dict with threshold and weight for each factor
    """
    score = 0
    
    # 1. 偏离均线（越负越好，说明超卖）
    if row['ma20_bias'] < params['ma20_bias_threshold']:
        score += params['ma20_bias_weight']
    
    # 2. RSI超卖
    if row['rsi14'] < params['rsi_threshold']:
        score += params['rsi_weight']
    
    # 3. 资金净流入
    if row['total_net_5d'] > params['flow_threshold']:
        score += params['flow_weight']
    
    # 4. 低波动（波动率分位数<阈值）
    if row['vol_20d_pct'] < params['vol_threshold']:
        score += params['vol_weight']
    
    # 5. 长期趋势向上
    if row['mom_60d'] > params['mom_threshold']:
        score += params['mom_weight']
    
    return score

# ============================================================
# 5. 回测函数
# ============================================================
def backtest_strategy(df, params, market_filter=True, stop_loss=None, 
                      top_n=15, hold_days=5, cost=0.0015):
    """
    回测规则型策略
    params: 策略参数
    market_filter: 是否使用市场过滤器
    stop_loss: 止损比例（None表示不止损）
    top_n: 选股数量
    hold_days: 持有天数
    cost: 交易成本（双边）
    """
    # 计算得分
    df = df.copy()
    df['score'] = df.apply(lambda row: rule_based_score(row, params), axis=1)
    
    # 获取所有交易日
    trade_dates = sorted(df['date_int'].unique())
    
    # 每hold_days天调仓一次
    rebal_dates = trade_dates[::hold_days]
    
    portfolio_returns = []
    position_count = 0
    stop_loss_count = 0
    
    for i, rebal_date in enumerate(rebal_dates[:-1]):  # 最后一个不处理
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else trade_dates[-1]
        
        # 市场过滤
        if market_filter:
            market_state = market_state_map.get(rebal_date, 'bull')
            if market_state == 'bear':
                # 熊市空仓
                continue
            elif market_state in ['cautious', 'weak']:
                # 震荡市减半仓位
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        # 选股日数据
        day_data = df[df['date_int'] == rebal_date].copy()
        if len(day_data) < 100:
            continue
        
        # 过滤条件
        day_data = day_data[
            (day_data['close'] >= 3) &  # 价格>3
            (day_data['close'] <= 1000) &  # 排除异常高价
            (~day_data['sym'].str.contains('ST|退市', na=False))  # 排除ST
        ]
        
        if len(day_data) < top_n:
            continue
        
        # 选股：得分最高的top_n只
        selected = day_data.nlargest(top_n, 'score')
        selected_syms = selected['sym'].tolist()
        
        # 持有期收益
        hold_period = df[
            (df['date_int'] > rebal_date) & 
            (df['date_int'] <= next_rebal) &
            (df['sym'].isin(selected_syms))
        ].copy()
        
        if len(hold_period) == 0:
            continue
        
        # 计算每日收益
        for sym in selected_syms:
            sym_data = hold_period[hold_period['sym'] == sym].sort_values('date_int')
            if len(sym_data) < 2:
                continue
            
            # 检查止损
            if stop_loss is not None:
                entry_price = selected[selected['sym'] == sym]['close'].values[0]
                sym_data = sym_data.copy()
                sym_data['cum_ret'] = sym_data['close'] / entry_price - 1
                
                # 如果触发止损，只持有到止损日
                stop_triggered = sym_data[sym_data['cum_ret'] <= stop_loss]
                if len(stop_triggered) > 0:
                    stop_day = stop_triggered.index[0]
                    sym_data = sym_data.loc[:stop_day]
                    stop_loss_count += 1
            
            # 计算收益
            if len(sym_data) >= 2:
                daily_ret = sym_data['close'].pct_change().dropna()
                portfolio_returns.extend((daily_ret * position_pct).tolist())
                position_count += 1
    
    if len(portfolio_returns) == 0:
        return {
            'annual_return': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'n_trades': 0,
            'stop_loss_count': stop_loss_count
        }
    
    # 计算指标
    returns = np.array(portfolio_returns)
    total_return = (1 + returns).prod() - 1
    n_years = len(rebal_dates) * hold_days / 252
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
        'n_trades': position_count,
        'stop_loss_count': stop_loss_count
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
    # 基线（原始策略）
    {
        'name': 'A_baseline',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': None,
        'hold_days': 5,
    },
    # +市场过滤器
    {
        'name': 'B_market_filter',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': None,
        'hold_days': 5,
    },
    # +止损-5%
    {
        'name': 'C_stop_loss_5pct',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': -0.05,
        'hold_days': 5,
    },
    # +止损-8%
    {
        'name': 'D_stop_loss_8pct',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    # +止损-10%
    {
        'name': 'E_stop_loss_10pct',
        'params': base_params.copy(),
        'market_filter': False,
        'stop_loss': -0.10,
        'hold_days': 5,
    },
    # 市场过滤+止损-8%
    {
        'name': 'F_filter_sl8pct',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    # 调整阈值：更宽松的入场条件
    {
        'name': 'G_relaxed_threshold',
        'params': {
            **base_params,
            'ma20_bias_threshold': -0.03,  # 更宽松
            'rsi_threshold': 40,  # 更宽松
            'vol_threshold': 0.4,  # 更宽松
        },
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    # 调整权重：资金流权重更高
    {
        'name': 'H_flow_heavy',
        'params': {
            **base_params,
            'flow_weight': 3,  # 资金流权重提高
            'ma20_bias_weight': 1,  # 均线偏离权重降低
        },
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 5,
    },
    # 延长持有期到10天
    {
        'name': 'I_hold_10d',
        'params': base_params.copy(),
        'market_filter': True,
        'stop_loss': -0.08,
        'hold_days': 10,
    },
    # 延长持有期到20天
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
    log(f"    交易次数: {result['n_trades']}")
    log(f"    止损次数: {result['stop_loss_count']}")
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

# 保存到文件
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
    f.write(f"\n\n## 纯规则型策略优化结果 ({datetime.now().strftime('%H:%M')})\n\n")
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
log("CEO决策：纯规则型策略优化完成")
log("=" * 60)
