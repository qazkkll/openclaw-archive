#!/usr/bin/env python3
"""
rule-alpha-v2.0 探索
方向1: 2019年Alpha负原因分析
方向2: 自适应因子权重（根据市场状态调整）
方向3: 改进选股条件（增加大单净流入过滤）
"""
import pandas as pd, numpy as np, json, time, os, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v2.0 探索实验")
print("="*60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1] 加载数据...")
t0 = time.time()

df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# 特征
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net', 'md_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
market_avg_r20 = df.groupby('date')['mkt_ret20'].first()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()

market_state_map = {}
for d in sorted(df['date'].unique()):
    r20 = market_avg_r20.get(d, 0) if d in market_avg_r20.index else 0
    ma60 = market_ma60.get(d, 0) if d in market_ma60.index else 0
    ma120 = market_ma120.get(d, 0) if d in market_ma120.index else 0
    ma_bull = ma60 > ma120
    mom_pos = r20 > 0
    if not ma_bull and not mom_pos:
        market_state_map[d] = 'bear'
    elif not ma_bull or not mom_pos:
        market_state_map[d] = 'cautious'
    else:
        market_state_map[d] = 'bull'

all_dates = sorted(df['date'].unique())
print(f"  {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

# ============================================================
# 2. 评分函数变体
# ============================================================
def score_v1(day):
    """v1.0 原始评分"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

def score_v2_flow_boost(day):
    """v2.0: 增强资金流权重，降低RSI权重"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 2.5
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 3  # +50%
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1  # -33%
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 2  # +100%
    s['score'] += s['md_net_5d'].fillna(0).rank(pct=True) * 1  # 新增中单
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

def score_v3_adaptive(day, market_state):
    """v3.0: 市场状态自适应因子权重"""
    s = day.copy()
    s['score'] = 0.0
    
    if market_state == 'bull':
        # 牛市：增加动量权重，减少反转
        s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 2
        s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 3
        s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 1.5
        s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1
        s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 2
        s['score'] += s['md_net_5d'].fillna(0).rank(pct=True) * 1
        s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1.5
    elif market_state == 'cautious':
        # 谨慎：增加低波动和反转权重
        s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
        s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
        s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2.5
        s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
        s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
        s['score'] += s['md_net_5d'].fillna(0).rank(pct=True) * 1
        s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    else:
        # 熊市：空仓（不评分）
        pass
    return s

def score_v4_value_filter(day):
    """v4.0: 增加大单净流入过滤条件"""
    s = day.copy()
    s['score'] = 0.0
    # 大单净流入必须为正
    s = s[s['lg_net_5d'].fillna(0) > 0].copy()
    if len(s) == 0:
        return s
    
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

def score_v5_combined(day, market_state):
    """v5.0: 综合改进 - 增强资金流 + 大单过滤 + 自适应"""
    s = day.copy()
    # 大单净流入过滤
    s = s[s['lg_net_5d'].fillna(0) > 0].copy()
    if len(s) == 0:
        return s
    
    s['score'] = 0.0
    
    if market_state == 'bull':
        s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 2
        s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 3
        s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 1.5
        s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1
        s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 2
        s['score'] += s['md_net_5d'].fillna(0).rank(pct=True) * 1
        s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1.5
    elif market_state == 'cautious':
        s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
        s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
        s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2.5
        s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
        s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
        s['score'] += s['md_net_5d'].fillna(0).rank(pct=True) * 1
        s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    else:
        pass
    return s

# ============================================================
# 3. 回测函数（精确版）
# ============================================================
def run_backtest(df_all, score_fn, test_start=20200101, test_end=20260616, 
                  hold_days=10, top_n=15, stop_loss=-0.01, cost=0.003,
                  use_market_filter=True, adaptive=False):
    """精确回测"""
    
    df_test = df_all[(df_all['date'] >= test_start) & (df_all['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    price_dict = {}
    for d in test_dates:
        day_data = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(day_data['sym'], day_data['close']))
    
    rebal_dates = test_dates[::hold_days]
    
    equity = 100000.0
    equity_curve = [(test_dates[0], equity)]
    trades = []
    
    for i, rd in enumerate(rebal_dates):
        if use_market_filter:
            state = market_state_map.get(rd, 'bull')
            if state == 'bear':
                next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
                for d in test_dates:
                    if rd < d <= next_rd:
                        equity_curve.append((d, equity))
                continue
            elif state in ['cautious']:
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        
        if adaptive:
            state = market_state_map.get(rd, 'bull')
            day = score_fn(day, state)
        else:
            day = score_fn(day)
        
        if len(day) == 0:
            next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
            for d in test_dates:
                if rd < d <= next_rd:
                    equity_curve.append((d, equity))
            continue
        
        picks = day.nlargest(top_n, 'score')
        
        entry_prices = {}
        for _, row in picks.iterrows():
            entry_prices[row['sym']] = row['close']
        
        equity *= (1 - cost * position_pct)
        
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
        hold_dates = [d for d in test_dates if rd < d <= next_rd]
        
        active_syms = set(entry_prices.keys())
        prev_day_prices = {sym: entry_prices[sym] for sym in active_syms}
        
        for hd in hold_dates:
            curr_prices = price_dict.get(hd, {})
            
            daily_port_ret = 0.0
            n_active = len(active_syms)
            if n_active == 0:
                equity_curve.append((hd, equity))
                continue
            
            weight_per_stock = position_pct / n_active
            stopped_out = []
            
            for sym in list(active_syms):
                if sym not in curr_prices:
                    continue
                    
                curr_p = curr_prices[sym]
                entry_p = entry_prices[sym]
                prev_p = prev_day_prices.get(sym, entry_p)
                
                cum_ret = curr_p / entry_p - 1
                
                if stop_loss is not None and cum_ret <= stop_loss:
                    prev_cum = prev_p / entry_p - 1
                    if prev_cum <= stop_loss:
                        day_ret = 0
                    else:
                        day_ret = stop_loss - prev_cum
                        stopped_out.append(sym)
                else:
                    day_ret = curr_p / prev_p - 1 if prev_p > 0 else 0
                
                daily_port_ret += day_ret * weight_per_stock
                prev_day_prices[sym] = curr_p
            
            equity *= (1 + daily_port_ret)
            equity_curve.append((hd, equity))
            
            for sym in stopped_out:
                active_syms.discard(sym)
        
        equity *= (1 - cost * position_pct)
        
        for sym, entry_p in entry_prices.items():
            exit_p = price_dict.get(next_rd, {}).get(sym, entry_p)
            ret = exit_p / entry_p - 1
            if stop_loss is not None and ret < stop_loss:
                ret = stop_loss
            trades.append({'sym': sym, 'date': rd, 'return': ret - cost})
    
    return trades, equity_curve

def calc_metrics(trades, eq_curve):
    """计算指标"""
    eq_arr = np.array([e[1] for e in eq_curve])
    eq_dates = np.array([e[0] for e in eq_curve])
    
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min()
    
    dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
    dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
    years = (dt2 - dt1).days / 365.25
    total_ret = eq_arr[-1] / eq_arr[0] - 1
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    ann_ret = daily_rets.mean() * 252
    ann_std = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    downside = daily_rets[daily_rets < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = ann_ret / (downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
    trade_rets = np.array([t['return'] for t in trades])
    win_rate = (trade_rets > 0).mean() if len(trade_rets) > 0 else 0
    
    return {
        'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'win_rate': win_rate, 'trades': len(trades), 'final_equity': eq_arr[-1],
    }

# ============================================================
# 4. 运行实验
# ============================================================
print("\n[2] 运行v2.0实验...")

configs = [
    ('v1.0_原始', score_v1, False, -0.01, 0.003, True),
    ('v2.0_资金流增强', score_v2_flow_boost, False, -0.01, 0.003, True),
    ('v3.0_自适应权重', score_v3_adaptive, True, -0.01, 0.003, True),
    ('v4.0_大单过滤', score_v4_value_filter, False, -0.01, 0.003, True),
    ('v5.0_综合改进', score_v5_combined, True, -0.01, 0.003, True),
    # 无市场过滤对照
    ('v1.0_无MF', score_v1, False, -0.01, 0.003, False),
    ('v5.0_无MF', score_v5_combined, True, -0.01, 0.003, False),
    # 不同止损
    ('v5.0_SL0%', score_v5_combined, True, None, 0.003, True),
    ('v5.0_SL2%', score_v5_combined, True, -0.02, 0.003, True),
    ('v5.0_SL3%', score_v5_combined, True, -0.03, 0.003, True),
    # 不同持有期
    ('v5.0_5d', score_v5_combined, True, -0.01, 0.003, True),
    ('v5.0_15d', score_v5_combined, True, -0.01, 0.003, True),
    ('v5.0_20d', score_v5_combined, True, -0.01, 0.003, True),
]

results = []
for name, score_fn, adaptive, sl, cost, mf in configs:
    print(f"  {name}...", end='', flush=True)
    t1 = time.time()
    
    trades, eq_curve = run_backtest(df, score_fn, 
                                     stop_loss=sl, cost=cost,
                                     use_market_filter=mf, adaptive=adaptive)
    metrics = calc_metrics(trades, eq_curve)
    metrics['name'] = name
    metrics['time'] = time.time() - t1
    results.append(metrics)
    
    print(f" Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']:.1%} DD={metrics['max_dd']:.1%} {metrics['time']:.0f}s")

# ============================================================
# 5. 2019年深度分析
# ============================================================
print("\n[3] 2019年Alpha负原因分析...")

# 2019年月度分析
df_2019 = df[(df['date'] >= 20190101) & (df['date'] <= 20191231)]
dates_2019 = sorted(df_2019['date'].unique())

# 月度市场状态
monthly_states = {}
for d in dates_2019:
    month = d // 100
    state = market_state_map.get(d, 'bull')
    if month not in monthly_states:
        monthly_states[month] = {'bull': 0, 'cautious': 0, 'bear': 0}
    monthly_states[month][state] += 1

print("2019年月度市场状态:")
for month in sorted(monthly_states.keys()):
    states = monthly_states[month]
    dominant = max(states, key=states.get)
    print(f"  {month}: {dominant} (bull={states['bull']}, cautious={states['cautious']}, bear={states['bear']})")

# 2019年选股特征分析
print("\n2019年Top15选股特征均值:")
dates_to_check = [d for d in dates_2019 if d % 10 == 0][:12]  # 每月取一个
for d in dates_to_check[:6]:
    day = df_2019[df_2019['date'] == d].copy()
    day = score_v1(day)
    if len(day) > 0:
        top15 = day.nlargest(15, 'score')
        print(f"  {d}: ret20={top15['ret20'].mean():.3f}, rsi={top15['rsi_14'].mean():.1f}, "
              f"total_net_5d={top15['total_net_5d'].mean():.0f}, vol20={top15['vol20'].mean():.4f}")

# ============================================================
# 6. 结果汇总
# ============================================================
print("\n" + "="*80)
print("📊 rule-alpha-v2.0 实验结果")
print("="*80)

results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
print(f"\n{'配置':<20} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'胜率':>7} {'交易':>6}")
print("-"*70)
for r in results_sorted:
    print(f"{r['name']:<20} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1%} "
          f"{r['win_rate']:>6.1%} {r['trades']:>6}")

# 最佳配置
best = results_sorted[0]
print(f"\n🏆 最佳配置: {best['name']}")
print(f"  CAGR: {best['cagr']:.2%}")
print(f"  Sharpe: {best['sharpe']:.2f}")
print(f"  Sortino: {best['sortino']:.2f}")
print(f"  MaxDD: {best['max_dd']:.2%}")

# ============================================================
# 7. 保存结果
# ============================================================
output = {
    'experiment': 'rule-alpha-v2.0',
    'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    'configs': configs,
    'results': results,
    'best': best,
}

with open('research/rule_alpha_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存: research/rule_alpha_v2_results.json")
print("="*60)
print("CEO决策: v2.0实验完成")
print("="*60)
