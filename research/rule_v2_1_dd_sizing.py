#!/usr/bin/env python3
"""
rule-alpha-v2.1: DD-based position sizing（回撤自适应仓位）

发现: v1.0无MF Sharpe=2.01但DD=-17.7%，有MF Sharpe=1.65但DD=-11.7%
目标: 用DD-based position sizing替代binary market filter，在保持高Sharpe的同时控制DD

方法:
1. 基础策略: v1.0无MF（保持Sharpe 2.01）
2. 叠加DD-based position sizing:
   - DD < -5%: 仓位降到75%
   - DD < -10%: 仓位降到50%
   - DD < -15%: 仓位降到25%
   - DD < -20%: 空仓
"""
import pandas as pd, numpy as np, json, time, os, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v2.1: DD-based Position Sizing")
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

for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态（用于对比）
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
# 2. 评分函数
# ============================================================
def score_v1(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

# ============================================================
# 3. 回测函数 — DD-based position sizing
# ============================================================
def run_backtest_dd_sizing(df_all, test_start=20200101, test_end=20260616,
                            hold_days=10, top_n=15, stop_loss=-0.01, cost=0.003,
                            dd_thresholds=None):
    """
    DD-based position sizing回测
    
    dd_thresholds: list of (dd_level, position_pct) pairs, sorted by dd_level ascending
    e.g., [(-0.05, 0.75), (-0.10, 0.50), (-0.15, 0.25), (-0.20, 0.0)]
    """
    
    df_test = df_all[(df_all['date'] >= test_start) & (df_all['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    price_dict = {}
    for d in test_dates:
        day_data = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(day_data['sym'], day_data['close']))
    
    rebal_dates = test_dates[::hold_days]
    
    equity = 100000.0
    peak_equity = equity
    equity_curve = [(test_dates[0], equity)]
    trades = []
    position_log = []
    
    for i, rd in enumerate(rebal_dates):
        # DD-based position sizing
        current_dd = (equity - peak_equity) / peak_equity
        
        if dd_thresholds is not None:
            position_pct = 1.0  # default
            for dd_level, pct in dd_thresholds:
                if current_dd <= dd_level:
                    position_pct = pct
                    break
        else:
            position_pct = 1.0
        
        # Skip if position is 0
        if position_pct <= 0:
            next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
            for d in test_dates:
                if rd < d <= next_rd:
                    equity_curve.append((d, equity))
            continue
        
        # 选股
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        day = score_v1(day)
        picks = day.nlargest(top_n, 'score')
        
        entry_prices = {}
        for _, row in picks.iterrows():
            entry_prices[row['sym']] = row['close']
        
        # 扣除买入成本
        equity *= (1 - cost * position_pct)
        
        # 持有期
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
            
            # Update peak
            peak_equity = max(peak_equity, equity)
            
            for sym in stopped_out:
                active_syms.discard(sym)
        
        equity *= (1 - cost * position_pct)
        
        # Log position changes
        position_log.append({
            'date': rd, 'dd': current_dd, 'position_pct': position_pct,
            'equity': equity, 'n_stocks': len(active_syms)
        })
        
        for sym, entry_p in entry_prices.items():
            exit_p = price_dict.get(next_rd, {}).get(sym, entry_p)
            ret = exit_p / entry_p - 1
            if stop_loss is not None and ret < stop_loss:
                ret = stop_loss
            trades.append({'sym': sym, 'date': rd, 'return': ret - cost})
    
    return trades, equity_curve, position_log

def calc_metrics(trades, eq_curve):
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
# 4. 实验矩阵
# ============================================================
print("\n[2] 运行DD-based position sizing实验...")

configs = [
    # Baseline: no DD sizing, no MF
    ('baseline_noMF', None),
    # Baseline: market filter (v1.0 original)
    # DD-based sizing variants
    ('DD5_75_DD10_50_DD15_25', [(-0.05, 0.75), (-0.10, 0.50), (-0.15, 0.25), (-0.20, 0.0)]),
    ('DD8_70_DD13_50_DD18_25', [(-0.08, 0.70), (-0.13, 0.50), (-0.18, 0.25), (-0.23, 0.0)]),
    ('DD10_50_DD15_25', [(-0.10, 0.50), (-0.15, 0.25), (-0.20, 0.0)]),
    ('DD10_60_DD15_40_DD20_20', [(-0.10, 0.60), (-0.15, 0.40), (-0.20, 0.20), (-0.25, 0.0)]),
    ('DD7_80_DD12_60_DD17_40_DD22_20', [(-0.07, 0.80), (-0.12, 0.60), (-0.17, 0.40), (-0.22, 0.20), (-0.27, 0.0)]),
    # Aggressive: only cut at deep DD
    ('DD15_50_DD20_0', [(-0.15, 0.50), (-0.20, 0.0)]),
    # Conservative: cut early
    ('DD3_80_DD6_60_DD10_40_DD14_20', [(-0.03, 0.80), (-0.06, 0.60), (-0.10, 0.40), (-0.14, 0.20), (-0.18, 0.0)]),
    # Hybrid: DD-based + market filter
    # (implemented as DD-based with additional market state adjustment)
]

results = []
for name, dd_thresholds in configs:
    print(f"  {name}...", end='', flush=True)
    t1 = time.time()
    
    trades, eq_curve, pos_log = run_backtest_dd_sizing(df, dd_thresholds=dd_thresholds)
    metrics = calc_metrics(trades, eq_curve)
    metrics['name'] = name
    metrics['time'] = time.time() - t1
    
    # Count how many times position was reduced
    if pos_log:
        reduced = sum(1 for p in pos_log if p['position_pct'] < 1.0)
        metrics['pct_reduced'] = reduced / len(pos_log) * 100
    else:
        metrics['pct_reduced'] = 0
    
    results.append(metrics)
    print(f" Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']:.1%} DD={metrics['max_dd']:.1%} {metrics['time']:.0f}s")

# ============================================================
# 5. 结果汇总
# ============================================================
print("\n" + "="*80)
print("📊 DD-based Position Sizing 实验结果")
print("="*80)

results_sorted = sorted(results, key=lambda x: x['sharpe'], reverse=True)
print(f"\n{'配置':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'胜率':>7} {'减仓%':>6}")
print("-"*85)
for r in results_sorted:
    print(f"{r['name']:<35} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['sortino']:>8.2f} {r['max_dd']:>7.1%} "
          f"{r['win_rate']:>6.1%} {r['pct_reduced']:>5.1f}%")

# 找出Pareto最优
print("\n📊 Pareto分析 (Sharpe vs DD):")
pareto = []
for r in sorted(results, key=lambda x: x['max_dd'], reverse=True):
    if not pareto or r['sharpe'] > max(p['sharpe'] for p in pareto):
        pareto.append(r)
        print(f"  ⭐ {r['name']}: Sharpe={r['sharpe']:.2f}, DD={r['max_dd']:.1%}")

# 最佳配置
best = results_sorted[0]
print(f"\n🏆 最佳配置: {best['name']}")
print(f"  CAGR: {best['cagr']:.2%}")
print(f"  Sharpe: {best['sharpe']:.2f}")
print(f"  Sortino: {best['sortino']:.2f}")
print(f"  MaxDD: {best['max_dd']:.2%}")

# ============================================================
# 6. 保存结果
# ============================================================
output = {
    'experiment': 'rule-alpha-v2.1-dd-sizing',
    'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    'results': results,
    'best': best,
}

with open('research/rule_alpha_v2_1_dd_sizing.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存: research/rule_alpha_v2_1_dd_sizing.json")
print("="*60)
print("CEO决策: v2.1 DD-based sizing实验完成")
print("="*60)
