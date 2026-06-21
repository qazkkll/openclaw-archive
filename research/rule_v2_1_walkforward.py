#!/usr/bin/env python3
"""
rule-alpha-v2.1 Walk-Forward验证
DD3保守配置: DD-3%→80%, DD-6%→60%, DD-10%→40%, DD-14%→20%, DD-18%→空仓
"""
import pandas as pd, numpy as np, json, time, os, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v2.1 Walk-Forward验证")
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
# 3. 回测函数
# ============================================================
def run_backtest(df_all, test_start, test_end, hold_days=10, top_n=15, 
                  stop_loss=-0.01, cost=0.003, dd_thresholds=None):
    """DD-based position sizing回测"""
    
    df_test = df_all[(df_all['date'] >= test_start) & (df_all['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    if len(test_dates) < 20:
        return None
    
    price_dict = {}
    for d in test_dates:
        day_data = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(day_data['sym'], day_data['close']))
    
    rebal_dates = test_dates[::hold_days]
    
    equity = 100000.0
    peak_equity = equity
    equity_curve = [(test_dates[0], equity)]
    trades = []
    
    for i, rd in enumerate(rebal_dates):
        current_dd = (equity - peak_equity) / peak_equity
        
        if dd_thresholds is not None:
            position_pct = 1.0
            for dd_level, pct in dd_thresholds:
                if current_dd <= dd_level:
                    position_pct = pct
                    break
        else:
            position_pct = 1.0
        
        if position_pct <= 0:
            next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
            for d in test_dates:
                if rd < d <= next_rd:
                    equity_curve.append((d, equity))
            continue
        
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        day = score_v1(day)
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
            peak_equity = max(peak_equity, equity)
            
            for sym in stopped_out:
                active_syms.discard(sym)
        
        equity *= (1 - cost * position_pct)
        
        for sym, entry_p in entry_prices.items():
            exit_p = price_dict.get(next_rd, {}).get(sym, entry_p)
            ret = exit_p / entry_p - 1
            if stop_loss is not None and ret < stop_loss:
                ret = stop_loss
            trades.append({'sym': sym, 'date': rd, 'return': ret - cost})
    
    if len(equity_curve) < 2:
        return None
    
    eq_arr = np.array([e[1] for e in equity_curve])
    eq_dates = np.array([e[0] for e in equity_curve])
    
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
# 4. Walk-Forward验证
# ============================================================
print("\n[2] Walk-Forward验证...")

# 定义fold: 2年训练 + 1年测试，滑动1年
folds = [
    (20160101, 20180101, 20180101, 20190101),  # train: 2016-2017, test: 2018
    (20170101, 20190101, 20190101, 20200101),  # train: 2017-2018, test: 2019
    (20180101, 20200101, 20200101, 20210101),  # train: 2018-2019, test: 2020
    (20190101, 20210101, 20210101, 20220101),  # train: 2019-2020, test: 2021
    (20200101, 20220101, 20220101, 20230101),  # train: 2020-2021, test: 2022
    (20210101, 20230101, 20230101, 20240101),  # train: 2021-2022, test: 2023
    (20220101, 20240101, 20240101, 20250101),  # train: 2022-2023, test: 2024
    (20230101, 20250101, 20250101, 20260616),  # train: 2023-2024, test: 2025-2026
]

# 配置
configs = [
    ('v1.0_noMF (baseline)', None),
    ('v1.0_MF (old prod)', 'market_filter'),
    ('DD3保守 (v2.1)', [(-0.03, 0.80), (-0.06, 0.60), (-0.10, 0.40), (-0.14, 0.20), (-0.18, 0.0)]),
    ('DD7温和', [(-0.07, 0.80), (-0.12, 0.60), (-0.17, 0.40), (-0.22, 0.20), (-0.27, 0.0)]),
]

# 市场状态（用于MF）
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

# 需要修改回测函数以支持MF
def run_backtest_mf(df_all, test_start, test_end, hold_days=10, top_n=15, 
                     stop_loss=-0.01, cost=0.003):
    """Market filter回测"""
    
    df_test = df_all[(df_all['date'] >= test_start) & (df_all['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    if len(test_dates) < 20:
        return None
    
    price_dict = {}
    for d in test_dates:
        day_data = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(day_data['sym'], day_data['close']))
    
    rebal_dates = test_dates[::hold_days]
    
    equity = 100000.0
    equity_curve = [(test_dates[0], equity)]
    trades = []
    
    for i, rd in enumerate(rebal_dates):
        state = market_state_map.get(rd, 'bull')
        if state == 'bear':
            next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
            for d in test_dates:
                if rd < d <= next_rd:
                    equity_curve.append((d, equity))
            continue
        elif state == 'cautious':
            position_pct = 0.5
        else:
            position_pct = 1.0
        
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        day = score_v1(day)
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
    
    if len(equity_curve) < 2:
        return None
    
    eq_arr = np.array([e[1] for e in equity_curve])
    eq_dates = np.array([e[0] for e in equity_curve])
    
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

# 运行WF
wf_results = {}
for config_name, config in configs:
    print(f"\n  {config_name}:")
    fold_results = []
    
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds):
        if config == 'market_filter':
            result = run_backtest_mf(df, test_start, test_end)
        else:
            result = run_backtest(df, test_start, test_end, dd_thresholds=config)
        
        if result is not None:
            fold_results.append(result)
            print(f"    Fold {fold_idx+1} ({test_start}-{test_end}): Sharpe={result['sharpe']:.2f} CAGR={result['cagr']:.1%} DD={result['max_dd']:.1%}")
        else:
            print(f"    Fold {fold_idx+1} ({test_start}-{test_end}): SKIPPED (insufficient data)")
    
    if fold_results:
        avg_sharpe = np.mean([r['sharpe'] for r in fold_results])
        std_sharpe = np.std([r['sharpe'] for r in fold_results])
        avg_cagr = np.mean([r['cagr'] for r in fold_results])
        avg_dd = np.mean([r['max_dd'] for r in fold_results])
        avg_winrate = np.mean([r['win_rate'] for r in fold_results])
        
        wf_results[config_name] = {
            'avg_sharpe': avg_sharpe, 'std_sharpe': std_sharpe,
            'avg_cagr': avg_cagr, 'avg_dd': avg_dd, 'avg_winrate': avg_winrate,
            'n_folds': len(fold_results), 'fold_details': fold_results
        }
        
        print(f"    平均: Sharpe={avg_sharpe:.2f}±{std_sharpe:.2f} CAGR={avg_cagr:.1%} DD={avg_dd:.1%}")

# ============================================================
# 5. 结果汇总
# ============================================================
print("\n" + "="*80)
print("📊 Walk-Forward验证结果")
print("="*80)

print(f"\n{'配置':<25} {'Sharpe':>8} {'±':>4} {'CAGR':>8} {'MaxDD':>8} {'胜率':>7} {'Fold':>5}")
print("-"*65)
for name, r in sorted(wf_results.items(), key=lambda x: x[1]['avg_sharpe'], reverse=True):
    print(f"{name:<25} {r['avg_sharpe']:>7.2f} {r['std_sharpe']:>5.2f} {r['avg_cagr']:>7.1%} {r['avg_dd']:>7.1%} "
          f"{r['avg_winrate']:>6.1%} {r['n_folds']:>5}")

# ============================================================
# 6. 保存结果
# ============================================================
output = {
    'experiment': 'rule-alpha-v2.1-walk-forward',
    'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    'folds': [(f[2], f[3]) for f in folds],
    'results': {k: {kk: vv for kk, vv in v.items() if kk != 'fold_details'} for k, v in wf_results.items()},
    'detailed_results': wf_results,
}

with open('research/rule_alpha_v2_1_walkforward.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存: research/rule_alpha_v2_1_walkforward.json")
print("="*60)
print("CEO决策: v2.1 Walk-Forward验证完成")
print("="*60)
