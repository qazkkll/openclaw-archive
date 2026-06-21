#!/usr/bin/env python3
"""
rule-alpha-v1.0 — 正确的每日回撤验证
CEO决策: 验证真实每日回撤
方法: 先用原始方法验证一致性，再添加每日跟踪
"""
import pandas as pd, numpy as np, json, time, os
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v1.0 — 正确的每日回撤验证")
print("="*60)

# ============================================================
# 1. 加载数据（与原始backtest_v5完全一致）
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

# 特征（与原始完全一致）
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_bias'] = df.groupby('date')['ma20_bias'].transform('mean')
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

# 前向收益
for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

all_dates_full = sorted(df['date'].unique())
print(f"  全量: {len(df):,}行, {df['sym'].nunique()}只, {all_dates_full[0]}~{all_dates_full[-1]}, {time.time()-t0:.0f}秒")

# ============================================================
# 2. 评分函数（与原始完全一致）
# ============================================================
def score_optimized(day):
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
# 3. 市场状态
# ============================================================
market_avg_r20 = df.groupby('date')['mkt_ret20'].first()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()

market_state_map = {}
for d in all_dates_full:
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

# ============================================================
# 4. 原始回测方法（验证一致性）
# ============================================================
def run_backtest_original(df_data, dates, hold_days=10, top_n=15, stop_loss=-0.03, cost=0.003, market_filter=True):
    """与原始backtest_v5完全一致的方法"""
    rebal_dates = dates[::hold_days]
    trades = []
    equity = 100000.0
    equity_curve = []
    
    for rd in rebal_dates:
        day = df_data[df_data['date'] == rd].copy()
        if len(day) < top_n:
            continue
        
        # 市场过滤
        if market_filter:
            state = market_state_map.get(rd, 'bull')
            if state == 'bear':
                equity_curve.append((rd, equity))
                continue
            elif state in ['cautious']:
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        day = score_optimized(day)
        picks = day.nlargest(top_n, 'score')
        
        fwd_col = f'fwd_{hold_days}d'
        rets = picks[fwd_col].fillna(0).values
        if stop_loss is not None:
            rets = np.where(rets < stop_loss, stop_loss, rets)
        rets = rets - cost
        
        for i, (_, row) in enumerate(picks.iterrows()):
            trades.append({'sym': row['sym'], 'date': rd, 'score': row['score'],
                          'fwd_ret': row.get(fwd_col, 0), 'net_ret': rets[i]})
        
        avg_ret = rets.mean() * position_pct
        equity *= (1 + avg_ret)
        equity_curve.append((rd, equity))
    
    return trades, equity_curve

def calc_metrics(trades, equity_curve, name, hold_days):
    if not trades:
        return {'strategy': name, 'trades': 0}
    
    rets = np.array([t['net_ret'] for t in trades])
    eq = np.array([e[1] for e in equity_curve])
    eq_dates = [e[0] for e in equity_curve]
    
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()
    
    dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
    dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
    years = (dt2 - dt1).days / 365.25
    total_ret = eq[-1] / eq[0] - 1
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak
    max_dd = dd.max()
    
    tpy = 252 / hold_days
    ann_ret = avg_ret * tpy
    ann_std = rets.std() * np.sqrt(tpy)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    downside = rets[rets < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = ann_ret / (downside_std * np.sqrt(tpy)) if downside_std > 0 else 0
    
    avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_loss = rets[rets < 0].mean() if (rets < 0).any() else 0
    
    return {
        'strategy': name, 'trades': len(trades), 'win_rate': round(win_rate, 4),
        'avg_return': round(avg_ret, 4), 'cagr': round(cagr, 4),
        'sharpe': round(sharpe, 4), 'sortino': round(sortino, 4),
        'max_dd': round(max_dd, 4), 'final_equity': round(eq[-1], 0),
        'avg_win': round(avg_win, 4), 'avg_loss': round(avg_loss, 4),
        'pl_ratio': round(abs(avg_win/avg_loss), 2) if avg_loss != 0 else 0,
    }

import datetime

# ============================================================
# 5. 运行原始方法验证
# ============================================================
print("\n[2] 原始方法验证（应该与report一致）...")

test_dates = sorted(df[(df['date'] >= 20200101) & (df['date'] <= 20260616)]['date'].unique())
df_test = df[(df['date'] >= 20200101) & (df['date'] <= 20260616)]

# SL3% + 市场过滤
trades_orig, eq_orig = run_backtest_original(df_test, test_dates, stop_loss=-0.03, market_filter=True)
m_orig = calc_metrics(trades_orig, eq_orig, 'SL3%+MF', 10)
print(f"  SL3%+MF: CAGR={m_orig['cagr']:.2%} Sharpe={m_orig['sharpe']:.2f} DD={m_orig['max_dd']:.2%}")

# SL3% 无过滤
trades_no_mf, eq_no_mf = run_backtest_original(df_test, test_dates, stop_loss=-0.03, market_filter=False)
m_no_mf = calc_metrics(trades_no_mf, eq_no_mf, 'SL3%', 10)
print(f"  SL3%无MF: CAGR={m_no_mf['cagr']:.2%} Sharpe={m_no_mf['sharpe']:.2f} DD={m_no_mf['max_dd']:.2%}")

# ============================================================
# 6. 添加每日回撤跟踪（在原始方法基础上）
# ============================================================
print("\n[3] 每日回撤跟踪...")

# 对SL3%+MF配置，跟踪每日回撤
# 方法：在每个rebal期间，用fwd_Nd的逐日版本来跟踪
def run_daily_dd_tracking(df_data, dates, hold_days=10, top_n=15, stop_loss=-0.03, cost=0.003, market_filter=True):
    """在原始方法基础上添加每日权益跟踪"""
    rebal_dates = dates[::hold_days]
    equity = 100000.0
    daily_equity = []
    trades = []
    
    # 构建价格查找表
    price_lookup = {}
    for d in dates:
        day_data = df_data[df_data['date'] == d][['sym', 'close']].set_index('sym')
        price_lookup[d] = day_data['close'].to_dict()
    
    for i, rd in enumerate(rebal_dates):
        day = df_data[df_data['date'] == rd].copy()
        if len(day) < top_n:
            daily_equity.append((rd, equity))
            continue
        
        # 市场过滤
        if market_filter:
            state = market_state_map.get(rd, 'bull')
            if state == 'bear':
                daily_equity.append((rd, equity))
                continue
            elif state in ['cautious']:
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        day = score_optimized(day)
        picks = day.nlargest(top_n, 'score')
        selected_syms = picks['sym'].tolist()
        entry_prices = {row['sym']: row['close'] for _, row in picks.iterrows()}
        
        # 扣除买入成本
        equity *= (1 - cost * position_pct)
        
        # 下一个调仓日
        next_rd = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        hold_dates_in_period = [d for d in dates if rd < d <= next_rd]
        
        # 每日跟踪
        active = set(selected_syms)
        prev_prices = {sym: entry_prices[sym] for sym in selected_syms}
        
        for hd in hold_dates_in_period:
            curr_prices = price_lookup.get(hd, {})
            
            daily_rets = []
            stopped = []
            for sym in list(active):
                if sym in curr_prices and sym in prev_prices:
                    curr_p = curr_prices[sym]
                    entry_p = entry_prices[sym]
                    prev_p = prev_prices[sym]
                    
                    # 止损检查（从入场价算起）
                    cum_ret = curr_p / entry_p - 1
                    if stop_loss is not None and cum_ret <= stop_loss:
                        stopped.append(sym)
                        # 止损时的收益
                        sl_ret = stop_loss
                        daily_rets.append(sl_ret * position_pct / len(active))
                    else:
                        # 正常日收益
                        day_ret = curr_p / prev_p - 1
                        daily_rets.append(day_ret * position_pct / len(active))
                    
                    prev_prices[sym] = curr_p
                else:
                    daily_rets.append(0)
            
            for sym in stopped:
                active.discard(sym)
            
            if daily_rets:
                port_ret = sum(daily_rets)
                equity *= (1 + port_ret)
            
            daily_equity.append((hd, equity))
        
        # 卖出成本
        equity *= (1 - cost * position_pct)
        daily_equity.append((next_rd, equity))
        
        # 记录交易
        fwd_col = f'fwd_{hold_days}d'
        for _, row in picks.iterrows():
            ret = row.get(fwd_col, 0)
            if stop_loss is not None and ret < stop_loss:
                ret = stop_loss
            trades.append({'sym': row['sym'], 'date': rd, 'net_ret': ret - cost})
    
    return trades, daily_equity

print("  运行SL3%+MF每日跟踪...")
t1 = time.time()
trades_daily, eq_daily = run_daily_dd_tracking(df_test, test_dates, stop_loss=-0.03, market_filter=True)

eq_arr = np.array([e[1] for e in eq_daily])
dates_arr = np.array([e[0] for e in eq_daily])

# 每日收益率
daily_rets = np.diff(eq_arr) / eq_arr[:-1]
daily_rets = daily_rets[np.isfinite(daily_rets)]

# 最大回撤（每日）
peak = np.maximum.accumulate(eq_arr)
dd = (eq_arr - peak) / peak
max_dd = dd.min()

# CAGR
years = len(eq_arr) / 252
total_ret = eq_arr[-1] / eq_arr[0] - 1
cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0

# Sharpe
ann_ret = daily_rets.mean() * 252
ann_std = daily_rets.std() * np.sqrt(252)
sharpe = ann_ret / ann_std if ann_std > 0 else 0

print(f"  完成 ({time.time()-t1:.0f}秒)")
print(f"\n  === SL3%+MF 每日跟踪结果 ===")
print(f"  CAGR: {cagr:.2%}")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  MaxDD (daily): {max_dd:.2%}")
print(f"  最终权益: {eq_arr[-1]:,.0f}")

# 找Top 5回撤期
dd_periods = []
dd_start = None
for k in range(len(dd)):
    if dd[k] < -0.005 and dd_start is None:
        dd_start = k
    elif dd[k] >= -0.005 and dd_start is not None:
        dd_periods.append((dates_arr[dd_start], dates_arr[k-1], dd[dd_start:k].min()))
        dd_start = None

dd_periods.sort(key=lambda x: x[2])
print(f"\n  Top 5 回撤期:")
for start, end, depth in dd_periods[:5]:
    print(f"    {start}~{end}: {depth:.2%}")

# 年度回撤
print(f"\n  年度最大回撤:")
for year in range(2020, 2027):
    year_mask = (dates_arr // 10000 == year)
    if year_mask.any():
        year_dd = dd[year_mask].min()
        print(f"    {year}: {year_dd:.2%}")

# ============================================================
# 7. 所有配置对比
# ============================================================
print("\n[4] 所有配置对比...")
configs = [
    ('SL3%+MF', -0.03, True),
    ('SL5%+MF', -0.05, True),
    ('SL3%', -0.03, False),
    ('SL5%', -0.05, False),
]

for name, sl, mf in configs:
    trades, eq = run_backtest_original(df_test, test_dates, stop_loss=sl, market_filter=mf)
    m = calc_metrics(trades, eq, name, 10)
    print(f"  {name}: CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.2f} DD={m['max_dd']:.2%} 胜率={m['win_rate']:.1%}")

# ============================================================
# 8. Walk-Forward验证（9折）
# ============================================================
print("\n[5] Walk-Forward验证...")
folds = [
    ('WF1: 2016-17→2018', 20160101, 20171231, 20180101, 20181231),
    ('WF2: 2017-18→2019', 20170101, 20181231, 20190101, 20191231),
    ('WF3: 2018-19→2020', 20180101, 20191231, 20200101, 20201231),
    ('WF4: 2019-20→2021', 20190101, 20201231, 20210101, 20211231),
    ('WF5: 2020-21→2022', 20200101, 20211231, 20220101, 20221231),
    ('WF6: 2021-22→2023', 20210101, 20221231, 20230101, 20231231),
    ('WF7: 2022-23→2024', 20220101, 20231231, 20240101, 20241231),
    ('WF8: 2023-24→2025', 20230101, 20241231, 20250101, 20251231),
    ('WF9: 2024-25→2026', 20240101, 20251231, 20260101, 20260616),
]

wf_sharpes = []
for name, ts, te, vs, ve in folds:
    test_d = sorted(df[(df['date'] >= vs) & (df['date'] <= ve)]['date'].unique())
    df_fold = df[(df['date'] >= vs) & (df['date'] <= ve)]
    trades, eq = run_backtest_original(df_fold, test_d)
    m = calc_metrics(trades, eq, name, 10)
    wf_sharpes.append(m['sharpe'])
    print(f"  {name}: Sharpe={m['sharpe']:.2f} CAGR={m['cagr']:.2%} DD={m['max_dd']:.2%}")

print(f"\n  WF Sharpe: {np.mean(wf_sharpes):.2f} ± {np.std(wf_sharpes):.2f}")
print(f"  全正: {'✅' if all(s > 0 for s in wf_sharpes) else '❌'}")

# ============================================================
# 9. 保存
# ============================================================
output = {
    'version': 'rule-alpha-v1.0',
    'test_period': '2020-01-02 ~ 2026-06-16',
    'original_method': {
        'cagr': m_orig['cagr'],
        'sharpe': m_orig['sharpe'],
        'max_dd_rebal': m_orig['max_dd'],
    },
    'daily_tracking': {
        'cagr': round(cagr, 4),
        'sharpe': round(sharpe, 4),
        'max_dd_daily': round(max_dd, 4),
        'final_equity': round(eq_arr[-1], 0),
    },
    'top5_dd_periods': [(str(s), str(e), round(d, 4)) for s, e, d in dd_periods[:5]],
    'wf_sharpe_mean': round(np.mean(wf_sharpes), 4),
    'wf_sharpe_std': round(np.std(wf_sharpes), 4),
}

with open('research/rule_alpha_v1_daily_dd.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: research/rule_alpha_v1_daily_dd.json")
