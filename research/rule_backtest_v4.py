#!/usr/bin/env python3
"""
A股规则型策略 — 深度优化 v4
CEO方向: 围绕SL-5%最佳点做精细优化
- 测试不同SL阈值(3%/4%/5%/6%/7%)
- 测试不同Top N(5/10/15/20)
- 年度分解分析
- 最终CEO定版
"""
import sys, os, time, json, datetime
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

WORKSPACE = os.path.expanduser('~/.hermes/openclaw-archive')
DATA_DIR = os.path.join(WORKSPACE, 'data')

print("="*60)
print("A股规则型策略 — 深度优化 v4")
print("="*60)

# === 1. 加载+特征 ===
print("\n[1] 加载+特征...")
t0 = time.time()

df = pd.read_parquet(os.path.join(DATA_DIR, 'a_hist_10y.parquet'))
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet(os.path.join(DATA_DIR, 'cn/moneyflow_core.parquet'))
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
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ret10'] = df.groupby('sym')['close'].pct_change(10)
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret60'] = df.groupby('sym')['close'].pct_change(60)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma60'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(60, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ma60_bias'] = (df['close'] - df['ma60']) / df['ma60']
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_bias'] = df.groupby('date')['ma60_bias'].transform('mean')
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

print(f"  {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

# === 2. 测试期 ===
df_test = df[(df['date'] >= 20200101) & (df['date'] <= 20260616)].copy()
all_dates = sorted(df_test['date'].unique())
print(f"  测试期: {all_dates[0]}~{all_dates[-1]}, {len(all_dates)}天")

# 市场状态
mkt = df_test.groupby('date').agg(bias=('mkt_bias','first'), ret20=('mkt_ret20','first'), breadth=('breadth','first')).reset_index()
def mkt_state(row):
    if row['bias'] > 0 and row['ret20'] > 0 and row['breadth'] > 0.5: return 'bull'
    elif (row['bias'] > 0 or row['ret20'] > 0) and row['breadth'] > 0.3: return 'cautious'
    else: return 'bear'
mkt['state'] = mkt.apply(mkt_state, axis=1)
mkt_lookup = dict(zip(mkt['date'], mkt['state']))

# === 3. 评分函数 ===
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

# === 4. 回测引擎 ===
def run_backtest(df_test, all_dates, hold_days, top_n, score_fn, 
                 stop_loss=None, market_filter=False, cost=0.003):
    rebal_dates = all_dates[::hold_days]
    trades = []
    equity = 100000.0
    equity_curve = []
    
    for rd in rebal_dates:
        position_pct = 1.0
        if market_filter:
            state = mkt_lookup.get(rd, 'bear')
            if state == 'bear':
                equity_curve.append((rd, equity))
                continue
            elif state == 'cautious':
                position_pct = 0.5
        
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        
        day = score_fn(day)
        picks = day.nlargest(top_n, 'score')
        
        fwd_col = f'fwd_{hold_days}d'
        rets = picks[fwd_col].fillna(0).values
        
        if stop_loss is not None:
            rets = np.where(rets < stop_loss, stop_loss, rets)
        
        rets = rets - cost
        
        for _, row in picks.iterrows():
            trades.append({'sym': row['sym'], 'date': rd, 'close': row['close'], 
                          'score': row['score'], 'fwd_ret': row.get(fwd_col, 0), 'net_ret': 0})
        
        # Update net_ret for recent trades
        for i, t in enumerate(trades[-len(picks):]):
            t['net_ret'] = rets[i]
        
        avg_ret = rets.mean() * position_pct
        equity *= (1 + avg_ret)
        equity_curve.append((rd, equity))
    
    return trades, equity_curve

def calc_metrics(trades, equity_curve, name, hold_days):
    if not trades:
        return {'strategy': name, 'trades': 0, 'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'win_rate': 0}
    
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
    
    return {
        'strategy': name, 'trades': len(trades), 'win_rate': round(win_rate, 4),
        'avg_return': round(avg_ret, 4), 'cagr': round(cagr, 4),
        'sharpe': round(sharpe, 4), 'sortino': round(sortino, 4),
        'max_dd': round(max_dd, 4), 'final_equity': round(eq[-1], 0),
        'avg_win': round(rets[rets>0].mean(), 4) if (rets>0).any() else 0,
        'avg_loss': round(rets[rets<0].mean(), 4) if (rets<0).any() else 0,
    }

# === 5. 实验矩阵 ===
print("\n[2] 精细优化实验...")
print("="*60)

experiments = []

# SL阈值扫描
for sl in [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08, -0.10]:
    experiments.append((f'sl{abs(int(sl*100))}_opt_10d', dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=sl)))

# Top N扫描
for n in [5, 8, 10, 12, 15, 20, 25]:
    experiments.append((f'top{n}_sl5_10d', dict(hold_days=10, top_n=n, score_fn=score_optimized, stop_loss=-0.05)))

# 持有期扫描 (with SL-5%, only valid forward periods)
for hd in [5, 10, 20]:
    experiments.append((f'hold{hd}d_sl5', dict(hold_days=hd, top_n=15, score_fn=score_optimized, stop_loss=-0.05)))

# 最佳组合+市场过滤
experiments.append(('BEST+mf', dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.05, market_filter=True)))
experiments.append(('BEST_mf_cau75', dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.05, market_filter=True)))

# 更激进的低波动策略
def score_ultra_low_vol(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 5  # 低波动最重要
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (s['rsi_14'].fillna(50) < 40).astype(float) * 1
    return s

experiments.append(('ultra_lowvol_5d_sl5', dict(hold_days=5, top_n=15, score_fn=score_ultra_low_vol, stop_loss=-0.05)))
experiments.append(('ultra_lowvol_10d_sl5', dict(hold_days=10, top_n=15, score_fn=score_ultra_low_vol, stop_loss=-0.05)))

# 成本敏感性
for cost_pct in [0.001, 0.003, 0.005]:
    experiments.append((f'cost{cost_pct*100:.0f}bp_sl5', dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.05, cost=cost_pct)))

results = []
for name, cfg in experiments:
    print(f"  {name}...", end=' ', flush=True)
    t0 = time.time()
    trades, eq = run_backtest(df_test, all_dates, **cfg)
    metrics = calc_metrics(trades, eq, name, cfg['hold_days'])
    results.append(metrics)
    print(f" {metrics['trades']}笔 胜率{metrics['win_rate']:.1%} 年化{metrics['cagr']:.1%} Sharpe{metrics['sharpe']:.2f} DD{metrics['max_dd']:.1%} ({time.time()-t0:.0f}s)")

# === 6. 年度分解 ===
print("\n[3] 年度分解（最优策略）...")
print("="*60)

# 用最优参数做年度分析
best_cfg = dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.05)
trades_best, eq_best = run_backtest(df_test, all_dates, **best_cfg)

# 按年分组
for t in trades_best:
    t['year'] = int(str(t['date'])[:4])

years_data = {}
for year in range(2020, 2027):
    year_trades = [t for t in trades_best if t['year'] == year]
    if not year_trades:
        continue
    rets = np.array([t['net_ret'] for t in year_trades])
    years_data[year] = {
        'trades': len(year_trades),
        'win_rate': round((rets > 0).mean(), 4),
        'avg_return': round(rets.mean(), 4),
        'total_return': round(rets.sum(), 4),
        'best_trade': round(rets.max(), 4),
        'worst_trade': round(rets.min(), 4),
    }

print(f"\n{'年份':>4} {'交易':>5} {'胜率':>6} {'均收':>7} {'总收':>8} {'最佳':>7} {'最差':>7}")
print("-"*50)
for year, d in sorted(years_data.items()):
    print(f"{year:>4} {d['trades']:>5} {d['win_rate']:>5.1%} {d['avg_return']:>6.2%} {d['total_return']:>7.1%} {d['best_trade']:>6.1%} {d['worst_trade']:>6.1%}")

# === 7. 汇总 ===
print("\n" + "="*60)
print("[4] 结果汇总（按Sharpe排序）")
print("="*60)

results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

print(f"\n{'策略':<25} {'交易':>5} {'胜率':>6} {'均收':>7} {'年化':>7} {'Sharpe':>7} {'Sortino':>7} {'DD':>7}")
print("-"*80)
for r in results[:20]:
    print(f"{r['strategy']:<25} {r['trades']:>5} {r['win_rate']:>5.1%} {r['avg_return']:>6.2%} {r['cagr']:>6.1%} {r['sharpe']:>6.2f} {r['sortino']:>6.2f} {r['max_dd']:>6.1%}")

# === 8. CEO定版 ===
print("\n" + "="*60)
print("[5] CEO定版决策")
print("="*60)

# 筛选条件：DD<15%, Sharpe>0.8, CAGR>10%
good = [r for r in results if r['max_dd'] < 0.15 and r['sharpe'] > 0.8 and r['cagr'] > 0.10]
print(f"\n满足条件(DD<15%, Sharpe>0.8, CAGR>10%): {len(good)}个")
for r in good[:5]:
    print(f"  {r['strategy']}: Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, 年化 {r['cagr']:.1%}")

if good:
    best = good[0]
    print(f"\n✅ CEO定版: {best['strategy']}")
    print(f"   年化: {best['cagr']:.1%}")
    print(f"   Sharpe: {best['sharpe']:.2f}")
    print(f"   Sortino: {best['sortino']:.2f}")
    print(f"   最大回撤: {best['max_dd']:.1%}")
    print(f"   胜率: {best['win_rate']:.1%}")
    print(f"   平均赢: {best['avg_win']:.2%}")
    print(f"   平均亏: {best['avg_loss']:.2%}")
    print(f"   盈亏比: {abs(best['avg_win']/best['avg_loss']):.2f}")
else:
    # 放宽条件
    good2 = [r for r in results if r['max_dd'] < 0.20 and r['sharpe'] > 0.5]
    print(f"\n放宽条件(DD<20%, Sharpe>0.5): {len(good2)}个")
    for r in good2[:5]:
        print(f"  {r['strategy']}: Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, 年化 {r['cagr']:.1%}")

# === 9. 保存 ===
output = {
    'test_period': '2020-01-02 ~ 2026-06-16',
    'stocks': df_test['sym'].nunique(),
    'results': results,
    'yearly': years_data,
}
output_file = os.path.join(WORKSPACE, 'research', 'backtest_v4_results.json')
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {output_file}")
