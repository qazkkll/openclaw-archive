#!/usr/bin/env python3
"""
A股规则型策略回测 — 全向量化 v3 (秒级完成)
CEO决策: 预计算所有调仓日的信号和退出收益, 不做逐日循环
"""
import sys, os, time, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

WORKSPACE = os.path.expanduser('~/.hermes/openclaw-archive')
DATA_DIR = os.path.join(WORKSPACE, 'data')

print("="*60)
print("A股规则型策略回测 v3 — 全向量化")
print("="*60)

# === 1. 加载+特征 ===
print("\n[1] 加载+特征计算...")
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
df = df.merge(mf[['sym','date','sm_net','md_net','lg_net','elg_net','total_net']], on=['sym','date'], how='left')

# 过滤
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

# RSI
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

# 资金流聚合
for col in ['total_net', 'lg_net', 'md_net', 'elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场宽度
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_bias'] = df.groupby('date')['ma60_bias'].transform('mean')
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')

# 标签
for hd in [5, 10, 20, 30]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

# 板块
df['board'] = '主板'
df.loc[df['sym'].str.startswith('30'), 'board'] = '创业板'

print(f"  {len(df):,}行, {df['sym'].nunique()}只, 耗时{time.time()-t0:.0f}秒")

# === 2. 测试期 ===
df_test = df[(df['date'] >= 20200101) & (df['date'] <= 20260616)].copy()
all_dates = sorted(df_test['date'].unique())
print(f"  测试期: {all_dates[0]}~{all_dates[-1]}, {len(all_dates)}天, {len(df_test):,}行")

# === 3. 市场状态 ===
mkt = df_test.groupby('date').agg(
    bias=('mkt_bias', 'first'),
    ret20=('mkt_ret20', 'first'),
    breadth=('breadth', 'first')
).reset_index()

def market_state(row):
    if row['bias'] > 0 and row['ret20'] > 0 and row['breadth'] > 0.5:
        return 'bull'
    elif (row['bias'] > 0 or row['ret20'] > 0) and row['breadth'] > 0.3:
        return 'cautious'
    else:
        return 'bear'

mkt['state'] = mkt.apply(market_state, axis=1)
mkt_lookup = dict(zip(mkt['date'], mkt['state']))

# === 4. 向量化回测 ===
def run_backtest(df_test, all_dates, hold_days, top_n, score_fn, 
                 stop_loss=None, market_filter=False, entry_filter=None, cost=0.003):
    """
    核心回测:
    - 每隔hold_days调仓
    - 用score_fn对当日所有股票评分
    - 选top_n, 计算fwd_{hold_days}d收益
    - 返回trades列表 + equity curve
    """
    # 调仓日
    rebal_dates = all_dates[::hold_days]
    
    trades = []
    equity = 100000.0
    equity_curve = []
    
    for rd in rebal_dates:
        # 市场过滤
        if market_filter:
            state = mkt_lookup.get(rd, 'bear')
            if state == 'bear':
                equity_curve.append((rd, equity))
                continue
            elif state == 'cautious':
                position_pct = 0.5
            else:
                position_pct = 1.0
        else:
            position_pct = 1.0
        
        # 当日数据
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < top_n:
            continue
        
        # 入场过滤
        if entry_filter:
            for feat, (lo, hi) in entry_filter.items():
                if feat in day.columns:
                    if lo is not None:
                        day = day[day[feat] >= lo]
                    if hi is not None:
                        day = day[day[feat] <= hi]
            if len(day) < 1:
                equity_curve.append((rd, equity))
                continue
        
        # 评分
        day = score_fn(day)
        
        # Top N
        picks = day.nlargest(top_n, 'score')
        
        # 获取收益 (用fwd_{hold_days}d标签)
        fwd_col = f'fwd_{hold_days}d'
        rets = picks[fwd_col].fillna(0).values
        
        # 止损
        if stop_loss is not None:
            rets = np.where(rets < stop_loss, stop_loss, rets)
        
        # 扣成本
        rets = rets - cost
        
        # 记录trades
        for _, row in picks.iterrows():
            trades.append({
                'sym': row['sym'],
                'date': rd,
                'close': row['close'],
                'score': row['score'],
                'fwd_ret': row[fwd_col] if fwd_col in row else 0,
                'net_ret': rets[list(picks.index).index(row.name)] if row.name in picks.index else 0,
            })
        
        # 更新权益
        avg_ret = rets.mean() * position_pct
        equity *= (1 + avg_ret)
        equity_curve.append((rd, equity))
    
    return trades, equity_curve

# === 5. 评分函数 ===
def score_default(day):
    """反转+资金流+低波动+超卖"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.5, 0.5) * 2
    rk = s['total_net_5d'].fillna(0).rank(pct=True)
    s['score'] += rk * 2
    rk = s['vol20'].fillna(s['vol20'].median()).rank(pct=True, ascending=True)
    s['score'] += (1 - rk) * 1
    s['score'] += (s['rsi_14'].fillna(50) < 40).astype(float) * 1
    return s

def score_reversal(day):
    return day.assign(score=(-day['ret20'].fillna(0)).clip(-0.5, 0.5))

def score_flow(day):
    return day.assign(score=day['total_net_5d'].fillna(0).rank(pct=True))

def score_low_vol(day):
    return day.assign(score=-day['vol20'].fillna(day['vol20'].median()))

def score_rsi(day):
    return day.assign(score=-(day['rsi_14'].fillna(50) - 50))

def score_optimized(day):
    """优化版: 更强反转+资金流+低波动+超卖+大单"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3  # 反转权重加大
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2  # 资金流
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2  # 低波动权重加大
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5  # 更严格的超卖
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1  # 大单
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1  # 均线偏离
    return s

def score_conservative(day):
    """保守版: 强调低波动+资金流入+中等反转"""
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 3  # 低波动最重要
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.2, 0.2) * 1  # 温和反转
    s['score'] += (s['rsi_14'].fillna(50) < 45).astype(float) * 1
    return s

# === 6. 实验矩阵 ===
print("\n[2] 开始回测...")
print("="*60)

experiments = [
    # 基础策略
    ('A_default_5d',     dict(hold_days=5,  top_n=15, score_fn=score_default)),
    ('B_reversal_5d',    dict(hold_days=5,  top_n=15, score_fn=score_reversal)),
    ('C_flow_5d',        dict(hold_days=5,  top_n=15, score_fn=score_flow)),
    ('D_low_vol_5d',     dict(hold_days=5,  top_n=15, score_fn=score_low_vol)),
    ('E_rsi_5d',         dict(hold_days=5,  top_n=15, score_fn=score_rsi)),
    ('F_optimized_5d',   dict(hold_days=5,  top_n=15, score_fn=score_optimized)),
    ('G_conservative_5d',dict(hold_days=5,  top_n=15, score_fn=score_conservative)),
    
    # 不同持有期
    ('H_default_10d',    dict(hold_days=10, top_n=15, score_fn=score_default)),
    ('I_default_20d',    dict(hold_days=20, top_n=15, score_fn=score_default)),
    ('J_optimized_10d',  dict(hold_days=10, top_n=15, score_fn=score_optimized)),
    ('K_optimized_20d',  dict(hold_days=20, top_n=15, score_fn=score_optimized)),
    ('L_conservative_10d',dict(hold_days=10,top_n=15, score_fn=score_conservative)),
    ('M_conservative_20d',dict(hold_days=20,top_n=15, score_fn=score_conservative)),
    
    # 不同Top N
    ('N_top10_opt_10d',  dict(hold_days=10, top_n=10, score_fn=score_optimized)),
    ('O_top20_opt_10d',  dict(hold_days=10, top_n=20, score_fn=score_optimized)),
    ('P_top30_opt_10d',  dict(hold_days=10, top_n=30, score_fn=score_optimized)),
    
    # 止损
    ('Q_sl5_opt_10d',    dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.05)),
    ('R_sl8_opt_10d',    dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.08)),
    ('S_sl10_opt_10d',   dict(hold_days=10, top_n=15, score_fn=score_optimized, stop_loss=-0.10)),
    
    # 市场过滤
    ('T_mf_opt_5d',      dict(hold_days=5,  top_n=15, score_fn=score_optimized, market_filter=True)),
    ('U_mf_opt_10d',     dict(hold_days=10, top_n=15, score_fn=score_optimized, market_filter=True)),
    ('V_mf_conservative_10d',dict(hold_days=10, top_n=15, score_fn=score_conservative, market_filter=True)),
    
    # 入场过滤
    ('W_entry_oversold_10d', dict(hold_days=10, top_n=15, score_fn=score_optimized, entry_filter={'ret20': (None, -0.05)})),
    ('X_entry_flow_10d',     dict(hold_days=10, top_n=15, score_fn=score_optimized, entry_filter={'total_net_5d': (0, None)})),
    ('Y_entry_both_10d',     dict(hold_days=10, top_n=15, score_fn=score_optimized, entry_filter={'ret20': (None, -0.03), 'total_net_5d': (0, None)})),
    
    # 最优组合
    ('Z_best_combo', dict(hold_days=10, top_n=15, score_fn=score_optimized, market_filter=True, stop_loss=-0.08)),
]

results = []
for name, cfg in experiments:
    print(f"  {name}...", end=' ', flush=True)
    t0 = time.time()
    trades, eq = run_backtest(df_test, all_dates, **cfg)
    
    if not trades:
        print("无交易")
        results.append({'strategy': name, 'trades': 0, 'win_rate': 0, 'avg_return': 0, 'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'final_equity': 100000})
        continue
    
    rets = np.array([t['net_ret'] for t in trades])
    eq_vals = np.array([e[1] for e in eq])
    eq_dates = [e[0] for e in eq]
    
    win_rate = (rets > 0).mean()
    avg_ret = rets.mean()
    n_trades = len(trades)
    
    # CAGR from equity curve (convert YYYYMMDD to actual days)
    import datetime
    dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
    dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
    years = (dt2 - dt1).days / 365.25
    total_ret = eq_vals[-1] / eq_vals[0] - 1
    cagr = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
    
    # Max DD
    peak = np.maximum.accumulate(eq_vals)
    dd = (peak - eq_vals) / peak
    max_dd = dd.max()
    
    # Sharpe
    tpy = 252 / cfg['hold_days']
    ann_ret = avg_ret * tpy
    ann_std = rets.std() * np.sqrt(tpy)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    
    r = {
        'strategy': name, 'trades': n_trades, 'win_rate': round(win_rate, 4),
        'avg_return': round(avg_ret, 4), 'cagr': round(cagr, 4),
        'sharpe': round(sharpe, 4), 'max_dd': round(max_dd, 4),
        'final_equity': round(eq_vals[-1], 0),
    }
    results.append(r)
    print(f"  {n_trades}笔 胜率{win_rate:.1%} 均收{avg_ret:.2%} 年化{cagr:.1%} Sharpe{sharpe:.2f} DD{max_dd:.1%} ({time.time()-t0:.0f}s)")

# === 7. 汇总 ===
print("\n" + "="*60)
print("[3] 结果汇总（按Sharpe排序）")
print("="*60)

results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

print(f"\n{'策略':<28} {'交易':>5} {'胜率':>6} {'均收':>7} {'年化':>7} {'Sharpe':>7} {'DD':>7} {'终值':>10}")
print("-"*85)
for r in results:
    print(f"{r['strategy']:<28} {r['trades']:>5} {r['win_rate']:>5.1%} {r['avg_return']:>6.2%} {r['cagr']:>6.1%} {r['sharpe']:>6.2f} {r['max_dd']:>6.1%} {r['final_equity']:>10,.0f}")

# === 8. CEO判断 ===
print("\n" + "="*60)
print("[4] CEO决策")
print("="*60)

good = [r for r in results if r['max_dd'] < 0.20 and r['sharpe'] > 0.5]
print(f"\nDD<20% + Sharpe>0.5: {len(good)}个")
for r in good[:5]:
    print(f"  {r['strategy']}: Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, 年化 {r['cagr']:.1%}")

target = [r for r in results if r['max_dd'] < 0.15 and r['cagr'] > 0.10]
print(f"\nDD<15% + 年化>10%: {len(target)}个")
for r in target[:5]:
    print(f"  {r['strategy']}: Sharpe {r['sharpe']:.2f}, DD {r['max_dd']:.1%}, 年化 {r['cagr']:.1%}")

# === 9. 保存 ===
output_file = os.path.join(WORKSPACE, 'research', 'backtest_v3_results.json')
with open(output_file, 'w') as f:
    json.dump({'results': results}, f, indent=2, ensure_ascii=False, default=str)
print(f"\n结果已保存: {output_file}")
