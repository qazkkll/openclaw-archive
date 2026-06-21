#!/usr/bin/env python3
"""
rule-alpha-v1.0 — 动态仓位优化
测试不同cautious/bull仓位比例，找到最优风险收益平衡
"""
import pandas as pd, numpy as np, json, datetime, time, warnings
warnings.filterwarnings('ignore')
import os
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("🔄 rule-alpha-v1.0 动态仓位优化")
print("="*60)
t0 = time.time()

# ============================================================
# 数据加载 (与验证框架一致)
# ============================================================
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

df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
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

print(f"  数据: {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

# 评分函数
def score_fn(day):
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
# 回测框架
# ============================================================
def run_bt(cautious_pct, bull_pct, stop_loss, hold_days=10, cost=0.003):
    test_start, test_end = 20200101, 20260616
    df_test = df[(df['date'] >= test_start) & (df['date'] <= test_end)]
    test_dates = sorted(df_test['date'].unique())
    
    price_dict = {}
    for d in test_dates:
        dd = df_test[df_test['date'] == d]
        price_dict[d] = dict(zip(dd['sym'], dd['close']))
    
    rebal_dates = test_dates[::hold_days]
    equity = 100000.0
    equity_curve = [(test_dates[0], equity)]
    trades = []
    
    for i, rd in enumerate(rebal_dates):
        state = market_state_map.get(rd, 'bull')
        if state == 'bear':
            pos_pct = 0
        elif state == 'cautious':
            pos_pct = cautious_pct
        else:
            pos_pct = bull_pct
        
        if pos_pct == 0:
            next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
            for d in test_dates:
                if rd < d <= next_rd:
                    equity_curve.append((d, equity))
            continue
        
        day = df_test[df_test['date'] == rd].copy()
        if len(day) < 15:
            continue
        scored = score_fn(day)
        picks = scored.nlargest(15, 'score')
        entry_prices = dict(zip(picks['sym'], picks['close']))
        equity *= (1 - cost * pos_pct)
        
        next_rd = rebal_dates[i+1] if i+1 < len(rebal_dates) else test_dates[-1]
        hold_dates = [d for d in test_dates if rd < d <= next_rd]
        active = set(entry_prices.keys())
        prev_p = {s: entry_prices[s] for s in active}
        
        for hd in hold_dates:
            curr = price_dict.get(hd, {})
            dr = 0.0
            na = len(active)
            if na == 0:
                equity_curve.append((hd, equity))
                continue
            w = pos_pct / na
            stopped = []
            for sym in list(active):
                if sym not in curr:
                    continue
                cp, ep, pp = curr[sym], entry_prices[sym], prev_p.get(sym, entry_prices[sym])
                cum = cp/ep - 1
                if stop_loss and cum <= stop_loss:
                    ppc = pp/ep - 1
                    if ppc <= stop_loss:
                        sr = 0
                    else:
                        sr = stop_loss - ppc
                        stopped.append(sym)
                else:
                    sr = cp/pp - 1 if pp > 0 else 0
                dr += sr * w
                prev_p[sym] = cp
            equity *= (1 + dr)
            equity_curve.append((hd, equity))
            for sym in stopped:
                active.discard(sym)
        
        equity *= (1 - cost * pos_pct)
        for sym, ep in entry_prices.items():
            xp = price_dict.get(next_rd, {}).get(sym, ep)
            ret = xp/ep - 1
            if stop_loss and ret < stop_loss:
                ret = stop_loss
            trades.append(ret - cost)
    
    eq_arr = np.array([e[1] for e in equity_curve])
    eq_dates = np.array([e[0] for e in equity_curve])
    daily_rets = np.diff(eq_arr) / eq_arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]
    
    dt1 = datetime.datetime.strptime(str(eq_dates[0]), '%Y%m%d')
    dt2 = datetime.datetime.strptime(str(eq_dates[-1]), '%Y%m%d')
    years = (dt2 - dt1).days / 365.25
    cagr = (eq_arr[-1]/eq_arr[0]) ** (1/years) - 1
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
    peak = np.maximum.accumulate(eq_arr)
    max_dd = ((eq_arr - peak) / peak).min()
    wr = (np.array(trades) > 0).mean() if trades else 0
    
    return cagr, sharpe, max_dd, wr, eq_arr[-1]

# ============================================================
# 测试矩阵
# ============================================================
print("\n" + "="*70)
print("📊 动态仓位优化结果")
print("="*70)

configs = [
    # (cautious_pct, bull_pct, stop_loss, name)
    # 当前生产配置
    (0.50, 1.0, -0.03, "当前: C50/B100/SL3%"),
    (0.50, 1.0, -0.02, "C50/B100/SL2%"),
    
    # 保守配置
    (0.30, 1.0, -0.02, "C30/B100/SL2%"),
    (0.30, 0.7, -0.02, "C30/B70/SL2%"),
    (0.20, 1.0, -0.02, "C20/B100/SL2%"),
    (0.20, 0.8, -0.02, "C20/B80/SL2%"),
    
    # 激进配置
    (0.70, 1.0, -0.02, "C70/B100/SL2%"),
    (0.70, 1.0, -0.03, "C70/B100/SL3%"),
    (1.00, 1.0, -0.03, "满仓/SL3%"),
    (1.00, 1.0, -0.02, "满仓/SL2%"),
    
    # 不同SL
    (0.50, 1.0, -0.01, "C50/B100/SL1%"),
    (0.50, 1.0, -0.04, "C50/B100/SL4%"),
    (0.50, 1.0, -0.05, "C50/B100/SL5%"),
    (0.50, 1.0, None,  "C50/B100/无SL"),
    
    # 不同持有期
    (0.50, 1.0, -0.02, "C50/B100/SL2%/5d", 5),
    (0.50, 1.0, -0.02, "C50/B100/SL2%/15d", 15),
    (0.50, 1.0, -0.02, "C50/B100/SL2%/20d", 20),
]

results = []
for cfg in configs:
    if len(cfg) == 5:
        c_pct, b_pct, sl, name, hold = cfg
    else:
        c_pct, b_pct, sl, name = cfg
        hold = 10
    
    print(f"  {name}...", end=" ", flush=True)
    t1 = time.time()
    cagr, sharpe, max_dd, wr, final = run_bt(c_pct, b_pct, sl, hold)
    results.append({
        'name': name, 'cagr': cagr, 'sharpe': sharpe, 'max_dd': max_dd,
        'win_rate': wr, 'final': final,
        'cautious': c_pct, 'bull': b_pct, 'sl': sl, 'hold': hold,
    })
    print(f"CAGR={cagr:.1%} Sharpe={sharpe:.2f} DD={max_dd:.1%} ({time.time()-t1:.0f}s)")

# 排序
results.sort(key=lambda x: x['sharpe'], reverse=True)

print(f"\n{'配置':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'胜率':>7}")
print("-" * 65)
for r in results:
    sl_str = f"SL{r['sl']*100:.0f}%" if r['sl'] else "无SL"
    print(f"{r['name']:<30} {r['cagr']:>7.1%} {r['sharpe']:>8.2f} {r['max_dd']:>7.1%} {r['win_rate']:>6.1%}")

# Pareto frontier
print("\n📈 Pareto最优 (Sharpe/MaxDD):")
pareto = []
for r in results:
    is_dominated = False
    for r2 in results:
        if r2['sharpe'] > r['sharpe'] and r2['max_dd'] > r['max_dd']:  # less negative DD = better
            is_dominated = True
            break
    if not is_dominated:
        pareto.append(r)

pareto.sort(key=lambda x: x['sharpe'], reverse=True)
for r in pareto:
    print(f"  {r['name']:<30} Sharpe={r['sharpe']:.2f} DD={r['max_dd']:.1%} CAGR={r['cagr']:.1%}")

# 保存
output = {
    'experiment': 'position_sizing_optimization',
    'date': '2026-06-21',
    'results': [{k: v for k, v in r.items()} for r in results]
}
with open('research/position_sizing_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ 保存 research/position_sizing_results.json")
print(f"⏱️ 总耗时: {time.time()-t0:.0f}秒")
