#!/usr/bin/env python3
"""
rule-alpha-v3.0 — Fast backtest using pre-computed features
Tests: 6 scoring functions with SL1% + DD-based position sizing
"""
import pandas as pd, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"📊 rule-alpha-v3.0 backtest {time.strftime('%Y-%m-%d %H:%M')}")

# ============================================================
# 1. Load Pre-Computed Features
# ============================================================
print("Loading features_v2.parquet...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)

# Filter
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close']>=3) & (df['close']<=200)].copy()
df = df[df['volume']>0].copy()
df = df.sort_values(['sym','date']).reset_index(drop=True)
print(f"  {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Extra Features
# ============================================================
print("Computing extra features...")

# Forward 10d return (for IC analysis)
df['fwd10'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10)/x - 1)

# Flow momentum: 5d net vs 20d net/4
df['flow_mom'] = df['total_net_5'] - df['total_net_20'] / 4

# Volume-price divergence
df['vol_price_div'] = -df['r5'] * (df['vol_r'].fillna(1) - 1)

# Market breadth & state (per date)
df['breadth'] = df.groupby('date')['r5'].transform(lambda x: (x>0).mean())
df['mkt_ret20'] = df.groupby('date')['r20'].transform('mean')

# MA20 bias (from close vs d20 — d20 is deviation from MA20 as fraction)
df['ma20_bias'] = df['d20'].fillna(0)

# Rename for convenience
df.rename(columns={'rsi14':'rsi_14', 'lg_net_5':'lg_net_5d', 'md_net_5':'md_net_5d',
                   'elg_net_5':'elg_net_5d', 'total_net_5':'total_net_5d'}, inplace=True)

print(f"  Done ({time.time()-t0:.0f}s)")

# ============================================================
# 3. Pre-group by date for fast lookup
# ============================================================
print("Pre-grouping by date...")
all_dates = sorted(df['date'].unique())
date_map = {}
for d, g in df.groupby('date'):
    date_map[d] = g
print(f"  {len(all_dates)} dates ({time.time()-t0:.0f}s)")

# ============================================================
# 4. Scoring Functions
# ============================================================
def score_v21(g):
    s = np.zeros(len(g))
    s += np.clip(-g['r20'].fillna(0).values, -0.3, 0.3) * 3
    s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 2
    s += (1 - pd.Series(g['vol20'].fillna(g['vol20'].median()).values).rank(pct=True).values) * 2
    s += (g['rsi_14'].fillna(50).values < 35).astype(float) * 1.5
    s += pd.Series(g['lg_net_5d'].fillna(0).values).rank(pct=True).values * 1
    s += np.clip(-g['ma20_bias'].fillna(0).values, -0.2, 0.2) * 1
    return s

def score_flow_heavy(g):
    """60% flow, 40% tech"""
    s = np.zeros(len(g))
    s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 2.5
    s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 2.0
    s += pd.Series(g['elg_net_5d'].fillna(0).values).rank(pct=True).values * 1.5
    s += pd.Series(g['lg_net_5d'].fillna(0).values).rank(pct=True).values * 1.0
    s += np.clip(-g['r20'].fillna(0).values, -0.3, 0.3) * 2
    s += (1 - pd.Series(g['vol20'].fillna(g['vol20'].median()).values).rank(pct=True).values) * 1
    return s

def score_all_flow(g):
    """Pure flow — no technicals"""
    s = np.zeros(len(g))
    s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 2
    s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 2
    s += pd.Series(g['elg_net_5d'].fillna(0).values).rank(pct=True).values * 2
    s += pd.Series(g['lg_net_5d'].fillna(0).values).rank(pct=True).values * 1.5
    s += pd.Series(g['flow_mom'].fillna(0).values).rank(pct=True).values * 0.5
    return s

def score_flow_reversal(g):
    """Flow + reversal only (top 2 stable factors)"""
    s = np.zeros(len(g))
    s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 3
    s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 2
    s += np.clip(-g['r20'].fillna(0).values, -0.3, 0.3) * 3
    return s

def score_extended(g):
    """v2.1 + new factors"""
    s = np.zeros(len(g))
    s += np.clip(-g['r20'].fillna(0).values, -0.3, 0.3) * 2.5
    s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 2
    s += (1 - pd.Series(g['vol20'].fillna(g['vol20'].median()).values).rank(pct=True).values) * 1.5
    s += (g['rsi_14'].fillna(50).values < 35).astype(float) * 1
    s += pd.Series(g['lg_net_5d'].fillna(0).values).rank(pct=True).values * 0.5
    s += np.clip(-g['ma20_bias'].fillna(0).values, -0.2, 0.2) * 0.5
    s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 1.5
    s += pd.Series(g['elg_net_5d'].fillna(0).values).rank(pct=True).values * 1
    s += pd.Series(g['flow_mom'].fillna(0).values).rank(pct=True).values * 0.5
    s += pd.Series(g['vol_price_div'].fillna(0).values).rank(pct=True).values * 0.5
    return s

def score_regime_adaptive(g, regime='bull'):
    """Different weights per market state"""
    s = np.zeros(len(g))
    if regime == 'bull':
        s += np.clip(-g['r20'].fillna(0).values, -0.3, 0.3) * 3.5
        s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 2.5
        s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 1.5
        s += (1 - pd.Series(g['vol20'].fillna(g['vol20'].median()).values).rank(pct=True).values) * 1
    elif regime == 'cautious':
        s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 3
        s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 2
        s += pd.Series(g['elg_net_5d'].fillna(0).values).rank(pct=True).values * 1.5
        s += (1 - pd.Series(g['vol20'].fillna(g['vol20'].median()).values).rank(pct=True).values) * 2
        s += np.clip(-g['ma20_bias'].fillna(0).values, -0.2, 0.2) * 1
    else:  # bear
        s += pd.Series(g['total_net_5d'].fillna(0).values).rank(pct=True).values * 3
        s += pd.Series(g['md_net_5d'].fillna(0).values).rank(pct=True).values * 3
        s += (1 - pd.Series(g['vol20'].fillna(g['vol20'].median()).values).rank(pct=True).values) * 2.5
        s += np.clip(-g['r20'].fillna(0).values, -0.3, 0.3) * 2
        s += (g['rsi_14'].fillna(50).values < 30).astype(float) * 1.5
    return s

# ============================================================
# 5. Backtest Engine
# ============================================================
HOLD = 10; TOP_N = 15; SL = -0.01; COST = 0.0015
DD_THR = [(-0.03,0.80),(-0.06,0.60),(-0.10,0.40),(-0.14,0.20),(-0.18,0.00)]

def run_bt(score_fn, label, warmup_idx=140, regime_aware=False):
    print(f"\n  ▶ {label}")
    t1 = time.time()
    
    n = len(all_dates)
    eq = 100000.0; peak = eq; dd = 0.0
    pos = {}  # sym -> {price, shares, idx, cp}
    d_eq = []; trades = []; last_rb = -999
    
    for i in range(warmup_idx, n):
        d = all_dates[i]
        g = date_map[d]
        
        # Mark-to-market + SL
        pv = 0
        for sym in list(pos.keys()):
            p = pos[sym]
            rows = g[g['sym']==sym]
            if len(rows)>0:
                cp = rows.iloc[0]['close']
                p['cp'] = cp
                r = cp/p['price']-1
                if r <= SL:
                    eq += p['shares']*cp*(1-COST/2)
                    trades.append({'ret':r,'reason':'SL'})
                    del pos[sym]
                else:
                    pv += p['shares']*cp
            else:
                pv += p['shares']*p.get('cp', p['price'])
        
        teq = eq + pv
        if teq > peak: peak = teq
        dd = teq/peak - 1
        d_eq.append(teq)
        
        if i - last_rb < HOLD: continue
        last_rb = i
        
        # DD-based sizing
        pp = 1.0
        for dl, pc in DD_THR:
            if dd <= dl: pp = pc; break
        
        # Market overlay
        mr = g['mkt_ret20'].mean() if 'mkt_ret20' in g.columns else 0
        br = g['breadth'].mean() if 'breadth' in g.columns else 0.5
        
        if mr < -0.05 and br < 0.35:
            regime = 'bear'; pp = min(pp, 0.5)
        elif mr < 0 or br < 0.4:
            regime = 'cautious'; pp = min(pp, 0.8)
        else:
            regime = 'bull'
        
        # Score
        if regime_aware:
            scores = score_fn(g, regime)
        else:
            scores = score_fn(g)
        
        g2 = g.copy(); g2['_s'] = scores
        mask = (g2['close']>=3)&(g2['close']<=200)&(~g2['sym'].str.contains('ST|退市',na=False))&(g2['volume']>0)
        g2 = g2[mask]
        if len(g2) < TOP_N: continue
        
        top = g2.nlargest(TOP_N, '_s')
        tgt = set(top['sym'].tolist())
        
        # Sell non-target
        for sym in list(pos.keys()):
            if sym not in tgt:
                p = pos[sym]; cp = p.get('cp',p['price'])
                eq += p['shares']*cp*(1-COST/2)
                trades.append({'ret':cp/p['price']-1,'reason':'rebal'})
                del pos[sym]
        
        # Buy new
        cash = eq*pp; per = cash/TOP_N
        for _, row in top.iterrows():
            sym = row['sym']
            if sym in pos or per<=0 or eq<=0: continue
            price = row['close']
            shares = per/(price*(1+COST/2))
            cost = shares*price*(1+COST/2)
            if cost > eq: continue
            eq -= cost
            pos[sym] = {'price':price,'shares':shares,'idx':i,'cp':price}
    
    # Close remaining
    for sym, p in list(pos.items()):
        cp = p.get('cp',p['price'])
        eq += p['shares']*cp*(1-COST/2)
        trades.append({'ret':cp/p['price']-1,'reason':'end'})
    
    # Metrics
    darr = np.array(d_eq)
    days_span = (all_dates[-1] - all_dates[warmup_idx]).days
    days_span = max(days_span, 1)
    cagr = (darr[-1]/darr[0])**(365/days_span) - 1
    
    dr = np.diff(darr)/darr[:-1]
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    neg = dr[dr<0]
    sortino = dr.mean()/neg.std()*np.sqrt(252) if len(neg)>0 and neg.std()>0 else 0
    
    # Max DD from daily
    running_max = np.maximum.accumulate(darr)
    drawdowns = darr/running_max - 1
    max_dd = drawdowns.min()
    
    tl = pd.DataFrame(trades) if trades else pd.DataFrame()
    nt = len(tl)
    if nt > 0:
        wr = (tl['ret']>0).mean()
        aw = tl[tl['ret']>0]['ret'].mean() if (tl['ret']>0).any() else 0
        al = tl[tl['ret']<0]['ret'].mean() if (tl['ret']<0).any() else 0
        slr = (tl['reason']=='SL').mean()
        plr = abs(aw/al) if al!=0 else 0
    else:
        wr=aw=al=slr=plr=0
    
    elapsed = time.time()-t1
    r = {'label':label,'cagr':round(cagr*100,2),'sharpe':round(sharpe,3),'sortino':round(sortino,3),
         'max_dd':round(max_dd*100,2),'win_rate':round(wr*100,1),'pl_ratio':round(plr,2),
         'avg_win':round(aw*100,2),'avg_loss':round(al*100,2),'sl_rate':round(slr*100,1),
         'n_trades':nt,'elapsed':round(elapsed,1)}
    print(f"    ✅ Sharpe={sharpe:.3f} CAGR={cagr*100:.1f}% DD={max_dd*100:.1f}% WR={wr*100:.1f}% P/L={plr:.2f} SL={slr*100:.0f}% ({nt}t, {elapsed:.0f}s)")
    return r

# ============================================================
# 6. Run All
# ============================================================
print("\n" + "="*60)
print("🚀 Running 6 strategies...")
print("="*60)

results = []
for fn, label, ra in [
    (score_v21, "A: v2.1 baseline", False),
    (score_flow_heavy, "B: Flow-heavy (60/40)", False),
    (score_all_flow, "C: Pure flow (no tech)", False),
    (score_flow_reversal, "D: Flow+reversal only", False),
    (score_extended, "E: Extended (v2.1+new)", False),
    (score_regime_adaptive, "F: Regime-adaptive", True),
]:
    r = run_bt(fn, label, regime_aware=ra)
    if r: results.append(r)

# Summary
print("\n" + "="*60)
print("📊 RESULTS (sorted by Sharpe)")
print("="*60)
rdf = pd.DataFrame(results).sort_values('sharpe', ascending=False)
print(f"\n{'Strategy':<35} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'P/L':>6} {'SL%':>6} {'#T':>6}")
print("-"*90)
for _, r in rdf.iterrows():
    mark = " ⭐" if r['sharpe'] == rdf['sharpe'].max() else ""
    print(f"{r['label']:<35} {r['sharpe']:>8.3f} {r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['win_rate']:>6.1f} {r['pl_ratio']:>6.2f} {r['sl_rate']:>6.1f} {r['n_trades']:>6}{mark}")

# Save
with open('research/rule_alpha_v3_experiments.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved: research/rule_alpha_v3_experiments.json")
print(f"Total: {time.time()-t0:.0f}s")
