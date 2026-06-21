#!/usr/bin/env python3
"""
rule-alpha-v3.0 — Full universe backtest with groupby.transform()
Faster than per-stock numpy loops on large datasets.
"""
import pandas as pd, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"📊 rule-alpha-v3.0 FULL backtest {time.strftime('%Y-%m-%d %H:%M')}")

# ============================================================
# 1. Load & Merge
# ============================================================
print("1. Loading data...")
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code':'sym','Date':'date','O':'open','H':'high','L':'low','C':'close','V':'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm','md','lg','elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')
for c in ['total_net','lg_net','md_net','elg_net']:
    df[c] = df[c].fillna(0)

# Filter
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close']>=3) & (df['close']<=200)].copy()
df = df[df['volume']>0].copy()
df = df.sort_values(['sym','date']).reset_index(drop=True)
print(f"   {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Compute Features (groupby.transform — vectorized)
# ============================================================
print("2. Computing features (groupby.transform)...")
t1 = time.time()

# Returns
for w in [5, 10, 20]:
    df[f'ret{w}'] = df.groupby('sym')['close'].pct_change(w)
print(f"   Returns done ({time.time()-t1:.0f}s)")

# MA20 bias
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20'].replace(0, np.nan)
print(f"   MA20 done ({time.time()-t1:.0f}s)")

# Volatility 20d
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=5).std())
print(f"   Vol20 done ({time.time()-t1:.0f}s)")

# RSI(14)
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)
print(f"   RSI done ({time.time()-t1:.0f}s)")

# Money flow 5d sums
for col in ['total_net','lg_net','md_net','elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
print(f"   Flow5d done ({time.time()-t1:.0f}s)")

# Money flow 20d sum (for flow momentum)
df['total_net_20d'] = df.groupby('sym')['total_net'].transform(lambda x: x.rolling(20, min_periods=1).sum())
df['flow_mom'] = df['total_net_5d'] - df['total_net_20d'] / 4
print(f"   Flow momentum done ({time.time()-t1:.0f}s)")

# Volume ratio 5/20
df['vol5'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['vol20r'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['vol_price_div'] = -df['ret5'] * (df['vol5'] / df['vol20r'].replace(0, np.nan) - 1)
print(f"   VPB done ({time.time()-t1:.0f}s)")

# Forward 10d return
df['fwd10'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

# Market state
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x.fillna(0) > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform(lambda x: x.fillna(0).mean())

# Cleanup
df = df.replace([np.inf, -np.inf], np.nan)
print(f"   ALL features done ({time.time()-t1:.0f}s)")

# ============================================================
# 3. Pre-group by date
# ============================================================
print("3. Pre-grouping by date...")
all_dates = sorted(df['date'].unique())
date_map = {d: g.reset_index(drop=True) for d, g in df.groupby('date')}
print(f"   {len(all_dates)} dates ({time.time()-t0:.0f}s)")

# ============================================================
# 4. Scoring Functions (using .values for speed)
# ============================================================
def rrank(arr):
    """Fast percentile rank."""
    n = len(arr)
    if n <= 1: return np.zeros(n)
    order = arr.argsort()
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    return rank / max(n - 1, 1)

def score_v21(g):
    """v2.1 baseline — static weights."""
    n = len(g)
    s = np.zeros(n)
    s += np.clip(-g['ret20'].values, -0.3, 0.3) * 3
    s += rrank(g['total_net_5d'].values) * 2
    v20 = g['vol20'].values
    v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
    s += (1 - rrank(v20)) * 2
    s += (g['rsi_14'].values < 35).astype(float) * 1.5
    s += rrank(g['lg_net_5d'].values) * 1
    s += np.clip(-g['ma20_bias'].values, -0.2, 0.2) * 1
    return s

def score_flow_heavy(g):
    """60% flow, 40% tech."""
    n = len(g); s = np.zeros(n)
    s += rrank(g['total_net_5d'].values) * 2.5
    s += rrank(g['md_net_5d'].values) * 2.0
    s += rrank(g['elg_net_5d'].values) * 1.5
    s += rrank(g['lg_net_5d'].values) * 1.0
    s += np.clip(-g['ret20'].values, -0.3, 0.3) * 2
    v20 = g['vol20'].values
    v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
    s += (1 - rrank(v20)) * 1
    return s

def score_all_flow(g):
    """Pure flow — no technicals."""
    n = len(g); s = np.zeros(n)
    s += rrank(g['total_net_5d'].values) * 2
    s += rrank(g['md_net_5d'].values) * 2
    s += rrank(g['elg_net_5d'].values) * 2
    s += rrank(g['lg_net_5d'].values) * 1.5
    s += rrank(g['flow_mom'].values) * 0.5
    return s

def score_flow_reversal(g):
    """Flow + reversal only."""
    n = len(g); s = np.zeros(n)
    s += rrank(g['total_net_5d'].values) * 3
    s += rrank(g['md_net_5d'].values) * 2
    s += np.clip(-g['ret20'].values, -0.3, 0.3) * 3
    return s

def score_extended(g):
    """v2.1 + new factors."""
    n = len(g); s = np.zeros(n)
    s += np.clip(-g['ret20'].values, -0.3, 0.3) * 2.5
    s += rrank(g['total_net_5d'].values) * 2
    v20 = g['vol20'].values
    v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
    s += (1 - rrank(v20)) * 1.5
    s += (g['rsi_14'].values < 35).astype(float) * 1
    s += rrank(g['lg_net_5d'].values) * 0.5
    s += np.clip(-g['ma20_bias'].values, -0.2, 0.2) * 0.5
    s += rrank(g['md_net_5d'].values) * 1.5
    s += rrank(g['elg_net_5d'].values) * 1
    s += rrank(g['flow_mom'].values) * 0.5
    s += rrank(g['vol_price_div'].values) * 0.5
    return s

def score_regime_adaptive(g, regime='bull'):
    """Different weights per regime."""
    n = len(g); s = np.zeros(n)
    if regime == 'bull':
        s += np.clip(-g['ret20'].values, -0.3, 0.3) * 3.5
        s += rrank(g['total_net_5d'].values) * 2.5
        s += rrank(g['md_net_5d'].values) * 1.5
        v20 = g['vol20'].values
        v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
        s += (1 - rrank(v20)) * 1
    elif regime == 'cautious':
        s += rrank(g['total_net_5d'].values) * 3
        s += rrank(g['md_net_5d'].values) * 2
        s += rrank(g['elg_net_5d'].values) * 1.5
        v20 = g['vol20'].values
        v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
        s += (1 - rrank(v20)) * 2
        s += np.clip(-g['ma20_bias'].values, -0.2, 0.2) * 1
    else:
        s += rrank(g['total_net_5d'].values) * 3
        s += rrank(g['md_net_5d'].values) * 3
        v20 = g['vol20'].values
        v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
        s += (1 - rrank(v20)) * 2.5
        s += np.clip(-g['ret20'].values, -0.3, 0.3) * 2
        s += (g['rsi_14'].values < 30).astype(float) * 1.5
    return s

# ============================================================
# 5. Backtest Engine
# ============================================================
HOLD=10; TOP_N=15; SL=-0.01; COST=0.0015
DD_THR=[(-0.03,0.80),(-0.06,0.60),(-0.10,0.40),(-0.14,0.20),(-0.18,0.00)]

def run_bt(score_fn, label, warmup=160, regime_aware=False):
    print(f"\n  ▶ {label}")
    t1 = time.time()
    nd = len(all_dates)
    eq=100000.0; peak=eq; dd=0.0; pos={}; d_eq=[]; trades=[]; last_rb=-999
    
    for i in range(warmup, nd):
        d = all_dates[i]
        g = date_map[d]
        if len(g) == 0: continue
        
        # Mark-to-market + SL
        pv=0
        for sy in list(pos.keys()):
            p=pos[sy]
            rows=g[g['sym']==sy]
            if len(rows)>0:
                cp=float(rows.iloc[0]['close']); p['cp']=cp
                r=cp/p['price']-1
                if r<=SL:
                    eq+=p['shares']*cp*(1-COST/2)
                    trades.append(r); del pos[sy]
                else: pv+=p['shares']*cp
            else: pv+=p['shares']*p.get('cp',p['price'])
        
        teq=eq+pv
        if teq>peak: peak=teq
        dd=teq/peak-1
        d_eq.append(teq)
        
        if i-last_rb<HOLD: continue
        last_rb=i
        
        pp=1.0
        for dl,pc in DD_THR:
            if dd<=dl: pp=pc; break
        
        mr=float(g['mkt_ret20'].mean()) if 'mkt_ret20' in g.columns else 0
        br=float(g['breadth'].mean()) if 'breadth' in g.columns else 0.5
        
        if mr<-0.05 and br<0.35: reg='bear'; pp=min(pp,0.5)
        elif mr<0 or br<0.4: reg='cautious'; pp=min(pp,0.8)
        else: reg='bull'
        
        scores = score_fn(g, reg) if regime_aware else score_fn(g)
        
        g2=g.copy(); g2['_s']=scores
        mask=(g2['close']>=3)&(g2['close']<=200)&(~g2['sym'].str.contains('ST|退市',na=False))&(g2['volume']>0)
        g2=g2[mask]
        if len(g2)<TOP_N: continue
        
        top=g2.nlargest(TOP_N,'_s')
        tgt=set(top['sym'].tolist())
        
        for sy in list(pos.keys()):
            if sy not in tgt:
                p=pos[sy]; cp=p.get('cp',p['price'])
                eq+=p['shares']*cp*(1-COST/2)
                trades.append(cp/p['price']-1); del pos[sy]
        
        cash=eq*pp; per=cash/TOP_N
        for _,row in top.iterrows():
            sy=row['sym']
            if sy in pos or per<=0 or eq<=0: continue
            price=float(row['close'])
            shares=per/(price*(1+COST/2))
            cost=shares*price*(1+COST/2)
            if cost>eq: continue
            eq-=cost
            pos[sy]={'price':price,'shares':shares,'cp':price}
    
    for sy,p in list(pos.items()):
        cp=p.get('cp',p['price'])
        eq+=p['shares']*cp*(1-COST/2)
        trades.append(cp/p['price']-1)
    
    darr=np.array(d_eq)
    if len(darr)<10: return None
    
    days_span=max(all_dates[-1]-all_dates[warmup], 100)
    cagr=(darr[-1]/darr[0])**(365/days_span)-1
    dr=np.diff(darr)/darr[:-1]
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    neg=dr[dr<0]
    sortino=dr.mean()/neg.std()*np.sqrt(252) if len(neg)>0 and neg.std()>0 else 0
    rm=np.maximum.accumulate(darr)
    max_dd=(darr/rm-1).min()
    
    ta=np.array(trades)
    nt=len(ta)
    if nt>0:
        wr=(ta>0).mean()
        aw=ta[ta>0].mean() if (ta>0).any() else 0
        al=ta[ta<0].mean() if (ta<0).any() else 0
        plr=abs(aw/al) if al!=0 else 0
    else: wr=aw=al=plr=0
    
    elapsed=time.time()-t1
    r={'label':label,'cagr':round(cagr*100,2),'sharpe':round(sharpe,3),'sortino':round(sortino,3),
       'max_dd':round(max_dd*100,2),'win_rate':round(wr*100,1),'pl_ratio':round(plr,2),
       'avg_win':round(aw*100,2),'avg_loss':round(al*100,2),'n_trades':nt,'elapsed':round(elapsed,1)}
    print(f"    ✅ Sharpe={sharpe:.3f} CAGR={cagr*100:.1f}% DD={max_dd*100:.1f}% WR={wr*100:.1f}% P/L={plr:.2f} ({nt}t, {elapsed:.0f}s)")
    return r

# ============================================================
# 6. Run
# ============================================================
print("\n" + "="*60)
print("🚀 Running 6 strategies...")
print("="*60)

results = []
for fn,label,ra in [
    (score_v21, "A: v2.1 baseline", False),
    (score_flow_heavy, "B: Flow-heavy (60/40)", False),
    (score_all_flow, "C: Pure flow", False),
    (score_flow_reversal, "D: Flow+reversal", False),
    (score_extended, "E: Extended (v2.1+new)", False),
    (score_regime_adaptive, "F: Regime-adaptive", True),
]:
    r=run_bt(fn,label,regime_aware=ra)
    if r: results.append(r)

print("\n" + "="*60)
print("📊 RESULTS (sorted by Sharpe)")
print("="*60)
rdf=pd.DataFrame(results).sort_values('sharpe',ascending=False)
print(f"\n{'Strategy':<35} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'P/L':>6} {'#T':>6}")
print("-"*80)
for _,r in rdf.iterrows():
    mark=" ⭐" if r['sharpe']==rdf['sharpe'].max() else ""
    print(f"{r['label']:<35} {r['sharpe']:>8.3f} {r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['win_rate']:>6.1f} {r['pl_ratio']:>6.2f} {r['n_trades']:>6}{mark}")

with open('research/rule_alpha_v3_full_universe.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved. Total: {time.time()-t0:.0f}s")
