#!/usr/bin/env python3
"""
rule-alpha-v3.3 — Regime-Conditional Scoring
Key insight: Reversal has NEGATIVE alpha in bull markets, POSITIVE in bear.
Flow is mildly positive in all regimes.
Solution: Score differently based on market state.

Also testing:
- Momentum filter: only buy reversal when market is down
- Dynamic factor weight by regime
"""
import pandas as pd, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"📊 rule-alpha-v3.3 regime-conditional {time.strftime('%Y-%m-%d %H:%M')}")

# ============================================================
# 1. Load & Features
# ============================================================
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

df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close']>=3) & (df['close']<=200)].copy()
df = df[df['volume']>0].copy()
df = df.sort_values(['sym','date']).reset_index(drop=True)

# Features
for w in [5,10,20,60]:
    df[f'ret{w}'] = df.groupby('sym')['close'].pct_change(w)
for w in [5,10,20]:
    df[f'ma{w}'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(w, min_periods=3).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20'].replace(0, np.nan)
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=5).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan)).fillna(50)
df['rsi_14'] = df['rsi_14'].fillna(50)
for col in ['total_net','lg_net','md_net','elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
df['total_net_20d'] = df.groupby('sym')['total_net'].transform(lambda x: x.rolling(20, min_periods=1).sum())
df['flow_mom'] = df['total_net_5d'] - df['total_net_20d'] / 4
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x.fillna(0) > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform(lambda x: x.fillna(0).mean())
df = df.replace([np.inf, -np.inf], np.nan)
print(f"Data: {len(df):,} rows, {df['sym'].nunique()} ({time.time()-t0:.0f}s)")

all_dates = sorted(df['date'].unique())
date_map = {d: g.reset_index(drop=True) for d, g in df.groupby('date')}
print(f"{len(all_dates)} dates ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Scoring Functions
# ============================================================
def rrank(arr):
    n=len(arr)
    if n<=1: return np.zeros(n)
    order=arr.argsort(); rank=np.empty(n,dtype=float); rank[order]=np.arange(n,dtype=float)
    return rank/max(n-1,1)

def score_regime_conditional(g, regime='bull'):
    """Regime-conditional: flow everywhere, reversal only in bear"""
    n=len(g); s=np.zeros(n)
    
    # FLOW: always present, always positive IC
    s += rrank(g['total_net_5d'].fillna(0).values)*2.5
    s += rrank(g['md_net_5d'].fillna(0).values)*2
    s += rrank(g['elg_net_5d'].fillna(0).values)*1.5
    
    # LOW VOL: always useful
    v20=g['vol20'].fillna(g['vol20'].median()).values
    v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*1.5
    
    if regime == 'bull':
        # Bull: NO reversal. Flow + low vol + mild RSI
        rsi = g['rsi_14'].fillna(50).values
        s += ((rsi >= 40) & (rsi <= 70)).astype(float)*1  # Not overbought
        # MA alignment
        above_ma = (g['close'].values > g['ma20'].fillna(0).values).astype(float)
        s += above_ma*1
        
    elif regime == 'cautious':
        # Cautious: mild reversal + flow + RSI
        s += np.clip(-g['ret20'].fillna(0).values,-0.15,0.15)*1  # Mild reversal
        rsi = g['rsi_14'].fillna(50).values
        s += ((rsi >= 25) & (rsi <= 50)).astype(float)*1
        
    else:  # bear
        # Bear: STRONG reversal + flow + deep oversold
        s += np.clip(-g['ret20'].fillna(0).values,-0.3,0.3)*3  # Strong reversal
        s += np.clip(-g['ma20_bias'].fillna(0).values,-0.2,0.2)*1.5  # MA deviation
        rsi = g['rsi_14'].fillna(50).values
        s += (rsi < 35).astype(float)*1.5  # Oversold
        s += rrank(g['lg_net_5d'].fillna(0).values)*1  # Big money flow
    
    return s

def score_bear_only_reversal(g, regime='bull'):
    """Only active in bear. Cash in bull/cautious."""
    if regime != 'bear':
        return np.zeros(len(g))
    n=len(g); s=np.zeros(n)
    s += np.clip(-g['ret20'].fillna(0).values,-0.3,0.3)*3
    s += rrank(g['total_net_5d'].fillna(0).values)*2
    s += rrank(g['md_net_5d'].fillna(0).values)*1.5
    v20=g['vol20'].fillna(g['vol20'].median()).values
    v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*2
    rsi = g['rsi_14'].fillna(50).values
    s += (rsi < 30).astype(float)*2
    return s

def score_flow_always_reversal_bear(g, regime='bull'):
    """Flow in all regimes, reversal boost in bear"""
    n=len(g); s=np.zeros(n)
    s += rrank(g['total_net_5d'].fillna(0).values)*2.5
    s += rrank(g['md_net_5d'].fillna(0).values)*2
    s += rrank(g['elg_net_5d'].fillna(0).values)*1.5
    v20=g['vol20'].fillna(g['vol20'].median()).values
    v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*1
    if regime == 'bear':
        s += np.clip(-g['ret20'].fillna(0).values,-0.3,0.3)*3
        rsi = g['rsi_14'].fillna(50).values
        s += (rsi < 35).astype(float)*1.5
    return s

def score_v21_regime_filter(g, regime='bull'):
    """v2.1 scoring but zero out in bull (where reversal has negative alpha)"""
    if regime == 'bull':
        # In bull, use flow-only (reversal has negative alpha)
        n=len(g); s=np.zeros(n)
        s += rrank(g['total_net_5d'].fillna(0).values)*3
        s += rrank(g['md_net_5d'].fillna(0).values)*2
        s += rrank(g['elg_net_5d'].fillna(0).values)*1.5
        v20=g['vol20'].fillna(g['vol20'].median()).values
        v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
        s += (1-rrank(v20))*1.5
        return s
    else:
        # In cautious/bear, use original v2.1
        n=len(g); s=np.zeros(n)
        s += np.clip(-g['ret20'].fillna(0).values,-0.3,0.3)*3
        s += rrank(g['total_net_5d'].fillna(0).values)*2
        v20=g['vol20'].fillna(g['vol20'].median()).values
        v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
        s += (1-rrank(v20))*2
        s += (g['rsi_14'].fillna(50).values<35).astype(float)*1.5
        s += rrank(g['lg_net_5d'].fillna(0).values)*1
        s += np.clip(-g['ma20_bias'].fillna(0).values,-0.2,0.2)*1
        return s

# ============================================================
# 3. Backtest
# ============================================================
HOLD=10; TOP_N=15; SL=-0.01; COST=0.0015
DD_THR=[(-0.03,0.80),(-0.06,0.60),(-0.10,0.40),(-0.14,0.20),(-0.18,0.00)]

def yyyymmdd_to_dt(d):
    return pd.Timestamp(year=d//10000, month=(d//100)%100, day=d%100)

def run_bt(score_fn, label, sl_pct=-1, warmup=180, regime_aware=True):
    sl = sl_pct/100
    nd=len(all_dates); eq=100000.0; peak=eq; dd=0.0
    pos={}; d_eq=[]; d_dates=[]; trades=[]; last_rb=-999
    
    for i in range(warmup, nd):
        d=all_dates[i]; g=date_map[d]
        if len(g)==0: continue
        
        pv=0
        for sy in list(pos.keys()):
            p=pos[sy]; rows=g[g['sym']==sy]
            if len(rows)>0:
                cp=float(rows.iloc[0]['close']); p['cp']=cp; r=cp/p['price']-1
                if sl<0 and r<=sl:
                    eq+=p['shares']*cp*(1-COST/2)
                    trades.append(r); del pos[sy]
                else: pv+=p['shares']*cp
            else: pv+=p['shares']*p.get('cp',p['price'])
        
        teq=eq+pv
        if teq>peak: peak=teq
        dd=teq/peak-1
        d_eq.append(teq); d_dates.append(d)
        
        if i-last_rb<HOLD: continue
        last_rb=i
        
        pp=1.0
        for dl,pc in DD_THR:
            if dd<=dl: pp=pc; break
        
        mr=float(g['mkt_ret20'].mean())
        br=float(g['breadth'].mean())
        if mr<-0.05 and br<0.35: regime='bear'; pp=min(pp,0.5)
        elif mr<0 or br<0.4: regime='cautious'; pp=min(pp,0.8)
        else: regime='bull'
        
        scores = score_fn(g, regime) if regime_aware else score_fn(g)
        
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
            price=float(row['close']); shares=per/(price*(1+COST/2))
            cost=shares*price*(1+COST/2)
            if cost>eq: continue
            eq-=cost
            pos[sy]={'price':price,'shares':shares,'idx':i,'cp':price}
    
    for sy,p in list(pos.items()):
        cp=p.get('cp',p['price']); eq+=p['shares']*cp*(1-COST/2)
        trades.append(cp/p['price']-1)
    
    darr=np.array(d_eq)
    if len(darr)<10: return None
    dt_s=yyyymmdd_to_dt(d_dates[0]); dt_e=yyyymmdd_to_dt(d_dates[-1])
    days_span=max((dt_e-dt_s).days,100)
    cagr=(darr[-1]/darr[0])**(365.0/days_span)-1
    dr=np.diff(darr)/darr[:-1]
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    neg=dr[dr<0]
    sortino=dr.mean()/neg.std()*np.sqrt(252) if len(neg)>0 and neg.std()>0 else 0
    rm=np.maximum.accumulate(darr); max_dd=(darr/rm-1).min()
    
    ta=np.array(trades); nt=len(ta)
    if nt>0:
        wr=(ta>0).mean(); aw=ta[ta>0].mean() if (ta>0).any() else 0
        al=ta[ta<0].mean() if (ta<0).any() else 0
        plr=abs(aw/al) if al!=0 else 0
    else: wr=aw=al=plr=0
    
    return {'label':label,'cagr':round(cagr*100,1),'sharpe':round(sharpe,3),'sortino':round(sortino,3),
            'max_dd':round(max_dd*100,1),'wr':round(wr*100,1),'plr':round(plr,2),'nt':nt}

# ============================================================
# 4. Run
# ============================================================
print("\n" + "="*60)
print("🚀 Regime-Conditional Scoring")
print("="*60)

results = []
for fn, label, ra in [
    (score_regime_conditional, "A: Regime-conditional (flow+bear-rev)", True),
    (score_bear_only_reversal, "B: Bear-only reversal", True),
    (score_flow_always_reversal_bear, "C: Flow always + bear reversal", True),
    (score_v21_regime_filter, "D: v2.1 regime-filtered", True),
    (lambda g: score_v21_regime_filter(g,'bull'), "E: v2.1 original (baseline)", False),
]:
    t1=time.time()
    print(f"  ▶ {label}...", end="", flush=True)
    r = run_bt(fn, label, sl_pct=-1, regime_aware=ra)
    if r:
        r['elapsed']=round(time.time()-t1,1)
        results.append(r)
        print(f" Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% DD={r['max_dd']:.1f}% WR={r['wr']:.1f}% P/L={r['plr']:.2f} ({r['elapsed']:.0f}s)")

# Test best with SL-3%
best = max(results, key=lambda x: x['sharpe'])
best_label = best['label']
best_fn_map = {
    "A: Regime-conditional (flow+bear-rev)": score_regime_conditional,
    "B: Bear-only reversal": score_bear_only_reversal,
    "C: Flow always + bear reversal": score_flow_always_reversal_bear,
    "D: v2.1 regime-filtered": score_v21_regime_filter,
}
if best_label in best_fn_map:
    for sl in [-3, -5]:
        r = run_bt(best_fn_map[best_label], f"{best_label}_SL{sl}%", sl_pct=sl, regime_aware=True)
        if r:
            r['elapsed']=0; results.append(r)
            print(f"  {best_label} SL{sl}%: Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% DD={r['max_dd']:.1f}%")

# Summary
print("\n" + "="*60)
print("📊 RESULTS (sorted by Sharpe)")
print("="*60)
rdf=pd.DataFrame(results).sort_values('sharpe',ascending=False)
print(f"\n{'Strategy':<45} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'DD%':>8} {'WR%':>6} {'P/L':>6}")
print("-"*90)
for _,r in rdf.iterrows():
    mark=" ⭐" if r['sharpe']==rdf['sharpe'].max() else ""
    print(f"{r['label']:<45} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} {r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['wr']:>6.1f} {r['plr']:>6.2f}{mark}")

with open('research/rule_alpha_v3_3_regime.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved. Total: {time.time()-t0:.0f}s")
