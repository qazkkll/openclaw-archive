#!/usr/bin/env python3
"""
rule-alpha-v3.1b — SL Sweep with CORRECT CAGR calculation
Fixed YYYYMMDD→days bug.
"""
import pandas as pd, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"📊 rule-alpha-v3.1b SL sweep (fixed) {time.strftime('%Y-%m-%d %H:%M')}")

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
print(f"Data: {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

# Features
for w in [5,10,20]:
    df[f'ret{w}'] = df.groupby('sym')['close'].pct_change(w)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=5).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20'].replace(0, np.nan)
df['vol20'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(20, min_periods=5).std())
delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)
for col in ['total_net','lg_net','md_net','elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
df['total_net_20d'] = df.groupby('sym')['total_net'].transform(lambda x: x.rolling(20, min_periods=1).sum())
df['flow_mom'] = df['total_net_5d'] - df['total_net_20d'] / 4
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x.fillna(0) > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform(lambda x: x.fillna(0).mean())
df = df.replace([np.inf, -np.inf], np.nan)
print(f"Features done ({time.time()-t0:.0f}s)")

all_dates = sorted(df['date'].unique())
date_map = {d: g.reset_index(drop=True) for d, g in df.groupby('date')}

# Date conversion for CAGR
def yyyymmdd_to_dt(d):
    return pd.Timestamp(year=d//10000, month=(d//100)%100, day=d%100)

# ============================================================
# 2. Scoring
# ============================================================
def rrank(arr):
    n = len(arr)
    if n <= 1: return np.zeros(n)
    order = arr.argsort()
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    return rank / max(n - 1, 1)

def score_v21(g):
    n = len(g); s = np.zeros(n)
    s += np.clip(-g['ret20'].values, -0.3, 0.3) * 3
    s += rrank(g['total_net_5d'].values) * 2
    v20 = g['vol20'].values
    v20 = np.where(np.isnan(v20), np.nanmedian(v20), v20)
    s += (1 - rrank(v20)) * 2
    s += (g['rsi_14'].values < 35).astype(float) * 1.5
    s += rrank(g['lg_net_5d'].values) * 1
    s += np.clip(-g['ma20_bias'].values, -0.2, 0.2) * 1
    return s

# ============================================================
# 3. Backtest
# ============================================================
HOLD=10; TOP_N=15; COST=0.0015
DD_THR=[(-0.03,0.80),(-0.06,0.60),(-0.10,0.40),(-0.14,0.20),(-0.18,0.00)]

def run_bt(sl_pct, use_dd=True, warmup=160):
    sl = sl_pct / 100 if sl_pct != 0 else 0
    
    nd = len(all_dates)
    eq=100000.0; peak=eq; dd=0.0; pos={}; d_eq=[]; d_dates=[]; trades=[]; last_rb=-999
    
    for i in range(warmup, nd):
        d = all_dates[i]
        g = date_map[d]
        if len(g) == 0: continue
        
        # Mark-to-market
        pv=0
        for sy in list(pos.keys()):
            p=pos[sy]
            rows=g[g['sym']==sy]
            if len(rows)>0:
                cp=float(rows.iloc[0]['close']); p['cp']=cp
                r=cp/p['price']-1
                if sl < 0 and r <= sl:
                    eq+=p['shares']*cp*(1-COST/2)
                    trades.append({'ret':r,'reason':'SL','days':i-p.get('idx',i)})
                    del pos[sy]
                else: pv+=p['shares']*cp
            else: pv+=p['shares']*p.get('cp',p['price'])
        
        teq=eq+pv
        if teq>peak: peak=teq
        dd=teq/peak-1
        d_eq.append(teq)
        d_dates.append(d)
        
        # Rebalance
        if i-last_rb<HOLD: continue
        last_rb=i
        
        # DD-based sizing
        pp=1.0
        if use_dd:
            for dl,pc in DD_THR:
                if dd<=dl: pp=pc; break
        
        # Market overlay
        mr=float(g['mkt_ret20'].mean())
        br=float(g['breadth'].mean())
        if mr<-0.05 and br<0.35: pp=min(pp,0.5)
        elif mr<0 or br<0.4: pp=min(pp,0.8)
        
        # Score & rank
        scores = score_v21(g)
        g2=g.copy(); g2['_s']=scores
        mask=(g2['close']>=3)&(g2['close']<=200)&(~g2['sym'].str.contains('ST|退市',na=False))&(g2['volume']>0)
        g2=g2[mask]
        if len(g2)<TOP_N: continue
        
        top=g2.nlargest(TOP_N,'_s')
        tgt=set(top['sym'].tolist())
        
        # Sell non-target
        for sy in list(pos.keys()):
            if sy not in tgt:
                p=pos[sy]; cp=p.get('cp',p['price'])
                eq+=p['shares']*cp*(1-COST/2)
                trades.append({'ret':cp/p['price']-1,'reason':'rebal','days':i-p.get('idx',i)})
                del pos[sy]
        
        # Buy new
        cash=eq*pp; per=cash/TOP_N
        for _,row in top.iterrows():
            sy=row['sym']
            if sy in pos or per<=0 or eq<=0: continue
            price=float(row['close'])
            shares=per/(price*(1+COST/2))
            cost=shares*price*(1+COST/2)
            if cost>eq: continue
            eq-=cost
            pos[sy]={'price':price,'shares':shares,'idx':i,'cp':price}
    
    # Close remaining
    for sy,p in list(pos.items()):
        cp=p.get('cp',p['price'])
        eq+=p['shares']*cp*(1-COST/2)
        trades.append({'ret':cp/p['price']-1,'reason':'end','days':0})
    
    darr=np.array(d_eq)
    if len(darr)<10: return None
    
    # CORRECT CAGR using actual dates
    dt_start = yyyymmdd_to_dt(d_dates[0])
    dt_end = yyyymmdd_to_dt(d_dates[-1])
    days_span = (dt_end - dt_start).days
    if days_span < 100: days_span = 100
    
    cagr=(darr[-1]/darr[0])**(365.0/days_span)-1
    
    dr=np.diff(darr)/darr[:-1]
    sharpe=dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    neg=dr[dr<0]
    sortino=dr.mean()/neg.std()*np.sqrt(252) if len(neg)>0 and neg.std()>0 else 0
    rm=np.maximum.accumulate(darr)
    max_dd=(darr/rm-1).min()
    
    tl=pd.DataFrame(trades) if trades else pd.DataFrame()
    nt=len(tl)
    if nt>0:
        wr=(tl['ret']>0).mean()
        aw=tl[tl['ret']>0]['ret'].mean() if (tl['ret']>0).any() else 0
        al=tl[tl['ret']<0]['ret'].mean() if (tl['ret']<0).any() else 0
        slr=(tl[tl['reason']=='SL'].shape[0]/nt) if sl<0 else 0
        plr=abs(aw/al) if al!=0 else 0
    else: wr=aw=al=slr=plr=0
    
    return {'sl':sl_pct,'dd':use_dd,'cagr':round(cagr*100,1),'sharpe':round(sharpe,3),
            'sortino':round(sortino,3),'max_dd':round(max_dd*100,1),'wr':round(wr*100,1),
            'plr':round(plr,2),'slr':round(slr*100,0),'nt':nt,
            'aw':round(aw*100,2),'al':round(al*100,2)}

# ============================================================
# 4. Sweep
# ============================================================
print("\n" + "="*60)
print("🔍 SL SWEEP (v2.1 baseline, full universe)")
print("="*60)

results = []
for sl_val in [0, -1, -2, -3, -5, -8]:
    for use_dd in [True, False]:
        t1=time.time()
        r=run_bt(sl_val, use_dd=use_dd)
        if r:
            r['elapsed']=round(time.time()-t1,1)
            results.append(r)
            dd_str="DD" if use_dd else "noDD"
            print(f"  SL{sl_val}%_{dd_str}: Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% DD={r['max_dd']:.1f}% WR={r['wr']:.1f}% P/L={r['plr']:.2f} SL={r['slr']:.0f}% ({r['nt']}t, {r['elapsed']:.0f}s)")

# Summary
print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
rdf=pd.DataFrame(results).sort_values('sharpe',ascending=False)
print(f"\n{'Config':<20} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'P/L':>6} {'SL%':>5} {'#T':>6}")
print("-"*80)
for _,r in rdf.iterrows():
    dd_str="DD" if r['dd'] else "noDD"
    label=f"SL{r['sl']}%_{dd_str}"
    mark=" ⭐" if r['sharpe']==rdf['sharpe'].max() else ""
    print(f"{label:<20} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} {r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['wr']:>6.1f} {r['plr']:>6.2f} {r['slr']:>5.0f} {r['nt']:>6}{mark}")

with open('research/rule_alpha_v3_1_sl_sweep_fixed.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved. Total: {time.time()-t0:.0f}s")
