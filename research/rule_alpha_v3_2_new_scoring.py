#!/usr/bin/env python3
"""
rule-alpha-v3.2 — New Scoring Functions
The v2.1 scoring has NEGATIVE alpha. Testing fundamentally different approaches.

Key finding from v3.1 diagnostic:
- IC = 0.0476 (barely positive)
- Top-15 alpha = -0.07% per 10d (NEGATIVE)
- Only 4/11 years positive alpha
- The "reversal + flow" combination picks losers

New approaches:
A. Pure flow (no reversal) — flow is the only stable factor
B. Momentum positive (buy winners, not losers) 
C. Flow + RSI mean reversion (oversold bounce)
D. Volume surge + flow (high volume = institutional activity)
E. Trend following (MA alignment + momentum)
F. Combined: flow trend + reversal with MA filter
"""
import pandas as pd, numpy as np, json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"📊 rule-alpha-v3.2 new scorings {time.strftime('%Y-%m-%d %H:%M')}")

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
df['vol5'] = df.groupby('sym')['close'].transform(lambda x: x.pct_change().rolling(5, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=3).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan)).fillna(50)
df['rsi_14'] = df['rsi_14'].fillna(50)

# MACD
df['ema12'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=12, min_periods=5).mean())
df['ema26'] = df.groupby('sym')['close'].transform(lambda x: x.ewm(span=26, min_periods=5).mean())
df['macd'] = df['ema12'] - df['ema26']
df['macd_sig'] = df.groupby('sym')['macd'].transform(lambda x: x.ewm(span=9, min_periods=3).mean())
df['macd_hist'] = df['macd'] - df['macd_sig']

# Money flow
for col in ['total_net','lg_net','md_net','elg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())
    df[f'{col}_20d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(20, min_periods=1).sum())

# Volume surge (vol_5d / vol_20d)
df['vol_avg5'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
df['vol_avg20'] = df.groupby('sym')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['vol_surge'] = df['vol_avg5'] / df['vol_avg20'].replace(0, np.nan)

# Market state
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x.fillna(0) > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform(lambda x: x.fillna(0).mean())

# Forward returns for diagnostics
df['fwd10'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10)/x-1)

df = df.replace([np.inf, -np.inf], np.nan)
print(f"Data: {len(df):,} rows, {df['sym'].nunique()} stocks ({time.time()-t0:.0f}s)")

all_dates = sorted(df['date'].unique())
date_map = {d: g.reset_index(drop=True) for d, g in df.groupby('date')}
print(f"{len(all_dates)} dates ({time.time()-t0:.0f}s)")

# ============================================================
# 2. Scoring Functions
# ============================================================
def rrank(arr):
    n=len(arr)
    if n<=1: return np.zeros(n)
    order=arr.argsort()
    rank=np.empty(n,dtype=float); rank[order]=np.arange(n,dtype=float)
    return rank/max(n-1,1)

def score_v21_old(g):
    """Original v2.1 (known to have negative alpha)"""
    n=len(g); s=np.zeros(n)
    s += np.clip(-g['ret20'].fillna(0).values,-0.3,0.3)*3
    s += rrank(g['total_net_5d'].fillna(0).values)*2
    v20=g['vol20'].fillna(g['vol20'].median()).values; v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*2
    s += (g['rsi_14'].fillna(50).values<35).astype(float)*1.5
    s += rrank(g['lg_net_5d'].fillna(0).values)*1
    s += np.clip(-g['ma20_bias'].fillna(0).values,-0.2,0.2)*1
    return s

def score_pure_flow(g):
    """Pure money flow (only stable factor)"""
    n=len(g); s=np.zeros(n)
    s += rrank(g['total_net_5d'].fillna(0).values)*3
    s += rrank(g['md_net_5d'].fillna(0).values)*2
    s += rrank(g['elg_net_5d'].fillna(0).values)*2
    s += rrank(g['lg_net_5d'].fillna(0).values)*1.5
    # Flow momentum (acceleration)
    fmom = g['total_net_5d'].fillna(0).values - g['total_net_20d'].fillna(0).values/4
    s += rrank(fmom)*1
    return s

def score_momentum(g):
    """Trend following — buy winners"""
    n=len(g); s=np.zeros(n)
    s += g['ret20'].fillna(0).values*3  # POSITIVE momentum (not reversed!)
    s += rrank(g['ret60'].fillna(0).values)*2  # 60d trend
    s += (g['macd_hist'].fillna(0).values > 0).astype(float)*2  # MACD bullish
    s += (g['ma5'].fillna(0).values > g['ma20'].fillna(0).values).astype(float)*1.5  # MA alignment
    s += (1-rrank(g['vol20'].fillna(g['vol20'].median()).values))*1  # Low vol
    return s

def score_flow_rsi_bounce(g):
    """Flow + RSI oversold bounce"""
    n=len(g); s=np.zeros(n)
    s += rrank(g['total_net_5d'].fillna(0).values)*3
    s += rrank(g['md_net_5d'].fillna(0).values)*2
    s += rrank(g['elg_net_5d'].fillna(0).values)*1.5
    rsi = g['rsi_14'].fillna(50).values
    # RSI sweet spot: 25-40 (oversold but not collapsing)
    s += ((rsi >= 25) & (rsi <= 40)).astype(float)*2
    s += ((rsi >= 20) & (rsi <= 50)).astype(float)*0.5
    # Low vol = more reliable bounce
    v20=g['vol20'].fillna(g['vol20'].median()).values; v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*1.5
    return s

def score_vol_flow(g):
    """Volume surge + money flow (institutional activity)"""
    n=len(g); s=np.zeros(n)
    # Volume surge
    vs = g['vol_surge'].fillna(1).values
    s += rrank(vs)*2
    # High volume + positive flow = accumulation
    flow = g['total_net_5d'].fillna(0).values
    vol_flow = vs * np.sign(flow) * np.abs(flow)
    s += rrank(vol_flow)*3
    # Money flow
    s += rrank(g['md_net_5d'].fillna(0).values)*2
    s += rrank(g['lg_net_5d'].fillna(0).values)*1.5
    # Low vol bonus
    v20=g['vol20'].fillna(g['vol20'].median()).values; v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*1
    return s

def score_combined_v3(g):
    """Combined: flow trend + RSI + vol + MA filter"""
    n=len(g); s=np.zeros(n)
    # Flow (core, 40% weight)
    s += rrank(g['total_net_5d'].fillna(0).values)*2.5
    s += rrank(g['md_net_5d'].fillna(0).values)*1.5
    # RSI range (20% weight)
    rsi = g['rsi_14'].fillna(50).values
    s += ((rsi >= 30) & (rsi <= 60)).astype(float)*1.5
    # Low vol (15% weight)
    v20=g['vol20'].fillna(g['vol20'].median()).values; v20=np.where(np.isnan(v20),np.nanmedian(v20),v20)
    s += (1-rrank(v20))*1
    # MA filter (15% weight) — price above MA20 = trend intact
    above_ma = (g['close'].values > g['ma20'].fillna(0).values).astype(float)
    s += above_ma*1
    # Mild reversal (10% weight) — small dip is OK
    ret20 = g['ret20'].fillna(0).values
    s += np.clip(-ret20, -0.1, 0.1)*0.5  # Very mild reversal weight
    return s

# ============================================================
# 3. Backtest Engine
# ============================================================
HOLD=10; TOP_N=15; COST=0.0015
DD_THR=[(-0.03,0.80),(-0.06,0.60),(-0.10,0.40),(-0.14,0.20),(-0.18,0.00)]

def yyyymmdd_to_dt(d):
    return pd.Timestamp(year=d//10000, month=(d//100)%100, day=d%100)

def run_bt(score_fn, label, sl_pct=-3, warmup=180):
    sl = sl_pct / 100
    nd = len(all_dates)
    eq=100000.0; peak=eq; dd=0.0; pos={}; d_eq=[]; d_dates=[]; trades=[]; last_rb=-999
    
    for i in range(warmup, nd):
        d = all_dates[i]
        g = date_map[d]
        if len(g)==0: continue
        
        pv=0
        for sy in list(pos.keys()):
            p=pos[sy]; rows=g[g['sym']==sy]
            if len(rows)>0:
                cp=float(rows.iloc[0]['close']); p['cp']=cp
                r=cp/p['price']-1
                if sl<0 and r<=sl:
                    eq+=p['shares']*cp*(1-COST/2)
                    trades.append({'ret':r,'reason':'SL'}); del pos[sy]
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
        if mr<-0.05 and br<0.35: pp=min(pp,0.5)
        elif mr<0 or br<0.4: pp=min(pp,0.8)
        
        scores = score_fn(g)
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
                trades.append({'ret':cp/p['price']-1,'reason':'rebal'}); del pos[sy]
        
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
        trades.append({'ret':cp/p['price']-1,'reason':'end'})
    
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
    
    tl=pd.DataFrame(trades) if trades else pd.DataFrame()
    nt=len(tl)
    if nt>0:
        wr=(tl['ret']>0).mean()
        aw=tl[tl['ret']>0]['ret'].mean() if (tl['ret']>0).any() else 0
        al=tl[tl['ret']<0]['ret'].mean() if (tl['ret']<0).any() else 0
        slr=(tl[tl['reason']=='SL'].shape[0]/nt)
        plr=abs(aw/al) if al!=0 else 0
    else: wr=aw=al=slr=plr=0
    
    return {'label':label,'cagr':round(cagr*100,1),'sharpe':round(sharpe,3),'sortino':round(sortino,3),
            'max_dd':round(max_dd*100,1),'wr':round(wr*100,1),'plr':round(plr,2),
            'slr':round(slr*100,0),'nt':nt,'aw':round(aw*100,2),'al':round(al*100,2)}

# ============================================================
# 4. Run
# ============================================================
print("\n" + "="*60)
print("🚀 Testing 6 new scoring functions (SL-3%, DD-based)")
print("="*60)

results = []
for fn, label in [
    (score_v21_old, "A: v2.1 old (baseline)"),
    (score_pure_flow, "B: Pure flow (no reversal)"),
    (score_momentum, "C: Momentum (buy winners)"),
    (score_flow_rsi_bounce, "D: Flow + RSI bounce"),
    (score_vol_flow, "E: Vol surge + flow"),
    (score_combined_v3, "F: Combined v3 (flow+RSI+vol+MA)"),
]:
    t1=time.time()
    print(f"  ▶ {label}...", end="", flush=True)
    r = run_bt(fn, label, sl_pct=-3)
    if r:
        r['elapsed']=round(time.time()-t1,1)
        results.append(r)
        print(f" Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% DD={r['max_dd']:.1f}% WR={r['wr']:.1f}% P/L={r['plr']:.2f} ({r['elapsed']:.0f}s)")

# Also test best with SL-1% and SL0%
best = max(results, key=lambda x: x['sharpe'])
best_label = best['label']
best_fn = [fn for fn, l in [
    (score_v21_old, "A: v2.1 old (baseline)"),
    (score_pure_flow, "B: Pure flow (no reversal)"),
    (score_momentum, "C: Momentum (buy winners)"),
    (score_flow_rsi_bounce, "D: Flow + RSI bounce"),
    (score_vol_flow, "E: Vol surge + flow"),
    (score_combined_v3, "F: Combined v3 (flow+RSI+vol+MA)"),
] if l == best_label][0]

print(f"\n  Best: {best_label}")
for sl in [-1, 0]:
    r = run_bt(best_fn, f"{best_label}_SL{sl}%", sl_pct=sl)
    if r:
        results.append(r)
        print(f"  {best_label} SL{sl}%: Sharpe={r['sharpe']:.3f} CAGR={r['cagr']:.1f}% DD={r['max_dd']:.1f}%")

# Summary
print("\n" + "="*60)
print("📊 RESULTS")
print("="*60)
rdf=pd.DataFrame(results).sort_values('sharpe',ascending=False)
print(f"\n{'Strategy':<40} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'DD%':>8} {'WR%':>6} {'P/L':>6} {'SL%':>5}")
print("-"*90)
for _,r in rdf.iterrows():
    mark=" ⭐" if r['sharpe']==rdf['sharpe'].max() else ""
    print(f"{r['label']:<40} {r['sharpe']:>8.3f} {r['sortino']:>8.3f} {r['cagr']:>8.1f} {r['max_dd']:>8.1f} {r['wr']:>6.1f} {r['plr']:>6.2f} {r['slr']:>5.0f}{mark}")

with open('research/rule_alpha_v3_2_new_scoring.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved. Total: {time.time()-t0:.0f}s")
