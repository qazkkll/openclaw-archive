#!/usr/bin/env python3
"""
V4.1 Enhanced Sweep - Based on original bt_v41_sweep.py framework
Adds: ADX confirmation + Three-regime switching (bull/sideways/bear)
"""
import json, os, warnings, time
import numpy as np
from itertools import product
warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
OUTPUT = "/home/admin/.openclaw/workspace/data/bt_v41_enhanced_results.json"

pool_data = json.load(open(UNIVERSE))
tickers = pool_data['tickers']
print("US stocks: %d" % len(tickers))

# SPY data for regime
import yfinance as yf
spy_raw = yf.download('SPY', start="2013-01-01", end="2026-06-01", progress=False)
spy = spy_raw['Close'].squeeze()
spy_dates = [d.strftime('%Y-%m-%d') for d in list(spy_raw.index)]
spy_prices = [float(spy.iloc[i]) for i in range(len(spy))]

def get_spy(date):
    for off in range(5):
        d = date[:8] + str(int(date[8:10]) + off + 1).zfill(2) if off > 0 else date
        if d in spy_dates: return spy_prices[spy_dates.index(d)]
    return None

# SPY MA200
spy_ma200 = {}
for i, d in enumerate(spy_dates):
    if i >= 199: spy_ma200[d] = sum(spy_prices[i-199:i+1])/200

def regime(date):
    sp = get_spy(date)
    if sp is None: return 'bear'
    ma = spy_ma200.get(date, sp)
    dev = (sp-ma)/ma*100
    if dev > 5: return 'bull'
    elif dev < -5: return 'bear'
    return 'sideways'

# Load stocks with metrics
print("Loading stocks...")
loaded = {}
for t in tickers:
    try:
        raw = json.load(open(f"{CACHE}/{t}.json"))['data']
        n = len(raw)
        c = [float(raw[i]['close']) for i in range(n)]
        has_hl = 'high' in raw[0]
        
        # ADX(14) if HL available
        adx_vals = [0.0]*n
        if has_hl:
            hh = [float(raw[i].get('high',c[i])) for i in range(n)]
            ll = [float(raw[i].get('low',c[i])) for i in range(n)]
            tr, pdm, mdm = [0.0]*n, [0.0]*n, [0.0]*n
            for i in range(1,n):
                tr[i]=max(hh[i]-ll[i], abs(hh[i]-c[i-1]), abs(ll[i]-c[i-1]))
                pdm[i]=max(hh[i]-hh[i-1],0) if hh[i]>hh[i-1] else 0
                mdm[i]=max(ll[i-1]-ll[i],0) if ll[i]<ll[i-1] else 0
            str_, spd, smd = [0.0]*n, [0.0]*n, [0.0]*n
            if n>=15:
                str_[14]=sum(tr[1:15])/14; spd[14]=sum(pdm[1:15])/14; smd[14]=sum(mdm[1:15])/14
                for i in range(15,n):
                    str_[i]=(str_[i-1]*13+tr[i])/14
                    spd[i]=(spd[i-1]*13+pdm[i])/14
                    smd[i]=(smd[i-1]*13+mdm[i])/14
            for i in range(14,n):
                if str_[i]>0:
                    dip=spd[i]/str_[i]*100; dim=smd[i]/str_[i]*100
                    s=dip+dim
                    dx=abs(dip-dim)/s*100 if s>0 else 0
                    if i==27: adx_vals[i]=sum(dx for dx in [0])+dx
            # Proper ADX
            dx_arr=[]
            for i in range(14,n):
                if str_[i]>0:
                    dip=spd[i]/str_[i]*100; dim=smd[i]/str_[i]*100
                    s=dip+dim
                    dx_arr.append(abs(dip-dim)/s*100 if s>0 else 0)
                else: dx_arr.append(0)
            for i in range(14,n):
                idx=i-14
                if idx>=13: adx_vals[i]=sum(dx_arr[idx-13:idx+1])/14
        # Not has_hl -> adx_vals stays 0
        
        # Build per-date metrics
        res = {}
        for i in range(60, n):
            d = raw[i]['date']; pr = c[i]
            hp52 = max(c[max(0,i-251):i+1]); p52 = pr/hp52*100 if hp52>0 else 100
            m = {}
            for p in [15,20,25,30]:
                if i>=p: m[p] = (pr/c[i-p]-1)*100
            adx = adx_vals[i] if i < len(adx_vals) else 0
            res[d] = {'p':pr, 'p52':p52, 'adx':adx, **m}
        loaded[t] = res
    except: pass
print("  Loaded: %d stocks" % len(loaded))

dates_list = sorted(set(d for t in loaded for d in loaded[t]))
years = sorted(set(int(d[:4]) for d in dates_list))
print("  Years: %s-%s (%d)" % (years[0], years[-1], len(years)))

# Parameter grid
PARAMS = {
    'ds': [40, 50, 60],       # deduct_start
    'dc': [0.3, 0.5],         # deduct_coeff
    'md': [25, 30],            # momentum_days
    'tn': [3, 5],              # top_n
    'hd': [15, 20],            # hold_days
    'ax': [0, 15, 20],         # adx_threshold (0=disabled)
    'bm': [0, 1],              # bear_mode: 0=reduce, 1=adx-only
}
total = 1
for v in PARAMS.values(): total *= len(v)
print("Grid: %d combinations" % total)

def run_one(params):
    ds, dc, md, tn, hd, ax, bm = params
    yearly_rets = []
    
    for y in years:
        sd = '%d-01-02' % y; ed = '%d-12-31' % y
        yr_dates = [d for d in dates_list if sd <= d <= ed]
        if len(yr_dates) < 60: continue
        rets = []
        
        for si in range(hd, len(yr_dates)-hd, hd):
            d_buy = yr_dates[si]; d_sell = yr_dates[min(si+hd, len(yr_dates)-1)]
            d_mom = yr_dates[max(0, si-md)]
            regime_type = regime(d_buy)
            cand = []
            
            for t, td in loaded.items():
                vb = td.get(d_buy); vp = td.get(d_mom)
                if not vb or not vb['p'] or not vp: continue
                mom = (vb['p']/vp['p']-1)*100 if md not in vb else vb[md]
                p52 = vb['p52']; adx_v = vb['adx']
                
                score = mom * (1 - min(max(0, (p52-ds)/(100-ds))*dc, 1))
                
                # Regime adjustments
                if regime_type == 'bear':
                    if bm == 0: score *= 0.5
                    elif ax > 0 and adx_v < ax: score = 0
                elif regime_type == 'sideways' and ax > 0 and adx_v < ax:
                    score *= 0.5
                
                if score > 0: cand.append((score, t, vb['p']))
            
            if len(cand) < tn: continue
            cand.sort(key=lambda x: -x[0])
            
            pr = []
            for _, t, bp in cand[:tn]:
                vs = loaded[t].get(d_sell)
                if vs and bp > 0: pr.append((vs['p']/bp-1)*100)
            if pr: rets.append(np.mean(pr))
        
        if rets: yearly_rets.append(sum(rets))
    
    cum = sum(yearly_rets)
    ny = len([r for r in yearly_rets if r != 0])
    if ny == 0: return {'cum':0,'ann':0,'sharpe':0,'mdd':0,'wr':0}
    
    ann = ((1+cum/100)**(1/ny)-1)*100 if cum>-100 else 0
    sharpe = np.mean(yearly_rets)/np.std(yearly_rets)*(12**0.5) if len(yearly_rets)>2 and np.std(yearly_rets)>0 else 0
    cv=100;pk=100;mdd=0
    for r in yearly_rets: cv*=1+r/100; pk=max(pk,cv); mdd=max(mdd,(pk-cv)/pk*100)
    wr = sum(1 for r in yearly_rets if r>0)/len(yearly_rets)*100 if yearly_rets else 0
    
    return {'cum':round(cum,1),'ann':round(ann,2),'sharpe':round(sharpe,2),'mdd':round(mdd,1),'wr':round(wr,1)}

# Run sweep
print("Running...")
t0 = time.time()
results = []
keys = list(PARAMS.keys())

for i, vals in enumerate(product(*(PARAMS[k] for k in keys))):
    r = run_one(list(vals))
    if r and r['cum'] != 0:
        results.append({'params':dict(zip(keys,vals)),**r})
    if (i+1)%200==0:
        el=time.time()-t0; best=max([r['ann'] for r in results]) if results else 0
        print('  %d/%d (%.0fs) best=%.2f%%'%(i+1,total,el,best),flush=True)

results.sort(key=lambda x:-x['ann'])
print('\n'+'='*80)
print('V4.1 Enhanced TOP 10')
print('='*80)
for i,r in enumerate(results[:10]):
    print('  #%d: ann=%.2f%% cum=%.1f%% sharpe=%.2f mdd=%.1f%% wr=%.0f%% | %s'%(
        i+1,r['ann'],r['cum'],r['sharpe'],r['mdd'],r['wr'],r['params']))
print('='*80)
json.dump(results[:50],open(OUTPUT,'w'),indent=2)
print('Saved to %s (%.0fs)'%(OUTPUT,time.time()-t0))
