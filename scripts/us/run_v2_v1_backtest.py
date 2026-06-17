#!/usr/bin/env python3
"""小钳轮动V2 vs V1 回测 2020-2025 - 快速版"""
import json, math, sys, time
from collections import defaultdict
from datetime import datetime

V2P = {'buy':62,'sell':40,'top':3,'hold':4,'rebal':10,'maxp':12,'pers':3}
V1P = {'buy':62,'sell':48,'top':4,'hold':4,'rebal':5,'maxp':8,'pers':2}
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}

print("Loading data...")
with open('/home/admin/.openclaw/workspace/data/backtest_hist_v3_extended.json') as f:
    hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f:
    smap = json.load(f)

dates = sorted(set(d for c in hist for d in hist[c].get('dates',[]) if '2020-01-01' <= d <= '2026-05-14'))
print(f"Dates: {len(dates)} ({dates[0]} ~ {dates[-1]})")

ss = defaultdict(list)
for c in hist: ss[smap.get(c,'其他')].append(c)
ssec = dict(ss)

def get_idx(code, dt):
    d = hist.get(code)
    if not d or not d.get('dates'): return -1
    try: return d['dates'].index(dt)
    except: pass
    for x in reversed(d['dates']):
        if x <= dt: return d['dates'].index(x)
    return -1

def calc_ind(code):
    d = hist.get(code)
    if not d: return None
    c = d.get('close',[]); h = d.get('high',[]); l = d.get('low',[])
    if len(c) < 60: return None
    n = len(c)
    def sma(a,p): return [None]*(p-1) + [sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def ema(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    m5=sma(c,5); m20=sma(c,20); m60=sma(c,60)
    e12=ema(c,12); e26=ema(c,26); ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9); mh=[ml[i]-sg[i] for i in range(n)]
    gl,ll=[],[]
    for i in range(1,n): diff=c[i]-c[i-1]; gl.append(max(diff,0)); ll.append(max(-diff,0))
    rsi=[None]*14
    ag=sum(gl[:14])/14 if len(gl)>=14 else 0; al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i<len(gl): ag=(ag*13+gl[i])/14; al=(al*13+ll[i])/14
    adx=[None]*30
    for i in range(30,n):
        tr=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        dm_plus=max(0,h[i]-h[i-1]); dm_minus=max(0,l[i-1]-l[i])
        atr=sum(max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(i-13,i+1))/14
        if atr>0 and (dm_plus+dm_minus)>0: dx=abs(dm_plus-dm_minus)/(dm_plus+dm_minus)*100
        else: dx=0
        adx.append(dx)
    p52=[None]*251
    for i in range(251,n):
        lo=min(c[i-250:i+1]); hi=max(c[i-250:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52}

print("Calc indicators...")
inds = {}
for code in hist:
    ind = calc_ind(code)
    if ind: inds[code] = ind

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score(code, di):
    ind = inds.get(code)
    if not ind: return 0
    mh = saf(ind['mh'], di)
    if mh is None or mh <= 0: return 0
    mhp = saf(ind['mh'], di-1)
    ms = 20 if (mh>0 and mhp and mh>mhp) else 12 if mh>0 else 0
    p = saf(ind['p52'], di)
    ws = 0
    if p is not None:
        if p<20: ws=20
        elif p<35: ws=15
        elif p<50: ws=10
        elif p<65: ws=6
        elif p<80: ws=3
    pr=saf(ind['c'],di); m5=saf(ind['m5'],di); m20=saf(ind['m20'],di); m60=saf(ind['m60'],di)
    mas = (7 if pr and m20 and pr>m20 else 0)+(7 if m5 and m20 and m5>m20 else 0)+(6 if m20 and m60 and m20>m60 else 0)
    av=saf(ind['adx'],di)
    ads = -5
    if av is not None:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
    rv=saf(ind['rsi'],di)
    rs = 0
    if rv is not None:
        if rv<25: rs=20
        elif rv<35: rs=14
        elif rv<50: rs=10
        elif rv<65: rs=6
        elif rv<75: rs=2
        elif rv>=75: rs=-5
    tr = av is not None and av >= 22
    wl = [25,15,15,25,20] if tr else [10,30,15,10,35]
    total = ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(total/sum(wl)*100, 100)

def sec_mom(i):
    mom = {}
    for sec, codes in ssec.items():
        rets = []
        for c in codes[:20]:
            di = get_idx(c, dates[i])
            if di < 0: continue
            c0 = saf(inds[c]['c'], di) if c in inds else None
            di20 = get_idx(c, dates[max(0,i-20)])
            c20 = saf(inds[c]['c'], di20) if di20>=0 and c in inds else None
            if c0 and c20 and c20>0:
                rets.append((c0-c20)/c20*100)
        if len(rets) >= 2:
            mom[sec] = sum(rets)/len(rets)
    return mom

def simulate(s, e, p, name):
    cash=1000000.0; pos={}; daily=[]
    for i in range(s, e):
        dt = dates[i]
        if (i-s) % p['rebal'] == 0:
            mom = sec_mom(i)
            if not mom: continue
            rk = sorted(mom.items(), key=lambda x:-x[1])
            ts = [r[0] for r in rk[:p['top']]]
            hs = [r[0] for r in rk[:p['hold']]]
            for c in list(pos.keys()):
                if pos[c]['s'] not in hs:
                    di = get_idx(c, dt)
                    if di>=0 and c in inds:
                        pr = saf(inds[c]['c'], di)
                        if pr and pr>0:
                            pnl = (pr-pos[c]['e'])/pos[c]['e']*100
                            cash += pos[c]['v']*(1+pnl/100)
                    del pos[c]
            for sec in ts:
                if len(pos)>=p['maxp']: break
                cand=[]
                for c in ssec.get(sec, []):
                    if c in pos or c in ETFS: continue
                    di = get_idx(c, dt)
                    if di<0 or c not in inds: continue
                    sc = score(c, di)
                    if sc>=p['buy']:
                        pr = saf(inds[c]['c'], di)
                        if pr and pr>0:
                            cand.append((c,sc,pr))
                cand.sort(key=lambda x:-x[1])
                for c,sc,pr in cand[:p['pers']]:
                    if c in pos or len(pos)>=p['maxp']: break
                    if name=='V2':
                        pct = min(0.10+(sc-62)*0.01, 0.25)
                        inv = min(cash*pct, cash*0.95)
                    else:
                        inv = min(cash*0.15, cash*0.95)
                    if inv<20000: continue
                    pos[c]={'e':pr,'v':inv,'s':sec}; cash-=inv
        for c in list(pos.keys()):
            di = get_idx(c, dt)
            if di<0 or c not in inds: continue
            sc = score(c, di)
            pr = saf(inds[c]['c'], di)
            m20 = saf(inds[c]['m20'], di)
            mh = saf(inds[c]['mh'], di)
            if sc<p['sell'] or (pr and m20 and mh is not None and pr<m20 and mh<0):
                if pr and pr>0:
                    pnl = (pr-pos[c]['e'])/pos[c]['e']*100
                    cash += pos[c]['v']*(1+pnl/100)
                del pos[c]
        tv = cash
        for c,ph in pos.items():
            di = get_idx(c, dt)
            if di>=0 and c in inds:
                pr = saf(inds[c]['c'], di)
                if pr and pr>0:
                    tv += ph['v']*(pr/ph['e'])
                else:
                    tv += ph['v']
            else:
                tv += ph['v']
        daily.append(tv)
    for c,ph in list(pos.items()):
        di = get_idx(c, dates[e-1])
        if di>=0 and c in inds:
            pr = saf(inds[c]['c'], di)
            if pr and pr>0:
                pnl = (pr-ph['e'])/ph['e']*100
                cash += ph['v']*(1+pnl/100)
            else:
                cash += ph['v']
        else:
            cash += ph['v']
    ret = (cash-1000000)/1000000*100
    peak = max(daily) if daily else 1000000
    mdd = max(((peak-v)/peak*100) for v in daily) if daily else 0
    dr = [(daily[j]-daily[j-1])/daily[j-1]*100 for j in range(1,len(daily)) if daily[j-1]>0]
    sr = 0
    if len(dr)>5:
        avg = sum(dr)/len(dr); var = sum((r-avg)**2 for r in dr)/len(dr); std = max(var**0.5,0.001)
        sr = round(avg/std*15.8,2)
    return {'ret':round(ret,2),'mdd':round(mdd,2),'sr':sr,'final':round(cash)}

# Year ranges
years = []
for y in [2020,2021,2022,2023,2024,2025]:
    s = next((i for i,dt in enumerate(dates) if dt>=f'{y}-01-01'), None)
    e = next((i for i,dt in enumerate(dates) if dt>=f'{y+1}-01-01'), len(dates))
    if s and e and e-s>30:
        years.append((y,s,e))

idx_ret = {2020:27.21,2021:-5.20,2022:-21.63,2023:-11.38,2024:14.68,2025:-3.67}

# Run V2
print("\nV2 Strategy")
v2r = {}
for y,s,e in years:
    print(f"DEBUG simulate {y} {s}->{e} keys={list(V2P.keys())}")
    r = simulate(s,e,V2P,'V2')
    v2r[y] = r
    rr=r['ret']; md=r['mdd']; sr=r['sr']
    print(f"  {y}: {rr:+6.2f}%  drawdown {md:.2f}%  sharpe {sr}")

# Run V1
print("\nV1 Strategy")
v1r = {}
for y,s,e in years:
    print(f"DEBUG simulate {y} {s}->{e} keys={list(V2P.keys())}")
    r = simulate(s,e,V1P,'V1')
    v1r[y] = r
    rr=r['ret']; md=r['mdd']; sr=r['sr']
    print(f"  {y}: {rr:+6.2f}%  drawdown {md:.2f}%  sharpe {sr}")

# Output table
print("\n" + "="*65)
print("V2 vs V1 vs CSI300 (2020-2025)")
print("="*65)
h = "Year    V2%        V1%        CSI300%    V2-vs-V1   V2-vs-Idx"
print(h)
print("-"*len(h))

tv2=1000000; tv1=1000000; tidx=1000000
for y,s,e in years:
    print(f"DEBUG simulate {y} {s}->{e} keys={list(V2P.keys())}")
    v2=v2r[y]['ret']; v1=v1r[y]['ret']; idx=idx_ret.get(y,0)
    tv2*=1+v2/100; tv1*=1+v1/100; tidx*=1+idx/100
    print(f"{y:4d}  {v2:+7.2f}%  {v1:+7.2f}%  {idx:+7.2f}%  {v2-v1:+7.2f}%  {v2-idx:+7.2f}%")

cv2=(tv2/1000000-1)*100; cv1=(tv1/1000000-1)*100; cidx=(tidx/1000000-1)*100
print("-"*len(h))
print(f"Tot  {cv2:+7.2f}%  {cv1:+7.2f}%  {cidx:+7.2f}%  {cv2-cv1:+7.2f}%  {cv2-cidx:+7.2f}%")

print(f"\nFinal value (1M start):")
print(f"  V2:     {tv2:>12,.0f} ({cv2:+.2f}%)")
print(f"  V1:     {tv1:>12,.0f} ({cv1:+.2f}%)")
print(f"  CSI300: {tidx:>12,.0f} ({cidx:+.2f}%)")

# Save
out = {
    'date': datetime.now().isoformat(),
    'range': [dates[years[0][1]], dates[years[-1][2]-1]],
    'v2': {str(y):v2r[y] for y,s,e in years},
    'v1': {str(y):v1r[y] for y,s,e in years},
    'index_returns': idx_ret,
    'summary': {
        'v2_cum': round(cv2,2), 'v1_cum': round(cv1,2), 'idx_cum': round(cidx,2),
        'v2_out_v1': round(cv2-cv1,2), 'v2_out_idx': round(cv2-cidx,2),
    }
}
with open('/home/admin/.openclaw/workspace/models/v2_vs_v1_2020_2025.json','w') as f:
    json.dump(out,f,indent=2)
print("\nDone! Saved to models/v2_vs_v1_2020_2025.json")
