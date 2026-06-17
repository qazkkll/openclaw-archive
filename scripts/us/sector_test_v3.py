#!/usr/bin/env python3
"""板块限制测试 - 完整修复"""
import json, sys, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

with open('/home/admin/.openclaw/workspace/data/backtest_hist_v3_extended.json') as f: hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)
ETFS={'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}
codes=[c for c in hist if c not in ETFS and len(hist[c].get('close',[]))>500 and smap.get(c,'其他')!='其他']
adates=sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2020-01-01'<=d<='2026-05-14'))

# Fix date order
for code in codes:
    d=hist[code]
    if d['dates'][0] > d['dates'][min(30,len(d['dates'])-1)]:
        d['dates'].reverse()
        d['close'].reverse()
        d['high'].reverse()
        d['low'].reverse()

cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes}

def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None:
                return cdates[code][x]
    return -1

def ind(code):
    d=hist.get(code)
    if not d: return None
    c=d.get('close',[]);h=d.get('high',[]);l=d.get('low',[]);n=len(c)
    if n<60: return None
    def sma(a,p):return[None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def ema(a,p):k=2/(p+1);r=[a[0]];[r.append(v*k+r[-1]*(1-k)) for v in a[1:]];return r
    m5=sma(c,5);m20=sma(c,20);m60=sma(c,60)
    e12=ema(c,12);e26=ema(c,26);ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9);mh=[ml[i]-sg[i] for i in range(n)]
    gl,ll=[],[]
    for i in range(1,n):
        diff=c[i]-c[i-1];gl.append(max(diff,0));ll.append(max(-diff,0))
    rsi=[None]*14
    ag=sum(gl[:14])/14 if len(gl)>=14 else 0
    al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
    adx=[None]*27;tr_h,dp_h,dm_h=[],[],[]
    for i in range(1,n):
        tr=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        dp=max(0,h[i]-h[i-1]);dm=max(0,l[i-1]-l[i])
        tr_h.append(tr);dp_h.append(dp);dm_h.append(dm)
        if i<14:continue
        tr14=sum(tr_h[-14:]);dp14=sum(dp_h[-14:]);dm14=sum(dm_h[-14:]);atr=tr14/14
        if atr==0:adx.append(0);continue
        dip=dp14/14/atr*100;dim=dm14/14/atr*100
        if dip+dim==0:adx.append(0);continue
        dx=abs(dip-dim)/(dip+dim)*100
        if i<27:adx.append(dx);continue
        adx.append((sum(a for a in adx[-13:] if a is not None)+dx)/14)
    while len(adx)<n:adx.append(None)
    p52=[None]*251
    for i in range(251,n):
        lo=min(c[i-250:i+1]);hi=max(c[i-250:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52}

inds={}
for code in codes:
    i=ind(code)
    if i:inds[code]=i

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score_stock(code,di):
    ind_=inds.get(code)
    if not ind_: return 0
    mh=saf(ind_['mh'],di);mhp=saf(ind_['mh'],di-1)
    ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=20
        elif mh>0 and mh>mhp: ms=12
        elif mh>0: ms=6
    if ms<=0: return 0
    p52v=saf(ind_['p52'],di)
    ws=0
    if p52v is not None:
        if p52v<20: ws=20
        elif p52v<35: ws=15
        elif p52v<50: ws=10
        elif p52v<65: ws=6
        elif p52v<80: ws=3
    pr=saf(ind_['c'],di);m5=saf(ind_['m5'],di);m20=saf(ind_['m20'],di);m60=saf(ind_['m60'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=saf(ind_['adx'],di)
    ads=-5
    if av is not None:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
    rv=saf(ind_['rsi'],di)
    rs=0
    if rv is not None:
        if rv<25: rs=20
        elif rv<35: rs=14
        elif rv<50: rs=10
        elif rv<65: rs=6
        elif rv<75: rs=2
        elif rv>=75: rs=-5
    tr=av is not None and av>=22
    wl=[25,15,15,25,20] if tr else[10,30,15,10,35]
    ttl=ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(ttl/sum(wl)*100,100)

def sector_momentum(i,ss_d):
    mom={}
    for sec,cls in ss_d.items():
        rets=[]
        for c in cls[:15]:
            di1=gi(c,adates[i]);di2=gi(c,adates[max(0,i-20)])
            if di1<0 or di2<0 or c not in inds:continue
            p1=saf(inds[c]['c'],di1);p2=saf(inds[c]['c'],di2)
            if p1 and p2 and p2>0:rets.append((p1-p2)/p2*100)
        if len(rets)>=2:mom[sec]=sum(rets)/len(rets)
    return mom

def run(p,ss_d,excluded_s,excl_buy):
    cash=1000000.0;pos={};yy=[]
    for y in range(2021,2026):
        s=next((i for i,dt in enumerate(adates) if dt>=f'{y}-01-01'),None)
        e=next((i for i,dt in enumerate(adates) if dt>=f'{y+1}-01-01'),len(adates))
        if s and e and e-s>60:yy.append((y,s,e))
    
    for y,s,e in yy:
        for i in range(s,e):
            dt=adates[i]
            if (i-s)%p['rebal']==0:
                mom=sector_momentum(i,ss_d)
                if not mom:continue
                rk=sorted(mom.items(),key=lambda x:-x[1])
                ts=[r[0] for r in rk[:p['top']]]
                hs=[r[0] for r in rk[:p['hold']]]
                # Sell non-hold positions
                for c in list(pos.keys()):
                    if pos[c]['s'] not in hs:
                        di=gi(c,dt)
                        if di>=0:
                            pr=saf(inds[c]['c'],di)
                            if pr and pr>0:
                                cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                        del pos[c]
                # Buy candidates
                buy_list = []
                for sec in ts:
                    buy_t=excl_buy if sec in excluded_s else p['buy']
                    for c in ss_d.get(sec,[]):
                        if c in pos:continue
                        di=gi(c,dt)
                        if di<0 or c not in inds:continue
                        sc=score_stock(c,di)
                        if sc>=buy_t:
                            pr=saf(inds[c]['c'],di)
                            if pr and pr>0:
                                buy_list.append((c,sc,pr,sec))
                # Sort by score and buy best per sector
                from collections import defaultdict
                by_sec=defaultdict(list)
                for c,sc,pr,sec in buy_list:
                    by_sec[sec].append((c,sc,pr))
                for sec in ts:
                    if sec not in by_sec:continue
                    by_sec[sec].sort(key=lambda x:-x[1])
                    for c,sc,pr in by_sec[sec][:p['per_sec']]:
                        if len(pos)>=p['maxp']:break
                        inv=min(cash*p['pct'],cash*0.95)
                        if inv<20000:continue
                        pos[c]={'e':pr,'v':inv,'s':sec}
                        cash-=inv
            # Sell check (daily)
            for c in list(pos.keys()):
                di=gi(c,dt)
                if di<0 or c not in inds:continue
                sc=score_stock(c,di)
                if sc==0:continue
                pr=saf(inds[c]['c'],di)
                if not pr:continue
                m20=saf(inds[c]['m20'],di)
                mh=saf(inds[c]['mh'],di)
                if sc<p['sell'] or (pr and m20 and mh is not None and pr<m20 and mh<0):
                    cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                    del pos[c]
    
    final=cash
    for c,px in pos.items():
        di=gi(c,adates[yy[-1][2]-1])
        if di>=0 and c in inds:
            pr=saf(inds[c]['c'],di)
            if pr and pr>0:final+=px['v']*pr/px['e']
            else:final+=px['v']
        else:final+=px['v']
    return round((final/1000000-1)*100,2)

# Sector dicts
ss_all=defaultdict(list)
ss_ex1=defaultdict(list)
for c in codes:
    sec=smap.get(c,'其他')
    ss_all[sec].append(c)
    if sec!='地产基建':ss_ex1[sec].append(c)

BASE={'buy':62,'sell':48,'top':4,'hold':4,'rebal':7,'maxp':5,'per_sec':2,'pct':0.15}

tests=[('V2.5当前(排除地产)',ss_ex1,{'地产基建'},62),
       ('V2.5全行业',ss_all,set(),62),
       ('V2.5地产回(买65)',ss_all,{'地产基建'},65),
       ('V2.5地产回(买68)',ss_all,{'地产基建'},68)]

print(f"\n🏃 {len(tests)}个版本 (2021-2025)\n")
for name,ss_d,excluded_s,excl_buy in tests:
    sys.stdout.write(f"  {name:<30s} ")
    sys.stdout.flush()
    cum=run(BASE,ss_d,excluded_s,excl_buy)
    print(f"累计{cum:+.2f}%")
