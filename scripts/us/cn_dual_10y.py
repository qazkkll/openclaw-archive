#!/usr/bin/env python3
"""A股双模式 2015-2025 十年对比"""
import json, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json') as f: hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)
ETFS={'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}
EX={'地产基建','农业','交通物流'}
codes=[c for c in hist if c not in ETFS and len(hist[c].get('close',[]))>500 and smap.get(c,'其他') not in EX]
adates=sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2015-01-01'<=d<='2026-05-14'))
print(f"📊 {len(codes)}只 | 📅 {len(adates)}天 ({adates[0]}~{adates[-1]})")

cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes}
def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

def ci(code):
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
    for i in range(1,n):diff=c[i]-c[i-1];gl.append(max(diff,0));ll.append(max(-diff,0))
    rsi=[None]*14;ag=sum(gl[:14])/14 if len(gl)>=14 else 0;al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):rsi.append(100-100/(1+ag/al) if al>0 else 100)
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
    for i in range(251,n):lo=min(c[i-250:i+1]);hi=max(c[i-250:i+1]);p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52}

print("⚙️ 指标...")
inds={}
for code in codes:
    i=ci(code)
    if i:inds[code]=i
print(f"  ✅ {len(inds)}只")

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score(code,di):
    ind=inds.get(code)
    if not ind:return 0
    mh=saf(ind['mh'],di);mhp=saf(ind['mh'],di-1);ms=0
    if mh and mhp:
        if mh>0 and mhp<=0:ms=20
        elif mh>0 and mh>mhp:ms=12
        elif mh>0:ms=6
    if ms<=0:return 0
    pv=saf(ind['p52'],di);ws=0
    if pv:
        if pv<20: ws=20
        elif pv<35: ws=15
        elif pv<50: ws=10
        elif pv<65: ws=6
        elif pv<80: ws=3
    pr=saf(ind['c'],di);m5=saf(ind['m5'],di);m20=saf(ind['m20'],di);m60=saf(ind['m60'],di)
    mas=0
    if pr and m20 and pr>m20:mas+=7
    if m5 and m20 and m5>m20:mas+=7
    if m20 and m60 and m20>m60:mas+=6
    av=saf(ind['adx'],di);ads=-5
    if av:
        if av>=35:ads=20
        elif av>=28:ads=15
        elif av>=22:ads=10
        elif av>=18:ads=5
    rv=saf(ind['rsi'],di);rs=0
    if rv:
        if rv<25:rs=20
        elif rv<35:rs=14
        elif rv<50:rs=10
        elif rv<65:rs=6
        elif rv<75:rs=2
        elif rv>=75:rs=-5
    tr=av and av>=22
    wl=[25,15,15,25,20] if tr else[10,30,15,10,35]
    ttl=ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(ttl/sum(wl)*100,100)

def strength(dt):
    a=0;t=0
    for code in codes:
        d=hist[code]
        try:di=d['dates'].index(dt)
        except:continue
        if di<20:continue
        ma=sum(d['close'][di-19:di+1])/20
        t+=1
        if d['close'][di]>ma:a+=1
    return round(a/t*100,1) if t>0 else 50

ss=defaultdict(list)
for c in codes:
    sec=smap.get(c,'其他')
    if sec not in EX:ss[sec].append(c)

def sm(i):
    mom={}
    for sec,cls in ss.items():
        rets=[]
        for c in cls[:20]:
            d1=gi(c,adates[i]);d2=gi(c,adates[max(0,i-20)])
            if d1<0 or d2<0 or c not in inds:continue
            p1=saf(inds[c]['c'],d1);p2=saf(inds[c]['c'],d2)
            if p1 and p2 and p2>0:rets.append((p1-p2)/p2*100)
        if len(rets)>=2:mom[sec]=sum(rets)/len(rets)
    return mom

P={'buy':62,'sell':48,'top':4,'hold':4,'rebal':7,'maxp':5,'per_sec':2,'pct':0.15}

def run(dual):
    cash=1000000.0;pos={}
    yy=[]
    for y in range(2016,2026):
        s=next((i for i,dt in enumerate(adates) if dt>=f'{y}-01-01'),None)
        e=next((i for i,dt in enumerate(adates) if dt>=f'{y+1}-01-01'),len(adates))
        if s and e and e-s>60:yy.append((y,s,e))
    
    for y,s,e in yy:
        for i in range(s,e):
            dt=adates[i]
            if (i-s)%P['rebal']==0:
                if dual:
                    st=strength(dt)
                    if st<20:
                        for c in list(pos.keys()):
                            di=gi(c,dt)
                            if di>=0:
                                pr=saf(inds[c]['c'],di)
                                if pr and pr>0:cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                            del pos[c]
                        continue
                mom=sm(i)
                if not mom:continue
                rk=sorted(mom.items(),key=lambda x:-x[1])
                ts=[r[0] for r in rk[:P['top']]];hs=[r[0] for r in rk[:P['hold']]]
                for c in list(pos.keys()):
                    if pos[c]['s'] not in hs:
                        di=gi(c,dt)
                        if di>=0:
                            pr=saf(inds[c]['c'],di)
                            if pr and pr>0:cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                        del pos[c]
                bcs=defaultdict(list)
                for sec in ts:
                    for c in ss.get(sec,[]):
                        if c in pos: continue
                        di=gi(c,dt)
                        if di<0 or c not in inds: continue
                        sco=score(c,di)
                        if sco>=P['buy']:
                            pr=saf(inds[c]['c'],di)
                            if pr and pr>0:bcs[sec].append((c,sco,pr))
                for sec in ts:
                    bcs[sec].sort(key=lambda x:-x[1])
                    for c,sco,pr in bcs[sec][:P['per_sec']]:
                        if len(pos)>=P['maxp']:break
                        inv=min(cash*P['pct'],cash*0.95)
                        if inv<20000:continue;pos[c]={'e':pr,'v':inv,'s':sec};cash-=inv
            for c in list(pos.keys()):
                di=gi(c,dt)
                if di<0 or c not in inds: continue
                sco=score(c,di)
                pr=saf(inds[c]['c'],di)
                m20=saf(inds[c]['m20'],di);mh=saf(inds[c]['mh'],di)
                if sco<P['sell'] or(pr and m20 and mh is not None and pr<m20 and mh<0):
                    if pr and pr>0:cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e']);del pos[c]
    for c,px in pos.items():
        di=gi(c,adates[yy[-1][2]-1])
        if di>=0 and c in inds:
            pr=saf(inds[c]['c'],di)
            if pr and pr>0:cash+=px['v']*(1+(pr-px['e'])/px['e'])
            else:cash+=px['v']
        else:cash+=px['v']
    return round((cash/1000000-1)*100,2),round(cash)

print("\n🏃 十年回测...\n")
for name,dual in [('V2.5原版',False),('V2.5双模式',True)]:
    cum,final=run(dual)
    nac=10
    ann=round(((1+cum/100)**(1/nac)-1)*100,2) if cum>0 else 0
    print(f"{name:<20s} 累计{cum:+.2f}%  年化{ann:.1f}%  100万->¥{final:,}")

print(f"\n沪深300同期(2016-2025): +1.60%")
