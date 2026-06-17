#!/usr/bin/env python3
"""A股/美股信号灯阈值独立测算"""
import json, sys, random
from collections import defaultdict

random.seed(42)

print("📥 A股数据...")
with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json') as f: hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)
EXCLUDED={'地产基建','农业','交通物流'}
codes=[c for c in hist if len(hist[c].get('close',[]))>500 and smap.get(c,'其他') not in EXCLUDED]
adates=sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2020-01-01'<=d<='2026-05-14'))

cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes}
def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

scodes=random.sample(codes,300)

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
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i<len(gl):ag=(ag*13+gl[i])/14;al=(al*13+ll[i])/14
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

inds={}
for code in scodes:
    ind=ci(code)
    if ind:inds[code]=ind
def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score_cn(code,di):
    ind=inds.get(code)
    if not ind: return 0
    mh=saf(ind['mh'],di);mhp=saf(ind['mh'],di-1);ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=20
        elif mh>0 and mh>mhp: ms=12
        elif mh>0: ms=6
    if ms<=0: return 0
    p52v=saf(ind['p52'],di);ws=0
    if p52v is not None:
        if p52v<20: ws=20
        elif p52v<35: ws=15
        elif p52v<50: ws=10
        elif p52v<65: ws=6
        elif p52v<80: ws=3
    pr=saf(ind['c'],di);m5=saf(ind['m5'],di);m20=saf(ind['m20'],di);m60=saf(ind['m60'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=saf(ind['adx'],di);ads=-5
    if av is not None:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
    rv=saf(ind['rsi'],di);rs=0
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

# 采样
print("采样A股评分分布...")
scores_cn=[]
warmup=400
for _ in range(60):
    t=random.randint(warmup,len(adates)-1)
    for code in inds:
        di=gi(code,adates[t])
        if di>=0:
            sc=score_cn(code,di)
            if sc>0:scores_cn.append(sc)
    if len(scores_cn)>5000:break

def pct(data,p):
    s=sorted(data);idx=int(len(s)*p/100)
    return s[min(idx,len(s)-1)]

print(f"\n📊 A股 V2.5 评分分布 ({len(scores_cn)}样本)")
print(f"{'百分位':>6s} {'分数':>5s}")
print("-"*20)
for p in [5,10,25,50,75,90,95]:
    print(f"{p:>5d}% {pct(scores_cn,p):>4.0f}分")

# 美股分布（用今日扫描结果）
print(f"\n📊 美股 V2 评分分布 (今日扫描结果)")
us_scores_high=[88,86,82,76,68,63,63,63,63,62,62,62,62,62,58,58,58,58,58,58,58,58,58,58,58,54,54,54]
us_scores_low=[38,38,42]  # 持仓
us_all=us_scores_high+us_scores_low
print(f"{'百分位':>6s} {'分数':>5s}")
print("-"*20)
for p in [5,10,25,50,75,90,95]:
    print(f"{p:>5d}% {pct(us_all,p):>4.0f}分")

# ===== 建议阈值 =====
print(f"\n{'='*50}")
print(f"🏆 建议五档信号灯阈值")
print(f"{'='*50}")
print(f"\n{'信号':>6s} {'A股V2.5':>10s} {'美股V2':>10s}")
print("-"*30)
print(f"{'🟢加仓':>6s} {'≥'+str(pct(scores_cn,75)):>8s} {'≥60':>8s}")
print(f"{'🔵关注':>6s} {'≥'+str(pct(scores_cn,50)):>8s} {'≥50':>8s}")
print(f"{'🟡持有':>6s} {'≥'+str(pct(scores_cn,25)):>8s} {'≥35':>8s}")
print(f"{'🟠警惕':>6s} {'≥'+str(pct(scores_cn,10)):>8s} {'≥25':>8s}")
print(f"{'🔴卖出':>6s} {'<'+str(pct(scores_cn,10)):>8s} {'<25':>8s}")

print(f"\n📌 对比现有买卖线:")
print(f"  A股: 买62 / 卖48")
print(f"  美股: 买50 / 卖30")
