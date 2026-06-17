#!/usr/bin/env python3
import json, os, warnings, numpy as np
warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
pool = json.load(open(UNIVERSE))
tickers = pool['tickers']

def ema(arr, p):
    if len(arr) < p: return []
    k=2/(p+1); r=[arr[0]]
    for v in arr[1:]: r.append(v*k+r[-1]*(1-k))
    return r

print("加载数据...")
loaded = {}
for t in tickers:
    try:
        raw = json.load(open(f"{CACHE}/{t}.json"))['data']
        n = len(raw)
        c = [float(raw[i]['close']) for i in range(n)]
        e12=ema(c,12); e26=ema(c,26)
        lc=min(len(e12),len(e26)); ml=[e12[j]-e26[j] for j in range(lc)]
        sg=ema(ml,9); l2=min(len(ml),len(sg)); mh=[ml[j]-sg[j] for j in range(l2)]
        res={}
        for i in range(125,n):
            d=raw[i]['date']; pr=c[i]
            hp52=max(c[max(0,i-251):i+1]); p52=pr/hp52*100 if hp52>0 else 100
            mom30=(pr/c[i-30]-1)*100
            ded=max(0,(p52-40)/60)*0.7; v42=mom30*(1-min(ded,1))
            hn=mh[i-1] if i-1<len(mh) else 0; hp=mh[i-2] if i-2<len(mh) else 0
            mg=hn>0
            ms=15 if mg and hn>0 and hp<=0 else (9 if mg and hn>hp else (5 if hn>0 else -3))
            rng=max(c[max(0,i-19):i+1])-min(c[max(0,i-19):i+1])
            ae=rng/pr*100 if pr>0 else 0; a=20 if ae>=0.08 else 15 if ae>=0.05 else 10 if ae>=0.03 else 5 if ae>=0.015 else -5
            ma20=sum(c[max(0,i-19):i+1])/min(20,i+1); ma50=sum(c[max(0,i-49):i+1])/min(50,i+1)
            ma=(5 if pr>ma20 else 0)+(5 if pr>ma50 else 0)+(5 if ma20>ma50 else 0)
            gn=sum(max(0,c[j]-c[j-1]) for j in range(i-13,i+1))
            ls=sum(max(0,c[j-1]-c[j]) for j in range(i-13,i+1))
            r=100 if ls==0 else (100-100/(1+(gn/14)/(ls/14)) if ls>0 else 50)
            rs=20 if r<25 else 15 if r<35 else 10 if r<50 else 6 if r<65 else 3 if r<75 else -5
            ps=30 if p52<20 else 24 if p52<35 else 18 if p52<50 else 10 if p52<65 else 5 if p52<80 else 0
            v2=0
            if mg:
                v2=max(0,ms+a+ma+rs+ps)
            res[d]=dict(p=pr, v42=v42, v2=v2)
        loaded[t]=res
    except Exception as e:
        pass

print(f"  {len(loaded)}只 / {len(tickers)}")
ad=sorted(set(d for td in loaded.values() for d in td.keys() if '2014-01-01'<=d<='2025-12-31'))
print(f"  交易日: {ad[0]}~{ad[-1]} ({len(ad)}天)")

# deep check - every stock, first 5 dates
all_ok = True
for t,td in loaded.items():
    for d in list(td.keys())[:5]:
        x = td[d]
        if not isinstance(x, dict) or 'v42' not in x:
            print(f"  BAD: {t}@{d} type={type(x)} keys={x.keys() if hasattr(x,'keys') else 'N/A'}")
            all_ok = False
if all_ok:
    print("  ✅ 深度检查通过")

import yfinance as yf, datetime as dtm
spy=yf.download('SPY',start="2013-06-01",end="2026-06-01",progress=False)
sm={d.strftime('%Y-%m-%d'):float(spy['Close'].squeeze().iloc[i]) for i,d in enumerate(spy.index)}
qqq=yf.download('QQQ',start="2013-06-01",end="2026-06-01",progress=False)
qm={d.strftime('%Y-%m-%d'):float(qqq['Close'].squeeze().iloc[i]) for i,d in enumerate(qqq.index)}

def br(m,y):
    def gb(s):
        dt=dtm.date(int(s[:4]),int(s[5:7]),int(s[8:10]))
        for o in range(-3,4):
            d=(dt+dtm.timedelta(days=o)).strftime('%Y-%m-%d')
            if d in m: return m[d]
        return None
    s=gb(f"{y}-01-02"); e=gb(f"{y}-12-31")
    return (e/s-1)*100 if s and e else 0

YEARS=list(range(2014,2026))

def run(y, hd=20):
    sd,ed=f"{y}-01-02",f"{y}-12-31"
    yd=[d for d in ad if sd<=d<=ed]
    if len(yd)<60: return 0.0, 0.0
    r42,r2=[],[]
    for si in range(hd, len(yd)-hd, hd):
        db=yd[si]
        ds=yd[min(si+hd, len(yd)-1)]
        items42=[]
        for t,td in loaded.items():
            if db in td:
                x=td[db]
                items42.append((x['v42'], t, x['p']))
        items42.sort(key=lambda x:-x[0])
        
        items2=[]
        for t,td in loaded.items():
            if db in td:
                x=td[db]
                if x['v2']>0:
                    items2.append((x['v2'], t, x['p']))
        items2.sort(key=lambda x:-x[0])
        
        if len(items42)>=5:
            rr=[]
            for _,t,bp in items42[:5]:
                vs=loaded[t].get(ds)
                if vs and vs['p']>0 and bp>0:
                    rr.append((vs['p']/bp - 1)*100)
            if rr:
                r42.append(np.mean(rr))
        if len(items2)>=5:
            rr=[]
            for _,t,bp in items2[:5]:
                vs=loaded[t].get(ds)
                if vs and vs['p']>0 and bp>0:
                    rr.append((vs['p']/bp - 1)*100)
            if rr:
                r2.append(np.mean(rr))
    return sum(r42) if r42 else 0.0, sum(r2) if r2 else 0.0

print(f"\n{'='*100}")
print(f"{'年份':>6s}  {'V4.2动量':>12s}  {'V2逆向':>12s}  {'70/30混合':>12s}  {'SPY':>8s}  {'QQQ':>8s}")
print(f"{'='*100}")

va,vt,vb,sa,qa=[],[],[],[],[]
for y in YEARS:
    a,b=run(y)
    m=a*0.7+b*0.3
    s=br(sm,y); q=br(qm,y)
    va.append(a); vt.append(b); vb.append(m); sa.append(s); qa.append(q)
    print(f"{y:>6d}  {a:>+12.1f}%  {b:>+12.1f}%  {m:>+12.1f}%  {s:>+8.1f}%  {q:>+8.1f}%")

def st(v):
    v=[x for x in v if x!=0]; n=len(v); t=sum(v)
    an=((1+t/100)**(1/n)-1)*100 if t>-100 else 0
    sp=np.mean(v)/np.std(v)*(12**0.5) if len(v)>2 and np.std(v)>0 else 0
    cv=100;pk=100;md=0
    for r in v:
        cv*=(1+r/100)
        if cv>pk:pk=cv
        d=(pk-cv)/pk*100
        if d>md:md=d
    wr=sum(1 for r in v if r>0)/n*100
    return an,sp,md,wr

sva=st(va); svt=st(vt); svb=st(vb)
print(f"{'='*100}")
print(f"{'累计':>6s}  {sum(va):>+12.1f}%  {sum(vt):>+12.1f}%  {sum(vb):>+12.1f}%  {sum(sa):>+8.1f}%  {sum(qa):>+8.1f}%")
print(f"\n{'指标':>20s}  {'V4.2':>10s}  {'V2逆向':>10s}  {'70/30混合':>10s}")
print("-"*55)
print(f"{'年化':>20s}  {sva[0]:>+9.1f}%  {svt[0]:>+9.1f}%  {svb[0]:>+9.1f}%")
print(f"{'夏普':>20s}  {sva[1]:>10.2f}  {svt[1]:>10.2f}  {svb[1]:>10.2f}")
print(f"{'最大回撤':>20s}  {sva[2]:>9.1f}%  {svt[2]:>9.1f}%  {svb[2]:>9.1f}%")
print(f"{'胜率(年)':>20s}  {sva[3]:>9.1f}%  {svt[3]:>9.1f}%  {svb[3]:>9.1f}%")

for n,v in [("V4.2",va),("V2逆向",vt),("70/30混合",vb)]:
    bs=sum(1 for i in range(12) if v[i]>sa[i])
    bq=sum(1 for i in range(12) if v[i]>qa[i])
    print(f"跑赢SPY {n:>10s}: {bs}/12  跑赢QQQ: {bq}/12")

print(f"\n✅ 完成")
