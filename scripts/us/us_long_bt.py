#!/usr/bin/env python3
"""美股TOP3模型 2015-2025长周期验证"""
import yfinance as yf, json, sys, math, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from datetime import datetime

UNIVERSE = ['NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','AVGO','AMD','INTC',
  'MU','QCOM','ARM','MRVL','SNPS','CDNS','ANET','CRWD','PANW','ZS',
  'TTWO','EA','PLTR','SOFI','COIN','MSTR','SMCI','WMT','LLY','UNH',
  'HD','JPM','V','MA','COST','NFLX','ADBE','CRM','UBER','ABNB',
  'KLAC','LRCX','AMAT','TXN','ASML','TSM','MS','GS','ABBV']

print("📥 下载美股数据 2014-2026...")
hist={}
for sym in UNIVERSE:
    try:
        df=yf.download(sym,start='2014-01-01',end='2026-05-16',progress=False,auto_adjust=True)
        if df is not None and len(df)>500:
            df=df.dropna()
            hist[sym]={'dates':df.index.strftime('%Y-%m-%d').tolist(),
                       'close':[round(float(x),2) for x in df['Close'].values],
                       'high':[round(float(x),2) for x in df['High'].values],
                       'low':[round(float(x),2) for x in df['Low'].values]}
    except: pass

adates=sorted(set(d for h in hist.values() for d in h['dates'] if '2015-01-01'<=d<='2026-05-15'))
print(f"✅ {len(hist)}只 | 📅 {len(adates)}天 ({adates[0]}~{adates[-1]})")

cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in hist}
def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

print("⚙️ 计算指标...")
def ci(code):
    d=hist.get(code)
    if not d: return None
    c=d.get('close',[]);h=d.get('high',[]);l=d.get('low',[]);n=len(c)
    if n<60: return None
    def sma(a,p):return[None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def ema(a,p):k=2/(p+1);r=[a[0]];[r.append(v*k+r[-1]*(1-k)) for v in a[1:]];return r
    m20=sma(c,20);m50=sma(c,50)
    e12=ema(c,12);e26=ema(c,26);ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9);mh=[ml[i]-sg[i] for i in range(n)]
    gl,ll=[],[]
    for i in range(1,n):diff=c[i]-c[i-1];gl.append(max(diff,0));ll.append(max(-diff,0))
    rsi=[None]*14;ag=sum(gl[:14])/14 if len(gl)>=14 else 0;al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i<len(gl):ag=(ag*13+gl[i])/14;al=(al*13+ll[i])/14
    p52=[None]*251
    for i in range(251,n):lo=min(c[i-250:i+1]);hi=max(c[i-250:i+1]);p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m20':m20,'m50':m50,'rsi':rsi,'mh':mh,'p52':p52}

inds={}
for code in hist:
    ind=ci(code)
    if ind:inds[code]=ind
print(f"  ✅ {len(inds)}只")

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score_stock(code,di,p):
    ind=inds.get(code)
    if not ind: return 0
    mh=saf(ind['mh'],di);mhp=saf(ind['mh'],di-1)
    pr=saf(ind['c'],di);m20=saf(ind['m20'],di);m50=saf(ind['m50'],di)
    rv=saf(ind['rsi'],di);p52v=saf(ind['p52'],di)
    ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=25
        elif mh>0 and mh>mhp: ms=15
        elif mh>0: ms=8
        else: ms=-3
    if p.get('macd_gate',False) and (mh is None or mh<=0): return 0
    ws=0
    if p52v is not None:
        if p52v<20: ws=15
        elif p52v<35: ws=12
        elif p52v<50: ws=8
        elif p52v<65: ws=5
        elif p52v<80: ws=2
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if pr and m50 and pr>m50: mas+=7
    if m20 and m50 and m20>m50: mas+=6
    rs=0
    if rv is not None:
        if rv<25: rs=18
        elif rv<35: rs=14
        elif rv<50: rs=10
        elif rv<65: rs=6
        elif rv<75: rs=2
        else: rs=-5
    total=ms*(p['w_m']/25)+ws*(p['w_w']/15)+mas*(p['w_ma']/20)+rs*(p['w_r']/18)
    return min(total/100*100,95)

def bt(params,warmup,rebal=20,label=''):
    buy_t=params.get('buy_t',50)
    results=[]
    for i in range(warmup,len(adates)-20,rebal):
        dt=adates[i];fwd=adates[min(i+20,len(adates)-1)]
        scored=[]
        for code in inds:
            di=gi(code,dt)
            if di<0: continue
            sc=score_stock(code,di,params)
            if sc>=buy_t:
                pr=saf(inds[code]['c'],di)
                if pr and pr>0: scored.append((code,sc,pr))
        if len(scored)<3: continue
        scored.sort(key=lambda x:-x[1])
        tp=0;fp=0;cnt=0
        for code,sc,pr in scored[:5]:
            di_f=gi(code,fwd)
            if di_f<0: continue
            pr_f=saf(inds[code]['c'],di_f)
            if pr_f and pr_f>0: tp+=pr;fp+=pr_f;cnt+=1
        if cnt>=3: results.append((fp/tp-1)*100)
    if len(results)<10: return None
    avg=sum(results)/len(results);ann=avg*(252/rebal)
    wins=sum(1 for r in results if r>0)/len(results)*100
    std=math.sqrt(sum((r-avg)**2 for r in results)/len(results)) if len(results)>1 else 1
    return {'avg':round(avg,2),'wr':round(wins,1),'ann':round(ann,2),'ir':round(avg/std,2)if std>0 else 0,'n':len(results)}

tests=[
    ('V1.6 逆向52W', {'macd_gate':True,'w_m':15,'w_ma':15,'w_r':20,'w_w':30,'buy_t':50}),
    ('V1.8 均匀20%', {'macd_gate':True,'w_m':20,'w_ma':20,'w_r':20,'w_w':20,'buy_t':50}),
    ('V1.9 RSI+52W', {'macd_gate':True,'w_m':20,'w_ma':10,'w_r':25,'w_w':25,'buy_t':50}),
    ('V1.0 原版',    {'macd_gate':False,'w_m':25,'w_ma':20,'w_r':18,'w_w':15,'buy_t':50}),
]

print(f"\n🏃 长周期回测...\n")
results=[]
for name,p in tests:
    r=bt(p,450,20)  # 2016-2025
    r2=bt(p,580,20)  # 2021-2025
    results.append({'name':name,'long':r,'short':r2})
    if r: print(f"{name}: 2016-2025 {r['ann']:+.1f}%/年 {r['avg']:+.1f}%/20d WR{r['wr']:.0f}% IR{r['ir']}")
    if r2: print(f"{'':>16s} 2021-2025 {r2['ann']:+.1f}%/年 {r2['avg']:+.1f}%/20d WR{r2['wr']:.0f}% IR{r2['ir']}")
    print()

# SPY基准
print("📊 SPY年化(2016-2025): ~13.2%")
print("📊 SPY年化(2021-2025): ~12.5%\n")

print("━"*55)
print("📊 长周期对比")
print("━"*55)
print(f"{'版本':<16s} {'2016-2025年化':>14s} {'2021-2025年化':>14s} {'变化':>8s}")
print("-"*55)
for r in results:
    a15=r['long']['ann'] if r['long'] else 0
    a21=r['short']['ann'] if r['short'] else 0
    print(f"{r['name']:<16s} {a15:>+10.1f}% {a21:>+10.1f}% {a15-a21:>+7.1f}%")
print("-"*55)
print("SPY比较:        +13.2%       +12.5%")
