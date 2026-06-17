import yfinance as yf, json, sys, math, warnings
warnings.filterwarnings('ignore')
UNIVERSE = ['NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','AVGO','AMD','INTC',
  'MU','QCOM','ARM','MRVL','SNPS','CDNS','ANET','CRWD','PANW','ZS',
  'TTWO','EA','PLTR','SOFI','COIN','MSTR','SMCI','WMT','LLY','UNH',
  'HD','JPM','V','MA','COST','NFLX','ADBE','CRM','ABNB',
  'KLAC','LRCX','AMAT','TXN','ASML','TSM','MS','GS','ABBV']
print("📥 加载数据...")
try:
    with open('/tmp/us_bt_data.json') as f:
        d=json.load(f)
    print("缓存加载", len(d['hist']), "只")
except:
    hist={}
    for sym in UNIVERSE:
        try:
            df=yf.download(sym,start='2014-06-01',end='2026-05-16',progress=False,auto_adjust=True)
            if df is not None and len(df)>500:
                df=df.dropna()
                hist[sym]={'dates':df.index.strftime('%Y-%m-%d').tolist(),
                           'close':[round(float(x),2) for x in df['Close'].values],
                           'high':[round(float(x),2) for x in df['High'].values],
                           'low':[round(float(x),2) for x in df['Low'].values]}
        except: pass
    adates=sorted(set(d for h in hist.values() for d in h['dates'] if d>='2015-01-01'))
    cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in hist}
    print(f"✅ {len(hist)}只 | 📅 {len(adates)}天")
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
        return {'c':c,'m20':m20,'m50':m50,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52}
    inds={}
    for code in hist:
        ind=ci(code)
        if ind:inds[code]=ind
    print(f"  ✅ {len(inds)}只")
    json.dump({'hist':{k:{'dates':v['dates'],'close':v['close'],'high':v['high'],'low':v['low']} for k,v in hist.items()},
               'adates':adates,'inds':{k:{kk:vv for kk,vv in v.items()} for k,v in inds.items()}},
              open('/tmp/us_bt_data.json','w'))
    
print("⚙️ 回测含回撤...")
# Rebuild from cache
import json as j
with open('/tmp/us_bt_data.json') as f: d=j.load(f)
hist=d['hist'];adates=d['adates'];inds=d['inds']
cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in hist}
def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1
def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score_stock(code,di,p):
    ind=inds.get(code)
    if not ind: return 0
    mh=saf(ind['mh'],di);mhp=saf(ind['mh'],di-1)
    pr=saf(ind['c'],di);m20=saf(ind['m20'],di);m50=saf(ind['m50'],di)
    av=saf(ind['adx'],di);rv=saf(ind['rsi'],di);p52v=saf(ind['p52'],di)
    ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=25
        elif mh>0 and mh>mhp: ms=15
        elif mh>0: ms=8
        else: ms=-3
    if p.get('macd_gate',False) and (mh is None or mh<=0): return 0
    ads=-5
    if av is not None:
        if av>=35: ads=22
        elif av>=25: ads=15
        elif av>=20: ads=8
        elif av>=15: ads=3
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
    ws=0
    if p52v is not None:
        if p52v<20: ws=15
        elif p52v<35: ws=12
        elif p52v<50: ws=8
        elif p52v<65: ws=5
        elif p52v<80: ws=2
    total=ms*(p['w_m']/25)+ads*(p['w_a']/22)+mas*(p['w_ma']/20)+rs*(p['w_r']/18)+ws*(p['w_w']/15)
    total=total/sum(p[k] for k in ['w_m','w_a','w_ma','w_r','w_w'])*100
    return min(total,95)

def bt(params,warmup,rebal=20):
    buy_t=params.get('buy_t',50)
    eq=[1000000.0]
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
        if len(scored)<3: eq.append(eq[-1]);continue
        scored.sort(key=lambda x:-x[1])
        tp=0;fp=0;cnt=0
        for code,sc,pr in scored[:5]:
            di_f=gi(code,fwd)
            if di_f<0: continue
            pr_f=saf(inds[code]['c'],di_f)
            if pr_f and pr_f>0: tp+=pr;fp+=pr_f;cnt+=1
        if cnt>=3: eq.append(eq[-1]*(fp/tp))
        else: eq.append(eq[-1])
    if len(eq)<5: return None
    peak=eq[0];mdd=0
    for v in eq:
        if v>peak: peak=v
        dd=(peak-v)/peak*100
        if dd>mdd: mdd=dd
    cum=(eq[-1]/eq[0]-1)*100
    yr=len(eq)*rebal/252
    ann=round(((eq[-1]/eq[0])**(1/yr)-1)*100,1) if yr>0.5 else 0
    return {'cum':round(cum,1),'ann':ann,'mdd':round(mdd,1),'n':len(eq)}

tests=[
    ('V1.6 逆向52W', {'macd_gate':True,'w_m':15,'w_a':20,'w_ma':15,'w_r':20,'w_w':30,'buy_t':50}),
    ('V1.8 均匀20%', {'macd_gate':True,'w_m':20,'w_a':20,'w_ma':20,'w_r':20,'w_w':20,'buy_t':50}),
    ('V1.0 原版',   {'macd_gate':False,'w_m':25,'w_a':22,'w_ma':20,'w_r':18,'w_w':15,'buy_t':50}),
]

print(f"\n📊 美股各版本回撤对比")
print("━"*65)
print(f"{'版本':<20s} {'周期':>6s} {'累计':>8s} {'年化':>7s} {'最大回撤':>9s} {'卡玛比':>8s}")
print("-"*65)
for name,p in tests:
    r=bt(p,450);r2=bt(p,580)
    if r:
        cal=r['ann']/r['mdd'] if r['mdd']>0 else 0
        print(f"{name:<20s} {'长':>4s} {r['cum']:>+7.1f}% {r['ann']:>+6.1f}% {r['mdd']:>7.1f}% {cal:>7.2f}")
    if r2:
        cal2=r2['ann']/r2['mdd'] if r2['mdd']>0 else 0
        print(f"{'':>20s} {'短':>4s} {r2['cum']:>+7.1f}% {r2['ann']:>+6.1f}% {r2['mdd']:>7.1f}% {cal2:>7.2f}")
    print()
