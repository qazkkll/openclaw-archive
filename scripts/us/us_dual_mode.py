#!/usr/bin/env python3
"""美股双模式：动量追涨 vs 逆向防守"""
import yfinance as yf, json, warnings
warnings.filterwarnings('ignore')

try:
    with open('/tmp/us_bt_data.json') as f: d=json.load(f)
except:
    import sys; print("无缓存"); sys.exit(1)

hist=d['hist'];adates=d['adates'];inds=d['inds']
cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in hist}

def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d0=hist.get(code)
    if d0 and d0.get('dates'):
        for x in reversed(d0['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score_v2(code,di):
    ind=inds.get(code)
    if not ind: return 0
    mh=saf(ind['mh'],di);mhp=saf(ind['mh'],di-1)
    ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=25
        elif mh>0 and mh>mhp: ms=15
        elif mh>0: ms=8
        else: ms=-3
    if mh is None or mh<=0: return 0
    pr=saf(ind['c'],di);m20=saf(ind['m20'],di);m50=saf(ind['m50'],di)
    av=saf(ind['adx'],di);rv=saf(ind['rsi'],di);p52v=saf(ind['p52'],di)
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
    total=(ms*15/25+ads*20/22+mas*15/20+rs*20/18+ws*30/15)/100*100
    return min(total,95)

print("📥 SPY...")
spy=yf.download('SPY',start='2014-06-01',end='2026-05-16',progress=False,auto_adjust=True)
spy_c=spy['Close'].values;spy_dates=spy.index.strftime('%Y-%m-%d').tolist()

def spy_regime(dt):
    for i in range(len(spy_dates)-1,-1,-1):
        if spy_dates[i]<=dt:
            c=spy_c[i];m200=sum(spy_c[max(0,i-199):i+1])/min(200,i+1)
            if c>m200*1.02:return 'bull'
            if c<m200*0.98:return 'bear'
            return 'choppy'
    return 'choppy'

print("🏃 双模式...\n")

for mode,label in [(score_v2,'V2逆向'),(None,'动量追涨')]:
    eq=[1000000.0];rc={'bull':0,'bear':0,'choppy':0}
    for i in range(580,len(adates)-20,20):
        dt=adates[i];fwd=adates[min(i+20,len(adates)-1)]
        rg=spy_regime(dt);rc[rg]+=1
        
        if mode is not None:
            # V2评分
            scored=[(c,score_v2(c,gi(c,dt)),saf(inds[c]['c'],gi(c,dt))) for c in inds if gi(c,dt)>=0]
            scored=[s for s in scored if s[1]>=30 and s[2] and s[2]>0]
        else:
            # 动量追涨
            scored=[]
            for code in inds:
                di=gi(code,dt);di20=gi(code,adates[max(0,i-20)])
                if di<0 or di20<0:continue
                p1=saf(inds[code]['c'],di);p2=saf(inds[code]['c'],di20)
                if p1 and p2 and p2>0:
                    ret=(p1-p2)/p2*100
                    if ret>0:scored.append((code,round(ret,1),p1))
        
        if len(scored)<3:eq.append(eq[-1]);continue
        scored.sort(key=lambda x:-x[1])
        tp=0;fp=0;cnt=0
        for code,sc,pr in scored[:5]:
            di_f=gi(code,fwd)
            if di_f<0:continue
            pr_f=saf(inds[code]['c'],di_f)
            if pr_f and pr_f>0:tp+=pr;fp+=pr_f;cnt+=1
        if cnt>=3:eq.append(eq[-1]*(fp/tp))
        else:eq.append(eq[-1])
    
    cum=(eq[-1]/eq[0]-1)*100
    yr=len(eq)*20/252
    ann=round(((eq[-1]/eq[0])**(1/yr)-1)*100,1) if yr>0.5 else 0
    peak=eq[0];mdd=0
    for v in eq:
        if v>peak:peak=v
        dd=(peak-v)/peak*100
        if dd>mdd:mdd=dd
    rc_str=' '.join(f"{k}:{v}" for k,v in rc.items())
    print(f"{label:<12s} 年化{ann:+.1f}% 累计{cum:+6.1f}% 回撤{mdd:5.1f}% [{rc_str}]")

print("\n✅ 完成")
