#!/usr/bin/env python3
"""美股V3双模 vs V2 - 2015-2025 长周期对比"""
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
    mh=saf(ind['mh'],di);mhp=saf(ind['mh'],di-1);ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=25
        elif mh>0 and mh>mhp: ms=15
        elif mh>0: ms=8
        else: ms=-3
    if mh is None or mh<=0: return 0
    pr=saf(ind['c'],di);m20=saf(ind['m20'],di);m50=saf(ind['m50'],di)
    av=saf(ind['adx'],di);rv=saf(ind['rsi'],di);pw=saf(ind['p52'],di)
    ads=-5
    if av:
        if av>=35: ads=22
        elif av>=25: ads=15
        elif av>=20: ads=8
        elif av>=15: ads=3
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if pr and m50 and pr>m50: mas+=7
    if m20 and m50 and m20>m50: mas+=6
    rs=0
    if rv:
        if rv<25: rs=18
        elif rv<35: rs=14
        elif rv<50: rs=10
        elif rv<65: rs=6
        elif rv<75: rs=2
        else: rs=-5
    ws=0
    if pw:
        if pw<20: ws=15
        elif pw<35: ws=12
        elif pw<50: ws=8
        elif pw<65: ws=5
        elif pw<80: ws=2
    total=(ms*15/25+ads*20/22+mas*15/20+rs*20/18+ws*30/15)/100*100
    return min(total,95)

print("📥 SPY...")
spy=yf.download('SPY',start='2014-01-01',end='2026-05-16',progress=False,auto_adjust=True)
spy_c=spy['Close'].values;spy_d=spy.index.strftime('%Y-%m-%d').tolist()

def spy_mode(dt):
    for i in range(len(spy_d)-1,-1,-1):
        if spy_d[i]<=dt:
            c=float(spy_c[i]);m200=sum(float(spy_c[j]) for j in range(max(0,i-199),i+1))/min(200,i+1)
            return 'bull' if c>m200 else 'bear'
    return 'bear'

def run_year(y,mode_name):
    s=next((i for i,dt in enumerate(adates) if dt>=f'{y}-01-01'),None)
    e=next((i for i,dt in enumerate(adates) if dt>=f'{y+1}-01-01'),len(adates))
    if s is None or e is None or e-s<60: return None
    
    cash=1000000.0
    use_hybrid=(mode_name=='hybrid')
    
    for i in range(s,e,20):
        if i+20>=len(adates):break
        dt=adates[i];fwd=adates[i+20]
        rg=spy_mode(dt)
        use_mom=(mode_name=='mom' or (use_hybrid and rg=='bull'))
        
        if use_mom:
            scored=[]
            for c in inds:
                di=gi(c,dt);di20=gi(c,adates[max(0,i-20)])
                if di<0 or di20<0:continue
                p1=saf(inds[c]['c'],di);p2=saf(inds[c]['c'],di20)
                if p1 and p2 and p2>0:
                    ret=(p1-p2)/p2*100
                    if ret>0:scored.append((c,round(ret,1),p1))
        else:
            scored=[(c,score_v2(c,gi(c,dt)),saf(inds[c]['c'],gi(c,dt))) for c in inds if gi(c,dt)>=0]
            scored=[s for s in scored if s[1]>=30 and s[2] and s[2]>0]
        
        if len(scored)<3:continue
        scored.sort(key=lambda x:-x[1])
        tp=0;fp=0;cnt=0
        for code,sc,pr in scored[:5]:
            di_f=gi(code,fwd)
            if di_f<0:continue
            pr_f=saf(inds[code]['c'],di_f)
            if pr_f and pr_f>0:tp+=pr;fp+=pr_f;cnt+=1
        if cnt>=3:cash*=fp/tp
    
    ret=round((cash/1000000-1)*100,2)
    return ret

# SPY annual returns
spy_ann={}
for y in range(2016,2026):
    s_idx=None;e_idx=None
    for i in range(len(spy_d)):
        if spy_d[i]>=f'{y}-01-01' and s_idx is None: s_idx=i
        if spy_d[i]>=f'{y+1}-01-01' and e_idx is None: e_idx=i
    if s_idx and e_idx:
        spy_ann[y]=round((float(spy_c[e_idx])/float(spy_c[s_idx])-1)*100,2)

years=list(range(2016,2026))

print(f"\n📊 V3双模 vs V2 vs SPY (2016-2025, 10年)")
print(f"{'='*70}")
h=f"{'年份':>6s} {'V2逆向':>10s} {'V3双模':>10s} {'纯动量':>10s} {'SPY':>10s} {'V3跑V2':>10s}"
print(h);print("-"*len(h))

v2d={};v3d={};momd={}
for y in years:
    v2=run_year(y,'v2');v3=run_year(y,'hybrid');mom=run_year(y,'mom')
    spy_r=spy_ann.get(y,0)
    v2d[y]=v2;v3d[y]=v3;momd[y]=mom
    
    v2s=f'{v2:+.2f}%' if v2 else 'N/A'
    v3s=f'{v3:+.2f}%' if v3 else 'N/A'
    ms=f'{mom:+.2f}%' if mom else 'N/A'
    vs=f'{v3-v2:+.2f}%' if v3 and v2 else 'N/A'
    print(f"{y:>6d} {v2s:>10s} {v3s:>10s} {ms:>10s} {spy_r:>+9.2f}% {vs:>10s}")

# Cumulative
cv2=1000000;cv3=1000000;cm=1000000;cspy=1000000
for y in years:
    if v2d[y]:cv2*=1+v2d[y]/100
    if v3d[y]:cv3*=1+v3d[y]/100
    if momd[y]:cm*=1+momd[y]/100
    cspy*=1+spy_ann.get(y,0)/100

print("-"*len(h))
print(f"{'累计':>6s} {round((cv2/1000000-1)*100,2):>+9.2f}% {round((cv3/1000000-1)*100,2):>+9.2f}% {round((cm/1000000-1)*100,2):>+9.2f}% {round((cspy/1000000-1)*100,2):>+9.2f}% {round((cv3/1000000-cv2/1000000)*100,2):>+9.2f}%")

def ann(cv,y): return round((cv/1000000)**(1/y)*100-100,2) if y>0 else 0
yrs=len(years)
print(f"{'年化':>6s} {ann(cv2,yrs):>+8.1f}% {ann(cv3,yrs):>+8.1f}% {ann(cm,yrs):>+8.1f}% {ann(cspy,yrs):>+8.1f}%")

# Max drawdown
max_l2=min(v for v in v2d.values() if v is not None)
max_l3=min(v for v in v3d.values() if v is not None)
max_m=min(v for v in momd.values() if v is not None)
print(f"\n最差年: V2 {max_l2:+.1f}% | V3 {max_l3:+.1f}% | 纯动量 {max_m:+.1f}%")
print(f"100万终值: V2 ¥{round(cv2):,} | V3 ¥{round(cv3):,}")
print(f"V3双模跑赢V2: {round((cv3/cv2-1)*100,1)}%")
