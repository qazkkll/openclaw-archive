#!/usr/bin/env python3
"""V3 A股第三轮：去行业限制 + 纯股票筛选 + 简化卖出规则"""
import json, sys, warnings, time
warnings.filterwarnings('ignore')
from collections import defaultdict

print("📥 加载数据...")
t0=time.time()
with open('data/backtest_hist_yahoo.json') as f: hist=json.load(f)
with open('data/sector_map.json') as f: smap=json.load(f)
EXCLUDED={'地产基建','农业','交通物流'}
ETFS={'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}
codes=[c for c in hist if c not in ETFS and len(hist[c].get('close',[]))>500]
adates=sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2015-01-01'<=d<='2026-05-14'))
print(f"📊 {len(codes)}只股票 📅 {len(adates)}天 耗时{time.time()-t0:.1f}s")

ss_excl=defaultdict(list); ss_all=defaultdict(list)
for c in codes:
    sec=smap.get(c,'其他')
    if sec not in EXCLUDED: ss_excl[sec].append(c)
    ss_all[sec].append(c)

cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes if hist[c].get('dates')}
def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

def ci(code):
    d=hist.get(code); 
    if not d: return None
    c=d.get('close',[]);h=d.get('high',[]);l=d.get('low',[]);n=len(c)
    if n<60: return None
    def sma(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def ema(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    m5=sma(c,5); m20=sma(c,20); m60=sma(c,60)
    e12=ema(c,12); e26=ema(c,26); ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9); mh=[ml[i]-sg[i] for i in range(n)]
    gl,ll=[],[]
    for i in range(1,n): diff=c[i]-c[i-1]; gl.append(max(diff,0)); ll.append(max(-diff,0))
    rsi=[None]*14; ag=sum(gl[:14])/14 if len(gl)>=14 else 0; al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i<len(gl): ag=(ag*13+gl[i])/14; al=(al*13+ll[i])/14
    adx=[None]*27; tr_h,dp_h,dm_h=[],[],[]
    for i in range(1,n):
        tr=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])); dp=max(0,h[i]-h[i-1]); dm=max(0,l[i-1]-l[i])
        tr_h.append(tr);dp_h.append(dp);dm_h.append(dm)
        if i<14: continue
        tr14=sum(tr_h[-14:]);dp14=sum(dp_h[-14:]);dm14=sum(dm_h[-14:]);atr=tr14/14
        if atr==0: adx.append(0); continue
        dip=dp14/14/atr*100; dim=dm14/14/atr*100
        if dip+dim==0: adx.append(0); continue
        dx=abs(dip-dim)/(dip+dim)*100
        if i<27: adx.append(dx); continue
        adx.append((sum(a for a in adx[-13:] if a is not None)+dx)/14)
    while len(adx)<n: adx.append(None)
    p52=[None]*251
    for i in range(251,n):
        lo=min(c[i-250:i+1]); hi=max(c[i-250:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52}

t1=time.time()
inds={}
for code in codes:
    ind=ci(code)
    if ind: inds[code]=ind
print(f"📊 {len(inds)}只指标完成 耗时{time.time()-t1:.1f}s")

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

def score_v1(code, di):
    ind=inds.get(code)
    if not ind: return 0
    mh=saf(ind['mh'],di); mhp=saf(ind['mh'],di-1)
    ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=20
        elif mh>0 and mh>mhp: ms=12
        elif mh>0: ms=6
    if ms<=0: return 0
    p52=saf(ind['p52'],di)
    ws=0
    if p52 is not None:
        if p52<20: ws=20
        elif p52<35: ws=15
        elif p52<50: ws=10
        elif p52<65: ws=6
        elif p52<80: ws=3
    pr=saf(ind['c'],di); m5=saf(ind['m5'],di); m20=saf(ind['m20'],di); m60=saf(ind['m60'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=saf(ind['adx'],di)
    ads=-5
    if av is not None:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
    rv=saf(ind['rsi'],di)
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
    return min(ttl/sum(wl)*100, 100)

# sim_stock: 无行业限制，直接选评分最高的N只
def sim_stock(s, e, p, sector_dict):
    cash=1000000.0; pos={}; daily=[]
    for i in range(s, e):
        dt=adates[i]
        if (i-s)%p['rebal']==0:
            # 评分所有股票
            scorings=[]
            for code in inds:
                sc=score_v1(code, i)
                if sc>=p['buy']:
                    pr=saf(inds[code]['c'], i)
                    if pr and pr>0: scorings.append((code, sc, pr))
            scorings.sort(key=lambda x:-x[1])
            # 清仓不在top的
            top_codes=set(x[0] for x in scorings[:p['maxp']*2])
            for c in list(pos.keys()):
                if c not in top_codes:
                    pr=saf(inds[c]['c'], i)
                    if pr and pr>0:
                        cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                    del pos[c]
            # 买入top
            for code, sc, pr in scorings:
                if len(pos)>=p['maxp']: break
                if code in pos: continue
                inv=min(cash*p['pct'], cash*0.95)
                if inv<20000: continue
                pos[code]={'e':pr,'v':inv}; cash-=inv
        # 每日卖出检查
        for c in list(pos.keys()):
            sc=score_v1(c, i)
            pr=saf(inds[c]['c'], i)
            m20=saf(inds[c]['m20'], i); mh=saf(inds[c]['mh'], i)
            today_sell=False
            if p.get('sell_simple'):
                # 简化卖出：只按评分
                if sc<p['sell']: today_sell=True
            else:
                # 原版卖出
                if sc<p['sell'] or (pr and m20 and mh is not None and pr<m20 and mh<0):
                    today_sell=True
            if today_sell:
                if pr and pr>0: cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                del pos[c]
        tv=cash
        for c,px in pos.items():
            pr=saf(inds[c]['c'], i)
            if pr and pr>0: tv+=px['v']*pr/px['e']
            else: tv+=px['v']
        daily.append(tv)
    for c,px in list(pos.items()):
        di=gi(c, adates[e-1])
        pr=saf(inds[c]['c'], di) if di>=0 and c in inds else None
        if pr and pr>0: cash+=px['v']*(1+(pr-px['e'])/px['e'])
        else: cash+=px['v']
    ret=(cash-1000000)/1000000*100
    peak=max(daily) if daily else 1000000
    mdd=max(((peak-v)/peak*100) for v in daily) if daily else 0
    dr=[(daily[j]-daily[j-1])/daily[j-1]*100 for j in range(1,len(daily)) if daily[j-1]>0]
    sr=0
    if len(dr)>5:
        avg=sum(dr)/len(dr); var=sum((r-avg)**2 for r in dr)/len(dr); std=max(var**0.5,0.001)
        sr=round(avg/std*15.8,2)
    return round(ret,2), round(mdd,2), sr

# 原版sim（带行业筛选）
def sim_orig(s, e, p, sector_dict):
    cash=1000000.0; pos={}; daily=[]
    for i in range(s, e):
        dt=adates[i]
        if (i-s)%p['rebal']==0:
            mom={}
            for sec,cls in sector_dict.items():
                rets=[]
                for c in cls[:20]:
                    di=gi(c,dt); di20=gi(c,adates[max(0,i-20)])
                    if di<0 or di20<0 or c not in inds: continue
                    pr=saf(inds[c]['c'],di); p20=saf(inds[c]['c'],di20)
                    if pr and p20 and p20>0: rets.append((pr-p20)/p20*100)
                if len(rets)>=2: mom[sec]=sum(rets)/len(rets)
            if not mom: continue
            rk=sorted(mom.items(), key=lambda x:-x[1])
            ts=[r[0] for r in rk[:p['top']]]
            hs=[r[0] for r in rk[:p['hold']]]
            for c in list(pos.keys()):
                if pos[c]['s'] not in hs:
                    di=gi(c,dt)
                    pr=saf(inds[c]['c'],di) if di>=0 else None
                    if pr and pr>0:
                        cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                    del pos[c]
            scs=defaultdict(list)
            for sec in ts:
                for c in sector_dict.get(sec,[]):
                    if c in pos: continue
                    di=gi(c,dt)
                    if di<0 or c not in inds: continue
                    sc=score_v1(c,di)
                    if sc>=p['buy']:
                        pr=saf(inds[c]['c'],di)
                        if pr and pr>0: scs[sec].append((c,sc,pr))
            for sec in ts:
                scs[sec].sort(key=lambda x:-x[1])
                for c,sc,pr in scs[sec][:p['per_sec']]:
                    if len(pos)>=p['maxp']: break
                    inv=min(cash*p['pct'],cash*0.95)
                    if inv<20000: continue
                    pos[c]={'e':pr,'v':inv,'s':sec}; cash-=inv
        for c in list(pos.keys()):
            sc=score_v1(c,i)
            pr=saf(inds[c]['c'],i)
            m20=saf(inds[c]['m20'],i); mh=saf(inds[c]['mh'],i)
            if sc<p['sell'] or (pr and m20 and mh is not None and pr<m20 and mh<0):
                if pr and pr>0: cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                del pos[c]
        tv=cash
        for c,px in pos.items():
            pr=saf(inds[c]['c'],i)
            if pr and pr>0: tv+=px['v']*pr/px['e']
            else: tv+=px['v']
        daily.append(tv)
    for c,px in list(pos.items()):
        di=gi(c,adates[e-1])
        pr=saf(inds[c]['c'],di) if di>=0 and c in inds else None
        if pr and pr>0: cash+=px['v']*(1+(pr-px['e'])/px['e'])
        else: cash+=px['v']
    ret=(cash-1000000)/1000000*100
    peak=max(daily) if daily else 1000000
    mdd=max(((peak-v)/peak*100) for v in daily) if daily else 0
    dr=[(daily[j]-daily[j-1])/daily[j-1]*100 for j in range(1,len(daily)) if daily[j-1]>0]
    sr=0
    if len(dr)>5:
        avg=sum(dr)/len(dr); var=sum((r-avg)**2 for r in dr)/len(dr); std=max(var**0.5,0.001)
        sr=round(avg/std*15.8,2)
    return round(ret,2), round(mdd,2), sr

years=[]
for y in range(2015,2026):
    s=next((i for i,dt in enumerate(adates) if dt>=f'{y}-01-01'), None)
    e=next((i for i,dt in enumerate(adates) if dt>=f'{y+1}-01-01'), len(adates))
    if s and e and e-s>30: years.append((y,s,e))
csi={2015:5.58,2016:-11.28,2017:21.78,2018:-25.31,2019:36.07,2020:27.21,2021:-5.20,2022:-21.63,2023:-11.38,2024:14.68,2025:-3.67}

# ========== 第三轮配置 ==========
configs = []
# U组：全市场直接选票（无行业限制）
for sell in [48, 50, 53]:
    configs.append(('U-6只/卖{sell}/7天直选'.format(sell=sell), {'buy':62,'sell':sell,'rebal':7,'maxp':6,'pct':0.16,'sell_simple':True}, 'stock'))
# V组：全市场直选 + 简化卖出规则
for sell in [48, 50, 53]:
    configs.append(('V-6只/卖{sell}/7天直选简卖'.format(sell=sell), {'buy':62,'sell':sell,'rebal':7,'maxp':6,'pct':0.16,'sell_simple':True}, 'stock'))
# W组：全市场直选 + 8只
for sell in [48, 50]:
    configs.append(('W-8只/卖{sell}/7天直选'.format(sell=sell), {'buy':62,'sell':sell,'rebal':7,'maxp':8,'pct':0.125,'sell_simple':True}, 'stock'))
# X组：带行业但不额外卖（简化卖出）
for sell in [50, 53]:
    configs.append(('X-6只/卖{sell}/7天行简卖'.format(sell=sell), {'buy':62,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16,'sell_simple':True}, 'orig_excl'))
# Y组：降低买入门槛
for buy in [58, 60]:
    for sell in [48, 50]:
        configs.append(('Y-6只/买{buy}/卖{sell}/7天直选'.format(buy=buy,sell=sell), {'buy':buy,'sell':sell,'rebal':7,'maxp':6,'pct':0.16,'sell_simple':True}, 'stock'))

results=[]
for name, p, mode in configs:
    t2=time.time()
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    cum=1000000; srs=[]; yearly=[]
    for y,s,e in years:
        if mode=='stock':
            r,mdd,sr=sim_stock(s,e,p,None)
        else:
            r,mdd,sr=sim_orig(s,e,p,ss_excl)
        cum*=1+r/100; srs.append(sr)
        c=csi.get(y,0)
        tag='✅' if r>c else ('🟡' if r>c-10 else '❌')
        yearly.append((y,r,mdd,c,tag))
        print(f"  {y}: {r:+7.2f}% (CSI:{c:+.1f}%) DD{mdd:.1f}% {tag}")
    cr=round((cum/1000000-1)*100,2); nac=len(years)
    ann=round((cum/1000000)**(1/nac)*100-100,2) if cum>0 else 0
    avg_sr=round(sum(srs)/len(srs),2)
    avg_dd=round(sum(y[2] for y in yearly)/len(yearly),1)
    print(f"  {'─'*50}")
    print(f"  累计: {cr:+.2f}% | 年化: {ann}% | 夏普: {avg_sr} | 均回撤: {avg_dd}%")
    print(f"  耗时: {time.time()-t2:.1f}s")
    results.append((name, cr, ann, avg_sr, avg_dd))

print(f"\n\n{'='*70}")
print(f"🏆 排名")
print(f"{'='*70}")
results.sort(key=lambda x:-x[2])
print(f"{'配置':<48} {'累计':>8} {'年化':>6} {'夏普':>6} {'均回撤':>7}")
print(f"{'─'*48} {'─'*8} {'─'*6} {'─'*6} {'─'*7}")
for name, cr, ann, sr, dd in results:
    print(f"{name:<48} {cr:>+7.2f}% {ann:>5.2f}% {sr:>5.2f} {dd:>6.1f}%")
