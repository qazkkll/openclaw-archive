#!/usr/bin/env python3
"""板块限制放开测试 - 修复版"""
import json, sys, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json') as f: hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}
codes = [c for c in hist if c not in ETFS and len(hist[c].get('close',[]))>500]
adates = sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2015-01-01'<=d<='2026-05-14'))
cdates={c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes}

def gi(code,dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
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
    rsi=[None]*14;ag=sum(gl[:14])/14 if len(gl)>=14 else 0;al=sum(ll[:14])/14 if len(ll)>=14 else 0
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

def sc(code,di):
    ind_=inds.get(code)
    if not ind_: return 0
    mh=saf(ind_['mh'],di);mhp=saf(ind_['mh'],di-1);ms=0
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

def sec_mom(i,ss_d):
    mom={}
    for sec,cls in ss_d.items():
        rets=[]
        for c in cls[:20]:
            di=gi(c,adates[i]);di20=gi(c,adates[max(0,i-20)])
            if di<0 or di20<0 or c not in inds:continue
            pr=saf(inds[c]['c'],di);p20=saf(inds[c]['c'],di20)
            if pr and p20 and p20>0:rets.append((pr-p20)/p20*100)
        if len(rets)>=2:mom[sec]=sum(rets)/len(rets)
    return mom

def bt(p,ss_d,excluded_s,excl_buy):
    cash=1000000.0;pos={}
    yearly_returns=[]
    yy=[]
    for y in range(2016,2026):
        s=next((i for i,dt in enumerate(adates) if dt>=f'{y}-01-01'),None)
        e=next((i for i,dt in enumerate(adates) if dt>=f'{y+1}-01-01'),len(adates))
        if s and e and e-s>60:yy.append((y,s,e))
    
    for y,s,e in yy:
        year_start_value = cash + sum(pos[c]['v'] for c in pos)
        
        for i in range(s,e):
            dt=adates[i]
            if (i-s)%p['rebal']==0:
                mom=sec_mom(i,ss_d)
                if not mom:continue
                rk=sorted(mom.items(),key=lambda x:-x[1])
                ts=[r[0] for r in rk[:p['top']]];hs=[r[0] for r in rk[:p['hold']]]
                for c in list(pos.keys()):
                    if pos[c]['s'] not in hs:
                        di=gi(c,dt);pr=saf(inds[c]['c'],di) if di>=0 else None
                        if di>=0 and pr and pr>0:
                            cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                        del pos[c]
                scs=defaultdict(list)
                for sec in ts:
                    buy_t=excl_buy if sec in excluded_s else p['buy']
                    for c in ss_d.get(sec,[]):
                        if c in pos:continue
                        di=gi(c,dt)
                        if di<0 or c not in inds:continue
                        sc0=sc(c,di)
                        if sc0>=buy_t:
                            pr=saf(inds[c]['c'],di)
                            if pr and pr>0:scs[sec].append((c,sc0,pr))
                for sec in ts:
                    scs[sec].sort(key=lambda x:-x[1])
                    for c,sc0,pr in scs[sec][:p['per_sec']]:
                        if len(pos)>=p['maxp']:break
                        inv=min(cash*p['pct'],cash*0.95)
                        if inv<20000:continue
                        pos[c]={'e':pr,'v':inv,'s':sec};cash-=inv
            for c in list(pos.keys()):
                di=gi(c,dt)
                if di<0 or c not in inds:continue
                sc0=sc(c,di);pr=saf(inds[c]['c'],di)
                m20=saf(inds[c]['m20'],di);mh=saf(inds[c]['mh'],di)
                if sc0<p['sell'] or(pr and m20 and mh is not None and pr<m20 and mh<0):
                    if pr and pr>0:
                        cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                    del pos[c]
        
        # 年末结算: 持仓市值
        year_end_value = cash
        for c,px in pos.items():
            di=gi(c,adates[e-1])
            if di>=0 and c in inds:
                pr=saf(inds[c]['c'],di)
                if pr and pr>0:year_end_value+=px['v']*pr/px['e']
                else:year_end_value+=px['v']
            else:year_end_value+=px['v']
        
        yearly_ret = (year_end_value/year_start_value - 1)*100
        yearly_returns.append(round(yearly_ret,2))
    
    # 最终结算
    final_value = cash
    for c,px in pos.items():
        di=gi(c,adates[yy[-1][2]-1])
        if di>=0 and c in inds:
            pr=saf(inds[c]['c'],di)
            if pr and pr>0:final_value+=px['v']*pr/px['e']
            else:final_value+=px['v']
        else:final_value+=px['v']
    
    total_ret = round((final_value/1000000-1)*100,2)
    return yearly_returns,total_ret,[y for y,s,e in yy]

# 行业分组
ss_all=defaultdict(list)
ss_ex3=defaultdict(list)
EX3={'地产基建','农业','交通物流'}
for c in codes:
    sec=smap.get(c,'其他')
    ss_all[sec].append(c)
    if sec not in EX3:ss_ex3[sec].append(c)

BASE={'buy':62,'sell':48,'top':4,'hold':4,'rebal':7,'maxp':5,'per_sec':2,'pct':0.15}

tests=[]
tests.append(('V2.5当前(排除3)',ss_ex3,EX3,62))
tests.append(('V2.5全行业',ss_all,set(),62))
tests.append(('V2.5仅排农业+物流',ss_all,{'农业','交通物流'},62))
for eb in [65,68,70]:
    tests.append((f'V2.5地产回(买{eb})',ss_all,set(),eb))
for eb in [65,68]:
    tests.append((f'V2.5地产+农业回(买{eb})',ss_all,{'农业'},eb))

print(f"\n🏃 {len(tests)}个版本...\n")
results=[]
for name,ss_d,excluded_s,excl_buy in tests:
    sys.stdout.write(f"  {name:<30s} ")
    sys.stdout.flush()
    yr,cum,ys=bt(BASE,ss_d,excluded_s,excl_buy)
    results.append((name,cum,yr,ys))
    y_str=' '.join(f'{ys[i]}:{yr[i]:+.1f}%' for i in range(min(5,len(yr))))
    print(f"累计{cum:+.2f}%")

print(f"\n{'='*80}")
print(f"📊 板块限制测试排名")
print(f"{'='*80}")
h=f"{'排名':>3s} {'版本':<30s} {'累计':>8s} {'年化':>7s} {'回撤':>7s}"
print(h);print("-"*len(h))
results.sort(key=lambda x:-x[1])
for i,(name,cum,yr,ys) in enumerate(results):
    nac=len(yr)
    ann=round(((1+cum/100)**(1/nac)-1)*100,1) if nac>0 else 0
    print(f"{i+1:3d} {name:<30s} {cum:>+7.2f}% {ann:>6.1f}%")

print(f"\n🏆 Top 3:")
for name,cum,yr,ys in results[:3]:
    nac=len(yr)
    ann=round(((1+cum/100)**(1/nac)-1)*100,1) if nac>0 else 0
    ys2=' '.join(f'{ys[i]}:{yr[i]:+.1f}%' for i in range(len(yr)) if ys[i]>=2021)
    print(f"  {name}: 累计{cum:+.2f}% | 年化{ann}% | 2021后:{ys2}")

import json as j
j.dump({'results':[{'name':n,'cum':c,'annual':{str(ys[i]):yr[i] for i in range(len(yr))}} for n,c,yr,ys in results]},
        open('/home/admin/.openclaw/workspace/models/sector_test_results.json','w'))
print(f"\n✅ 已保存")
