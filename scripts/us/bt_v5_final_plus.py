#!/usr/bin/env python3
"""V5 Final Plus - 多周期加权动量 + 波动率平价"""
import json, os, warnings, time, numpy as np
warnings.filterwarnings('ignore')
CACHE="/home/admin/.openclaw/workspace/data/cache"
UNIVERSE="/home/admin/.openclaw/workspace/data/sp500_universe.json"
pool=json.load(open(UNIVERSE));tickers=pool['tickers']
print("Loading...",flush=True)

loaded={}
for t in tickers:
    try:
        raw=json.load(open(f"{CACHE}/{t}.json"))['data']
        n=len(raw);c=[float(raw[i]['close']) for i in range(n)]
        rs=[50.0]*n
        if n>14:
            ag=sum(max(c[j]-c[j-1],0) for j in range(1,15))/14
            al=sum(max(c[j-1]-c[j],0) for j in range(1,15))/14
            rs[14]=100-100/(1+ag/al) if al>0 else 100
            for i in range(15,n):
                ag=(ag*13+max(c[i]-c[i-1],0))/14
                al=(al*13+max(c[i-1]-c[i],0))/14
                rs[i]=100-100/(1+ag/al) if al>0 else 100
        # Daily returns for volatility
        dr=[0.0]+[(c[i]/c[i-1]-1)*100 for i in range(1,n)]
        result={}
        for i in range(60,n):
            d=raw[i]['date'];pr=c[i]
            hp52=max(c[max(0,i-251):i+1]);p52=pr/hp52*100 if hp52>0 else 100
            vol=np.std(dr[i-19:i+1])*(252**0.5) if i>=19 else 1
            result[d]={'p':pr,'p52':p52,'rsi':rs[i],'vol':vol,
                'm5':(pr/c[i-5]-1)*100 if i>=5 else 0,
                'm10':(pr/c[i-10]-1)*100 if i>=10 else 0,
                'm20':(pr/c[i-20]-1)*100 if i>=20 else 0,
                'm30':(pr/c[i-30]-1)*100 if i>=30 else 0,
                'm60':(pr/c[i-60]-1)*100 if i>=60 else 0}
        loaded[t]=result
    except:pass

dates_list=sorted(set(d for t in loaded for d in loaded[t]))
years=sorted(set(int(d[:4]) for d in dates_list))
print("Stocks: %d, Years: %d"%(len(loaded),len(years)))

def run_bt(p):
    HD=p.get('hd',10);TN=p.get('tn',3);IC=p.get('ic',2);RMAX=p.get('rmax',59)
    MULTI=p.get('multi',False);VOLPAR=p.get('volpar',False)
    yearly_rets=[]
    for y in years:
        if y>2025:continue
        yr_dates=[d for d in dates_list if '%d-01-02'%y<=d<='%d-12-31'%y]
        if len(yr_dates)<60:continue
        rets=[]
        for si in range(HD,len(yr_dates)-HD,HD):
            db=yr_dates[si];ds=yr_dates[min(si+HD,len(yr_dates)-1)]
            cand=[]
            for t,td in loaded.items():
                vb=td.get(db)
                if not vb or not vb['p']:continue
                if MULTI:  # Weighted multi-timeframe
                    m5=vb.get('m5',0);m30=vb.get('m30',0);m60=vb.get('m60',0)
                    score=m5*0.25+m30*0.5+m60*0.25
                else:  # Traditional with MM
                    m30=vb.get('m30',0);m5=vb.get('m5',0)
                    if m30<=0 or m5<=0:continue
                    score=m30*(1-min(max(0,(vb['p52']-50)/50)*0.3,1))
                if score<=0:continue
                if vb.get('rsi',50)>=RMAX:continue
                cand.append((score,t,vb['p'],vb.get('vol',1)))
            if not cand:continue
            cand.sort(key=lambda x:-x[0])
            if IC>0:
                f=[];sc={}
                for s,t,p,v in cand:
                    sec={'PG':'ConsDef','SCHW':'Fin'}.get(t,'Tech')
                    if sc.get(sec,0)>=IC:continue
                    sc[sec]=sc.get(sec,0)+1;f.append((t,p,v))
                pos=f[:TN]
            else:pos=[(t,p,v) for s,t,p,v in cand[:TN]]
            if not pos:continue
            pr=[]
            if VOLPAR:  # Volatility parity weighting
                total_inv=sum(1/max(p[2],1) for p in pos)
                for t,bp,v in pos:
                    vs=loaded[t].get(ds)
                    if vs and bp>0:
                        w=(1/max(v,1))/total_inv
                        pr.append((vs['p']/bp-1)*100*w)
            else:
                for t,bp,v in pos:
                    vs=loaded[t].get(ds)
                    if vs and bp>0:pr.append((vs['p']/bp-1)*100)
            if pr:rets.append(np.mean(pr)*3 if VOLPAR else np.mean(pr))  # Normalize
        if rets:yearly_rets.append(sum(rets))
    cum=sum(yearly_rets);ny=len([r for r in yearly_rets if r!=0])
    ann=((1+cum/100)**(1/ny)-1)*100 if cum>-100 and ny>0 else 0
    sh=np.mean(yearly_rets)/np.std(yearly_rets)*(12**0.5) if len(yearly_rets)>2 and np.std(yearly_rets)>0 else 0
    cv=100;pk=100;md=0
    for r in yearly_rets:cv*=1+r/100;pk=max(pk,cv);md=max(md,(pk-cv)/pk*100)
    return{'ann':round(ann,2),'sharpe':round(sh,2),'mdd':round(md,1)}

# ── TESTS ──
BASE={'hd':10,'tn':3,'ic':2,'rmax':59,'multi':False,'volpar':False}
print("\n=== 对比 ===")
tests=[('V5基准',{'hd':10,'tn':3,'ic':2,'rmax':59,'multi':False,'volpar':False})]
tests.append(('加权动量(0.25+0.5+0.25)',{**BASE,'multi':True}))
tests.append(('+波动率平价',{**BASE,'multi':True,'volpar':True}))
tests.append(('波动率仅',{**BASE,'volpar':True}))
# Test different weights
for w5 in [0.2,0.25,0.3]:
    for w30 in [0.4,0.5,0.6]:
        w60=1-w5-w30
        tests.append(('W%.1f+%.1f+%.1f'%(w5,w30,w60),{**BASE,'multi':True,'w5':w5,'w30':w30,'w60':w60}))

for label,params in tests:
    r=run_bt(params)
    m=' <<<' if r==tests[0] else ''
    print(f"  {label:>25s}: {r['ann']:>+6.2f}% ann {r['sharpe']:>5.2f} sharpe {r['mdd']:>4.1f}% mdd")

# ── If weighted multi works, find best combo ──
print("\n=== 加权动量+HD+RSI 最优 ===")
best=0
for hd in [10,15]:
    for rmax in [59,65]:
        for w5 in [0.2,0.25]:
            for w30 in [0.4,0.5]:
                w60=1-w5-w30
                p={**BASE,'hd':hd,'rmax':rmax,'multi':True,'w5':w5,'w30':w30,'w60':w60}
                r=run_bt(p)
                if r['ann']>best:
                    best=r['ann']
                    print(f"  🔥 {hd}d RSI<{rmax} W({w5}+{w30}+{w60}): {r['ann']}% ann {r['sharpe']} sharpe {r['mdd']}% mdd")

print("\n✅ DONE")
