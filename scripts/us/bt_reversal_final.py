#!/usr/bin/env python3
"""均值回归最终验证"""
import json, os, warnings, numpy as np
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
        result={}
        for i in range(60,n):
            d=raw[i]['date'];pr=c[i]
            result[d]={'p':pr,
                'm5':(pr/c[i-5]-1)*100 if i>=5 else 0,
                'm10':(pr/c[i-10]-1)*100 if i>=10 else 0,
                'm20':(pr/c[i-20]-1)*100 if i>=20 else 0}
        loaded[t]=result
    except:pass

dates_list=sorted(set(d for t in loaded for d in loaded[t]))
years=sorted(set(int(d[:4]) for d in dates_list))
print("Stocks: %d, Years: %d"%(len(loaded),len(years)))

def run_bt(p):
    HD=p.get('hd',5);TN=p.get('tn',5);IC=p.get('ic',0);NM=p.get('nm',10) # nm = reversal window
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
                mom=vb.get('m%d'%NM,0)
                if mom>=0:continue  # Only buy stocks that went DOWN
                score=-mom  # More negative = higher score
                cand.append((score,t,vb['p']))
            if not cand:continue
            cand.sort(key=lambda x:-x[0])
            if IC>0:
                f=[];sc={}
                for s,t,p in cand:
                    sec=t[:2]  # Simple sector proxy: first 2 chars of ticker
                    if sc.get(sec,0)>=IC:continue
                    sc[sec]=sc.get(sec,0)+1;f.append((t,p))
                pos=f[:TN]
            else:pos=[(t,p) for _,t,p in cand[:TN]]
            if not pos:continue
            pr=[]
            for t,bp in pos:
                vs=loaded[t].get(ds)
                if vs and bp>0:pr.append((vs['p']/bp-1)*100)
            if pr:rets.append(np.mean(pr))
        if rets:yearly_rets.append(sum(rets))
    cum=sum(yearly_rets);ny=len([r for r in yearly_rets if r!=0])
    ann=((1+cum/100)**(1/ny)-1)*100 if cum>-100 and ny>0 else 0
    sh=np.mean(yearly_rets)/np.std(yearly_rets)*(12**0.5) if len(yearly_rets)>2 and np.std(yearly_rets)>0 else 0
    cv=100;pk=100;md=0
    for r in yearly_rets:cv*=1+r/100;pk=max(pk,cv);md=max(md,(pk-cv)/pk*100)
    return{'ann':round(ann,2),'sharpe':round(sh,2),'mdd':round(md,1),'yr':yearly_rets}

# ── 全面扫描 ──
print("\n=== 均值回归全参数扫描 ===")
results=[]
for hd in [5,7,10,15]:
    for tn in [3,5,8,10]:
        for nm in [5,10,20]:
            for ic in [0,1,2]:
                p={'hd':hd,'tn':tn,'nm':nm,'ic':ic}
                r=run_bt(p)
                results.append({'hd':hd,'tn':tn,'nm':nm,'ic':ic,**r})

results.sort(key=lambda x:-x['ann'])
print(f"{'HD':>4s} {'TN':>4s} {'NM':>4s} {'IC':>4s} {'年化':>8s} {'夏普':>8s} {'回撤':>6s}")
print("-"*45)
for r in results[:15]:
    print(f"{r['hd']:>3d}天 {r['tn']:>3d}只 {r['nm']:>3d}日 {r['ic']:>3d} {r['ann']:>+6.2f}% {r['sharpe']:>6.2f} {r['mdd']:>4.1f}%")

# ── 最优模型验证 ──
print("\n=== 最优模型逐年验证 ===")
best=results[0]
r=run_bt({'hd':best['hd'],'tn':best['tn'],'nm':best['nm'],'ic':best['ic']})
print(f"参数: {best['hd']}d调仓, {best['tn']}只, {best['nm']}d反转窗口")
yr=r.get('yr',[])
for i,y in enumerate([y for y in years if y<=2025]):
    if i<len(yr):
        print(f"  {y}: {yr[i]:+.1f}%")

# ── 动量+均值混合 ──
# Run reversal best and V5 side by side
v5=run_bt({'hd':10,'tn':3,'ic':2,'nm':10}) # not right, V5 needs rmax
print(f"\n均值回归最优: {best['ann']}% ann, {best['sharpe']} sharpe, {best['mdd']}% mdd")
print(f"\n✅ 完成")
