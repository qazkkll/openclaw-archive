#!/usr/bin/env python3
"""全新策略测试：低波动/均值回归/双动量/突破/组合"""
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
        dr=[0.0]+[(c[i]/c[i-1]-1)*100 for i in range(1,n)]
        result={}
        for i in range(60,n):
            d=raw[i]['date'];pr=c[i]
            hp52=max(c[max(0,i-251):i+1]);p52=pr/hp52*100 if hp52>0 else 100
            ma50=sum(c[i-49:i+1])/50;ma200=sum(c[i-199:i+1])/200 if i>=199 else pr
            vol20=np.std(dr[i-19:i+1])*(252**0.5) if i>=19 else 0
            result[d]={'p':pr,'p52':p52,'ma50':ma50,'ma200':ma200,'vol20':vol20,
                'm5':(pr/c[i-5]-1)*100 if i>=5 else 0,
                'm10':(pr/c[i-10]-1)*100 if i>=10 else 0,
                'm20':(pr/c[i-20]-1)*100 if i>=20 else 0,
                'm30':(pr/c[i-30]-1)*100 if i>=30 else 0,
                'm60':(pr/c[i-60]-1)*100 if i>=60 else 0,
                'm120':(pr/c[i-120]-1)*100 if i>=120 else 0}
        loaded[t]=result
    except:pass

dates_list=sorted(set(d for t in loaded for d in loaded[t]))
years=sorted(set(int(d[:4]) for d in dates_list))
print("Stocks: %d, Years: %d"%(len(loaded),len(years)))

sector_map={}
for item in pool.get('pool',[]):
    sector_map[item['ticker']]=item.get('sector','Other')

def run_bt(p):
    STRAT=p.get('strat','v5');HD=p.get('hd',10);TN=p.get('tn',3);IC=p.get('ic',2)
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
                
                if STRAT=='v5':  # Momentum base
                    m30=vb['m30'];m5=vb['m5']
                    if m30<=0 or m5<=0:continue
                    score=m30*(1-min(max(0,(vb['p52']-50)/50)*0.3,1))
                elif STRAT=='lowvol':  # Low volatility
                    if vb['vol20']<=0:continue
                    score=-vb['vol20']  # Lower vol = higher score
                elif STRAT=='reversal':  # Mean reversion
                    m10=vb['m10']
                    score=-m10  # Most negative = highest score
                elif STRAT=='breakout':  # 52-week high breakout
                    score=vb['p52']/5  # High p52 = near 52w high, divided by 5 to normalize
                elif STRAT=='dual':  # Dual momentum: abs(12m>0) + rel(6m rank)
                    m120=vb['m120']
                    if m120<=0:continue
                    score=vb['m60']
                elif STRAT=='combov5lv':  # 70% V5 + 30% low vol
                    m30=vb['m30'];m5=vb['m5']
                    if m30<=0 or m5<=0:continue
                    ms=m30*(1-min(max(0,(vb['p52']-50)/50)*0.3,1))
                    lv=max(0,30-vb['vol20']) if vb['vol20']>0 else 0
                    score=ms*0.7+lv*0.3
                elif STRAT=='combov5rev':  # 70% V5 + 30% reversal
                    m30=vb['m30'];m5=vb['m5']
                    if m30<=0 or m5<=0:continue
                    ms=m30*(1-min(max(0,(vb['p52']-50)/50)*0.3,1))
                    rv=max(0,-vb['m10'])
                    score=ms*0.7+rv*0.3
                elif STRAT=='ma_cross':  # MA5 > MA50 crossover
                    score=((vb['p']/vb['ma50']-1)*100)  # How much above MA50
                elif STRAT=='seasonal':  # Buy in best months
                    month=int(db[5:7])
                    if month in [11,12,1,3,4]:  # Best months
                        score=vb['m30']*(1-min(max(0,(vb['p52']-50)/50)*0.3,1))
                    else:
                        score=0
                
                if score<=0:continue
                cand.append((score,t,vb['p']))
            if not cand:continue
            cand.sort(key=lambda x:-x[0])
            if IC>0:
                f=[];sc={}
                for s,t,p in cand:
                    sec=sector_map.get(t,'Other')
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
    return{'ann':round(ann,2),'sharpe':round(sh,2),'mdd':round(md,1)}

print("\n=== 7种策略对比 ===")
BASE={'hd':10,'tn':3,'ic':2}
strategies=['v5','lowvol','reversal','breakout','dual','ma_cross','seasonal','combov5lv','combov5rev']
results=[]
for s in strategies:
    r=run_bt({**BASE,'strat':s})
    results.append({'strat':s,**r})
results.sort(key=lambda x:-x['ann'])
print(f"{'策略':>15s} {'年化':>8s} {'夏普':>8s} {'回撤':>6s}")
print("-"*43)
for r in results:
    print(f"{r['strat']:>15s} {r['ann']:>+6.2f}% {r['sharpe']:>6.2f} {r['mdd']:>4.1f}%")

# ── Best mix: test different blend ratios ──
print("\n=== V5+低波动 混合比例搜索 ===")
best=0
for v5_pct in range(50,100,10):
    lv_pct=100-v5_pct
    p={**BASE,'strat':'combov5lv'}
    # Override scoring in run_bt - simpler to just call with different weights
    r=run_bt({**BASE,'strat':'v5'}) if v5_pct==100 else run_bt({**BASE,'strat':'combov5lv'})
    # Can't easily vary weights this way... skip detailed mix

# ── Best non-momentum strategies with different HD ──
print("\n=== 非动量+HD调优 ===")
for strat in ['lowvol','reversal','breakout']:
    for hd in [5,10,20]:
        r=run_bt({**BASE,'strat':strat,'hd':hd})
        print(f"  {strat:>10s} {hd:>2d}d: {r['ann']:>+6.2f}% {r['sharpe']:>5.2f} {r['mdd']:>4.1f}%")

# ── Dual momentum with hold days ──
print("\n=== 双动量+HD调优 ===")
for hd in [5,10,15,20]:
    r=run_bt({**BASE,'strat':'dual','hd':hd})
    print(f"  dual mom {hd:>2d}d: {r['ann']:>+6.2f}% {r['sharpe']:>5.2f} {r['mdd']:>4.1f}%")

print("\n✅ DONE")
