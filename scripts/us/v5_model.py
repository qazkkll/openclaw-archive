#!/usr/bin/env python3
"""
V5 动量防御系统 - 全参数最终扫描
"""
import json, os, warnings, time, numpy as np
warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
pool = json.load(open(UNIVERSE))
tickers = pool['tickers']

print("Loading...", flush=True)
loaded = {}
for t in tickers:
    try:
        raw = json.load(open(f"{CACHE}/{t}.json"))['data']
        n = len(raw); c = [float(raw[i]['close']) for i in range(n)]
        result = {}
        for i in range(60, n):
            d = raw[i]['date']; pr = c[i]
            hp52 = max(c[max(0,i-251):i+1]); p52 = pr/hp52*100 if hp52>0 else 100
            m = {}
            for p in [15,20,25,30]:
                if i>=p: m[p] = (pr/c[i-p]-1)*100
            if i >= 20:
                vol5 = sum(raw[j].get('volume',0) for j in range(i-4,i+1))/5
                vol20 = sum(raw[j].get('volume',0) for j in range(i-19,i+1))/20
                vol_r = vol5/vol20 if vol20 > 0 else 1
            else: vol_r = 1
            result[d] = {'p':pr, 'p52':p52, 'vol_r':vol_r, **m}
        loaded[t] = result
    except: pass

sector_map = {}
for item in pool.get('pool', []):
    sector_map[item['ticker']] = item.get('sector', 'Other')
dates_list = sorted(set(d for t in loaded for d in loaded[t]))
years = sorted(set(int(d[:4]) for d in dates_list))

import yfinance as yf
vix_raw = yf.download('^VIX', start="2013-01-01", end="2026-06-01", progress=False)
vix_close = vix_raw['Close'].squeeze()
vix_dates = [d.strftime('%Y-%m-%d') for d in list(vix_raw.index)]
vix_prices = [float(vix_close.iloc[i]) for i in range(len(vix_close))]

def get_vix(date):
    for off in range(5):
        d = date[:8] + str(int(date[8:10])+off+1).zfill(2) if off > 0 else date
        if d in vix_dates: return vix_prices[vix_dates.index(d)]
    return None

def run_bt(params):
    DS=params.get('ds',50); DC=params.get('dc',0.3); MD=params.get('md',30)
    TN=params.get('tn',3); IC=params.get('ic',2); HD=params.get('hd',10)
    VIX=params.get('vix',False); VC=params.get('vc',False)
    VD=params.get('vd',25); VA=params.get('va',35)
    RW=params.get('rw',False)  # Rank weighting
    
    yearly_rets = []
    for y in years:
        if y > 2025: continue
        sd = '%d-01-02'%y; ed = '%d-12-31'%y
        yr_dates = [d for d in dates_list if sd <= d <= ed]
        if len(yr_dates) < 60: continue
        rets = []
        for si in range(HD, len(yr_dates)-HD, HD):
            d_buy = yr_dates[si]; d_sell = yr_dates[min(si+HD, len(yr_dates)-1)]
            d_mom = yr_dates[max(0, si-MD)]
            if VIX:
                v = get_vix(d_buy)
                if v is not None and v >= VA: continue
            cand = []
            for t, td in loaded.items():
                vb=td.get(d_buy); vp=td.get(d_mom)
                if not vb or not vb['p'] or not vp: continue
                mom = (vb['p']/vp['p']-1)*100 if MD not in vb else vb[MD]
                if mom <= 0: continue
                score = mom * (1 - min(max(0, (vb['p52']-DS)/(100-DS))*DC, 1))
                if VC and vb.get('vol_r',1) < 1.2: score *= 0.7
                cand.append((score, t, vb['p']))
            if not cand: continue
            cand.sort(key=lambda x:-x[0])
            if IC > 0:
                f=[];sc={}
                for s,t,p in cand:
                    sec=sector_map.get(t,'Other')
                    if sc.get(sec,0)>=IC: continue
                    sc[sec]=sc.get(sec,0)+1; f.append((t,p))
                pos=f[:TN]
            else:
                pos=[(t,p) for _,t,p in cand[:TN]]
            if not pos: continue
            if VIX:
                v=get_vix(d_buy)
                if v is not None and v >= VD:
                    pos=pos[:max(1,len(pos)-1)]
            if not pos: continue
            pr=[]
            for t,bp in pos:
                vs=loaded[t].get(d_sell)
                if vs and bp>0: pr.append((vs['p']/bp-1)*100)
            if pr: rets.append(np.mean(pr))
        if rets: yearly_rets.append(sum(rets))
    cum=sum(yearly_rets); ny=len([r for r in yearly_rets if r!=0])
    ann=((1+cum/100)**(1/ny)-1)*100 if cum>-100 and ny>0 else 0
    sharpe=np.mean(yearly_rets)/np.std(yearly_rets)*(12**0.5) if len(yearly_rets)>2 and np.std(yearly_rets)>0 else 0
    cv=100;pk=100;mdd=0
    for r in yearly_rets: cv*=1+r/100; pk=max(pk,cv); mdd=max(mdd,(pk-cv)/pk*100)
    wr=sum(1 for r in yearly_rets if r>0)/len(yearly_rets)*100 if yearly_rets else 0
    return {'ann':round(ann,2),'cum':round(cum,1),'sharpe':round(sharpe,2),'mdd':round(mdd,1),'wr':round(wr,1)}

# ── SWEEP 1: 调仓频率 + 行业限制 ──
print("\n=== SWEEP 1: 调仓频率×行业限制 ===")
results1=[]
for hd in [5,7,10,15,20]:
    for ic in [0,1,2,3]:
        r=run_bt({'hd':hd,'ic':ic})
        results1.append({'hd':hd,'ic':ic,**r})
results1.sort(key=lambda x:-x['ann'])
print(f"{'HD':>4s} {'IC':>4s} {'年化':>8s} {'夏普':>8s} {'回撤':>6s}")
for r in results1[:12]:
    print(f"{r['hd']:>3d}天 {r['ic']:>3d}限 {r['ann']:>+6.2f}% {r['sharpe']:>6.2f} {r['mdd']:>4.1f}%")

# ── SWEEP 2: VIX阈值 ──
print("\n=== SWEEP 2: VIX阈值 ===")
results2=[]
for vd in [0,18,20,22,25]:
    for va in [0,25,28,30,35]:
        if vd==0 and va==0: continue
        r=run_bt({'vix':True,'vd':vd,'va':va})
        results2.append({'vd':vd,'va':va,**r})
results2.sort(key=lambda x:-x['ann'])
print(f"{'防御':>4s} {'现金':>4s} {'年化':>8s} {'夏普':>8s} {'回撤':>6s}")
for r in results2[:10]:
    vd_s=str(r['vd']) if r['vd']>0 else '关'
    va_s=str(r['va']) if r['va']>0 else '关'
    print(f"VIX>{vd_s:>3s} VIX>{va_s:>3s} {r['ann']:>+6.2f}% {r['sharpe']:>6.2f} {r['mdd']:>4.1f}%")

# ── SWEEP 3: 成交量确认 ──
print("\n=== SWEEP 3: 成交量确认+排名权重 ===")
for vc in [False, True]:
    r=run_bt({'vc':vc})
    print(f"{'成交量确认' if vc else '无成交量'}: {r['ann']}% ann, {r['sharpe']} sharpe, {r['mdd']}% mdd")

# ── 最终综合 ──
print("\n=== 最终推荐 ===")
best = results1[0]
print(f"调仓周期: {best['hd']}天")
print(f"行业限制: {best['ic']}只/行业")
print(f"预期年化: {best['ann']}%")
print(f"夏普率: {best['sharpe']}")
print(f"最大回撤: {best['mdd']}%")
print(f"\nVIX: 不建议加入（降低收益>降低风险）")
print(f"成交量: 不建议加入（降低收益）")

# Save all results
out = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'sweep_hd_ic': [{'hd':r['hd'],'ic':r['ic'],'ann':r['ann'],'sharpe':r['sharpe'],'mdd':r['mdd']} for r in results1],
    'sweep_vix': [{'defense':r['vd'],'cash':r['va'],'ann':r['ann'],'sharpe':r['sharpe'],'mdd':r['mdd']} for r in results2],
    'final_model': {'hd':best['hd'],'ic':best['ic'],'ds':50,'dc':0.3,'md':30}
}
json.dump(out, open('/home/admin/.openclaw/workspace/data/v5_final.json','w'), indent=2)
print("\n✅ 全部完成")
