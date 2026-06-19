#!/usr/bin/env python3
"""V3 A股模式第二轮优化：换评分体系冲7-8%"""
import json, sys, warnings, time
warnings.filterwarnings('ignore')
from collections import defaultdict

print("📥 加载数据...")
t0 = time.time()
with open('data/backtest_hist_yahoo.json') as f: hist = json.load(f)
with open('data/sector_map.json') as f: smap = json.load(f)
EXCLUDED = {'地产基建','农业','交通物流'}
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}
codes = [c for c in hist if c not in ETFS and len(hist[c].get('close',[]))>500]
adates = sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2015-01-01'<=d<='2026-05-14'))
print(f"📊 {len(codes)}只股票 📅 {len(adates)}天 耗时{time.time()-t0:.1f}s")

ss_excl = defaultdict(list); ss_all = defaultdict(list)
for c in codes:
    sec = smap.get(c, '其他')
    if sec not in EXCLUDED: ss_excl[sec].append(c)
    ss_all[sec].append(c)

cdates = {c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes if hist[c].get('dates')}
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
    m5=sma(c,5); m20=sma(c,20); m60=sma(c,60); m120=sma(c,120)
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
    # 20日动量
    mom20 = [None]*20
    for i in range(20,n): mom20.append((c[i]-c[i-20])/c[i-20]*100)
    # 量比（价格变化率）
    roc5 = [None]*5
    for i in range(5,n): roc5.append((c[i]-c[i-5])/c[i-5]*100)
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'m120':m120,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52,'mom20':mom20,'roc5':roc5}

t1=time.time()
inds={}
for code in codes:
    ind=ci(code)
    if ind: inds[code]=ind
print(f"📊 {len(inds)}只指标计算完成 耗时{time.time()-t1:.1f}s")

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

# === 评分系统1: V1原始（有MACD门）===
def _p52_score(p52):
    if p52 is None: return 0
    if p52<20: return 20
    if p52<35: return 15
    if p52<50: return 10
    if p52<65: return 6
    if p52<80: return 3
    return 0

def _adx_score(av):
    if av is None: return -5
    if av>=35: return 20
    if av>=28: return 15
    if av>=22: return 10
    if av>=18: return 5
    return -5

def _rsi_score(rv, deadly=False):
    if rv is None: return 0
    if rv<25: return 20
    if rv<35: return 14
    if rv<50: return 10
    if rv<65: return 6
    if rv<75: return 2
    if deadly and rv>=75: return -5
    return 0

def _mom20_score(mom):
    if mom is None: return 0
    if mom>15: return 20
    if mom>10: return 15
    if mom>5: return 10
    if mom>0: return 5
    if mom>-5: return 2
    return 0

def score_v1(code, di):
    ind=inds.get(code)
    if not ind: return 0
    mh=saf(ind['mh'],di); mhp=saf(ind['mh'],di-1); ms=0
    if mh and mhp:
        if mh>0 and mhp<=0: ms=20
        elif mh>0 and mh>mhp: ms=12
        elif mh>0: ms=6
    if ms<=0: return 0  # MACD门
    p52=saf(ind['p52'],di); ws=_p52_score(p52)
    pr=saf(ind['c'],di); m5=saf(ind['m5'],di); m20=saf(ind['m20'],di); m60=saf(ind['m60'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=saf(ind['adx'],di); ads=_adx_score(av)
    rv=saf(ind['rsi'],di); rs=_rsi_score(rv, deadly=True)
    tr=av is not None and av>=22
    wl=[25,15,15,25,20] if tr else[10,30,15,10,35]
    ttl=ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(ttl/sum(wl)*100, 100)

# === 评分系统2: V1放松MACD门（MACD为负但趋势好也给分）===
def _macd_score(code, di, relaxed=False):
    ind=inds.get(code)
    if not ind: return 0, False
    mh=saf(ind['mh'],di); mhp=saf(ind['mh'],di-1)
    if not mh or not mhp: return 0, False
    if mh>0 and mhp<=0: return 20, True
    if mh>0 and mh>mhp: return 12, True
    if mh>0: return 6, True
    if relaxed:
        if mh<0 and mh>mhp: return 3, True
        if mh<0 and mhp<0 and mh>mhp: return 1, True
    return 0, False

def score_v1_relaxed(code, di):
    ind=inds.get(code)
    if not ind: return 0
    ms, ok = _macd_score(code, di, relaxed=True)
    if not ok: return 0
    p52=saf(ind['p52'],di); ws=_p52_score(p52)
    pr=saf(ind['c'],di); m5=saf(ind['m5'],di); m20=saf(ind['m20'],di); m60=saf(ind['m60'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=saf(ind['adx'],di); ads=_adx_score(av)
    rv=saf(ind['rsi'],di); rs=_rsi_score(rv)
    tr=av is not None and av>=22
    wl=[25,15,15,25,20] if tr else[10,30,15,10,35]
    ttl=ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(ttl/sum(wl)*100, 100)

# === 评分系统3: V5.1均匀五分（每个factor等权，无MACD门）===
def score_v51(code, di):
    ind=inds.get(code)
    if not ind: return 0
    ms, _ = _macd_score(code, di, relaxed=True)
    p52=saf(ind['p52'],di); ws=_p52_score(p52)
    pr=saf(ind['c'],di); m5=saf(ind['m5'],di); m20=saf(ind['m20'],di); m60=saf(ind['m60'],di); m120=saf(ind['m120'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=5
    if m5 and m20 and m5>m20: mas+=5
    if m20 and m60 and m20>m60: mas+=5
    if m20 and m120 and m20>m120: mas+=5
    av=saf(ind['adx'],di)
    ads=0
    if av is not None:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
        elif av>=15: ads=2
    mom=saf(ind['mom20'],di); ms2=_mom20_score(mom)
    ttl=ms+ws+mas+ads+ms2
    return min(ttl, 100)

# === 评分系统4: 纯动量（不分factor，只用20日涨幅评分）===
def score_momentum(code, di):
    ind=inds.get(code)
    if not ind: return 0
    mom=saf(ind['mom20'],di)
    if mom is None or mom<=0: return 0
    return min(mom*2, 100)  # 20%涨幅=40分，50%涨幅=100分

# === 评分系统5: 动量+均线质量 ===
def score_mom_ma(code, di):
    ind=inds.get(code)
    if not ind: return 0
    mom=saf(ind['mom20'],di)
    if mom is None or mom<=0: return 0
    pr=saf(ind['c'],di); m20=saf(ind['m20'],di); m60=saf(ind['m60'],di)
    mas=0
    if pr and m20 and pr>m20: mas+=25
    if m5:=saf(ind['m5'],di):
        if m5 and m20 and m5>m20: mas+=25
    if m20 and m60 and m20>m60: mas+=25
    mh=saf(ind['mh'],di)
    if mh and mh>0: mas+=25
    score_val = mom + mas
    return min(max(score_val, 0), 100)

scoring_systems = {
    'v1': score_v1,
    'v1_relaxed': score_v1_relaxed,
    'v51': score_v51,
    'momentum': score_momentum,
    'mom_ma': score_mom_ma,
}

def sm(i, sector_dict):
    mom = {}
    for sec, cls in sector_dict.items():
        rets = []
        for c in cls[:20]:
            di=gi(c,adates[i]); di20=gi(c,adates[max(0,i-20)])
            if di<0 or di20<0 or c not in inds: continue
            pr=saf(inds[c]['c'],di); p20=saf(inds[c]['c'],di20)
            if pr and p20 and p20>0: rets.append((pr-p20)/p20*100)
        if len(rets)>=2: mom[sec]=sum(rets)/len(rets)
    return mom

def sim(s, e, p, sector_dict, score_fn):
    cash=1000000.0; pos={}; daily=[]
    for i in range(s, e):
        dt=adates[i]
        if (i-s)%p['rebal']==0:
            mom=sm(i, sector_dict)
            if not mom: continue
            rk=sorted(mom.items(), key=lambda x:-x[1])
            ts=[r[0] for r in rk[:p['top']]]
            hs=[r[0] for r in rk[:p['hold']]]
            for c in list(pos.keys()):
                if pos[c]['s'] not in hs:
                    di=gi(c,dt)
                    if di>=0: pr=saf(inds[c]['c'], di)
                    if di>=0 and pr and pr>0:
                        cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                    del pos[c]
            scs=defaultdict(list)
            for sec in ts:
                for c in sector_dict.get(sec, []):
                    if c in pos: continue
                    di=gi(c,dt)
                    if di<0 or c not in inds: continue
                    sc=score_fn(c,di)
                    if sc>=p['buy']:
                        pr=saf(inds[c]['c'],di)
                        if pr and pr>0: scs[sec].append((c,sc,pr))
            for sec in ts:
                scs[sec].sort(key=lambda x:-x[1])
                for c,sc,pr in scs[sec][:p['per_sec']]:
                    if len(pos)>=p['maxp']:break
                    inv=min(cash*p['pct'],cash*0.95)
                    if inv<20000: continue
                    pos[c]={'e':pr,'v':inv,'s':sec}; cash-=inv
        for c in list(pos.keys()):
            di=gi(c,dt)
            if di<0 or c not in inds: continue
            sc=score_fn(c,di)
            pr=saf(inds[c]['c'],di)
            m20=saf(inds[c]['m20'],di); mh=saf(inds[c]['mh'],di)
            if sc<p['sell'] or (pr and m20 and mh is not None and pr<m20 and mh<0):
                if pr and pr>0: cash+=pos[c]['v']*(1+(pr-pos[c]['e'])/pos[c]['e'])
                del pos[c]
        tv=cash
        for c,px in pos.items():
            di=gi(c,dt)
            if di>=0 and c in inds:
                pr=saf(inds[c]['c'],di)
                if pr and pr>0: tv+=px['v']*pr/px['e']
                else: tv+=px['v']
            else: tv+=px['v']
        daily.append(tv)
    for c,px in list(pos.items()):
        di=gi(c,adates[e-1])
        if di>=0 and c in inds:
            pr=saf(inds[c]['c'],di)
            if pr and pr>0: cash+=px['v']*(1+(pr-px['e'])/px['e'])
            else: cash+=px['v']
        else: cash+=px['v']
    ret=(cash-1000000)/1000000*100
    peak=max(daily) if daily else 1000000
    mdd=max(((peak-v)/peak*100) for v in daily) if daily else 0
    dr=[(daily[j]-daily[j-1])/daily[j-1]*100 for j in range(1,len(daily)) if daily[j-1]>0]
    sr=0
    if len(dr)>5:
        avg=sum(dr)/len(dr); var=sum((r-avg)**2 for r in dr)/len(dr); std=max(var**0.5,0.001)
        sr=round(avg/std*15.8, 2)
    return round(ret,2), round(mdd,2), sr

years=[]
for y in range(2015,2026):
    s=next((i for i,dt in enumerate(adates) if dt>=f'{y}-01-01'), None)
    e=next((i for i,dt in enumerate(adates) if dt>=f'{y+1}-01-01'), len(adates))
    if s and e and e-s>30: years.append((y,s,e))

csi={2015:5.58,2016:-11.28,2017:21.78,2018:-25.31,2019:36.07,2020:27.21,2021:-5.20,2022:-21.63,2023:-11.38,2024:14.68,2025:-3.67}

# ===== 第二轮测试配置 =====
configs = []

# 基准：最佳第一轮结果
configs.append(('REF-6只/卖53/v1排除', {'scoring':'v1','buy':62,'sell':53,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))

# K组：v1_relaxed评分 + 调参
for sell in [50, 53, 55]:
    configs.append((f'K-6只/卖{sell}/v1relaxed', {'scoring':'v1_relaxed','buy':60,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))
for sell in [48, 50, 53]:
    configs.append((f'K-8只/卖{sell}/v1relaxed', {'scoring':'v1_relaxed','buy':60,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':8,'per_sec':2,'pct':0.125}, True))

# L组：v5.1评分（等权5因子，无MACD门）
for sell in [45, 48, 50, 53]:
    configs.append((f'L-6只/卖{sell}/v51', {'scoring':'v51','buy':55,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))
for sell in [45, 48, 50]:
    configs.append((f'L-8只/卖{sell}/v51', {'scoring':'v51','buy':55,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':8,'per_sec':2,'pct':0.125}, True))
for sell in [45, 48]:
    configs.append((f'L-10只/卖{sell}/v51', {'scoring':'v51','buy':55,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':10,'per_sec':2,'pct':0.10}, True))

# M组：纯动量（只买涨得最猛的）
for sell in [30, 35, 40]:
    configs.append((f'M-6只/卖{sell}/动量', {'scoring':'momentum','buy':30,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))
for sell in [30, 35]:
    configs.append((f'M-8只/卖{sell}/动量', {'scoring':'momentum','buy':30,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':8,'per_sec':2,'pct':0.125}, True))

# N组：动量+均线质量
for sell in [40, 45, 50]:
    configs.append((f'N-6只/卖{sell}/momMA', {'scoring':'mom_ma','buy':50,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))
for sell in [40, 45]:
    configs.append((f'N-8只/卖{sell}/momMA', {'scoring':'mom_ma','buy':50,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':8,'per_sec':2,'pct':0.125}, True))

# O组：v5.1 + 全行业
for sell in [48, 50]:
    configs.append((f'O-6只/卖{sell}/v51全行', {'scoring':'v51','buy':55,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, False))

# P组：v5.1 + 高买入门槛
for sell in [48, 50]:
    configs.append((f'P-6只/卖{sell}/v51买60', {'scoring':'v51','buy':60,'sell':sell,'top':4,'hold':4,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))

# Q组：v5.1 + 5天调仓
for sell in [48, 50]:
    configs.append((f'Q-6只/卖{sell}/v51/5天', {'scoring':'v51','buy':55,'sell':sell,'top':4,'hold':4,'rebal':5,'maxp':6,'per_sec':2,'pct':0.16}, True))

# R组：v5.1 + 10天调仓
for sell in [48, 50]:
    configs.append((f'R-6只/卖{sell}/v51/10天', {'scoring':'v51','buy':55,'sell':sell,'top':4,'hold':4,'rebal':10,'maxp':6,'per_sec':2,'pct':0.16}, True))

# S组：v1_relaxed + 前6行业
for sell in [50, 53]:
    configs.append((f'S-6只/卖{sell}/v1relaxed前6', {'scoring':'v1_relaxed','buy':60,'sell':sell,'top':6,'hold':6,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))

# T组：v51 + 前6行业
for sell in [48, 50]:
    configs.append((f'T-6只/卖{sell}/v51前6', {'scoring':'v51','buy':55,'sell':sell,'top':6,'hold':6,'rebal':7,'maxp':6,'per_sec':2,'pct':0.16}, True))

results=[]
for name, p, use_excl in configs:
    sector_dict = ss_excl if use_excl else ss_all
    score_fn = scoring_systems[p['scoring']]
    t2=time.time()
    print(f"\n{'='*60}")
    print(f"📊 {name}")
    print(f"{'='*60}")
    cum=1000000; srs=[]; yearly=[]
    for y,s,e in years:
        r, mdd, sr = sim(s, e, p, sector_dict, score_fn)
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
print(f"{'配置':<42} {'累计':>8} {'年化':>6} {'夏普':>6} {'均回撤':>7}")
print(f"{'─'*42} {'─'*8} {'─'*6} {'─'*6} {'─'*7}")
for name, cr, ann, sr, dd in results:
    print(f"{name:<42} {cr:>+7.2f}% {ann:>5.2f}% {sr:>5.2f} {dd:>6.1f}%")
