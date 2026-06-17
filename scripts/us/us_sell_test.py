#!/usr/bin/env python3
"""美股卖出阈值全面测试"""
import json, math, sys

try:
    with open('/tmp/us_bt_data.json') as f: d=json.load(f)
    print("✅ 缓存加载", len(d['hist']), "只")
except:
    print("❌ 无缓存, 先下载一次us_dd_check.py")
    sys.exit(1)

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

def bt(params,warmup=450,rebal=20):
    buy_t=params.get('buy_t',50)
    sell_t=params.get('sell_t',30)
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
        # 卖出门槛过滤：只保留分数>=sell_t的
        buyable=[s for s in scored if s[1]>=sell_t]
        if len(buyable)<3: eq.append(eq[-1]);continue
        tp=0;fp=0;cnt=0
        for code,sc,pr in buyable[:5]:
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

base={'macd_gate':True,'w_m':15,'w_a':20,'w_ma':15,'w_r':20,'w_w':30}

tests=[]
for buy_t in [50,55,60]:
    for sell_t in [25,30,35,40,45]:
        tests.append({'name':f'买{buy_t}/卖{sell_t}','p':{**base,'buy_t':buy_t,'sell_t':sell_t}})

print(f"🏃 {len(tests)}个组合\n")
results=[]
for t in tests:
    sys.stdout.write(f"  {t['name']}... ")
    sys.stdout.flush()
    r=bt(t['p'])
    if r:
        cal=r['ann']/r['mdd'] if r['mdd']>0 else 0
        results.append({**t,**r,'cal':round(cal,2)})
        print(f"✅ {r['ann']:+5.1f}%/年 | 回撤{r['mdd']:5.1f}% | 夏普(卡玛){cal:.2f}")
    else:
        print("❌")

# 按收益排
print(f"\n{'='*70}")
print(f"📊 美股卖出阈值测试 (15权重)")
print(f"{'='*70}")
h=f"{'版本':<14s} {'年化':>7s} {'累计':>8s} {'回撤':>7s} {'卡玛':>6s} {'次数':>5s}"
print(h);print("-"*len(h))
for r in sorted(results,key=lambda x:-x['ann']):
    print(f"{r['name']:<14s} {r['ann']:>+6.1f}% {r['cum']:>+7.1f}% {r['mdd']:>6.1f}% {r['cal']:>5.2f} {r['n']:>5d}")

# 按卡玛排
print(f"\n🏆 按风险调整收益（卡玛比）排序:")
for r in sorted(results,key=lambda x:-x['cal'])[:5]:
    print(f"  {r['name']:<14s} 年化{r['ann']:+.1f}% 回撤{r['mdd']:.1f}% 卡玛{r['cal']:.2f}")

# 当前版本
print(f"\n📌 当前: 买50/卖30 → 查表中")
cur=next((r for r in results if r['name']=='买50/卖30'),None)
if cur:
    print(f"  年化{cur['ann']:+.1f}% 回撤{cur['mdd']:.1f}% 卡玛{cur['cal']:.2f}")
