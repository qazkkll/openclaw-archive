#!/usr/bin/env python3
"""美股买卖阈值全面网格搜索"""
import json, math, sys

print("📥 加载缓存...")
try:
    with open('/tmp/us_bt_data.json') as f: d = json.load(f)
    print(f"✅ {len(d['hist'])}只")
except:
    print("❌ 无缓存，先跑 us_dd_check.py")
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
    total=(ms*(15/25) + ads*(20/22) + mas*(15/20) + rs*(20/18) + ws*(30/15)) / 100 * 100
    return min(total,95)

def bt(params,warmup=450,rebal=20):
    """完整回测：买/卖分别用不同阈值"""
    buy_t=params['buy'];sell_t=params['sell']
    eq=[1000000.0]
    
    for i in range(warmup,len(adates)-20,rebal):
        dt=adates[i];fwd=adates[min(i+20,len(adates)-1)]
        
        # 1. 买入：评分>=buy_t的股票
        scored=[]
        for code in inds:
            di=gi(code,dt)
            if di<0: continue
            sc=score_stock(code,di,params)
            if sc>=buy_t:
                pr=saf(inds[code]['c'],di)
                if pr and pr>0: scored.append((code,sc,pr))
        
        if len(scored)<3:
            eq.append(eq[-1])
            continue
        
        # 2. 剔除评分低于sell_t的（宽松筛选）
        #   注意：sell_t是卖出线，买入时如果评分低于sell_t，说明已接近卖出区
        safe=[s for s in scored if s[1]>=sell_t]
        if len(safe)<3:
            eq.append(eq[-1])
            continue
        
        # 3. 选前5只等权买入
        safe.sort(key=lambda x:-x[1])
        top=safe[:5]
        
        tp=0;fp=0;cnt=0
        for code,sc,pr in top:
            di_f=gi(code,fwd)
            if di_f<0: continue
            pr_f=saf(inds[code]['c'],di_f)
            if pr_f and pr_f>0:
                tp+=pr;fp+=pr_f;cnt+=1
        
        if cnt>=3:
            eq.append(eq[-1]*(fp/tp))
        else:
            eq.append(eq[-1])
    
    if len(eq)<5: return None
    
    peak=eq[0];mdd=0
    for v in eq:
        if v>peak: peak=v
        dd=(peak-v)/peak*100
        if dd>mdd: mdd=dd
    
    cum=(eq[-1]/eq[0]-1)*100
    yr=len(eq)*rebal/252
    ann=round(((eq[-1]/eq[0])**(1/yr)-1)*100,1) if yr>0.5 else 0
    
    # 胜率：多少期是正收益
    wins=0;total_obs=len(eq)-1
    for j in range(1,len(eq)):
        if eq[j]>eq[j-1]: wins+=1
    wr=round(wins/total_obs*100,1) if total_obs>0 else 0
    
    return {'ann':ann,'mdd':round(mdd,1),'cum':round(cum,1),'wr':wr,'n':len(eq)}

# ===== 网格搜索 =====
base={'macd_gate':True,'w_m':15,'w_a':20,'w_ma':15,'w_r':20,'w_w':30}

buy_vals=list(range(30,71,5))   # 30,35,40,45,50,55,60,65,70
sell_vals=list(range(20,66,5))  # 20,25,30,35,40,45,50,55,60,65

tests=[]
for buy in buy_vals:
    for sell in sell_vals:
        if sell < buy and sell <= 55:  # 卖出必须低于买入
            tests.append({'buy':buy,'sell':sell})

total=len(tests)
print(f"\n🏃 网格搜索 {total} 个组合...")

results=[]
for idx,t in enumerate(tests):
    if (idx+1)%20==0:
        print(f"  {idx+1}/{total}")
        sys.stdout.flush()
    
    r=bt({**base,'buy':t['buy'],'sell':t['sell']})
    if r:
        cal=r['ann']/r['mdd'] if r['mdd']>0 else 0
        results.append({**t,**r,'cal':round(cal,2)})

# 按年化排序
results.sort(key=lambda x:-x['ann'])
print(f"\n{'='*80}")
print(f"📊 美股买卖阈值网格搜索 (V1.6)")
print(f"{'='*80}")
h=f"{'买':>4s}|{'卖':>4s}|{'年化':>7s}|{'累计':>9s}|{'回撤':>7s}|{'卡玛':>6s}|{'胜率':>6s}"
print(h);print("-"*len(h))
for r in results[:30]:
    print(f"{r['buy']:>4d}|{r['sell']:>4d}|{r['ann']:>+6.1f}%|{r['cum']:>+8.1f}%|{r['mdd']:>6.1f}%|{r['cal']:>5.2f}|{r['wr']:>5.1f}%")

# 按卡玛排序
results.sort(key=lambda x:-x['cal'])
print(f"\n🏆 按风险调整(卡玛) TOP 10:")
h2=f"{'买':>4s}|{'卖':>4s}|{'年化':>7s}|{'回撤':>7s}|{'卡玛':>6s}|{'胜率':>6s}"
print(h2);print("-"*len(h2))
for r in results[:10]:
    print(f"{r['buy']:>4d}|{r['sell']:>4d}|{r['ann']:>+6.1f}%|{r['mdd']:>6.1f}%|{r['cal']:>5.2f}|{r['wr']:>5.1f}%")

# 保存
json.dump({'results':results},open('/home/admin/.openclaw/workspace/models/us_threshold_sweep.json','w'))
print(f"\n✅ 已保存 ({len(results)}个组合)")
