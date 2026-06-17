#!/usr/bin/env python3
"""美股6模型对比(修正版)"""
import json, math, sys, os
from datetime import datetime

DATA_FILE = '/home/admin/.openclaw/workspace/data/us_hist_v3.json'
OUTPUT = '/home/admin/.openclaw/workspace/iteration_log/us_v3_results.json'
WARMUP = 260

# Stock lists (global)
SP500_TOP = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','BRK-B','TSLA','AVGO',
    'JPM','V','XOM','UNH','LLY','COST','PG','HD','MA','JNJ',
    'MRK','CVX','ABBV','CRM','BAC','ORCL','NFLX','KO','AMD','PEP',
    'WMT','ADBE','MCD','CSCO','ABT','TMO','GE','IBM','DHR','CAT',
    'TXN','INTU','QCOM','VZ','CMCSA','WFC','PM','NEE','RTX','SPGI',
    'LOW','MS','BA','PM','AMAT','AXP','T','UNP','HON','BKNG',
    'SYK','LMT','ISRG','PLTR','BLK','BKNG','AMGN','PANW','MDT','DE',
    'SCHW','C','MU','PGR','NOW','GILD','ADP','CB','ETN','UBER',
    'MDLZ','TMUS','EQIX','MMC','ICE','SO','CI','COF','INTC',
    'DUK','ELV','AON','ZTS','REGN','SHW','PYPL','USB','VRTX','NOC',
][:100]
SECTOR_ETFS = ['XLK','XLC','XLY','XLP','XLV','XLF','XLE','XLU','XLI','XLB','XLRE','QQQ','SPY','SHY','IEF','GLD']

with open(DATA_FILE) as f: d = json.load(f)

# Add back any missing SP500 stocks that have data
for t in SP500_TOP:
    if t not in d: SP500_TOP.remove(t)

dates = d[list(d.keys())[0]]['dates']
N = min(len(d[t]['close']) for t in d)
dates = d[list(d.keys())[0]]['dates']

print(f"数据: {len(d)}只×{N}天 ({dates[0]}~{dates[-1]})")

# ==== 指标计算 ====
def ema(a,n):
    k=2/(n+1);r=[a[0]]
    for v in a[1:]:r.append(v*k+r[-1]*(1-k))
    return r
def sms(a,n):
    return [None]*(n-1)+[sum(a[i-n+1:i+1])/n for i in range(n-1,len(a))]

def calc(c,h,l,v):
    n=len(c);m5=sms(c,5);m20=sms(c,20);m60=sms(c,60)
    gl=[];ll=[]
    for i in range(1,n):g=c[i]-c[i-1];gl.append(max(g,0));ll.append(max(-g,0))
    ra=[None]*14
    ag=sum(gl[:14])/14 if len(gl)>=14 else 0;al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        if al==0:ra.append(100)
        else:ra.append(100-100/(1+ag/al))
        if i<len(gl):ag=(ag*13+gl[i])/14;al=(al*13+ll[i])/14
    e12=ema(c,12);e26=ema(c,26)
    ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9);mh=[ml[i]-sg[i] for i in range(n)]
    p52=[None]*252
    for i in range(252,n):
        lo=min(c[i-251:i+1]);hi=max(c[i-251:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    ax=[None]*14
    for i in range(14,n):
        rg=max(c[i-13:i+1])-min(c[i-13:i+1]);s=sum(c[i-13:i+1])
        ax.append(rg/s*1400 if s>0 else 0)
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'r':ra,'mh':mh,'p':p52,'a':ax,'v':v}

ind={}
for t, h in d.items():
    ind[t]=calc(h['close'],h['high'],h['low'],h.get('volume',[0]*len(h['close'])))

def sf(arr,i):
    return arr[i] if 0<=i<len(arr) and arr[i] is not None else None

# ==== 评分函数 ====
def score_small(i,t):
    """小钳评分 0-100"""
    id=ind.get(t)
    if not id: return 0
    h=sf(id['mh'],i);hp=sf(id['mh'],i-1)
    ms=0
    if h and hp:
        if h>0 and hp<=0: ms=20
        elif h>0 and h>hp: ms=12
        elif h>0: ms=6
    p=sf(id['p'],i)
    ws=0
    if p:
        if p<20: ws=20
        elif p<35: ws=15
        elif p<50: ws=10
        elif p<65: ws=6
        elif p<80: ws=3
    pr=sf(id['c'],i);m5=sf(id['m5'],i);m20=sf(id['m20'],i);m60=sf(id['m60'],i)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=sf(id['a'],i)
    ads=-5
    if av:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
    rv=sf(id['r'],i)
    rs=0
    if rv:
        if rv<25: rs=20
        elif rv<35: rs=14
        elif rv<50: rs=10
        elif rv<65: rs=6
        elif rv<75: rs=2
        elif rv>=75: rs=-5
    if ms<=0: return 0
    wl=[25,15,15,25,20] if (av and av>=22) else [10,30,15,10,35]
    sw=sum(wl)
    total=ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(total/sw*100,100)

def quant_score(i,t):
    """Fama-French风格多因子"""
    id=ind.get(t)
    if not id: return 0
    f=score_factors(i,t)
    ms,ws,mas,ads,rs=f['ms'],f['ws'],f['mas'],f['ads'],f['rs']
    pr_n=sf(id['c'],i)
    pr_12=sf(id['c'],max(252,i-252))
    mom=((pr_n-pr_12)/pr_12*100) if (pr_n and pr_12 and pr_12>0) else -10
    # 低波
    rets=[]
    for j in range(max(260,i-20),i):
        pn=sf(id['c'],j);pp=sf(id['c'],j-1)
        if pn and pp and pp>0:rets.append(abs((pn-pp)/pp*100))
    vol=sum(rets)/len(rets) if rets else 20
    lv=max(0,20-vol)
    return mom*0.3+lv*0.2+ms*0.2+ws*0.15+rs*0.15

def score_factors(i,t):
    id=ind.get(t)
    if not id: return {'ms':0,'ws':0,'mas':0,'ads':0,'rs':0}
    h=sf(id['mh'],i);hp=sf(id['mh'],i-1)
    ms=0
    if h and hp:
        if h>0 and hp<=0: ms=20
        elif h>0 and h>hp: ms=12
        elif h>0: ms=6
    p=sf(id['p'],i)
    ws=0
    if p:
        if p<20: ws=20
        elif p<35: ws=15
        elif p<50: ws=10
        elif p<65: ws=6
        elif p<80: ws=3
    pr=sf(id['c'],i);m5=sf(id['m5'],i);m20=sf(id['m20'],i);m60=sf(id['m60'],i)
    mas=0
    if pr and m20 and pr>m20: mas+=7
    if m5 and m20 and m5>m20: mas+=7
    if m20 and m60 and m20>m60: mas+=6
    av=sf(id['a'],i)
    ads=-5
    if av:
        if av>=35: ads=20
        elif av>=28: ads=15
        elif av>=22: ads=10
        elif av>=18: ads=5
    rv=sf(id['r'],i)
    rs=0
    if rv:
        if rv<25: rs=20
        elif rv<35: rs=14
        elif rv<50: rs=10
        elif rv<65: rs=6
        elif rv<75: rs=2
        elif rv>=75: rs=-5
    return {'ms':ms,'ws':ws,'mas':mas,'ads':ads,'rs':rs}

def ma_trend_score(i):
    spy=ind.get('SPY',{}).get('c',[])
    if not spy or i<200: return 0
    ma50=sum([sf(spy,j) for j in range(i-49,i+1)])/50
    ma200=sum([sf(spy,j) for j in range(i-199,i+1)])/200
    pr=sf(spy,i)
    if pr and ma50 and ma200:
        if pr>ma50>ma200: return 100
        elif pr>ma200: return 60
    return 0

def hold_bench(s,e):
    """SPY为基准"""
    sp=ind.get('SPY',{}).get('c',[])
    a=sf(sp,s);b=sf(sp,e-1)
    return round((b-a)/a*100,2) if a and b else 0

# ==== 回测引擎 ====
def run(model, s, e, params=None):
    if params is None: params={}
    cash=1000000.0; pos={}
    
    for i in range(s, e):
        if model == 'ma_trend':
            sc=ma_trend_score(i)
            sp=ind.get('SPY',{}).get('c',[])
            p=sf(sp,i)
            if sc>=60 and not pos and p:
                pos['SPY']={'ep':p,'v':cash}; cash=0
            elif sc<40 and pos:
                cash+=pos['SPY']['v']*(1+(p-pos['SPY']['ep'])/pos['SPY']['ep']); pos={}
        
        elif model == 'sector_rotation':
            if (i-s)%params.get('rebal',10)==0:
                mom={}
                for sec in SECTOR_ETFS:
                    id=ind.get(sec)
                    if not id: continue
                    pn=sf(id['c'],i);pb=sf(id['c'],max(0,i-20))
                    if pn and pb and pb>0: mom[sec]=(pn-pb)/pb*100
                if mom:
                    rk=sorted(mom.items(),key=lambda x:-x[1])
                    t=[r[0] for r in rk[:params.get('top_n',3)]]
                    for c in list(pos.keys()):
                        if c not in [r[0] for r in rk[:5]]:
                            p=sf(ind[c]['c'],i)
                            if p: cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep']); del pos[c]
                    for sec in t:
                        if len(pos)>=params.get('max',6): break
                        sc=score_small(i,sec)
                        if sc>=62:
                            p=sf(ind[sec]['c'],i)
                            if p:
                                inv=min(200000,cash*0.2)
                                if inv>20000:
                                    pos[sec]={'ep':p,'v':inv}; cash-=inv
        
        elif model == 'dual_momentum':
            sp=ind.get('SPY',{}).get('c',[])
            qq=ind.get('QQQ',{}).get('c',[])
            if not sp or not qq: continue
            sp_r=((sf(sp,i)-sf(sp,i-252))/sf(sp,i-252)*100) if sf(sp,i-252) else -99
            qq_r=((sf(qq,i)-sf(qq,i-252))/sf(qq,i-252)*100) if sf(qq,i-252) else -99
            sp_p=sf(sp,i);qq_p=sf(qq,i)
            
            if sp_r > -15:  # 绝对动量:没大崩就持股
                if not pos:
                    target='QQQ' if qq_r>sp_r*1.05 else 'SPY'
                    p=sf(ind[target]['c'],i)
                    if p: pos[target]={'ep':p,'v':cash}; cash=0
                else:
                    cur=list(pos.keys())[0]
                    target='QQQ' if qq_r>sp_r*1.05 else 'SPY'
                    if cur!=target:
                        p=sf(ind[cur]['c'],i)
                        if p:
                            cash+=pos[cur]['v']*(1+(p-pos[cur]['ep'])/pos[cur]['ep']); pos={}
                        p2=sf(ind[target]['c'],i)
                        if p2: pos[target]={'ep':p2,'v':cash}; cash=0
            else:
                # 逃到债券
                pos={}
                ie=ind.get('IEF',{}).get('c',[])
                pi=sf(ie,i)
                if pi: pos['IEF']={'ep':pi,'v':cash}; cash=0
        
        elif model == 'hybrid':
            # 双动量门禁 → 行业轮动
            sp=ind.get('SPY',{}).get('c',[])
            if sp:
                sp_r=((sf(sp,i)-sf(sp,i-252))/sf(sp,i-252)*100) if sf(sp,i-252) else -99
                if sp_r<-15:
                    pos={}
                    ie=ind.get('IEF',{}).get('c',[])
                    pi=sf(ie,i)
                    if pi: pos['IEF']={'ep':pi,'v':cash}; cash=0
                    continue
            
            if (i-s)%params.get('rebal',10)==0:
                mom={}
                for sec in ['XLK','XLC','XLV','XLF','XLE','XLI','XLP','XLY','XLU','XLB','XLRE']:
                    id=ind.get(sec)
                    if not id: continue
                    pn=sf(id['c'],i);pb=sf(id['c'],max(0,i-20))
                    if pn and pb and pb>0: mom[sec]=(pn-pb)/pb*100
                if mom:
                    rk=sorted(mom.items(),key=lambda x:-x[1])
                    t=[r[0] for r in rk[:3]]
                    for c in list(pos.keys()):
                        if c not in [r[0] for r in rk[:5]]:
                            p=sf(ind[c]['c'],i)
                            if p: cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep']); del pos[c]
                    for sec in t:
                        if len(pos)>=6: break
                        sc=score_small(i,sec)
                        if sc>=60:
                            p=sf(ind[sec]['c'],i)
                            if p:
                                inv=min(200000,cash*0.2)
                                if inv>20000:
                                    pos[sec]={'ep':p,'v':inv}; cash-=inv
    
    # 平仓
    for c,p in list(pos.items()):
        id=ind.get(c)
        if id:
            pr=sf(id['c'],e-1)
            if pr and pr>0: cash+=p['v']*(1+(pr-p['ep'])/p['ep'])
    return round((cash-1000000)/1000000*100,2)

# ==== 跑 ====
s=WARMUP; e=N-20
print(f"回测区间: {dates[s]}~{dates[e-1]}")

h=hold_bench(s,e)
print(f"基准SPY: {h:+.2f}%\n")

models = ['sector_rotation','dual_momentum','ma_trend','hybrid']
labels = ['行业轮动(小钳)','双动量','均线趋势','混合(双动量+轮动)']
results=[]

for mdl,lb in zip(models,labels):
    r=run(mdl,s,e)
    vs=round(r-h,2)
    results.append((lb,r,vs))
    print(f"{lb:20s}: {r:+7.2f}% (vsHold:{vs:+7.2f}%)")

results.sort(key=lambda x:-x[1])
print(f"\n🏆 {results[0][0]} → {results[0][1]:+.2f}%")

# Save
os.makedirs('iteration_log',exist_ok=True)
with open(OUTPUT,'w') as f:
    json.dump({'results':[{'n':n,'r':r,'v':v} for n,r,v in results],'hold':h,'period':f'{dates[s]}~{dates[e-1]}'},f,indent=2)
print(f"\n✅ {datetime.now().strftime('%H:%M')}")
PYEOF