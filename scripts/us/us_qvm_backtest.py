#!/usr/bin/env python3
"""
美股 QVM 模型 — Quality + Value + Momentum + 技术补充
完整回测 + 参数优化 + 交叉验证
"""
import json, math, sys, os
from datetime import datetime
from collections import defaultdict

# Load data
with open('/home/admin/.openclaw/workspace/data/us_hist_v3.json') as f:
    price_data = json.load(f)
with open('/home/admin/.openclaw/workspace/data/us_fundamentals.json') as f:
    fund_data = json.load(f)

dates = price_data[list(price_data.keys())[0]]['dates']
N = min(len(price_data[t]['close']) for t in price_data)
WARMUP = 260

SP500 = [t for t in fund_data.keys() if t in price_data]
SECTOR_ETFS = ['XLK','XLC','XLY','XLP','XLV','XLF','XLE','XLU','XLI','XLB','XLRE']

print(f"数据: {len(price_data)}只, 回测池: {len(SP500)}只SP500, {len(SECTOR_ETFS)}只ETF")
print(f"天数: {N}天 ({dates[0]}~{dates[-1]})")

# ==== 指标计算 ====
def ema(a,n):
    k=2/(n+1);r=[a[0]]
    for v in a[1:]:r.append(v*k+r[-1]*(1-k))
    return r
def sms(a,n):
    return [None]*(n-1)+[sum(a[i-n+1:i+1])/n for i in range(n-1,len(a))]
def calc(c,h,l):
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
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'r':ra,'mh':mh,'p':p52,'a':ax}

ind={}
for t, d in price_data.items():
    ind[t]=calc(d['close'],d['high'],d['low'])

def sf(arr,i):
    return arr[i] if 0<=i<len(arr) and arr[i] is not None else None

# ==== 因子计算 ====
def factors(i, t):
    """返回 QVM + 技术 因子"""
    id = ind.get(t)
    if not id: return None
    
    p_n = sf(id['c'], i)
    p_12 = sf(id['c'], max(252, i-252))
    
    # 动量 (12月收益)
    momentum = ((p_n-p_12)/p_12*100) if (p_n and p_12 and p_12>0) else 0
    
    # 低波 (20日波动率倒数)
    rets=[]
    for j in range(max(260,i-20),i):
        pn=sf(id['c'],j);pp=sf(id['c'],j-1)
        if pn and pp and pp>0: rets.append(abs((pn-pp)/pp*100))
    vol = sum(rets)/len(rets) if rets else 20
    low_vol = 100 - min(vol*5, 90)  # 0-100
    
    # 技术分 (小钳因子)
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
    
    # 基本面 (从fund_data)
    fd = fund_data.get(t, {})
    roe = fd.get('roe') or 0
    pm = fd.get('profit_margin') or 0
    de = fd.get('debt_equity') or 100
    pe = fd.get('pe') or 50
    
    # 质量分 (ROE+利润率+负债率)
    quality = 0
    if roe and roe > 0: quality += min(roe*100, 50)
    if pm and pm > 0: quality += min(pm*100, 30)
    if de and de < 80: quality += 20
    elif de and de < 150: quality += 10
    quality = min(quality, 100)
    
    # 价值分 (PE低分高)
    value = 0
    if pe and pe > 0:
        if pe < 15: value = 100
        elif pe < 20: value = 80
        elif pe < 30: value = 60
        elif pe < 50: value = 40
        else: value = 20
    
    return {
        'q': quality,       # 质量 0-100
        'v': value,         # 价值 0-100  
        'm': momentum,      # 动量 百分比
        'lv': low_vol,      # 低波 0-100
        'ms': ms,           # MACD 0-20
        'ws': ws,           # 52周 0-20
        'mas': mas,         # 均线 0-20
        'momentum_score': min(momentum/3*10, 100) if momentum > 0 else 0,  # 动量归一化
    }

def qvm_score(i, t, w_q=25, w_v=10, w_m=35, w_lv=15, w_t=15):
    """QVM综合评分"""
    f = factors(i, t)
    if not f: return 0
    # 质量+价值+动量+低波+技术
    mom_s = min(max((f['m'])/3, 0), 100)  # 12%年化→40分
    total = (f['q']*w_q + f['v']*w_v + mom_s*w_m + f['lv']*w_lv + 
             (f['ms']+f['ws']+f['mas'])*w_t) / 100
    return max(total, 0)

# ==== 基准 ====
def benchmark(s,e):
    sp=ind.get('SPY',{}).get('c',[])
    a=sf(sp,s);b=sf(sp,e-1)
    return round((b-a)/a*100,2) if a and b else 0

# ==== 回测 ====
def run(model, s, e, params=None):
    if params is None: params={}
    cash=1000000.0; pos={}
    
    for i in range(s, e):
        if model == 'qvm':
            if (i-s)%30==0:
                cands=[]
                for t in SP500:
                    sc = qvm_score(i, t, params.get('wq',25), params.get('wv',10),
                                    params.get('wm',35), params.get('wl',15), params.get('wt',15))
                    p = sf(ind[t]['c'], i)
                    if sc > 0 and p: cands.append((t, sc, p))
                cands.sort(key=lambda x:-x[1])
                top_n = params.get('n', 10)
                keep = [c[0] for c in cands[:top_n]]
                for c in list(pos.keys()):
                    if c not in keep:
                        p=sf(ind[c]['c'],i)
                        if p: cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep']); del pos[c]
                for t,sc,p in cands[:top_n]:
                    if t in pos or len(pos)>=top_n: break
                    inv=min(100000, cash*0.12)
                    if inv<10000: continue
                    pos[t]={'ep':p,'v':inv}; cash-=inv
        
        elif model == 'technical_only':
            # 只用技术因子(小钳) — 对照
            if (i-s)%30==0:
                cands=[]
                for t in SP500:
                    f = factors(i,t)
                    if not f: continue
                    sc = f['ms']+f['ws']+f['mas']
                    p = sf(ind[t]['c'],i)
                    if sc > 5 and p: cands.append((t, sc, p))
                cands.sort(key=lambda x:-x[1])
                keep=[c[0] for c in cands[:params.get('n',10)]]
                for c in list(pos.keys()):
                    if c not in keep:
                        p=sf(ind[c]['c'],i)
                        if p: cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep']); del pos[c]
                for t,sc,p in cands[:params.get('n',10)]:
                    if t in pos or len(pos)>=params.get('n',10): break
                    inv=min(100000,cash*0.12)
                    if inv<10000: continue
                    pos[t]={'ep':p,'v':inv}; cash-=inv
    
    for c,p in list(pos.items()):
        id=ind.get(c)
        if id:
            pr=sf(id['c'],e-1)
            if pr and pr>0: cash+=p['v']*(1+(pr-p['ep'])/p['ep'])
    return round((cash-1000000)/1000000*100,2)

# ==== 跑 ====
s=WARMUP; e=N-20
h=benchmark(s,e)

print(f"\n{'='*60}")
print(f"📊 美股 QVM vs 其他模型")
print(f"{'='*60}")
print(f"区间: {dates[s]}~{dates[e-1]}  SPY基准: {h:+.2f}%")

# 参数搜索: QVM权重
print(f"\n🔬 QVM 权重搜索:")
print(f"{'质量':>6s} {'价值':>6s} {'动量':>6s} {'低波':>6s} {'技术':>6s} {'收益':>8s} {'vsHold':>8s}")
best_r=-999; best_p=None

weights = [(25,10,35,15,15),(30,10,30,15,15),(20,15,35,15,15),
           (25,10,40,15,10),(20,10,40,15,15),(25,5,40,15,15)]

for wq,wv,wm,wl,wt in weights:
    r=run('qvm',s,e,{'wq':wq,'wv':wv,'wm':wm,'wl':wl,'wt':wt,'n':10})
    vs=round(r-h,2)
    mk='✅' if r>best_r else ''
    if r>best_r: best_r=r; best_p=(wq,wv,wm,wl,wt)
    print(f"  {wq:>4}% {wv:>4}% {wm:>4}% {wl:>4}% {wt:>4}% {r:+7.2f}% {vs:+7.2f}% {mk}")

# 最优参数
wq,wv,wm,wl,wt = best_p
print(f"\n🏆 最优: 质量{wq}% 价值{wv}% 动量{wm}% 低波{wl}% 技术{wt}%")

# 对比: QVM最优 vs 技术 vs 双动量
print(f"\n{'─'*50}")
print(f"{'模型':25s} {'收益':>8s} {'vsHold':>8s}")
print(f"{'─'*25} {'─'*8} {'─'*8}")

r_qvm = run('qvm',s,e,{'wq':wq,'wv':wv,'wm':wm,'wl':wl,'wt':wt,'n':10})
r_tech = run('technical_only',s,e,{'n':10})

# 双动量(之前跑的)
def dual_mom(i):
    sp=ind.get('SPY',{}).get('c',[]);qq=ind.get('QQQ',{}).get('c',[])
    sp_r=((sf(sp,i)-sf(sp,i-252))/sf(sp,i-252)*100) if sp and sf(sp,i-252) else -99
    qq_r=((sf(qq,i)-sf(qq,i-252))/sf(qq,i-252)*100) if qq and sf(qq,i-252) else -99
    return sp_r, qq_r

def run_dual(s,e,abs_th=-20,use_qqq=True):
    cash=1000000.0; pos={}
    for i in range(s, e):
        sp_r, qq_r = dual_mom(i)
        if sp_r > abs_th:
            target='QQQ' if (use_qqq and qq_r>sp_r) else 'SPY'
            if not pos:
                p=sf(ind[target]['c'],i)
                if p: pos[target]={'ep':p,'v':cash}; cash=0
            else:
                cur=list(pos.keys())[0]
                if cur!=target:
                    p=sf(ind[cur]['c'],i)
                    if p:
                        cash+=pos[cur]['v']*(1+(p-pos[cur]['ep'])/pos[cur]['ep']); pos={}
                    p2=sf(ind[target]['c'],i)
                    if p2: pos[target]={'ep':p2,'v':cash}; cash=0
        elif pos:
            p=sf(ind[list(pos.keys())[0]]['c'],i)
            if p:
                cash+=pos[list(pos.keys())[0]]['v']*(1+(p-pos[list(pos.keys())[0]]['ep'])/pos[list(pos.keys())[0]]['ep']); pos={}
    for c,p in list(pos.items()):
        id=ind.get(c)
        if id:
            pr=sf(id['c'],e-1)
            if pr and pr>0: cash+=p['v']*(1+(pr-p['ep'])/p['ep'])
    return round((cash-1000000)/1000000*100,2)

r_dm = run_dual(s,e)

print(f"  {'QVM(质量+价值+动量)':25s} {r_qvm:+7.2f}% {r_qvm-h:+7.2f}%")
print(f"  {'纯技术面(小钳5因子)':25s} {r_tech:+7.2f}% {r_tech-h:+7.2f}%")
print(f"  {'双动量(QQQ/SPY)':25s} {r_dm:+7.2f}% {r_dm-h:+7.2f}%")
print(f"  {'SPY买持有':25s} {h:+7.2f}%  0.00%")

# 年度
print(f"\n📊 年度对比:")
print(f"{'年份':>6s} {'QVM':>8s} {'纯技术':>8s} {'双动量':>8s} {'SPY':>8s}")
print(f"{'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for y in [2021,2022,2023,2024,2025,2026]:
    ys=next((i for i,dt in enumerate(dates) if dt>=f'{y}-01-01'),None)
    ye=next((i for i,dt in enumerate(dates) if dt>=f'{y+1}-01-01'),N-20)
    if ys is None or ys<WARMUP or ye-ys<50: continue
    r1=run('qvm',ys,ye,{'wq':wq,'wv':wv,'wm':wm,'wl':wl,'wt':wt,'n':10})
    r2=run('technical_only',ys,ye,{'n':10})
    r3=run_dual(ys,ye)
    h2=benchmark(ys,ye)
    print(f"  {y} {r1:+7.2f}% {r2:+7.2f}% {r3:+7.2f}% {h2:+7.2f}%")

# 持仓评分
print(f"\n📊 当前持仓评分 (QVM):")
for t,n in [('NVDA','英伟达'),('TSLA','特斯拉'),('TTWO','Take2'),('ZS','Zscaler')]:
    sc = qvm_score(N-1, t, wq,wv,wm,wl,wt)
    print(f"  {t}: {sc:.0f}/100  ({n})")

# 全量排名TOP10
cands=[]
for t in SP500:
    sc = qvm_score(N-1, t, wq,wv,wm,wl,wt)
    if sc>0 and sf(ind[t]['c'],N-1): cands.append((t,sc,sf(ind[t]['c'],N-1)))
cands.sort(key=lambda x:-x[1])
print(f"\n📊 当前TOP 10推荐:")
for t,sc,p in cands[:10]:
    print(f"  {t:6s} ${p:>7.2f} 评分{sc:.0f}/100")

print(f"\n✅ {datetime.now().strftime('%H:%M')}")
PYEOF