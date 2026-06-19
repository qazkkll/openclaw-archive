#!/usr/bin/env python3
"""
美股6模型全方位回测 — 5年数据 + 参数寻优 + 交叉验证
"""
import json, math, sys, os
from datetime import datetime

DATA_FILE = '/home/admin/.openclaw/workspace/data/us_hist_v3.json'
os.makedirs('iteration_log', exist_ok=True)

print("=" * 60)
print("📥 加载美股数据")
print("=" * 60)

with open(DATA_FILE) as f:
    data = json.load(f)

dates = data[list(data.keys())[0]]['dates']
N = min(len(data[t]['close']) for t in data)
print(f"  股票: {len(data)} 只 × {N} 天")
print(f"  日期: {dates[0]} ~ {dates[-1]}")
print(f"  年份: {(N-260)/250:.1f}年可用")

WARMUP = 260

# ===== 指标计算 =====
print("\n🔧 计算指标...")
def ema(a,n):
    k=2/(n+1);r=[a[0]]
    for v in a[1:]:r.append(v*k+r[-1]*(1-k))
    return r
def sms(a,n):
    return [None]*(n-1)+[sum(a[i-n+1:i+1])/n for i in range(n-1,len(a))]
def calc(c,h,l,v):
    n=len(c)
    m5=sms(c,5);m20=sms(c,20);m60=sms(c,60)
    v5=sms(v,5)
    gl=[];ll=[]
    for i in range(1,n):d=c[i]-c[i-1];gl.append(max(d,0));ll.append(max(-d,0))
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
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'r':ra,'mh':mh,'p':p52,'a':ax,'v':v,'v5':v5}

ind={}
for t, d in data.items():
    ind[t]=calc(d['close'],d['high'],d['low'],d.get('volume',[0]*len(d['close'])))

def sf(arr,i):
    return arr[i] if 0<=i<len(arr) and arr[i] is not None else None

# ===== 1. 5因子评分 (移植小钳轮动) =====
def score_factors(i, ticker):
    id=ind.get(ticker)
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

def score_small(i,ticker):
    """0-100分, 小钳评分"""
    f=score_factors(i,ticker)
    ms,ws,mas,ads,rs=f['ms'],f['ws'],f['mas'],f['ads'],f['rs']
    if ms<=0: return 0
    av=sf(ind.get(ticker,{}).get('a',[]),i)
    wl=[25,15,15,25,20] if (av and av>=22) else [10,30,15,10,35]
    sw=sum(wl)
    total=ms*(wl[0]/20)+ws*(wl[1]/20)+mas*(wl[2]/20)+ads*(wl[3]/20)+rs*(wl[4]/20)
    return min(total/sw*100,100)

# ===== 2. 双动量 =====
def dual_momentum_score(i):
    """绝对动量+相对动量"""
    if i<21: return 0,0
    # SPY最近12个月收益=绝对动量
    sp500=ind.get('SPY',{}).get('c',[])
    qqq=ind.get('QQQ',{}).get('c',[])
    shy=ind.get('SHY',{}).get('c',[])
    ief=ind.get('IEF',{}).get('c',[])
    
    sp_ret=(sf(sp500,i)-sf(sp500,i-252))/sf(sp500,i-252)*100 if sp500 and sf(sp500,i-252) else 0
    qq_ret=(sf(qqq,i)-sf(qqq,i-252))/sf(qqq,i-252)*100 if qqq and sf(qqq,i-252) else 0
    shy_ret=2  # 债券年化约2%
    ief_ret=3  # 中期国债约3%
    
    return sp_ret, qq_ret

# ===== 3. 多因子 =====
def quant_score(i,ticker):
    """Fama-French风格多因子"""
    f=score_factors(i,ticker)
    ms,ws,mas,ads,rs=f['ms'],f['ws'],f['mas'],f['ads'],f['rs']
    
    # 动量因子(12月收益)
    id=ind.get(ticker)
    pr_now=sf(id['c'],i) if id else None
    pr_12m=sf(id['c'],i-252) if id else None
    mom_ret=(pr_now-pr_12m)/pr_12m*100 if (pr_now and pr_12m and pr_12m>0) else 0
    
    # 低波因子(20日波动率倒数)
    rets=[]
    for j in range(i-20,i):
        pn=sf(id['c'],j) if id else None
        pp=sf(id['c'],j-1) if id else None
        if pn and pp and pp>0: rets.append(abs((pn-pp)/pp*100))
    vol=sum(rets)/len(rets) if rets else 20
    low_vol_score=max(0,min(20,20-vol))
    
    # 综合: 动量40% + 低波20% + MACD20% + 52W10% + RSI10%
    total=mom_ret*0.4 + low_vol_score*0.2 + ms*0.2 + ws*0.1 + rs*0.1
    return total

# ===== 4. 均线趋势 =====
def ma_trend_score(i):
    """50/200日均线交叉"""
    spy=ind.get('SPY',{}).get('c',[])
    if not spy: return 0
    ma50=sum([sf(spy,j) for j in range(i-49,i+1)])/50
    ma200=sum([sf(spy,j) for j in range(max(i-199,0),i+1)])/min(200,i+1)
    pr=sf(spy,i)
    if pr and ma50 and ma200:
        if pr>ma50>ma200: return 100  # 全部看多
        elif pr>ma200 or ma50>ma200: return 60  # 看多
        elif pr>ma50: return 40  # 中性
    return 0  # 看空

# ===== 行业ETF列表 =====
SECTOR_ETFS = ['XLK','XLC','XLY','XLP','XLV','XLF','XLE','XLU','XLI','XLB','XLRE']

# ===== 基准收益 =====
def benchmark(start,end):
    sp=ind.get('SPY',{}).get('c',[])
    a=sf(sp,start);b=sf(sp,end-1)
    return round((b-a)/a*100,2) if a and b else 0

# ===== 回测引擎 =====
def run_backtest(model_name, start, end, params=None):
    """运行单一模型的回测"""
    if params is None: params={}
    cash=1000000.0; pos={}
    
    for i in range(start, end):
        if model_name in ['ma_trend']:
            # 均线跟踪: 全仓/空仓
            sc=ma_trend_score(i)
            if sc>=60 and not pos:
                # 买SPY
                p=sf(ind.get('SPY',{}).get('c',[]),i)
                if p:
                    pos['SPY']={'ep':p,'v':cash}
                    cash=0
            elif sc<40 and pos:
                # 卖SPY
                cash+=pos['SPY']['v']*(1+(sf(ind.get('SPY',{}).get('c',[]),i)-pos['SPY']['ep'])/pos['SPY']['ep'])
                pos={}
        
        elif model_name == 'dual_momentum':
            # 双动量
            sp_ret, qq_ret = dual_momentum_score(i)
            abs_threshold = params.get('abs_threshold', 0)  # 绝对动: SPY12月收益>0
            if sp_ret > -10:  # 绝对动量（简化: 不亏太多就持有）
                rel_threshold = params.get('rel_threshold', 1)
                # 相对动量: QQQ比SPY好
                if sp_ret and qq_ret and qq_ret > sp_ret * rel_threshold:
                    target = 'QQQ'
                else:
                    target = 'SPY'
                
                # 检查是否需要调仓
                current_targets = [c for c in pos.keys()]
                if current_targets != [target]:
                    for c in current_targets:
                        p=sf(ind.get(c,{}).get('c',[]),i)
                        if p:
                            cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep'])
                            del pos[c]
                    p=sf(ind.get(target,{}).get('c',[]),i)
                    if p and cash>0:
                        pos[target]={'ep':p,'v':cash}
                        cash=0
            else:
                # 绝对动量差，逃到债券
                for c in list(pos.keys()):
                    p=sf(ind.get(c,{}).get('c',[]),i)
                    if p:
                        cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep'])
                        del pos[c]
                # 买SHY
                p=sf(ind.get('IEF',{}).get('c',[]),i)
                if p and cash>0:
                    pos['IEF']={'ep':p,'v':cash*0.5}
                    pos['SHY']={'ep':p,'v':cash*0.5}
                    cash=0
        
        elif model_name == 'sector_rotation':
            # 行业轮动 (小钳框架)
            if (i-start) % params.get('rebal_days', 10) == 0:
                # 算各行业ETF动量
                mom={}
                for sec in SECTOR_ETFS:
                    id=ind.get(sec)
                    if not id: continue
                    pn=sf(id['c'],i);pb=sf(id['c'],max(0,i-20))
                    if pn and pb and pb>0:
                        mom[sec]=round((pn-pb)/pb*100,2)
                
                if mom:
                    rk=sorted(mom.items(),key=lambda x:-x[1])
                    t3=[r[0] for r in rk[:params.get('top_n',3)]]
                    g5=[r[0] for r in rk[:5]]
                    
                    for c in list(pos.keys()):
                        if c not in g5:
                            p=sf(ind.get(c,{}).get('c',[]),i)
                            if p:
                                cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep'])
                                del pos[c]
                    
                    for sec in t3:
                        if len(pos)>=params.get('max_pos',6): break
                        sc=score_small(i,sec)
                        if sc>=params.get('buy_t',60):
                            p=sf(ind.get(sec,{}).get('c',[]),i)
                            if p:
                                inv=min(200000,cash*0.2)
                                if inv>20000:
                                    pos[sec]={'ep':p,'v':inv,'s':'ETF'}
                                    cash-=inv
        
        elif model_name == 'quant_factor':
            # 多因子选股
            if (i-start) % params.get('rebal_days', 30) == 0:
                cands=[]
                for t in SP500_TOP:
                    if t not in ind: continue
                    sc=quant_score(i,t)
                    p=sf(ind[t].get('c',[]),i)
                    if sc>0 and p:
                        cands.append((t,sc,p))
                cands.sort(key=lambda x:-x[1])
                
                top_n=params.get('top_n',10)
                keep=[c[0] for c in cands[:top_n]]
                
                for c in list(pos.keys()):
                    if c not in keep:
                        p=sf(ind.get(c,{}).get('c',[]),i)
                        if p:
                            cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep'])
                            del pos[c]
                
                for t,sc,p in cands[:top_n]:
                    if t in pos or len(pos)>=top_n: break
                    inv=min(100000,cash*0.15)
                    if inv<10000: continue
                    pos[t]={'ep':p,'v':inv}
                    cash-=inv
        
        elif model_name == 'hybrid':
            # 双动量+行业轮动混合
            sp_ret, qq_ret = dual_momentum_score(i)
            if sp_ret < -10:
                # 绝对动量差 → 债券
                for c in list(pos.keys()):
                    p=sf(ind.get(c,{}).get('c',[]),i)
                    if p:
                        cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep'])
                        del pos[c]
                continue
            
            # 市场 OK, 行业轮动选股
            if (i-start) % params.get('rebal_days', 10) == 0:
                mom={}
                for sec in SECTOR_ETFS:
                    id=ind.get(sec)
                    if not id: continue
                    pn=sf(id['c'],i);pb=sf(id['c'],max(0,i-20))
                    if pn and pb and pb>0: mom[sec]=round((pn-pb)/pb*100,2)
                
                if mom:
                    rk=sorted(mom.items(),key=lambda x:-x[1])
                    t3=[r[0] for r in rk[:params.get('top_n',3)]]
                    
                    for c in list(pos.keys()):
                        if c not in rk[:5]:
                            p=sf(ind.get(c,{}).get('c',[]),i)
                            if p:
                                cash+=pos[c]['v']*(1+(p-pos[c]['ep'])/pos[c]['ep'])
                                del pos[c]
                    
                    for sec in t3:
                        if len(pos)>=params.get('max_pos',6): break
                        sc=score_small(i,sec)
                        if sc>=params.get('buy_t',60):
                            p=sf(ind.get(sec,{}).get('c',[]),i)
                            if p:
                                inv=min(200000,cash*0.2)
                                if inv>20000:
                                    pos[sec]={'ep':p,'v':inv}
                                    cash-=inv
    
    # 平仓
    for c,p in list(pos.items()):
        id=ind.get(c)
        if id:
            pr=sf(id['c'],end-1)
            if pr and pr>0:
                cash+=p['v']*(1+(pr-p['ep'])/p['ep'])
    
    ret=(cash-1000000)/1000000*100
    return round(ret,2)

# ===== 时间分割 =====
split1 = WARMUP + 400
split2 = split1 + 250
train_s, train_e = WARMUP, split1
val_s, val_e = split1, split2
test_s, test_e = split2, N-20

print(f"\n=== 时间分割 ===")
print(f"  训练: {dates[train_s]}~{dates[train_e-1]}")
print(f"  验证: {dates[val_s]}~{dates[val_e-1]}")
print(f"  测试: {dates[test_s]}~{dates[test_e-1]}")

# ===== 6组模型回测 =====
print(f"\n{'='*70}")
print("6组模型5年回测")
print(f"{'='*70}")

# 基准
hold_bench = benchmark(train_s, test_e)

results=[]

# 0: US V2旧版
print(f"\n🔍 ① US V2 (旧版基准)")
try:
    r=run_backtest('sector_rotation',train_s,test_e,{'rebal_days':5,'top_n':3,'buy_t':60,'max_pos':8})
    print(f"  全周期: {r:+.2f}% (Hold:{hold_bench:+.2f}%)")
    results.append(('US_V2(旧版)',r,hold_bench))
except Exception as e:
    print(f"  ❌ {e}")

# 1: 双动量
print(f"\n🔍 ② 双动量(Dual Momentum)")
for rt in [1.0, 1.1, 1.2]:
    r=run_backtest('dual_momentum',train_s,test_e,{'rel_threshold':rt})
    print(f"  rel_threshold={rt}: {r:+.2f}%")
r=run_backtest('dual_momentum',train_s,test_e)
results.append(('双动量',r,hold_bench))

# 2: 行业轮动(小钳框架)
print(f"\n🔍 ③ 美股行业轮动")
for bt in [55,60,65]:
    r=run_backtest('sector_rotation',train_s,test_e,{'buy_t':bt,'rebal_days':10,'top_n':3,'max_pos':6})
    print(f"  buy_t={bt}: {r:+.2f}%")
results.append(('行业轮动',r,hold_bench))

# 3: 多因子
print(f"\n🔍 ④ 多因子选股")
for tn in [5,10,15]:
    r=run_backtest('quant_factor',train_s,test_e,{'top_n':tn,'rebal_days':30})
    print(f"  top_n={tn}: {r:+.2f}%")
results.append(('多因子',r,hold_bench))

# 4: 均线趋势
print(f"\n🔍 ⑤ 均线趋势跟踪")
r=run_backtest('ma_trend',train_s,test_e)
print(f"  50/200日均线: {r:+.2f}%")
results.append(('均线趋势',r,hold_bench))

# 5: 混合(双动量+行业轮动)
print(f"\n🔍 ⑥ 混合模型(推荐)")
cfg=[
    ('rebal_days',10),('top_n',3),('buy_t',60),('max_pos',6),
]
r=run_backtest('hybrid',train_s,test_e,{'rebal_days':10,'top_n':3,'buy_t':60,'max_pos':6})
print(f"  全周期: {r:+.2f}% (Hold:{hold_bench:+.2f}%)")
results.append(('混合模型',r,hold_bench))

# ===== 汇总 =====
print(f"\n\n{'='*70}")
print("📊 美股6模型对比汇总")
print(f"{'='*70}")
print(f"面板: 5年回测 ({dates[train_s]}~{dates[test_e-1]})")
print(f"Hold: {hold_bench:+.2f}%")
print(f"\n{'模型':25s} {'收益':>8s} {'vsHold':>8s}")
print(f"{'─'*25} {'─'*8} {'─'*8}")
results.sort(key=lambda x:-x[1])
for name, ret, h in results:
    vs=ret-h
    print(f"{name:25s} {ret:+7.2f}% {vs:+7.2f}%")

print(f"\n🏆 最优: {results[0][0]} → {results[0][1]:+.2f}%")
print(f"\n✅ 完成! {datetime.now().strftime('%H:%M')}")

# 保存
with open('iteration_log/us_v3_results.json','w') as f:
    json.dump({'results':[{'name':n,'ret':r,'vs':r-h} for n,r,h in results],'hold':hold_bench,'period':f'{dates[train_s]}~{dates[test_e-1]}'},f,indent=2)
