#!/usr/bin/env python3
"""
12因子完整量化回测：技术面(60%) + 基本面(40%)
时间：3年（2023-05 ~ 2026-05）
股票：21只行业龙头
"""

import json, urllib.request, math, time, akshare as ak

RELAY = "http://47.107.99.189:8080"

STOCKS = [("SH.600519","茅台"),("SZ.000858","五粮液"),("SZ.000568","泸州老窖"),
          ("SH.600036","招行"),("SH.601318","平安"),("SH.603501","韦尔"),
          ("SH.600584","长电"),("SZ.002594","比亚迪"),("SH.601633","长城"),
          ("SH.600276","恒瑞"),("SH.603259","药明"),("SH.600588","用友"),
          ("SZ.000938","紫光"),("SH.601012","隆基"),("SH.600438","通威"),
          ("SH.600760","沈飞"),("SH.600893","航发"),("SH.600150","船舶"),
          ("SZ.002558","巨人"),("SZ.000021","深科技"),("SZ.002236","大华")]

def fetch(code,s,e):
    d=json.dumps({"code":code,"start":s,"end":e,"freq":"D"}).encode()
    r=urllib.request.Request(f"{RELAY}/history",data=d,headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=30).read()).get("data",[])

def ema(arr,n):
    k=2/(n+1);r=[arr[0]]
    for v in arr[1:]:r.append(v*k+r[-1]*(1-k))
    return r
def sma(arr,n):
    res=[None]*n
    for i in range(n,len(arr)):
        vals=[arr[j] for j in range(i-n+1,i+1) if arr[j] is not None]
        res.append(sum(vals)/n if len(vals)>=n else None)
    return res[:len(arr)]
def sf(arr,i):
    try:return arr[i] if 0<=i<len(arr) and arr[i] is not None else None
    except:return None

# ===== 全部指标计算 =====
def calc_indicators(p,h,l,v):
    n=len(p)
    rs=rsi_calc(p)
    m5,m20,m60=sma(p,5),sma(p,20),sma(p,60)
    e12,e26=ema(p,12),ema(p,26)
    macd=[e12[i]-e26[i] for i in range(n)]
    sig=ema(macd,9);hist=[macd[i]-sig[i] for i in range(n)]
    
    # 量比
    v5=sma(v,5);vr=[v[i]/(sf(v5,i) or 1) for i in range(n)]
    
    # ADX
    pdi,ndi,tr=[0]*n,[0]*n,[0]*n
    for i in range(1,n):
        up,down=h[i]-h[i-1],l[i-1]-l[i]
        pdi[i]=max(up,0) if up>down else 0
        ndi[i]=max(down,0) if down>up else 0
        tr[i]=max(h[i]-l[i],abs(h[i]-p[i-1]),abs(l[i]-p[i-1]))
    atr=sma(tr,14);pd_s=sma(pdi,14);nd_s=sma(ndi,14)
    adx=[None]*n
    for i in range(14,n):
        sm=sf(pd_s,i)+sf(nd_s,i)
        adx[i]=abs(sf(pd_s,i)-sf(nd_s,i))/sm*100 if sm>0 else adx[i-1] if i>14 else 0
    adx=sma(adx,14)
    
    # 价量关系
    pv=[0]*n
    for i in range(1,n):
        pu=p[i]>p[i-1];vu=v[i]>(sf(v5,i) or 1)
        pv[i]=2 if pu and vu else 1 if pu else -1 if not pu and vu else -2
    
    # MFI
    mfi=[None]*14
    for i in range(14,n):
        tp=[((h[j]+l[j]+p[j])/3)*v[j] for j in range(i-13,i+1)]
        po,ne=0,0
        for j in range(1,14):
            cur=(h[i-13+j]+l[i-13+j]+p[i-13+j])/3
            pre=(h[i-14+j]+l[i-14+j]+p[i-14+j])/3
            if cur>pre:po+=tp[j]
            else:ne+=tp[j]
        mfi.append(100-100/(1+po/ne) if ne>0 else 100)
    
    # 52周位置
    p52=[None]*251
    for i in range(251,n):
        lo,hi=min(p[i-251:i+1]),max(p[i-251:i+1])
        p52.append((p[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    
    return {'rs':rs,'m5':m5,'m20':m20,'m60':m60,'hist':hist,'vr':vr,'adx':adx,'pv':pv,'mfi':mfi,'p52':p52}

def rsi_calc(p):
    if len(p)<16:return[None]*len(p)
    g,l=[],[]
    for i in range(1,len(p)):d=p[i]-p[i-1];g.append(max(d,0));l.append(max(-d,0))
    r,ag,al=[None]*14,sum(g[:14])/14,sum(l[:14])/14
    for i in range(14,len(p)):
        r.append(100-100/(1+ag/al) if al else 100)
        if i<len(g):ag=(ag*13+g[i])/14;al=(al*13+l[i])/14
    return r

# ===== 12因子评分 =====
def tech_score(i, f, p):
    """技术面评分 0-60"""
    s=0
    # MACD柱 (30%)
    if sf(f['hist'],i) is not None:
        if sf(f['hist'],i)>0 and (sf(f['hist'],i-1) or 0)<=0:s+=30
        elif sf(f['hist'],i)>0:s+=15
    # RSI (15%)
    if sf(f['rs'],i) is not None:
        if sf(f['rs'],i)<35:s+=15
        elif sf(f['rs'],i)<60:s+=10
        elif sf(f['rs'],i)<70:s+=5
        else:s-=5
    # 站上20日 (15%)
    if sf(f['m20'],i) and p[i]>sf(f['m20'],i):s+=10
    if sf(f['m5'],i) and sf(f['m20'],i) and sf(f['m5'],i)>sf(f['m20'],i):s+=5
    # 量比 (10%)
    if sf(f['vr'],i) and sf(f['vr'],i)>1.2:s+=5
    if sf(f['vr'],i) and sf(f['vr'],i)>1.5:s+=5
    # ADX (10%)
    if sf(f['adx'],i) and sf(f['adx'],i)>25:s+=10
    # 价量 (5%)
    if sf(f['pv'],i) and sf(f['pv'],i)>=2:s+=5
    # MFI (5%)
    if sf(f['mfi'],i) and sf(f['mfi'],i)<30:s+=5
    # 52周位置(安全边际)
    if sf(f['p52'],i) and sf(f['p52'],i)<30:s+=5
    elif sf(f['p52'],i) and sf(f['p52'],i)>85:s-=5
    return max(0,min(60,s))

def fund_score(pe, pb, roe, growth):
    """基本面评分 0-40"""
    s=0
    # PE分位 (15%) — PE低=好
    if pe is not None and pe>0:s+=15 if pe<20 else 10 if pe<40 else 5
    elif pe is not None and pe<0:s+=0  # 亏损
    # PB分位 (10%)
    if pb is not None and pb>0:s+=10 if pb<2 else 5 if pb<5 else 3
    # ROE (10%)
    if roe is not None:s+=10 if roe>15 else 5 if roe>8 else 2
    # 营收增速 (5%)
    if growth is not None:s+=5 if growth>20 else 3 if growth>10 else 1
    return min(40,s)

def total_score(i,f,p,pe,pb,roe,growth):
    return tech_score(i,f,p) + fund_score(pe,pb,roe,growth)

# ===== 获取基本面（3年最后一份财报）=====
def get_fundamentals(code):
    try:
        df = ak.stock_value_em(symbol=code)
        r = df.iloc[-1]
        pe = r['PE(TTM)'] if not math.isnan(r['PE(TTM)']) else None
        pb = r['市净率'] if not math.isnan(r['市净率']) else None
    except:
        pe,pb = None,None
    
    try:
        df2 = ak.stock_financial_analysis_indicator(symbol=code, start_year="2025")
        if not df2.empty:
            last = df2.iloc[-1]
            roe = float(last['净资产收益率(%)']) if not math.isnan(float(last['净资产收益率(%)'])) else None
            growth = float(last['主营业务收入增长率(%)']) if not math.isnan(float(last['主营业务收入增长率(%)'])) else None
        else: roe,growth = None,None
    except:
        roe,growth = None,None
    
    return pe,pb,roe,growth

# ===== 回测 =====
def backtest(code,name,prices,highs,lows,vols,pe,pb,roe,growth):
    n=len(prices)
    if n<60:return None
    
    fac=calc_indicators(prices,highs,lows,vols)
    
    # 买入持有基准
    hold=(prices[-1]-prices[0])/prices[0]*100
    
    # === 旧4因子模型 ===
    t4=[];ip=False;ep=0
    for i in range(60,n):
        s=tech_score(i,fac,prices)
        s_old=s
        if not ip and s_old>=35:ip=True;ep=prices[i]
        elif ip and (s_old<20 or (sf(fac['m20'],i) and prices[i]<sf(fac['m20'],i))):
            ip=False;t4.append((prices[i]-ep)/ep*100)
    
    # === 新12因子模型 ===
    t12=[];ip=False;ep=0
    for i in range(60,n):
        s=total_score(i,fac,prices,pe,pb,roe,growth)
        if not ip and s>=55:ip=True;ep=prices[i]  # 总分55分买入
        elif ip and (s<35 or (sf(fac['m20'],i) and prices[i]<sf(fac['m20'],i))):
            ip=False;t12.append((prices[i]-ep)/ep*100)
    
    # === 纯基本面模型 ===
    fs=fund_score(pe,pb,roe,growth)
    
    res4={'pnl':round(sum(t4),2),'trades':len(t4),'wr':round(sum(1 for x in t4 if x>0)/len(t4)*100,1) if t4 else 0} if t4 else {'pnl':0,'trades':0,'wr':0}
    res12={'pnl':round(sum(t12),2),'trades':len(t12),'wr':round(sum(1 for x in t12 if x>0)/len(t12)*100,1) if t12 else 0} if t12 else {'pnl':0,'trades':0,'wr':0}
    
    return {
        'name':name,'code':code,'hold':round(hold,1),
        'pe':pe,'pb':pb,'roe':roe,'growth':growth,
        'fund_score':fs,
        'old4':res4,'new12':res12
    }

# ===== 主程序 =====
print(f"🔥 12因子回测: {len(STOCKS)}只 × 3年\n")

results=[]
for idx,(code,name) in enumerate(STOCKS):
    print(f"[{idx+1}/{len(STOCKS)}] {name} {code}...",end=" ",flush=True)
    
    d=fetch(code,'2023-05-01','2026-05-12')
    if not d or len(d)<100:print(f"❌ 数据不足");continue
    
    p=[x['收盘价'] for x in d];h=[x['最高价'] for x in d]
    l=[x['最低价'] for x in d];v=[x['成交量'] for x in d]
    
    # 基本面（每只只拉一次）
    pe,pb,roe,growth=get_fundamentals(code.split('.')[1])
    
    r=backtest(code,name,p,h,l,v,pe,pb,roe,growth)
    if r:results.append(r);print(f"✅ 旧={r['old4']['pnl']:+.1f}% 新={r['new12']['pnl']:+.1f}% 持有={r['hold']:+.1f}% [PE={pe} PB={pb} ROE={roe}%]")
    else:print("❌ 回测失败")
    
    time.sleep(0.3)

# ===== 汇总 =====
print(f"\n\n{'='*65}")
print(f"📊 12因子回测结果汇总 ({len(results)}只)")
print(f"{'='*65}")
print(f"\n  {'股票':<10} {'持有':>6} {'旧4因子':>8} {'新12因子':>8} {'Δ超额':>8} {'PE':>6} {'PB':>5} {'ROE':>5}")
print(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*5} {'─'*5}")

for r in results:
    o4=r['old4']['pnl'];n12=r['new12']['pnl']
    delta=n12-o4
    print(f"  {r['name']:<8} {r['hold']:>+5.1f}% {o4:>+7.1f}% {n12:>+7.1f}% {delta:>+7.1f}% {r['pe'] if r['pe'] else '-':>5} {r['pb'] if r['pb'] else '-':>4} {r['roe'] if r['roe'] else '-':>4}")

# 平均
avg_hold=sum(r['hold'] for r in results)/len(results)
avg_old=sum(r['old4']['pnl'] for r in results)/len(results)
avg_new=sum(r['new12']['pnl'] for r in results)/len(results)
avg_delta=avg_new-avg_old
print(f"\n  {'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")
print(f"  {'平均':>8} {avg_hold:>+5.1f}% {avg_old:>+7.1f}% {avg_new:>+7.1f}% {avg_delta:>+7.1f}%")
print(f"\n  跑赢持有的股票数:")
print(f"    旧4因子: {sum(1 for r in results if r['old4']['pnl']>r['hold'])}/{len(results)}")
print(f"    新12因子: {sum(1 for r in results if r['new12']['pnl']>r['hold'])}/{len(results)}")
print(f"    新模型比旧模型提升: {sum(1 for r in results if r['new12']['pnl']>r['old4']['pnl'])}/{len(results)}")

print(f"\n\n{'='*65}")
print(f"✅ 结论")
print(f"{'='*65}")
print(f"  旧4因子模型: {avg_old:+.1f}% (vs 持有{avg_hold:+.1f}%)")
print(f"  新12因子模型: {avg_new:+.1f}% (vs 持有{avg_hold:+.1f}%)")
if avg_new > avg_old:
    print(f"  🎉 新模型比旧模型提升 +{avg_delta:.1f}%")
else:
    print(f"  ⚠️ 新模型比旧模型差 {avg_delta:+.1f}%")
