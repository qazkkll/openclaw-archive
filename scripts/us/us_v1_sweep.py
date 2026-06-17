#!/usr/bin/env python3
"""
美股评分系统 V1.x 暴力穷举
2021-2025 回测, 寻找最优权重组合
"""
import yfinance as yf, json, sys, itertools, math, warnings, time
warnings.filterwarnings('ignore')
from collections import defaultdict
from datetime import datetime

print("📥 下载美股数据...")
UNIVERSE = ['NVDA','AAPL','MSFT','GOOGL','AMZN','META','TSLA','AVGO','AMD','INTC',
  'MU','QCOM','ARM','MRVL','SNPS','CDNS','ANET','CRWD','PANW','ZS',
  'TTWO','EA','PLTR','SOFI','COIN','MSTR','SMCI','WMT','LLY','UNH',
  'HD','JPM','V','MA','COST','NFLX','ADBE','CRM','UBER','ABNB',
  'KLAC','LRCX','AMAT','TXN','NXPI','ASML','TSM','MS','GS','ABBV']

hist = {}; errors = 0
for sym in UNIVERSE:
    try:
        df = yf.download(sym, start='2019-01-01', end='2026-05-16', progress=False, auto_adjust=True)
        if df is not None and len(df) > 500:
            df = df.dropna()
            dates = df.index.strftime('%Y-%m-%d').tolist()
            closes = [round(float(x),2) for x in df['Close'].values]
            highs = [round(float(x),2) for x in df['High'].values]
            lows = [round(float(x),2) for x in df['Low'].values]
            volumes = [int(x) for x in df['Volume'].values]
            hist[sym] = {'dates':dates,'close':closes,'high':highs,'low':lows,'volume':volumes}
    except:
        errors += 1

print(f"✅ {len(hist)}只, {errors}失败")
all_dates = sorted(set(d for h in hist.values() for d in h['dates'] if '2021-01-01'<=d<='2026-05-15'))
print(f"📅 {len(all_dates)}天 ({all_dates[0]}~{all_dates[-1]})")

cdates = {c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in hist if hist[c].get('dates')}
def gi(code, dt):
    cm=cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d=hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x<=dt and cdates[code].get(x) is not None: return cdates[code][x]
    return -1

print("⚙️ 计算指标...")
def calc_ind(code):
    d=hist.get(code)
    if not d: return None
    c=d.get('close',[]);h=d.get('high',[]);l=d.get('low',[]);n=len(c)
    if n<60: return None
    def sma(a,p):return[None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def ema(a,p):k=2/(p+1);r=[a[0]];[r.append(v*k+r[-1]*(1-k)) for v in a[1:]];return r
    m20=sma(c,20);m50=sma(c,50);m200=sma(c,200)
    e12=ema(c,12);e26=ema(c,26);ml=[e12[i]-e26[i] for i in range(n)]
    sg=ema(ml,9);mh=[ml[i]-sg[i] for i in range(n)]
    gl,ll=[],[]
    for i in range(1,n):diff=c[i]-c[i-1];gl.append(max(diff,0));ll.append(max(-diff,0))
    rsi=[None]*14;ag=sum(gl[:14])/14 if len(gl)>=14 else 0;al=sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i<len(gl):ag=(ag*13+gl[i])/14;al=(al*13+ll[i])/14
    # ADX 14期
    adx=[None]*27;tr_h,dp_h,dm_h=[],[],[]
    for i in range(1,n):
        tr=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
        dp=max(0,h[i]-h[i-1]);dm=max(0,l[i-1]-l[i])
        tr_h.append(tr);dp_h.append(dp);dm_h.append(dm)
        if i<14:continue
        tr14=sum(tr_h[-14:]);dp14=sum(dp_h[-14:]);dm14=sum(dm_h[-14:]);atr=tr14/14
        if atr==0:adx.append(0);continue
        dip=dp14/14/atr*100;dim=dm14/14/atr*100
        if dip+dim==0:adx.append(0);continue
        dx=abs(dip-dim)/(dip+dim)*100
        if i<27:adx.append(dx);continue
        adx.append((sum(a for a in adx[-13:] if a is not None)+dx)/14)
    while len(adx)<n:adx.append(None)
    p52=[None]*251
    for i in range(251,n):lo=min(c[i-250:i+1]);hi=max(c[i-250:i+1]);p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'c':c,'m20':m20,'m50':m50,'rsi':rsi,'mh':mh,'adx':adx,'p52':p52}

inds={}
for code in hist:
    ind=calc_ind(code)
    if ind:inds[code]=ind
print(f"  ✅ {len(inds)}只")

def saf(arr,i):
    return arr[i] if arr and 0<=i<len(arr) and arr[i] is not None else None

# ===== 通用评分函数（参数化） =====
def score_stock(code, di, params):
    """params = {macd_gate, w_macd, w_adx, w_ma, w_rsi, w_52w, buy_t, sell_t}"""
    ind = inds.get(code)
    if not ind: return 0
    
    mh = saf(ind['mh'], di); mhp = saf(ind['mh'], di-1)
    pr = saf(ind['c'], di); m20 = saf(ind['m20'], di); m50 = saf(ind['m50'], di)
    av = saf(ind['adx'], di); rv = saf(ind['rsi'], di); p52 = saf(ind['p52'], di)
    
    # MACD
    ms = 0
    if mh and mhp:
        if mh > 0 and mhp <= 0: ms = 25  # 金叉
        elif mh > 0 and mh > mhp: ms = 15  # 柱上升
        elif mh > 0: ms = 8  # 柱正
        else: ms = -3  # 柱负
    
    # MACD门
    if params.get('macd_gate', False) and (mh is None or mh <= 0):
        return 0
    
    # ADX
    ads = -5
    if av is not None:
        if av >= 35: ads = 22
        elif av >= 25: ads = 15
        elif av >= 20: ads = 8
        elif av >= 15: ads = 3
    
    # 均线
    mas = 0
    if pr and m20 and pr > m20: mas += 7
    if pr and m50 and pr > m50: mas += 7
    if m20 and m50 and m20 > m50: mas += 6
    
    # RSI
    rs = 0
    if rv is not None:
        if rv < 25: rs = 18
        elif rv < 35: rs = 14
        elif rv < 50: rs = 10
        elif rv < 65: rs = 6
        elif rv < 75: rs = 2
        else: rs = -5
    
    # 52周
    ws = 0
    if p52 is not None:
        if p52 < 20: ws = 15
        elif p52 < 35: ws = 12
        elif p52 < 50: ws = 8
        elif p52 < 65: ws = 5
        elif p52 < 80: ws = 2
    
    # 加权总分
    w = params
    total = ms*(w['w_macd']/25) + ws*(w['w_52w']/15) + mas*(w['w_ma']/20) + ads*(w['w_adx']/22) + rs*(w['w_rsi']/18)
    return min(total/100*100, 95)  # normalize to 0-95

# ===== 回测 =====
def backtest(params, warmup=200, rebal=20):
    """回测某个参数组合。返回平均超额"""
    buy_t = params.get('buy_t', 50)
    
    results = []
    for i in range(warmup, len(all_dates)-20, rebal):
        dt = all_dates[i]
        fwd = all_dates[min(i+20, len(all_dates)-1)]
        
        # 评分所有股票
        scored = []
        for code in inds:
            di = gi(code, dt)
            if di < 0: continue
            sc = score_stock(code, di, params)
            if sc >= buy_t:
                pr = saf(inds[code]['c'], di)
                if pr and pr > 0:
                    scored.append((code, sc, pr))
        
        if len(scored) < 3: continue
        
        # 选前N只
        scored.sort(key=lambda x: -x[1])
        top = scored[:5]  # 选前5
        
        # 等权买入
        total_pr = 0; fwd_pr = 0; cnt = 0
        for code, sc, pr in top:
            di_f = gi(code, fwd)
            if di_f < 0: continue
            pr_f = saf(inds[code]['c'], di_f)
            if pr_f and pr_f > 0:
                total_pr += pr
                fwd_pr += pr_f
                cnt += 1
        
        if cnt >= 3:
            ret = (fwd_pr/total_pr - 1) * 100
            results.append(ret)
    
    if len(results) < 10: return None
    
    avg = sum(results)/len(results)
    wins = sum(1 for r in results if r > 0)
    wr = wins/len(results)*100
    ann = avg * (252/20)  # 年化
    
    # 信息比
    std = math.sqrt(sum((r-avg)**2 for r in results)/len(results)) if len(results)>1 else 1
    ir = avg/std if std > 0 else 0
    
    return {'avg':round(avg,2), 'wr':round(wr,1), 'ann':round(ann,2), 'ir':round(ir,2), 'n':len(results)}

# ===== 测试版本 =====
print(f"\n🏃 跑美股参数组合...")

# 基础版（当前美股V1）
base_params = {'macd_gate':False, 'w_macd':25, 'w_adx':22, 'w_ma':20, 'w_rsi':18, 'w_52w':15, 'buy_t':50, 'sell_t':30}

# 不同组合
versions = [
    # 当前标准
    {'ver':'V1.0', 'p':{**base_params}, 'desc':'当前美股V1标准'},
    
    # 加MACD门
    {'ver':'V1.1', 'p':{**base_params, 'macd_gate':True}, 'desc':'当前+V1MACD门'},
    
    # 调整权重
    {'ver':'V1.2', 'p':{**base_params, 'w_macd':30, 'w_adx':15, 'w_ma':15, 'w_rsi':20, 'w_52w':20, 'buy_t':55, 'macd_gate':True}, 'desc':'动量偏重30%+MACD门'},
    {'ver':'V1.3', 'p':{**base_params, 'w_macd':20, 'w_adx':30, 'w_ma':15, 'w_rsi':20, 'w_52w':15, 'buy_t':55, 'macd_gate':True}, 'desc':'趋势偏重ADX30%+MACD门'},
    {'ver':'V1.4', 'p':{**base_params, 'w_macd':20, 'w_adx':15, 'w_ma':25, 'w_rsi':20, 'w_52w':20, 'buy_t':55, 'macd_gate':True}, 'desc':'均线偏重25%+MACD门'},
    {'ver':'V1.5', 'p':{**base_params, 'w_macd':20, 'w_adx':18, 'w_ma':15, 'w_rsi':25, 'w_52w':22, 'buy_t':55, 'macd_gate':True}, 'desc':'RSI偏重25%+52W22%'},
    {'ver':'V1.6', 'p':{**base_params, 'w_macd':15, 'w_adx':20, 'w_ma':15, 'w_rsi':20, 'w_52w':30, 'buy_t':50, 'macd_gate':True}, 'desc':'逆向偏重52W30%'},
    {'ver':'V1.7', 'p':{**base_params, 'w_macd':25, 'w_adx':25, 'w_ma':20, 'w_rsi':15, 'w_52w':15, 'buy_t':60, 'macd_gate':True}, 'desc':'MACD+ADX各25%+买60'},
    {'ver':'V1.8', 'p':{**base_params, 'w_macd':20, 'w_adx':20, 'w_ma':20, 'w_rsi':20, 'w_52w':20, 'buy_t':50, 'macd_gate':True}, 'desc':'均匀各20%+MACD门'},
    {'ver':'V1.9', 'p':{**base_params, 'w_macd':20, 'w_adx':20, 'w_ma':10, 'w_rsi':25, 'w_52w':25, 'buy_t':50, 'macd_gate':True}, 'desc':'RSI+52W偏重'},
    {'ver':'V1.10', 'p':{**base_params, 'w_macd':30, 'w_adx':15, 'w_ma':15, 'w_rsi':15, 'w_52w':25, 'buy_t':60, 'macd_gate':True}, 'desc':'MACD30%+52W25%+买60'},
]

results = []
for v in versions:
    sys.stdout.write(f"  {v['ver']}: {v['desc']}... ")
    sys.stdout.flush()
    r = backtest(v['p'])
    if r:
        results.append({**v, **r})
        print(f"✅ 超额{r['avg']:+.1f}% 胜率{r['wr']:.0f}% 年化{r['ann']:+.1f}% IR{r['ir']:.2f}")
    else:
        print("❌ 数据不足")

results.sort(key=lambda x: -x['ann'])
print(f"\n{'='*80}")
print(f"🏆 美股评分 V1.x 排名")
print(f"{'='*80}")
h=f"{'排名':>3s} {'版本':<8s} {'超额20天':>9s} {'年化':>8s} {'胜率':>6s} {'IR':>5s} {'观测':>5s} 说明"
print(h);print("-"*len(h))
for i,r in enumerate(results):
    print(f"{i+1:3d} {r['ver']:<8s} {r['avg']:+7.2f}% {r['ann']:+6.1f}% {r['wr']:>5.1f}% {r['ir']:>4.2f} {r['n']:>5d}  {r['desc']}")

import json as j
j.dump({'versions':results,'date':datetime.now().isoformat()},
        open('/home/admin/.openclaw/workspace/models/us_v1_sweep.json','w'),indent=2)
print(f"\n✅ 已保存")
PYEOF