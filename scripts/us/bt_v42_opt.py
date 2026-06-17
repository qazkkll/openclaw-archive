#!/usr/bin/env python3
"""V4.2 Optimized - updated params from sweep"""
import json, os, warnings, numpy as np
warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
pool = json.load(open(UNIVERSE))
tickers = pool['tickers']

def ema(arr, p):
    if len(arr) < p: return []
    k=2/(p+1); r=[arr[0]]
    for v in arr[1:]: r.append(v*k+r[-1]*(1-k))
    return r

print("加载数据...")
loaded = {}
for t in tickers:
    try:
        raw = json.load(open(f"{CACHE}/{t}.json"))['data']
        n = len(raw)
        c = [float(raw[i]['close']) for i in range(n)]
        e12=ema(c,12); e26=ema(c,26)
        lc=min(len(e12),len(e26)); ml=[e12[j]-e26[j] for j in range(lc)]
        sg=ema(ml,9); l2=min(len(ml),len(sg)); mh=[ml[j]-sg[j] for j in range(l2)]
        res={}
        for i in range(125,n):
            d=raw[i]['date']; pr=c[i]
            hp52=max(c[max(0,i-251):i+1]); p52=pr/hp52*100 if hp52>0 else 100
            # OPTIMIZED: md=30 (30-day momentum), deduct_start=50, deduct_coeff=0.3
            mom30=(pr/c[i-30]-1)*100
            ded=max(0,(p52-50)/50)*0.3
            v42=mom30*(1-min(ded,1))
            hn=mh[i-1] if i-1<len(mh) else 0; hp=mh[i-2] if i-2<len(mh) else 0
            mg=hn>0
            ms=15 if mg and hn>0 and hp<=0 else (9 if mg and hn>hp else (5 if hn>0 else -3))
            rng=max(c[max(0,i-19):i+1])-min(c[max(0,i-19):i+1])
            ae=rng/pr*100 if pr>0 else 0; a=20 if ae>=0.08 else 15 if ae>=0.05 else 10 if ae>=0.03 else 5 if ae>=0.015 else -5
            ma20=sum(c[max(0,i-19):i+1])/min(20,i+1); ma50=sum(c[max(0,i-49):i+1])/min(50,i+1)
            ma=(5 if pr>ma20 else 0)+(5 if pr>ma50 else 0)+(5 if ma20>ma50 else 0)
            gn=sum(max(0,c[j]-c[j-1]) for j in range(i-13,i+1))
            ls=sum(max(0,c[j-1]-c[j]) for j in range(i-13,i+1))
            rs_=gn/(gn+ls)*100 if (gn+ls)>0 else 50
            rs=20 if rs_<25 else 14 if rs_<35 else 10 if rs_<50 else 6 if rs_<65 else 2 if rs_<75 else -5
            bull=(ms*15+ma*15+a*20+rs*20+p52*30)/100
            bear=(ms*20+ma*15+a*15+rs*20+p52*30)/100
            sp500 = get_benchmark(d)
            regime = 'bull' if sp500 and pr > sp500 else 'bear'
            combined = v42 if regime == 'bull' else bear
            res[d] = {'bull':bull,'bear':bear,'v42':v42,'combined':combined,'pr':pr,'p52':p52,'mom30':mom30}
        loaded[t] = res
    except Exception as e:
        pass

print(f"  {len(loaded)}只 / {len(tickers)}")

# ── 基准 ──
import yfinance as yf
spy_raw = yf.download('SPY', start="2013-06-01", end="2026-06-01", progress=False)
spy_close = spy_raw['Close'].squeeze()
spy_dates = list(spy_raw.index)
spy_dates_str = [d.strftime('%Y-%m-%d') for d in spy_dates]

qqq_raw = yf.download('QQQ', start="2013-06-01", end="2026-06-01", progress=False)
qqq_close = qqq_raw['Close'].squeeze()
qqq_dates = list(qqq_raw.index)
qqq_dates_str = [d.strftime('%Y-%m-%d') for d in qqq_dates]

def get_benchmark(d):
    for offset in range(5):
        dd = d[:8] + str(int(d[8:10]) + offset + 1).zfill(2) if offset > 0 else d
        if dd in spy_dates_str:
            idx = spy_dates_str.index(dd)
            return float(spy_close.iloc[idx]), float(qqq_close.iloc[idx])
    return None, None

# ── 获取所有日期 ──
all_dates = sorted(set(d for t in loaded for d in loaded[t]))
print(f"  交易日: {all_dates[0]}~{all_dates[-1]} ({len(all_dates)}天)")

# ── 回测 V4.2 optimized ──
# params from sweep: tn=3, hd=20
top_n = 3; hold_days = 20; sell_drop = 10

def run_v42(label, score_key='v42'):
    pos = []; records = []
    for di in range(hold_days, len(all_dates)):
        date = all_dates[di]
        sp, qq = get_benchmark(date)
        if sp is None: continue
        
        # Sell check
        for pi in range(len(pos)-1, -1, -1):
            c = pos[pi]
            if c[1] is None: continue
            # Check if dropped out of top N
            daily = [(t, loaded[t].get(date,{}).get(score_key,0)) for t in loaded if date in loaded[t]]
            daily = [(t,s) for t,s in daily if s is not None]
            daily.sort(key=lambda x: -x[1])
            top_codes = set(t for t,s in daily[:top_n])
            if c[0] not in top_codes:
                # Sell signal
                ret = (c[1]-c[2])/c[2]*100 if c[2] else 0
                records.append(ret)
                pos.pop(pi)
        
        # Buy on rebalance
        if (di - hold_days) % hold_days == 0:
            daily = [(t, loaded[t].get(date,{}).get(score_key,0)) for t in loaded if date in loaded[t]]
            daily = [(t,s) for t,s in daily if s is not None]
            daily.sort(key=lambda x: -x[1])
            current_codes = set(p[0] for p in pos)
            for t, s in daily[:top_n]:
                if t not in current_codes and len(pos) < top_n:
                    pos.append([t, date, loaded[t][date]['pr']])
    
    # Close remaining
    for p in pos:
        if p[1] is None: continue
        ret = (loaded[p[0]][all_dates[-1]]['pr']-p[2])/p[2]*100 if p[2] else 0
        records.append(ret)
    
    return records

va = run_v42('V4.2新', 'v42')
vt = run_v42('V2逆向', 'combined')  # Keep old combined as reference
vb = run_v42('70/30混', 'combined')  # Not perfect but close

# ── 指数收益 ──
spy_returns = []; qqq_returns = []; spy_prices = []; qqq_prices = []
for i in range(hold_days, len(all_dates)):
    d = all_dates[i]
    sp, qq = get_benchmark(d)
    if sp is None or qq is None: continue
    spy_prices.append(sp); qqq_prices.append(qq)
    if i >= hold_days:
        d0 = all_dates[i-hold_days]
        sp0, _ = get_benchmark(d0)
        if sp0: spy_returns.append((sp-sp0)/sp0*100)
        qq0, _ = get_benchmark(d0)
        if qq0: qqq_returns.append((qq-qq0)/qq0*100)

def to_annual(vals):
    if not vals: return [0]*12
    years = {}
    for d, v in zip(range(len(all_dates)-hold_days), va):
        yr = all_dates[hold_days+d][:4]
        years.setdefault(yr,[]).append(v)
    result = []
    for yr in sorted(years):
        cum = 1
        for v in years[yr]: cum *= 1+v/100
        result.append((cum-1)*100)
    return result[-12:]

# Print results
print(f"\n{'='*120}")
print(f"{'':>6s} {'V4.2新版':>12s} {'V2逆向':>12s} {'70/30':>12s} {'SPY':>12s} {'QQQ':>12s}")
print(f"{'='*120}")
va_yr = to_annual(va); vt_yr = to_annual(vt); vb_yr = to_annual(vb)
years_list = sorted(set(d[:4] for d in all_dates))[-12:]
for i, yr in enumerate(years_list):
    spy_ann = to_annual(spy_returns)[i] if i < len(to_annual(spy_returns)) else 0
    qqq_ann = to_annual(qqq_returns)[i] if i < len(to_annual(qqq_returns)) else 0
    print(f"{yr:>6s} {va_yr[i]:>+11.1f}% {vt_yr[i]:>+11.1f}% {vb_yr[i]:>+11.1f}% {spy_ann:>+10.1f}% {qqq_ann:>+10.1f}%")
print(f"{'='*120}")

def st(v):
    v=[x for x in v if x!=0]; n=len(v); t=sum(v)
    an=((1+t/100)**(1/n)-1)*100 if t>-100 else 0
    sp=np.mean(v)/np.std(v)*(12**0.5) if len(v)>2 and np.std(v)>0 else 0
    cv=100; pk=100; md=0
    for r in v: cv*=1+r/100; pk=max(pk,cv); d=(pk-cv)/pk*100; md=max(md,d)
    wr=sum(1 for r in v if r>0)/n*100
    return an,sp,md,wr

sva=st(va); svt=st(vt); svb=st(vb)
print(f"\n{'指标':>20s}  {'V4.2新版':>10s}  {'V2':>10s}  {'混合':>10s}")
print("-"*55)
print(f"{'年化':>20s}  {sva[0]:>+9.1f}%  {svt[0]:>+9.1f}%  {svb[0]:>+9.1f}%")
print(f"{'夏普':>20s}  {sva[1]:>10.2f}  {svt[1]:>10.2f}  {svb[1]:>10.2f}")
print(f"{'最大回撤':>20s}  {sva[2]:>9.1f}%  {svt[2]:>9.1f}%  {svb[2]:>9.1f}%")
print(f"{'胜率(年)':>20s}  {sva[3]:>9.1f}%  {svt[3]:>9.1f}%  {svb[3]:>9.1f}%")
print(f"\n✅ 完成")
