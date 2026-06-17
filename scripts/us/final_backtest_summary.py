#!/usr/bin/env python3
"""
小钳轮动 v1 最终版 — 严谨全量回测
混合8集群(16→8合并) + 均衡参数 + ETF
每年度独立验证
"""
import json, math, sys
from collections import defaultdict
from datetime import datetime

print("=" * 60)
print("📥 加载数据")
print("=" * 60)

with open('/home/admin/.openclaw/workspace/data/backtest_hist_v3.json') as f:
    hist = json.load(f)
with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f:
    sector_map = json.load(f)

dates = hist[list(hist.keys())[0]]['dates']
N = min(len(hist[c]['close']) for c in hist)
print(f"  股票: {len(hist)} 只 × {N} 天 ({dates[0]} ~ {dates[-1]})")
print(f"  行业: {len(set(sector_map.values()))}类")

ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}

# 行业分组
ss = defaultdict(list)
for code in hist:
    ss[sector_map.get(code,'其他')].append(code)
sector_stocks = dict(ss)
print(f"\n  行业分布:")
for sec, codes in sorted(sector_stocks.items(), key=lambda x:-len(x[1])):
    etf_in = sum(1 for c in codes if c in ETFS)
    print(f"    {sec}: {len(codes)}只{' (含'+str(etf_in)+'只ETF)' if etf_in else ''}")

# ==== 指标 ====
def ema(a,n):
    k=2/(n+1);r=[a[0]]
    for v in a[1:]:r.append(v*k+r[-1]*(1-k))
    return r
def sms(a,n):
    return [None]*(n-1)+[sum(a[i-n+1:i+1])/n for i in range(n-1,len(a))]
def calc(c,h,l):
    n=len(c);m5=sms(c,5);m20=sms(c,20);m60=sms(c,60)
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
    return {'c':c,'m5':m5,'m20':m20,'m60':m60,'r':ra,'mh':mh,'p':p52,'a':ax}

print("\n🔧 计算指标...")
ind={}
for code, d in hist.items():
    ind[code]=calc(d['close'],d['high'],d['low'])

def sf(arr,i):
    return arr[i] if 0<=i<len(arr) and arr[i] is not None else None

def score(i,code):
    id=ind.get(code)
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

def sec_mom(i):
    m={}
    for sec, codes in sector_stocks.items():
        rets=[]
        for c in codes[:15]:
            id=ind.get(c)
            if not id: continue
            pn=sf(id['c'],i);pb=sf(id['c'],max(0,i-20))
            if pn and pb and pb>0: rets.append((pn-pb)/pb*100)
        if len(rets)>=3: m[sec]=sum(rets)/len(rets)
    return m

def simulate(s, e, verbose=False):
    """完整模拟器, 返回详细统计"""
    cash=1000000.0; pos={}; trades=[]; daily=[]
    cfg = {'top_n':4,'hold_n':4,'buy_t':62,'sell_t':48,'rebal_days':5,'max_pos':8,'per_sec':2}
    
    for i in range(s, e):
        if (i-s) % cfg['rebal_days'] == 0:
            mom=sec_mom(i)
            if mom:
                rk=sorted(mom.items(), key=lambda x:-x[1])
                t=[r[0] for r in rk[:cfg['top_n']]]
                g=[r[0] for r in rk[:cfg['hold_n']]]
                
                for c in list(pos.keys()):
                    if pos[c]['s'] not in g:
                        id=ind.get(c); pr=sf(id['c'],i) if id else None
                        if pr and pr>0:
                            pnl=(pr-pos[c]['ep'])/pos[c]['ep']*100
                            cash+=pos[c]['v']*(1+pnl/100)
                            del pos[c]
                for sec in t:
                    if len(pos)>=cfg['max_pos']: break
                    cs=sector_stocks.get(sec,[]); ca=[]
                    for c in cs:
                        if c in pos: continue
                        sc=score(i,c)
                        if sc>=cfg['buy_t']:
                            pr=sf(ind[c]['c'],i)
                            if pr and pr>0: ca.append((c,sc,pr,c in ETFS))
                    ca.sort(key=lambda x:(-x[1], x[3]))
                    for c,sc,pr,is_etf in ca[:cfg['per_sec']]:
                        if c in pos or len(pos)>=cfg['max_pos']: break
                        inv=min(120000,cash*0.15)
                        if inv<20000: continue
                        pos[c]={'ep':pr,'v':inv,'s':sec}
                        cash-=inv
                        if verbose:
                            trades.append(f"BUY {c} @{pr:.2f} sc={sc:.0f} {'ETF' if is_etf else '股票'} {sec}")
        
        for c in list(pos.keys()):
            sc=score(i,c)
            id=ind.get(c); pr=sf(id['c'],i) if id else None
            m20=sf(id['m20'],i) if id else None; mh=sf(id['mh'],i) if id else None
            if sc<cfg['sell_t'] or (pr and m20 and pr<m20 and mh and mh<0):
                if pr and pr>0:
                    pnl=(pr-pos[c]['ep'])/pos[c]['ep']*100
                    cash+=pos[c]['v']*(1+pnl/100)
                    if verbose and abs(pnl)>3:
                        trades.append(f"SELL {c} @{pr:.2f} pnl={pnl:+.1f}%")
                    del pos[c]
        
        pv = sum(
            sf(ind[c]['c'],i)/p['ep']*p['v'] if ind.get(c) and sf(ind[c]['c'],i) else p['v']
            for c,p in pos.items()
        )
        daily.append(cash+pv)
    
    for c,p in list(pos.items()):
        id=ind.get(c)
        if id:
            pr=sf(id['c'],e-1)
            if pr and pr>0:
                pnl=(pr-p['ep'])/p['ep']*100
                cash+=p['v']*(1+pnl/100)
    
    ret=(cash-1000000)/1000000*100
    
    # 最大回撤
    peak=daily[0] if daily else 1000000; mdd=0
    for v in daily:
        if v>peak: peak=v
        dd=(peak-v)/peak*100
        if dd>mdd: mdd=dd
    
    # 夏普
    dr=[]
    for j in range(1,len(daily)):
        r2=(daily[j]-daily[j-1])/daily[j-1]*100; dr.append(r2)
    sr=0
    if dr and sum(dr)!=0:
        avg=sum(dr)/len(dr); var=sum((r-avg)**2 for r in dr)/len(dr); std=max(var**0.5, 0.001)
        sr=round(avg/std*15.8, 2)
    
    return {'ret':round(ret,2),'mdd':round(mdd,2),'sr':sr,'final':round(cash),'trades':trades}

def hold_ret(s,e):
    cl=list(ind.keys())[:100]
    ps,pe=[],[]
    for c in cl:
        a,b=sf(ind[c]['c'],s),sf(ind[c]['c'],e-1)
        if a and b: ps.append(a);pe.append(b)
    return round((sum(pe)/len(pe)-sum(ps)/len(ps))/(sum(ps)/len(ps))*100,2) if ps else 0

# ===== 各年份回测 =====
print(f"\n{'='*70}")
print("📊 小钳轮动 v1 最终版 — 年度回测")
print(f"{'='*70}")

# 定义各年时间段
WARMUP = 260
years = []
for y in [2022, 2023, 2024, 2025, 2026]:
    s = next((i for i,dt in enumerate(dates) if dt>=f'{y}-01-01'), None)
    e = next((i for i,dt in enumerate(dates) if dt>=f'{y+1}-01-01'), N)
    if s is None: continue
    if e > N: e = N
    if e - s < 30: continue
    years.append((y, s, e, f"{dates[s]}~{dates[e-1]}"))

# 先跑全周期
full_start = years[0][1]
full_end = years[-1][2]
full = simulate(full_start, full_end, verbose=False)
full_hold = hold_ret(full_start, full_end)

print(f"\n全周期 ({dates[full_start]}~{dates[full_end-1]}):")
print(f"  小钳轮动: {full['ret']:+7.2f}%")
print(f"  大盘Hold: {full_hold:+7.2f}%")
print(f"  超额收益: {full['ret']-full_hold:+7.2f}%")
print(f"  最大回撤: {full['mdd']:.2f}%")
print(f"  夏普比率: {full['sr']}")
print(f"  100万 → ¥{full['final']:,}")
print()

# 各年
print(f"{'年份':>6s} {'交易日':>6s} {'小钳轮动':>10s} {'大盘Hold':>10s} {'超额':>10s} {'最大回撤':>8s} {'夏普':>6s} {'100万→':>12s}")
print(f"{'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*12}")

total_compound = 1.0
total_hold_compound = 1.0

for year, s, e, label in years:
    r = simulate(s, e)
    h = hold_ret(s, e)
    vs = round(r['ret'] - h, 2)
    
    # 如果需要年化复利
    compound_factor = 1 + r['ret']/100
    total_compound *= compound_factor
    total_hold_compound *= (1 + h/100)
    
    # 年末资产
    if year == years[0][0]:
        annual_final = 1000000 * compound_factor
    else:
        # Cumulative
        pass
    
    print(f"  {year} ({label[:10]}) {e-s:5d} {r['ret']:+8.2f}% {h:+8.2f}% {vs:+8.2f}% {r['mdd']:>7.2f}% {r['sr']:>5.1f} ¥{1000000*compound_factor:>10,.0f}")

cum_ret = round((total_compound-1)*100, 2)
cum_hold_ret = round((total_hold_compound-1)*100, 2)
cum_final = round(1000000 * total_compound)
print(f"{'─'*6} {'─'*6} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*6} {'─'*12}")
print(f"{'累计':>6s} {'':6s} {cum_ret:+8.2f}% {cum_hold_ret:+8.2f}% {cum_ret-cum_hold_ret:+8.2f}% {'':8s} {'':6s} ¥{cum_final:>10,.0f}")

# 保存结果
output = {
    'model': '小钳轮动 v1 最终版',
    'parameters': {
        'buy_threshold': 62, 'sell_threshold': 48,
        'top_sectors': 4, 'hold_sectors': 4,
        'rebalance_days': 5, 'max_positions': 8, 'per_sector': 2,
        'sector_classes': 8, 'etfs': 10,
    },
    'full_period': {
        'from': dates[full_start],
        'to': dates[full_end-1],
        'return': full['ret'],
        'hold': full_hold,
        'outperform': round(full['ret']-full_hold, 2),
        'max_drawdown': full['mdd'],
        'sharpe': full['sr'],
        'final': full['final'],
    },
    'yearly': {},
}
for year, s, e, label in years:
    r = simulate(s, e)
    h = hold_ret(s, e)
    output['yearly'][str(year)] = {
        'days': e-s, 'return': r['ret'], 'hold': h,
        'outperform': round(r['ret']-h, 2),
        'mdd': r['mdd'], 'sr': r['sr'],
    }
output['cumulative'] = {
    'return': cum_ret, 'hold': cum_hold_ret,
    'outperform': round(cum_ret-cum_hold_ret, 2),
    'final': cum_final,
}

with open('iteration_log/final_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ 结果已保存到 iteration_log/final_results.json")
print(f"🕐 {datetime.now().strftime('%H:%M')}")
