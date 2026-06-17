#!/usr/bin/env python3
"""V4 美股 · 完整系数扫描 + 回测"""
import json, itertools, numpy as np, os
from bisect import bisect_right

CACHE_DIR = "/home/admin/.openclaw/workspace/data/cache"
TICKERS = ['NVDA','AMD','MU','INTC','AVGO','QCOM','AMAT','MSFT','CRM',
    'GOOGL','AMZN','META','HD','COST','JPM','V','MA','UNH','LLY','TSLA',
    'AAPL','NFLX','ORCL','CAT','GE','WMT','DIS','BA']

spy = json.load(open(f"{CACHE_DIR}/spy.json"))
SPY_DATES = spy['dates']

def load_ticker(t):
    raw = json.load(open(f"{CACHE_DIR}/{t}.json"))['data']
    return raw  # 保留原始list格式

def gv_list(lst, d):
    """从list[{date,close}]中获取某日数据"""
    for i in range(len(lst)-1, -1, -1):
        if lst[i]['date'] <= d:
            return lst[i]
    return None

def calc_metrics(raw):
    """从原始OHLCV列表预计算指标"""
    result = {}
    for i in range(60, len(raw)):
        row = raw[i]
        d = row['date']
        pr = float(row['close'])
        a20 = float(np.mean([float(raw[j]['close']) for j in range(i-19,i+1)]))
        a50 = float(np.mean([float(raw[j]['close']) for j in range(i-49,i+1)]))
        hp52 = max(float(raw[j]['close']) for j in range(i-251,i+1))
        p52 = pr/hp52*100 if hp52>0 else 100
        m15 = (pr/float(raw[i-15]['close'])-1)*100
        m20 = (pr/float(raw[i-20]['close'])-1)*100
        result[d] = {'p':pr,'a20':a20,'a50':a50,'p52':p52,'m15':m15,'m20':m20}
    return result

# 预计算所有指标
print("加载并预计算数据...")
all_data = {}
for i, t in enumerate(TICKERS):
    try:
        raw = load_ticker(t)
        if len(raw) > 200:
            all_data[t] = calc_metrics(raw)
    except:
        pass
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(TICKERS)}", flush=True)
print(f"  {len(all_data)} 只")

PARAMS = {
    'momentum_days': [10, 15, 20, 25, 30],
    'max_52w': [85, 90, 95, 100],
    'top_n': [3, 5, 8],
    'hold_days': [10, 15, 20, 25, 30],
}
total_c = 1
for v in PARAMS.values(): total_c *= len(v)

PERIODS = [
    ("5年", "2020-01-02", "2025-12-31"),
    ("10年", "2015-01-02", "2025-12-31"),
]

for pname, sd, ed in PERIODS:
    print(f"\n{'='*65}")
    print(f"扫描: {pname} ({sd}-{ed})")
    print(f"{'='*65}")
    
    si_s = next(i for i,d in enumerate(SPY_DATES) if d >= sd)
    si_e = next(i for i,d in enumerate(SPY_DATES) if d >= ed)
    
    results = []
    
    for idx, (md, m52, tn, hd) in enumerate(itertools.product(*PARAMS.values())):
        strategy_rets = []
        ss = si_s + (md if md > 20 else 20)
        
        for si in range(ss, si_e - hd, hd):
            d_pr = SPY_DATES[si - md]; d_by = SPY_DATES[si]; d_sl = SPY_DATES[min(si+hd, si_e)]
            mom = []
            for t, td in all_data.items():
                vb = td.get(d_by); vp = td.get(d_pr)
                if vb and vp and vb['p'] > 1:
                    mv = vb['m15'] if md==15 else vb['m20'] if md==20 else (vb['p']/vp['p']-1)*100
                    mom.append((t, mv, vb['p52']))
            if len(mom) < tn: continue
            mom.sort(key=lambda x: x[1], reverse=True)
            fl = [x for x in mom if x[2] < m52]
            if len(fl) < tn: fl = mom[:tn]
            tp = fl[:tn]
            rr = []
            for t,_,_ in tp:
                v_b = td.get(d_by) if (td:=all_data.get(t)) else None
                v_s = td.get(d_sl) if td else None
                if v_b and v_s and v_b['p'] > 1:
                    rr.append((v_s['p']/v_b['p']-1)*100)
            if rr: strategy_rets.append(np.mean(rr))
        
        if strategy_rets:
            results.append((sum(strategy_rets), np.mean(strategy_rets),
                          100*sum(1 for r in strategy_rets if r>0)/len(strategy_rets),
                          min(strategy_rets), md, m52, tn, hd))
        
        if (idx+1) % 50 == 0:
            print(f"  {idx+1}/{total_c}", flush=True)
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n{'='*65}")
    print(f"🏆 {pname} TOP 10")
    print(f"{'='*65}")
    print(f"{'#':>3s} {'总收益':>8s} {'均/期':>7s} {'胜率':>5s} {'最差':>6s}  参数")
    print("-" * 55)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:3d} {r[0]:>+8.1f}% {r[1]:>+7.2f}% {r[2]:>4.1f}% {r[3]:>+6.1f}%  {r[4]:2d}日 {r[5]:3d}% {r[6]}只 {r[7]}天")
    
    # 基准
    for r in results:
        if r[4]==20 and r[5]==100 and r[6]==5 and r[7]==20:
            print(f"\n基准(20/100/5/20): +{r[0]:.1f}%")
            print(f"最优: +{results[0][0]:.1f}%  (+{results[0][0]-r[0]:.1f}%)")
            break

print("\n✅ 完成")
