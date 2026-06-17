#!/usr/bin/env python3
"""
美股多因子评分模型 · 系数暴力扫描
框架：6因子综合评分（动量+MACD+ADX+均线+RSI+52周位）
目标：找到最优权重组合
"""

import json, numpy as np, os, itertools, sys
from bisect import bisect_right

CACHE = "/home/admin/.openclaw/workspace/data/cache"
ALL = sorted([f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json'])
spy = json.load(open(f"{CACHE}/spy.json"))
SPY_DATES = spy['dates']

# 加载并预计算所有因子（只读 close，实时算因子）
def load_factors(ticker):
    raw = json.load(open(f"{CACHE}/{ticker}.json"))['data']
    closes = [float(r['close']) for r in raw]
    dates = [r['date'] for r in raw]
    result = {}
    for i in range(60, len(raw)):
        d = dates[i]; c = closes
        cp = c[i]
        # 动量
        m20 = (cp/c[i-20]-1)*100
        m10 = (cp/c[i-10]-1)*100
        m30 = (cp/c[i-30]-1)*100
        # 52周位置
        hp52 = max(c[max(0,i-251):i+1])
        p52 = cp/hp52*100 if hp52 > 0 else 100
        # 均线
        ma20 = np.mean(c[max(0,i-19):i+1])
        ma50 = np.mean(c[max(0,i-49):i+1])
        ma200 = np.mean(c[max(0,i-199):i+1]) if i >= 199 else 0
        # MACD
        # MACD
        def ema(arr, p):
            k = 2/(p+1); r = arr[0]
            for v in arr[1:]: r = v*k + r*(1-k)
            return r
        macd_val = ema(c[max(0,i-25):i+1],12) - ema(c[max(0,i-51):i+1],26) if i >= 51 else None
        # RSI
        gains = sum(max(0, c[j]-c[j-1]) for j in range(max(13,i-13),i+1))/14
        losses = sum(max(0, c[j-1]-c[j]) for j in range(max(13,i-13),i+1))/14
        rsi = 100-100/(1+gains/(losses+0.001)) if losses > 0 else 100
        # ADX 简化版
        tr = max(c[i]-c[i-1], 0) if i > 0 else 0
        adx = 25  # 简化
        
        result[d] = {
            'p': cp, 'm20': m20, 'p52': p52,
            'a20': ma20, 'a50': ma50, 'a200': ma200,
            'rsi': rsi, 'adx': adx,
            'macd': macd_val
        }
    return result, dates, closes

# 预计算
print("预计算因子...")
all_data = {}
for t in ALL:
    try:
        fd, _, _ = load_factors(t)
        all_data[t] = fd
    except: pass
    if len(all_data) % 20 == 0: print(f"  {len(all_data)}/{len(ALL)}", flush=True)
print(f"  加载 {len(all_data)} 只")

def score_stock(vb, month_weight=0):
    """多因子评分，返回0-100分。month_weight控制动量权重比例"""
    if not vb: return 0
    
    # 原始因子
    m20 = vb['m20']                     # 20日动量(%)
    p52 = vb['p52']                     # 52周位置(%)
    cp = vb['p']; a20 = vb['a20']; a50 = vb['a50']; a200 = vb['a200']
    rsi = vb['rsi']; macd = vb.get('macd')
    
    # MACD门控：如果MACD可用且≤0则0分；如果MACD不可用(数据不足)则跳过MACD门
    if macd is not None and macd <= 0: return 0
    
    # 各因子得分(0-20)
    # 动量得分: 0-30%之间线性, >30%得满分, <0得0
    ms = min(20, max(0, m20 / 30 * 20))
    
    # 52周位置(越低越好)
    ws = 20 if p52 < 20 else 15 if p52 < 35 else 10 if p52 < 50 else 6 if p52 < 65 else 3 if p52 < 80 else 0
    
    # 均线系统(站上MA20/50/200加分)
    mas = (7 if cp > a20 else 0) + (7 if cp > a50 else 0) + (6 if a200 and cp > a200 else 0)
    
    # ADX(用动量变化代替)
    ads = 10 if abs(m20) > 5 else 5 if abs(m20) > 2 else 0
    
    # RSI(超卖加分，超买扣分)
    rs = 10 if rsi < 35 else 6 if rsi < 50 else 2 if rsi < 65 else -3 if rsi < 75 else -5
    
    # 综合
    total = ms + ws + mas + ads + rs
    
    return min(round(total), 100)

def run_strat_v3(sd, ed, md=20, tn=5, hd=20):
    """V3纯动量：按20日动量排序"""
    si_s = bisect_right(SPY_DATES, sd) - 1
    si_e = bisect_right(SPY_DATES, ed) - 1
    ss = si_s + max(md, 25)
    period_rets = []
    for si in range(ss, si_e - hd, hd):
        dp = SPY_DATES[si - md]; db = SPY_DATES[si]; ds = SPY_DATES[min(si + hd, si_e)]
        cand = []
        for t, td in all_data.items():
            vb = td.get(db); vp = td.get(dp)
            if not vb or not vp or vb['p'] < 1: continue
            cand.append((vb['m20'], t, vb['p']))
        if len(cand) < tn: continue
        cand.sort(key=lambda x: x[0], reverse=True)
        rr = []
        for _, t, v_b in cand[:tn]:
            vs = all_data[t].get(ds)
            if vs and v_b > 1: rr.append((vs['p'] / v_b - 1) * 100)
        if rr: period_rets.append(np.mean(rr))
    return period_rets

def run_strat_v4(sd, ed, md=20, tn=5, hd=20, pp=0.5):
    """V4比例扣分：20日动量 × 52周位置扣分"""
    si_s = bisect_right(SPY_DATES, sd) - 1
    si_e = bisect_right(SPY_DATES, ed) - 1
    ss = si_s + max(md, 25)
    period_rets = []
    for si in range(ss, si_e - hd, hd):
        dp = SPY_DATES[si - md]; db = SPY_DATES[si]; ds = SPY_DATES[min(si + hd, si_e)]
        cand = []
        for t, td in all_data.items():
            vb = td.get(db); vp = td.get(dp)
            if not vb or not vp or vb['p'] < 1: continue
            mom = vb['m20']; p52 = vb['p52']
            sc = mom * (1 - max(0, (p52 - 50) / 50 * pp))
            cand.append((sc, t, vb['p']))
        if len(cand) < tn: continue
        cand.sort(key=lambda x: x[0], reverse=True)
        rr = []
        for _, t, v_b in cand[:tn]:
            vs = all_data[t].get(ds)
            if vs and v_b > 1: rr.append((vs['p'] / v_b - 1) * 100)
        if rr: period_rets.append(np.mean(rr))
    return period_rets

def run_strat_mfs(sd, ed, md=20, tn=5, hd=20):
    """MFS多因子评分：6因子综合评分"""
    si_s = bisect_right(SPY_DATES, sd) - 1
    si_e = bisect_right(SPY_DATES, ed) - 1
    ss = si_s + max(md, 30)
    period_rets = []
    for si in range(ss, si_e - hd, hd):
        dp = SPY_DATES[si - md]; db = SPY_DATES[si]; ds = SPY_DATES[min(si + hd, si_e)]
        cand = []
        for t, td in all_data.items():
            vb = td.get(db); vp = td.get(dp)
            if not vb or not vp or vb['p'] < 1: continue
            sc = score_stock(vb)
            if sc > 0: cand.append((sc, t, vb['p']))
        if len(cand) < tn: continue
        cand.sort(key=lambda x: x[0], reverse=True)
        rr = []
        for _, t, v_b in cand[:tn]:
            vs = all_data[t].get(ds)
            if vs and v_b > 1: rr.append((vs['p'] / v_b - 1) * 100)
        if rr: period_rets.append(np.mean(rr))
    return period_rets

# ===== 对比测试 =====
SD = "2021-07-01"; ED = "2025-12-31"      # 从7月开始，给MACD留足预热数据

# SPY
si_s = bisect_right(SPY_DATES, SD) - 1
si_e = bisect_right(SPY_DATES, ED) - 1

print(f"\n{'='*85}")
print("多因子评分 vs 动量 vs 比例扣分 | 2021-2025")
print(f"{'='*85}")

for label, fn in [
    ("V3纯动量(无过滤)", lambda: run_strat_v3(SD, ED)),
    ("V4比例扣分50%",    lambda: run_strat_v4(SD, ED, pp=0.5)),
    ("V4比例扣分70%",    lambda: run_strat_v4(SD, ED, pp=0.7)),
    ("多因子评分模型",    lambda: run_strat_mfs(SD, ED)),
]:
    rets = fn()
    if rets:
        cum = 1
        for r in rets: cum *= (1 + r/100)
        total = (cum - 1) * 100
        ann = ((1 + total/100) ** (1/5) - 1) * 100 if total > -100 else 0
        print(f"{label:>20s}: +{total:>+8.1f}%  年化{ann:>+6.2f}%")

# 逐年
print(f"\n{'='*85}")
print("逐年分解")
print(f"{'='*85}")

YEARS = [2021, 2022, 2023, 2024, 2025]
for label, fn in [
    ("V3纯动量", lambda y: run_strat_v3(f"{y}-01-01", f"{y}-12-31")),
    ("V4扣70%", lambda y: run_strat_v4(f"{y}-01-01", f"{y}-12-31", pp=0.7)),
    ("MFS多因子", lambda y: run_strat_mfs(f"{y}-01-01", f"{y}-12-31")),
]:
    line = f"{label:>10s}:"
    for y in YEARS:
        rets = fn(y)
        if rets:
            cum = 1
            for r in rets: cum *= (1 + r/100)
            line += f" {(cum-1)*100:>+8.1f}%"
        else:
            line += f" {'N/A':>8s}"
    print(line)

# 单独跑多因子评分各年
print(f"\n{'='*85}")
print("多因子评分模型·逐年详细")
print(f"{'='*85}")
print(f"{'年份':>5s} {'收益':>10s} {'申数':>6s} {'说明':>30s}")
for y in YEARS:
    rets = run_strat_mfs(f"{y}-01-01", f"{y}-12-31")
    if rets:
        cum = 1
        for r in rets: cum *= (1 + r/100)
        ret = (cum - 1) * 100
        print(f"{y:>5d} {ret:>+10.1f}% {len(rets):>6d} {'':>30s}")

print("\n✅ 完成")
