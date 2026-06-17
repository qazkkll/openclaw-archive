#!/usr/bin/env python3
"""V4 美股 · 逐年回测 + 因子分解"""
import json, numpy as np
from bisect import bisect_right

CACHE_DIR = "/home/admin/.openclaw/workspace/data/cache"
TICKERS = ['NVDA','AMD','MU','INTC','AVGO','QCOM','AMAT','MSFT','CRM',
    'GOOGL','AMZN','META','HD','COST','JPM','V','MA','UNH','LLY','TSLA',
    'AAPL','NFLX','ORCL','CAT','GE','WMT','DIS','BA']

spy = json.load(open(f"{CACHE_DIR}/spy.json"))
SPY_DATES = spy['dates']

def load_raw(t):
    return json.load(open(f"{CACHE_DIR}/{t}.json"))['data']

def calc_metrics(raw):
    result = {}
    for i in range(60, len(raw)):
        row = raw[i]; d = row['date']
        pr = float(row['close'])
        a20 = float(np.mean([float(raw[j]['close']) for j in range(i-19,i+1)]))
        a50 = float(np.mean([float(raw[j]['close']) for j in range(i-49,i+1)]))
        hp52 = max(float(raw[j]['close']) for j in range(i-251,i+1))
        p52 = pr/hp52*100 if hp52>0 else 100
        m15 = (pr/float(raw[i-15]['close'])-1)*100
        m20 = (pr/float(raw[i-20]['close'])-1)*100
        m25 = (pr/float(raw[i-25]['close'])-1)*100
        result[d] = {'p':pr,'a20':a20,'a50':a50,'p52':p52,'m15':m15,'m20':m20,'m25':m25}
    return result

print("加载数据...")
all_data = {}
for t in TICKERS:
    try:
        raw = load_raw(t)
        if len(raw) > 200: all_data[t] = calc_metrics(raw)
    except: pass
print(f"  {len(all_data)} 只")

# 最佳参数（从扫描结果选取）
BEST_PARAMS = [
    ("V4推荐", 25, 85, 5, 15),    # 25日动量, 85%过滤, 5只, 15天
    ("V4激进", 25, 85, 3, 25),    # 5年最优
    ("V3基准", 20, 100, 5, 20),   # 当前
]

YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
YEAR_DATES = {y: f"{y}-01-02" for y in YEARS}
YEAR_END = {y: f"{y}-12-31" for y in YEARS}

def run_strategy(data, md, m52, tn, hd, sd, ed):
    si_s = next(i for i,d in enumerate(SPY_DATES) if d >= sd)
    si_e = next(i for i,d in enumerate(SPY_DATES) if d >= ed)
    ss = si_s + (md if md > 20 else 20)
    rets = []
    for si in range(ss, si_e - hd, hd):
        d_pr = SPY_DATES[si - md]; d_by = SPY_DATES[si]; d_sl = SPY_DATES[min(si+hd, si_e)]
        mom = []
        for t, td in data.items():
            vb = td.get(d_by); vp = td.get(d_pr)
            if vb and vp and vb['p'] > 1:
                mv = vb.get(f'm{md}') or (vb['p']/vp['p']-1)*100
                mom.append((t, mv, vb['p52']))
        if len(mom) < tn: continue
        mom.sort(key=lambda x: x[1], reverse=True)
        fl = [x for x in mom if x[2] < m52]
        if len(fl) < tn: fl = mom[:tn]
        for t,_,_ in fl[:tn]:
            v_b = data[t].get(d_by); v_s = data[t].get(d_sl)
            if v_b and v_s and v_b['p'] > 1:
                rets.append((v_s['p']/v_b['p']-1)*100)
    return rets if rets else [0]

# 逐年回测
print("\n" + "=" * 85)
print("逐年回测")
print("=" * 85)

header = f"{'年份':>6s}"
for name,_,_,_,_ in BEST_PARAMS:
    header += f" {name:>10s}"
header += f" {'SPY':>8s}"
print(header)
print("-" * 85)

all_spy_rets = []
all_strat_rets = {name: [] for name,_,_,_,_ in BEST_PARAMS}

for y in YEARS:
    sd = f"{y}-01-02"; ed = f"{y}-12-31"
    
    spy_start = float(spy['close'][0])
    for i, d in enumerate(SPY_DATES):
        if d >= sd:
            spy_start = float(spy['close'][i])
            break
    spy_end = float(spy['close'][-1])
    for i, d in enumerate(SPY_DATES):
        if d >= ed:
            spy_end = float(spy['close'][i])
            break
    spy_yr = (spy_end/spy_start-1)*100
    
    line = f"{y:6d}"
    for name, md, m52, tn, hd in BEST_PARAMS:
        rets = run_strategy(all_data, md, m52, tn, hd, sd, ed)
        avg = np.mean(rets) if rets else 0
        tot = sum(rets) if rets else 0
        line += f" {tot:>+10.1f}%"
        all_strat_rets[name].append(tot)
    line += f" {spy_yr:>+8.1f}%"
    all_spy_rets.append(spy_yr)
    print(line)

print("-" * 85)
print(f"{'累计':>6s}", end="")
for name, md, m52, tn, hd in BEST_PARAMS:
    total = sum(all_strat_rets[name])
    print(f" {total:>+10.1f}%", end="")
print(f" {sum(all_spy_rets):>+8.1f}%")

# 年化
import math
print(f"{'年化':>6s}", end="")
for name, md, m52, tn, hd in BEST_PARAMS:
    total = sum(all_strat_rets[name])
    annualized = ((1+total/100)**(1/len(YEARS))-1)*100 if total > -100 else 0
    print(f" {annualized:>+9.2f}%", end="")
spy_ann = ((1+sum(all_spy_rets)/100)**(1/len(YEARS))-1)*100
print(f" {spy_ann:>+8.2f}%")
print()

# 胜率
print(f"{'胜率':>6s}", end="")
for name, md, m52, tn, hd in BEST_PARAMS:
    wins = sum(1 for r in all_strat_rets[name] if r > 0)
    print(f" {wins/len(YEARS)*100:>9.1f}%", end="")
print(f" {sum(1 for r in all_spy_rets if r>0)/len(YEARS)*100:>8.1f}%")

# 因子贡献度（对V4推荐参数）
print("\n" + "=" * 85)
print("因子贡献度分解 (V4推荐: 25日动量, 85%过滤, 5只, 15天)")
print("每个因子去掉后的收益变化")
print("=" * 85)

md, m52, tn, hd = 25, 85, 5, 15
baseline_rets = []
for y in YEARS:
    sd = f"{y}-01-02"; ed = f"{y}-12-31"
    rets = run_strategy(all_data, md, m52, tn, hd, sd, ed)
    baseline_rets.append(sum(rets) if rets else 0)
baseline_total = sum(baseline_rets)

factors = [
    ("去掉52周过滤(m52=100)", 25, 100, 5, 15),
    ("换20日动量(md=20)", 20, 85, 5, 15),
    ("换10天调仓(hd=10)", 25, 85, 5, 10),
    ("换3只持仓(tn=3)", 25, 85, 3, 15),
    ("换8只持仓(tn=8)", 25, 85, 8, 15),
]

print(f"{'因子':>25s} {'累计':>8s} {'vs基准':>8s}")
print("-" * 45)
print(f"{'基准(V4推荐)':>25s} {baseline_total:>+8.1f}% {'--':>8s}")

for fname, fmd, fm52, ftn, fhd in factors:
    total = 0
    for y in YEARS:
        sd = f"{y}-01-02"; ed = f"{y}-12-31"
        rets = run_strategy(all_data, fmd, fm52, ftn, fhd, sd, ed)
        total += sum(rets) if rets else 0
    diff = total - baseline_total
    print(f"{fname:>25s} {total:>+8.1f}% {diff:>+8.1f}%")

print("\n✅ 完成")
