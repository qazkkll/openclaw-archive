#!/usr/bin/env python3
"""V4美股 · 三大框架逐年回测（2014-2025）"""
import json, numpy as np, os
from bisect import bisect_right

CACHE = "/home/admin/.openclaw/workspace/data/cache"
TICKERS = [f.replace('.json','') for f in os.listdir(CACHE) if f not in ['spy.json','us_pc.json','us_pc10y.json','fw_comparison.json']]
spy = json.load(open(f"{CACHE}/spy.json"))
SPY_DATES = spy['dates']

def get_idx(d):
    return bisect_right(SPY_DATES, d) - 1

def prep(ticker, sd_idx, ed_idx):
    raw = json.load(open(f"{CACHE}/{ticker}.json"))['data']
    result = {}
    for i in range(60, len(raw)):
        row = raw[i]; d = row['date']
        di = bisect_right(SPY_DATES, d) - 1
        if di < sd_idx - 60 or di > ed_idx + 30: continue
        pr = float(row['close'])
        hp52 = max(float(raw[j]['close']) for j in range(i-251,i+1))
        p52 = pr/hp52*100 if hp52>0 else 100
        m25 = (pr/float(raw[i-25]['close'])-1)*100
        m15 = (pr/float(raw[i-15]['close'])-1)*100
        m30 = (pr/float(raw[i-30]['close'])-1)*100
        result[d] = {'p':pr,'p52':p52,'m15':m15,'m25':m25,'m30':m30}
    return result

SD = "2014-01-02"; ED = "2025-12-31"

# 预计算
print("加载数据...")
si_s = get_idx(SD); si_e = get_idx(ED)
print(f"区间: {SD}~{ED} ({si_e-si_s}天)")

all_data = {}
for t in TICKERS:
    try:
        all_data[t] = prep(t, si_s, si_e)
    except: pass
print(f"  {len(all_data)} 只")

# 三大框架最优参数
TOP_CONFIGS = [
    ("B-比例扣分", {"md":30, "tn":3, "hd":30, "pp":0.5}),
    ("C-硬过滤",   {"md":15, "tn":3, "hd":25, "th":80}),
    ("A-纯动量",   {"md":25, "tn":3, "hd":10}),
]

def run_year(name, params, year):
    sd = f"{year}-01-02"; ed = f"{year}-12-31"
    st = get_idx(sd); en = get_idx(ed)
    ss = st + max(params.get('md',20), 25)
    md = params['md']; tn = params['tn']; hd = params['hd']
    pp = params.get('pp',0); th = params.get('th',0)
    
    rets = []
    for si in range(ss, en - hd, hd):
        d_pr = SPY_DATES[si-md]; d_by = SPY_DATES[si]; d_sl = SPY_DATES[min(si+hd, en)]
        cand = []
        for t, td in all_data.items():
            vb = td.get(d_by); vp = td.get(d_pr)
            if not vb or not vp or vb['p']<1: continue
            mom = {25:vb.get('m25',(vb['p']/vp['p']-1)*100),15:vb.get('m15'),30:vb.get('m30')}.get(md,(vb['p']/vp['p']-1)*100)
            p52 = vb['p52']
            if name == 'A-纯动量': sc = mom
            elif name == 'B-比例扣分':
                adj = 1 - max(0, (p52-50)/50 * pp)
                sc = mom * adj
            elif name == 'C-硬过滤':
                if p52 > th: continue
                sc = mom
            cand.append((sc, t))
        
        if len(cand) < tn: continue
        cand.sort(key=lambda x:x[0], reverse=True)
        rr = []
        for _, t in cand[:tn]:
            v_b = all_data[t].get(d_by); v_s = all_data[t].get(d_sl)
            if v_b and v_s and v_b['p']>1: rr.append((v_s['p']/v_b['p']-1)*100)
        if rr: rets.append(np.mean(rr))
    return rets

YEARS = list(range(2014, 2026))

print(f"\n{'='*100}")
print(f"三大框架 · 逐年回测 (2014-2025)")
print(f"{'='*100}")
header = f"{'年份':>6s}"
for n,_ in TOP_CONFIGS: header += f" {n:>12s}"
header += f" {'SPY':>8s}"
print(header)
print("-" * 100)

all_strat = {n:[] for n,_ in TOP_CONFIGS}
all_spy = []

for y in YEARS:
    # SPY年度表现
    spy_start = float(spy['close'][get_idx(f"{y}-01-02")])
    spy_end = float(spy['close'][get_idx(f"{y}-12-31")])
    spy_yr = (spy_end/spy_start-1)*100
    
    line = f"{y:>6d}"
    for n, p in TOP_CONFIGS:
        rets = run_year(n, p, y)
        total = sum(rets) if rets else 0
        line += f" {total:>+12.1f}%"
        all_strat[n].append(total)
    line += f" {spy_yr:>+8.1f}%"
    all_spy.append(spy_yr)
    print(line)

# 汇总
print("-" * 100)
print(f"{'累计':>6s}", end="")
for n,_ in TOP_CONFIGS:
    total = sum(all_strat[n])
    ann = ((1+total/100)**(1/len(YEARS))-1)*100 if total>-100 else 0
    print(f" {total:>+12.1f}%", end="")
print(f" {sum(all_spy):>+8.1f}%")

print(f"{'年化':>6s}", end="")
for n,_ in TOP_CONFIGS:
    total = sum(all_strat[n])
    ann = ((1+total/100)**(1/len(YEARS))-1)*100 if total>-100 else 0
    print(f" {ann:>+11.2f}%", end="")
spy_ann = ((1+sum(all_spy)/100)**(1/len(YEARS))-1)*100
print(f" {spy_ann:>+8.2f}%")

print(f"{'胜率':>6s}", end="")
for n,_ in TOP_CONFIGS:
    wins = sum(1 for r in all_strat[n] if r > 0)
    print(f" {wins/len(YEARS)*100:>11.1f}%", end="")
print(f" {sum(1 for r in all_spy if r>0)/len(YEARS)*100:>8.1f}%")

# 超额收益
print(f"\n{'='*100}")
print("超额收益分析（vs SPY）")
print(f"{'='*100}")
for n,_ in TOP_CONFIGS:
    total_s = sum(all_strat[n])
    total_spy_s = sum(all_spy)
    excess = total_s - total_spy_s
    print(f"{n:>15s}: 策略+{total_s:.1f}% vs SPY+{total_spy_s:.1f}% = 超额+{excess:.1f}%")
    # 各年跑赢次数
    beat = sum(1 for i in range(len(YEARS)) if all_strat[n][i] > all_spy[i])
    print(f"{'':>15s}  跑赢SPY年份: {beat}/{len(YEARS)} ({100*beat//len(YEARS)}%)")

# 累积曲线（简化）
print(f"\n{'='*100}")
print("累积收益曲线")
print(f"{'='*100}")
print(f"{'年份':>6s}", end="")
for n,_ in TOP_CONFIGS:
    print(f" {n:>12s}", end="")
print(f" {'SPY':>8s}")
print("-" * 100)
cum = {n:0 for n,_ in TOP_CONFIGS}
cum_spy = 0
for i, y in enumerate(YEARS):
    for n,_ in TOP_CONFIGS:
        cum[n] += all_strat[n][i]
    cum_spy += all_spy[i]
    line = f"{y:>6d}"
    for n,_ in TOP_CONFIGS:
        line += f" {cum[n]:>+12.1f}%"
    line += f" {cum_spy:>+8.1f}%"
    print(line)

print("\n✅ 完成")
