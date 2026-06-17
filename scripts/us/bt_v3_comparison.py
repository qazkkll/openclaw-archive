#!/usr/bin/env python3
"""V4美股 · V3同条件对比测试（SP500 Top100级池，2021-2026）"""
import json, numpy as np, os
from bisect import bisect_right

CACHE = "/home/admin/.openclaw/workspace/data/cache"
ALL = sorted([f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json'])

# 读全量数据（只读close）
def load_close(ticker):
    raw = json.load(open(f"{CACHE}/{ticker}.json"))['data']
    return {d['date']: d['close'] for d in raw}

spy_dates = json.load(open(f"{CACHE}/spy.json"))['dates']

# 预计算：对每个ticker，在SPY日期索引上对齐价格
print(f"加载 {len(ALL)} 只...")
all_prices = {}
for t in ALL:
    try:
        close_dict = load_close(t)
        # 转为SPY日期对齐列表
        prices = []
        for d in spy_dates:
            v = close_dict.get(d)
            if v is None:
                prices.append(None)
            else:
                prices.append(float(v))
        all_prices[t] = prices
    except:
        pass
print(f"  成功: {len(all_prices)} 只")

def run_strat(penalty_rate, start_d, end_d, md=20, tn=5, hd=20):
    """运行策略，返回年总收益%"""
    si_s = bisect_right(spy_dates, start_d) - 1
    si_e = bisect_right(spy_dates, end_d) - 1
    ss = si_s + max(md, 25)
    
    period_rets = []
    for si in range(ss, si_e - hd, hd):
        d_pr = spy_dates[si - md]
        d_buy = spy_dates[si]
        d_sell = spy_dates[min(si + hd, si_e)]
        
        # 计算每个ticker的动量
        cand = []
        for t, prices in all_prices.items():
            v_prev = prices[spy_dates.index(d_pr)] if d_pr in spy_dates else None
            v_now = prices[spy_dates.index(d_buy)] if d_buy in spy_dates else None
            if v_prev and v_now and v_prev > 1 and v_now > 1:
                mom = (v_now / v_prev - 1) * 100
                if penalty_rate > 0:
                    # 52周高分（近252天最高）
                    idx_buy = spy_dates.index(d_buy)
                    idx_p = max(0, idx_buy - 252)
                    hp52 = max(p for p in prices[idx_p:idx_buy+1] if p is not None)
                    p52 = v_now / hp52 * 100 if hp52 > 0 else 100
                    adj = 1 - max(0, (p52 - 50) / 50 * penalty_rate)
                    score = mom * adj
                else:
                    score = mom
                cand.append((score, t, v_now))
        
        if len(cand) < tn: continue
        cand.sort(key=lambda x: x[0], reverse=True)
        top = cand[:tn]
        
        rr = []
        for _, t, v_b in top:
            v_s = all_prices[t][spy_dates.index(d_sell)] if d_sell in spy_dates else None
            if v_s and v_b > 1:
                rr.append((v_s / v_b - 1) * 100)
        if rr:
            period_rets.append(np.mean(rr))
    
    if not period_rets: return 0
    cum = 1
    for r in period_rets:
        cum *= (1 + r / 100)
    return (cum - 1) * 100

# V3同条件：2021-2026，5年
SD = "2021-01-01"
ED = "2026-05-15"

# 计算SPY同期收益
spy = json.load(open(f"{CACHE}/spy.json"))
si_s = bisect_right(spy_dates, SD) - 1
si_e = bisect_right(spy_dates, ED) - 1
spy_ret = (float(spy['close'][si_e]) / float(spy['close'][si_s]) - 1) * 100

print(f"\n{'='*80}")
print(f"V3同条件对比 | 池:{len(all_prices)}只 | {SD}~{ED} ({si_e-si_s}天)")
print(f"{'='*80}")

STRATS = [
    ("V3纯动量(无过滤)", 0.0, 20, 5, 20),
    ("V4比例扣分50%", 0.5, 20, 5, 20),
    ("V4比例扣分70%", 0.7, 20, 5, 20),
]

# 累计收益
for n, pp, md, tn, hd in STRATS:
    ret = run_strat(pp, SD, ED, md, tn, hd)
    ann = ((1 + ret / 100) ** (1 / 5.4) - 1) * 100  # 5年零5个月
    print(f"{n:>20s}: 累计+{ret:>+8.1f}%  年化{ann:>+6.2f}%")
print(f"{'SPY':>20s}: 累计+{spy_ret:>+8.1f}%  年化{((1+spy_ret/100)**(1/5.4)-1)*100:>+6.2f}%")

# 逐年
print(f"\n{'='*80}")
print("逐年对比")
print(f"{'='*80}")
print(f"{'年份':>5s}", end="")
for n,_,_,_,_ in STRATS: print(f" {n:>16s}", end="")
print(f" {'SPY':>8s}")
print("-" * 65)

YEARS = [2021, 2022, 2023, 2024, 2025]
all_r = {n:[] for n,_,_,_,_ in STRATS}
all_s = []

for y in YEARS:
    sy = f"{y}-01-01"; ey = f"{y}-12-31"
    si_sy = bisect_right(spy_dates, sy) - 1
    si_ey = bisect_right(spy_dates, ey) - 1
    spy_y = (float(spy['close'][si_ey]) / float(spy['close'][si_sy]) - 1) * 100
    
    l = f"{y:>5d}"
    for n, pp, md, tn, hd in STRATS:
        ret = run_strat(pp, sy, ey, md, tn, hd)
        l += f" {ret:>+16.1f}%"
        all_r[n].append(ret)
    l += f" {spy_y:>+8.1f}%"
    all_s.append(spy_y)
    print(l)

print("-" * 65)
print(f"{'累计':>5s}", end="")
for n,_,_,_,_ in STRATS:
    t = sum(all_r[n]); print(f" {t:>+16.1f}%", end="")
print(f" {sum(all_s):>+8.1f}%")

print(f"{'年化':>5s}", end="")
for n,_,_,_,_ in STRATS:
    t = sum(all_r[n])
    a = ((1 + t / 100) ** (1 / len(YEARS)) - 1) * 100 if t > -100 else 0
    print(f" {a:>+15.2f}%", end="")
print(f" {((1+sum(all_s)/100)**(1/len(YEARS))-1)*100:>+8.2f}%")

print(f"\n{'='*80}")
print("比例扣分 vs V3纯动量 差值（超额）")
print(f"{'='*80}")
print(f"{'年份':>5s} {'扣50%超额':>12s} {'扣70%超额':>12s}")
for i, y in enumerate(YEARS):
    d50 = all_r['V4比例扣分50%'][i] - all_r['V3纯动量(无过滤)'][i]
    d70 = all_r['V4比例扣分70%'][i] - all_r['V3纯动量(无过滤)'][i]
    print(f"{y:>5d} {d50:>+12.1f}% {d70:>+12.1f}%")
print("-" * 35)
print(f"{'结论':>5s}")
