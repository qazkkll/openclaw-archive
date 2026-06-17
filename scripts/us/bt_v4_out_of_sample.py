#!/usr/bin/env python3
"""V4美股 · 轻量样本外验证"""
import json, numpy as np, os
from bisect import bisect_right

CACHE = "/home/admin/.openclaw/workspace/data/cache"
TICKERS = [f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json']
spy = json.load(open(f"{CACHE}/spy.json"))
SPY_DATES = spy['dates']

def load_p(t):
    raw = json.load(open(f"{CACHE}/{t}.json"))['data']
    return {r['date']: float(r['close']) for r in raw}

all_p = {}
for t in TICKERS:
    try: all_p[t] = load_p(t)
    except: pass
print(f"候选池: {len(all_p)}只")

def run(pp, sd, ed, md=20, tn=5, hd=20):
    si_s = bisect_right(SPY_DATES, sd) - 1
    si_e = bisect_right(SPY_DATES, ed) - 1
    ss = si_s + max(md, 25)
    picks = []
    for si in range(ss, si_e-hd, hd):
        dp=SPY_DATES[si-md]; db=SPY_DATES[si]; ds=SPY_DATES[min(si+hd,si_e)]
        cand=[]
        for t,p in all_p.items():
            vb=p.get(db); vp=p.get(dp)
            if not vb or not vp or vp<=1: continue
            mom=(vb/vp-1)*100
            if pp==0: sc=mom
            else:
                hp=vb
                for ii in range(max(0,si-252),si+1):
                    pv=p.get(SPY_DATES[ii])
                    if pv and pv>hp: hp=pv
                p52=vb/hp*100 if hp>0 else 100
                sc=mom*(1-max(0,(p52-50)/50*pp))
            cand.append((sc,t))
        if len(cand)<tn: continue
        cand.sort(key=lambda x:x[0],reverse=True)
        period_rets=[]
        for _,t in cand[:tn]:
            vb=all_p[t].get(db); vs=all_p[t].get(ds)
            if vb and vs and vb>1: period_rets.append((vs/vb-1)*100)
        if period_rets: picks.append(np.mean(period_rets))
    return picks

def compound(rets):
    cum=1
    for r in rets: cum*=(1+r/100)
    return (cum-1)*100

# 只测关键参数组合
CONFIGS = [
    ("V3基准", 0, 20, 5, 20),
    ("V4扣50%", 0.5, 20, 5, 20),
    ("V4扣70%", 0.7, 20, 5, 20),
    ("V4扣70_25d", 0.7, 25, 5, 20),
    ("V4扣70_30d", 0.7, 30, 5, 20),
    ("V4扣90%", 0.9, 20, 5, 20),
]

# 训练期: 2014-2019
train_sd="2014-01-02"; train_ed="2019-12-31"
# 验证期: 2020-2025
test_sd="2020-01-02"; test_ed="2025-12-31"

print(f"\n{'='*70}")
print("样本外验证 | 训练(2014-2019) → 验证(2020-2025)")
print(f"{'='*70}")

print(f"\n{'策略':>20s} {'训练收益':>10s} {'验证收益':>10s} {'超额vsV3':>10s}")
print("-"*55)

results = []
for name, pp, md, tn, hd in CONFIGS:
    tr = run(pp,train_sd,train_ed,md,tn,hd)
    train_val = compound(tr) if tr else 0
    vr = run(pp,test_sd,test_ed,md,tn,hd)
    test_val = compound(vr) if vr else 0
    results.append((name, train_val, test_val))

# V3基准
v3_train, v3_test = results[0][1], results[0][2]

for name, train_val, test_val in results:
    excess = test_val - v3_test
    if name == "V3基准":
        print(f"{name:>20s} {train_val:>+9.1f}% {test_val:>+9.1f}% {'--':>10s}")
    else:
        mark = "✅" if excess > 0 else "❌"
        print(f"{name:>20s} {train_val:>+9.1f}% {test_val:>+9.1f}% {excess:>+9.1f}% {mark}")

print(f"\n{'='*70}")
print("验证结论:")
best = max(results[1:], key=lambda x: x[2])
if best[2] > v3_test:
    print(f"  ✅ 验证通过 — {best[0]}在验证期跑赢V3 (+{best[2]-v3_test:.1f}%)")
    print(f"  🏆 推荐: {best[0]}")
else:
    print(f"  ❌ 验证未通过")
print(f"  V3基准验证期收益: {v3_test:+.1f}%")
PYEOF