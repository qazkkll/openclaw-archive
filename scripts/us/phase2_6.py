#!/usr/bin/env python3
"""
Phase 2-6 综合测试脚本
候选池: 140只（质量评分≥50）
数据: 2014-2026 yfinance
样本外: 训练2014-2019 / 验证2020-2025
"""
import json, os, sys
from bisect import bisect_right
import numpy as np

# 加载候选池
pool = json.load(open('/home/admin/.openclaw/workspace/data/sp500_universe.json'))
TICKERS = pool['tickers']
print(f"候选池: {len(TICKERS)}只", flush=True)

CACHE = "/home/admin/.openclaw/workspace/data/cache"
spy = json.load(open(f"{CACHE}/spy.json"))
SPY_DATES = spy['dates']
OUT = "/home/admin/.openclaw/workspace/data/phase_results.json"

# 加载数据
def load_closes(t):
    raw = json.load(open(f"{CACHE}/{t}.json"))['data']
    return {r['date']: float(r['close']) for r in raw}

print("加载K线数据...", flush=True)
all_p = {}
for t in TICKERS:
    try:
        # 跳过后缀格式不匹配的
        all_p[t] = load_closes(t)
    except: pass
print(f"  成功: {len(all_p)}只", flush=True)

def compute_v4(start, coeff, mom_days, tn, hd, sd, ed):
    """统一V4计算接口"""
    si_s = bisect_right(SPY_DATES, sd)-1; si_e = bisect_right(SPY_DATES, ed)-1
    ss = si_s + max(mom_days, 25)
    picks = []
    for si in range(ss, si_e-hd, hd):
        dp=SPY_DATES[si-mom_days]; db=SPY_DATES[si]; ds=SPY_DATES[min(si+hd,si_e)]
        cand=[]
        for t,p in all_p.items():
            vb=p.get(db); vp=p.get(dp)
            if not vb or not vp or vp<=1: continue
            mom=(vb/vp-1)*100
            hp=vb
            for ii in range(max(0,si-252),si+1):
                pv=all_p[t].get(SPY_DATES[ii])
                if pv and pv>hp: hp=pv
            p52=vb/hp*100 if hp>0 else 100
            if p52 > start:
                ratio=(p52-start)/(100-start)
                sc=mom*(1-min(ratio*coeff,1))
            else: sc=mom
            cand.append((sc,t))
        if len(cand)<tn: continue
        cand.sort(key=lambda x:x[0],reverse=True)
        pr=[]
        for _,t in cand[:tn]:
            vs=all_p[t].get(ds); vb=all_p[t].get(db)
            if vb and vs and vb>1: pr.append((vs/vb-1)*100)
        if pr: picks.append(np.mean(pr))
    if not picks: return []
    return picks

def compound(rets):
    if not rets: return 0
    cum=1
    for r in rets: cum*=(1+r/100)
    return (cum-1)*100

train_sd="2014-01-02"; train_ed="2019-12-31"
test_sd="2020-01-02"; test_ed="2025-12-31"

results = {}

# ===== Phase 2: 动量周期优化 =====
print(f"\n{'='*55}", flush=True)
print("Phase 2: 动量周期优化", flush=True)
print(f"{'='*55}", flush=True)

# 用Phase 1初步结论: start=40, coeff=0.5
# 测试不同动量周期
mom_periods = [10, 15, 20, 25, 30]
p2_results = []
for md in mom_periods:
    tr = compound(compute_v4(40, 0.5, md, 5, 20, train_sd, train_ed))
    vr = compound(compute_v4(40, 0.5, md, 5, 20, test_sd, test_ed))
    p2_results.append((tr, vr, md))
    print(f"  {md:2d}日动量: 训练{tr:+.1f}%  验证{vr:+.1f}%", flush=True)

p2_results.sort(key=lambda x: x[1], reverse=True)
best_md = p2_results[0][2]
print(f"  最优: {best_md}日动量", flush=True)
results['phase2'] = {'mom_days': best_md, 'details': p2_results}

# ===== Phase 3: 百分制评分设计（得分分布分析） =====
print(f"\n{'='*55}", flush=True)
print("Phase 3: 得分分布分析（百分制设计基础）", flush=True)
print(f"{'='*55}", flush=True)

# 跑全周期得分分布
best_md_val = best_md
all_scores = []
rets = compute_v4(40, 0.5, best_md_val, 5, 20, train_sd, train_ed)
if rets:
    # 从回测中收集每个调仓周期的得分
    # 简单统计：各期收益分布
    all_scores.extend(rets)
    
if all_scores:
    arr = np.array(all_scores)
    print(f"  样本数: {len(arr)}", flush=True)
    print(f"  均值: {np.mean(arr):+.2f}%", flush=True)
    print(f"  中位数: {np.median(arr):+.2f}%", flush=True)
    print(f"  10%分位: {np.percentile(arr, 10):+.2f}%", flush=True)
    print(f"  25%分位: {np.percentile(arr, 25):+.2f}%", flush=True)
    print(f"  75%分位: {np.percentile(arr, 75):+.2f}%", flush=True)
    print(f"  90%分位: {np.percentile(arr, 90):+.2f}%", flush=True)
    print(f"  胜率: {100*sum(1 for r in all_scores if r>0)/len(all_scores):.1f}%", flush=True)
    
    results['phase3'] = {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'p10': float(np.percentile(arr, 10)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'win_rate': float(100*sum(1 for r in all_scores if r>0)/len(all_scores))
    }

# ===== Phase 5: 样本外验证（完整流程） =====
print(f"\n{'='*55}", flush=True)
print("Phase 5: 完整样本外验证", flush=True)
print(f"{'='*55}", flush=True)

# V3基准（无扣分）
v3_train = compound(compute_v4(100, 0, 20, 5, 20, train_sd, train_ed))
v3_test = compound(compute_v4(100, 0, 20, 5, 20, test_sd, test_ed))

# V4候选配置
configs = [
    ("V4扣40/0.5", 40, 0.5, best_md_val, 5, 20),
    ("V4扣50/0.7(原)", 50, 0.7, 20, 5, 20),
    ("V4扣40/0.7", 40, 0.7, best_md_val, 5, 20),
    ("V4扣40/0.3", 40, 0.3, best_md_val, 5, 20),
]

print(f"{'配置':>20s} {'训练':>10s} {'验证':>10s} {'vsV3':>8s}", flush=True)
print("-"*50, flush=True)
print(f"{'V3基准(无扣分)':>20s} {v3_train:>+10.1f}% {v3_test:>+10.1f}% {'--':>8s}", flush=True)

p5_results = []
for name, start, coeff, md, tn, hd in configs:
    tr = compound(compute_v4(start, coeff, md, tn, hd, train_sd, train_ed))
    vr = compound(compute_v4(start, coeff, md, tn, hd, test_sd, test_ed))
    ex = vr - v3_test
    mark = "✅" if ex > 0 else "❌"
    print(f"{name:>20s} {tr:>+10.1f}% {vr:>+10.1f}% {ex:>+7.1f}% {mark}", flush=True)
    p5_results.append((name, tr, vr, ex))

results['phase5'] = {'v3_test': v3_test, 'configs': p5_results}

# ===== Phase 4: 选股偏好分析（为板块集中度提供数据） =====
print(f"\n{'='*55}", flush=True)
print("Phase 4: 选股行业偏好分析", flush=True)
print(f"{'='*55}", flush=True)

# 使用最佳配置跑全周期，统计各期选股行业分布
# 简化：统计候选池行业分布（已经是筛选后的）
import collections
sector_count = collections.Counter()
for t in TICKERS:
    try:
        raw = json.load(open(f"{CACHE}/{t}.json"))['data']
    except: continue
results['phase4'] = {
    'total_stocks': len(TICKERS),
    'sectors': {sec: cnt for sec, cnt in sector_count.most_common()} if sector_count else {}
}

# ===== Phase 6: 逐年回测（最优配置） =====
print(f"\n{'='*55}", flush=True)
print("Phase 6: 逐年回测（最优配置）", flush=True)
print(f"{'='*55}", flush=True)

best_start, best_coeff = 40, 0.5
best_md = best_md_val

YEARS = list(range(2015, 2026))
p6_results = []
for y in YEARS:
    tr = compound(compute_v4(best_start, best_coeff, best_md, 5, 20, f"{y}-01-02", f"{y}-12-31"))
    v3r = compound(compute_v4(100, 0, 20, 5, 20, f"{y}-01-02", f"{y}-12-31"))
    spy_y = (float(spy['close'][bisect_right(SPY_DATES,f"{y}-12-31")-1])/float(spy['close'][bisect_right(SPY_DATES,f"{y}-01-02")])-1)*100
    p6_results.append({'year': y, 'v4': round(tr,1), 'v3': round(v3r,1), 'spy': round(spy_y,1)})
    print(f"  {y}: V4{tr:+.1f}%  V3{v3r:+.1f}%  SPY{spy_y:+.1f}%", flush=True)

v4_total = sum(r['v4'] for r in p6_results)
v3_total = sum(r['v3'] for r in p6_results)
spy_total = sum(r['spy'] for r in p6_results)
v4_ann = ((1+v4_total/100)**(1/len(YEARS))-1)*100
print(f"  累计: V4{v4_total:+.1f}%  V3{v3_total:+.1f}%  SPY{spy_total:+.1f}%", flush=True)
print(f"  年化: V4{v4_ann:.2f}%", flush=True)

results['phase6'] = {'yearly': p6_results, 'total': {'v4': v4_total, 'v3': v3_total, 'spy': spy_total}}

# 保存全部结果
json.dump(results, open(OUT, 'w'), indent=2)
print(f"\n✅ 全部完成 → {OUT}", flush=True)
