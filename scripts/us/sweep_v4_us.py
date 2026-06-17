#!/usr/bin/env python3
"""V4 美股 · 系数暴力扫描（基于缓存）"""
import json, itertools, numpy as np
from bisect import bisect_right

CACHE = json.load(open("/home/admin/.openclaw/workspace/data/us_pc.json"))
DATA = CACHE['data']
TICKERS = CACHE['tickers']
SPY_DATES = CACHE['spy_dates']

def gv(ticker, d):
    """获取某只股票某天的预计算数据"""
    td = DATA.get(ticker, {})
    # 找到该日期或之前最近的交易日
    keys = sorted(td.keys())
    i = bisect_right(keys, d) - 1
    return td.get(keys[i]) if i >= 0 else None

# 参数空间
PARAMS = {
    'momentum_days': [10, 15, 20, 25, 30],
    'max_52w': [85, 90, 95, 100],
    'top_n': [3, 5, 8],
    'hold_days': [10, 15, 20, 25, 30],
}

total_combo = 1
for v in PARAMS.values(): total_combo *= len(v)

print(f"参数空间: {total_combo}组合")
print("正在扫描...")

results = []

for idx, values in enumerate(itertools.product(*PARAMS.values())):
    md, m52, tn, hd = values
    
    strategy_rets = []
    start_step = md if md > 20 else 20
    
    for si in range(start_step, len(SPY_DATES) - hd, hd):
        d_prev = SPY_DATES[si - md]
        d_buy = SPY_DATES[si]
        d_sell = SPY_DATES[min(si + hd, len(SPY_DATES) - 1)]
        
        # 计算动量
        mom_list = []
        for t in TICKERS:
            vb = gv(t, d_buy)
            vp = gv(t, d_prev)
            if vb and vp and vb['p'] > 1 and vp['p'] > 1:
                # 选择正确的动量周期
                if md == 10: mom_val = vb['m10']
                elif md == 20: mom_val = vb['m20']
                elif md == 30: mom_val = vb['m30']
                else: mom_val = (vb['p'] / vp['p'] - 1) * 100
                mom_list.append((t, mom_val, vb['p52']))
        
        if len(mom_list) < tn: continue
        mom_list.sort(key=lambda x: x[1], reverse=True)
        
        # 52周过滤
        filtered = [x for x in mom_list if x[2] < m52]
        if len(filtered) < tn:
            filtered = mom_list[:tn]
        
        top = filtered[:tn]
        
        pd_rets = []
        for t, _, _ in top:
            v_b = gv(t, d_buy)
            v_s = gv(t, d_sell)
            if v_b and v_s and v_b['p'] > 1:
                pd_rets.append((v_s['p'] / v_b['p'] - 1) * 100)
        
        if pd_rets:
            strategy_rets.append(np.mean(pd_rets))
    
    if strategy_rets:
        total_ret = sum(strategy_rets)
        avg_ret = np.mean(strategy_rets)
        wr = sum(1 for r in strategy_rets if r > 0) / len(strategy_rets) * 100
        worst = min(strategy_rets)
        results.append((total_ret, avg_ret, wr, worst, md, m52, tn, hd))
    
    if (idx + 1) % 50 == 0:
        print(f"  {idx+1}/{total_combo}...", flush=True)

# 排序
results.sort(key=lambda x: x[0], reverse=True)

print("\n" + "=" * 80)
print("🏆 TOP 15 (按总收益)")
print("=" * 80)
print(f"{'#':>3s} {'总收益':>7s} {'均/期':>7s} {'胜率':>5s} {'最差':>6s} 参数")
print("-" * 55)
for i, (tr, ar, wr, wd, md, m52, tn, hd) in enumerate(results[:15]):
    print(f"{i+1:3d} {tr:>+7.1f}% {ar:>+7.2f}% {wr:>4.1f}% {wd:>+6.1f}%  {md:2d}日 {m52:3d}% {tn}只 {hd}天")

# 基准（纯动量）
base = None
for r in results:
    if r[4]==20 and r[5]==100 and r[6]==5 and r[7]==20:
        base = r
        break
if base:
    print(f"\n对比基准 (20日 100% 5只 20天): +{base[0]:.1f}%")
    print(f"最优: +{results[0][0]:.1f}% (较基准{results[0][0]-base[0]:+.1f}%)")

# 稳键型（最优胜率）
print("\n🏆 TOP 10 (按胜率)")
safe = sorted([r for r in results if r[3] > -15], key=lambda x: x[2], reverse=True)
for i, (tr, ar, wr, wd, md, m52, tn, hd) in enumerate(safe[:10]):
    print(f"{i+1:3d} 胜{wr:>4.1f}% +{tr:>+7.1f}% {md:2d}日 {m52:3d}% {tn}只 {hd}天")

print("\n✅ 完成")
