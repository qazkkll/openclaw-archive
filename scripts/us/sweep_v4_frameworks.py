#!/usr/bin/env python3
"""
V4 美股 · 五种框架暴力扫描
不设剔除逻辑，用比例扣分替代硬排除
"""

import json, itertools, numpy as np, os
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
        # RSI
        if i >= 14:
            gains = sum(max(0, float(raw[j]['close'])-float(raw[j-1]['close'])) for j in range(i-13,i+1))
            losses = sum(max(0, float(raw[j-1]['close'])-float(raw[j]['close'])) for j in range(i-13,i+1))
            rsi = 100-100/(1+gains/14/(losses/14+0.001)) if losses > 0 else 100
        else: rsi = 50
        # ADX简化版
        if i >= 28:
            tr = max(float(raw[j]['close'])-float(raw[j]['close']), 0)
            adx = 20  # placeholder
        else: adx = 20
        result[d] = {'p':pr,'a20':a20,'a50':a50,'p52':p52,'rsi':rsi,'adx':adx}
    return result

print("加载数据...")
all_data = {}
for t in TICKERS:
    try:
        raw = load_raw(t)
        if len(raw) > 200: all_data[t] = calc_metrics(raw)
    except: pass
print(f"  {len(all_data)} 只")

# ========== 定义5种框架 ==========
FRAMEWORKS = []

# 框架A: 纯动量（基准）
# 直接用动量排序，不设任何过滤/扣分
# 参数: momentum_days, top_n, hold_days
FRAMEWORKS.append({
    'name': 'A-纯动量',
    'params': {'momentum_days': [10,15,20,25,30], 'top_n': [3,5,8], 'hold_days': [10,15,20,25,30]},
    'desc': '纯动量排序选股，无过滤无扣分'
})

# 框架B: 动量+比例扣分（52周位置）
# adjusted_score = momentum × (1 - max(0, (p52-50)/50 × penalty))
# penalty参数: [0.3, 0.5, 0.7]
FRAMEWORKS.append({
    'name': 'B-比例扣分',
    'params': {'momentum_days': [10,15,20,25,30], 'top_n': [3,5,8], 'hold_days': [10,15,20,25,30],
               'penalty_pct': [0.3, 0.5, 0.7]},
    'desc': '按52周位置比例扣减动量分：adjusted=m×(1-max(0,p52-50)/50×penalty)'
})

# 框架C: 硬过滤
# 剔除p52 > threshold 的股票，剩下的按动量排序
# threshold参数: [80, 85, 90, 95, 100]
FRAMEWORKS.append({
    'name': 'C-硬过滤',
    'params': {'momentum_days': [10,15,20,25,30], 'top_n': [3,5,8], 'hold_days': [10,15,20,25,30],
               'threshold': [80, 85, 90, 95, 100]},
    'desc': '剔除52周位置大于threshold的股票'
})

# 框架D: 全评分模式（V2逆向防守适用于所有市场）
# score = (momentum_norm × w_m) + (p52_score × w_p) + (rsi_score × w_r)
# 权重组合: 多种
FRAMEWORKS.append({
    'name': 'D-全评分',
    'params': {'momentum_days': [10,15,20,25,30], 'top_n': [3,5,8], 'hold_days': [10,15,20,25,30],
               'weight_m': [30, 40, 50], 'weight_p': [30, 40, 50], 'weight_r': [10, 20, 30]},
    'desc': '综合评分：动量(标准化)+52周位评分+RSI评分'
})

# 框架E: 动量+扣分+RSI过热惩罚
# adjusted = momentum × (1 - 52周扣分率) - RSI超买惩罚
FRAMEWORKS.append({
    'name': 'E-双扣分',
    'params': {'momentum_days': [10,15,20,25,30], 'top_n': [3,5,8], 'hold_days': [10,15,20,25,30],
               'penalty_52w': [0.3, 0.5, 0.7], 'rsi_penalty': [0, 0.2, 0.4]},
    'desc': '动量扣分(52周) + RSI超买惩罚(rsi>70时额外扣分)'
})

def score_framework(fw_name, data, md, tn, hd, sd, ed, **extra):
    """在指定区间运行框架并返回策略收益"""
    si_s = next(i for i,d in enumerate(SPY_DATES) if d >= sd)
    si_e = next(i for i,d in enumerate(SPY_DATES) if d >= ed)
    ss = si_s + max(md, 25)
    
    rets = []
    for si in range(ss, si_e - hd, hd):
        d_pr = SPY_DATES[si - md]; d_by = SPY_DATES[si]; d_sl = SPY_DATES[min(si+hd, si_e)]
        
        candidates = []
        for t, td in data.items():
            v_b = td.get(d_by); v_p = td.get(d_pr)
            if not v_b or not v_p or v_b['p'] < 1: continue
            
            mom = (v_b['p']/v_p['p']-1)*100
            p52 = v_b['p52']
            
            if fw_name == 'A-纯动量':
                score = mom
            
            elif fw_name == 'B-比例扣分':
                penalty = extra.get('penalty_pct', 0.5)
                adj = 1 - max(0, (p52-50)/50 * penalty)
                score = mom * adj
            
            elif fw_name == 'C-硬过滤':
                threshold = extra.get('threshold', 95)
                if p52 > threshold: continue
                score = mom
            
            elif fw_name == 'D-全评分':
                w_m = extra.get('weight_m', 40) / 100
                w_p = extra.get('weight_p', 40) / 100
                w_r = extra.get('weight_r', 20) / 100
                # 标准化动量(0-100)
                mom_norm = max(0, min(100, 50 + mom))
                # 52周位评分(越低越好)
                p52_score = max(0, 100 - p52)
                # RSI评分(中性偏好)
                rsi = v_b.get('rsi', 50)
                rsi_score = 100 - abs(rsi - 50) * 2  # rsi=50时100分
                score = mom_norm*w_m + p52_score*w_p + rsi_score*w_r
            
            elif fw_name == 'E-双扣分':
                penalty_52w = extra.get('penalty_52w', 0.5)
                rsi_penalty = extra.get('rsi_penalty', 0.2)
                p52_adj = max(0, (p52-50)/50 * penalty_52w)
                rsi = v_b.get('rsi', 50)
                rsi_adj = max(0, (rsi-70)/30 * rsi_penalty) if rsi > 70 else 0
                score = mom * (1 - p52_adj) - mom * rsi_adj
            
            candidates.append((score, t))
        
        if len(candidates) < tn: continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:tn]
        
        pd_rets = []
        for _, t in top:
            v_b = data[t].get(d_by); v_s = data[t].get(d_sl)
            if v_b and v_s and v_b['p'] > 1:
                pd_rets.append((v_s['p']/v_b['p']-1)*100)
        
        if pd_rets:
            rets.append(np.mean(pd_rets))
    
    return rets

# ========== 暴力扫描 ==========
print("\n开始暴力扫描（5框架 × 参数组合）...")
sd = "2020-01-02"; ed = "2025-12-31"  # 用完整数据的区间

all_results = {}

for fw in FRAMEWORKS:
    name = fw['name']
    keys = list(fw['params'].keys())
    total_c = 1
    for v in fw['params'].values(): total_c *= len(v)
    
    print(f"\n{'='*60}")
    print(f"框架 {name}: {fw['desc']}")
    print(f"  参数组合: {total_c}")
    print(f"{'='*60}")
    
    fw_results = []
    
    for idx, values in enumerate(itertools.product(*fw['params'].values())):
        params = dict(zip(keys, values))
        
        rets = score_framework(name, all_data, params.get('momentum_days',20),
                              params.get('top_n',5), params.get('hold_days',20),
                              sd, ed, **{k:v for k,v in params.items() if k not in ['momentum_days','top_n','hold_days']})
        
        if rets:
            total = sum(rets)
            avg = np.mean(rets)
            wr = 100*sum(1 for r in rets if r>0)/len(rets)
            worst = min(rets)
            best = max(rets)
            fw_results.append((total, avg, wr, worst, best, params))
        
        if (idx+1) % 50 == 0:
            print(f"  {idx+1}/{total_c}", flush=True)
    
    fw_results.sort(key=lambda x: x[0], reverse=True)
    all_results[name] = fw_results
    
    # 输出TOP5
    print(f"\n🏆 {name} TOP 5:")
    print(f"{'#':>3s} {'总收益':>8s} {'均/期':>7s} {'胜率':>5s} {'最差':>7s} {'最好':>7s} 参数")
    print("-" * 70)
    for i, r in enumerate(fw_results[:5]):
        params_str = " ".join(f"{k}={v}" for k,v in r[5].items())
        print(f"{i+1:3d} {r[0]:>+8.1f}% {r[1]:>+7.2f}% {r[2]:>4.1f}% {r[3]:>+7.1f}% {r[4]:>+7.1f}%  {params_str}")

# 保存全部结果
cache = {}
for name, results in all_results.items():
    cache[name] = [(r[0],r[1],r[2],r[3],r[4],r[5]) for r in results[:20]]
json.dump(cache, open(f"{CACHE_DIR}/fw_comparison.json", 'w'))

print("\n✅ 扫描完成")
