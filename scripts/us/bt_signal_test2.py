#!/usr/bin/env python3
"""精简版信号质量测试"""
import json, numpy as np, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, 'data', 'backtest_hist_v3.json')) as f:
    raw = json.load(f)

# 只用10只股票
all_data = {}
for code in list(raw.keys())[:10]:
    item = raw[code]
    if isinstance(item, dict):
        closes = item.get('close',[])
        dates = item.get('dates',[])
        s = {}
        for i,d in enumerate(dates):
            if i < len(closes):
                s[d] = closes[i]
        if s: all_data[code] = s

print(f'测试 {len(all_data)} 只股票')

def sma(arr, p):
    if len(arr) < p: return [None]*len(arr)
    res = [None]*(p-1)
    for i in range(p-1, len(arr)):
        res.append(float(np.mean(arr[i-p+1:i+1])))
    return res

# 测试1: 站上MA60信号  
print('\n=== 站上MA60信号测试 ===')
total_buy = 0
total_returns_10d = []
for code in all_data:
    closes = list(all_data[code].values())
    dates = list(all_data[code].keys())
    ma60 = sma(closes, 60)
    
    for i in range(120, len(closes)-20, 5):
        if ma60[i] is not None and closes[i] > ma60[i]:
            # 信号触发
            total_buy += 1
            if i+10 < len(closes):
                ret = (closes[i+10]/closes[i] - 1) * 100
                total_returns_10d.append(ret)

if total_returns_10d:
    avg = np.mean(total_returns_10d)
    med = np.median(total_returns_10d)
    pos = sum(1 for x in total_returns_10d if x>0)/len(total_returns_10d)*100
    print(f'信号次数: {total_buy}')
    print(f'有效样本: {len(total_returns_10d)}')
    print(f'10日后平均收益: {avg:+.2f}%')
    print(f'10日胜率: {pos:.0f}%')

# 测试2: 海龟60日新高
print('\n=== 60日新高信号测试 ===')
turtle_returns = []
for code in all_data:
    closes = list(all_data[code].values())
    for i in range(60, len(closes)-20, 5):
        if closes[i] >= max(closes[i-59:i+1]):
            if i+10 < len(closes):
                ret = (closes[i+10]/closes[i] - 1) * 100
                turtle_returns.append(ret)

if turtle_returns:
    avg = np.mean(turtle_returns)
    pos = sum(1 for x in turtle_returns if x>0)/len(turtle_returns)*100
    print(f'信号次数: {len(turtle_returns)}')
    print(f'10日后平均收益: {avg:+.2f}%')
    print(f'10日胜率: {pos:.0f}%')

# 测试3: 随机基准
print('\n=== 随机基准 ===')
import random
random.seed(42)
rand_returns = []
for code in all_data:
    closes = list(all_data[code].values())
    for _ in range(50):
        i = random.randint(120, len(closes)-20)
        ret = (closes[i+10]/closes[i] - 1) * 100
        rand_returns.append(ret)

avg = np.mean(rand_returns)
pos = sum(1 for x in rand_returns if x>0)/len(rand_returns)*100
print(f'样本数: {len(rand_returns)}')
print(f'10日后平均收益: {avg:+.2f}%')
print(f'胜率: {pos:.0f}%')
print('\n✅ 测试完成')
