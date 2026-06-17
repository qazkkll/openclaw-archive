#!/usr/bin/env python3
"""信号质量测试：测策略信号触发后N天的平均收益"""
import json, numpy as np, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, 'data', 'backtest_hist_v3.json')) as f:
    raw = json.load(f)

codes = list(raw.keys())[:100]
all_data = {}
for code in codes:
    item = raw[code]
    if not isinstance(item, dict): continue
    closes = item.get('close',[])
    dates = item.get('dates',[])
    stock = {}
    for i,d in enumerate(dates):
        stock[d] = {'close':closes[i] if i<len(closes) else 0}
    if stock: all_data[code] = stock

def sma(arr, p):
    if len(arr)<p: return [None]*len(arr)
    return [None]*(p-1)+[float(np.mean(arr[i-p+1:i+1])) for i in range(p-1,len(arr))]

signals = {
    '60日新高': lambda c,i: i>=60 and c[i] >= max(c[i-59:i+1]),
    '站上MA60': lambda c,i: i>=60 and (m:=sma(c,60))[i] is not None and c[i] > m[i],
    'MA30持续向上': lambda c,i: i>=30 and (m:=sma(c,30))[i] is not None and m[i] > m[i-5] if i>=5 else False,
}

print(f'股票: {len(all_data)}只')
for name, fn in signals.items():
    fwd = [5,10,20]
    results = {fd:[] for fd in fwd}
    count = 0
    for code in list(all_data.keys())[:50]:
        cd = all_data[code]
        closes = [cd[d]['close'] for d in sorted(cd.keys())]
        for i in range(120, len(closes)-20, 5):
            if fn(closes, i):
                count += 1
                for fd in fwd:
                    if i+fd < len(closes):
                        results[fd].append((closes[i+fd]/closes[i]-1)*100)
    print(f'\n{name} ({count}次):')
    for fd in fwd:
        r = results[fd]
        if r:
            avg = np.mean(r); med = np.median(r); pos = sum(1 for x in r if x>0)/len(r)*100
            print(f'  {fd}天后: 平均{avg:+.2f}% 中位数{med:+.2f}% 胜率{pos:.0f}%')

# 基准: 随机买入
print(f'\n基准对比:')
all_returns = {fd:[] for fd in [5,10,20]}
import random
random.seed(42)
for code in list(all_data.keys())[:50]:
    closes = [all_data[code][d]['close'] for d in sorted(all_data[code].keys())]
    for _ in range(50):
        i = random.randint(120, len(closes)-20)
        for fd in [5,10,20]:
            all_returns[fd].append((closes[i+fd]/closes[i]-1)*100)
for fd in [5,10,20]:
    r = all_returns[fd]
    avg = np.mean(r); pos = sum(1 for x in r if x>0)/len(r)*100
    print(f'  随机买入{fd}天后: 平均{avg:+.2f}% 胜率{pos:.0f}%')
