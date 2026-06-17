#!/usr/bin/env python3
"""美股信号质量测试"""
import json, numpy as np, random

with open('data/backtest_hist_yahoo.json') as f:
    raw = json.load(f)

all_data = {}
for code in list(raw.keys())[:200]:
    item = raw[code]
    if isinstance(item, dict):
        closes = item.get('close',[])
        dates = item.get('dates',[])
        s = {}
        for i,d in enumerate(dates):
            if i < len(closes):
                s[d] = closes[i]
        if s: all_data[code] = s

print(f'美股: {len(all_data)}只')

def sma(arr, p):
    if len(arr) < p: return [None]*len(arr)
    res = [None]*(p-1)
    for i in range(p-1, len(arr)):
        res.append(float(np.mean(arr[i-p+1:i+1])))
    return res

random.seed(42)

for fd in [5, 10, 20, 60]:
    turtle = []
    ma60 = []
    rlist = []
    
    for code in all_data:
        closes = list(all_data[code].values())
        ma = sma(closes, 60)
        
        # 海龟60日新高
        for i in range(60, len(closes)-fd, 10):
            if closes[i] >= max(closes[i-59:i+1]):
                if i+fd < len(closes):
                    turtle.append((closes[i+fd]/closes[i]-1)*100)
        
        # 站上MA60
        for i in range(120, len(closes)-fd, 10):
            if ma[i] and closes[i] > ma[i]:
                if i+fd < len(closes):
                    ma60.append((closes[i+fd]/closes[i]-1)*100)
        
        # 随机
        for _ in range(30):
            i = random.randint(120, len(closes)-fd-1)
            if i+fd < len(closes):
                rlist.append((closes[i+fd]/closes[i]-1)*100)
    
    print(f'\n=== {fd}日后 ===')
    for name, d in [('海龟60日高', turtle), ('站上MA60突破', ma60), ('随机基准', rlist)]:
        if d:
            avg = np.mean(d); pos = sum(1 for x in d if x>0)/len(d)*100
            print(f'  {name}: 均{avg:+.2f}% 胜{pos:.0f}% ({len(d)}次)')

print('\n策略原理:')
print('海龟60日新高 = 价格创60日最高收盘价 → 趋势确认突破信号')
print('站上MA60突破 = 收盘价突破60日均线 → 中期趋势转多信号')
print('InStock使用这些信号筛选候选股，再结合成交量/形态/基本面人工判断')
