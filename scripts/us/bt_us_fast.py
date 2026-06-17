#!/usr/bin/env python3
"""美股策略速测"""
import json, numpy as np, random, sys

with open('data/backtest_hist_yahoo.json') as f:
    raw = json.load(f)

codes = list(raw.keys())[:30]
data = {}
for code in codes:
    item = raw[code]
    if isinstance(item, dict):
        c = item.get('close',[])
        d = item.get('dates',[])
        s = {}
        for i,dt in enumerate(d):
            if i < len(c): s[dt] = c[i]
        if s: data[code] = s

print(f'测试 {len(data)} 只美股')
random.seed(42)

# SMA金叉
for fd in [5,10,20]:
    sig = []
    for code in data:
        c = list(data[code].values())
        for i in range(60,len(c)-fd,5):
            # 20/50 SMA金叉简化
            if i>=50:
                ma20 = np.mean(c[i-19:i+1])
                ma50 = np.mean(c[i-49:i+1])
                ma20_1 = np.mean(c[i-20:i])
                ma50_1 = np.mean(c[i-50:i])
                if ma20_1 <= ma50_1 and ma20 > ma50:
                    sig.append((c[i+fd]/c[i]-1)*100)
    if sig:
        print(f'SMA金叉 {fd}日: 均{np.mean(sig):+.2f}% 胜{sum(1 for x in sig if x>0)/len(sig)*100:.0f}% ({len(sig)}次)')

# RSI超卖
for fd in [5,10,20]:
    sig = []
    for code in data:
        c = list(data[code].values())
        for i in range(20,len(c)-fd,5):
            gains=[max(c[j]-c[j-1],0) for j in range(i-13,i+1)]
            losses=[max(-(c[j]-c[j-1]),0) for j in range(i-13,i+1)]
            ag=sum(gains)/14; al=sum(losses)/14
            rsi=100-100/(1+ag/al) if al>0 else 100
            if rsi < 30:
                sig.append((c[i+fd]/c[i]-1)*100)
    if sig:
        print(f'RSI超卖 {fd}日: 均{np.mean(sig):+.2f}% 胜{sum(1 for x in sig if x>0)/len(sig)*100:.0f}% ({len(sig)}次)')

# 随机
print('\n随机基准:')
for fd in [5,10,20]:
    r=[]
    for code in data:
        c=list(data[code].values())
        for _ in range(20):
            i=random.randint(60,len(c)-fd-1)
            r.append((c[i+fd]/c[i]-1)*100)
    print(f'  {fd}日: 均{np.mean(r):+.2f}% 胜{sum(1 for x in r if x>0)/len(r)*100:.0f}%')
