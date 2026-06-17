#!/usr/bin/env python3
"""美股经典策略信号测试 — 轻量版"""
import json, numpy as np, random

with open('data/backtest_hist_yahoo.json') as f:
    raw = json.load(f)

all_data = {}
for code in list(raw.keys())[:200]:
    item = raw[code]
    if isinstance(item, dict):
        closes = item.get('close',[])
        s = {}
        for i,d in enumerate(item.get('dates',[])):
            if i < len(closes): s[d] = closes[i]
        if s: all_data[code] = s

def sma(arr,p):
    if len(arr)<p: return [None]*len(arr)
    r=[None]*(p-1)
    for i in range(p-1,len(arr)): r.append(float(np.mean(arr[i-p+1:i+1])))
    return r

random.seed(42)
strategies = {}

# 1. SMA Crossover (backtrader经典)
def sma_cross(closes, i, fast=20, slow=50):
    if i<slow: return False
    sma_f = sma(closes[:i+1], fast)
    sma_s = sma(closes[:i+1], slow)
    if sma_f[i] and sma_f[i-1] and sma_s[i] and sma_s[i-1]:
        return sma_f[i-1] <= sma_s[i-1] and sma_f[i] > sma_s[i]  # 金叉
    return False

# 2. RSI超卖反弹
def rsi_rebound(closes, i, period=14):
    if i<period: return False
    gains=[];losses=[]
    for j in range(1,period+1):
        d=closes[i-period+j]-closes[i-period+j-1]
        gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains)/period; al=sum(losses)/period
    rsi=100-100/(1+ag/al) if al>0 else 100
    return rsi < 30  # 超卖

# 3. Bollinger Band突破
def bollinger_breakout(closes, i, period=20):
    if i<period: return False
    window=closes[i-period:i+1]
    m=np.mean(window); s=np.std(window)
    return closes[i] > m + 2*s  # 突破上轨

strategies = {
    'SMA金叉(20/50)': sma_cross,
    'RSI超卖(<30)': rsi_rebound,
    '布林上轨突破': bollinger_breakout,
}

for name, fn in strategies.items():
    for fd in [5,10,20]:
        data=[]
        for code in list(all_data.keys())[:100]:
            closes=list(all_data[code].values())
            for i in range(60,len(closes)-fd,5):
                if fn(closes,i) and i+fd<len(closes):
                    data.append((closes[i+fd]/closes[i]-1)*100)
        avg=np.mean(data) if data else 0
        pos=sum(1 for x in data if x>0)/len(data)*100 if data else 0
        print(f'{name:20} {fd}日后: {avg:+.2f}% 胜率{pos:.0f}% ({len(data)}次)')
    print()

# 基准: 随机
print('随机基准:')
for fd in [5,10,20]:
    r=[]
    for code in list(all_data.keys())[:100]:
        closes=list(all_data[code].values())
        for _ in range(30):
            i=random.randint(60,len(closes)-fd-1)
            if i+fd<len(closes): r.append((closes[i+fd]/closes[i]-1)*100)
    avg=np.mean(r); pos=sum(1 for x in r if x>0)/len(r)*100
    print(f'  {fd}日后: {avg:+.2f}% 胜率{pos:.0f}%')
