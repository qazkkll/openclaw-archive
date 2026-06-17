#!/usr/bin/env python3
"""A股权重资金回测 — 100万模拟 + 256种权重搜索"""
import json, math, time, itertools
from collections import OrderedDict

with open('data/test_50_hist.json') as f:
    hist = json.load(f)
with open('data/test_stocks_50.json') as f:
    stocks = json.load(f)

def ema(arr, n):
    k=2/(n+1); r=[arr[0]]
    for v in arr[1:]: r.append(v*k+r[-1]*(1-k))
    return r

def sma(arr, n):
    return [None]*(n-1)+[sum(arr[i-n+1:i+1])/n for i in range(n-1,len(arr))]

def calc_all(closes, highs, lows):
    n=len(closes)
    ma5,ma20,ma60=sma(closes,5),sma(closes,20),sma(closes,60)
    g,l=[],[]
    for i in range(1,n):
        d=closes[i]-closes[i-1]; g.append(max(d,0)); l.append(max(-d,0))
    rsi=[None]*14; ag,al=sum(g[:14])/14,sum(l[:14])/14
    for i in range(14,n):
        rsi.append(100-100/(1+ag/al) if al else 100)
        if i<len(g): ag=(ag*13+g[i])/14; al=(al*13+l[i])/14
    e12,e26=ema(closes,12),ema(closes,26)
    macd=[e12[i]-e26[i] for i in range(n)]
    sig=ema(macd,9)
    hist_m=[macd[i]-sig[i] for i in range(n)]
    pos52=[None]*252
    for i in range(252,n):
        lo,hi=min(closes[i-251:i+1]),max(closes[i-251:i+1])
        pos52.append((closes[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'close':closes,'ma5':ma5,'ma20':ma20,'ma60':ma60,'rsi':rsi,'macd_hist':hist_m,'pos52':pos52}

def sf(arr,i):
    return arr[i] if 0<=i<len(arr) and arr[i] is not None else None

def calc_score(i, ind, w):
    """评分函数 v6 (非线性变换 + 自定义权重 w=[w_macd,w_52w,w_ma,w_adx,w_rsi])"""
    s=0
    h=sf(ind['macd_hist'],i); hp=sf(ind['macd_hist'],i-1)
    if h and hp and h>0 and hp<=0:
        confirmed = i>2 and sf(ind['macd_hist'],i) and sf(ind['macd_hist'],i-1) and sf(ind['macd_hist'],i-2) and sf(ind['macd_hist'],i) > sf(ind['macd_hist'],i-1) > sf(ind['macd_hist'],i-2)
        s += w[0] * (1.0 if confirmed else 0.8)
    elif h and hp and h>hp and h>0: s += w[0]*0.6
    elif h and h>0: s += w[0]*0.25
    else: s -= w[0]*0.1
    p=sf(ind['pos52'],i)
    if p: s += w[1] * (1 - math.sqrt(min(p,100)/100))  # sqrt
    price=sf(ind['close'],i); m20=sf(ind['ma20'],i); m5=sf(ind['ma5'],i)
    if price and m20 and price>m20: s+=w[2]*0.35
    if m5 and m20 and m5>m20: s+=w[2]*0.35
    s+=w[2]*0.3
    s+=w[3]*0.5  # ADX est
    r=sf(ind['rsi'],i)
    if r:
        if r<50: s+=w[4]/(1+((r-25)/15)**2)  # hyperbolic
        else: s+=w[4]/(1+((75-r)/15)**2)
    return s

# Calculate indicators for all 50 stocks
print("计算技术指标...")
ind_data = {}
for code, d in hist.items():
    if len(d['close']) < 300: continue
    ind = calc_all(d['close'], d['high'], d['low'])
    ind_data[code] = ind

# Align to common date range (use shortest stock)
min_len = min(len(d['close']) for d in ind_data.values())
print(f"对齐到 {min_len} 天")

# Use last 300 days for backtest
test_start = min_len - 300
test_end = min_len

# ===== 资金模拟回测 =====
def run_capital_backtest(ind_data, w, buy_pct=60, sell_pct=35, max_pos=5, pos_size=0.15):
    """
    资金模拟: 100万启动
    固定仓位制: 每只投入=初始资本×pos_size, 不滚动放大
    score>=buy_pct→买入, score<=sell_pct或破MA20+MACD负→卖出
    """
    cash = 1000000.0
    FIXED_POS = 1000000.0 * pos_size  # 每只固定投入, 如15万
    positions = {}
    trades = []
    
    for i in range(test_start, test_end):
        # Sell
        to_sell = []
        for code, pos in list(positions.items()):
            ind = ind_data.get(code)
            if not ind: continue
            score = calc_score(i, ind, w)
            price = sf(ind['close'], i)
            ma20 = sf(ind['ma20'], i)
            macd = sf(ind['macd_hist'], i)
            if (score < sell_pct) or (price and ma20 and price < ma20 and macd and macd < 0):
                if price and price > 0:
                    pnl = (price - pos['entry_price']) / pos['entry_price']
                    cash += FIXED_POS * (1 + pnl)
                    trades.append(pnl * 100)
                    del positions[code]
        
        # Buy
        slots = max_pos - len(positions)
        budget = int(cash / FIXED_POS)
        can_buy = min(budget, slots)
        if can_buy > 0:
            candidates = []
            for code in ind_data:
                if code in positions: continue
                score = calc_score(i, ind_data[code], w)
                if score >= buy_pct:
                    price = sf(ind_data[code]['close'], i)
                    if price and price > 0:
                        candidates.append((code, score, price))
            candidates.sort(key=lambda x: x[1], reverse=True)
            for code, score, price in candidates[:can_buy]:
                if code in positions: continue
                positions[code] = {'entry_price': price}
                cash -= FIXED_POS
    
    # Close last positions
    for code, pos in list(positions.items()):
        ind = ind_data.get(code)
        if ind:
            price = sf(ind['close'], test_end-1)
            if price and price > 0:
                pnl = (price - pos['entry_price']) / pos['entry_price']
                cash += FIXED_POS * (1 + pnl)
                trades.append(pnl * 100)
    
    total_ret = (cash - 1000000) / 1000000 * 100
    wins = [t for t in trades if t > 0]
    return {'ret': round(total_ret,2), 'wr': round(len(wins)/len(trades)*100,1) if trades else 0, 'trades': len(trades)}
    
    # Close remaining positions at end
    for code, pos in positions.items():
        ind = ind_data.get(code)
        if ind:
            price = sf(ind['close'], test_end-1)
            if price and price > 0:
                ret = (price - pos['entry_capital']/pos['cost_shares']) / (pos['entry_capital']/pos['cost_shares']) * 100
                capital += pos['entry_capital'] * (1 + ret/100)
                trades.append(ret)
            else:
                capital += pos['entry_capital']
        else:
            capital += pos['entry_capital']
    
    total_ret = (capital - 1000000) / 1000000 * 100
    wins = [t for t in trades if t > 0]
    return {'ret': round(total_ret,2), 'wr': round(len(wins)/len(trades)*100,1) if trades else 0, 'trades': len(trades)}

# ===== 权重网格搜索 =====
print("\n权重网格搜索 (256种组合)...")
print(f"{'MACD':>5} {'52W':>5} {'MA':>5} {'ADX':>5} {'RSI':>5} {'收益%':>7} {'胜率':>5}")
print("-"*45)

results = []
combo = 0

for w_macd in [15,20,25,30]:
    for w_52w in [15,20,25,30]:
        for w_ma in [15,20,25]:
            for w_adx in [10,15,20]:
                for w_rsi in [10,15,20]:
                    if w_macd+w_52w+w_ma+w_adx+w_rsi != 100: continue
                    combo += 1
                    w = [w_macd, w_52w, w_ma, w_adx, w_rsi]
                    r = run_capital_backtest(ind_data, w)
                    results.append({'w': w, 'ret': r['ret'], 'wr': r['wr'], 'trades': r['trades']})

results.sort(key=lambda x: x['ret'], reverse=True)

print(f"\nTop 10:")
for r in results[:10]:
    print(f"{r['w'][0]:5d} {r['w'][1]:5d} {r['w'][2]:5d} {r['w'][3]:5d} {r['w'][4]:5d} {r['ret']:>+7.2f}% {r['wr']:5.1f}%")

# v5.1 uniform benchmark
v51_result = run_capital_backtest(ind_data, [20,20,20,20,20])
print(f"\n--- v5.1均匀[20,20,20,20,20]: {v51_result['ret']:+.2f}% | 胜率{v51_result['wr']:.1f}%")

# Output best weight
best = results[0]['w'] if results else [20,20,20,20,20]
print(f"最优权重: [{best[0]},{best[1]},{best[2]},{best[3]},{best[4]}]")

# Also test different buy/sell thresholds on best weight
print(f"\n--- 最优权重不同门槛测试 ---")
for buy in [50,55,60,65]:
    for sell in [30,35,40]:
        r = run_capital_backtest(ind_data, best, buy_pct=buy, sell_pct=sell)
        print(f"  买入≥{buy} 卖出≤{sell}: {r['ret']:+.2f}% | 胜率{r['wr']:.1f}% | {r['trades']}笔")

# Save
output = {
    'stocks_tested': 50,
    'total_combinations': combo,
    'v51_benchmark': v51_result,
    'best_weight': best,
    'top10': results[:10],
    'top1_detail': results[0] if results else None
}
with open('data/weight_search_final.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✅ 已保存到 data/weight_search_final.json")
