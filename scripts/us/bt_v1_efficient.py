#!/usr/bin/env python3
"""
V1效率回测 — 两步走:
1) 逐只股票时间顺序算分 → 缓存 (无前视偏差)
2) 用缓存评分跑回测 (快)
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from score_engine import compute_indicators, v1_score_from_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = f'{ROOT}/data/v1_scores_cache.json'

t0 = time.time()

# Load data
with open(f'{ROOT}/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)

print('构建数据结构...', flush=True)
ALL_DATA = {}
for code, item in YAHOO.items():
    if not isinstance(item, dict):
        continue
    dates = item.get('dates', [])
    closes = item.get('close', [])
    highs = item.get('high', [])
    lows = item.get('low', [])
    opens = item.get('open', [])
    sd = {}
    for i, d in enumerate(dates):
        if i < len(closes):
            sd[d] = {'c': closes[i], 'h': highs[i] if i < len(highs) else closes[i],
                     'l': lows[i] if i < len(lows) else closes[i],
                     'o': opens[i] if i < len(opens) else closes[i]}
    if sd:
        ALL_DATA[code] = sd

all_dates = sorted(set(d for c in ALL_DATA for d in ALL_DATA[c].keys()))
print(f'{len(ALL_DATA)}只, {len(all_dates)}交易日, {time.time()-t0:.0f}s', flush=True)

# Step 1: Pre-compute scores for each stock (time-ordered, no look-ahead)
if not os.path.exists(CACHE_FILE):
    print('Step 1: 预计算评分 (逐只逐天, 无前视偏差)...', flush=True)
    SCORE_CACHE = {}
    codes = list(ALL_DATA.keys())
    
    for ci, code in enumerate(codes[:400]):  # 400只(速度快+覆盖广)
        if (ci+1) % 50 == 0:
            print(f'  {ci+1}/{min(400,len(codes))} | mem:{__import__("psutil").Process().memory_percent():.0f}% | {time.time()-t0:.0f}s', flush=True)
        
        cd = ALL_DATA[code]
        stock_dates = sorted(cd.keys())
        stock_scores = {}
        
        for di in range(200, len(stock_dates)):
            d = stock_dates[di]
            # 只用截至今天的数据
            closes = [cd[stock_dates[j]]['c'] for j in range(max(0, di-200), di+1) if stock_dates[j] in cd]
            highs = [cd[stock_dates[j]]['h'] for j in range(max(0, di-200), di+1) if stock_dates[j] in cd]
            lows = [cd[stock_dates[j]]['l'] for j in range(max(0, di-200), di+1) if stock_dates[j] in cd]
            
            if len(closes) >= 60:
                try:
                    s = v1_score_from_data(closes, highs, lows)
                    if s and s > 0:
                        stock_scores[d] = round(s, 1)
                except:
                    pass
        
        if stock_scores:
            SCORE_CACHE[code] = stock_scores
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(SCORE_CACHE, f)
    print(f'评分缓存: {len(SCORE_CACHE)}只, {time.time()-t0:.0f}s', flush=True)
else:
    print('加载评分缓存...', flush=True)
    with open(CACHE_FILE) as f:
        SCORE_CACHE = json.load(f)
    print(f'缓存: {len(SCORE_CACHE)}只', flush=True)

# Step 2: Run backtest using cached scores
print()
print('Step 2: 回测...')
print('═' * 60)

def run_bt(buy_th, sell_th, max_pos=8, cost=0.003, name=''):
    cash = 100000.0
    pos = {}
    
    start_idx = all_dates.index(list(list(SCORE_CACHE.values())[0].keys())[0])
    
    for di in range(start_idx, len(all_dates) - 1):
        date = all_dates[di]
        next_date = all_dates[di + 1]
        
        # Get scores for this date from cache
        daily_scores = {}
        for code in SCORE_CACHE:
            s = SCORE_CACHE[code].get(date, 0)
            if s > 0:
                daily_scores[code] = s
        
        if not daily_scores:
            continue
        
        ranked = sorted(daily_scores.items(), key=lambda x: -x[1])
        
        # 卖出
        for c in list(pos.keys()):
            if daily_scores.get(c, 0) < sell_th:
                h = pos.pop(c)
                sp = ALL_DATA.get(c, {}).get(next_date, {}).get('o', 0)
                if sp > 0:
                    cash += h['shares'] * sp * (1 - cost)
        
        # 买入
        if len(pos) < max_pos:
            cand = [(c, s) for c, s in ranked[:30] if s >= buy_th and c not in pos]
            for code, score in cand[:max_pos - len(pos)]:
                bp = ALL_DATA.get(code, {}).get(next_date, {}).get('o', 0)
                if bp <= 0: continue
                invest = cash * (1.0/max_pos) * 0.95
                shares = invest / bp
                if shares > 0:
                    cash -= invest
                    pos[code] = {'shares': shares, 'buy_p': bp}
    
    # 终值
    final = cash
    for c, h in pos.items():
        p = ALL_DATA.get(c, {}).get(all_dates[-1], {}).get('c', 0)
        if p > 0: final += h['shares'] * p
    
    ret = (final/100000 - 1)*100
    yrs = (len(all_dates)-start_idx)/245
    ann = ((final/100000)**(1/max(yrs,1))-1)*100 if final > 0 else -100
    print(f'  {name:<20} {ret:>+7.1f}%  {ann:>+6.1f}%')
    return ret

run_bt(62, 50, 8, 0.003, 'V1_62/50')
run_bt(55, 40, 8, 0.003, 'V1_55/40')
run_bt(65, 45, 8, 0.003, 'V1_65/45')
run_bt(62, 35, 8, 0.003, 'V1_62/35')

print(f'\n总耗时: {time.time()-t0:.0f}s')
