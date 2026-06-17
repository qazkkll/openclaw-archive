#!/usr/bin/env python3
"""
V1正确回测 — 使用真实score_engine.py
数据: backtest_hist_yahoo.json (原始OHLCV)
交易: 当天算评分 → 次日开盘买卖
无前视偏差
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from score_engine import compute_indicators, v1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print('加载数据...', flush=True)
t0 = time.time()

with open(f'{ROOT}/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)

# Build ordered data
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
            sd[d] = {
                'c': closes[i], 'h': highs[i] if i < len(highs) else closes[i],
                'l': lows[i] if i < len(lows) else closes[i],
                'o': opens[i] if i < len(opens) else closes[i],
            }
    if sd:
        ALL_DATA[code] = sd

all_dates = sorted(set(d for c in ALL_DATA for d in ALL_DATA[c].keys()))
print(f'  {len(ALL_DATA)}只股票, {len(all_dates)}交易日, {time.time()-t0:.1f}s', flush=True)

def get_score(code, date_idx, days=200):
    """用真实V1引擎算评分(只用截至当天的数据)"""
    code_data = ALL_DATA.get(code)
    if not code_data:
        return 0
    
    # 获取过去days天的OHLCV
    closes, highs, lows = [], [], []
    for i in range(max(0, date_idx - days), date_idx + 1):
        d = all_dates[i]
        if d in code_data:
            row = code_data[d]
            closes.append(row['c'])
            highs.append(row['h'])
            lows.append(row['l'])
    
    if len(closes) < 60:
        return 0
    
    try:
        score = v1_score_from_data(closes, highs, lows)
        return score if score else 0
    except:
        return 0

# Import the real scoring function
from score_engine import v1_score_from_data

def run(buy=62, sell=50, max_pos=8, cost=0.003, name=''):
    cash = 100000.0
    pos = {}
    trades = []
    
    print(f'  回测 {name}...', end=' ', flush=True)
    ts = time.time()
    
    start = 250
    for di in range(start, len(all_dates) - 1):
        date = all_dates[di]
        next_date = all_dates[di + 1]
        
        # 计算评分(当天收盘数据)
        scores = {}
        for code in list(ALL_DATA.keys())[:200]:  # 限200只加快速度
            s = get_score(code, di)
            if s > 0:
                scores[code] = s
        
        if not scores:
            continue
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        # 卖出(次日开盘)
        for c in list(pos.keys()):
            if scores.get(c, 0) < sell:
                h = pos.pop(c)
                sell_p = ALL_DATA.get(c, {}).get(next_date, {}).get('o', 0)
                if sell_p > 0:
                    proceeds = h['shares'] * sell_p * (1 - cost)
                    cash += proceeds
                    trades.append(('sell', next_date, c))
        
        # 买入(次日开盘)
        if len(pos) < max_pos:
            cand = [(c, s) for c, s in ranked[:30] if s >= buy and c not in pos]
            for code, score in cand[:max_pos - len(pos)]:
                buy_p = ALL_DATA.get(code, {}).get(next_date, {}).get('o', 0)
                if buy_p <= 0:
                    continue
                invest = cash * (1.0 / max_pos) * 0.95
                shares = invest / buy_p
                if shares > 0:
                    cash -= invest
                    pos[code] = {'shares': shares, 'buy_p': buy_p}
                    trades.append(('buy', next_date, code))
    
    # 终值
    final = cash
    for c, h in pos.items():
        p = ALL_DATA.get(c, {}).get(all_dates[-1], {}).get('c', 0)
        if p > 0:
            final += h['shares'] * p
    
    ret = (final/100000 - 1) * 100
    yrs = (len(all_dates) - start) / 245
    ann = ((final/100000) ** (1/max(yrs,1)) - 1) * 100 if final > 0 else -100
    
    buys = sum(1 for t in trades if t[0] == 'buy')
    sells = sum(1 for t in trades if t[0] == 'sell')
    
    print(f'回报{ret:+.1f}% 年化{ann:+.1f}% 交易{buys}买/{sells}卖 {time.time()-ts:.0f}s', flush=True)
    return {'ret': ret, 'ann': ann, 'buys': buys, 'sells': sells, 'final': final}

print()
print('🏁 V1真实回测(真实score_engine.py, 200只样本)')
print('═' * 60)
print()

r = run(62, 50, 8, 0.003, 'V1真实(200只样本)')

print()
print(f'总耗时: {time.time()-t0:.0f}s')
