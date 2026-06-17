#!/usr/bin/env python3
"""Re-score all stocks using baostock data (with proper high/low) for correct ADX"""
import json, os, time, sys
sys.path.insert(0, '/home/admin/.openclaw/workspace/scripts')
from score_engine import compute_indicators, v1_score

DATA = '/home/admin/.openclaw/workspace/data/backtest_baostock.json'
OUT = '/home/admin/.openclaw/workspace/data/v1_scores_baostock.json'
CACHE = '/home/admin/.openclaw/workspace/data/cache'

print('Loading data...', flush=True)
Y = json.load(open(DATA))
stocks = list(Y.keys())[:200]  # Limit to 200 for speed
print('Stocks: %d' % len(stocks), flush=True)

# Score on a sample of dates (every 10th day)
t0 = time.time()
scores = {}
total = 0
for ci, code in enumerate(stocks):
    cd = Y[code].get('dates',[])
    cl = Y[code].get('close',[])
    hi = Y[code].get('high',[])
    lo = Y[code].get('low',[])
    if not cd or len(cd) < 60:
        continue
    day_scores = {}
    for di in range(60, len(cd), 10):  # every 10th day
        date = cd[di]
        start = max(0, di-250)
        c = cl[start:di+1]
        h = hi[start:di+1]
        l = lo[start:di+1]
        if len(c) < 60: continue
        ind = compute_indicators(c, h, l)
        if ind is None: continue
        s = v1_score(ind, len(c)-1)
        if s > 0:
            day_scores[date] = round(s, 1)
            total += 1
    if day_scores:
        scores[code] = day_scores
    if (ci+1) % 100 == 0:
        el = time.time()-t0
        print('  %d/%d scored=%d (%.0fs, %.1f/s)' % (ci+1, len(stocks), len(scores), el, total/el if el>0 else 0), flush=True)

json.dump(scores, open(OUT, 'w'))
elapsed = time.time()-t0
print('DONE: %d stocks, %d scores (%.0fs)' % (len(scores), total, elapsed), flush=True)
