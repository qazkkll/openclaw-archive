#!/usr/bin/env python3
"""Re-score yahoo stocks with estimated high/low for proper ADX"""
import json, os, time, sys
sys.path.insert(0, '/home/admin/.openclaw/workspace/scripts')
from score_engine import compute_indicators, v1_score
import ohlcv_estimator as hl

YFILE = '/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json'
OFILE = '/home/admin/.openclaw/workspace/data/v1_scores_hl.json'

print('Loading...', flush=True)
Y = json.load(open(YFILE))
stocks = [c for c in Y if c != '000001'][:100]
print('Stocks: %d' % len(stocks), flush=True)

t0 = time.time()
scores = {}
total_scored = 0

for si, code in enumerate(stocks):
    cd = Y[code].get('dates',[])
    cl = Y[code].get('close',[])
    if not cd or len(cd) < 60: continue
    day_scores = {}
    for di in range(60, len(cd), 10):  # every 10th day
        date = cd[di]
        start = max(0, di - 250)
        c = cl[start:di+1]
        # Estimate high/low
        h = []; l = []
        for p in c:
            hh, ll = hl.get_hl(code, p)
            h.append(hh); l.append(ll)
        ind = compute_indicators(c, h, l)
        if ind is None: continue
        s = v1_score(ind, len(c)-1)
        if s > 0:
            day_scores[date] = round(s, 1)
            total_scored += 1
    if day_scores:
        scores[code] = day_scores
    if (si+1) % 20 == 0:
        el = time.time()-t0
        print('  %d/%d (%.0fs, %.1f/s)' % (si+1, len(stocks), el, total_scored/el if el>0 else 0), flush=True)

json.dump(scores, open(OFILE, 'w'))
print('DONE: %d stocks, %d scores (%.0fs)' % (len(scores), total_scored, time.time()-t0), flush=True)
