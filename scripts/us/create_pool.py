#!/usr/bin/env python3
"""创建美股质量池 - SP500基础"""
import json, yfinance as yf, warnings, numpy as np
warnings.filterwarnings('ignore')

# 直接用SP500列表
tickers = json.load(open('/home/admin/.openclaw/workspace/data/sp500_universe.json'))['tickers']
print('SP500基础: %d只' % len(tickers))

pool = []
for t in tickers:
    try:
        h = yf.Ticker(t).history(period='3mo')
        c = list(h['Close'])
        if len(c) < 30: continue
        pr = c[-1]
        m30 = (pr/c[-30]-1)*100 if c[-30] > 0 else 0
        pool.append({'ticker': t, 'price': round(pr, 2), 'mom30': round(m30, 1)})
    except: pass

pool.sort(key=lambda x: -abs(x['mom30']))
json.dump({'date': '2026-06-01', 'total': len(pool), 'pool': pool},
          open('/home/admin/.openclaw/workspace/data/us_quality_pool.json', 'w'))
print('质量池: %d只, 已保存' % len(pool))
for i, p in enumerate(pool[:10]):
    print('  %d. %-6s $%7.2f  30天%+.1f%%' % (i+1, p['ticker'], p['price'], p['mom30']))
