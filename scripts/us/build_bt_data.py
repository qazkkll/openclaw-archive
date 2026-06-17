#!/usr/bin/env python3
"""Convert baostock cache (OHLCV) to backtest format with high/low"""
import json, os, time

CACHE = '/home/admin/.openclaw/workspace/data/cache'
OUT = '/home/admin/.openclaw/workspace/data/backtest_baostock.json'

t0 = time.time()
result = {}
count = 0

for f in os.listdir(CACHE):
    if not f.endswith('.json') or f == '_index.json':
        continue
    with open(os.path.join(CACHE, f), encoding='utf-8') as fh:
        d = json.load(fh)
    if isinstance(d, list):
        data = d
    else:
        data = d.get('data', [])
    if len(data) < 60:
        continue
    if isinstance(d, dict):
        code = d.get('code', '')
        board = d.get('board', '')
    else:
        code = ''
        board = ''
    if not code and data:
        code = data[0].get('code', data[0].get('ts_code', '').split('.')[0])
    
    dates = []
    closes = []
    highs = []
    lows = []
    opens = []
    for row in data:
        if isinstance(row, dict):
            raw = row.get('day', row.get('trade_date',''))
            if raw and len(raw) == 8 and '-' not in raw:
                raw = raw[:4] + '-' + raw[4:6] + '-' + raw[6:8]
            dates.append(raw)
            closes.append(row.get('close',0))
            highs.append(row.get('high', row.get('close',0)))
            lows.append(row.get('low', row.get('close',0)))
            opens.append(row.get('open', row.get('close',0)))
        else:
            continue
    result[code] = {
        'dates': dates,
        'close': closes,
        'high': highs,
        'low': lows,
        'open': opens,
    }
    count += 1

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

print('Converted %d stocks (%.0fs)' % (count, time.time()-t0))
print('Output: %s (%.1fMB)' % (OUT, os.path.getsize(OUT)/1024/1024))
