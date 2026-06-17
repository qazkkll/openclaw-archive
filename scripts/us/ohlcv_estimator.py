#!/usr/bin/env python3
"""OHLCV estimator: adds high/low to close-only data using cached baostock volatility"""
import json, os

CACHE = '/home/admin/.openclaw/workspace/data/cache'
_VOL = {}  # {code: avg_atr_pct}
_DEFAULT_VOL = 4.0  # avg A-share volatility ~4%

def _load_volatility():
    if _VOL: return
    for f in os.listdir(CACHE):
        if f.endswith('.json') and not f.startswith('mf_') and not f.startswith('ll_') and f != '_index.json':
            try:
                d = json.load(open(os.path.join(CACHE, f)))
                data = d.get('data', []) if isinstance(d, dict) else d
                code = d.get('code', f.replace('.json',''))
                atrs = []
                for r in data:
                    if isinstance(r, dict):
                        h, l, c = r.get('high',0), r.get('low',0), r.get('close',0)
                        if h > 0 and l > 0 and c > 0:
                            atr = (h - l) / c * 100
                            if atr < 20: atrs.append(atr)
                if atrs: _VOL[code] = sum(atrs)/len(atrs)
            except: pass
    if not _VOL: _VOL['_default'] = _DEFAULT_VOL

def get_hl(code, price):
    """Return (estimated_high, estimated_low) for a stock at given price"""
    _load_volatility()
    vol = _VOL.get(code, _VOL.get('_default', _DEFAULT_VOL))
    half = vol / 200
    return round(price * (1 + half), 2), round(price * (1 - half), 2)
