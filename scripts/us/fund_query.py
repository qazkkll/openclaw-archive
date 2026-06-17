#!/usr/bin/env python3
"""基本面目查询器 — 通过 akshare 拿 PE/PB/ROE，供 Node.js 调用"""
import sys, json, akshare as ak

def get_fund(code):
    try:
        df = ak.stock_value_em(symbol=code)
        if df.empty: return None
        r = df.iloc[-1]
        pe = r['PE(TTM)'] if not pd.isna(r['PE(TTM)']) else None
        pb = r['市净率'] if not pd.isna(r['市净率']) else None
        mc = r['总市值']
        return {'code': code, 'pe': float(pe) if pe else None, 'pb': float(pb) if pb else None, 'market_cap': float(mc)}
    except: return None

import pandas as pd
code = sys.argv[1] if len(sys.argv) > 1 else ''
if code:
    r = get_fund(code)
    print(json.dumps(r or {'error':'无数据'}, ensure_ascii=False))
else:
    # 批量模式: 从 stdin 读代码列表
    codes = [line.strip() for line in sys.stdin if line.strip()]
    results = {}
    for c in codes[:20]:  # 一次最多20个
        r = get_fund(c)
        if r: results[c] = r
    print(json.dumps(results, ensure_ascii=False))
