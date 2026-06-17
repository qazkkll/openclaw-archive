#!/usr/bin/env python3
"""Test minishare news API"""
import minishare as ms
token = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'
df = ms.pro_api(token).news(start_date='2026-06-01 00:00:00', end_date='2026-06-01 23:59:59')
print('News count: %d' % len(df))
if len(df) > 0:
    for i, row in df.head(5).iterrows():
        print('[%s] %s' % (row.get('datetime','?'), str(row.get('title',''))[:80]))
