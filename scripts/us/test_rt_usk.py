#!/usr/bin/env python3
import minishare as ms
api = ms.pro_api('Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06')

# Test rt_us_k
print('=== rt_us_k (美股实时K线) ===')
try:
    df = api.query('rt_us_k', ts_code='AAPL', freq='D')
    print('Rows: %d' % len(df))
    if len(df) > 0:
        print('Columns:', list(df.columns))
        print('Last:', dict(df.iloc[-1]))
except Exception as e:
    print('FAIL:', str(e))

print()

# Also try with different params
try:
    df = api.query('rt_us_k', ts_code='AAPL')
    print('Without freq: %d rows' % len(df))
    if len(df) > 0:
        print('Last:', dict(df.iloc[-1]))
except Exception as e:
    print('Without freq FAIL:', str(e))
