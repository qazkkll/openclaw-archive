#!/usr/bin/env python3
import minishare as ms
api = ms.pro_api('Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06')
# Test US stock data
for api_name in ['us_daily', 'daily', 'stock_us_daily']:
    try:
        df = api.query(api_name, ts_code='AAPL', start_date='20260501', end_date='20260601')
        print('%s: %d rows' % (api_name, len(df)))
        if len(df) > 0:
            print('  Columns:', list(df.columns)[:5])
            print('  Last:', dict(df.iloc[-1]) if len(df) > 0 else '')
    except Exception as e:
        print('%s: FAIL - %s' % (api_name, str(e)[:60]))
