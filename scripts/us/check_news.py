#!/usr/bin/env python3
import minishare as ms
token = 'Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06'
df = ms.pro_api(token).news(start_date='2026-06-01 00:00:00', end_date='2026-06-01 23:59:59')
print('Columns:', list(df.columns))
print()
row = df.iloc[0].to_dict()
for k, v in row.items():
    print('%s: %s' % (k, str(v)[:100]))
