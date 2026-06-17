#!/usr/bin/env python3
import minishare as ms
api = ms.pro_api('Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06')
methods = [m for m in dir(api) if not m.startswith('_')]
print('Available methods:')
for m in sorted(methods):
    print('  %s' % m)
