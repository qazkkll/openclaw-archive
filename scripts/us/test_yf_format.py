#!/usr/bin/env python3
"""测试yfinance返回格式"""
import yfinance as yf
import pandas as pd

# 下载3只测试
df = yf.download(['AAPL','MSFT','GOOGL'], period='5y', group_by='ticker', threads=True, progress=False, auto_adjust=True)
print('type:', type(df))
print('columns:', df.columns)
print('columns type:', type(df.columns))
if isinstance(df.columns, pd.MultiIndex):
    print('LEVELS:', df.columns.levels)
    print('Level 0:', df.columns.get_level_values(0).unique().tolist())
    print('AAPL columns:', df['AAPL'].columns.tolist())
    print('first AAPL row:')
    r = df['AAPL'].iloc[0]
    print(r)
    print('AAPL head:')
    print(df['AAPL'].head())
else:
    print('Single level columns:', df.columns.tolist())
    print('head:')
    print(df.head())
