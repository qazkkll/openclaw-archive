#!/usr/bin/env python3
"""建立美股质量池"""
import yfinance as yf, json, warnings, numpy as np, time
warnings.filterwarnings('ignore')

# 用SP500 + NASDAQ 100做基础
sp500 = list(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'])
nasdaq100 = list(pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'])
tickers = list(set(sp500 + nasdaq100))
print('基础池: %d只' % len(tickers))
import pandas as pd
print('Need pandas for HTML... using static list instead')
