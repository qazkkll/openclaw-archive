#!/usr/bin/env python3
"""全市场美股V5-R5扫描"""
import yfinance as yf, json, warnings, time, numpy as np
warnings.filterwarnings('ignore')

# Get all US stock tickers from yfinance
print('Getting all US tickers...', flush=True)
sp500 = list(pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'])
nasdaq = list(pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]['Ticker'])
# Combine and deduplicate
all_tickers = list(set(sp500 + nasdaq))
print('Total unique: %d' % len(all_tickers), flush=True)
import pandas as pd
print('Failed - need pandas with HTML support. Trying alternative...', flush=True)
