#!/usr/bin/env python3
"""
补充缺失的megacap股票到10y数据
"""
import pandas as pd
import yfinance as yf
import time
import os

ROOT = '/home/hermes/.hermes/openclaw-archive'
PARQUET_10Y = os.path.join(ROOT, 'data/us/us_hist_yf_10y.parquet')

# 当前10y数据
df_10y = pd.read_parquet(PARQUET_10Y)
existing = set(df_10y['sym'].unique())
print(f'当前10y: {len(existing)}只')

# 需要补充的megacap股票（从megacap列表 + 当前持仓）
MISSING = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'ASML', 'ANET', 'AVGO',
    'ABBV', 'ABT', 'ACN', 'ADBE', 'AMD', 'ARM', 'BRK-B', 'CRWD',
    'DDOG', 'DE', 'DHR', 'EA', 'GE', 'GS', 'HD', 'HOOD', 'INTU',
    'LCID', 'MU', 'NOW', 'PINS', 'PLTR', 'RBLX', 'RIVN', 'RTX',
    'SBUX', 'SHOP', 'SNAP', 'SOFI', 'SYK', 'T', 'TTWO', 'TXN',
    'U', 'UBER', 'UNP', 'UPS', 'XOM'
]

# 过滤掉已存在的
to_download = [t for t in MISSING if t not in existing]
print(f'需要补充: {len(to_download)}只')
print(f'列表: {to_download}')

# 下载
new_data = []
failed = []

for i, ticker in enumerate(to_download):
    try:
        print(f'  [{i+1}/{len(to_download)}] {ticker}...', end=' ', flush=True)
        tk = yf.Ticker(ticker)
        hist = tk.history(period='max')
        if len(hist) < 252:  # 至少1年数据
            print(f'数据不足({len(hist)}行)')
            continue
        
        # 转换格式
        hist = hist.reset_index()
        hist['sym'] = ticker
        hist = hist.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'Dividends': 'dividends', 'Stock Splits': 'stock splits'
        })
        hist = hist[['sym', 'date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']]
        hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)
        
        new_data.append(hist)
        print(f'{len(hist)}行')
        time.sleep(0.5)  # 避免请求过快
    except Exception as e:
        print(f'失败: {e}')
        failed.append(ticker)

if new_data:
    # 合并
    df_new = pd.concat(new_data, ignore_index=True)
    print(f'\n新下载: {len(df_new)}行, {df_new["sym"].nunique()}只')
    
    # 合并到10y数据
    df_combined = pd.concat([df_10y, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['sym', 'date'], keep='last')
    df_combined = df_combined.sort_values(['sym', 'date']).reset_index(drop=True)
    
    # 保存
    df_combined.to_parquet(PARQUET_10Y, index=False)
    print(f'✅ 保存: {len(df_combined)}行, {df_combined["sym"].nunique()}只')
else:
    print('没有新数据')

if failed:
    print(f'⚠️ 失败: {failed}')

# 验证
df_check = pd.read_parquet(PARQUET_10Y)
print(f'\n验证: {df_check["sym"].nunique()}只')
for t in ['AAPL', 'MSFT', 'ASML', 'ANET', 'NVDA', 'AMZN']:
    if t in df_check['sym'].unique():
        print(f'  ✅ {t}')
    else:
        print(f'  ❌ {t}')
