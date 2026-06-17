"""免费数据源测试 v3"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import time, requests

time.sleep(3)

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

try:
    tk = yf.Ticker('AAPL', session=session)
    info = tk.info
    print('yfinance custom session: OK')
    print(f'  pb: {info.get("priceToBook")}')
    print(f'  roe: {info.get("returnOnEquity")}')
    print(f'  rev_growth: {info.get("revenueGrowth")}')
    print(f'  earningsGrowth: {info.get("earningsGrowth")}')
    print(f'  debtToEquity: {info.get("debtToEquity")}')
    print(f'  grossMargins: {info.get("grossMargins")}')
except Exception as e:
    print(f'yfinance: {str(e)[:100]}')
