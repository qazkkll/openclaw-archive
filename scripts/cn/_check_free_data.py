"""测试免费美股基本面数据源"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request, json, time

# 方案1: Yahoo Finance 直接API（不用yfinance库）
url = 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/AAPL?modules=defaultKeyStatistics,financialData'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    r = urllib.request.urlopen(req, timeout=15)
    data = json.loads(r.read())
    ks = data['quoteSummary']['result'][0].get('defaultKeyStatistics', {})
    fd = data['quoteSummary']['result'][0].get('financialData', {})
    print('Yahoo Finance 直接API: ✅')
    for k in ['priceToBook','returnOnEquity','revenueGrowth','earningsGrowth','debtToEquity','grossMargins','profitMargins']:
        v = ks.get(k) or fd.get(k) or {}
        raw = v.get('raw')
        print(f'  {k}: {raw}')
except Exception as e:
    print(f'Yahoo Finance 直接API: ❌ {e}')

# 方案2: 试试大智慧美股API
time.sleep(1)
try:
    url2 = 'https://push.sina.com.cn/api/us_finance?symbol=AAPL'
    req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
    r2 = urllib.request.urlopen(req2, timeout=15)
    print('\n新浪美股API: ✅', r2.status)
except Exception as e:
    print(f'\n新浪美股API: ❌ {e}')
