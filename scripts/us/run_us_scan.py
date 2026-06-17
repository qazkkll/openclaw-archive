#!/usr/bin/env python3
"""
🍤 美股全量扫描 — S&P 500头部 + 持仓/关注
"""
import sys, json, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from scoring import score, is_a_stock
from compliance import check_compliance
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# S&P 500 头部+关注
TICKERS = [
    'AAPL','MSFT','AMZN','NVDA','GOOGL','META','BRK.B','TSLA','AVGO','JPM',
    'V','JNJ','WMT','MA','PG','XOM','UNH','CVX','HD','MRK','BAC','PEP',
    'ABBV','KO','COST','ADBE','CRM','NFLX','TMO','PFE','ABT','ACN','DHR',
    'WFC','NKE','DIS','LIN','TXN','PM','QCOM','IBM','SPGI','UPS','CAT',
    'LOW','RTX','BA','MDT','GS','SCHW','HON','AMD','DE','AXP','TMUS',
    'C','BLK','LMT','NOW','BKNG','SYK','ELV','PLD','MDLZ','GILD','ADP',
    'ISRG','MMC','CL','ZTS','WM','EOG','APD','DUK','SO','MS','CB',
    'NOC','CI','MO','GD','ITW','GE','FIS','SHW','MCO','AMAT','SBUX',
    'BDX','TGT','EQIX','REGN','PGR','HCA','ADI','CME','ATVI','MU',
    'CSCO','INTC','NOK','ZS','SNOW','TEAM','PANW','ORCL','CRWD','DDOG'
]

print(f'📊 美股V4.2扫描 · {len(TICKERS)}只')
print()

results = []
for i, t in enumerate(TICKERS):
    try:
        d = yf.download(t, period='6mo', interval='1d', progress=False)
        if len(d) < 30:
            continue
        close = list(d['Close'].values.flatten())
        cur = close[-1]
        if cur < 3: continue
        
        s = score(t, details=True)
        prev = close[-2] if len(close) > 1 else cur
        ma20 = sum(close[-20:]) / 20
        h52 = max(close[-252:]) if len(close) >= 252 else max(close)
        l52 = min(close[-252:]) if len(close) >= 252 else min(close)
        pos52 = (cur - l52) / (h52 - l52) * 100 if h52 != l52 else 50
        
        results.append({'ticker': t, 'score': s['score'], 'price': round(cur, 1), 'ma20': round(ma20, 1), 'pos52': round(pos52, 0)})
        
        if (i+1) % 30 == 0:
            print(f'  进度: {i+1}/{len(TICKERS)} | 有效: {len(results)}')
        time.sleep(0.3)
    except:
        continue

results.sort(key=lambda x: x['score'], reverse=True)

with open(os.path.join(ROOT, 'data', 'us_scored.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n完成: {len(results)}只有效')
print()
print('=== 🏆 美股V4.2排名 ===')
for i, r in enumerate(results[:15]):
    em = '🟢' if r['score'] >= 20 else ('🟡' if r['score'] >= 0 else '🔴')
    ma = '✅' if r['price'] > r['ma20'] else '❌'
    print(f'#{i+1} {em} {r["ticker"]}: {r["score"]}分  ${r["price"]}  MA20${r["ma20"]}{ma}')

check_compliance('US全量扫描', stocks_count=len(results), scoring='V4.2', source='yfinance')
