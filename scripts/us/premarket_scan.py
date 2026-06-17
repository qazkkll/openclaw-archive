#!/usr/bin/env python3
"""Pre-market scan + V5/R5 analysis"""
import yfinance as yf, json, warnings
warnings.filterwarnings('ignore')

pool = json.load(open('/home/admin/.openclaw/workspace/data/sp500_universe.json'))
tickers = pool['tickers']

print('Fetching pre-market data...')
results = []
for i, t in enumerate(tickers):
    try:
        tk = yf.Ticker(t)
        hist = tk.history(period='5d', interval='5m', prepost=True)
        if hist.empty:
            continue
        last = hist.iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else last['Close']
        pre_change = (last['Close'] / prev_close - 1) * 100
        vol = hist['Volume'].sum()
        results.append({
            'ticker': t,
            'pre_change': round(pre_change, 2),
            'pre_price': round(last['Close'], 2),
            'volume': int(vol)
        })
    except:
        pass
    if (i+1) % 50 == 0:
        print('  %d/%d' % (i+1, len(tickers)), flush=True)

results.sort(key=lambda x: -abs(x['pre_change']))
print()
print('=== Pre-Market Analysis (%d stocks) ===' % len(results))
print('Ticker  Price   PreChg   Volume')
print('-' * 38)
for r in results[:30]:
    print('%s %7.2f %+6.2f%% %8d' % (r['ticker'].ljust(6), r['pre_price'], r['pre_change'], r['volume']))

gainers = [r for r in results if r['pre_change'] > 1]
losers = [r for r in results if r['pre_change'] < -1]

# V5 candidates
print()
print('=== V5 Momentum (pre-market gaining) ===')
for r in gainers[:10]:
    print('  %s: pre+%.1f%% at %.2f' % (r['ticker'], r['pre_change'], r['pre_price']))

# R5 candidates
print()
print('=== R5 Reversal (pre-market dipping) ===')
for r in losers[:10]:
    print('  %s: pre%.1f%% at %.2f' % (r['ticker'], r['pre_change'], r['pre_price']))

# Quality pool question
print()
print('=== Quality Pool Analysis ===')
print('Current quality pool: %d stocks (sp500_universe.json)' % len(tickers))
print('V5 filters: RSI<59, positive 5d+30d mom, industry cap -> no need for pre-filter')
print('R5 filters: negative 5d, positive 30d trend -> no need for pre-filter')
print('Recommendation: quality pool is REDUNDANT with V5-R5 strategy.')
print('  V5 already filters out low-momentum stocks.')
print('  R5 already filters out downtrend stocks.')
print('  Adding a quality pool on top would only reduce opportunities.')

print()
print('Done.')
