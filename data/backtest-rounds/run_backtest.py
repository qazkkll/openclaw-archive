#!/usr/bin/env python3
"""Round 2: US stock backtest using yfinance"""
import json, sys, time
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed")
    sys.exit(1)

ROOT = '/home/hermes/.hermes/openclaw-archive'
TRACK_FILE = f'{ROOT}/data/recommendations.json'

# Load recommendations
with open(TRACK_FILE) as f:
    data = json.load(f)

recs = data['recommendations']
us_recs = [r for r in recs if r.get('market') == 'us']
print(f'Total recommendations: {len(recs)}')
print(f'US recommendations: {len(us_recs)}')

# Ticker mapping
SPECIAL = {'SPY': 'SPY', 'QQQ': 'QQQ', 'VIX': '^VIX', 'IWM': 'IWM', 'DIA': 'DIA', 'Semiconductor': 'SOXX', 'S&P 500': 'SPY'}

def to_ticker(target):
    t = target.strip()
    if t in SPECIAL:
        return SPECIAL[t]
    if ' ' not in t and not t.isdigit():
        return t
    parts = t.split()
    code = parts[0]
    if code.isdigit() and len(code) == 6:
        return f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
    return code

# Score function matching auto_scorer.py logic
def compute_score(direction, ret_5d):
    THRESHOLD = 0.02
    if direction == 'bullish':
        if ret_5d > THRESHOLD: return 1.0
        elif ret_5d > 0: return 0.5
        elif ret_5d > -THRESHOLD: return 0.25
        else: return 0.0
    elif direction == 'bearish':
        if ret_5d < -THRESHOLD: return 1.0
        elif ret_5d < 0: return 0.5
        elif ret_5d < THRESHOLD: return 0.25
        else: return 0.0
    else:  # neutral
        if abs(ret_5d) < THRESHOLD: return 1.0
        elif abs(ret_5d) < 0.05: return 0.5
        else: return 0.0

# Process each US recommendation
results = []
for r in us_recs:
    target = r.get('target', '')
    date_str = r.get('date', '')
    direction = r.get('direction', '')
    confidence = r.get('confidence', 0)
    source = r.get('source', '')
    rec_id = r.get('id', '')
    
    ticker = to_ticker(target)
    
    # Parse date (handle both YYYYMMDD and YYYY-MM-DD)
    try:
        if '-' in date_str:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            dt = datetime.strptime(date_str, '%Y%m%d')
    except:
        results.append({'id': rec_id, 'target': target, 'ticker': ticker, 'direction': direction, 
                       'confidence': confidence, 'source': source, 'status': 'date_parse_error'})
        continue
    
    # Get price on date
    try:
        start = (dt - timedelta(days=3)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=15)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        
        if hist.empty:
            results.append({'id': rec_id, 'target': target, 'ticker': ticker, 'direction': direction,
                          'confidence': confidence, 'source': source, 'status': 'no_data'})
            time.sleep(0.5)
            continue
        
        target_date = dt.date()
        # Find closest price on or before target date
        base_price = None
        base_date = None
        for d in sorted(hist.index, reverse=True):
            if d.date() <= target_date:
                base_price = float(hist.loc[d, 'Close'])
                base_date = d.date()
                break
        
        if base_price is None:
            base_price = float(hist.iloc[0]['Close'])
            base_date = hist.index[0].date()
        
        # Find 5th trading day after base_date
        future_dates = [d for d in sorted(hist.index) if d.date() > base_date]
        if len(future_dates) >= 5:
            d5 = future_dates[4]
            price_5d = float(hist.loc[d5, 'Close'])
            ret_5d = (price_5d - base_price) / base_price
            score = compute_score(direction, ret_5d)
            
            results.append({
                'id': rec_id, 'target': target, 'ticker': ticker, 'direction': direction,
                'confidence': confidence, 'source': source,
                'base_date': str(base_date), 'base_price': round(base_price, 2),
                'exit_date': str(d5.date()), 'exit_price': round(price_5d, 2),
                'ret_5d': round(ret_5d * 100, 2), 'score': score,
                'status': 'ok'
            })
        else:
            results.append({'id': rec_id, 'target': target, 'ticker': ticker, 'direction': direction,
                          'confidence': confidence, 'source': source, 'status': 'insufficient_future_data'})
        
        time.sleep(0.5)
    except Exception as e:
        results.append({'id': rec_id, 'target': target, 'ticker': ticker, 'direction': direction,
                       'confidence': confidence, 'source': source, 'status': f'error: {str(e)[:100]}'})
        time.sleep(0.5)

# === Summary ===
ok = [r for r in results if r['status'] == 'ok']
print(f"\n{'='*80}")
print(f"US Backtest Results | Total {len(results)}, Valid {len(ok)}")
print(f"{'='*80}")

if ok:
    avg_score = sum(r['score'] for r in ok) / len(ok)
    correct = sum(1 for r in ok if r['score'] >= 0.5)
    print(f"\nOverall hit rate: {correct}/{len(ok)} = {correct/len(ok)*100:.1f}%")
    print(f"Average score: {avg_score:.3f}")
    print(f"Avg 5-day return: {sum(r['ret_5d'] for r in ok)/len(ok):+.2f}%")
    
    # By direction
    print(f"\n--- By Direction ---")
    for d in ['bullish', 'bearish', 'neutral']:
        group = [r for r in ok if r['direction'] == d]
        if group:
            avg_s = sum(r['score'] for r in group) / len(group)
            avg_r = sum(r['ret_5d'] for r in group) / len(group)
            c = sum(1 for r in group if r['score'] >= 0.5)
            print(f"  {d}: {len(group)} | hit {c}/{len(group)}={c/len(group)*100:.0f}% | avg_score={avg_s:.2f} | avg_ret={avg_r:+.2f}%")
    
    # By source
    print(f"\n--- By Source ---")
    for src in sorted(set(r['source'] for r in ok)):
        group = [r for r in ok if r['source'] == src]
        avg_s = sum(r['score'] for r in group) / len(group)
        avg_r = sum(r['ret_5d'] for r in group) / len(group)
        c = sum(1 for r in group if r['score'] >= 0.5)
        print(f"  {src}: {len(group)} | hit {c}/{len(group)}={c/len(group)*100:.0f}% | avg_score={avg_s:.2f} | avg_ret={avg_r:+.2f}%")
    
    # By confidence bracket
    print(f"\n--- By Confidence ---")
    for lo, hi, label in [(0, 0.5, 'low(<0.5)'), (0.5, 0.7, 'mid(0.5-0.7)'), (0.7, 1.01, 'high(>0.7)')]:
        group = [r for r in ok if lo <= r['confidence'] < hi]
        if group:
            avg_s = sum(r['score'] for r in group) / len(group)
            avg_r = sum(r['ret_5d'] for r in group) / len(group)
            c = sum(1 for r in group if r['score'] >= 0.5)
            print(f"  {label}: {len(group)} | hit {c}/{len(group)}={c/len(group)*100:.0f}% | avg_score={avg_s:.2f} | avg_ret={avg_r:+.2f}%")

    # Detail table
    print(f"\n--- Detail ---")
    for r in sorted(ok, key=lambda x: x['score'], reverse=True):
        emoji = 'WIN' if r['score'] >= 0.5 else 'PARTIAL' if r['score'] > 0 else 'FAIL'
        print(f"  {emoji:8s} {r['target']:12s} {r['direction']:8s} conf={r['confidence']:.2f} {r['source']:12s} ${r['base_price']:.2f}->${r['exit_price']:.2f} ret={r['ret_5d']:+.2f}% score={r['score']:.2f}")

# Errors
errors = [r for r in results if r['status'] != 'ok']
if errors:
    print(f"\n--- Could not score ({len(errors)}) ---")
    for r in errors:
        print(f"  {r['target']} ({r.get('ticker','')}): {r['status']}")

# Save results
with open(f'{ROOT}/data/backtest-rounds/backtest-us-raw.json', 'w') as f:
    json.dump({'results': results, 'summary': {
        'total': len(results), 'ok': len(ok), 'errors': len(errors),
        'avg_score': sum(r['score'] for r in ok)/len(ok) if ok else 0,
        'hit_rate': sum(1 for r in ok if r['score'] >= 0.5)/len(ok)*100 if ok else 0
    }}, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to data/backtest-rounds/backtest-us-raw.json")
