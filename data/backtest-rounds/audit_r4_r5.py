#!/usr/bin/env python3
"""Round 4-5 audit helper: score distribution + pending rec tracking"""
import yfinance as yf
import pandas as pd
import numpy as np
import json

# === Round 5: Pending US Recommendations ===
tickers_map = {
    'SPY': 'SPY',
    'QQQ': 'QQQ',
    'VIX': '^VIX',
    'SMH': 'SMH'
}

data = yf.download(list(tickers_map.values()), start='2026-06-20', end='2026-06-25', progress=False)

# Pending US recs
pending = [
    {'id': 'R3dd1de2f', 'target': 'VIX', 'ticker': '^VIX', 'direction': 'bearish', 'date': '20260622', 'confidence': 0.65, 'rec_price': 17.28},
    {'id': 'Rac20116d', 'target': 'SPY', 'ticker': 'SPY', 'direction': 'bullish', 'date': '20260622', 'confidence': 0.6, 'rec_price': 744.39},
    {'id': 'Rd7677a38', 'target': 'QQQ', 'ticker': 'QQQ', 'direction': 'bullish', 'date': '20260622', 'confidence': 0.65, 'rec_price': 737.95},
    {'id': 'R0561918b', 'target': 'SPY', 'ticker': 'SPY', 'direction': 'neutral', 'date': '20260623', 'confidence': 0.55, 'rec_price': 733.58},
    {'id': 'R62bed8c1', 'target': 'Semiconductor', 'ticker': 'SMH', 'direction': 'bullish', 'date': '20260623', 'confidence': 0.65, 'rec_price': 622.05},
    {'id': 'Rd83637ca', 'target': 'S&P 500', 'ticker': 'SPY', 'direction': 'neutral', 'date': '20260623', 'confidence': 0.5, 'rec_price': 733.58},
    {'id': 'R3ebeeba5', 'target': 'VIX', 'ticker': '^VIX', 'direction': 'bearish', 'date': '20260623', 'confidence': 0.5, 'rec_price': 19.49},
]

print("=== Pending US Recommendations - Current Status ===")
results = []
for rec in pending:
    tk = rec['ticker']
    try:
        close = data['Close'][tk].dropna()
        curr = float(close.iloc[-1]) if len(close) > 0 else rec['rec_price']
        rec_p = rec['rec_price']
        
        if rec['direction'] == 'bearish':
            chg = (rec_p - curr) / rec_p * 100
        elif rec['direction'] == 'bullish':
            chg = (curr - rec_p) / rec_p * 100
        else:
            chg = (curr - rec_p) / rec_p * 100
        
        status = "WIN" if chg > 0.5 else "LOSS" if chg < -0.5 else "FLAT"
        results.append({'id': rec['id'], 'target': rec['target'], 'dir': rec['direction'], 'conf': rec['confidence'], 'rec_price': rec_p, 'curr_price': round(curr, 2), 'chg_pct': round(chg, 2), 'status': status})
        print(f"  {rec['id']}: {rec['target']} {rec['direction']} conf={rec['confidence']} | rec=${rec_p} -> curr=${curr:.2f} | {chg:+.2f}% [{status}]")
    except Exception as e:
        results.append({'id': rec['id'], 'target': rec['target'], 'dir': rec['direction'], 'conf': rec['confidence'], 'rec_price': rec['rec_price'], 'curr_price': None, 'chg_pct': None, 'status': 'ERROR'})
        print(f"  {rec['id']}: {rec['target']} {rec['direction']} | ERROR: {e}")

# Summary
wins = sum(1 for r in results if r['status'] == 'WIN')
losses = sum(1 for r in results if r['status'] == 'LOSS')
flats = sum(1 for r in results if r['status'] == 'FLAT')
errors = sum(1 for r in results if r['status'] == 'ERROR')
valid = [r for r in results if r['chg_pct'] is not None]
avg_chg = np.mean([r['chg_pct'] for r in valid]) if valid else 0

print(f"\nSummary: {wins} WIN / {losses} LOSS / {flats} FLAT / {errors} ERROR")
print(f"Average change: {avg_chg:+.2f}%")
print(f"Hit rate (>0.5%): {wins}/{len(valid)} = {wins/len(valid)*100:.1f}%" if valid else "N/A")

# Save
with open('data/backtest-rounds/pending_us_status.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to data/backtest-rounds/pending_us_status.json")
