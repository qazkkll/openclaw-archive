#!/usr/bin/env python3
"""
🦅 Falcon Insider Trading IC Validation (OpenInsider)
======================================================
Scrapes insider trading data from OpenInsider.com and computes IC/ICIR.

Signals tested:
  - net_buy_count: net insider buys (buys - sells) in last 90 days
  - net_buy_value: net insider buy value ($) in last 90 days
  - ceo_cfo_buy: CEO/CFO specific buy signal (binary)
  - cluster_buy: 3+ insiders buying in 90 days (binary)
  - insider_buy_ratio: buys / (buys + sells)
"""

import sys, json, time, os, re
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import urllib.request

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = WORKSPACE / "data" / "falcon"
INSIDER_DIR = DATA_DIR / "insider_trading"
INSIDER_DIR.mkdir(exist_ok=True)

SPX_PATH = WORKSPACE / "data" / "config" / "sp500_symbols.json"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"


def load_spx_tickers():
    with open(SPX_PATH) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("tickers", [])


def scrape_openinsider(ticker, days=90):
    """Scrape insider trading from OpenInsider for a single ticker."""
    url = f'http://openinsider.com/search?q={ticker}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        body = resp.read().decode()
    except Exception as e:
        return []
    
    # Find the data table (table with most rows, typically table 9)
    tables = re.findall(r'<table[^>]*>(.*?)</table>', body, re.DOTALL)
    data_table = None
    max_rows = 0
    for t in tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, re.DOTALL)
        if len(rows) > max_rows:
            max_rows = len(rows)
            data_table = t
    
    if not data_table or max_rows < 3:
        return []
    
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', data_table, re.DOTALL)
    records = []
    
    for row in rows[1:]:  # Skip header
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) < 10:
            continue
        
        # Clean cells
        clean = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        
        # Parse fields
        try:
            filing_date = clean[1][:10] if clean[1] else ''
            trade_date = clean[2][:10] if clean[2] else ''
            insider_name = clean[4] if len(clean) > 4 else ''
            title = clean[5] if len(clean) > 5 else ''
            trade_type = clean[6] if len(clean) > 6 else ''
            price_str = clean[7].replace('$', '').replace(',', '') if len(clean) > 7 else '0'
            qty_str = clean[8].replace(',', '').replace('+', '') if len(clean) > 8 else '0'
            value_str = clean[11].replace('$', '').replace(',', '').replace('+', '') if len(clean) > 11 else '0'
            
            # Parse numbers
            try:
                price = float(price_str) if price_str and price_str != '' else 0
            except:
                price = 0
            try:
                qty = int(qty_str) if qty_str and qty_str != '' else 0
            except:
                qty = 0
            try:
                value = float(value_str) if value_str and value_str != '' else 0
            except:
                value = 0
            
            # Determine buy/sell
            is_buy = any(kw in trade_type.upper() for kw in ['PURCHASE', 'P -', 'BUY'])
            is_sell = any(kw in trade_type.upper() for kw in ['SALE', 'S -', 'SELL'])
            
            # Check for CEO/CFO
            is_ceo_cfo = any(kw in title.upper() for kw in ['CEO', 'CFO', 'CHIEF'])
            
            records.append({
                'ticker': ticker,
                'filing_date': filing_date,
                'trade_date': trade_date,
                'insider': insider_name,
                'title': title,
                'type': trade_type,
                'price': price,
                'qty': qty,
                'value': abs(value),
                'is_buy': is_buy,
                'is_sell': is_sell,
                'is_ceo_cfo': is_ceo_cfo,
            })
        except Exception:
            continue
    
    return records


def download_insider_data(tickers, max_tickers=100):
    """Download insider trading for tickers. Cache to disk."""
    cache_file = INSIDER_DIR / "openinsider_cache.json"
    
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
    
    to_fetch = [t for t in tickers if t not in cache][:max_tickers]
    print(f"📥 Fetching insider data for {len(to_fetch)} tickers ({len(cache)} cached)...")
    
    for i, ticker in enumerate(to_fetch):
        records = scrape_openinsider(ticker)
        cache[ticker] = records
        
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(to_fetch)}] {ticker}: {len(records)} records")
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            time.sleep(1)
        
        time.sleep(0.3)
    
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    total = sum(len(v) for v in cache.values())
    with_data = sum(1 for v in cache.values() if len(v) > 0)
    print(f"  ✅ {with_data} tickers with data, {total} total records")
    return cache


def compute_signals(cache, lookback_days=90):
    """Compute insider trading signals for each ticker."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    signals = {}
    
    for ticker, records in cache.items():
        if not records:
            signals[ticker] = {
                'net_buy_count': 0, 'net_buy_value': 0,
                'insider_buy_ratio': 0.5, 'cluster_buy': 0, 'ceo_cfo_buy': 0,
            }
            continue
        
        recent = [r for r in records if r.get('trade_date', '') >= cutoff]
        if not recent:
            signals[ticker] = {
                'net_buy_count': 0, 'net_buy_value': 0,
                'insider_buy_ratio': 0.5, 'cluster_buy': 0, 'ceo_cfo_buy': 0,
            }
            continue
        
        buys = [r for r in recent if r.get('is_buy')]
        sells = [r for r in recent if r.get('is_sell')]
        
        net_buy_count = len(buys) - len(sells)
        net_buy_value = sum(r.get('value', 0) for r in buys) - sum(r.get('value', 0) for r in sells)
        
        total = len(buys) + len(sells)
        buy_ratio = len(buys) / total if total > 0 else 0.5
        
        unique_buyers = set(r.get('insider', '') for r in buys)
        cluster = 1 if len(unique_buyers) >= 3 else 0
        
        ceo_cfo = 1 if any(r.get('is_ceo_cfo') for r in buys) else 0
        
        signals[ticker] = {
            'net_buy_count': net_buy_count,
            'net_buy_value': net_buy_value,
            'insider_buy_ratio': buy_ratio,
            'cluster_buy': cluster,
            'ceo_cfo_buy': ceo_cfo,
        }
    
    return signals


def compute_time_series_ic(cache, prices_df, lookback_days=90, forward_days=30):
    """Compute time-series IC by quarterly rolling."""
    price_dates = sorted(prices_df.index.astype(str))
    test_dates = price_dates[::63]  # Every quarter
    test_dates = [d for d in test_dates if d <= price_dates[-(forward_days+5)]]
    test_dates = test_dates[-8:]  # Last 2 years
    
    if len(test_dates) < 3:
        return None
    
    signal_names = ['net_buy_count', 'net_buy_value', 'insider_buy_ratio', 'cluster_buy', 'ceo_cfo_buy']
    all_ics = {s: [] for s in signal_names}
    
    for test_date in test_dates:
        td_idx = price_dates.index(test_date)
        if td_idx + forward_days >= len(price_dates):
            continue
        exit_date = price_dates[td_idx + forward_days]
        fwd_ret = (prices_df.loc[exit_date] / prices_df.loc[test_date] - 1).dropna()
        
        cutoff = (pd.Timestamp(test_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        for sig_name in signal_names:
            sig_values = {}
            for ticker, records in cache.items():
                if not records or ticker not in fwd_ret.index:
                    continue
                
                relevant = [r for r in records 
                           if r.get('trade_date', '') <= test_date 
                           and r.get('trade_date', '') >= cutoff]
                if not relevant:
                    continue
                
                buys = [r for r in relevant if r.get('is_buy')]
                sells = [r for r in relevant if r.get('is_sell')]
                
                if sig_name == 'net_buy_count':
                    sig_values[ticker] = len(buys) - len(sells)
                elif sig_name == 'net_buy_value':
                    sig_values[ticker] = sum(r.get('value', 0) for r in buys) - sum(r.get('value', 0) for r in sells)
                elif sig_name == 'insider_buy_ratio':
                    total = len(buys) + len(sells)
                    sig_values[ticker] = len(buys) / total if total > 0 else 0.5
                elif sig_name == 'cluster_buy':
                    unique_buyers = set(r.get('insider', '') for r in buys)
                    sig_values[ticker] = 1 if len(unique_buyers) >= 3 else 0
                elif sig_name == 'ceo_cfo_buy':
                    sig_values[ticker] = 1 if any(r.get('is_ceo_cfo') for r in buys) else 0
            
            common = set(sig_values.keys()) & set(fwd_ret.index)
            if len(common) < 20:
                continue
            
            sig_arr = np.array([sig_values[t] for t in common])
            ret_arr = np.array([fwd_ret[t] for t in common])
            valid = ~(np.isnan(sig_arr) | np.isnan(ret_arr))
            if valid.sum() < 20:
                continue
            
            ic, _ = spearmanr(sig_arr[valid], ret_arr[valid])
            if not np.isnan(ic):
                all_ics[sig_name].append(ic)
    
    results = {}
    for sig_name, ics in all_ics.items():
        if len(ics) < 2:
            results[sig_name] = {'mean_ic': None, 'icir': None, 't_stat': None, 'n_quarters': len(ics), 'status': 'insufficient'}
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        icir = mean_ic / std_ic if std_ic > 0 else 0
        t_stat = icir * np.sqrt(len(ics))
        results[sig_name] = {
            'mean_ic': round(mean_ic, 4),
            'std_ic': round(std_ic, 4),
            'icir': round(icir, 3),
            't_stat': round(t_stat, 2),
            'n_quarters': len(ics),
            'significant': abs(t_stat) > 1.96,
            'ics': [round(float(x), 4) for x in ics],
            'status': '✅' if abs(t_stat) > 1.96 else '❌',
        }
    
    return results


def main():
    print("=" * 60)
    print("  🦅 Falcon Insider Trading IC Validation (OpenInsider)")
    print("=" * 60)
    t0 = time.time()
    
    tickers = load_spx_tickers()
    print(f"📋 SPX Universe: {len(tickers)} tickers")
    
    # Download
    cache = download_insider_data(tickers, max_tickers=100)
    
    # Stats
    total_records = sum(len(v) for v in cache.values())
    tickers_with_data = sum(1 for v in cache.values() if len(v) > 0)
    print(f"\n📊 Data: {tickers_with_data} tickers, {total_records} records")
    
    # Load prices
    print("\n📊 Loading prices...")
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {prices.shape}")
    
    # Compute signals
    signals = compute_signals(cache, lookback_days=90)
    
    # Cross-sectional IC
    print("\n" + "=" * 60)
    print("  Cross-Sectional IC (latest snapshot, 30d forward)")
    print("=" * 60)
    
    price_dates = sorted(prices.index.astype(str))
    entry_date = price_dates[-35]
    exit_date = price_dates[-1]
    fwd_ret = (prices.loc[exit_date] / prices.loc[entry_date] - 1).dropna()
    
    signal_names = ['net_buy_count', 'net_buy_value', 'insider_buy_ratio', 'cluster_buy', 'ceo_cfo_buy']
    
    print(f"\n  {'Signal':<25} {'IC':>8} {'N':>6}")
    print("  " + "-" * 40)
    
    for sig_name in signal_names:
        sig_values = {t: signals[t][sig_name] for t in signals if t in fwd_ret.index}
        common = set(sig_values.keys()) & set(fwd_ret.index)
        if len(common) < 20:
            print(f"  {sig_name:<25} {'N/A':>8} {len(common):>6}")
            continue
        
        sig_arr = np.array([sig_values[t] for t in common])
        ret_arr = np.array([fwd_ret[t] for t in common])
        valid = ~(np.isnan(sig_arr) | np.isnan(ret_arr))
        if valid.sum() < 20:
            print(f"  {sig_name:<25} {'N/A':>8} {int(valid.sum()):>6}")
            continue
        
        ic, pval = spearmanr(sig_arr[valid], ret_arr[valid])
        sig = '✅' if pval < 0.05 else '❌'
        print(f"  {sig_name:<25} {ic:>+8.4f} {int(valid.sum()):>6} {sig}")
    
    # Time-series IC
    print("\n" + "=" * 60)
    print("  Time-Series IC (quarterly, 2 years)")
    print("=" * 60)
    
    ts_ic = compute_time_series_ic(cache, prices, lookback_days=90, forward_days=30)
    
    if ts_ic:
        print(f"\n  {'Signal':<25} {'Mean IC':>8} {'ICIR':>8} {'t-stat':>8} {'Qtrs':>5} {'Status':>6}")
        print("  " + "-" * 65)
        for sig_name, result in ts_ic.items():
            if result.get('mean_ic') is not None:
                print(f"  {sig_name:<25} {result['mean_ic']:>+8.4f} {result['icir']:>8.3f} {result['t_stat']:>8.2f} {result['n_quarters']:>5} {result['status']:>6}")
            else:
                print(f"  {sig_name:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8} {result['n_quarters']:>5} {result['status']:>6}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("  📋 VERDICT")
    print("=" * 60)
    
    if ts_ic:
        significant = [name for name, r in ts_ic.items() if r.get('significant')]
        if significant:
            print(f"\n  ✅ Significant: {significant}")
            print(f"  → Add to Falcon as new factor group")
        else:
            print(f"\n  ❌ No significant signals")
            marginal = [name for name, r in ts_ic.items() if r.get('t_stat') and abs(r['t_stat']) > 1.0]
            if marginal:
                print(f"  ⚠️ Marginal (t>1.0): {marginal}")
    
    elapsed = time.time() - t0
    print(f"\n⏱️ Total: {elapsed:.0f}s")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'tickers_with_data': tickers_with_data,
        'total_records': total_records,
        'time_series_ic': ts_ic,
    }
    with open(DATA_DIR / "insider_ic_results.json", 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"💾 Saved: {DATA_DIR / 'insider_ic_results.json'}")


if __name__ == '__main__':
    main()
