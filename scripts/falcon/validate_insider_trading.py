#!/usr/bin/env python3
"""
🦅 Falcon Insider Trading IC Validation
=========================================
Pull insider trading data from FMP API, compute IC/ICIR against forward returns.

Signals tested:
  - net_buy_count: net insider buys (buys - sells) in last 90 days
  - net_buy_value: net insider buy value ($) in last 90 days
  - ceo_cfo_buy: CEO/CFO specific buy signal (binary)
  - cluster_buy: 3+ insiders buying in 90 days (binary)
  - insider_buy_ratio: buys / (buys + sells)
"""

import sys, json, time, os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Load .env
env_path = Path("/home/hermes/.hermes/openclaw-archive/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = WORKSPACE / "data" / "falcon"
INSIDER_DIR = DATA_DIR / "insider_trading"
INSIDER_DIR.mkdir(exist_ok=True)

FMP_KEY = os.environ.get("FMP_API_KEY", "")
if not FMP_KEY:
    print("❌ FMP_API_KEY not found in .env")
    sys.exit(1)

SPX_PATH = WORKSPACE / "data" / "config" / "sp500_symbols.json"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"


def load_spx_tickers():
    with open(SPX_PATH) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("tickers", [])


def fetch_insider_trading(ticker, api_key):
    """Fetch insider trading from FMP API v4."""
    import urllib.request
    url = f"https://financialmodelingprep.com/api/v4/insider-trading?symbol={ticker}&apikey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return data
    except Exception as e:
        return []


def download_all_insider(tickers, api_key, batch_size=50, max_tickers=None):
    """Download insider trading for all tickers. Cache to disk."""
    cache_file = INSIDER_DIR / "insider_raw.json"
    
    # Load existing cache
    cache = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
    
    to_fetch = [t for t in tickers if t not in cache]
    if max_tickers:
        to_fetch = to_fetch[:max_tickers]
    
    print(f"📥 Fetching insider data for {len(to_fetch)} tickers ({len(cache)} cached)...")
    
    for i, ticker in enumerate(to_fetch):
        data = fetch_insider_trading(ticker, api_key)
        cache[ticker] = data
        
        if (i + 1) % batch_size == 0:
            print(f"  [{i+1}/{len(to_fetch)}] {ticker}: {len(data)} records")
            # Save intermediate
            with open(cache_file, 'w') as f:
                json.dump(cache, f)
            time.sleep(0.5)  # Rate limit
        
        time.sleep(0.1)  # Rate limit
    
    # Final save
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    
    print(f"  ✅ Saved {len(cache)} tickers to {cache_file}")
    return cache


def compute_insider_signals(raw_data, lookback_days=90):
    """Compute insider trading signals from raw FMP data.
    
    Returns: dict of ticker -> signal values
    """
    signals = {}
    
    for ticker, records in raw_data.items():
        if not records or not isinstance(records, list):
            continue
        
        # Filter to lookback window
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        recent = []
        for r in records:
            if isinstance(r, dict):
                date_str = r.get('transactionDate', r.get('filingDate', ''))
                if date_str and date_str >= cutoff:
                    recent.append(r)
        
        if not recent:
            signals[ticker] = {
                'net_buy_count': 0,
                'net_buy_value': 0,
                'insider_buy_ratio': 0.5,  # neutral
                'cluster_buy': 0,
                'ceo_cfo_buy': 0,
                'n_transactions': 0,
            }
            continue
        
        buys = [r for r in recent if r.get('transactionType', '').lower() in 
                ['purchase', 'buy', 'p-purchase']]
        sells = [r for r in recent if r.get('transactionType', '').lower() in 
                 ['sale', 'sell', 's-sale', 's-sale+oe']]
        
        net_buy_count = len(buys) - len(sells)
        
        # Net buy value
        buy_value = sum(float(r.get('securitiesTransacted', 0) or 0) * 
                       float(r.get('price', 0) or 0) for r in buys)
        sell_value = sum(float(r.get('securitiesTransacted', 0) or 0) * 
                        float(r.get('price', 0) or 0) for r in sells)
        net_buy_value = buy_value - sell_value
        
        # Buy ratio
        total = len(buys) + len(sells)
        buy_ratio = len(buys) / total if total > 0 else 0.5
        
        # Cluster buy: 3+ unique insiders buying
        unique_buyers = set()
        for r in buys:
            name = r.get('reportingName', r.get(' insiderRelation', ''))
            if name:
                unique_buyers.add(name)
        cluster = 1 if len(unique_buyers) >= 3 else 0
        
        # CEO/CFO buy
        ceo_cfo = 0
        for r in buys:
            title = str(r.get('typeOfOwner', r.get('position', ''))).lower()
            if 'ceo' in title or 'cfo' in title or 'chief' in title:
                ceo_cfo = 1
                break
        
        signals[ticker] = {
            'net_buy_count': net_buy_count,
            'net_buy_value': net_buy_value,
            'insider_buy_ratio': buy_ratio,
            'cluster_buy': cluster,
            'ceo_cfo_buy': ceo_cfo,
            'n_transactions': len(recent),
        }
    
    return signals


def compute_ic_for_insider_signals(tickers, signals, prices_df, forward_days=30):
    """Compute IC (Spearman rank correlation) between insider signals and forward returns."""
    
    # Use latest signal vs recent forward return
    # Since insider data is cross-sectional (latest snapshot), we test:
    # signal(today) vs return(today → today+30d)
    
    price_dates = sorted(prices_df.index.astype(str))
    latest_price_date = price_dates[-1]
    
    # Forward return from latest available date
    entry_date = price_dates[-min(forward_days+5, len(price_dates))]
    exit_date = latest_price_date
    
    if entry_date not in prices_df.index or exit_date not in prices_df.index:
        print(f"❌ Price dates not found: {entry_date} or {exit_date}")
        return None
    
    fwd_ret = (prices_df.loc[exit_date] / prices_df.loc[entry_date] - 1).dropna()
    
    # Build signal DataFrame
    signal_names = ['net_buy_count', 'net_buy_value', 'insider_buy_ratio', 
                    'cluster_buy', 'ceo_cfo_buy']
    
    results = {}
    for sig_name in signal_names:
        sig_values = {}
        for t in tickers:
            if t in signals and t in fwd_ret.index:
                val = signals[t].get(sig_name, None)
                if val is not None:
                    sig_values[t] = val
        
        if len(sig_values) < 30:
            results[sig_name] = {'ic': np.nan, 'n': len(sig_values), 'status': 'too few'}
            continue
        
        common = set(sig_values.keys()) & set(fwd_ret.index)
        if len(common) < 30:
            results[sig_name] = {'ic': np.nan, 'n': len(common), 'status': 'too few common'}
            continue
        
        sig_arr = np.array([sig_values[t] for t in common])
        ret_arr = np.array([fwd_ret[t] for t in common])
        
        valid = ~(np.isnan(sig_arr) | np.isnan(ret_arr))
        if valid.sum() < 30:
            results[sig_name] = {'ic': np.nan, 'n': int(valid.sum()), 'status': 'too few valid'}
            continue
        
        ic, pval = spearmanr(sig_arr[valid], ret_arr[valid])
        results[sig_name] = {
            'ic': round(float(ic), 4),
            'pval': round(float(pval), 4),
            'n': int(valid.sum()),
            'significant': pval < 0.05,
            'status': '✅' if pval < 0.05 else '❌',
        }
    
    return results


def compute_time_series_ic(raw_data, prices_df, lookback_days=90, forward_days=30):
    """Compute time-series IC by rolling quarterly snapshots.
    
    More robust than single cross-section: tests signal at multiple time points.
    """
    price_dates = sorted(prices_df.index.astype(str))
    
    # Quarterly test dates (every 63 trading days ≈ 3 months)
    test_dates = price_dates[::63]
    # Need enough history for forward return
    test_dates = [d for d in test_dates if d <= price_dates[-(forward_days+5)]]
    test_dates = test_dates[-8:]  # Last 8 quarters ≈ 2 years
    
    if len(test_dates) < 3:
        return None
    
    signal_names = ['net_buy_count', 'net_buy_value', 'insider_buy_ratio', 
                    'cluster_buy', 'ceo_cfo_buy']
    
    all_ics = {s: [] for s in signal_names}
    
    for test_date in test_dates:
        # Forward return
        td_idx = price_dates.index(test_date)
        if td_idx + forward_days >= len(price_dates):
            continue
        exit_date = price_dates[td_idx + forward_days]
        
        fwd_ret = (prices_df.loc[exit_date] / prices_df.loc[test_date] - 1).dropna()
        
        # For each ticker, compute insider signals as of test_date
        # (using records filed before test_date)
        cutoff = (pd.Timestamp(test_date) - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        for sig_name in signal_names:
            sig_values = {}
            for ticker, records in raw_data.items():
                if not records or not isinstance(records, list):
                    continue
                if ticker not in fwd_ret.index:
                    continue
                
                # Filter records before test_date
                relevant = [r for r in records if isinstance(r, dict) 
                           and r.get('transactionDate', r.get('filingDate', '')) <= test_date
                           and r.get('transactionDate', r.get('filingDate', '')) >= cutoff]
                
                if not relevant:
                    continue
                
                buys = [r for r in relevant if r.get('transactionType', '').lower() in 
                        ['purchase', 'buy', 'p-purchase']]
                sells = [r for r in relevant if r.get('transactionType', '').lower() in 
                         ['sale', 'sell', 's-sale']]
                
                if sig_name == 'net_buy_count':
                    sig_values[ticker] = len(buys) - len(sells)
                elif sig_name == 'net_buy_value':
                    bv = sum(float(r.get('securitiesTransacted', 0) or 0) * 
                            float(r.get('price', 0) or 0) for r in buys)
                    sv = sum(float(r.get('securitiesTransacted', 0) or 0) * 
                            float(r.get('price', 0) or 0) for r in sells)
                    sig_values[ticker] = bv - sv
                elif sig_name == 'insider_buy_ratio':
                    total = len(buys) + len(sells)
                    sig_values[ticker] = len(buys) / total if total > 0 else 0.5
                elif sig_name == 'cluster_buy':
                    unique_buyers = set(r.get('reportingName', '') for r in buys)
                    sig_values[ticker] = 1 if len(unique_buyers) >= 3 else 0
                elif sig_name == 'ceo_cfo_buy':
                    has = 0
                    for r in buys:
                        title = str(r.get('typeOfOwner', '')).lower()
                        if 'ceo' in title or 'cfo' in title:
                            has = 1
                            break
                    sig_values[ticker] = has
            
            common = set(sig_values.keys()) & set(fwd_ret.index)
            if len(common) < 30:
                continue
            
            sig_arr = np.array([sig_values[t] for t in common])
            ret_arr = np.array([fwd_ret[t] for t in common])
            valid = ~(np.isnan(sig_arr) | np.isnan(ret_arr))
            if valid.sum() < 30:
                continue
            
            ic, _ = spearmanr(sig_arr[valid], ret_arr[valid])
            if not np.isnan(ic):
                all_ics[sig_name].append(ic)
    
    # Summary
    results = {}
    for sig_name, ics in all_ics.items():
        if len(ics) < 2:
            results[sig_name] = {'mean_ic': np.nan, 'icir': np.nan, 'n_quarters': len(ics), 'status': 'insufficient'}
            continue
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        icir = mean_ic / std_ic if std_ic > 0 else 0
        t_stat = icir * np.sqrt(len(ics))
        results[sig_name] = {
            'mean_ic': round(float(mean_ic), 4),
            'std_ic': round(float(std_ic), 4),
            'icir': round(float(icir), 3),
            't_stat': round(float(t_stat), 2),
            'n_quarters': len(ics),
            'significant': abs(t_stat) > 1.96,
            'ics_per_quarter': [round(float(x), 4) for x in ics],
            'status': '✅' if abs(t_stat) > 1.96 else '❌',
        }
    
    return results


def main():
    print("=" * 60)
    print("  🦅 Falcon Insider Trading IC Validation")
    print("=" * 60)
    t0 = time.time()
    
    # Load tickers
    tickers = load_spx_tickers()
    print(f"📋 SPX Universe: {len(tickers)} tickers")
    
    # Download insider trading data
    raw_data = download_all_insider(tickers, FMP_KEY, max_tickers=100)
    
    # Count records
    total_records = sum(len(v) for v in raw_data.values() if isinstance(v, list))
    tickers_with_data = sum(1 for v in raw_data.values() if isinstance(v, list) and len(v) > 0)
    print(f"\n📊 Data summary: {tickers_with_data} tickers with data, {total_records} total records")
    
    # Load prices
    print("\n📊 Loading prices...")
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {prices.shape}")
    
    # Cross-sectional IC (single snapshot)
    print("\n" + "=" * 60)
    print("  Cross-Sectional IC (latest snapshot)")
    print("=" * 60)
    
    signals = compute_insider_signals(raw_data, lookback_days=90)
    cs_ic = compute_ic_for_insider_signals(tickers, signals, prices, forward_days=30)
    
    if cs_ic:
        print(f"\n  {'Signal':<25} {'IC':>8} {'p-value':>8} {'N':>6} {'Status':>6}")
        print("  " + "-" * 55)
        for sig_name, result in cs_ic.items():
            if isinstance(result.get('ic'), float) and not np.isnan(result['ic']):
                print(f"  {sig_name:<25} {result['ic']:>+8.4f} {result['pval']:>8.4f} {result['n']:>6} {result['status']:>6}")
            else:
                print(f"  {sig_name:<25} {'N/A':>8} {'N/A':>8} {result.get('n',0):>6} {result.get('status',''):>6}")
    
    # Time-series IC (quarterly rolling)
    print("\n" + "=" * 60)
    print("  Time-Series IC (quarterly rolling, 2 years)")
    print("=" * 60)
    
    ts_ic = compute_time_series_ic(raw_data, prices, lookback_days=90, forward_days=30)
    
    if ts_ic:
        print(f"\n  {'Signal':<25} {'Mean IC':>8} {'ICIR':>8} {'t-stat':>8} {'Quarters':>8} {'Status':>6}")
        print("  " + "-" * 65)
        for sig_name, result in ts_ic.items():
            if isinstance(result.get('mean_ic'), float) and not np.isnan(result['mean_ic']):
                print(f"  {sig_name:<25} {result['mean_ic']:>+8.4f} {result['icir']:>8.3f} {result['t_stat']:>8.2f} {result['n_quarters']:>8} {result['status']:>6}")
            else:
                print(f"  {sig_name:<25} {'N/A':>8} {'N/A':>8} {'N/A':>8} {result.get('n_quarters',0):>8} {result.get('status',''):>6}")
    
    # Verdict
    print("\n" + "=" * 60)
    print("  📋 VERDICT")
    print("=" * 60)
    
    if ts_ic:
        significant = [name for name, r in ts_ic.items() 
                      if isinstance(r.get('t_stat'), float) and abs(r['t_stat']) > 1.96]
        if significant:
            print(f"\n  ✅ Significant signals: {significant}")
            print(f"  → These can be added to Falcon as a new factor group")
        else:
            print(f"\n  ❌ No statistically significant insider trading signals")
            print(f"  → Insider trading data does not add value to Falcon")
            
            # Check if any are marginally interesting
            marginal = [name for name, r in ts_ic.items() 
                       if isinstance(r.get('t_stat'), float) and abs(r['t_stat']) > 1.0]
            if marginal:
                print(f"  ⚠️ Marginally interesting (t>1.0): {marginal}")
    
    elapsed = time.time() - t0
    print(f"\n⏱️ Total: {elapsed:.0f}s")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'tickers_with_data': tickers_with_data,
        'total_records': total_records,
        'cross_sectional_ic': cs_ic,
        'time_series_ic': ts_ic,
    }
    output_path = DATA_DIR / "insider_trading_ic_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"💾 Saved: {output_path}")


if __name__ == '__main__':
    main()
