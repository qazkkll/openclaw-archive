#!/usr/bin/env python3
"""
拉取R2K股票2018-2021年历史价格数据
用yfinance批量下载
"""
import json
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "fmp_premium" / "snapshots"
FALCON_DIR = PROJECT_ROOT / "data" / "falcon"

MERGED_FILE = SNAPSHOTS_DIR / "russell_prices_updated.json"
EXTENDED_FILE = SNAPSHOTS_DIR / "russell_prices_extended.json"

START_DATE = "2018-01-01"
END_DATE = "2021-12-31"

BATCH_SIZE = 50  # yfinance batch size


def main():
    print("=== R2K Historical Price Extension (2018-2021) ===\n")
    
    # Load existing data
    print("Loading existing data...")
    with open(MERGED_FILE) as f:
        existing = json.load(f)
    
    print(f"Existing tickers: {len(existing)}")
    
    # Check which tickers need more history
    tickers_to_fetch = []
    for ticker, records in existing.items():
        if not records:
            tickers_to_fetch.append(ticker)
            continue
        dates = sorted([r['date'] for r in records if 'date' in r])
        if dates and dates[0] > "2019-01-01":
            tickers_to_fetch.append(ticker)
    
    print(f"Tickers needing 2018-2021 data: {len(tickers_to_fetch)}")
    
    if not tickers_to_fetch:
        print("All tickers already have sufficient history!")
        return
    
    # Fetch in batches using yfinance download
    all_new_data = {}
    total_batches = (len(tickers_to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(tickers_to_fetch), BATCH_SIZE):
        batch = tickers_to_fetch[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"\nBatch {batch_num}/{total_batches}: {len(batch)} tickers...")
        
        try:
            # yfinance batch download
            data = yf.download(batch, start=START_DATE, end=END_DATE, 
                             group_by='ticker', progress=False, threads=True)
            
            if data is not None and not data.empty:
                for ticker in batch:
                    try:
                        if len(batch) == 1:
                            df = data
                        else:
                            df = data[ticker] if ticker in data.columns.get_level_values(0) else None
                        
                        if df is not None and not df.empty:
                            df = df.dropna(subset=['Close'])
                            records = []
                            for idx, row in df.iterrows():
                                records.append({
                                    "date": idx.strftime("%Y-%m-%d"),
                                    "open": float(row['Open']),
                                    "high": float(row['High']),
                                    "low": float(row['Low']),
                                    "close": float(row['Close']),
                                    "volume": int(row['Volume']),
                                })
                            if records:
                                all_new_data[ticker] = records
                    except Exception as e:
                        continue
        except Exception as e:
            print(f"  Batch error: {e}")
        
        found = len([t for t in batch if t in all_new_data])
        print(f"  Got data for {found}/{len(batch)} tickers")
        
        # Rate limiting
        if batch_num < total_batches:
            time.sleep(1)
    
    print(f"\n=== Summary ===")
    print(f"Tickers fetched: {len(all_new_data)}")
    
    # Merge with existing data
    print("\nMerging with existing data...")
    merged = {}
    extended_count = 0
    
    for ticker in existing:
        old_records = existing[ticker]
        old_dates = {r['date'] for r in old_records if 'date' in r}
        
        if ticker in all_new_data:
            new_records = all_new_data[ticker]
            # Only add records that don't exist
            to_add = [r for r in new_records if r['date'] not in old_dates]
            if to_add:
                merged[ticker] = sorted(to_add + old_records, key=lambda x: x['date'])
                extended_count += 1
            else:
                merged[ticker] = old_records
        else:
            merged[ticker] = old_records
    
    print(f"Extended {extended_count} tickers with earlier data")
    
    # Save
    print(f"\nSaving to {EXTENDED_FILE}...")
    with open(EXTENDED_FILE, "w") as f:
        json.dump(merged, f)
    
    # Stats
    total_records = sum(len(v) for v in merged.values())
    print(f"Total records: {total_records:,}")
    
    # Show date range improvement
    sample_tickers = [t for t in list(merged.keys())[:10] if merged[t]]
    print("\nDate range comparison (sample):")
    for t in sample_tickers[:5]:
        old_dates = sorted([r['date'] for r in existing[t] if 'date' in r])
        new_dates = sorted([r['date'] for r in merged[t] if 'date' in r])
        print(f"  {t}: {old_dates[0]}→{old_dates[-1]} ({len(old_dates)}d) → {new_dates[0]}→{new_dates[-1]} ({len(new_dates)}d)")


if __name__ == "__main__":
    main()
