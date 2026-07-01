#!/usr/bin/env python3
"""
T0.2: Download Russell 2000 price data (2025-01-01 to 2026-06-30)
using yfinance and merge with existing data.

Outputs:
  - russell_prices_2025_2026.json (new download only)
  - russell_prices_updated.json (merged: original + new data)

Red line: NEVER modify russell_prices.json
"""
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, date

import pandas as pd
import yfinance as yf

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
SNAPSHOTS_DIR = PROJECT_ROOT / "data" / "fmp_premium" / "snapshots"

SOURCE_FILE = SNAPSHOTS_DIR / "russell_prices.json"
NEW_DATA_FILE = SNAPSHOTS_DIR / "russell_prices_2025_2026.json"
MERGED_FILE = SNAPSHOTS_DIR / "russell_prices_updated.json"
LOG_FILE = SNAPSHOTS_DIR / "russell_download.log"

START_DATE = "2025-01-01"
END_DATE = "2026-06-30"

BATCH_SIZE = 50  # tickers per batch to avoid timeout
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries

# ============================================================
# Logging setup
# ============================================================
logger = logging.getLogger("russell_download")
logger.setLevel(logging.INFO)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(fh)


def load_existing_prices(filepath: Path) -> dict:
    """Load existing prices JSON file. Returns dict {ticker: [records]}."""
    logger.info(f"Loading existing prices from {filepath.name}...")
    with open(filepath, "r") as f:
        data = json.load(f)
    tickers = list(data.keys())
    logger.info(f"  Loaded {len(tickers)} tickers")
    if tickers:
        first = tickers[0]
        dates = [r["date"] for r in data[first]]
        logger.info(f"  Date range: {min(dates)} ~ {max(dates)}")
    return data


def download_batch(tickers: list[str], start: str, end: str) -> dict:
    """
    Download price data for a batch of tickers using yfinance.
    Returns dict {ticker: [records]} or empty on failure.
    """
    results = {}
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )
        
        if data.empty:
            logger.warning("  Download returned empty data")
            return results
            
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    df = data
                else:
                    # Multi-level columns: (Price, Ticker)
                    if ticker not in data.columns.get_level_values(0):
                        logger.debug(f"  {ticker}: not found in download result")
                        continue
                    df = data[ticker]
                
                if df is None or df.empty:
                    logger.debug(f"  {ticker}: empty dataframe")
                    continue
                
                # Drop rows with all NaN
                df = df.dropna(how="all")
                if df.empty:
                    continue
                    
                records = []
                for idx, row in df.iterrows():
                    d = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                    
                    # Skip if all price fields are NaN
                    open_val = float(row.get("Open", 0) or 0)
                    close_val = float(row.get("Close", 0) or 0)
                    high_val = float(row.get("High", 0) or 0)
                    low_val = float(row.get("Low", 0) or 0)
                    vol_val = float(row.get("Volume", 0) or 0)
                    
                    if open_val == 0 and close_val == 0:
                        continue  # Skip zero-price rows
                        
                    records.append({
                        "date": d,
                        "open": round(open_val, 4),
                        "high": round(high_val, 4),
                        "low": round(low_val, 4),
                        "close": round(close_val, 4),
                        "volume": round(vol_val, 2),
                        "vwap": None,  # yfinance doesn't provide VWAP
                    })
                
                if records:
                    results[ticker] = records
                    
            except Exception as e:
                logger.debug(f"  {ticker}: error processing - {e}")
                
    except Exception as e:
        logger.error(f"  Batch download failed: {e}")
        
    return results


def merge_prices(existing: dict, new_data: dict) -> dict:
    """
    Merge existing prices with new data.
    New data overwrites existing dates (more recent/more accurate).
    """
    merged = {}
    stats = {"kept_from_existing": 0, "added_new": 0, "updated": 0}
    
    # Start with existing data
    for ticker, records in existing.items():
        date_map = {r["date"]: r for r in records}
        merged[ticker] = date_map
        stats["kept_from_existing"] += len(records)
    
    # Overlay new data
    for ticker, records in new_data.items():
        if ticker not in merged:
            merged[ticker] = {}
            stats["added_new"] += len(records)
        else:
            existing_dates = set(merged[ticker].keys())
            new_dates = set(r["date"] for r in records)
            stats["added_new"] += len(new_dates - existing_dates)
            stats["updated"] += len(new_dates & existing_dates)
        
        for r in records:
            merged[ticker][r["date"]] = r
    
    # Convert back to lists
    result = {}
    for ticker, date_map in merged.items():
        result[ticker] = sorted(date_map.values(), key=lambda x: x["date"])
    
    return result, stats


def main():
    logger.info("=" * 60)
    logger.info("T0.2 Russell 2000 Price Download (2025-2026)")
    logger.info("=" * 60)
    
    # Step 1: Load existing prices
    existing = load_existing_prices(SOURCE_FILE)
    all_tickers = sorted(existing.keys())
    logger.info(f"Total tickers to download: {len(all_tickers)}")
    
    # Step 2: Download in batches
    new_data = {}
    failed_tickers = []
    success_count = 0
    
    total_batches = (len(all_tickers) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, len(all_tickers), BATCH_SIZE):
        batch = all_tickers[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        
        logger.info(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} tickers) ---")
        
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                batch_data = download_batch(batch, START_DATE, END_DATE)
                
                if batch_data:
                    new_data.update(batch_data)
                    success_count += len(batch_data)
                    logger.info(f"  ✓ Got {len(batch_data)}/{len(batch)} tickers")
                    break
                else:
                    logger.warning(f"  Attempt {attempt}: no data returned")
                    if attempt < RETRY_ATTEMPTS:
                        logger.info(f"  Retrying in {RETRY_DELAY}s...")
                        time.sleep(RETRY_DELAY)
                        
            except Exception as e:
                logger.error(f"  Attempt {attempt} failed: {e}")
                if attempt < RETRY_ATTEMPTS:
                    logger.info(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
        
        # Track failed tickers
        failed_in_batch = [t for t in batch if t not in new_data]
        failed_tickers.extend(failed_in_batch)
        
        # Rate limiting between batches
        if batch_idx + BATCH_SIZE < len(all_tickers):
            time.sleep(1)
    
    # Step 3: Save new data
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Step 3: Saving new data (2025-2026)")
    logger.info(f"{'=' * 60}")
    
    with open(NEW_DATA_FILE, "w") as f:
        json.dump(new_data, f, indent=2)
    
    new_total_records = sum(len(r) for r in new_data.values())
    logger.info(f"  Saved {len(new_data)} tickers, {new_total_records} records")
    logger.info(f"  File: {NEW_DATA_FILE}")
    
    # Step 4: Merge with existing
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Step 4: Merging with existing data")
    logger.info(f"{'=' * 60}")
    
    merged, stats = merge_prices(existing, new_data)
    
    logger.info(f"  Existing records: {stats['kept_from_existing']:,}")
    logger.info(f"  New records added: {stats['added_new']:,}")
    logger.info(f"  Records updated: {stats['updated']:,}")
    
    # Step 5: Save merged result
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Step 5: Saving merged result")
    logger.info(f"{'=' * 60}")
    
    with open(MERGED_FILE, "w") as f:
        json.dump(merged, f, indent=2)
    
    merged_total_records = sum(len(r) for r in merged.values())
    logger.info(f"  Saved {len(merged)} tickers, {merged_total_records:,} records")
    logger.info(f"  File: {MERGED_FILE}")
    
    # Verify date range
    all_dates = set()
    for records in merged.values():
        all_dates.update(r["date"] for r in records)
    
    if all_dates:
        logger.info(f"  Date range: {min(all_dates)} ~ {max(all_dates)}")
    
    # Report failures
    if failed_tickers:
        logger.warning(f"\n⚠️ Failed to download {len(failed_tickers)} tickers:")
        for i in range(0, len(failed_tickers), 20):
            logger.warning(f"  {', '.join(failed_tickers[i:i+20])}")
    
    # Final summary
    logger.info(f"\n{'=' * 60}")
    logger.info("✅ T0.2 Complete")
    logger.info(f"{'=' * 60}")
    logger.info(f"  New data file: {NEW_DATA_FILE}")
    logger.info(f"  Merged file:   {MERGED_FILE}")
    logger.info(f"  Download log:  {LOG_FILE}")
    logger.info(f"  Total tickers: {len(merged)}")
    logger.info(f"  Total records: {merged_total_records:,}")
    
    return 0 if not failed_tickers else 1


if __name__ == "__main__":
    sys.exit(main())
