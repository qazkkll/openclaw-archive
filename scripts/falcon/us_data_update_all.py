#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
us_data_update_all.py — Comprehensive US Stock Data Updater
============================================================
Updates ALL Falcon data sources in one run:
  1. OHLCV prices       → data/falcon/us_prices_daily.parquet
  2. VIX historical     → data/us/vix_10y.parquet
  3. Sector ETFs        → data/us/sector_etf_daily.parquet
  4. SPX index          → data/us/spx_daily.parquet
  5. FMP fundamentals   → data/falcon/fmp_*.json
  6. FMP news           → data/falcon/fmp_news_cache.json
  7. FMP earnings cal   → data/falcon/earnings_calendar.json
  8. FMP grades/ratings → data/falcon/fmp_grades.json

Usage:
  python3 us_data_update_all.py [--prices] [--fmp] [--macro] [--news] [--all]

Default: --all
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Configuration ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = PROJECT_ROOT / "data"
FALCON_DIR = DATA_DIR / "falcon"
US_DIR = DATA_DIR / "us"

# Output paths
PRICES_PARQUET = FALCON_DIR / "us_prices_daily.parquet"
VIX_PARQUET = US_DIR / "vix_10y.parquet"
SECTOR_PARQUET = US_DIR / "sector_etf_daily.parquet"
SPX_PARQUET = US_DIR / "spx_daily.parquet"
NEWS_CACHE = FALCON_DIR / "fmp_news_cache.json"
EARNINGS_CAL = FALCON_DIR / "earnings_calendar.json"
GRADES_FILE = FALCON_DIR / "fmp_grades.json"
FMP_FUNDAMENTALS = {
    "ratios": FALCON_DIR / "fmp_ratios_historical.json",
    "key_metrics": FALCON_DIR / "fmp_key_metrics.json",
    "financial_growth": FALCON_DIR / "fmp_financial_growth.json",
    "analyst_estimates": FALCON_DIR / "analyst_historical.json",
}

# FMP API endpoints
FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_ENDPOINTS = {
    "ratios": "/ratios-ttm?symbol={ticker}&apikey={key}",
    "key_metrics": "/key-metrics-ttm?symbol={ticker}&apikey={key}",
    "financial_growth": "/financial-growth?symbol={ticker}&period=quarter&limit=10&apikey={key}",
    "analyst_estimates": "/analyst-estimates?symbol={ticker}&period=quarter&limit=10&apikey={key}",
}

# Sector ETFs (11 GICS sectors — XLK appears twice in original spec, deduplicated)
SECTOR_ETFS = ["XLK", "XLV", "XLF", "XLE", "XLI", "XLB", "XLP", "XLU", "XLC", "XLRE", "XLY"]

# yfinance batch size (too many tickers at once can timeout)
YF_BATCH_SIZE = 50

# FMP rate limit: ~300 req/min on paid plan; be conservative
FMP_WORKERS = 4
YF_WORKERS = 8

# History duration
YEARS_HISTORY = 10
SECTOR_YEARS = 1

# Top N tickers for news fetch
NEWS_TOP_N = 20

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("us_data_update")

# ─── Helpers ──────────────────────────────────────────────────────────────────


def load_env():
    """Load API keys from .env file."""
    env_path = PROJECT_ROOT / ".env"
    keys = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    keys[k.strip()] = v.strip()
    # Also try dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass
    return keys


def get_fmp_api_key() -> str:
    """Get FMP API key from env or .env file."""
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        env_keys = load_env()
        key = env_keys.get("FMP_API_KEY", "")
    if not key:
        log.warning("FMP_API_KEY not found — FMP endpoints will be skipped")
    return key


def load_spx_tickers() -> List[str]:
    """Load SPX tickers from features_v02.parquet or fallback to static list."""
    parquet = FALCON_DIR / "features_v02.parquet"
    if parquet.exists():
        try:
            df = pd.read_parquet(parquet, columns=["ticker"])
            tickers = sorted(df["ticker"].unique().tolist())
            log.info(f"Loaded {len(tickers)} SPX tickers from features_v02.parquet")
            return tickers
        except Exception as e:
            log.warning(f"Failed to load tickers from parquet: {e}")

    # Fallback: load from us_all_tickers.json
    tickers_file = US_DIR / "us_all_tickers.json"
    if tickers_file.exists():
        try:
            with open(tickers_file) as f:
                tickers = json.load(f)
            if isinstance(tickers, list) and tickers:
                log.info(f"Loaded {len(tickers)} tickers from us_all_tickers.json")
                return sorted(tickers)
        except Exception:
            pass

    log.error("No ticker source found — cannot proceed")
    sys.exit(1)


def load_json_cache(path: Path) -> Dict[str, Any]:
    """Load existing JSON cache, return empty dict on failure."""
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            log.info(f"Loaded cache: {path.name} ({len(data)} entries)")
            return data
        except Exception as e:
            log.warning(f"Failed to load {path.name}: {e}")
    return {}


def save_json_cache(path: Path, data: Dict[str, Any]):
    """Save data to JSON with atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)
    log.info(f"Saved {path.name} ({len(data)} entries)")


def fmp_api_get(endpoint: str, params: str = "", timeout: int = 30) -> Any:
    """Make a GET request to FMP API. Returns parsed JSON or None."""
    import urllib.request
    import urllib.error

    url = f"{FMP_BASE}{endpoint}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            data = json.loads(raw)
            return data
    except urllib.error.HTTPError as e:
        if e.code == 429:
            log.warning(f"FMP rate limit hit on {endpoint}")
            time.sleep(2)
            return None
        elif e.code == 401:
            log.error(f"FMP auth error: {endpoint}")
            return None
        else:
            log.warning(f"FMP HTTP {e.code}: {endpoint}")
            return None
    except Exception as e:
        log.warning(f"FMP request failed: {endpoint} — {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 1. OHLCV Prices (yfinance, 10y)
# ═══════════════════════════════════════════════════════════════════════════════


def _yf_fetch_single(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV for a single ticker via yfinance."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        # Normalize columns
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" not in df.columns:
            # Sometimes the index is named 'Date'
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["ticker"] = ticker
        # Keep only OHLCV
        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        return df
    except Exception as e:
        log.debug(f"yfinance failed for {ticker}: {e}")
        return None


def update_prices(tickers: List[str]) -> Tuple[int, int]:
    """
    Update OHLCV prices for all tickers.
    Returns (success_count, new_row_count).
    """
    log.info(f"{'='*60}")
    log.info(f"Updating OHLCV prices for {len(tickers)} tickers...")
    log.info(f"{'='*60}")

    # Load existing data for incremental update
    existing_df = pd.DataFrame()
    last_date_map: Dict[str, str] = {}

    if PRICES_PARQUET.exists():
        try:
            existing_df = pd.read_parquet(PRICES_PARQUET)
            existing_df["date"] = existing_df["date"].astype(str)
            # Build per-ticker max date map
            for ticker in tickers:
                sub = existing_df[existing_df["ticker"] == ticker]
                if not sub.empty:
                    last_date_map[ticker] = sub["date"].max()
            log.info(f"Existing prices: {len(existing_df):,} rows, "
                     f"{existing_df['ticker'].nunique()} tickers")

            # Check if all tickers are up to date (within 1 day)
            if last_date_map:
                overall_max = max(last_date_map.values())
                days_gap = (datetime.now() - datetime.strptime(overall_max, "%Y-%m-%d")).days
                if days_gap <= 1:
                    log.info(f"  Prices already up to date (last: {overall_max})")
                    return len(tickers), 0
        except Exception as e:
            log.warning(f"Failed to load existing prices: {e}")

    # Date range
    end_date = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=YEARS_HISTORY * 365)).strftime("%Y-%m-%d")

    # Fetch in batches using yfinance download for speed
    all_new_frames = []
    success_count = 0
    fail_count = 0
    t0 = time.time()

    for i in range(0, len(tickers), YF_BATCH_SIZE):
        batch = tickers[i : i + YF_BATCH_SIZE]

        # Determine start date per-batch: use earliest last_date among batch
        batch_starts = []
        for t in batch:
            if t in last_date_map:
                # Start from day after last known date
                try:
                    dt = datetime.strptime(last_date_map[t], "%Y-%m-%d")
                    batch_starts.append((dt + timedelta(days=1)).strftime("%Y-%m-%d"))
                except ValueError:
                    batch_starts.append(default_start)
            else:
                batch_starts.append(default_start)

        # Use the earliest start for batch download (yfinance doesn't support per-ticker start)
        batch_start = min(batch_starts)

        # Skip batch if all tickers are up to date
        if all(
            ds > end_date for ds in batch_starts
        ):
            success_count += len(batch)
            continue

        try:
            if len(batch) == 1:
                df = yf.download(
                    batch[0],
                    start=batch_start,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if df is not None and not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    df["ticker"] = batch[0]
                    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
                    df = df[[c for c in keep if c in df.columns]]
                    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                    # Filter to only new rows
                    if batch[0] in last_date_map:
                        df = df[df["date"] > last_date_map[batch[0]]]
                    if not df.empty:
                        all_new_frames.append(df)
                    success_count += 1
            else:
                data = yf.download(
                    batch,
                    start=batch_start,
                    end=end_date,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if data is None or data.empty:
                    fail_count += len(batch)
                    continue

                for ticker in batch:
                    try:
                        if ticker not in data.columns.get_level_values(0):
                            fail_count += 1
                            continue
                        tdf = data[ticker].dropna(subset=["Close"])
                        if tdf.empty:
                            fail_count += 1
                            continue
                        tdf = tdf.reset_index()
                        tdf.columns = [c.lower().replace(" ", "_") for c in tdf.columns]
                        tdf["ticker"] = ticker
                        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
                        tdf = tdf[[c for c in keep if c in tdf.columns]]
                        tdf["date"] = pd.to_datetime(tdf["date"]).dt.strftime("%Y-%m-%d")
                        # Incremental: only new rows
                        if ticker in last_date_map:
                            tdf = tdf[tdf["date"] > last_date_map[ticker]]
                        if not tdf.empty:
                            all_new_frames.append(tdf)
                        success_count += 1
                    except Exception:
                        fail_count += 1
        except Exception as e:
            log.warning(f"Batch download failed for {batch[:3]}...: {e}")
            fail_count += len(batch)

        done = min(i + YF_BATCH_SIZE, len(tickers))
        if done % 200 == 0 or done == len(tickers):
            log.info(
                f"  Progress: {done}/{len(tickers)} "
                f"({success_count} ok, {fail_count} failed, "
                f"{time.time()-t0:.0f}s)"
            )

    # Merge and save
    new_rows = 0
    if all_new_frames:
        new_df = pd.concat(all_new_frames, ignore_index=True)
        new_df = new_df.drop_duplicates(subset=["ticker", "date"], keep="last")
        new_rows = len(new_df)

        if not existing_df.empty:
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
        else:
            combined = new_df

        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
        combined.to_parquet(PRICES_PARQUET, index=False)
        log.info(f"Saved prices: {len(combined):,} rows, {combined['ticker'].nunique()} tickers")
    elif existing_df.empty:
        log.error("No price data fetched and no existing data")
    else:
        log.info("No new price data to add — all up to date")

    elapsed = time.time() - t0
    log.info(
        f"Prices done: {success_count}/{len(tickers)} tickers, "
        f"+{new_rows:,} new rows, {elapsed:.0f}s"
    )
    return success_count, new_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 2. VIX Historical (yfinance ^VIX, 10y)
# ═══════════════════════════════════════════════════════════════════════════════


def update_vix() -> int:
    """Update VIX historical data. Returns new row count."""
    log.info(f"{'='*60}")
    log.info("Updating VIX (^VIX) 10y history...")
    log.info(f"{'='*60}")

    # Determine start date for incremental
    start_date = (datetime.now() - timedelta(days=YEARS_HISTORY * 365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    if VIX_PARQUET.exists():
        try:
            existing = pd.read_parquet(VIX_PARQUET)
            # Handle multi-level columns from previous yfinance download
            if isinstance(existing.columns, pd.MultiIndex):
                existing.columns = [c[0] if isinstance(c, tuple) else c for c in existing.columns]
            # Find last date
            if "Date" in existing.columns:
                last = pd.to_datetime(existing["Date"]).max()
            elif "date" in existing.columns:
                last = pd.to_datetime(existing["date"]).max()
            else:
                last = None
            if last is not None:
                days_gap = (datetime.now() - last).days
                if days_gap <= 1:
                    log.info(f"  VIX already up to date (last: {last.strftime('%Y-%m-%d')})")
                    return len(existing)
                start_date = (last + timedelta(days=1)).strftime("%Y-%m-%d")
                log.info(f"  Incremental from {start_date}")
        except Exception as e:
            log.warning(f"  Failed to read existing VIX: {e}")

    try:
        vix = yf.Ticker("^VIX")
        df = vix.history(start=start_date, end=end_date, auto_adjust=False)
        if df is None or df.empty:
            log.info("  No new VIX data")
            return 0

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        # Normalize: ensure we have date column
        if "date" not in df.columns:
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Strip timezone from date
        if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # Keep standard columns
        keep_cols = ["date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Merge with existing
        if VIX_PARQUET.exists():
            try:
                old = pd.read_parquet(VIX_PARQUET)
                if isinstance(old.columns, pd.MultiIndex):
                    old.columns = [c[0] if isinstance(c, tuple) else c for c in old.columns]
                if "Date" in old.columns:
                    old = old.rename(columns={"Date": "date"})
                old["date"] = pd.to_datetime(old["date"]).dt.strftime("%Y-%m-%d")
                keep_old = ["date", "open", "high", "low", "close", "volume"]
                old = old[[c for c in keep_old if c in old.columns]]
                combined = pd.concat([old, df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["date"], keep="last")
                combined = combined.sort_values("date").reset_index(drop=True)
                df = combined
            except Exception:
                pass

        df.to_parquet(VIX_PARQUET, index=False)
        log.info(f"  Saved VIX: {len(df):,} rows, {df['date'].min()} ~ {df['date'].max()}")
        return len(df)
    except Exception as e:
        log.error(f"  VIX update failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Sector ETFs (yfinance, 1y)
# ═══════════════════════════════════════════════════════════════════════════════


def update_sector_etfs() -> int:
    """Update sector ETF daily data. Returns new row count."""
    log.info(f"{'='*60}")
    log.info(f"Updating Sector ETFs ({len(SECTOR_ETFS)} sectors, 1y)...")
    log.info(f"{'='*60}")

    start_date = (datetime.now() - timedelta(days=SECTOR_YEARS * 365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Check existing for incremental
    # Track whether we already have recent data (skip if up-to-date)
    incremental_mode = False
    if SECTOR_PARQUET.exists():
        try:
            existing = pd.read_parquet(SECTOR_PARQUET)
            if "date" in existing.columns:
                last = pd.to_datetime(existing["date"]).max()
                # If data is from today or yesterday, nothing to fetch
                days_gap = (datetime.now() - last).days
                if days_gap <= 1:
                    log.info(f"  Sector ETFs already up to date (last: {last.strftime('%Y-%m-%d')})")
                    return len(existing)
                inc_start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
                if inc_start > start_date:
                    start_date = inc_start
                    incremental_mode = True
                    log.info(f"  Incremental from {start_date}")
        except Exception:
            pass

    try:
        # Download all sector ETFs in one batch
        data = yf.download(
            SECTOR_ETFS,
            start=start_date,
            end=end_date,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data is None or data.empty:
            if incremental_mode:
                log.info("  No new sector ETF data (market may not have opened yet)")
            else:
                log.warning("  No sector ETF data returned")
            # Return existing row count if we have data
            if SECTOR_PARQUET.exists():
                try:
                    return len(pd.read_parquet(SECTOR_PARQUET))
                except Exception:
                    pass
            return 0

        all_frames = []
        for etf in SECTOR_ETFS:
            try:
                if etf not in data.columns.get_level_values(0):
                    log.warning(f"  {etf}: no data in download")
                    continue
                edf = data[etf].dropna(subset=["Close"])
                if edf.empty:
                    continue
                edf = edf.reset_index()
                edf.columns = [c.lower().replace(" ", "_") for c in edf.columns]
                edf["ticker"] = etf
                keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
                edf = edf[[c for c in keep if c in edf.columns]]
                edf["date"] = pd.to_datetime(edf["date"]).dt.strftime("%Y-%m-%d")
                all_frames.append(edf)
            except Exception as e:
                log.warning(f"  {etf} processing failed: {e}")

        if not all_frames:
            log.warning("  No sector ETF data parsed")
            return 0

        new_df = pd.concat(all_frames, ignore_index=True)

        # Merge with existing
        if SECTOR_PARQUET.exists():
            try:
                old = pd.read_parquet(SECTOR_PARQUET)
                if "date" in old.columns:
                    old["date"] = old["date"].astype(str)
                    combined = pd.concat([old, new_df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
                    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
                    new_df = combined
            except Exception:
                pass

        new_df.to_parquet(SECTOR_PARQUET, index=False)
        log.info(
            f"  Saved sector ETFs: {len(new_df):,} rows, "
            f"{new_df['ticker'].nunique()} ETFs"
        )
        return len(new_df)
    except Exception as e:
        log.error(f"  Sector ETF update failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SPX Index (yfinance ^GSPC, 10y)
# ═══════════════════════════════════════════════════════════════════════════════


def update_spx() -> int:
    """Update S&P 500 index data. Returns new row count."""
    log.info(f"{'='*60}")
    log.info("Updating SPX (^GSPC) 10y history...")
    log.info(f"{'='*60}")

    start_date = (datetime.now() - timedelta(days=YEARS_HISTORY * 365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Incremental
    if SPX_PARQUET.exists():
        try:
            existing = pd.read_parquet(SPX_PARQUET)
            if "date" in existing.columns:
                last = pd.to_datetime(existing["date"]).max()
                days_gap = (datetime.now() - last).days
                if days_gap <= 1:
                    log.info(f"  SPX already up to date (last: {last.strftime('%Y-%m-%d')})")
                    return len(existing)
                inc_start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
                if inc_start > start_date:
                    start_date = inc_start
                    log.info(f"  Incremental from {start_date}")
        except Exception:
            pass

    try:
        spx = yf.Ticker("^GSPC")
        df = spx.history(start=start_date, end=end_date, auto_adjust=True)
        if df is None or df.empty:
            log.info("  No new SPX data")
            return 0

        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" not in df.columns:
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Strip timezone
        if hasattr(df["date"].dtype, "tz") and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["ticker"] = "^GSPC"
        keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]

        # Merge
        if SPX_PARQUET.exists():
            try:
                old = pd.read_parquet(SPX_PARQUET)
                if "date" in old.columns:
                    old["date"] = old["date"].astype(str)
                    combined = pd.concat([old, df], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["date"], keep="last")
                    combined = combined.sort_values("date").reset_index(drop=True)
                    df = combined
            except Exception:
                pass

        df.to_parquet(SPX_PARQUET, index=False)
        log.info(f"  Saved SPX: {len(df):,} rows, {df['date'].min()} ~ {df['date'].max()}")
        return len(df)
    except Exception as e:
        log.error(f"  SPX update failed: {e}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FMP Fundamentals (ratios, key-metrics, growth, analyst-estimates)
# ═══════════════════════════════════════════════════════════════════════════════


def _fetch_fmp_fundamental(ticker: str, endpoint_key: str, api_key: str) -> Optional[list]:
    """Fetch a single fundamental endpoint for one ticker."""
    endpoint = FMP_ENDPOINTS[endpoint_key]
    url = endpoint.format(ticker=ticker, key=api_key)
    data = fmp_api_get(url)
    if data is None:
        return None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some endpoints wrap in a key
        for key in ["symbol", "ratios", "metrics", "financialGrowth", "estimates"]:
            if key in data:
                return data[key]
        # If dict with ticker as key
        if ticker in data:
            val = data[ticker]
            return val if isinstance(val, list) else [val] if val else []
    return []


def _validate_fmp_key(api_key: str) -> bool:
    """Quick validation: try one cheap FMP stable endpoint. Returns True if key works."""
    data = fmp_api_get(f"/ratios-ttm?symbol=AAPL&apikey={api_key}", timeout=10)
    if data is not None and isinstance(data, list) and len(data) > 0:
        return True
    # Second check: quote endpoint
    data2 = fmp_api_get(f"/quote?symbol=AAPL&apikey={api_key}", timeout=10)
    if data2 is not None and isinstance(data2, list) and len(data2) > 0:
        return True
    return False


def update_fmp_fundamentals(tickers: List[str], api_key: str) -> Tuple[int, int]:
    """Update FMP fundamentals for all tickers. Returns (success, fail)."""
    log.info(f"{'='*60}")
    log.info(f"Updating FMP fundamentals ({len(tickers)} tickers, 4 endpoints)...")
    log.info(f"{'='*60}")

    # Validate API key first
    if not _validate_fmp_key(api_key):
        log.warning("  FMP API key invalid or expired — skipping fundamentals update")
        log.warning("  Existing cached data will be preserved")
        return 0, 0

    t0 = time.time()
    total_success = 0
    total_fail = 0

    for ep_name in FMP_ENDPOINTS:
        log.info(f"  Fetching {ep_name}...")
        cache = load_json_cache(FMP_FUNDAMENTALS[ep_name])
        success = 0
        fail = 0

        def fetch_one(ticker):
            return ticker, _fetch_fmp_fundamental(ticker, ep_name, api_key)

        with ThreadPoolExecutor(max_workers=FMP_WORKERS) as executor:
            futures = {executor.submit(fetch_one, t): t for t in tickers}
            done = 0
            for future in as_completed(futures):
                done += 1
                ticker = futures[future]
                try:
                    _, result = future.result()
                    if result and len(result) > 0:
                        # Merge: append new records, deduplicate by date
                        existing = cache.get(ticker, [])
                        if existing:
                            existing_map = {r.get("date", ""): r for r in existing}
                            for rec in result:
                                d = rec.get("date", "")
                                if d:
                                    # Keep newest version of each date
                                    if d not in existing_map or d >= max(existing_map.keys()):
                                        existing_map[d] = rec
                            cache[ticker] = sorted(existing_map.values(), key=lambda r: r.get("date", ""))
                        else:
                            cache[ticker] = result
                        success += 1
                    else:
                        fail += 1
                except Exception as e:
                    log.debug(f"  {ep_name} {ticker} error: {e}")
                    fail += 1

                if done % 50 == 0:
                    log.info(f"    {ep_name}: {done}/{len(tickers)} ({success} ok, {fail} fail)")

        save_json_cache(FMP_FUNDAMENTALS[ep_name], cache)
        total_success += success
        total_fail += fail
        log.info(f"  {ep_name} done: {success}/{len(tickers)} ({fail} failed)")

    elapsed = time.time() - t0
    log.info(
        f"FMP fundamentals done: {total_success}/{len(tickers) * 4} "
        f"endpoint-tickers, {elapsed:.0f}s"
    )
    return total_success, total_fail


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FMP News (latest 3 days for top 20 tickers)
# ═══════════════════════════════════════════════════════════════════════════════


def update_fmp_news(tickers: List[str], api_key: str) -> int:
    """Update FMP news cache for top N tickers. Returns new article count."""
    log.info(f"{'='*60}")
    log.info(f"Updating FMP news (top {NEWS_TOP_N} tickers, 3-day window)...")
    log.info(f"{'='*60}")

    # Validate API key
    if not _validate_fmp_key(api_key):
        log.warning("  FMP API key invalid or expired — skipping news update")
        return 0

    cache = load_json_cache(NEWS_CACHE)
    top_tickers = tickers[:NEWS_TOP_N]
    cutoff_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    total_new = 0
    t0 = time.time()

    def fetch_news(ticker):
        """Fetch news for a single ticker."""
        endpoint = f"/news/stock?symbol={ticker}&from={cutoff_date}&limit=50&apikey={api_key}"
        data = fmp_api_get(endpoint, timeout=15)
        return ticker, data if isinstance(data, list) else []

    with ThreadPoolExecutor(max_workers=FMP_WORKERS) as executor:
        futures = {executor.submit(fetch_news, t): t for t in top_tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                _, articles = future.result()
                if articles:
                    # Merge with existing
                    existing = cache.get(ticker, [])
                    existing_set = {(a.get("publishedDate", ""), a.get("title", "")) for a in existing}
                    for article in articles:
                        key = (article.get("publishedDate", ""), article.get("title", ""))
                        if key not in existing_set:
                            existing.append(article)
                    # Sort by date descending, keep last 100
                    existing.sort(key=lambda a: a.get("publishedDate", ""), reverse=True)
                    cache[ticker] = existing[:100]
                    total_new += len(articles)
            except Exception as e:
                log.warning(f"  News {ticker} failed: {e}")

    save_json_cache(NEWS_CACHE, cache)
    elapsed = time.time() - t0
    log.info(f"News done: {len(top_tickers)} tickers, +{total_new} articles, {elapsed:.0f}s")
    return total_new


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FMP Earnings Calendar (next 30 days)
# ═══════════════════════════════════════════════════════════════════════════════


def update_earnings_calendar(api_key: str) -> int:
    """Update FMP earnings calendar for next 30 days. Returns entry count."""
    log.info(f"{'='*60}")
    log.info("Updating FMP earnings calendar (next 30 days)...")
    log.info(f"{'='*60}")

    # Validate API key
    if not _validate_fmp_key(api_key):
        log.warning("  FMP API key invalid or expired — skipping earnings calendar")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

    # Fetch earnings calendar from FMP
    endpoint = f"/earnings-calendar?from={today}&to={end_date}&apikey={api_key}"
    data = fmp_api_get(endpoint, timeout=30)

    if data is None or not isinstance(data, list):
        # Try date-by-date approach if bulk fails
        log.info("  Bulk fetch failed, trying date-by-date...")
        all_entries = []
        for day_offset in range(31):
            d = (datetime.now() + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            ep = f"/earnings-calendar?from={d}&to={d}&apikey={api_key}"
            day_data = fmp_api_get(ep, timeout=15)
            if isinstance(day_data, list):
                all_entries.extend(day_data)
            time.sleep(0.2)  # Rate limit
        data = all_entries

    if not isinstance(data, list):
        log.warning("  No earnings calendar data returned")
        return 0

    # Filter to only SPX tickers
    spx_tickers = set(load_spx_tickers())
    filtered = [e for e in data if e.get("symbol", "") in spx_tickers]

    # Load existing and merge
    cache = load_json_cache(EARNINGS_CAL)
    existing_map = {}
    for e in cache.get("entries", []):
        key = (e.get("symbol", ""), e.get("date", ""))
        existing_map[key] = e

    new_count = 0
    for entry in filtered:
        key = (entry.get("symbol", ""), entry.get("date", ""))
        if key not in existing_map:
            new_count += 1
        existing_map[key] = entry

    output = {
        "last_updated": datetime.now().isoformat(),
        "from": today,
        "to": end_date,
        "entries": sorted(existing_map.values(), key=lambda e: (e.get("date", ""), e.get("symbol", ""))),
    }

    save_json_cache(EARNINGS_CAL, output)
    log.info(f"  Earnings calendar: {len(filtered)} SPX entries, +{new_count} new")
    return len(filtered)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FMP Grades/Ratings
# ═══════════════════════════════════════════════════════════════════════════════


def update_fmp_grades(tickers: List[str], api_key: str) -> int:
    """Update FMP analyst grades for all SPX tickers. Returns success count."""
    log.info(f"{'='*60}")
    log.info(f"Updating FMP grades/ratings ({len(tickers)} tickers)...")
    log.info(f"{'='*60}")

    # Validate API key
    if not _validate_fmp_key(api_key):
        log.warning("  FMP API key invalid or expired — skipping grades update")
        return 0

    cache = load_json_cache(GRADES_FILE)
    t0 = time.time()
    success = 0
    fail = 0

    def fetch_grade(ticker):
        """Fetch latest grade for a single ticker."""
        endpoint = f"/grades-historical?symbol={ticker}&limit=10&apikey={api_key}"
        data = fmp_api_get(endpoint, timeout=15)
        return ticker, data if isinstance(data, list) else []

    with ThreadPoolExecutor(max_workers=FMP_WORKERS) as executor:
        futures = {executor.submit(fetch_grade, t): t for t in tickers}
        done = 0
        for future in as_completed(futures):
            done += 1
            ticker = futures[future]
            try:
                _, grades = future.result()
                if grades:
                    existing = cache.get(ticker, [])
                    existing_dates = {g.get("gradeDate", g.get("date", "")) for g in existing}
                    new_grades = [g for g in grades if g.get("gradeDate", g.get("date", "")) not in existing_dates]
                    cache[ticker] = existing + new_grades
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                log.debug(f"  Grade {ticker} error: {e}")
                fail += 1

            if done % 50 == 0:
                log.info(f"    Grades: {done}/{len(tickers)} ({success} ok, {fail} fail)")

    save_json_cache(GRADES_FILE, cache)
    elapsed = time.time() - t0
    log.info(f"Grades done: {success}/{len(tickers)} tickers, {elapsed:.0f}s")
    return success


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Update all Falcon US stock data sources")
    parser.add_argument("--prices", action="store_true", help="Update OHLCV prices only")
    parser.add_argument("--fmp", action="store_true", help="Update FMP fundamentals only")
    parser.add_argument("--macro", action="store_true", help="Update VIX/SPX/Sectors only")
    parser.add_argument("--news", action="store_true", help="Update FMP news/earnings/grades only")
    parser.add_argument("--all", action="store_true", help="Update everything (default)")
    args = parser.parse_args()

    # Default to --all if nothing specified
    if not (args.prices or args.fmp or args.macro or args.news):
        args.all = True

    run_prices = args.all or args.prices
    run_macro = args.all or args.macro
    run_fmp = args.all or args.fmp
    run_news = args.all or args.news

    # Ensure directories exist
    FALCON_DIR.mkdir(parents=True, exist_ok=True)
    US_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    log.info("=" * 70)
    log.info("FALCON US DATA UPDATE — START")
    log.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Sections: {'prices ' if run_prices else ''}"
             f"{'macro ' if run_macro else ''}"
             f"{'fmp ' if run_fmp else ''}"
             f"{'news ' if run_news else ''}")
    log.info("=" * 70)

    # Load tickers
    tickers = load_spx_tickers()
    log.info(f"  SPX tickers: {len(tickers)}")

    # Load API key
    api_key = get_fmp_api_key()
    if not api_key and (run_fmp or run_news):
        log.warning("  FMP_API_KEY not available — FMP sections will be limited")

    results = {}

    # ── 1. OHLCV Prices ──
    if run_prices:
        try:
            s, n = update_prices(tickers)
            results["prices"] = f"{s}/{len(tickers)} tickers, +{n:,} rows"
        except Exception as e:
            log.error(f"Prices update failed: {e}")
            traceback.print_exc()
            results["prices"] = f"FAILED: {e}"

    # ── Macro: VIX, SPX, Sectors ──
    if run_macro:
        try:
            vix_rows = update_vix()
            results["vix"] = f"{vix_rows:,} rows"
        except Exception as e:
            log.error(f"VIX update failed: {e}")
            traceback.print_exc()
            results["vix"] = f"FAILED: {e}"

        try:
            spx_rows = update_spx()
            results["spx"] = f"{spx_rows:,} rows"
        except Exception as e:
            log.error(f"SPX update failed: {e}")
            traceback.print_exc()
            results["spx"] = f"FAILED: {e}"

        try:
            sector_rows = update_sector_etfs()
            results["sectors"] = f"{sector_rows:,} rows"
        except Exception as e:
            log.error(f"Sector ETF update failed: {e}")
            traceback.print_exc()
            results["sectors"] = f"FAILED: {e}"

    # ── 5. FMP Fundamentals ──
    if run_fmp and api_key:
        try:
            s, f = update_fmp_fundamentals(tickers, api_key)
            results["fmp_fundamentals"] = f"{s} success, {f} failed"
        except Exception as e:
            log.error(f"FMP fundamentals update failed: {e}")
            traceback.print_exc()
            results["fmp_fundamentals"] = f"FAILED: {e}"

    # ── 6-8. FMP News, Earnings, Grades ──
    if run_news and api_key:
        try:
            news_count = update_fmp_news(tickers, api_key)
            results["fmp_news"] = f"+{news_count} articles"
        except Exception as e:
            log.error(f"FMP news update failed: {e}")
            traceback.print_exc()
            results["fmp_news"] = f"FAILED: {e}"

        try:
            earn_count = update_earnings_calendar(api_key)
            results["earnings_calendar"] = f"{earn_count} entries"
        except Exception as e:
            log.error(f"Earnings calendar update failed: {e}")
            traceback.print_exc()
            results["earnings_calendar"] = f"FAILED: {e}"

        try:
            grade_count = update_fmp_grades(tickers, api_key)
            results["fmp_grades"] = f"{grade_count}/{len(tickers)} tickers"
        except Exception as e:
            log.error(f"FMP grades update failed: {e}")
            traceback.print_exc()
            results["fmp_grades"] = f"FAILED: {e}"

    # ── Summary ──
    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 70)
    log.info("FALCON US DATA UPDATE — SUMMARY")
    log.info("=" * 70)
    for section, summary in results.items():
        status = "✅" if "FAILED" not in summary else "❌"
        log.info(f"  {status} {section}: {summary}")
    log.info(f"  ⏱️  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info("=" * 70)

    # Verify output files
    log.info("")
    log.info("Output files:")
    for name, path in [
        ("OHLCV prices", PRICES_PARQUET),
        ("VIX", VIX_PARQUET),
        ("Sector ETFs", SECTOR_PARQUET),
        ("SPX index", SPX_PARQUET),
        ("FMP news", NEWS_CACHE),
        ("Earnings calendar", EARNINGS_CAL),
        ("FMP grades", GRADES_FILE),
    ] + [(f"FMP {k}", v) for k, v in FMP_FUNDAMENTALS.items()]:
        if path.exists():
            size = path.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f}MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f}KB"
            else:
                size_str = f"{size}B"
            log.info(f"  ✅ {name}: {path.name} ({size_str})")
        else:
            log.info(f"  ⚠️  {name}: {path.name} (not created)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
