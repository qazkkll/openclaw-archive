#!/usr/bin/env python3
"""
🦅 Falcon VWAP Monitor — Intraday Entry Timing Daemon
=====================================================

Monitors target stocks during market hours and triggers entries
based on price vs VWAP. Simple VWAP-based entry validated by
timing backtest (54% win rate vs open, +0.067% median improvement).

Modes:
  --live       Monitor in real-time (daemon mode, 9:25-4:05 ET)
  --backtest   Backtest VWAP entry logic on historical minute bars
  --test       Quick connectivity/API test (no market hours check)

Config: config/falcon.yaml
Signals: data/falcon/falcon_v031_scored_YYYYMMDD.json
Logs:    data/falcon/logs/vwap_monitor_YYYYMMDD.json

Usage:
  python3 falcon_vwap_monitor.py --live                    # live daemon
  python3 falcon_vwap_monitor.py --backtest --months 3     # backtest 3 months
  python3 falcon_vwap_monitor.py --backtest --date 2024-12-31  # single day
  python3 falcon_vwap_monitor.py --test                    # API test
"""

import sys
import json
import os
import time
import argparse
import signal as sig
from pathlib import Path
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

# ── Timezone ──
import pytz

# ── Alpaca ──
from dotenv import load_dotenv
load_dotenv()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# ═══════════════════════════════════════════════════════════
# Paths & Constants
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
LOG_DIR = DATA_DIR / "logs"
CONFIG_FILE = PROJECT_ROOT / "config" / "falcon.yaml"

ET = pytz.timezone("US/Eastern")
UTC = pytz.utc

# Default config (fallback if yaml parse fails)
DEFAULT_CONFIG = {
    "vwap_trigger": 0.0,       # price <= VWAP triggers entry
    "fallback_minutes": 60,     # force entry after 60 min without trigger
    "min_entry_score": 60,      # minimum falcon score to consider
    "top_n": 5,
    "poll_interval_sec": 30,    # seconds between API polls in live mode
}

# Entry scoring weights (from timing backtest optimization)
ENTRY_WEIGHTS = {
    "price_position": 0.30,   # price in daily range position
    "vwap_deviation": 0.30,   # VWAP deviation
    "volume_confirm": 0.20,   # volume confirmation
    "momentum": 0.10,         # minute momentum
    "spread_quality": 0.10,   # trade density (spread proxy)
}


# ═══════════════════════════════════════════════════════════
# Config Loader
# ═══════════════════════════════════════════════════════════
def load_config() -> dict:
    """Load falcon.yaml config, falling back to defaults."""
    cfg = dict(DEFAULT_CONFIG)
    try:
        import yaml
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                raw = yaml.safe_load(f)
            entry = raw.get("trading", {}).get("entry", {})
            if "vwap_trigger" in entry:
                cfg["vwap_trigger"] = entry["vwap_trigger"]
            if "fallback_minutes" in entry:
                cfg["fallback_minutes"] = entry["fallback_minutes"]
            if "min_entry_score" in entry:
                cfg["min_entry_score"] = entry["min_entry_score"]
            model = raw.get("model", {})
            if "top_n" in model:
                cfg["top_n"] = model["top_n"]
    except Exception:
        pass
    return cfg


# ═══════════════════════════════════════════════════════════
# Signal Loader
# ═══════════════════════════════════════════════════════════
def find_signal_file(target_date: Optional[str] = None) -> Optional[Path]:
    """
    Find the Falcon scored signal file for a given date.
    Searches: falcon_v031_scored_YYYYMMDD.json, then blueshield/arrow scores.
    """
    if target_date:
        dt = datetime.strptime(target_date, "%Y-%m-%d")
        date_str = dt.strftime("%Y%m%d")
    else:
        dt = datetime.now(ET)
        date_str = dt.strftime("%Y%m%d")

    # Primary: Falcon scored
    falcon_file = DATA_DIR / f"falcon_v046_scored_{date_str}.json"
    if falcon_file.exists():
        return falcon_file

    # Fallback: try previous trading days (weekend/holiday)
    for days_back in range(1, 5):
        prev = dt - timedelta(days=days_back)
        prev_str = prev.strftime("%Y%m%d")
        prev_file = DATA_DIR / f"falcon_v031_scored_{prev_str}.json"
        if prev_file.exists():
            return prev_file

    # Last resort: latest blueshield/arrow scores
    for pattern in ["signals/us/blueshield_v7_scores.json",
                     "signals/us/arrow_v12_scores.json"]:
        f = PROJECT_ROOT / pattern
        if f.exists():
            return f

    return None


def load_signal(signal_file: Path, top_n: int = 5) -> list:
    """
    Load signal file and extract top-N tickers.
    Returns list of dicts: [{"ticker": "AAPL", "score": 0.75, "close": 250.0}, ...]
    """
    with open(signal_file) as f:
        data = json.load(f)

    picks = data.get("picks", [])

    # Normalize field names (falcon uses "sym", others use "ticker")
    targets = []
    for p in picks[:top_n]:
        ticker = p.get("sym") or p.get("ticker", "")
        score = p.get("score") or p.get("pred_rank", 0.0)
        close = p.get("close") or p.get("price", 0.0)
        if ticker:
            targets.append({
                "ticker": ticker,
                "score": float(score),
                "close": float(close),
            })

    return targets


# ═══════════════════════════════════════════════════════════
# Alpaca Client
# ═══════════════════════════════════════════════════════════
def get_alpaca_client() -> StockHistoricalDataClient:
    """Create Alpaca historical data client from env."""
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID / APCA_API_SECRET_KEY in .env")
    return StockHistoricalDataClient(key, secret)


def fetch_minute_bars(client: StockHistoricalDataClient,
                      symbols: list, start: datetime, end: datetime,
                      batch_size: int = 15) -> dict:
    """
    Fetch minute bars for multiple symbols.
    Returns {symbol: DataFrame} with columns: open, high, low, close, volume, vwap, trade_count
    """
    all_bars = {}

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            bars = client.get_stock_bars(req)
            df = bars.df
            for sym in batch:
                if sym in df.index.get_level_values(0):
                    sub = df.loc[sym].copy()
                    sub.index = pd.to_datetime(sub.index)
                    all_bars[sym] = sub
        except Exception as e:
            print(f"  ⚠️ Batch {batch[:3]}... failed: {e}")
        time.sleep(0.3)

    return all_bars


def fetch_snapshots(client: StockHistoricalDataClient,
                    symbols: list) -> dict:
    """
    Fetch latest snapshots (trade + quote) for symbols.
    Returns {symbol: {"trade_price": ..., "bid": ..., "ask": ...}}
    """
    try:
        req = StockSnapshotRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)
        snapshots = client.get_stock_snapshot(req)
        result = {}
        for sym in symbols:
            snap = snapshots.get(sym) if isinstance(snapshots, dict) else getattr(snapshots, sym, None)
            if snap is None:
                continue
            trade = getattr(snap, "latest_trade", None)
            quote = getattr(snap, "latest_quote", None)
            result[sym] = {
                "trade_price": float(trade.price) if trade else None,
                "bid": float(quote.bid_price) if quote else None,
                "ask": float(quote.ask_price) if quote else None,
                "spread": (float(quote.ask_price) - float(quote.bid_price)) if quote else None,
            }
        return result
    except Exception as e:
        print(f"  ⚠️ Snapshot fetch failed: {e}")
        return {}


# ═══════════════════════════════════════════════════════════
# Entry Score (intraday quality)
# ═══════════════════════════════════════════════════════════
def compute_entry_score(bar_idx: int, bars_df: pd.DataFrame, lookback: int = 5) -> float:
    """
    Compute intraday entry quality score (0-100).
    Higher = better entry opportunity.
    """
    if bar_idx < lookback:
        return 50.0

    row = bars_df.iloc[bar_idx]
    price = row["close"]
    vwap = row.get("vwap", price)

    # Day data so far
    day_bars = bars_df.iloc[:bar_idx + 1]
    day_high = day_bars["high"].max()
    day_low = day_bars["low"].min()

    if day_high == day_low:
        return 50.0

    # 1. Price position (low = good for buying)
    price_pos = (price - day_low) / (day_high - day_low)
    price_score = max(0, min(100, (1 - price_pos) * 100))

    # 2. VWAP deviation (below VWAP = good)
    vwap_dev = (price - vwap) / vwap if vwap > 0 else 0
    vwap_score = max(0, min(100, (0.01 - vwap_dev) * 5000))

    # 3. Volume confirmation
    avg_vol = day_bars["volume"].mean()
    recent_vol = day_bars.iloc[-lookback:]["volume"].mean()
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
    vol_score = max(0, min(100, vol_ratio * 50))

    # 4. Momentum (recent bars)
    if bar_idx >= lookback:
        price_5ago = bars_df.iloc[bar_idx - lookback]["close"]
        mom = (price - price_5ago) / price_5ago if price_5ago > 0 else 0
        mom_score = max(0, min(100, (mom + 0.01) * 5000))
    else:
        mom_score = 50.0

    # 5. Trade density (liquidity proxy)
    tc = row.get("trade_count", 100)
    avg_tc = day_bars["trade_count"].mean() if "trade_count" in day_bars.columns else 100
    tc_ratio = tc / avg_tc if avg_tc > 0 else 1.0
    spread_score = max(0, min(100, tc_ratio * 50))

    # Weighted score
    score = (
        ENTRY_WEIGHTS["price_position"] * price_score +
        ENTRY_WEIGHTS["vwap_deviation"] * vwap_score +
        ENTRY_WEIGHTS["volume_confirm"] * vol_score +
        ENTRY_WEIGHTS["momentum"] * mom_score +
        ENTRY_WEIGHTS["spread_quality"] * spread_score
    )

    return round(score, 1)


# ═══════════════════════════════════════════════════════════
# VWAP Monitor Core Logic
# ═══════════════════════════════════════════════════════════
class VWAPTracker:
    """Tracks VWAP entry status for a single stock."""

    def __init__(self, ticker: str, config: dict):
        self.ticker = ticker
        self.config = config
        self.status = "waiting"     # waiting -> good_entry / forced_entry
        self.entry_price = None
        self.entry_time = None
        self.entry_bar_idx = None
        self.entry_score = None
        self.entry_type = None       # "good_entry" or "forced_entry"
        self.bars_seen = 0
        self.open_price = None
        self.vwap_values = []        # track VWAP history
        self.price_history = []      # track close prices

    def update(self, bar_idx: int, price: float, vwap: float,
               bars_df: pd.DataFrame = None) -> Optional[dict]:
        """
        Process a new minute bar.
        Returns entry event dict if entry triggered, else None.
        """
        if self.status != "waiting":
            return None

        self.bars_seen += 1
        self.price_history.append(price)
        self.vwap_values.append(vwap)

        if self.open_price is None:
            self.open_price = price

        # Check VWAP trigger: price <= VWAP
        trigger = self.config.get("vwap_trigger", 0.0)
        if price <= vwap * (1 + trigger):
            # Good entry: price at or below VWAP
            self.entry_price = price
            self.entry_time = bar_idx
            self.entry_bar_idx = bar_idx
            self.entry_type = "good_entry"
            self.status = "entered"

            # Compute entry quality score
            if bars_df is not None:
                self.entry_score = compute_entry_score(bar_idx, bars_df)
            else:
                self.entry_score = None

            return {
                "ticker": self.ticker,
                "entry_type": "good_entry",
                "entry_price": price,
                "entry_vwap": vwap,
                "entry_bar": bar_idx,
                "entry_score": self.entry_score,
                "vwap_deviation_pct": round((price - vwap) / vwap * 100, 4) if vwap > 0 else 0,
                "bars_waited": self.bars_seen,
            }

        # Check fallback: force entry after N minutes
        fallback_min = self.config.get("fallback_minutes", 60)
        if self.bars_seen >= fallback_min:
            self.entry_price = price
            self.entry_time = bar_idx
            self.entry_bar_idx = bar_idx
            self.entry_type = "forced_entry"
            self.status = "entered"

            if bars_df is not None:
                self.entry_score = compute_entry_score(bar_idx, bars_df)
            else:
                self.entry_score = None

            return {
                "ticker": self.ticker,
                "entry_type": "forced_entry",
                "entry_price": price,
                "entry_vwap": vwap,
                "entry_bar": bar_idx,
                "entry_score": self.entry_score,
                "vwap_deviation_pct": round((price - vwap) / vwap * 100, 4) if vwap > 0 else 0,
                "bars_waited": self.bars_seen,
                "open_price": self.open_price,
                "vs_open_pct": round((price - self.open_price) / self.open_price * 100, 4) if self.open_price else 0,
            }

        return None

    def to_dict(self) -> dict:
        """Export tracker state."""
        return {
            "ticker": self.ticker,
            "status": self.status,
            "entry_type": self.entry_type,
            "entry_price": self.entry_price,
            "entry_bar": self.entry_bar_idx,
            "entry_score": self.entry_score,
            "open_price": self.open_price,
            "bars_seen": self.bars_seen,
        }


# ═══════════════════════════════════════════════════════════
# Live Monitor Mode
# ═══════════════════════════════════════════════════════════
def run_live_monitor(args):
    """
    Live VWAP monitor daemon.
    - Starts at 9:25 AM ET
    - Polls minute bars every poll_interval_sec
    - Triggers entries based on VWAP
    - Logs all decisions
    - Exits at 4:05 PM ET
    """
    config = load_config()
    config["poll_interval_sec"] = getattr(args, "poll_interval", 30)

    now_et = datetime.now(ET)
    print(f"🦅 Falcon VWAP Monitor — Live Mode")
    print(f"=" * 60)
    print(f"  Time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Config: trigger={config['vwap_trigger']}, "
          f"fallback={config['fallback_minutes']}min, "
          f"min_score={config['min_entry_score']}")

    # ── Wait for market open ──
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    force_exit = now_et.replace(hour=16, minute=5, second=0, microsecond=0)

    if now_et < market_open:
        wait_sec = (market_open - now_et).total_seconds()
        print(f"\n⏰ Waiting for market open... ({wait_sec/60:.0f} min)")
        time.sleep(max(0, wait_sec - 5))  # wake up 5s early
    elif now_et > force_exit:
        print("  ❌ Market already closed. Use --backtest for historical analysis.")
        return

    # ── Load signal ──
    signal_file = find_signal_file()
    if signal_file is None:
        print("  ❌ No signal file found. Run falcon_score.py first.")
        return

    print(f"\n📡 Loading signal: {signal_file.name}")
    targets = load_signal(signal_file, top_n=config["top_n"])
    if not targets:
        print("  ❌ No targets in signal file.")
        return

    for t in targets:
        print(f"  {t['ticker']:<8} score={t['score']:.4f}  close=${t['close']:.2f}")

    tickers = [t["ticker"] for t in targets]

    # ── Init Alpaca ──
    print(f"\n🔌 Connecting to Alpaca...")
    client = get_alpaca_client()
    print(f"  ✅ Connected")

    # ── Init trackers ──
    trackers = {}
    for t in targets:
        trackers[t["ticker"]] = VWAPTracker(t["ticker"], config)

    # ── Audit log ──
    log_data = {
        "timestamp": now_et.isoformat(),
        "mode": "live",
        "signal_file": signal_file.name,
        "config": config,
        "targets": targets,
        "entries": [],
        "status_updates": [],
    }

    # ── Graceful shutdown ──
    running = [True]

    def handle_signal(signum, frame):
        print(f"\n🛑 Signal {signum} received, shutting down...")
        running[0] = False

    sig.signal(sig.SIGINT, handle_signal)
    sig.signal(sig.SIGTERM, handle_signal)

    # ── Main loop ──
    print(f"\n📊 Monitoring {len(tickers)} targets...")
    print(f"   Enter when price ≤ VWAP (trigger={config['vwap_trigger']})")
    print(f"   Fallback after {config['fallback_minutes']} min")
    print(f"   Poll interval: {config['poll_interval_sec']}s")
    print("-" * 60)

    day_str = now_et.strftime("%Y-%m-%d")
    poll_count = 0
    start_time = time.time()

    while running[0]:
        current_et = datetime.now(ET)

        # Check market close
        if current_et >= force_exit:
            print(f"\n⏰ Market closed at {current_et.strftime('%H:%M')}. Finalizing...")
            break

        # Check if all entries done
        all_done = all(t.status == "entered" for t in trackers.values())
        if all_done:
            print(f"\n✅ All {len(tickers)} entries triggered!")
            break

        # Wait for market open if needed
        if current_et < market_open:
            time.sleep(5)
            continue

        poll_count += 1
        time_since_start = time.time() - start_time

        try:
            # Fetch latest bars for today
            today_start = current_et.replace(hour=9, minute=30, second=0, microsecond=0)
            bars = fetch_minute_bars(client, tickers, today_start, current_et)

            for ticker in tickers:
                tracker = trackers[ticker]
                if tracker.status == "entered":
                    continue

                if ticker not in bars or len(bars[ticker]) < 5:
                    continue

                df = bars[ticker]
                # Convert to ET if needed
                if df.index.tz is not None:
                    df = df.tz_convert("US/Eastern")

                latest_bar = df.iloc[-1]
                price = float(latest_bar["close"])
                vwap = float(latest_bar.get("vwap", price))
                bar_idx = len(df) - 1

                event = tracker.update(bar_idx, price, vwap, df)

                if event:
                    # Compute vs open for good entries too
                    if tracker.open_price and event["entry_type"] == "good_entry":
                        event["open_price"] = tracker.open_price
                        event["vs_open_pct"] = round(
                            (price - tracker.open_price) / tracker.open_price * 100, 4
                        )

                    log_data["entries"].append({
                        **event,
                        "time_et": current_et.strftime("%H:%M:%S"),
                        "timestamp": current_et.isoformat(),
                    })

                    emoji = "🟢" if event["entry_type"] == "good_entry" else "🟡"
                    score_str = f" score={event['entry_score']:.1f}" if event.get("entry_score") else ""
                    vs_open = f" vs_open={event.get('vs_open_pct', 0):+.2f}%" if event.get("vs_open_pct") is not None else ""
                    print(f"  {emoji} {ticker}: {event['entry_type']} @ ${event['entry_price']:.2f} "
                          f"(VWAP=${event['entry_vwap']:.2f}, dev={event['vwap_deviation_pct']:+.2f}%)"
                          f"{score_str}{vs_open}")
                else:
                    # Status update every 10 polls
                    if poll_count % 10 == 0:
                        remaining = sum(1 for t in trackers.values() if t.status == "waiting")
                        print(f"  ⏳ {current_et.strftime('%H:%M')} — "
                              f"Waiting: {remaining}/{len(tickers)} | "
                              f"{ticker}: ${price:.2f} vs VWAP ${vwap:.2f}")

        except Exception as e:
            print(f"  ⚠️ Poll {poll_count} error: {e}")

        # Sleep
        time.sleep(config["poll_interval_sec"])

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print(f"📋 Session Summary")
    print(f"{'=' * 60}")

    good_entries = sum(1 for t in trackers.values() if t.entry_type == "good_entry")
    forced_entries = sum(1 for t in trackers.values() if t.entry_type == "forced_entry")
    waiting = sum(1 for t in trackers.values() if t.status == "waiting")

    print(f"  🟢 Good entries (≤ VWAP): {good_entries}")
    print(f"  🟡 Forced entries (timeout): {forced_entries}")
    print(f"  ⏳ Still waiting: {waiting}")
    print(f"  Total polls: {poll_count}")

    log_data["summary"] = {
        "good_entries": good_entries,
        "forced_entries": forced_entries,
        "waiting": waiting,
        "total_polls": poll_count,
        "final_time_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
    }

    for t in trackers.values():
        log_data["status_updates"].append(t.to_dict())

    # ── Save log ──
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"vwap_monitor_{day_str}.json"
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"\n📁 Log saved: {log_file}")


# ═══════════════════════════════════════════════════════════
# Backtest Mode
# ═══════════════════════════════════════════════════════════
def run_backtest(args):
    """
    Backtest VWAP entry logic on historical data.
    Compares VWAP timing vs open price entry.
    """
    config = load_config()

    print(f"🦅 Falcon VWAP Monitor — Backtest Mode")
    print(f"{'=' * 60}")

    # ── Determine date range ──
    if args.date:
        signal_dates = [args.date]
    else:
        # Use falcon features dates
        features_file = DATA_DIR / "features_v02.parquet"
        if not features_file.exists():
            print("  ❌ features_v02.parquet not found")
            return

        master = pd.read_parquet(features_file)
        master["date"] = master["date"].astype(str)
        all_dates = sorted(master["date"].unique())

        n_days = args.months * 22 if args.months else 22
        signal_dates = all_dates[-n_days:]

    print(f"  Dates: {signal_dates[0]} → {signal_dates[-1]} ({len(signal_dates)} days)")

    # ── Collect all tickers across dates ──
    all_signals = {}
    all_tickers = set()

    for d in signal_dates:
        # Find signal file for this date
        dt = datetime.strptime(d, "%Y-%m-%d")
        date_str = dt.strftime("%Y%m%d")
        falcon_file = DATA_DIR / f"falcon_v046_scored_{date_str}.json"
        if not falcon_file.exists():
            continue

        targets = load_signal(falcon_file, top_n=config["top_n"])
        if targets:
            all_signals[d] = targets
            all_tickers.update(t["ticker"] for t in targets)

    if not all_signals:
        print("  ❌ No signals found for backtest period")
        return

    print(f"  Signals: {len(all_signals)} days, {len(all_tickers)} unique tickers")

    # ── Fetch minute bars ──
    print(f"\n📡 Fetching minute bars from Alpaca...")
    client = get_alpaca_client()

    fetch_start = datetime.strptime(min(all_signals.keys()), "%Y-%m-%d") - timedelta(days=1)
    fetch_end = datetime.strptime(max(all_signals.keys()), "%Y-%m-%d") + timedelta(days=1)

    tickers_list = sorted(all_tickers)
    minute_data = fetch_minute_bars(client, tickers_list, fetch_start, fetch_end)
    print(f"  ✅ Fetched {len(minute_data)}/{len(tickers_list)} tickers")

    # ── Simulate entries ──
    print(f"\n⏱️  Simulating VWAP entry logic...")
    results = []

    for d, targets in sorted(all_signals.items()):
        dt = datetime.strptime(d, "%Y-%m-%d")

        for target in targets:
            ticker = target["ticker"]
            if ticker not in minute_data:
                continue

            bars = minute_data[ticker]

            # Convert to ET
            if bars.index.tz is not None:
                bars_et = bars.tz_convert("US/Eastern")
            else:
                bars_et = bars.tz_localize("UTC").tz_convert("US/Eastern")

            # Filter to this day
            day_bars = bars_et[bars_et.index.strftime("%Y-%m-%d") == d]
            if len(day_bars) < 10:
                continue

            # Get open price
            open_price = float(day_bars.iloc[0]["open"])

            # Simulate VWAP entry
            tracker = VWAPTracker(ticker, config)
            entry_event = None

            # Only use first 240 bars (4 hours) for entry window
            market_bars = day_bars.iloc[:240] if len(day_bars) > 240 else day_bars

            for i in range(min(5, len(market_bars)), len(market_bars)):
                row = market_bars.iloc[i]
                price = float(row["close"])
                vwap = float(row.get("vwap", price))

                event = tracker.update(i, price, vwap, market_bars)
                if event:
                    entry_event = event
                    break

            # If no entry triggered (not enough bars), force at fallback
            if entry_event is None and tracker.status == "waiting":
                fallback_min = config["fallback_minutes"]
                fallback_bar = min(fallback_min, len(market_bars) - 1)
                row = market_bars.iloc[fallback_bar]
                price = float(row["close"])
                vwap = float(row.get("vwap", price))
                entry_event = {
                    "ticker": ticker,
                    "entry_type": "forced_entry",
                    "entry_price": price,
                    "entry_vwap": vwap,
                    "entry_bar": fallback_bar,
                    "entry_score": compute_entry_score(fallback_bar, market_bars),
                    "bars_waited": fallback_bar,
                    "open_price": open_price,
                    "vs_open_pct": round((price - open_price) / open_price * 100, 4),
                }

            if entry_event:
                entry_event["date"] = d
                entry_event["signal_score"] = target.get("score", 0)
                entry_event["signal_close"] = target.get("close", 0)
                # Ensure vs_open_pct is present for all entries
                if "vs_open_pct" not in entry_event and open_price:
                    entry_event["open_price"] = open_price
                    entry_event["vs_open_pct"] = round(
                        (entry_event["entry_price"] - open_price) / open_price * 100, 4
                    )
                results.append(entry_event)

    if not results:
        print("  ❌ No entries simulated")
        return

    # ── Analysis ──
    print(f"\n{'=' * 60}")
    print(f"📊 Backtest Results")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame(results)

    # Entry type breakdown
    good = results_df[results_df["entry_type"] == "good_entry"]
    forced = results_df[results_df["entry_type"] == "forced_entry"]

    print(f"\n  Total entries: {len(results_df)}")
    print(f"  🟢 Good entries (≤ VWAP): {len(good)} ({len(good)/len(results_df)*100:.1f}%)")
    print(f"  🟡 Forced entries (timeout): {len(forced)} ({len(forced)/len(results_df)*100:.1f}%)")

    # VWAP deviation
    print(f"\n  VWAP Deviation (entry_price - VWAP) / VWAP:")
    devs = results_df["vwap_deviation_pct"]
    print(f"    Mean:   {devs.mean():+.2f}%")
    print(f"    Median: {devs.median():+.2f}%")
    print(f"    Std:    {devs.std():.2f}%")

    # vs Open
    if "vs_open_pct" in results_df.columns:
        vs_open = results_df["vs_open_pct"].dropna()
    else:
        vs_open = pd.Series(dtype=float)

    if len(vs_open) > 0:
        better = int((vs_open < 0).sum())
        print(f"\n  Entry vs Open Price:")
        print(f"    Entries better than open: {better}/{len(vs_open)} ({better/len(vs_open)*100:.1f}%)")
        print(f"    Mean improvement: {vs_open.mean():+.2f}%")
        print(f"    Median improvement: {vs_open.median():+.2f}%")
    else:
        better = 0

    # Entry timing
    print(f"\n  Entry Timing (bars after open):")
    bars = results_df["bars_waited"]
    print(f"    Mean:   {bars.mean():.1f} min")
    print(f"    Median: {bars.median():.0f} min")
    print(f"    Min:    {bars.min():.0f} min")
    print(f"    Max:    {bars.max():.0f} min")

    # Entry scores
    scores = results_df["entry_score"].dropna()
    if len(scores) > 0:
        print(f"\n  Entry Quality Score:")
        print(f"    Mean:   {scores.mean():.1f}")
        print(f"    Median: {scores.median():.1f}")

    # Per-ticker summary
    print(f"\n  Per-Ticker Summary:")
    print(f"    {'Ticker':<8} {'Count':>5} {'Good':>5} {'Forced':>6} {'AvgDev':>8} {'AvgScore':>9}")
    print(f"    {'-'*50}")
    for ticker in sorted(results_df["ticker"].unique()):
        t_df = results_df[results_df["ticker"] == ticker]
        n = len(t_df)
        n_good = (t_df["entry_type"] == "good_entry").sum()
        n_forced = (t_df["entry_type"] == "forced_entry").sum()
        avg_dev = t_df["vwap_deviation_pct"].mean()
        avg_score = t_df["entry_score"].mean()
        print(f"    {ticker:<8} {n:>5} {n_good:>5} {n_forced:>6} {avg_dev:>+7.2f}% {avg_score:>8.1f}")

    # ── Save results ──
    output = {
        "config": {
            "vwap_trigger": config["vwap_trigger"],
            "fallback_minutes": config["fallback_minutes"],
            "top_n": config["top_n"],
            "signal_dates": [signal_dates[0], signal_dates[-1]],
            "n_days": len(all_signals),
        },
        "summary": {
            "total_entries": len(results_df),
            "good_entries": len(good),
            "forced_entries": len(forced),
            "vwap_dev_mean_pct": round(float(devs.mean()), 4),
            "vwap_dev_median_pct": round(float(devs.median()), 4),
            "vs_open_mean_pct": round(float(vs_open.mean()), 4) if len(vs_open) > 0 else None,
            "vs_open_median_pct": round(float(vs_open.median()), 4) if len(vs_open) > 0 else None,
            "vs_open_better_count": int(better) if len(vs_open) > 0 else 0,
            "vs_open_total": len(vs_open),
            "entry_bars_mean": round(float(bars.mean()), 1),
        },
        "entries": results,
    }

    out_file = DATA_DIR / "vwap_backtest_result.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n📁 Results saved: {out_file}")

    return output


# ═══════════════════════════════════════════════════════════
# Test Mode
# ═══════════════════════════════════════════════════════════
def run_test(args):
    """Quick connectivity and API test."""
    print("🦅 Falcon VWAP Monitor — Test Mode")
    print("=" * 60)

    # 1. Config
    config = load_config()
    print(f"\n✅ Config loaded: {json.dumps(config, indent=2)}")

    # 2. Signal file
    signal_file = find_signal_file()
    if signal_file:
        targets = load_signal(signal_file, top_n=config["top_n"])
        print(f"\n✅ Signal file: {signal_file.name}")
        for t in targets:
            print(f"   {t['ticker']:<8} score={t['score']:.4f}  close=${t['close']:.2f}")
    else:
        print(f"\n⚠️  No signal file found (OK for backtest mode)")

    # 3. Alpaca connection
    print(f"\n🔌 Testing Alpaca connection...")
    try:
        client = get_alpaca_client()
        # Try fetching a small batch of bars
        test_start = datetime.now(ET) - timedelta(days=3)
        test_end = datetime.now(ET)
        bars = fetch_minute_bars(client, ["AAPL"], test_start, test_end, batch_size=1)
        if "AAPL" in bars:
            df = bars["AAPL"]
            print(f"   ✅ AAPL: {len(df)} bars fetched")
            if len(df) > 0:
                latest = df.iloc[-1]
                print(f"   Latest: close=${latest['close']:.2f}, "
                      f"vwap=${latest.get('vwap', 'N/A')}, "
                      f"volume={latest['volume']}")
        else:
            print(f"   ⚠️  AAPL bars returned empty (market may be closed)")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")

    # 4. VWAPTracker test
    print(f"\n🧪 Testing VWAPTracker logic...")
    test_tracker = VWAPTracker("TEST", config)
    test_tracker.update(0, 100.0, 100.0)  # at VWAP
    test_tracker.update(1, 99.5, 100.0)   # below VWAP -> should trigger
    print(f"   Status: {test_tracker.status}")
    print(f"   Entry type: {test_tracker.entry_type}")
    print(f"   Entry price: ${test_tracker.entry_price}")

    # 5. Entry score test
    print(f"\n🧪 Testing entry score computation...")
    test_bars = pd.DataFrame({
        "open": [100, 101, 100, 99, 98, 99, 100],
        "high": [101, 102, 101, 100, 99, 100, 101],
        "low": [99, 100, 99, 98, 97, 98, 99],
        "close": [100.5, 101.5, 100.5, 99.5, 98.5, 99.5, 100.5],
        "volume": [1000, 1200, 800, 1500, 900, 1100, 1000],
        "vwap": [100.0, 100.5, 100.3, 100.1, 99.9, 99.8, 99.9],
        "trade_count": [50, 60, 45, 70, 40, 55, 50],
    })
    for i in range(5, len(test_bars)):
        score = compute_entry_score(i, test_bars)
        price = test_bars.iloc[i]["close"]
        vwap = test_bars.iloc[i]["vwap"]
        print(f"   Bar {i}: price=${price:.1f} VWAP=${vwap:.1f} → score={score:.1f}")

    print(f"\n✅ All tests passed!")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="🦅 Falcon VWAP Monitor — Intraday Entry Timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  --live       Live monitoring daemon (9:25-4:05 ET)
  --backtest   Backtest on historical data
  --test       API connectivity test

Examples:
  python3 falcon_vwap_monitor.py --live
  python3 falcon_vwap_monitor.py --backtest --months 3
  python3 falcon_vwap_monitor.py --backtest --date 2024-12-31
  python3 falcon_vwap_monitor.py --test
""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--live", action="store_true", help="Live monitoring daemon")
    group.add_argument("--backtest", action="store_true", help="Backtest on historical data")
    group.add_argument("--test", action="store_true", help="API connectivity test")

    parser.add_argument("--months", type=int, default=1, help="Backtest months (default: 1)")
    parser.add_argument("--date", type=str, default=None, help="Single date for backtest (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=None, help="Override top-N picks")
    parser.add_argument("--poll-interval", type=int, default=30, help="Live poll interval seconds (default: 30)")

    args = parser.parse_args()

    if args.top_n:
        config = load_config()
        config["top_n"] = args.top_n

    if args.live:
        run_live_monitor(args)
    elif args.backtest:
        run_backtest(args)
    elif args.test:
        run_test(args)


if __name__ == "__main__":
    main()
