#!/usr/bin/env python3
"""
Falcon Weekly Review — Performance Analysis & Report Generator

Reads:
  - Trade logs from data/falcon/trades/
  - Scored signals from data/falcon/falcon_v031_scored_*.json
  - Actual price data from Alpaca (via API)
  - Config from config/falcon.yaml

Outputs:
  - Markdown report to data/falcon/reviews/YYYY-MM-DD_weekly_review.md
  - JSON summary to data/falcon/reviews/YYYY-MM-DD_weekly_review.json

Usage:
    python3 scripts/falcon/weekly_review.py                    # review last 7 days
    python3 scripts/falcon/weekly_review.py --days 30          # review last 30 days
    python3 scripts/falcon/weekly_review.py --start 2024-12-01 --end 2024-12-31
"""

import json
import os
import sys
import glob
from datetime import datetime, timedelta
from pathlib import Path

# ── Setup ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "falcon"
REVIEWS_DIR = DATA_DIR / "reviews"
TRADES_DIR = DATA_DIR / "trades"
CONFIG_PATH = BASE_DIR / "config" / "falcon.yaml"
ENV_PATH = BASE_DIR / ".env"

REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
TRADES_DIR.mkdir(parents=True, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')


def load_env():
    """Load .env file for Alpaca credentials."""
    env = {}
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env


def load_config():
    """Load falcon.yaml config."""
    try:
        import yaml
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Config load error: {e}")
        return {}


def load_trades(start_date, end_date):
    """Load trade logs within date range."""
    trades = []
    if not TRADES_DIR.exists():
        return trades

    for f in sorted(TRADES_DIR.glob("*.json")):
        try:
            with open(f) as fh:
                t = json.load(fh)
            # Check date range
            t_date = t.get("date", t.get("timestamp", ""))
            if t_date and start_date <= t_date <= end_date:
                trades.append(t)
        except Exception:
            continue

    return trades


def load_scored_signals():
    """Load all scored signal files."""
    signals = []
    pattern = str(DATA_DIR / "falcon_v031_scored_*.json")
    for f in sorted(glob.glob(pattern)):
        try:
            with open(f) as fh:
                sig = json.load(fh)
            signals.append(sig)
        except Exception:
            continue
    return signals


def fetch_alpaca_bars(symbols, start_date, end_date, env):
    """Fetch daily bars from Alpaca for given symbols and date range."""
    try:
        import requests
    except ImportError:
        print("⚠️  requests not installed, skipping Alpaca price fetch")
        return {}

    api_key = env.get("APCA_API_KEY_ID", "")
    api_secret = env.get("APCA_API_SECRET_KEY", "")
    base_url = env.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        print("⚠️  Alpaca credentials not found in .env, skipping price fetch")
        return {}

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    bars = {}
    # Alpaca allows up to ~200 symbols per request via the bars endpoint
    batch_size = 50
    symbols_list = list(symbols)

    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i+batch_size]
        symbols_param = ",".join(batch)
        url = f"https://data.alpaca.markets/v2/stocks/bars"
        params = {
            "symbols": symbols_param,
            "timeframe": "1Day",
            "start": start_date,
            "end": end_date,
            "limit": 10000,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                for sym, sym_bars in data.get("bars", {}).items():
                    bars[sym] = sym_bars
            else:
                print(f"⚠️  Alpaca API error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"⚠️  Alpaca request failed: {e}")

    return bars


def compute_trade_pnl(trades, bars):
    """Compute P&L for each trade using Alpaca price data."""
    results = []
    for t in trades:
        ticker = t.get("ticker", t.get("sym", ""))
        entry_date = t.get("entry_date", t.get("date", ""))
        exit_date = t.get("exit_date", "")
        entry_price = t.get("entry_price", t.get("avg_entry", 0))
        shares = t.get("shares", t.get("qty", 0))

        if not ticker or not entry_price:
            continue

        # Try to get exit price from bars
        exit_price = t.get("exit_price", 0)
        if not exit_price and ticker in bars:
            # Find the closest bar to exit_date
            for bar in bars[ticker]:
                bar_date = bar.get("t", "")[:10]
                if bar_date >= (exit_date or entry_date):
                    exit_price = bar.get("c", 0)
                    break

        if entry_price and exit_price:
            pnl = (exit_price - entry_price) * shares
            ret_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl = t.get("pnl", 0)
            ret_pct = t.get("return_pct", t.get("net_ret", 0))

        results.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "return_pct": ret_pct,
            "exit_reason": t.get("exit_reason", "unknown"),
        })

    return results


def compute_metrics(trade_results):
    """Compute aggregate performance metrics."""
    if not trade_results:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_return_pct": 0,
            "avg_return_pct": 0,
            "median_return_pct": 0,
            "sharpe_ratio": 0,
            "max_drawdown_pct": 0,
            "total_pnl": 0,
            "avg_win_pct": 0,
            "avg_loss_pct": 0,
            "best_trade_pct": 0,
            "worst_trade_pct": 0,
            "profit_factor": 0,
        }

    returns = [t["return_pct"] for t in trade_results]
    pnls = [t["pnl"] for t in trade_results]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    import statistics

    total_trades = len(trade_results)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_return = statistics.mean(returns) if returns else 0
    median_return = statistics.median(returns) if returns else 0
    std_return = statistics.stdev(returns) if len(returns) > 1 else 0

    # Sharpe (annualized, assuming ~252 trading days per year)
    # For simplicity, use daily returns
    sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0

    # Max drawdown (from cumulative returns)
    cumulative = []
    cum = 1.0
    for r in returns:
        cum *= (1 + r / 100)
        cumulative.append(cum)

    peak = cumulative[0]
    max_dd = 0
    for c in cumulative:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100
        if dd > max_dd:
            max_dd = dd

    total_return_pct = (cum - 1) * 100 if cumulative else 0
    total_pnl = sum(pnls)
    avg_win = statistics.mean(wins) if wins else 0
    avg_loss = statistics.mean(losses) if losses else 0
    best = max(returns) if returns else 0
    worst = min(returns) if returns else 0

    gross_wins = sum(r for r in returns if r > 0)
    gross_losses = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "total_return_pct": round(total_return_pct, 2),
        "avg_return_pct": round(avg_return, 2),
        "median_return_pct": round(median_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "best_trade_pct": round(best, 2),
        "worst_trade_pct": round(worst, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else 999,
    }


def generate_markdown_report(review_data):
    """Generate markdown report from review data."""
    metrics = review_data["metrics"]
    trades = review_data["trade_results"]
    signals = review_data["signals_summary"]
    config = review_data["config"]
    backtest = config.get("backtest_comparison", {})

    lines = []
    lines.append(f"# 🦅 Falcon Weekly Review")
    lines.append(f"**Period:** {review_data['start_date']} → {review_data['end_date']}")
    lines.append(f"**Generated:** {review_data['generated_at']}")
    lines.append("")

    # ── Performance Summary ──
    lines.append("## 📊 Performance Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Trades | {metrics['total_trades']} |")
    lines.append(f"| Win Rate | {metrics['win_rate']}% |")
    lines.append(f"| Total Return | {metrics['total_return_pct']}% |")
    lines.append(f"| Avg Return/Trade | {metrics['avg_return_pct']}% |")
    lines.append(f"| Sharpe Ratio | {metrics['sharpe_ratio']} |")
    lines.append(f"| Max Drawdown | {metrics['max_drawdown_pct']}% |")
    lines.append(f"| Profit Factor | {metrics['profit_factor']} |")
    lines.append(f"| Total P&L | ${metrics['total_pnl']:,.2f} |")
    lines.append(f"| Best Trade | {metrics['best_trade_pct']}% |")
    lines.append(f"| Worst Trade | {metrics['worst_trade_pct']}% |")
    lines.append("")

    # ── Backtest Comparison ──
    if backtest:
        lines.append("## 🔄 Backtest Comparison")
        lines.append("")
        bt_metrics = backtest.get("metrics", {})
        lines.append(f"| Metric | Live | Backtest (OOS) | Delta |")
        lines.append(f"|--------|------|----------------|-------|")

        for key, label in [("sharpe_ratio", "Sharpe"), ("win_rate", "Win Rate %"), ("max_drawdown_pct", "Max DD %"), ("avg_return_pct", "Avg Return %")]:
            live_val = metrics.get(key, "--")
            bt_val = bt_metrics.get(key, "--")
            if isinstance(live_val, (int, float)) and isinstance(bt_val, (int, float)):
                delta = live_val - bt_val
                delta_str = f"{delta:+.2f}"
            else:
                delta_str = "--"
            lines.append(f"| {label} | {live_val} | {bt_val} | {delta_str} |")

        lines.append("")

    # ── Signals Summary ──
    if signals:
        lines.append("## 🎯 Signals Generated")
        lines.append("")
        for sig in signals:
            date = sig.get("date", "--")
            picks = sig.get("picks", [])
            lines.append(f"### {date}")
            lines.append("")
            lines.append(f"| # | Ticker | Score | Fund Ratio | Analyst | Fund Metric |")
            lines.append(f"|---|--------|-------|------------|---------|-------------|")
            for i, p in enumerate(picks, 1):
                lines.append(f"| {i} | **{p.get('sym', '--')}** | {p.get('score', 0)*100:.1f}% | {p.get('fund_ratio', 0)*100:.1f}% | {p.get('analyst', 0)*100:.1f}% | {p.get('fund_metric', 0)*100:.1f}% |")
            lines.append("")

    # ── Per-Trade P&L ──
    if trades:
        lines.append("## 💰 Per-Trade P&L")
        lines.append("")
        lines.append(f"| Ticker | Entry | Exit | Return | P&L | Exit Reason |")
        lines.append(f"|--------|-------|------|--------|-----|-------------|")
        for t in sorted(trades, key=lambda x: x.get("return_pct", 0), reverse=True):
            ret = t.get("return_pct", 0)
            color = "🟢" if ret > 0 else "🔴" if ret < 0 else "⚪"
            lines.append(f"| {t.get('ticker', '--')} | {t.get('entry_date', '--')} | {t.get('exit_date', '--')} | {color} {ret:+.2f}% | ${t.get('pnl', 0):+,.2f} | {t.get('exit_reason', '--')} |")
        lines.append("")

    # ── Recommendations ──
    lines.append("## 📋 Recommendations")
    lines.append("")
    if metrics["win_rate"] < 50:
        lines.append("- ⚠️ Win rate below 50% — review scoring weights and entry criteria")
    if metrics["sharpe_ratio"] < 0:
        lines.append("- 🔴 Negative Sharpe — strategy is not generating risk-adjusted returns")
    if metrics["max_drawdown_pct"] > 20:
        lines.append("- 🔴 Max drawdown exceeds 20% — tighten stop-loss or reduce position sizing")
    if metrics["profit_factor"] < 1:
        lines.append("- 🔴 Profit factor below 1.0 — losses exceed gains")
    if metrics["win_rate"] >= 55 and metrics["sharpe_ratio"] > 0.5:
        lines.append("- 🟢 Performance in line with expectations — continue monitoring")
    if not trades:
        lines.append("- ℹ️ No trades executed this period — check entry conditions and market hours")
    lines.append("")
    lines.append("---")
    lines.append(f"*Auto-generated by Falcon Weekly Review Script*")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Falcon Weekly Review")
    parser.add_argument("--days", type=int, default=7, help="Review period in days (default: 7)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    now = datetime.now()
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"🦅 Falcon Weekly Review")
    print(f"   Period: {start_date} → {end_date}")
    print()

    # Load config
    config = load_config()
    env = load_env()

    # Load trades
    trades = load_trades(start_date, end_date)
    print(f"   📊 Found {len(trades)} trades in period")

    # Load scored signals
    all_signals = load_scored_signals()
    # Filter to period
    signals = [s for s in all_signals if start_date <= s.get("date", "") <= end_date]
    print(f"   🎯 Found {len(signals)} scored signals in period")

    # Get symbols from signals
    symbols = set()
    for sig in signals:
        for pick in sig.get("picks", []):
            symbols.add(pick.get("sym", ""))
    for t in trades:
        symbols.add(t.get("ticker", t.get("sym", "")))

    # Fetch Alpaca prices
    bars = {}
    if symbols:
        print(f"   📈 Fetching Alpaca bars for {len(symbols)} symbols...")
        bars = fetch_alpaca_bars(symbols, start_date, end_date, env)
        print(f"   ✅ Got bars for {len(bars)} symbols")

    # Compute trade P&L
    trade_results = compute_trade_pnl(trades, bars)

    # Compute metrics
    metrics = compute_metrics(trade_results)

    # Load backtest comparison
    bt_path = DATA_DIR / "oos_validation.json"
    backtest_comparison = {}
    if bt_path.exists():
        try:
            with open(bt_path) as f:
                oos = json.load(f)
            backtest_comparison = {
                "metrics": oos.get("results", {}).get("OOS", {}),
            }
        except Exception:
            pass

    # Build review data
    review_data = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "start_date": start_date,
        "end_date": end_date,
        "config": {
            "hold_days": config.get("trading", {}).get("hold_days"),
            "stop_loss": config.get("trading", {}).get("stop_loss"),
            "model": config.get("model", {}).get("name"),
            "weights": config.get("model", {}).get("weights", {}),
            "backtest_comparison": backtest_comparison,
        },
        "metrics": metrics,
        "trade_results": trade_results,
        "signals_summary": signals,
    }

    # Save JSON
    json_path = REVIEWS_DIR / f"{end_date}_weekly_review.json"
    with open(json_path, "w") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)
    print(f"   💾 JSON saved: {json_path}")

    # Generate and save markdown
    md_report = generate_markdown_report(review_data)
    md_path = REVIEWS_DIR / f"{end_date}_weekly_review.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"   📝 Markdown saved: {md_path}")

    # Print summary
    print()
    print("═" * 50)
    print(f"   Total Trades:  {metrics['total_trades']}")
    print(f"   Win Rate:      {metrics['win_rate']}%")
    print(f"   Total Return:  {metrics['total_return_pct']}%")
    print(f"   Sharpe:        {metrics['sharpe_ratio']}")
    print(f"   Max Drawdown:  {metrics['max_drawdown_pct']}%")
    print(f"   Profit Factor: {metrics['profit_factor']}")
    print("═" * 50)
    print()

    return review_data


if __name__ == "__main__":
    main()
