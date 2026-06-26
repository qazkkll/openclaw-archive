#!/usr/bin/env python3
"""
Comprehensive Backtest: V12 Green-Arrow LambdaMART (17 features)
================================================================
Tests multiple holding periods (5/10/15/20) and exit strategies
(fixed, trailing stop, RSI/BB early exit, stop-loss).

Out-of-sample: 2025-01-01 to present (needs ~200 days lookback from 2024-06-01)
"""

import sys, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(ROOT / "scripts" / "us"))

with open(ROOT / "config" / "central_config.json") as f:
    CFG = json.load(f)
FEATURES = CFG["features"]["lambdamart_v10_v12"]

# ─── Feature Computation ─────────────────────────────────────────────

def compute_features(df):
    """17 features, consistent with training"""
    df = df.sort_values(["sym", "date"])
    g = df.groupby("sym")["close"]

    df["ret5"] = g.transform(lambda x: x.pct_change(5)).astype(np.float32)
    df["ret20"] = g.transform(lambda x: x.pct_change(20)).astype(np.float32)
    df["ret60"] = g.transform(lambda x: x.pct_change(60)).astype(np.float32)
    df["momentum_6m"] = g.transform(lambda x: x.pct_change(126)).astype(np.float32)
    df["momentum_1m"] = g.transform(lambda x: x.pct_change(21)).astype(np.float32)

    ma20 = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma_bias20"] = ((df["close"] - ma20) / ma20.replace(0, np.nan)).astype(np.float32)

    df["vol20"] = g.transform(lambda x: x.rolling(20, min_periods=10).std()).astype(np.float32)
    vol60 = g.transform(lambda x: x.rolling(60, min_periods=20).std())
    df["vol_ratio"] = (df["vol20"] / vol60.replace(0, np.nan)).astype(np.float32)

    delta = g.transform(lambda x: x.diff()).astype(np.float32)
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    avg_loss = loss.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    df["rsi14"] = (100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))).astype(np.float32)

    ema12 = g.transform(lambda x: x.ewm(span=12, min_periods=6).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, min_periods=13).mean())
    macd = (ema12 - ema26).astype(np.float32)
    df["macd_hist"] = (macd - macd.groupby(df["sym"]).transform(
        lambda x: x.ewm(span=9, min_periods=5).mean())).astype(np.float32)

    bb_mid = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std = g.transform(lambda x: x.rolling(20, min_periods=10).std())
    df["bb_pos"] = ((df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)).astype(np.float32)

    high60 = g.transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = g.transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)

    fund = pd.read_parquet(ROOT / "data/us/fundamentals_latest.parquet", columns=["sym", "beta"])
    df = df.merge(fund, on="sym", how="left")
    df["beta_c"] = df["beta"].clip(-2, 5).fillna(0.73).astype(np.float32)

    vix = pd.read_parquet(ROOT / "data/us/vix_10y.parquet")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.reset_index()
    vix_date = [c for c in vix.columns if "date" in str(c).lower()][0]
    vix_val = [c for c in vix.columns if "close" in str(c).lower()][0]
    vix_df = pd.DataFrame({"date": pd.to_datetime(vix[vix_date]),
                           "vix_close": vix[vix_val].astype(np.float32)})
    df = df.merge(vix_df, on="date", how="left")

    spy = df[df["sym"] == "SPY"][["date", "close"]].sort_values("date")
    for d in [5, 20, 60]:
        spy[f"spy_ret{d}"] = spy["close"].pct_change(d).astype(np.float32)
    df = df.merge(spy[["date", "spy_ret5", "spy_ret20", "spy_ret60"]], on="date", how="left")

    df[FEATURES] = df[FEATURES].fillna(0)
    return df


# ─── Advanced Backtest Engine ────────────────────────────────────────

def run_backtest(df, scores_df, hold_days, top_n, exit_strategy, stop_loss_pct=-0.15):
    """
    Run backtest for a single configuration.
    
    exit_strategy: 'fixed' | 'trailing_stop' | 'rsi_bb_exit' | 'stop_only'
    """
    
    # Merge scores into price data
    # scores_df has: date, sym, pred (model score)
    
    # Build universe: pivot to get prices per (date, sym)
    price = df.pivot_table(index="date", columns="sym", values="close")
    
    # Also need high, low, rsi, bb_pos for early exit strategies
    high = df.pivot_table(index="date", columns="sym", values="high")
    low = df.pivot_table(index="date", columns="sym", values="low")
    rsi = df.pivot_table(index="date", columns="sym", values="rsi14")
    bb_pos = df.pivot_table(index="date", columns="sym", values="bb_pos")
    
    # Pivot scores to (date, sym)
    score_pivot = scores_df.pivot_table(index="date", columns="sym", values="pred")
    
    # Only evaluate from 2025-01-01
    eval_start = pd.Timestamp("2025-01-01")
    eval_dates = price.index[price.index >= eval_start]
    
    if len(eval_dates) == 0:
        return None
    
    # Track trades
    trades = []
    equity_curve = []
    cash = 100000.0
    initial_cash = cash
    
    # Position tracking: {sym: {entry_date, entry_price, entry_high, shares}}
    positions = {}
    
    available_dates = sorted(price.index)
    date_to_idx = {d: i for i, d in enumerate(available_dates)}
    
    for date in eval_dates:
        if date not in score_pivot.index:
            continue
        
        # Daily price changes for existing positions
        daily_port_return = 0.0
        
        # Check exits for existing positions
        exited_syms = []
        for sym, pos in positions.items():
            if sym not in price.columns or date not in price.index:
                continue
            
            current_price = price.loc[date, sym]
            if pd.isna(current_price):
                continue
            
            entry_price = pos["entry_price"]
            entry_high = pos["entry_high"]
            days_held = (date - pos["entry_date"]).days
            
            # Update trailing high
            if not pd.isna(high.loc[date, sym] if sym in high.columns and date in high.index else np.nan):
                day_high = high.loc[date, sym]
                if not pd.isna(day_high):
                    entry_high = max(entry_high, day_high)
                    pos["entry_high"] = entry_high
            
            daily_return = current_price / pos["last_price"] - 1
            daily_port_return += pos["weight"] * daily_return
            
            should_exit = False
            exit_reason = ""
            
            if exit_strategy == "fixed":
                if days_held >= hold_days:
                    should_exit = True
                    exit_reason = "hold_period"
            
            elif exit_strategy == "stop_only":
                pnl = current_price / entry_price - 1
                if pnl <= stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif days_held >= hold_days:
                    should_exit = True
                    exit_reason = "hold_period"
            
            elif exit_strategy == "trailing_stop":
                pnl = current_price / entry_price - 1
                trail_pnl = current_price / entry_high - 1
                # Exit if trailing from peak drops beyond threshold
                if entry_high > entry_price and trail_pnl <= stop_loss_pct * 0.5:
                    should_exit = True
                    exit_reason = "trailing_stop"
                elif pnl <= stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif days_held >= hold_days:
                    should_exit = True
                    exit_reason = "hold_period"
            
            elif exit_strategy == "rsi_bb_exit":
                pnl = current_price / entry_price - 1
                # Early exit if overbought
                day_rsi = rsi.loc[date, sym] if (sym in rsi.columns and date in rsi.index) else 50
                day_bb = bb_pos.loc[date, sym] if (sym in bb_pos.columns and date in bb_pos.index) else 0.5
                if pnl > 0 and not pd.isna(day_rsi) and not pd.isna(day_bb):
                    if day_rsi > 75 and day_bb > 0.9:
                        should_exit = True
                        exit_reason = "rsi_bb_overbought"
                # Stop loss
                if pnl <= stop_loss_pct:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif days_held >= hold_days:
                    should_exit = True
                    exit_reason = "hold_period"
            
            pos["last_price"] = current_price
            
            if should_exit:
                trade_return = current_price / entry_price - 1
                trades.append({
                    "sym": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "return": trade_return,
                    "days_held": days_held,
                    "exit_reason": exit_reason,
                })
                exited_syms.append(sym)
                cash += pos["shares"] * current_price
        
        for sym in exited_syms:
            del positions[sym]
        
        # Record equity
        port_value = cash
        for sym, pos in positions.items():
            if sym in price.columns and date in price.index:
                cp = price.loc[date, sym]
                if not pd.isna(cp):
                    port_value += pos["shares"] * cp
        equity_curve.append({"date": date, "equity": port_value})
        
        # New entries if we have room
        if len(positions) < top_n and date in score_pivot.index:
            # Get scores for this date, excluding already-held
            day_scores = score_pivot.loc[date].dropna()
            held = set(positions.keys())
            candidates = day_scores[~day_scores.index.isin(held)]
            
            # Filter by green arrow signal (top 5% percentile)
            if len(candidates) > 0:
                rank_pct = candidates.rank(pct=True, ascending=True)
                green_mask = rank_pct >= 0.95
                candidates = candidates[green_mask]
            
            # Take top N
            n_slots = top_n - len(positions)
            picks = candidates.nlargest(n_slots)
            
            if len(picks) > 0:
                weight_per = 1.0 / top_n
                available = cash
                for sym in picks.index:
                    if sym in price.columns and date in price.index:
                        cp = price.loc[date, sym]
                        if not pd.isna(cp) and cp > 0:
                            alloc = min(available, initial_cash * weight_per)
                            shares = alloc / cp
                            cash -= alloc
                            positions[sym] = {
                                "entry_date": date,
                                "entry_price": cp,
                                "entry_high": cp,
                                "last_price": cp,
                                "shares": shares,
                                "weight": weight_per,
                            }
                            available -= alloc
    
    # Close remaining positions at last price
    if trades:
        last_date = trades[-1]["exit_date"] if trades else eval_dates[-1]
    
    # Final equity
    if equity_curve:
        final_equity = equity_curve[-1]["equity"]
    else:
        final_equity = initial_cash
    
    total_return = (final_equity / initial_cash - 1) * 100
    
    # Compute metrics
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    eq_series = eq_df["equity"]
    
    n_days = len(eq_series)
    ann_factor = 252 / n_days if n_days > 0 else 1
    ann_ret = ((final_equity / initial_cash) ** ann_factor - 1) * 100
    
    daily_rets = eq_series.pct_change().dropna()
    if len(daily_rets) > 0 and daily_rets.std() > 0:
        sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
    else:
        sharpe = 0
    
    max_dd = ((eq_series / eq_series.cummax()) - 1).min() * 100
    
    win_trades = [t for t in trades if t["return"] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    
    avg_trade_ret = np.mean([t["return"] for t in trades]) * 100 if trades else 0
    avg_days_held = np.mean([t["days_held"] for t in trades]) if trades else 0
    
    # Exit reason breakdown
    exit_counts = {}
    for t in trades:
        r = t["exit_reason"]
        exit_counts[r] = exit_counts.get(r, 0) + 1
    
    # Yearly breakdown
    yearly = {}
    for t in trades:
        year = t["entry_date"].year
        if year not in yearly:
            yearly[year] = {"trades": 0, "wins": 0, "total_return": 0}
        yearly[year]["trades"] += 1
        if t["return"] > 0:
            yearly[year]["wins"] += 1
        yearly[year]["total_return"] += t["return"]
    
    yearly_results = {}
    for year, d in sorted(yearly.items()):
        yearly_results[year] = {
            "trades": d["trades"],
            "win_rate": d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0,
            "total_return_pct": d["total_return"] * 100,
            "avg_return_pct": d["total_return"] / d["trades"] * 100 if d["trades"] > 0 else 0,
        }
    
    return {
        "total_return_pct": round(total_return, 1),
        "annualized_return_pct": round(ann_ret, 1),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 1),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_trade_return_pct": round(avg_trade_ret, 2),
        "avg_days_held": round(avg_days_held, 1),
        "exit_counts": exit_counts,
        "yearly": yearly_results,
    }


# ─── SPY Benchmark ──────────────────────────────────────────────────

def compute_spy_benchmark():
    spy = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet",
                         columns=["date", "sym", "close"])
    spy = spy[(spy["sym"] == "SPY") & (spy["date"] >= "2025-01-01")].sort_values("date")
    
    total_ret = (spy["close"].iloc[-1] / spy["close"].iloc[0] - 1) * 100
    daily_rets = spy["close"].pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
    max_dd = ((spy["close"] / spy["close"].cummax()) - 1).min() * 100
    
    # Yearly
    spy["year"] = spy["date"].dt.year
    yearly = {}
    for year, grp in spy.groupby("year"):
        yr_ret = (grp["close"].iloc[-1] / grp["close"].iloc[0] - 1) * 100
        yearly[int(year)] = round(yr_ret, 1)
    
    return {
        "total_return_pct": round(total_ret, 1),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 1),
        "yearly": yearly,
    }


# ─── Main ───────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  COMPREHENSIVE BACKTEST: V12 Green-Arrow LambdaMART (17 features)")
    print("  OOS Period: 2025-01-01 to present | Lookback starts 2024-06-01")
    print("=" * 70)
    t0 = time.time()
    
    # ── Load Data ──
    print("\n[1/4] Loading price data (2024-06-01 to present)...")
    df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet",
                        columns=["date", "sym", "close", "volume", "high", "low"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2024-06-01"].copy()
    
    # Universe filter
    from universe_filter import filter_green_arrow
    df = filter_green_arrow(df)
    print(f"  Universe: {df['sym'].nunique()} stocks, {df['date'].nunique()} days")
    
    # ── Compute Features ──
    print("[2/4] Computing 17 features...")
    df = compute_features(df)
    
    # ── Model Scoring ──
    print("[3/4] Scoring with LambdaMART V12...")
    model = lgb.Booster(model_file=str(ROOT / "models/us/arrow_v12_lambdamart.txt"))
    
    # Score all (date, sym) combinations
    mask = df[FEATURES].notna().all(axis=1)
    df.loc[mask, "pred"] = model.predict(df.loc[mask, FEATURES])
    df["pred"] = df["pred"].fillna(0)
    
    # Extract score DataFrame
    scores_df = df[["date", "sym", "pred"]].copy()
    print(f"  Scored: {len(scores_df)} stock-day combinations")
    
    # ── SPY Benchmark ──
    print("[4/4] Computing SPY benchmark...")
    spy_bench = compute_spy_benchmark()
    
    # ── Run All Configurations ──
    configs = []
    for hold in [5, 10, 15, 20]:
        for strategy in ["fixed", "stop_only", "trailing_stop", "rsi_bb_exit"]:
            configs.append({
                "hold_days": hold,
                "top_n": 5,
                "exit_strategy": strategy,
                "stop_loss_pct": -0.15,
            })
    
    results = []
    for cfg in configs:
        label = f"H{cfg['hold_days']}_{cfg['exit_strategy']}"
        print(f"\n  Running {label}...", end=" ", flush=True)
        r = run_backtest(df, scores_df, cfg["hold_days"], cfg["top_n"],
                        cfg["exit_strategy"], cfg["stop_loss_pct"])
        if r:
            r["config"] = cfg
            r["label"] = label
            results.append(r)
            print(f"Trades={r['num_trades']}, Return={r['total_return_pct']:+.1f}%")
        else:
            print("NO DATA")
    
    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.0f}s")
    
    # ── Results Table ──
    print("\n" + "=" * 120)
    print("  RESULTS SUMMARY")
    print("=" * 120)
    
    header = f"{'Config':<25} {'Return%':>8} {'AnnRet%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'#Trade':>7} {'Win%':>6} {'AvgRet%':>8} {'AvgDays':>7}"
    print(header)
    print("-" * 120)
    
    # SPY row
    spy_row = f"{'SPY Benchmark':<25} {spy_bench['total_return_pct']:>+8.1f} {'--':>8} {spy_bench['sharpe']:>7.3f} {spy_bench['max_drawdown_pct']:>7.1f} {'--':>7} {'--':>6} {'--':>8} {'--':>7}"
    print(spy_row)
    print("-" * 120)
    
    for r in results:
        row = (
            f"{r['label']:<25}"
            f" {r['total_return_pct']:>+8.1f}"
            f" {r['annualized_return_pct']:>+8.1f}"
            f" {r['sharpe']:>7.3f}"
            f" {r['max_drawdown_pct']:>7.1f}"
            f" {r['num_trades']:>7d}"
            f" {r['win_rate_pct']:>6.1f}"
            f" {r['avg_trade_return_pct']:>+8.2f}"
            f" {r['avg_days_held']:>7.1f}"
        )
        print(row)
    
    # ── Best Config ──
    print("\n" + "=" * 70)
    print("  BEST CONFIGURATIONS")
    print("=" * 70)
    
    by_sharpe = sorted(results, key=lambda x: x["sharpe"], reverse=True)
    by_return = sorted(results, key=lambda x: x["total_return_pct"], reverse=True)
    by_winrate = sorted(results, key=lambda x: x["win_rate_pct"], reverse=True)
    
    print(f"  By Sharpe:    {by_sharpe[0]['label']} ({by_sharpe[0]['sharpe']:.3f})")
    print(f"  By Return:    {by_return[0]['label']} ({by_return[0]['total_return_pct']:+.1f}%)")
    print(f"  By Win Rate:  {by_winrate[0]['label']} ({by_winrate[0]['win_rate_pct']:.1f}%)")
    
    # ── Yearly Breakdown ──
    print("\n" + "=" * 70)
    print("  YEARLY BREAKDOWN (selected configs)")
    print("=" * 70)
    
    # Show yearly for SPY
    print(f"\n  SPY:")
    for year, ret in sorted(spy_bench["yearly"].items()):
        print(f"    {year}: {ret:+.1f}%")
    
    # Show yearly for key configs
    key_configs = ["H5_fixed", "H10_fixed", "H20_fixed", "H10_trailing_stop", "H10_rsi_bb_exit"]
    for r in results:
        if r["label"] in key_configs:
            print(f"\n  {r['label']}:")
            for year, yd in sorted(r["yearly"].items()):
                print(f"    {year}: {yd['total_return_pct']:+.1f}% ({yd['trades']} trades, {yd['win_rate']:.0f}% win)")
    
    # ── Exit Strategy Comparison ──
    print("\n" + "=" * 70)
    print("  EXIT STRATEGY COMPARISON (10-day hold)")
    print("=" * 70)
    
    for r in results:
        if r["config"]["hold_days"] == 10:
            print(f"\n  {r['label']}:")
            print(f"    Total Return: {r['total_return_pct']:+.1f}% | Sharpe: {r['sharpe']:.3f} | MaxDD: {r['max_drawdown_pct']:.1f}%")
            print(f"    Trades: {r['num_trades']} | Win Rate: {r['win_rate_pct']:.1f}% | Avg/Trade: {r['avg_trade_return_pct']:+.2f}%")
            print(f"    Exit Reasons: {r['exit_counts']}")
    
    # ── Save ──
    out_data = {
        "model": "arrow_v12_lambdamart",
        "features": 17,
        "period": "2025-01-01 to present",
        "data_start": "2024-06-01",
        "universe": "green_arrow $1-$10",
        "top_n": 5,
        "spy_benchmark": spy_bench,
        "configs": [],
        "timestamp": datetime.now().isoformat(),
    }
    for r in results:
        out_data["configs"].append({
            "label": r["label"],
            "hold_days": r["config"]["hold_days"],
            "exit_strategy": r["config"]["exit_strategy"],
            "stop_loss_pct": r["config"]["stop_loss_pct"],
            "total_return_pct": r["total_return_pct"],
            "annualized_return_pct": r["annualized_return_pct"],
            "sharpe": r["sharpe"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "num_trades": r["num_trades"],
            "win_rate_pct": r["win_rate_pct"],
            "avg_trade_return_pct": r["avg_trade_return_pct"],
            "avg_days_held": r["avg_days_held"],
            "exit_counts": r["exit_counts"],
            "yearly": r["yearly"],
        })
    
    out_path = ROOT / "data/backtest/arrow_v12_comprehensive.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {out_path}")
    print("  DONE")


if __name__ == "__main__":
    main()
