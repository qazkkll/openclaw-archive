#!/usr/bin/env python3
"""
🦅 Falcon V0.3.1 — OOS Validation Report
Split: IS (2022-2023), Validation (2024H1), OOS (2024H2)
Records Sharpe, MaxDD, WinRate for each period.
Compares in-sample vs OOS degradation.
"""
import pandas as pd, numpy as np, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")

# ═══════════════════════════════════════════════════
# Load SPX data (same as falcon_v03.py)
# ═══════════════════════════════════════════════════
def load_spx():
    print("📊 Loading S&P 500...")
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_historical.json"),
        ("analyst_historical", "analyst_historical.json"),
        ("fmp_key_metrics", "fmp_key_metrics.json"),
        ("fmp_financial_growth", "fmp_financial_growth.json"),
    ]:
        f = DATA_DIR / fname
        data[name] = json.load(open(f)) if f.exists() else {}
    data["fmp_insider"] = {}
    data["fmp_dcf"] = {}
    data["fmp_price_target"] = {}
    n = master["ticker"].nunique()
    print(f"  ✅ {n} tickers, {len(master)} rows")
    return master, data, n


# ═══════════════════════════════════════════════════
# SPX Optimal Config
# ═══════════════════════════════════════════════════
WEIGHTS = {
    "fund_ratio": 0.7,
    "analyst": 0.2,
    "fund_metric": 0.1,
    "tech": 0.0,
}
STRATEGY = "fixed"
PARAMS = {
    "hold_days": 30,
    "stop_loss": -0.15,
    "bear_alloc": 0.50,
}
TOP_N = 5


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    t0 = time.time()

    print("=" * 80)
    print("🦅 Falcon V0.3.1 — OOS Validation Report")
    print("=" * 80)

    # 1. Load data
    master, data, n_tickers = load_spx()

    # 2. Precompute PIT ranks once
    ranks_dict = precompute_pit_ranks(
        master,
        data["fmp_ratios_historical"],
        data["analyst_historical"],
        data["fmp_key_metrics"],
        data["fmp_financial_growth"],
        data["fmp_insider"],
        data["fmp_dcf"],
        data["fmp_price_target"],
    )

    # 3. Build price pivot and regime
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    # 4. Split dates
    all_dates = sorted(ranks_dict.keys())
    is_dates = [d for d in all_dates if "2022" in d or "2023" in d]
    val_dates = [d for d in all_dates if "2024" in d and int(d.split("-")[1]) <= 6]
    oos_dates = [d for d in all_dates if d >= "2024-07-01"]

    print(f"\n📅 Date splits:")
    print(f"   IS (2022-2023):  {len(is_dates)} days  [{is_dates[0]} → {is_dates[-1]}]")
    print(f"   Val (2024H1):    {len(val_dates)} days  [{val_dates[0]} → {val_dates[-1]}]")
    print(f"   OOS (2024H2):    {len(oos_dates)} days  [{oos_dates[0]} → {oos_dates[-1]}]")

    # 5. Run backtests
    print(f"\n⚙️  Config: weights={WEIGHTS}, strategy={STRATEGY}, "
          f"hold_days={PARAMS['hold_days']}, stop_loss={PARAMS['stop_loss']}, "
          f"bear_alloc={PARAMS['bear_alloc']}, top_n={TOP_N}")

    results = {}
    for label, dates in [("IS", is_dates), ("Validation", val_dates), ("OOS", oos_dates)]:
        print(f"\n🔄 Running {label} backtest ({len(dates)} days)...")
        bt = backtest_flexible(
            ranks_dict, price_pivot, dates, regime_above,
            WEIGHTS, STRATEGY, PARAMS, TOP_N
        )
        if bt:
            results[label] = bt
            print(f"   ✅ Sharpe={bt['sharpe']:.3f}  MaxDD={bt['dd']:.2f}%  "
                  f"Ret={bt['ret']:.2f}%  WR={bt['wr']:.1f}%  "
                  f"Trades={bt['trades']}  Rebalances={bt['rebalances']}")
        else:
            results[label] = None
            print(f"   ❌ Insufficient data for {label}")

    # 6. Report
    print(f"\n{'=' * 80}")
    print("📊 OOS VALIDATION REPORT")
    print(f"{'=' * 80}")
    print(f"{'Period':<14} {'Sharpe':>8} {'MaxDD%':>8} {'Return%':>9} {'WinRate%':>9} {'Trades':>7} {'Rebal':>6}")
    print("-" * 65)

    for label in ["IS", "Validation", "OOS"]:
        bt = results.get(label)
        if bt:
            print(f"{label:<14} {bt['sharpe']:>8.3f} {bt['dd']:>7.2f}% {bt['ret']:>8.2f}% "
                  f"{bt['wr']:>8.1f}% {bt['trades']:>7} {bt['rebalances']:>6}")
        else:
            print(f"{label:<14} {'N/A':>8} {'N/A':>8} {'N/A':>9} {'N/A':>9} {'N/A':>7} {'N/A':>6}")

    # 7. Degradation analysis
    print(f"\n{'=' * 80}")
    print("📉 DEGRADATION ANALYSIS (OOS / IS)")
    print(f"{'=' * 80}")

    is_bt = results.get("IS")
    oos_bt = results.get("OOS")
    val_bt = results.get("Validation")

    degradation = {}
    if is_bt and oos_bt:
        sr_ratio = oos_bt["sharpe"] / is_bt["sharpe"] if is_bt["sharpe"] != 0 else float("nan")
        dd_ratio = oos_bt["dd"] / is_bt["dd"] if is_bt["dd"] != 0 else float("nan")
        wr_ratio = oos_bt["wr"] / is_bt["wr"] if is_bt["wr"] != 0 else float("nan")

        degradation = {
            "sharpe_ratio_oos_is": round(sr_ratio, 4),
            "maxdd_ratio_oos_is": round(dd_ratio, 4),
            "winrate_ratio_oos_is": round(wr_ratio, 4),
            "sharpe_is": is_bt["sharpe"],
            "sharpe_oos": oos_bt["sharpe"],
            "maxdd_is": is_bt["dd"],
            "maxdd_oos": oos_bt["dd"],
            "winrate_is": is_bt["wr"],
            "winrate_oos": oos_bt["wr"],
        }

        sr_grade = "🟢" if sr_ratio > 0.7 else "🟡" if sr_ratio > 0.4 else "🔴"
        dd_grade = "🟢" if dd_ratio < 1.5 else "🟡" if dd_ratio < 2.5 else "🔴"
        wr_grade = "🟢" if wr_ratio > 0.85 else "🟡" if wr_ratio > 0.65 else "🔴"

        print(f"  Sharpe ratio (OOS/IS):  {sr_ratio:.4f}  {sr_grade}  "
              f"[{is_bt['sharpe']:.3f} → {oos_bt['sharpe']:.3f}]")
        print(f"  MaxDD ratio (OOS/IS):   {dd_ratio:.4f}  {dd_grade}  "
              f"[{is_bt['dd']:.2f}% → {oos_bt['dd']:.2f}%]")
        print(f"  WinRate ratio (OOS/IS): {wr_ratio:.4f}  {wr_grade}  "
              f"[{is_bt['wr']:.1f}% → {oos_bt['wr']:.1f}%]")

        # Overall grade
        grades = []
        if sr_ratio > 0.7: grades.append("PASS")
        else: grades.append("FAIL")
        if dd_ratio < 2.0: grades.append("PASS")
        else: grades.append("FAIL")
        if oos_bt["dd"] <= 28: grades.append("PASS")
        else: grades.append("FAIL")
        if oos_bt["wr"] >= 42: grades.append("PASS")
        else: grades.append("FAIL")

        pass_rate = grades.count("PASS") / len(grades) * 100
        overall = "🟢 ROBUST" if pass_rate >= 75 else "🟡 MARGINAL" if pass_rate >= 50 else "🔴 OVERFIT"

        print(f"\n  Overall: {overall} ({grades.count('PASS')}/{len(grades)} criteria pass)")
        degradation["overall_grade"] = overall
        degradation["pass_rate_pct"] = pass_rate
        degradation["criteria"] = grades
    else:
        print("  ⚠️  Cannot compute degradation (missing IS or OOS results)")

    # 8. Validation → OOS stability
    if val_bt and oos_bt:
        val_oos_sr = oos_bt["sharpe"] / val_bt["sharpe"] if val_bt["sharpe"] != 0 else float("nan")
        degradation["val_oos_sharpe_ratio"] = round(val_oos_sr, 4)
        print(f"\n  Val→OOS Sharpe stability: {val_oos_sr:.4f}  "
              f"[{val_bt['sharpe']:.3f} → {oos_bt['sharpe']:.3f}]")

    # 9. Save results
    output = {
        "config": {
            "weights": WEIGHTS,
            "strategy": STRATEGY,
            "params": PARAMS,
            "top_n": TOP_N,
        },
        "n_tickers": n_tickers,
        "date_ranges": {
            "IS": {"start": is_dates[0], "end": is_dates[-1], "n_days": len(is_dates)},
            "Validation": {"start": val_dates[0], "end": val_dates[-1], "n_days": len(val_dates)},
            "OOS": {"start": oos_dates[0], "end": oos_dates[-1], "n_days": len(oos_dates)},
        },
        "results": {k: v for k, v in results.items()},
        "degradation": degradation,
    }

    out_path = DATA_DIR / "oos_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {out_path}")

    print(f"\n⏱️  Total time: {time.time()-t0:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
