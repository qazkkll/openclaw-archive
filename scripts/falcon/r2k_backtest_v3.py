#!/usr/bin/env python3
"""
R2K 扩展回测 V3: 用扩展历史数据(2018-2026) + 4/4因子覆盖
==========================================
V2问题: 只有2022-2024数据，仅1个WF窗口
V3改进: 用russell_prices_extended.json(2018-2026)，可获得3+个WF窗口

Walk-Forward: 2yr train, 6mo test, hold_days=30, top_n=10
"""
import sys
import json
import time
import warnings
from pathlib import Path
from bisect import bisect_right
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))
from backtest_engine import BacktestEngine, DataQualityError

FMP_FILING_DELAY = 33
HOLD_DAYS = 30
TOP_N = 10

# Factor fields
RATIO_FIELDS = [
    "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
    "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
    "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
    "ebitdaMargin", "assetTurnover", "inventoryTurnover",
    "receivablesTurnover", "debtToEquityRatio", "currentRatio",
    "quickRatio", "financialLeverageRatio",
    "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
    "dividendYieldPercentage", "dividendPayoutRatio",
]
METRIC_FIELDS = [
    "earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
    "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
    "returnOnCapitalEmployed", "returnOnInvestedCapital",
    "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
    "cashConversionCycle", "capexToRevenue", "capexToDepreciation",
    "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
    "netDebtToEBITDA", "operatingReturnOnAssets",
]
GROWTH_FIELDS = [
    "revenueGrowth", "grossProfitGrowth", "ebitgrowth",
    "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
    "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
    "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
    "receivablesGrowth", "inventoryGrowth", "assetGrowth",
    "bookValueperShareGrowth", "debtGrowth",
]
ANALYST_FIELDS = ["eps_revision", "revenue_revision", "eps_dispersion", "num_analysts_eps"]
ANALYST_FIELD_ALIASES = {"num_analysts_eps": "numAnalystsEps"}
QOQ_MARGIN_FIELDS = ["grossProfitMargin", "netProfitMargin", "operatingProfitMargin", "ebitdaMargin"]


def build_pit_index(quarterly_data):
    if not quarterly_data:
        return ([], [])
    pairs = []
    for q in quarterly_data:
        if not isinstance(q, dict) or not q.get("date"):
            continue
        try:
            qdate = datetime.strptime(q["date"], "%Y-%m-%d")
            avail = (qdate + timedelta(days=FMP_FILING_DELAY)).strftime("%Y-%m-%d")
        except ValueError:
            continue
        pairs.append((avail, q))
    pairs.sort(key=lambda x: x[0])
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def build_analyst_index(analyst_data):
    if not analyst_data:
        return ([], [])
    pairs = [(r["date"], r) for r in analyst_data if isinstance(r, dict) and r.get("date")]
    pairs.sort(key=lambda x: x[0])
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def get_pit_from_index(avail_dates, entries, date):
    if not avail_dates:
        return {}
    idx = bisect_right(avail_dates, date) - 1
    if idx < 0:
        return {}
    return entries[idx]


def load_data():
    """Load all data sources."""
    print("📦 Loading data...")
    t0 = time.time()
    data_dir = PROJECT_ROOT / "data" / "falcon"

    # SPX features (already has ranked factors)
    features = pd.read_parquet(data_dir / "features_v04_1.parquet")
    features["date"] = features["date"].astype(str)
    spx_tickers = set(features["ticker"].unique())

    # SPX price pivot
    spx_prices = features.pivot_table(index="date", columns="ticker", values="close").sort_index()

    # R2K data - use extended prices
    r2k = {}
    for name in ["fmp_ratios_russell", "fmp_metrics_russell", "fmp_growth_russell",
                  "fmp_analyst_russell"]:
        with open(data_dir / f"{name}.json") as f:
            r2k[name] = json.load(f)
    
    # Use extended prices
    with open(PROJECT_ROOT / "data" / "fmp_premium" / "snapshots" / "russell_prices_extended.json") as f:
        r2k["russell_prices"] = json.load(f)

    # Filter R2K to 4/4 coverage only
    r2k_tickers_all = set(r2k["fmp_ratios_russell"].keys()) | set(r2k["fmp_metrics_russell"].keys())
    r2k_quality = {}
    for t in r2k_tickers_all:
        score = sum([
            len(r2k["fmp_ratios_russell"].get(t, [])) > 0,
            len(r2k["fmp_metrics_russell"].get(t, [])) > 0,
            len(r2k["fmp_growth_russell"].get(t, [])) > 0,
            len(r2k["fmp_analyst_russell"].get(t, [])) > 0,
        ])
        r2k_quality[t] = score

    r2k_full = {t for t, s in r2k_quality.items() if s == 4}

    # R2K price pivot (filtered to 4/4)
    all_prices = []
    for ticker, records in r2k["russell_prices"].items():
        if ticker not in r2k_full:
            continue
        for r in records:
            if r.get("date") and r.get("close"):
                all_prices.append({"ticker": ticker, "date": r["date"], "close": r["close"]})
    r2k_prices = pd.DataFrame(all_prices).pivot_table(index="date", columns="ticker", values="close").sort_index()

    print(f"  SPX: {len(spx_tickers)} tickers, {spx_prices.shape} prices")
    print(f"  R2K 4/4覆盖: {len(r2k_full)} tickers, {r2k_prices.shape} prices")
    print(f"  ⏱️ {time.time()-t0:.1f}s")

    return features, spx_prices, r2k, r2k_full, r2k_prices


def build_spx_ranks(features):
    """Build SPX cross-sectional ranks from features parquet."""
    print("📊 Building SPX ranks...")
    t0 = time.time()
    dates = sorted(features["date"].unique())
    ranks_dict = {}

    for date in dates:
        day = features[features["date"] == date].copy()
        if len(day) < 10:
            continue
        day = day.set_index("ticker")
        row = pd.DataFrame(index=day.index)

        r_cols = [c for c in day.columns if c.startswith("r_") and "_qoq" not in c]
        m_cols = [c for c in day.columns if c.startswith("m_")]
        g_cols = [c for c in day.columns if c.startswith("g_")]
        a_cols = [c for c in day.columns if c.startswith("a_")]
        qoq_cols = [c for c in day.columns if c.startswith("r_") and "_qoq" in c]

        for cols in [r_cols, m_cols, g_cols, a_cols, qoq_cols]:
            for c in cols:
                if day[c].notna().sum() > 5:
                    row[c] = day[c].rank(pct=True)

        r_rc = [c for c in row.columns if c.startswith("r_") and "_qoq" not in c]
        m_rc = [c for c in row.columns if c.startswith("m_")]
        g_rc = [c for c in row.columns if c.startswith("g_")]
        q_rc = [c for c in row.columns if c.startswith("r_") and "_qoq" in c]

        row["fund_ratio"] = row[r_rc].mean(axis=1) if r_rc else 0.5
        row["fund_metric"] = row[m_rc].mean(axis=1) if m_rc else 0.5
        row["growth_composite"] = row[g_rc].mean(axis=1) if g_rc else 0.5
        row["qoq"] = row[q_rc].mean(axis=1) if q_rc else 0.5
        row["fund_ratio"] = row[["fund_ratio", "fund_metric"]].mean(axis=1)
        ranks_dict[date] = row

    print(f"  ✅ {len(ranks_dict)} dates ({time.time()-t0:.1f}s)")
    return ranks_dict


def build_r2k_ranks(r2k, r2k_tickers):
    """Build R2K ranks using PIT data."""
    print(f"📊 Building R2K ranks ({len(r2k_tickers)} tickers)...")
    t0 = time.time()

    # Build PIT indices
    indices = {"ratios": {}, "metrics": {}, "growth": {}, "analyst": {}}
    for t in r2k_tickers:
        indices["ratios"][t] = build_pit_index(r2k["fmp_ratios_russell"].get(t, []))
        indices["metrics"][t] = build_pit_index(r2k["fmp_metrics_russell"].get(t, []))
        indices["growth"][t] = build_pit_index(r2k["fmp_growth_russell"].get(t, []))
        indices["analyst"][t] = build_analyst_index(r2k["fmp_analyst_russell"].get(t, []))

    # Get all dates from R2K prices
    all_prices = []
    for ticker, records in r2k["russell_prices"].items():
        if ticker not in r2k_tickers:
            continue
        for r in records:
            if r.get("date") and r.get("close"):
                all_prices.append(r["date"])
    all_dates = sorted(set(all_prices))

    ranks_dict = {}
    for di, date in enumerate(all_dates):
        day_tickers = [t for t in r2k_tickers if len(indices["ratios"][t][0]) > 0]
        if len(day_tickers) < 10:
            continue

        row_data = {}

        # Ratios
        for f in RATIO_FIELDS:
            vals = {}
            for t in day_tickers:
                pit = get_pit_from_index(*indices["ratios"][t], date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row_data[f"r_{f}"] = pd.Series(vals).rank(pct=True)

        # Metrics
        for f in METRIC_FIELDS:
            vals = {}
            for t in day_tickers:
                pit = get_pit_from_index(*indices["metrics"][t], date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row_data[f"m_{f}"] = pd.Series(vals).rank(pct=True)

        # Growth
        for f in GROWTH_FIELDS:
            vals = {}
            for t in day_tickers:
                pit = get_pit_from_index(*indices["growth"][t], date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row_data[f"g_{f}"] = pd.Series(vals).rank(pct=True)

        # Analyst
        for f in ANALYST_FIELDS:
            json_key = ANALYST_FIELD_ALIASES.get(f, f)
            vals = {}
            for t in day_tickers:
                pit = get_pit_from_index(*indices["analyst"][t], date)
                v = pit.get(json_key)
                if v is not None:
                    vals[t] = v
            if len(vals) > 5:
                row_data[f"a_{f}"] = pd.Series(vals).rank(pct=True)

        # QoQ margins
        for t in day_tickers:
            ad, en = indices["ratios"][t]
            if ad:
                current_pit = get_pit_from_index(ad, en, date)
                if current_pit and current_pit.get("date"):
                    idx = bisect_right(ad, date) - 1
                    if idx > 0:
                        prev = en[idx - 1]
                        for margin in QOQ_MARGIN_FIELDS:
                            curr = current_pit.get(margin)
                            prev_v = prev.get(margin)
                            key = f"r_{margin}_qoq"
                            if curr is not None and prev_v is not None and prev_v != 0:
                                if key not in row_data:
                                    row_data[key] = {}
                                row_data[key][t] = (curr - prev_v) / abs(prev_v)

        for key in list(row_data.keys()):
            if "_qoq" in key and isinstance(row_data[key], dict):
                if len(row_data[key]) > 10:
                    row_data[key] = pd.Series(row_data[key]).rank(pct=True)
                else:
                    del row_data[key]

        if not row_data:
            continue

        df = pd.DataFrame(row_data)
        r_cols = [c for c in df.columns if c.startswith("r_") and "_qoq" not in c]
        m_cols = [c for c in df.columns if c.startswith("m_")]
        g_cols = [c for c in df.columns if c.startswith("g_")]
        qoq_cols = [c for c in df.columns if c.startswith("r_") and "_qoq" in c]

        df["fund_ratio"] = df[r_cols].mean(axis=1) if r_cols else 0.5
        df["fund_metric"] = df[m_cols].mean(axis=1) if m_cols else 0.5
        df["growth_composite"] = df[g_cols].mean(axis=1) if g_cols else 0.5
        df["qoq"] = df[qoq_cols].mean(axis=1) if qoq_cols else 0.5
        df["fund_ratio"] = df[["fund_ratio", "fund_metric"]].mean(axis=1)
        ranks_dict[date] = df

        if (di + 1) % 100 == 0:
            print(f"  📊 {di+1}/{len(all_dates)} ({(di+1)/(time.time()-t0):.0f}/s)")

    print(f"  ✅ {len(ranks_dict)} dates ({time.time()-t0:.1f}s)")
    return ranks_dict


def merge_and_run(spx_ranks, r2k_ranks, spx_prices, r2k_prices, mode="merged"):
    """Run walk-forward backtest on merged or separate universe."""
    print(f"\n{'='*60}")
    print(f"🧪 Running backtest: {mode}")
    print(f"{'='*60}")
    t0 = time.time()

    common_dates = sorted(set(spx_ranks.keys()) & set(r2k_ranks.keys()) & set(spx_prices.index) & set(r2k_prices.index))
    if not common_dates:
        print("  ❌ No common dates!")
        return None

    print(f"  Common dates: {common_dates[0]} → {common_dates[-1]} ({len(common_dates)} days)")

    # Build combined ranks + prices
    combined_ranks = {}
    combined_prices = pd.concat([spx_prices, r2k_prices], axis=1)
    combined_prices = combined_prices.loc[:, ~combined_prices.columns.duplicated(keep='first')]

    for date in common_dates:
        spx_df = spx_ranks.get(date, pd.DataFrame())
        r2k_df = r2k_ranks.get(date, pd.DataFrame())

        if mode == "spx_only":
            combined = spx_df
        elif mode == "r2k_only":
            combined = r2k_df
        else:
            combined = pd.concat([spx_df, r2k_df])
            combined = combined[~combined.index.duplicated(keep='first')]
            # Re-rank composites cross-sectionally
            for col in ["fund_ratio", "growth_composite", "qoq"]:
                if col in combined.columns and combined[col].notna().sum() > 10:
                    combined[col] = combined[col].rank(pct=True)

        combined_ranks[date] = combined

    # Walk-forward
    all_dates_arr = sorted(combined_ranks.keys())
    train_window = 504  # ~2yr
    test_window = 126   # ~6mo

    windows = []
    start = 0
    while start + train_window + test_window <= len(all_dates_arr):
        train_end = start + train_window
        test_end = min(train_end + test_window, len(all_dates_arr))
        windows.append((all_dates_arr[start:train_end], all_dates_arr[train_end:test_end]))
        start += test_window

    print(f"  Walk-forward windows: {len(windows)}")

    # For each window, compute composite score and evaluate
    engine = BacktestEngine()

    all_returns = []
    window_results = []

    for wi, (train_dates, test_dates) in enumerate(windows):
        # Score stocks on each test date
        test_scores = {}
        test_price_changes = {}

        for test_date in test_dates:
            ranks = combined_ranks.get(test_date)
            if ranks is None or len(ranks) < 20:
                continue

            # Composite score (V0.4.4 weights)
            score_cols = []
            if "fund_ratio" in ranks.columns:
                score_cols.append(("fund_ratio", 0.45))
            if "growth_composite" in ranks.columns:
                score_cols.append(("growth_composite", 0.20))
            if "qoq" in ranks.columns:
                score_cols.append(("qoq", 0.20))
            if "cashflow" in ranks.columns:
                score_cols.append(("cashflow", 0.15))

            # Normalize weights
            total_w = sum(w for _, w in score_cols)
            score_cols = [(n, w / total_w) for n, w in score_cols]

            scores = pd.Series(dtype=float)
            for col, weight in score_cols:
                if col in ranks.columns:
                    s = ranks[col].dropna()
                    if len(s) > 0:
                        scores = scores.add(s * weight, fill_value=0)

            if len(scores) < 20:
                continue

            test_scores[test_date] = scores

        # Evaluate: top N stocks vs bottom N stocks
        for test_date in test_scores:
            scores = test_scores[test_date]
            top_stocks = scores.nlargest(TOP_N).index.tolist()
            bottom_stocks = scores.nsmallest(TOP_N).index.tolist()

            # Compute 30-day forward return
            try:
                test_idx = all_dates_arr.index(test_date)
            except ValueError:
                continue

            if test_idx + HOLD_DAYS >= len(all_dates_arr):
                continue

            end_date = all_dates_arr[test_idx + HOLD_DAYS]
            prices = combined_prices

            if test_date not in prices.index or end_date not in prices.index:
                continue

            top_returns = []
            bottom_returns = []
            for stock in top_stocks:
                if stock in prices.columns:
                    s_price = prices.loc[test_date, stock]
                    e_price = prices.loc[end_date, stock]
                    if pd.notna(s_price) and pd.notna(e_price) and s_price > 0:
                        top_returns.append((e_price - s_price) / s_price)

            for stock in bottom_stocks:
                if stock in prices.columns:
                    s_price = prices.loc[test_date, stock]
                    e_price = prices.loc[end_date, stock]
                    if pd.notna(s_price) and pd.notna(e_price) and s_price > 0:
                        bottom_returns.append((e_price - s_price) / s_price)

            if top_returns and bottom_returns:
                all_returns.append({
                    "date": test_date,
                    "top_mean": np.mean(top_returns),
                    "bottom_mean": np.mean(bottom_returns),
                    "long_short": np.mean(top_returns) - np.mean(bottom_returns),
                    "top_n": len(top_returns),
                    "bottom_n": len(bottom_returns),
                })

        window_results.append({
            "window": wi + 1,
            "train": f"{train_dates[0]} → {train_dates[-1]}",
            "test": f"{test_dates[0]} → {test_dates[-1]}",
        })

    if not all_returns:
        print("  ❌ No valid returns!")
        return None

    df_ret = pd.DataFrame(all_returns)

    # Summary
    win_rate = (df_ret["long_short"] > 0).mean()
    avg_top = df_ret["top_mean"].mean() * 100
    avg_ls = df_ret["long_short"].mean() * 100

    # Sharpe (on L/S spread)
    ls_series = df_ret.set_index("date")["long_short"]
    monthly_sharpe = ls_series.mean() / ls_series.std() if ls_series.std() > 0 else 0
    annual_sharpe = monthly_sharpe * np.sqrt(12)

    # Max drawdown — 基于top组合实际净值，不是L/S spread
    top_series = df_ret.set_index("date")["top_mean"]
    cum_top = (1 + top_series).cumprod()
    peak_top = cum_top.expanding().max()
    dd_top = (cum_top - peak_top) / peak_top
    max_dd = dd_top.min() * 100

    elapsed = time.time() - t0
    print(f"\n📊 Results: {mode}")
    print(f"  Total trades: {len(all_returns)}")
    print(f"  Win rate (L/S > 0): {win_rate:.1%}")
    print(f"  Avg 30d return (Top{TOP_N}): {avg_top:+.2f}%")
    print(f"  Avg 30d L/S spread: {avg_ls:+.2f}%")
    print(f"  Sharpe (monthly): {monthly_sharpe:.3f}")
    print(f"  Sharpe (annual): {annual_sharpe:.3f}")
    print(f"  Max drawdown: {max_dd:.1f}%")
    print(f"  Walk-forward windows: {len(windows)}")
    print(f"  ⏱️ {elapsed:.1f}s")

    # Per-window breakdown
    print(f"\n  Window breakdown:")
    for wr in window_results:
        print(f"    W{wr['window']}: train={wr['train']}, test={wr['test']}")

    return {
        "mode": mode,
        "trades": len(all_returns),
        "win_rate": float(win_rate),
        "avg_top_return": float(avg_top),
        "avg_ls_spread": float(avg_ls),
        "sharpe_monthly": float(monthly_sharpe),
        "sharpe_annual": float(annual_sharpe),
        "max_dd": float(max_dd),
        "windows": len(windows),
        "window_details": window_results,
    }


def main():
    print("=" * 60)
    print("🦅 R2K Extension Backtest V3 (2018-2026)")
    print("=" * 60)
    
    t0 = time.time()
    
    # Load data
    features, spx_prices, r2k, r2k_full, r2k_prices = load_data()
    
    # Build ranks
    spx_ranks = build_spx_ranks(features)
    r2k_ranks = build_r2k_ranks(r2k, r2k_full)
    
    # Run backtests
    results = {}
    for mode in ["spx_only", "r2k_only", "merged"]:
        result = merge_and_run(spx_ranks, r2k_ranks, spx_prices, r2k_prices, mode)
        if result:
            results[mode] = result
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "hold_days": HOLD_DAYS,
            "top_n": TOP_N,
            "train_window": 504,
            "test_window": 126,
            "data_source": "russell_prices_extended.json (2018-2026)",
            "factor_requirement": "4/4 coverage",
        },
        "r2k_quality": {
            "total_tickers": len(r2k.get("russell_prices", {})),
            "full_coverage_4_4": len(r2k_full),
        },
        "results": results,
    }
    
    output_file = PROJECT_ROOT / "data" / "falcon" / "r2k_backtest_v3_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Results saved to {output_file}")
    print(f"⏱️ Total time: {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
