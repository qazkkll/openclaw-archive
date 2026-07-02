#!/usr/bin/env python3
"""
R2K 扩展回测: SPX 476只 vs SPX + Russell 2000
================================================
评估扩展universe对模型表现的影响。

V0.4.4 weights: fund_ratio=0.45, growth_composite=0.20, qoq=0.20, cashflow=0.15
R2K限制: 无balance sheet/cashflow/income stmt → 无cashflow因子

Walk-Forward: 2yr train, 6mo test, hold_days=30, top_n=10
"""
import sys
import json
import time
from pathlib import Path
from bisect import bisect_right
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Path setup
PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

# ═══════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════
FMP_FILING_DELAY = 33  # days

# Factor field definitions (same as build_features_v041.py)
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


# ═══════════════════════════════════════════════════
# PIT Index Infrastructure
# ═══════════════════════════════════════════════════

def build_pit_index(quarterly_data):
    """Build PIT index: (avail_dates, entries) sorted by avail_date."""
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
    """Analyst PIT: no filing delay."""
    if not analyst_data:
        return ([], [])
    pairs = []
    for r in analyst_data:
        if not isinstance(r, dict) or not r.get("date"):
            continue
        pairs.append((r["date"], r))
    pairs.sort(key=lambda x: x[0])
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def get_pit_from_index(avail_dates, entries, date):
    """O(log n) PIT lookup."""
    if not avail_dates:
        return {}
    idx = bisect_right(avail_dates, date) - 1
    if idx < 0:
        return {}
    return entries[idx]


# ═══════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════

def load_spx_data():
    """Load SPX features and prices."""
    print("📦 Loading SPX data...")
    t0 = time.time()
    
    features = pd.read_parquet(PROJECT_ROOT / "data" / "falcon" / "features_v04_1.parquet")
    features["date"] = features["date"].astype(str)
    
    # Build price pivot
    price_pivot = features.pivot_table(index="date", columns="ticker", values="close")
    price_pivot = price_pivot.sort_index()
    
    print(f"  ✅ SPX: {features.shape[0]} rows, {features['ticker'].nunique()} tickers")
    print(f"  ✅ Price pivot: {price_pivot.shape}")
    print(f"  ✅ Date range: {price_pivot.index.min()} → {price_pivot.index.max()}")
    print(f"  ⏱️ {time.time()-t0:.1f}s")
    
    return features, price_pivot


def load_r2k_data():
    """Load Russell 2000 JSON data."""
    print("📦 Loading Russell 2000 data...")
    t0 = time.time()
    data_dir = PROJECT_ROOT / "data" / "falcon"
    
    data = {}
    for name in ["fmp_ratios_russell", "fmp_metrics_russell", "fmp_growth_russell", 
                  "fmp_analyst_russell", "russell_prices"]:
        path = data_dir / f"{name}.json"
        with open(path) as f:
            data[name] = json.load(f)
        n = len(data[name])
        print(f"  ✅ {name}: {n} tickers")
    
    print(f"  ⏱️ {time.time()-t0:.1f}s")
    return data


# ═══════════════════════════════════════════════════
# R2K Feature Construction
# ═══════════════════════════════════════════════════

def build_r2k_pit_indices(r2k_data):
    """Build PIT indices for all R2K data sources."""
    print("📊 Building R2K PIT indices...")
    t0 = time.time()
    
    # Collect all R2K tickers
    all_tickers = set()
    for key in ["fmp_ratios_russell", "fmp_metrics_russell", "fmp_growth_russell", "fmp_analyst_russell"]:
        all_tickers.update(r2k_data.get(key, {}).keys())
    
    indices = {}
    for t in all_tickers:
        # Ratios
        if "ratios" not in indices:
            indices["ratios"] = {}
        indices["ratios"][t] = build_pit_index(r2k_data.get("fmp_ratios_russell", {}).get(t, []))
        
        # Metrics
        if "metrics" not in indices:
            indices["metrics"] = {}
        indices["metrics"][t] = build_pit_index(r2k_data.get("fmp_metrics_russell", {}).get(t, []))
        
        # Growth
        if "growth" not in indices:
            indices["growth"] = {}
        indices["growth"][t] = build_pit_index(r2k_data.get("fmp_growth_russell", {}).get(t, []))
        
        # Analyst
        if "analyst" not in indices:
            indices["analyst"] = {}
        indices["analyst"][t] = build_analyst_index(r2k_data.get("fmp_analyst_russell", {}).get(t, []))
    
    print(f"  ✅ {len(all_tickers)} tickers indexed ({time.time()-t0:.1f}s)")
    return indices, all_tickers


def build_r2k_price_pivot(r2k_data):
    """Build price pivot from russell_prices.json."""
    prices_raw = r2k_data["russell_prices"]
    
    all_prices = []
    for ticker, records in prices_raw.items():
        for r in records:
            if r.get("date") and r.get("close"):
                all_prices.append({"ticker": ticker, "date": r["date"], "close": r["close"]})
    
    df = pd.DataFrame(all_prices)
    pivot = df.pivot_table(index="date", columns="ticker", values="close")
    pivot = pivot.sort_index()
    
    print(f"  ✅ R2K Price pivot: {pivot.shape}")
    return pivot


def compute_r2k_ranks(r2k_data, indices, r2k_tickers, all_dates):
    """Compute percentile-ranked factors for R2K tickers on each date.
    
    Returns: {date_str: DataFrame(ticker -> factor_rank)}
    """
    print("📊 Computing R2K PIT ranks...")
    t0 = time.time()
    
    ranks_dict = {}
    total = len(all_dates)
    
    for di, date in enumerate(all_dates):
        # Get R2K tickers with price data on this date
        day_tickers = []
        for t in r2k_tickers:
            r_idx = indices.get("ratios", {}).get(t, ([], []))
            m_idx = indices.get("metrics", {}).get(t, ([], []))
            g_idx = indices.get("growth", {}).get(t, ([], []))
            a_idx = indices.get("analyst", {}).get(t, ([], []))
            has_data = any(len(idx[0]) > 0 for idx in [r_idx, m_idx, g_idx, a_idx])
            if has_data:
                day_tickers.append(t)
        
        if len(day_tickers) < 10:
            continue
        
        # Build rank DataFrame for this date
        row_data = {}
        
        # Ratios rank
        for f in RATIO_FIELDS:
            vals = {}
            for t in day_tickers:
                ad, en = indices.get("ratios", {}).get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row_data[f"r_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Metrics rank
        for f in METRIC_FIELDS:
            vals = {}
            for t in day_tickers:
                ad, en = indices.get("metrics", {}).get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row_data[f"m_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Growth rank
        for f in GROWTH_FIELDS:
            vals = {}
            for t in day_tickers:
                ad, en = indices.get("growth", {}).get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row_data[f"g_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Analyst rank
        for f in ANALYST_FIELDS:
            json_key = ANALYST_FIELD_ALIASES.get(f, f)
            vals = {}
            for t in day_tickers:
                ad, en = indices.get("analyst", {}).get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(json_key)
                if v is not None:
                    vals[t] = v
            if len(vals) > 5:
                row_data[f"a_{f}"] = pd.Series(vals).rank(pct=True)
        
        # QoQ margins (from ratios)
        for t in day_tickers:
            r_idx = indices.get("ratios", {}).get(t, ([], []))
            if r_idx[0]:
                ad_r, en_r = r_idx
                current_pit = get_pit_from_index(ad_r, en_r, date)
                if current_pit and current_pit.get("date"):
                    avail_dates, entries = r_idx
                    idx = bisect_right(avail_dates, date) - 1
                    if idx > 0:
                        prev = entries[idx - 1]
                        for margin in QOQ_MARGIN_FIELDS:
                            curr = current_pit.get(margin)
                            prev_v = prev.get(margin)
                            key = f"r_{margin}_qoq"
                            if curr is not None and prev_v is not None and prev_v != 0:
                                if key not in row_data:
                                    row_data[key] = {}
                                row_data[key][t] = (curr - prev_v) / abs(prev_v)
        
        # Convert QoQ dict to Series and rank
        for key in list(row_data.keys()):
            if "_qoq" in key and isinstance(row_data[key], dict):
                if len(row_data[key]) > 10:
                    row_data[key] = pd.Series(row_data[key]).rank(pct=True)
                else:
                    del row_data[key]
        
        if not row_data:
            continue
        
        # Build DataFrame
        df = pd.DataFrame(row_data)
        
        # Compute group means (same logic as falcon_v03_engine.py)
        r_cols = [c for c in df.columns if c.startswith("r_") and "_qoq" not in c]
        m_cols = [c for c in df.columns if c.startswith("m_")]
        g_cols = [c for c in df.columns if c.startswith("g_")]
        a_cols = [c for c in df.columns if c.startswith("a_")]
        qoq_cols = [c for c in df.columns if c.startswith("r_") and "_qoq" in c]
        
        df["fund_ratio"] = df[r_cols].mean(axis=1) if r_cols else 0.5
        df["fund_metric"] = df[m_cols].mean(axis=1) if m_cols else 0.5
        df["fund_growth"] = df[g_cols].mean(axis=1) if g_cols else 0.5
        df["analyst"] = df[a_cols].mean(axis=1) if a_cols else 0.5
        df["qoq"] = df[qoq_cols].mean(axis=1) if qoq_cols else 0.5
        
        # Combined groups for backtest
        # fund_ratio composite = mean of r_* and m_* groups
        df["fund_ratio"] = df[["fund_ratio", "fund_metric"]].mean(axis=1)
        
        # growth_composite = g_* group mean
        df["growth_composite"] = df["fund_growth"]
        
        ranks_dict[date] = df
        
        if (di + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (di + 1) / elapsed if elapsed > 0 else 0
            print(f"  📊 {di+1}/{total} dates ({rate:.0f} dates/s)")
    
    print(f"  ✅ R2K ranks: {len(ranks_dict)} dates ({time.time()-t0:.1f}s)")
    return ranks_dict


def build_spx_ranks_from_features(features):
    """Build ranks dict from SPX features_v04_1.parquet.
    
    The features file already has raw factor values. We need to
    percentile-rank them cross-sectionally per date.
    """
    print("📊 Building SPX ranks from features...")
    t0 = time.time()
    
    dates = sorted(features["date"].unique())
    ranks_dict = {}
    
    for date in dates:
        day = features[features["date"] == date].copy()
        if len(day) < 10:
            continue
        
        day = day.set_index("ticker")
        
        row = pd.DataFrame(index=day.index)
        
        # Fund ratio components (r_* fields from features)
        r_cols = [c for c in day.columns if c.startswith("r_") and "_qoq" not in c]
        m_cols = [c for c in day.columns if c.startswith("m_")]
        g_cols = [c for c in day.columns if c.startswith("g_")]
        a_cols = [c for c in day.columns if c.startswith("a_")]
        qoq_cols = [c for c in day.columns if c.startswith("r_") and "_qoq" in c]
        
        # Rank each factor cross-sectionally
        for c in r_cols:
            if day[c].notna().sum() > 10:
                row[c] = day[c].rank(pct=True)
        
        for c in m_cols:
            if day[c].notna().sum() > 10:
                row[c] = day[c].rank(pct=True)
        
        for c in g_cols:
            if day[c].notna().sum() > 10:
                row[c] = day[c].rank(pct=True)
        
        for c in a_cols:
            if day[c].notna().sum() > 5:
                row[c] = day[c].rank(pct=True)
        
        for c in qoq_cols:
            if day[c].notna().sum() > 10:
                row[c] = day[c].rank(pct=True)
        
        # Compute group means
        r_rank_cols = [c for c in row.columns if c.startswith("r_") and "_qoq" not in c]
        m_rank_cols = [c for c in row.columns if c.startswith("m_")]
        g_rank_cols = [c for c in row.columns if c.startswith("g_")]
        qoq_rank_cols = [c for c in row.columns if c.startswith("r_") and "_qoq" in c]
        
        row["fund_ratio"] = row[r_rank_cols].mean(axis=1) if r_rank_cols else 0.5
        row["fund_metric"] = row[m_rank_cols].mean(axis=1) if m_rank_cols else 0.5
        row["fund_growth"] = row[g_rank_cols].mean(axis=1) if g_rank_cols else 0.5
        row["qoq"] = row[qoq_rank_cols].mean(axis=1) if qoq_rank_cols else 0.5
        
        # Combined groups
        row["fund_ratio"] = row[["fund_ratio", "fund_metric"]].mean(axis=1)
        row["growth_composite"] = row["fund_growth"]
        
        # Keep only the composite factors + raw ranked factors for the backtest
        ranks_dict[date] = row
    
    print(f"  ✅ SPX ranks: {len(ranks_dict)} dates ({time.time()-t0:.1f}s)")
    return ranks_dict


# ═══════════════════════════════════════════════════
# Merge Ranks
# ═══════════════════════════════════════════════════

def merge_ranks(spx_ranks, r2k_ranks):
    """Merge SPX and R2K ranks, computing cross-sectional ranks across all tickers."""
    print("📊 Merging SPX + R2K ranks (cross-sectional re-ranking)...")
    t0 = time.time()
    
    # Find common dates
    common_dates = sorted(set(spx_ranks.keys()) & set(r2k_ranks.keys()))
    print(f"  Common dates: {len(common_dates)}")
    
    merged = {}
    
    # Factor groups to re-rank
    factor_groups = {
        "fund_ratio": ["fund_ratio"],
        "growth_composite": ["growth_composite"],
        "qoq": ["qoq"],
    }
    
    for date in common_dates:
        spx_df = spx_ranks.get(date, pd.DataFrame())
        r2k_df = r2k_ranks.get(date, pd.DataFrame())
        
        if spx_df.empty and r2k_df.empty:
            continue
        
        # Concat
        combined = pd.concat([spx_df, r2k_df])
        combined = combined[~combined.index.duplicated(keep='first')]
        
        # Re-rank key composite factors cross-sectionally
        for group_name, factors in factor_groups.items():
            for f in factors:
                if f in combined.columns:
                    if combined[f].notna().sum() > 10:
                        combined[f] = combined[f].rank(pct=True)
        
        merged[date] = combined
    
    print(f"  ✅ Merged ranks: {len(merged)} dates ({time.time()-t0:.1f}s)")
    return merged


# ═══════════════════════════════════════════════════
# Combined Price Pivot
# ═══════════════════════════════════════════════════

def merge_price_pivots(spx_prices, r2k_prices):
    """Merge SPX and R2K price pivots."""
    print("📊 Merging price pivots...")
    
    # Find common dates
    common_dates = sorted(set(spx_prices.index) & set(r2k_prices.index))
    print(f"  Common dates: {len(common_dates)}")
    
    # Combine
    spx_sub = spx_prices.loc[spx_prices.index.isin(common_dates)]
    r2k_sub = r2k_prices.loc[r2k_prices.index.isin(common_dates)]
    
    combined = pd.concat([spx_sub, r2k_sub], axis=1)
    
    # Remove duplicate columns (keep SPX if overlap)
    combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
    
    print(f"  ✅ Combined price pivot: {combined.shape}")
    return combined


# ═══════════════════════════════════════════════════
# Backtest Runner
# ═══════════════════════════════════════════════════

def run_walk_forward_backtest(ranks, prices, weights, label, 
                               hold_days=30, top_n=10,
                               train_years=2, test_months=6):
    """Run walk-forward backtest and return results."""
    print(f"\n{'='*60}")
    print(f"🦅 Walk-Forward: {label}")
    print(f"  Weights: {weights}")
    print(f"  Top-N: {top_n}, Hold: {hold_days}d, Train: {train_years}yr, Test: {test_months}mo")
    print(f"{'='*60}")
    
    engine = BacktestEngine(cost=0.001, stop_loss=-0.15)
    
    try:
        result = engine.walk_forward(
            ranks, prices, weights,
            hold_days=hold_days, top_n=top_n,
            train_years=train_years, test_months=test_months
        )
        
        print(f"\n  📊 Results:")
        print(f"    Sharpe:  {result.sharpe:.3f}")
        print(f"    MaxDD:   {result.max_dd:.1%}")
        print(f"    CAGR:    {result.cagr:.1%}")
        print(f"    WinRate: {result.win_rate:.0%}")
        print(f"    Trades:  {result.n_trades}")
        
        if result.window_details:
            print(f"\n  📊 Window Details:")
            for w in result.window_details:
                if "sharpe" in w:
                    print(f"    {w['period']}: Sharpe={w['sharpe']:.3f}, MaxDD={w['max_dd']:.1%}, WR={w['win_rate']:.0%}")
                else:
                    print(f"    {w['period']}: {w.get('error', 'N/A')}")
        
        if result.warnings:
            print(f"  ⚠️ Warnings: {result.warnings}")
        
        return result
    
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return None


def analyze_top10_quality(ranks, prices, weights, top_n=10):
    """Analyze the quality of top-10 picks across the backtest period."""
    print(f"\n📊 Top-{top_n} Pick Quality Analysis:")
    
    engine = BacktestEngine(cost=0.001, stop_loss=-0.15)
    
    dates = sorted(ranks.keys())
    all_price_dates = sorted(prices.index.astype(str))
    
    # Sample dates for analysis (every 30 days)
    sample_dates = [d for d in dates if d in all_price_dates][::30]
    
    pick_returns = []
    pick_tickers = []
    pick_dates = []
    
    for date in sample_dates:
        # Get scores
        scores = engine._get_scores(ranks, date, weights)
        if scores is None:
            continue
        
        picks = scores.head(top_n).index.tolist()
        valid_picks = [t for t in picks if t in prices.columns and date in prices.index]
        
        if not valid_picks:
            continue
        
        # Calculate forward 30-day return
        date_idx = list(prices.index).index(date) if date in prices.index else -1
        if date_idx < 0 or date_idx + 30 >= len(prices.index):
            continue
        
        future_date = prices.index[date_idx + 30]
        
        for t in valid_picks:
            p0 = prices.loc[date, t]
            p1 = prices.loc[future_date, t]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                ret = (p1 - p0) / p0
                pick_returns.append(ret)
                pick_tickers.append(t)
                pick_dates.append(date)
    
    if not pick_returns:
        print("  No valid picks found")
        return
    
    rets = np.array(pick_returns)
    print(f"  Total picks analyzed: {len(rets)}")
    print(f"  Mean 30d return: {np.mean(rets):.2%}")
    print(f"  Median 30d return: {np.median(rets):.2%}")
    print(f"  Win rate: {np.mean(rets > 0):.1%}")
    print(f"  Sharpe (daily): {np.mean(rets) / np.std(rets) * np.sqrt(252/30):.3f}" if np.std(rets) > 0 else "  Sharpe: N/A")
    print(f"  Max gain: {np.max(rets):.2%}")
    print(f"  Max loss: {np.min(rets):.2%}")
    
    # Check for small-cap alpha
    r2k_tickers = set(pick_tickers) - set(["A", "AAPL", "ABBV", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP"])  # rough SPX check
    # Better: count how many picks are from each universe
    spx_count = sum(1 for t in pick_tickers if t in prices.columns and len(t) <= 5)  # rough heuristic
    
    return {
        "n_picks": len(rets),
        "mean_return": float(np.mean(rets)),
        "median_return": float(np.median(rets)),
        "win_rate": float(np.mean(rets > 0)),
        "sharpe": float(np.mean(rets) / np.std(rets) * np.sqrt(252/30)) if np.std(rets) > 0 else 0,
        "max_gain": float(np.max(rets)),
        "max_loss": float(np.min(rets)),
        "pick_tickers": pick_tickers,
    }


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 R2K Extension Backtest: SPX 476 vs SPX + Russell 2000")
    print("=" * 80)
    
    # ── Load data ──
    spx_features, spx_prices = load_spx_data()
    r2k_raw = load_r2k_data()
    
    # ── Build R2K data ──
    r2k_indices, r2k_tickers = build_r2k_pit_indices(r2k_raw)
    r2k_prices_raw = build_r2k_price_pivot(r2k_raw)
    
    # ── Get common dates ──
    spx_dates = set(spx_features["date"].unique())
    r2k_dates = set(r2k_prices_raw.index)
    common_dates = sorted(spx_dates & r2k_dates)
    print(f"\n  SPX dates: {len(spx_dates)}")
    print(f"  R2K dates: {len(r2k_dates)}")
    print(f"  Common dates: {len(common_dates)}")
    
    # ── Build ranks ──
    spx_ranks = build_spx_ranks_from_features(spx_features)
    r2k_ranks = compute_r2k_ranks(r2k_raw, r2k_indices, r2k_tickers, common_dates)
    
    # ── Merge for combined backtest ──
    merged_ranks = merge_ranks(spx_ranks, r2k_ranks)
    merged_prices = merge_price_pivots(spx_prices, r2k_prices_raw)
    
    # ── Model weights ──
    # V0.4.4 weights (SPX only, with cashflow)
    v044_weights = {
        "fund_ratio": 0.45,
        "growth_composite": 0.20,
        "qoq": 0.20,
        # cashflow not available for R2K — redistribute
    }
    
    # Adjusted weights (no cashflow, fair comparison)
    # Redistribute cashflow's 15% to fund_ratio (10%) and qoq (5%)
    adj_weights = {
        "fund_ratio": 0.55,  # 0.45 + 0.10
        "growth_composite": 0.20,
        "qoq": 0.25,  # 0.20 + 0.05
    }
    
    # Equal weight baseline
    eq_weights = {
        "fund_ratio": 1.0/3,
        "growth_composite": 1.0/3,
        "qoq": 1.0/3,
    }
    
    # ═══════════════════════════════════════════════════
    # Run backtests
    # ═══════════════════════════════════════════════════
    
    results = {}
    
    # ── Fair comparison: SPX-only on same date range as R2K ──
    print("\n" + "="*80)
    print("  TEST 0: SPX Only (Fair comparison, same date range as R2K)")
    print("="*80)
    # Filter SPX to R2K date range for fair comparison
    r2k_date_range = (min(r2k_prices_raw.index), max(r2k_prices_raw.index))
    spx_prices_fair = spx_prices.loc[(spx_prices.index >= r2k_date_range[0]) & (spx_prices.index <= r2k_date_range[1])]
    spx_ranks_fair = {d: r for d, r in spx_ranks.items() if r2k_date_range[0] <= d <= r2k_date_range[1]}
    results["spx_fair_adj"] = run_walk_forward_backtest(
        spx_ranks_fair, spx_prices_fair, adj_weights,
        f"SPX Only (Fair: {r2k_date_range[0]} to {r2k_date_range[1]})", top_n=10
    )

    # 1. SPX only (V0.4.4 weights — as reference)
    print("\n" + "="*80)
    print("  TEST 1: SPX Only (V0.4.4 weights)")
    print("="*80)
    results["spx_v044"] = run_walk_forward_backtest(
        spx_ranks, spx_prices, v044_weights, 
        "SPX Only (V0.4.4)", top_n=10
    )
    
    # 2. SPX only (adjusted weights — no cashflow)
    print("\n" + "="*80)
    print("  TEST 2: SPX Only (Adjusted weights, no cashflow)")
    print("="*80)
    results["spx_adj"] = run_walk_forward_backtest(
        spx_ranks, spx_prices, adj_weights,
        "SPX Only (Adjusted)", top_n=10
    )
    
    # 3. SPX + R2K combined (adjusted weights)
    print("\n" + "="*80)
    print("  TEST 3: SPX + R2K Combined (Adjusted weights)")
    print("="*80)
    results["spx_r2k_adj"] = run_walk_forward_backtest(
        merged_ranks, merged_prices, adj_weights,
        "SPX+R2K (Adjusted)", top_n=10
    )
    
    # 4. SPX + R2K combined (equal weight baseline)
    print("\n" + "="*80)
    print("  TEST 4: SPX + R2K Combined (Equal weight baseline)")
    print("="*80)
    results["spx_r2k_eq"] = run_walk_forward_backtest(
        merged_ranks, merged_prices, eq_weights,
        "SPX+R2K (Equal Weight)", top_n=10
    )
    
    # 5. SPX + R2K combined (V0.4.4 weights — ignoring cashflow=0)
    print("\n" + "="*80)
    print("  TEST 5: SPX + R2K (V0.4.4 weights, cashflow=0)")
    print("="*80)
    # Note: cashflow factor doesn't exist in merged ranks, so this will just use fund_ratio + growth_composite + qoq
    results["spx_r2k_v044"] = run_walk_forward_backtest(
        merged_ranks, merged_prices, v044_weights,
        "SPX+R2K (V0.4.4, no CF)", top_n=10
    )
    
    # ═══════════════════════════════════════════════════
    # Top-10 Quality Analysis
    # ═══════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("  TOP-10 PICK QUALITY ANALYSIS")
    print("="*80)
    
    spx_quality = analyze_top10_quality(spx_ranks, spx_prices, adj_weights)
    combined_quality = analyze_top10_quality(merged_ranks, merged_prices, adj_weights)
    
    # ═══════════════════════════════════════════════════
    # Summary Comparison Table
    # ═══════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("  📊 FINAL COMPARISON TABLE")
    print("="*80)
    
    header = f"{'Config':<35} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'WR':>8} {'Trades':>8}"
    print(header)
    print("-" * len(header))
    
    for name, r in results.items():
        if r is not None:
            print(f"{name:<35} {r.sharpe:>8.3f} {r.max_dd:>8.1%} {r.cagr:>8.1%} {r.win_rate:>8.0%} {r.n_trades:>8d}")
        else:
            print(f"{name:<35} {'FAILED':>8}")
    
    # ═══════════════════════════════════════════════════
    # Conclusions
    # ═══════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("  📝 CONCLUSIONS")
    print("="*80)
    
    if results.get("spx_adj") and results.get("spx_r2k_adj"):
        spx_result = results["spx_adj"]
        combo_result = results["spx_r2k_adj"]
        assert spx_result is not None and combo_result is not None
        spx_sharpe = spx_result.sharpe
        combo_sharpe = combo_result.sharpe
        spx_dd = spx_result.max_dd
        combo_dd = combo_result.max_dd
        spx_cagr = spx_result.cagr
        combo_cagr = combo_result.cagr
        
        print(f"\n  SPX Only:    Sharpe={spx_sharpe:.3f}, MaxDD={spx_dd:.1%}, CAGR={spx_cagr:.1%}")
        print(f"  SPX+R2K:     Sharpe={combo_sharpe:.3f}, MaxDD={combo_dd:.1%}, CAGR={combo_cagr:.1%}")
        
        if combo_sharpe > spx_sharpe * 1.05:
            print(f"\n  ✅ VERDICT: R2K扩展有益 (Sharpe提升 {(combo_sharpe/spx_sharpe-1)*100:.1f}%)")
        elif combo_sharpe > spx_sharpe * 0.95:
            print(f"\n  ⚠️ VERDICT: R2K扩展中性 (Sharpe变化 {(combo_sharpe/spx_sharpe-1)*100:.1f}%)")
        else:
            print(f"\n  ❌ VERDICT: R2K扩展有害 (Sharpe下降 {(1-combo_sharpe/spx_sharpe)*100:.1f}%)")
        
        if abs(combo_dd) < abs(spx_dd) * 0.9:
            print(f"  ✅ MaxDD改善: {spx_dd:.1%} → {combo_dd:.1%}")
        elif abs(combo_dd) > abs(spx_dd) * 1.1:
            print(f"  ⚠️ MaxDD恶化: {spx_dd:.1%} → {combo_dd:.1%}")
    
    print(f"\n  ⏱️ Total time: {time.time()-t0:.0f}s")
    
    return results


if __name__ == "__main__":
    results = main()
