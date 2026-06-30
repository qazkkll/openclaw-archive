"""T1.4: Merge features, targets, and news features into unified training dataset."""
import pandas as pd
import json
import numpy as np
from pathlib import Path

BASE = Path("/home/hermes/.hermes/openclaw-archive")

print("=== T1.4: Data Merge ===")

# 1. Load data
print("\n1. Loading data...")
features = pd.read_parquet(BASE / "data/falcon/features_v02.parquet")
targets = pd.read_parquet(BASE / "data/falcon/targets_v04.parquet")
news = pd.read_parquet(BASE / "data/falcon/news_features_v04.parquet")

print(f"  Features: {features.shape}")
print(f"  Targets:  {targets.shape}")
print(f"  News:     {news.shape}")

# 2. Merge features + targets on (date, ticker)
print("\n2. Merging features + targets...")
features["date"] = pd.to_datetime(features["date"]).dt.date
targets["date"] = pd.to_datetime(targets["date"]).dt.date

merged = features.merge(targets, on=["date", "ticker"], how="inner")
print(f"  After merge: {merged.shape} (lost {len(features) - len(merged)} rows)")

# 3. Align news features (monthly -> daily)
print("\n3. Aligning news features (monthly -> daily)...")
news["date"] = pd.to_datetime(news["date"]).dt.date
# News date is already month-end dates; forward-fill to daily
# Create a lookup: for each (ticker, month), assign news values to all trading days in that month
merged["year_month"] = pd.to_datetime(merged["date"]).dt.to_period("M")

# Convert news date to year_month for matching
news["year_month"] = pd.to_datetime(news["date"]).dt.to_period("M")
news_cols = [c for c in news.columns if c.startswith("news_")]
news_lookup = news[["ticker", "year_month"] + news_cols].copy()

merged = merged.merge(news_lookup, on=["ticker", "year_month"], how="left")
merged.drop(columns=["year_month"], inplace=True)
print(f"  After news merge: {merged.shape}")

# 4. Ensure correct column order
# Reorder: date, ticker, factors, news, targets
factor_cols = [c for c in features.columns if c not in ["date", "ticker"]]
target_cols = ["fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d", "fwd_ret_30d"]
final_cols = ["date", "ticker"] + factor_cols + news_cols + target_cols
merged = merged[final_cols]

print(f"\n4. Final shape: {merged.shape}")
print(f"   Columns: {list(merged.columns[:3])} ... {list(merged.columns[-7:])}")

# 5. Data quality checks
print("\n5. Data quality checks...")
merged["year"] = pd.to_datetime(merged["date"]).dt.year

quality_report = {}
all_pass = True

for year in sorted(merged["year"].unique()):
    yr_df = merged[merged["year"] == year]
    total_tickers = yr_df["ticker"].nunique()
    # Coverage: % of expected trading days (assume ~252)
    unique_dates = yr_df["date"].nunique()
    coverage_pct = unique_dates / 252 * 100
    
    yr_report = {
        "tickers": int(total_tickers),
        "trading_days": int(unique_dates),
        "coverage_pct": round(coverage_pct, 1),
        "rows": int(len(yr_df)),
    }
    
    # Check max date gap per ticker
    max_gap = 0
    for t in yr_df["ticker"].unique():
        t_df = yr_df[yr_df["ticker"] == t].sort_values("date")
        if len(t_df) > 1:
            gaps = pd.to_datetime(t_df["date"]).diff().dt.days.max()
            if pd.notna(gaps) and gaps > max_gap:
                max_gap = gaps
    yr_report["max_date_gap_days"] = int(max_gap)
    
    fail = False
    if coverage_pct < 80:
        yr_report["FAIL_coverage"] = True
        all_pass = False
        fail = True
    if total_tickers < 400:
        yr_report["FAIL_tickers"] = True
        all_pass = False
        fail = True
    if max_gap > 5:
        yr_report["FAIL_gap"] = True
        all_pass = False
        fail = True
    
    status = "✓" if not fail else "✗"
    print(f"  {year}: {status} tickers={total_tickers}, days={unique_dates}, coverage={coverage_pct:.1f}%, max_gap={max_gap}d")
    quality_report[str(year)] = yr_report

merged.drop(columns=["year"], inplace=True)

# Overall stats
overall = {
    "total_rows": int(len(merged)),
    "total_tickers": int(merged["ticker"].nunique()),
    "date_range": [str(merged["date"].min()), str(merged["date"].max())],
    "n_factors": len(factor_cols),
    "n_news_features": len(news_cols),
    "n_targets": len(target_cols),
    "all_checks_pass": all_pass,
    "by_year": quality_report,
}

# 6. Save
print("\n6. Saving training data...")
out_path = BASE / "data/falcon/training_data_v04.parquet"
merged.to_parquet(out_path, index=False)
print(f"  Saved: {out_path} ({merged.shape})")

# 7. Save quality report
print("\n7. Saving quality report...")
with open(BASE / "data/falcon/v04_data_quality.json", "w") as f:
    json.dump(overall, f, indent=2, default=str)
print(f"  Saved: data/falcon/v04_data_quality.json")

print(f"\n=== DONE ===")
print(f"Final dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
print(f"Date range: {merged['date'].min()} ~ {merged['date'].max()}")
print(f"Tickers: {merged['ticker'].nunique()}")
print(f"Quality: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
