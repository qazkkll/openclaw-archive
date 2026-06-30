#!/usr/bin/env python3
"""
T1.1 Factor Audit: Complete audit of all factors in features_v02.parquet
Outputs: data/falcon/v04_factor_audit.json
"""
import pandas as pd
import numpy as np
import json
from scipy import stats
from pathlib import Path

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
FEATURES_PATH = WORKSPACE / "data/falcon/features_v02.parquet"
PRICES_PATH = WORKSPACE / "data/falcon/us_prices_daily.parquet"
OUTPUT_PATH = WORKSPACE / "data/falcon/v04_factor_audit.json"

# ─── Factor Classification ───────────────────────────────────────────
# Technical factors: derived from price/volume at observation time → PIT-safe
TECHNICAL_PATTERNS = [
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align',
    'ma_cross_5_20', 'ma_cross_20_60', 'price_position',
    'ret1', 'ret5', 'ret10', 'ret20', 'ret30', 'ret60', 'ret90',
    'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
    'vol20', 'vol5', 'vol_ratio', 'vol_change', 'vol_regime',
    'rsi14', 'rsi_change', 'rsi_zone',
    'macd', 'macd_signal', 'macd_hist', 'macd_roc',
    'bb_std', 'bb_width', 'bb_pos',
    'ret_quality', 'range_ratio', 'avg_body', 'vwap_drift',
    'dd_60', 'ud_vol_ratio', 'beta',
]

# Fundamental factors from FMP ratios: NOT PIT-safe (have reporting lag)
FUNDAMENTAL_FMP_PATTERNS = [
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin',
    'ebitdaMargin', 'assetTurnover', 'inventoryTurnover',
    'receivablesTurnover', 'debtToEquityRatio', 'currentRatio',
    'quickRatio', 'financialLeverageRatio',
    'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
    'dividendYieldPercentage', 'dividendPayoutRatio',
]

# Quarter-over-quarter changes: NOT PIT-safe (derived from fundamental data)
FUNDAMENTAL_QOQ_PATTERNS = [
    'grossProfitMargin_qoq', 'netProfitMargin_qoq',
    'operatingProfitMargin_qoq', 'ebitdaMargin_qoq',
]

# Analyst factors: PIT-safe (forward-looking estimates available at observation time)
ANALYST_PATTERNS = [
    'eps_revision', 'revenue_revision',
    'num_analysts_eps', 'num_analysts_rev', 'eps_dispersion',
]

# Metadata (not factors, but in the parquet)
META_PATTERNS = ['ticker', 'date', 'fmp_covered', 'analyst_covered']


def classify_factor(col):
    """Classify a factor into type and PIT safety."""
    if col in META_PATTERNS:
        return ('meta', False, 'metadata')
    if col in TECHNICAL_PATTERNS:
        return ('technical', True, 'price/volume derived')
    if col in FUNDAMENTAL_FMP_PATTERNS:
        return ('fundamental', False, 'FMP ratios - reporting lag')
    if col in FUNDAMENTAL_QOQ_PATTERNS:
        return ('fundamental', False, 'QoQ change - reporting lag')
    if col in ANALYST_PATTERNS:
        return ('analyst', True, 'forward-looking estimates')
    # Default: classify by name heuristics
    col_lower = col.lower()
    if any(k in col_lower for k in ['ret', 'mom', 'vol', 'rsi', 'macd', 'bb_', 'ma_', 'price', 'vwap']):
        return ('technical', True, 'price-derived heuristic')
    return ('unknown', False, 'unclassified')


def compute_coverage_by_year(df, factor_cols):
    """Compute per-year non-NaN coverage for each factor."""
    df['year'] = pd.to_datetime(df['date']).dt.year
    coverage = {}
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        cov = {}
        for col in factor_cols:
            cov[col] = float(year_df[col].notna().mean())
        coverage[int(year)] = cov
    df.drop(columns=['year'], inplace=True)
    return coverage


def compute_latest_dates(df, factor_cols):
    """For each factor, find the latest date where it has data (per ticker)."""
    df_dated = df.copy()
    df_dated['_date'] = pd.to_datetime(df_dated['date'])
    latest = {}
    for col in factor_cols:
        mask = df_dated[col].notna()
        if mask.any():
            latest[col] = str(df_dated.loc[mask, '_date'].max().date())
        else:
            latest[col] = None
    return latest


def compute_forward_returns(df, prices_df, hold_days=30):
    """Compute 30-day forward returns for IC calculation."""
    prices = prices_df.copy()
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['ticker', 'date'])
    
    # Forward returns: close[t+30] / close[t] - 1
    prices['fwd_ret'] = prices.groupby('ticker')['close'].transform(
        lambda x: x.shift(-hold_days) / x - 1
    )
    return prices[['ticker', 'date', 'fwd_ret']]


def compute_ic(df, factor_cols, fwd_rets):
    """Compute Spearman IC (mean, std, ICIR) for each factor vs 30-day fwd return."""
    # Merge forward returns
    df_merged = df.copy()
    df_merged['date'] = pd.to_datetime(df_merged['date'])
    fwd_rets['date'] = pd.to_datetime(fwd_rets['date'])
    df_merged = df_merged.merge(fwd_rets, on=['ticker', 'date'], how='left')
    
    ic_results = {}
    for col in factor_cols:
        # Group by date, compute cross-sectional Spearman rank correlation
        daily_ics = []
        for date_val, group in df_merged.groupby('date'):
            valid = group[[col, 'fwd_ret']].dropna()
            if len(valid) < 30:  # need minimum stocks
                continue
            corr, _ = stats.spearmanr(valid[col], valid['fwd_ret'])
            if np.isfinite(corr):
                daily_ics.append(corr)
        
        if daily_ics:
            ic_mean = float(np.mean(daily_ics))
            ic_std = float(np.std(daily_ics))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            ic_results[col] = {
                'ic_mean': round(ic_mean, 6),
                'ic_std': round(ic_std, 6),
                'icir': round(icir, 4),
                'n_days': len(daily_ics),
            }
        else:
            ic_results[col] = {'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'n_days': 0}
    
    return ic_results


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Shape: {df.shape}, Date range: {df['date'].min()} - {df['date'].max()}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    
    # Identify factor columns (exclude meta)
    all_cols = list(df.columns)
    factor_cols = [c for c in all_cols if c not in ['ticker', 'date']]
    print(f"  Total columns: {len(all_cols)}, Factor columns: {len(factor_cols)}")
    
    # Classify factors
    classifications = {}
    for col in factor_cols:
        ftype, pit_safe, source = classify_factor(col)
        classifications[col] = {'type': ftype, 'pit_safe': pit_safe, 'source': source}
    
    print("\nClassification summary:")
    type_counts = {}
    for col, info in classifications.items():
        t = info['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    
    # Coverage by year
    print("\nComputing coverage by year...")
    coverage = compute_coverage_by_year(df, factor_cols)
    years = sorted(coverage.keys())
    print(f"  Years: {years}")
    
    # Find low coverage factors (<80% in any year)
    low_coverage = {}
    for col in factor_cols:
        low_years = [y for y in years if coverage[y].get(col, 0) < 0.8]
        if low_years:
            low_coverage[col] = low_years
    
    print(f"\nLow coverage factors (<80% in any year): {len(low_coverage)}")
    for col, yrs in sorted(low_coverage.items()):
        min_cov = min(coverage[y][col] for y in yrs)
        print(f"  {col}: {len(yrs)} years below 80%, min={min_cov:.1%} in {yrs}")
    
    # Latest dates
    print("\nComputing latest dates...")
    latest_dates = compute_latest_dates(df, factor_cols)
    
    # Forward returns and IC
    print("\nLoading prices for forward returns...")
    prices = pd.read_parquet(PRICES_PATH)
    print(f"  Prices shape: {prices.shape}")
    
    print("Computing 30-day forward returns...")
    fwd_rets = compute_forward_returns(df, prices, hold_days=30)
    n_valid = fwd_rets['fwd_ret'].notna().sum()
    print(f"  Valid forward returns: {n_valid}/{len(fwd_rets)}")
    
    print("Computing IC for all factors...")
    ic_results = compute_ic(df, factor_cols, fwd_rets)
    
    # Build output
    factors_output = []
    for col in factor_cols:
        info = classifications[col]
        ftype = info['type']
        pit_safe = info['pit_safe']
        source = info['source']
        cov_by_year = {str(y): round(coverage[y].get(col, 0), 4) for y in years}
        ic = ic_results.get(col, {'ic_mean': 0, 'ic_std': 0, 'icir': 0, 'n_days': 0})
        
        factors_output.append({
            'name': col,
            'type': ftype,
            'source': source,
            'coverage_by_year': cov_by_year,
            'latest_date': latest_dates.get(col),
            'pit_safe': pit_safe,
            'ic_mean': ic['ic_mean'],
            'ic_std': ic['ic_std'],
            'icir': ic['icir'],
            'ic_n_days': ic['n_days'],
        })
    
    # Summary
    total = len(factor_cols)
    technical = sum(1 for c in factor_cols if classifications[c]['type'] == 'technical')
    fundamental = sum(1 for c in factor_cols if classifications[c]['type'] == 'fundamental')
    analyst = sum(1 for c in factor_cols if classifications[c]['type'] == 'analyst')
    pit_safe_count = sum(1 for c in factor_cols if classifications[c]['pit_safe'])
    low_cov_count = len(low_coverage)
    
    # High IC factors (|ICIR| > 0.1)
    high_ic = [c for c in factor_cols if abs(ic_results.get(c, {}).get('icir', 0)) > 0.1]
    
    summary = {
        'total': total,
        'technical': technical,
        'fundamental': fundamental,
        'analyst': analyst,
        'pit_safe_count': pit_safe_count,
        'low_coverage_count': low_cov_count,
        'high_ic_count': len(high_ic),
        'high_ic_factors': high_ic,
        'date_range': {
            'min': str(df['date'].min().date()),
            'max': str(df['date'].max().date()),
        },
        'n_tickers': int(df['ticker'].nunique()),
        'n_rows': len(df),
    }
    
    output = {
        'factors': factors_output,
        'summary': summary,
    }
    
    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Output written to {OUTPUT_PATH}")
    print(f"   Total factors: {total}")
    print(f"   Technical: {technical}, Fundamental: {fundamental}, Analyst: {analyst}")
    print(f"   PIT-safe: {pit_safe_count}/{total}")
    print(f"   Low coverage (<80%): {low_cov_count}")
    print(f"   High IC (|ICIR|>0.1): {len(high_ic)}")
    
    return output


if __name__ == '__main__':
    main()
