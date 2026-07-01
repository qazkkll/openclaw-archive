#!/usr/bin/env python3
"""
🦅 Falcon V0.4.0 True Rank Inversion Test
================================================================
Test the TRUE rank inversion (Top5% vs Bottom20%) for V0.4.0 configuration.
Previously V0.4.0 only tested "Sharpe degradation stability" which is NOT
the standard rank inversion test.

V0.4.0 Config:
  - features: features_v02.parquet (80 columns)
  - weights: fund_ratio=0.70, fund_metric=0.15, log_metric=0.15
  - train_years=0.5, test_months=6, hold_days=30, top_n=10
  - cost=0.001, stop_loss=-0.15

Tests:
  1. True rank inversion: Top5% avg_return vs Bottom20% avg_return per window
  2. Factor IC/ICIR analysis for V0.4.0 factors
  3. Factor stability analysis
  4. Cross-factor correlation analysis
  5. Comparison with V0.4.1 results

Output:
  - data/falcon/v040_rank_inversion_results.json
  - data/falcon/rank_inversion_root_cause_analysis.json
"""
import sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

# ═══════════════════════════════════════════════════
#  Paths
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v02.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_RI = DATA_DIR / "v040_rank_inversion_results.json"
OUTPUT_ROOT_CAUSE = DATA_DIR / "rank_inversion_root_cause_analysis.json"

# ═══════════════════════════════════════════════════
#  V0.4.0 Factor Group Definitions (from v04_final_refined_v5.py)
# ═══════════════════════════════════════════════════
FACTOR_GROUPS = {
    'fund_ratio': [
        'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
        'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
        'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin',
        'ebitdaMargin', 'assetTurnover', 'inventoryTurnover',
        'receivablesTurnover', 'debtToEquityRatio', 'currentRatio',
        'quickRatio', 'financialLeverageRatio',
        'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
        'dividendYieldPercentage', 'dividendPayoutRatio',
    ],
    'fund_metric': [
        'earningsYield', 'evToEBITDA', 'evToFreeCashFlow', 'evToSales',
        'freeCashFlowYield', 'returnOnEquity', 'returnOnAssets',
        'returnOnCapitalEmployed', 'returnOnInvestedCapital',
        'returnOnTangibleAssets', 'incomeQuality', 'grahamNumber',
        'cashConversionCycle', 'capexToRevenue', 'capexToDepreciation',
        'researchAndDevelopementToRevenue', 'stockBasedCompensationToRevenue',
        'netDebtToEBITDA', 'operatingReturnOnAssets',
    ],
}

# Flip factors (higher = worse → invert rank)
FLIP_FACTORS = {
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'debtToEquityRatio', 'financialLeverageRatio', 'inventoryTurnover',
    'netDebtToEBITDA', 'capexToRevenue', 'capexToDepreciation',
    'researchAndDevelopementToRevenue', 'stockBasedCompensationToRevenue',
    'cashConversionCycle',
}

# V0.4.0 weights
V040_WEIGHTS = {
    'fund_ratio': 0.70,
    'fund_metric': 0.15,
    'log_metric': 0.15,
}


# ═══════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════
def load_data():
    """Load features_v02.parquet and prices."""
    print("📂 Loading data...")
    t0 = time.time()
    
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    print(f"  ✅ Features: {df.shape[0]} rows × {df.shape[1]} cols, {df['ticker'].nunique()} tickers")
    
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {price_pivot.shape[0]} days × {price_pivot.shape[1]} tickers")
    print(f"  ⏱️ Load time: {time.time()-t0:.1f}s")
    return df, price_pivot


# ═══════════════════════════════════════════════════
#  Cross-sectional Percentile Ranking
# ═══════════════════════════════════════════════════
def compute_cross_sectional_ranks(df, factor_cols):
    """Compute cross-sectional percentile ranks for each date."""
    print("📊 Computing cross-sectional percentile ranks...")
    t0 = time.time()
    
    from scipy.stats import rankdata
    
    dates = sorted(df['date'].unique())
    ranks = {}
    
    for date in dates:
        day_df = df[df['date'] == date].copy()
        if len(day_df) < 10:
            continue
        
        tickers = day_df['ticker'].values
        rank_df = pd.DataFrame(index=tickers)
        
        for col in factor_cols:
            if col not in day_df.columns:
                continue
            vals = day_df[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue
            
            ranks_raw = np.full_like(vals, np.nan)
            if valid.sum() > 0:
                ranks_raw[valid] = rankdata(vals[valid], method='average') / valid.sum()
            
            if col in FLIP_FACTORS:
                mask = ~np.isnan(ranks_raw)
                ranks_raw[mask] = 1.0 - ranks_raw[mask]
            
            rank_df[col] = ranks_raw
        
        ranks[date] = rank_df
    
    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)} days ranked ({elapsed:.0f}s)")
    return ranks


def compute_group_ranks(ranks, factor_groups):
    """Merge factor-level ranks into group-level ranks (equal-weighted mean)."""
    print("📊 Computing group ranks...")
    for date in list(ranks.keys()):
        df = ranks[date]
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                df[group_name] = df[available].mean(axis=1)
        ranks[date] = df
    print(f"  ✅ Group ranks added: {list(factor_groups.keys())}")
    return ranks


def add_combo_factors(ranks):
    """Add combo factors (log_metric) for each date."""
    for date in ranks:
        df = ranks[date]
        if 'fund_metric' in df.columns:
            df['log_metric'] = np.log(df['fund_metric'] + 1)
        ranks[date] = df
    print("  ✅ Combo factor 'log_metric' added")
    return ranks


# ═══════════════════════════════════════════════════
#  TRUE Rank Inversion Test (Top5% vs Bottom20%)
# ═══════════════════════════════════════════════════
def compute_true_rank_inversion(ranks, prices, weights, 
                                 train_months=6, test_months=6,
                                 hold_days=30):
    """
    TRUE rank inversion: Top5% vs Bottom20% forward return per WF window.
    
    For each Walk-Forward window:
      1. On each test date, rank stocks by combined score
      2. Take Top5% and Bottom20% 
      3. Compute their forward returns (hold_days)
      4. Check if Top5% avg_return > Bottom20% avg_return
    """
    print("\n🔍 Running TRUE Rank Inversion Test (Top5% vs Bottom20%)...")
    t0 = time.time()
    
    dates = sorted(ranks.keys())
    if not dates:
        return None
    
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    
    windows = []
    window_idx = 0
    
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
        
        test_start_str = str(train_end)[:10]
        test_end_str = str(test_end)[:10]
        
        # Get test dates
        test_dates = [d for d in sorted(prices.index.astype(str))
                      if test_start_str <= d <= test_end_str]
        
        if len(test_dates) < 10:
            window_idx += 1
            train_start += pd.DateOffset(months=test_months)
            continue
        
        # Compute forward returns for each date
        top5_returns = []
        bottom20_returns = []
        per_date_details = []
        
        for i, date in enumerate(test_dates):
            if date not in ranks:
                continue
            
            r = ranks[date]
            available = [f for f in weights if f in r.columns and weights[f] > 0]
            if not available:
                continue
            
            # Combined score
            combined = pd.Series(0.0, index=r.index)
            for f in available:
                combined = combined + weights[f] * r[f]
            scores = combined.dropna().sort_values(ascending=False)
            
            if len(scores) < 20:
                continue
            
            # Forward return (hold_days)
            if i + hold_days >= len(test_dates):
                continue
            future_date = test_dates[min(i + hold_days, len(test_dates) - 1)]
            
            if date not in prices.index or future_date not in prices.index:
                continue
            
            pr_today = prices.loc[date]
            pr_future = prices.loc[future_date]
            
            # Per-stock returns
            returns = {}
            for ticker in scores.index:
                if ticker in pr_today.index and ticker in pr_future.index:
                    if pd.notna(pr_today[ticker]) and pd.notna(pr_future[ticker]) and pr_today[ticker] > 0:
                        returns[ticker] = (pr_future[ticker] / pr_today[ticker]) - 1
            
            if len(returns) < 20:
                continue
            
            returns_series = pd.Series(returns)
            
            # Sort by score (descending)
            sorted_tickers = scores.sort_values(ascending=False).index
            sorted_returns = returns_series.reindex(sorted_tickers).dropna()
            
            if len(sorted_returns) < 20:
                continue
            
            # Top 5% and Bottom 20%
            n_top5 = max(1, int(len(sorted_returns) * 0.05))
            n_bottom20 = max(1, int(len(sorted_returns) * 0.20))
            
            top5_ret = sorted_returns.head(n_top5).mean()
            bottom20_ret = sorted_returns.tail(n_bottom20).mean()
            
            top5_returns.append(top5_ret)
            bottom20_returns.append(bottom20_ret)
            
            per_date_details.append({
                'date': date,
                'top5_return': float(top5_ret),
                'bottom20_return': float(bottom20_ret),
                'spread': float(top5_ret - bottom20_ret),
                'n_stocks': len(sorted_returns),
            })
        
        # Window-level aggregation
        if len(top5_returns) > 0:
            avg_top5 = np.mean(top5_returns)
            avg_bottom20 = np.mean(bottom20_returns)
            ri_passed = avg_top5 > avg_bottom20
            
            # Also compute median spread for robustness
            spreads = [t - b for t, b in zip(top5_returns, bottom20_returns)]
            median_spread = np.median(spreads)
            
            windows.append({
                'window_idx': window_idx,
                'period': f"{test_start_str} → {test_end_str}",
                'avg_top5_return': float(avg_top5),
                'avg_bottom20_return': float(avg_bottom20),
                'spread': float(avg_top5 - avg_bottom20),
                'median_spread': float(median_spread),
                'passed': bool(ri_passed),
                'n_dates': len(top5_returns),
                'positive_spread_pct': float(np.mean([s > 0 for s in spreads])),
            })
        
        window_idx += 1
        train_start += pd.DateOffset(months=test_months)
    
    elapsed = time.time() - t0
    print(f"  ✅ Rank inversion test completed ({elapsed:.0f}s, {len(windows)} windows)")
    
    if not windows:
        return None
    
    # Summary
    passed_windows = sum(1 for w in windows if w['passed'])
    total_windows = len(windows)
    
    return {
        'windows': windows,
        'total_windows': total_windows,
        'passed_windows': passed_windows,
        'pass_rate': passed_windows / total_windows if total_windows > 0 else 0,
        'overall_passed': passed_windows / total_windows > 0.6 if total_windows > 0 else False,
        'avg_spread': float(np.mean([w['spread'] for w in windows])),
        'median_spread': float(np.median([w['spread'] for w in windows])),
        'avg_positive_spread_pct': float(np.mean([w['positive_spread_pct'] for w in windows])),
    }


# ═══════════════════════════════════════════════════
#  Factor IC/ICIR Analysis
# ═══════════════════════════════════════════════════
def compute_factor_ic_analysis(df, factor_cols, price_pivot, hold_days=30):
    """
    Compute IC (Information Coefficient) and ICIR for each factor.
    IC = Spearman rank correlation between factor and forward return.
    ICIR = mean(IC) / std(IC)
    
    Uses price_pivot to compute forward returns (not dependent on fwd_ret columns).
    """
    print("\n📊 Computing factor IC/ICIR analysis...")
    t0 = time.time()
    
    from scipy.stats import spearmanr
    
    dates = sorted(df['date'].unique())
    price_dates = sorted(price_pivot.index.astype(str))
    
    # Pre-compute forward returns from price data
    print("  📊 Pre-computing forward returns from price data...")
    fwd_ret_map = {}  # date -> {ticker -> fwd_return}
    for i, date in enumerate(dates):
        if i + hold_days >= len(dates):
            continue
        future_date = dates[min(i + hold_days, len(dates) - 1)]
        
        if date in price_pivot.index and future_date in price_pivot.index:
            pr_today = price_pivot.loc[date]
            pr_future = price_pivot.loc[future_date]
            valid = pr_today.notna() & pr_future.notna() & (pr_today > 0)
            ret = ((pr_future[valid] / pr_today[valid]) - 1).to_dict()
            fwd_ret_map[date] = ret
    
    print(f"  ✅ Forward returns computed for {len(fwd_ret_map)} dates")
    
    # Compute IC per date per factor
    factor_ics = {col: [] for col in factor_cols}
    
    for date in dates:
        if date not in fwd_ret_map:
            continue
        
        fwd = fwd_ret_map[date]
        day_df = df[df['date'] == date]
        if len(day_df) < 20:
            continue
        
        # Map tickers to forward returns
        ticker_fwd = day_df['ticker'].map(fwd).values.astype(float)
        valid_fwd = ~np.isnan(ticker_fwd)
        
        if valid_fwd.sum() < 20:
            continue
        
        for col in factor_cols:
            if col not in day_df.columns:
                factor_ics[col].append(np.nan)
                continue
            
            vals = day_df[col].values.astype(float)
            valid = (~np.isnan(vals)) & valid_fwd
            if valid.sum() < 20:
                factor_ics[col].append(np.nan)
                continue
            
            ic, _ = spearmanr(vals[valid], ticker_fwd[valid])
            factor_ics[col].append(ic)
    
    # Compute ICIR
    results = []
    for col in factor_cols:
        ics = factor_ics[col]
        valid_ics = [x for x in ics if not np.isnan(x)]
        if len(valid_ics) < 30:
            continue
        
        ic_mean = np.mean(valid_ics)
        ic_std = np.std(valid_ics)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        t_stat = ic_mean / (ic_std / np.sqrt(len(valid_ics))) if ic_std > 0 else 0
        
        results.append({
            'name': col,
            'ic_mean': round(float(ic_mean), 6),
            'ic_std': round(float(ic_std), 6),
            'icir': round(float(icir), 4),
            't_stat': round(float(t_stat), 2),
            'n_dates': len(valid_ics),
            'abs_icir': round(float(abs(icir)), 4),
        })
    
    # Sort by abs_icir descending
    results.sort(key=lambda x: x['abs_icir'], reverse=True)
    
    elapsed = time.time() - t0
    print(f"  ✅ IC analysis completed for {len(results)} factors ({elapsed:.0f}s)")
    return results


# ═══════════════════════════════════════════════════
#  Factor Stability Analysis
# ═══════════════════════════════════════════════════
def compute_factor_stability(df, factor_cols, n_periods=4):
    """
    Compute factor stability: split the data into n_periods and check
    if factor IC is consistent across periods.
    """
    print("\n📊 Computing factor stability...")
    t0 = time.time()
    
    dates = sorted(df['date'].unique())
    period_size = len(dates) // n_periods
    
    stability_results = []
    
    for col in factor_cols:
        if col not in df.columns:
            continue
        
        period_ics = []
        for p in range(n_periods):
            start_idx = p * period_size
            end_idx = min((p + 1) * period_size, len(dates))
            period_dates = dates[start_idx:end_idx]
            
            period_df = df[df['date'].isin(period_dates)]
            
            # Compute mean rank for this factor in this period
            mean_rank = period_df[col].rank(pct=True).mean()
            std_rank = period_df[col].rank(pct=True).std()
            
            period_ics.append({
                'period': p,
                'mean_rank_pct': float(mean_rank) if not np.isnan(mean_rank) else None,
                'std_rank_pct': float(std_rank) if not np.isnan(std_rank) else None,
            })
        
        # Stability: lower std of mean_ranks across periods = more stable
        valid_periods = [p for p in period_ics if p['mean_rank_pct'] is not None]
        if len(valid_periods) >= 2:
            mean_ranks = [p['mean_rank_pct'] for p in valid_periods]
            stability_score = 1.0 - np.std(mean_ranks)  # Higher = more stable
        else:
            stability_score = None
        
        stability_results.append({
            'name': col,
            'stability_score': round(float(stability_score), 4) if stability_score is not None else None,
            'periods': period_ics,
        })
    
    stability_results.sort(key=lambda x: x.get('stability_score') or 0, reverse=True)
    
    elapsed = time.time() - t0
    print(f"  ✅ Stability analysis completed ({elapsed:.0f}s)")
    return stability_results


# ═══════════════════════════════════════════════════
#  Cross-factor Correlation Analysis
# ═══════════════════════════════════════════════════
def compute_factor_correlations(df, factor_groups):
    """
    Compute correlation between factor groups and identify
    which factors are redundant or conflicting.
    """
    print("\n📊 Computing factor correlations...")
    t0 = time.time()
    
    # Sample dates for efficiency
    dates = sorted(df['date'].unique())
    sample_dates = dates[::max(1, len(dates) // 50)]
    
    # Collect all factor columns
    all_factors = []
    for group, cols in factor_groups.items():
        for col in cols:
            if col in df.columns and col not in all_factors:
                all_factors.append(col)
    
    # Compute average cross-sectional correlation
    corr_matrices = []
    for date in sample_dates:
        day_df = df[df['date'] == date]
        if len(day_df) < 50:
            continue
        available = [f for f in all_factors if f in day_df.columns]
        if len(available) < 2:
            continue
        corr = day_df[available].corr(method='spearman')
        corr_matrices.append(corr)
    
    if not corr_matrices:
        return {}
    
    # Average correlation matrix
    avg_corr = sum(corr_matrices) / len(corr_matrices)
    
    # Group-level correlations
    group_names = list(factor_groups.keys())
    group_corr = pd.DataFrame(index=group_names, columns=group_names)
    
    for g1 in group_names:
        for g2 in group_names:
            cols1 = [c for c in factor_groups[g1] if c in avg_corr.columns]
            cols2 = [c for c in factor_groups[g2] if c in avg_corr.columns]
            if cols1 and cols2:
                sub_corr = avg_corr.loc[cols1, cols2]
                group_corr.loc[g1, g2] = float(sub_corr.values.mean())
    
    # Convert to dict
    group_corr_dict = {}
    for g1 in group_names:
        group_corr_dict[g1] = {}
        for g2 in group_names:
            val = group_corr.loc[g1, g2]
            group_corr_dict[g1][g2] = round(float(val), 4) if pd.notna(val) else None
    
    # Identify highly correlated factor pairs (>0.7)
    high_corr_pairs = []
    for i, f1 in enumerate(all_factors):
        for f2 in all_factors[i+1:]:
            if f1 in avg_corr.columns and f2 in avg_corr.columns:
                corr_val = avg_corr.loc[f1, f2]
                if pd.notna(corr_val) and abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'factor1': f1,
                        'factor2': f2,
                        'correlation': round(float(corr_val), 4),
                    })
    
    elapsed = time.time() - t0
    print(f"  ✅ Correlation analysis completed ({elapsed:.0f}s)")
    
    return {
        'group_correlations': group_corr_dict,
        'high_corr_pairs': sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:20],
        'n_factors_analyzed': len(all_factors),
        'n_dates_used': len(corr_matrices),
    }


# ═══════════════════════════════════════════════════
#  Root Cause Analysis
# ═══════════════════════════════════════════════════
def analyze_root_cause(ri_results, ic_results, stability_results, corr_results,
                       v041_ri_path=None):
    """
    Analyze root causes of rank inversion:
    1. Factor IC quality (weak IC → poor ranking)
    2. Factor stability (unstable → inconsistent ranking)
    3. Factor redundancy (high corr → diluted signal)
    4. V0.4.0 vs V0.4.1 comparison
    """
    print("\n🔍 Analyzing root causes...")
    
    analysis = {
        'summary': {},
        'factor_quality': {},
        'stability_assessment': {},
        'correlation_assessment': {},
        'v040_vs_v041_comparison': {},
        'root_causes': [],
        'recommendations': [],
    }
    
    # 1. Factor Quality (IC/ICIR)
    if ic_results:
        # Group IC analysis
        fund_ratio_factors = [r for r in ic_results if r['name'] in FACTOR_GROUPS.get('fund_ratio', [])]
        fund_metric_factors = [r for r in ic_results if r['name'] in FACTOR_GROUPS.get('fund_metric', [])]
        
        avg_fund_ratio_icir = np.mean([r['abs_icir'] for r in fund_ratio_factors]) if fund_ratio_factors else 0
        avg_fund_metric_icir = np.mean([r['abs_icir'] for r in fund_metric_factors]) if fund_metric_factors else 0
        
        n_strong = sum(1 for r in ic_results if r['abs_icir'] > 0.1)
        n_weak = sum(1 for r in ic_results if r['abs_icir'] < 0.05)
        n_negative = sum(1 for r in ic_results if r['icir'] < 0)
        
        analysis['factor_quality'] = {
            'avg_fund_ratio_abs_icir': round(float(avg_fund_ratio_icir), 4),
            'avg_fund_metric_abs_icir': round(float(avg_fund_metric_icir), 4),
            'n_strong_factors': n_strong,  # |ICIR| > 0.1
            'n_weak_factors': n_weak,      # |ICIR| < 0.05
            'n_negative_icir': n_negative,
            'total_factors': len(ic_results),
            'top5_factors': [
                {'name': r['name'], 'icir': r['icir'], 'abs_icir': r['abs_icir']}
                for r in ic_results[:5]
            ],
            'bottom5_factors': [
                {'name': r['name'], 'icir': r['icir'], 'abs_icir': r['abs_icir']}
                for r in ic_results[-5:]
            ],
        }
        
        # Root cause: weak IC
        if avg_fund_ratio_icir < 0.1:
            analysis['root_causes'].append({
                'cause': 'WEAK_FUND_RATIO_IC',
                'detail': f'fund_ratio average |ICIR|={avg_fund_ratio_icir:.4f} < 0.1, '
                          f'meaning the 20 fundamental ratios have weak predictive power.',
                'severity': 'HIGH',
            })
        
        if avg_fund_metric_icir < 0.1:
            analysis['root_causes'].append({
                'cause': 'WEAK_FUND_METRIC_IC',
                'detail': f'fund_metric average |ICIR|={avg_fund_metric_icir:.4f} < 0.1, '
                          f'meaning the 19 quality metrics have weak predictive power.',
                'severity': 'HIGH',
            })
    
    # 2. Stability Assessment
    if stability_results:
        unstable_factors = [s for s in stability_results 
                          if s['stability_score'] is not None and s['stability_score'] < 0.8]
        analysis['stability_assessment'] = {
            'n_unstable_factors': len(unstable_factors),
            'total_factors': len(stability_results),
            'avg_stability': round(float(np.mean([s['stability_score'] for s in stability_results 
                                                 if s['stability_score'] is not None])), 4),
            'unstable_factor_names': [s['name'] for s in unstable_factors[:10]],
        }
        
        if len(unstable_factors) > len(stability_results) * 0.3:
            analysis['root_causes'].append({
                'cause': 'FACTOR_INSTABILITY',
                'detail': f'{len(unstable_factors)}/{len(stability_results)} factors have '
                          f'stability < 0.8, indicating inconsistent ranking across periods.',
                'severity': 'MEDIUM',
            })
    
    # 3. Correlation Assessment
    if corr_results:
        n_high_corr = len(corr_results.get('high_corr_pairs', []))
        analysis['correlation_assessment'] = {
            'n_high_corr_pairs': n_high_corr,
            'group_correlations': corr_results.get('group_correlations', {}),
        }
        
        if n_high_corr > 10:
            analysis['root_causes'].append({
                'cause': 'FACTOR_REDUNDANCY',
                'detail': f'{n_high_corr} factor pairs have correlation > 0.7, '
                          f'signaling redundancy that dilutes the composite signal.',
                'severity': 'MEDIUM',
            })
    
    # 4. V0.4.0 vs V0.4.1 Comparison
    v041_data = None
    if v041_ri_path and Path(v041_ri_path).exists():
        with open(v041_ri_path) as f:
            v041_data = json.load(f)
    
    if v041_data and ri_results:
        v041_ri = v041_data.get('rank_inversion_test', {})
        v040_ri = ri_results
        
        comparison = {
            'v040': {
                'pass_rate': v040_ri.get('pass_rate', 0),
                'passed_windows': v040_ri.get('passed_windows', 0),
                'total_windows': v040_ri.get('total_windows', 0),
                'avg_spread': v040_ri.get('avg_spread', 0),
                'overall_passed': v040_ri.get('overall_passed', False),
                'config': 'fund_ratio=0.70, fund_metric=0.15, log_metric=0.15',
                'features': 'features_v02.parquet (80 cols)',
            },
            'v041': {
                'pass_rate': v041_ri.get('pass_rate', 0),
                'passed_windows': v041_ri.get('passed_windows', 0),
                'total_windows': v041_ri.get('total_windows', 0),
                'avg_spread': v041_ri.get('avg_spread', 0),
                'overall_passed': v041_ri.get('overall_passed', False),
                'config': 'fund_ratio=0.70, growth_composite=0.30',
                'features': 'features_v04_1.parquet (156 cols)',
            },
        }
        
        analysis['v040_vs_v041_comparison'] = comparison
        
        # Determine which is worse
        v040_rate = comparison['v040']['pass_rate']
        v041_rate = comparison['v041']['pass_rate']
        
        if v041_rate < v040_rate:
            analysis['root_causes'].append({
                'cause': 'V041_WORSE_THAN_V040',
                'detail': f'V0.4.1 rank inversion pass rate ({v041_rate:.1%}) is WORSE than '
                          f'V0.4.0 ({v040_rate:.1%}). The growth_composite factor DEGRADED ranking ability.',
                'severity': 'HIGH',
            })
        
        analysis['summary'] = {
            'v040_rank_inversion_passed': v040_ri.get('overall_passed', False),
            'v041_rank_inversion_passed': v041_ri.get('overall_passed', False),
            'v040_pass_rate': v040_rate,
            'v041_pass_rate': v041_rate,
            'which_is_worse': 'V0.4.1' if v041_rate < v040_rate else 'V0.4.0',
            'conclusion': '',
        }
    else:
        analysis['summary'] = {
            'v040_rank_inversion_passed': ri_results.get('overall_passed', False) if ri_results else None,
            'v040_pass_rate': ri_results.get('pass_rate', 0) if ri_results else 0,
        }
    
    # 5. Key Recommendations
    analysis['recommendations'] = [
        'V0.4.0 uses only fund_ratio (20 ratios) + fund_metric (19 metrics) + log transform. '
        'These 39 factors have average |ICIR| that needs to be checked.',
        'V0.4.1 adds growth_composite (fund_growth + analyst + income) which may be noisy '
        'and cause rank inversion in certain market conditions.',
        'The fundamental issue is that ALL these factors are derived from the same '
        'financial statement data, leading to high correlation and diluted signals.',
        'Consider using only the highest ICIR factors rather than equal-weighting all factors in a group.',
    ]
    
    # Final conclusion
    if analysis.get('summary', {}).get('conclusion') == '':
        v040_passed = analysis['summary'].get('v040_rank_inversion_passed', False)
        v041_passed = analysis['summary'].get('v041_rank_inversion_passed', False)
        
        if v040_passed and not v041_passed:
            analysis['summary']['conclusion'] = (
                'V0.4.0 PASSES rank inversion but V0.4.1 FAILS. '
                'The growth_composite factor introduced in V0.4.1 degraded the model\'s '
                'ranking ability. V0.4.0\'s simpler 3-factor config (fund_ratio + fund_metric + log_metric) '
                'maintains better Top5% > Bottom20% ordering.'
            )
        elif not v040_passed and not v041_passed:
            analysis['summary']['conclusion'] = (
                'BOTH V0.4.0 and V0.4.1 FAIL rank inversion. '
                'The fundamental problem is that the factor weights and combinations '
                'do not produce reliable Top5% > Bottom20% ordering. '
                'The equal-weight averaging within factor groups dilutes signal quality.'
            )
        elif v040_passed and v041_passed:
            analysis['summary']['conclusion'] = (
                'Both V0.4.0 and V0.4.1 PASS rank inversion. '
                'The ranking ability is maintained in both configurations.'
            )
    
    return analysis


# ═══════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4.0 True Rank Inversion Test")
    print("=" * 80)
    
    # 1. Load data
    df, price_pivot = load_data()
    
    # 2. Get all factor columns from the data
    all_factor_cols = [c for c in df.columns 
                       if c not in ['date', 'ticker', 'close', 'open', 'high', 'low', 'volume',
                                    'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'fwd_ret_30d',
                                    'fwd_ret_60d', 'fwd_ret_90d']]
    print(f"\n📊 Available factors: {len(all_factor_cols)}")
    
    # Filter to V0.4.0 factor groups
    v040_factors = []
    for group, cols in FACTOR_GROUPS.items():
        for col in cols:
            if col in df.columns and col not in v040_factors:
                v040_factors.append(col)
    print(f"📊 V0.4.0 factor columns: {len(v040_factors)}")
    
    # 3. Compute cross-sectional ranks
    ranks = compute_cross_sectional_ranks(df, v040_factors)
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)
    ranks = add_combo_factors(ranks)
    
    # 4. TRUE Rank Inversion Test
    ri_results = compute_true_rank_inversion(
        ranks, price_pivot, V040_WEIGHTS,
        train_months=6, test_months=6, hold_days=30
    )
    
    # 5. IC/ICIR Analysis
    ic_results = compute_factor_ic_analysis(df, v040_factors, price_pivot, hold_days=30)
    
    # 6. Stability Analysis
    stability_results = compute_factor_stability(df, v040_factors, n_periods=4)
    
    # 7. Correlation Analysis
    corr_results = compute_factor_correlations(df, FACTOR_GROUPS)
    
    # 8. Root Cause Analysis (with V0.4.1 comparison)
    v041_ri_path = DATA_DIR / "v041_fixed_validation_results.json"
    root_cause = analyze_root_cause(
        ri_results, ic_results, stability_results, corr_results,
        v041_ri_path=v041_ri_path
    )
    
    # ═══════════════════════════════════════════════
    #  Save Results
    # ═══════════════════════════════════════════════
    
    # Save rank inversion results
    ri_output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'task': 'V0.4.0 True Rank Inversion Test (Top5% vs Bottom20%)',
            'config': {
                'weights': V040_WEIGHTS,
                'features': 'features_v02.parquet',
                'train_months': 6,
                'test_months': 6,
                'hold_days': 30,
                'top_n': 10,
                'cost': 0.001,
                'stop_loss': -0.15,
            },
            'previous_fake_ri': {
                'note': 'V0.4.0 previously used "Sharpe degradation stability" as rank_inversion, '
                        'which is NOT the standard Top5% vs Bottom20% test.',
                'old_method': 'early_avg_sharpe vs recent_avg_sharpe, threshold=50% degradation',
                'old_result': 'PASSED (early=2.07, recent=1.659, degradation=19.9%)',
            },
        },
        'rank_inversion': ri_results,
        'ic_analysis': {
            'total_factors_analyzed': len(ic_results),
            'factors': ic_results,
        },
        'factor_groups_used': FACTOR_GROUPS,
        'v040_weights': V040_WEIGHTS,
    }
    
    with open(OUTPUT_RI, 'w') as f:
        json.dump(ri_output, f, indent=2, default=str)
    print(f"\n✅ Rank inversion results saved: {OUTPUT_RI}")
    
    # Save root cause analysis
    root_cause_output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'task': 'Rank Inversion Root Cause Analysis (V0.4.0 vs V0.4.1)',
        },
        'root_cause_analysis': root_cause,
        'factor_stability': stability_results,
        'factor_correlations': corr_results,
    }
    
    with open(OUTPUT_ROOT_CAUSE, 'w') as f:
        json.dump(root_cause_output, f, indent=2, default=str)
    print(f"✅ Root cause analysis saved: {OUTPUT_ROOT_CAUSE}")
    
    # ═══════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("📋 SUMMARY")
    print(f"{'='*80}")
    
    if ri_results:
        print(f"\n🔍 V0.4.0 TRUE Rank Inversion:")
        print(f"   Pass Rate: {ri_results['pass_rate']:.1%} ({ri_results['passed_windows']}/{ri_results['total_windows']})")
        print(f"   Overall: {'✅ PASS' if ri_results['overall_passed'] else '❌ FAIL'}")
        print(f"   Avg Spread (Top5% - Bottom20%): {ri_results['avg_spread']:.4f}")
        print(f"   Median Spread: {ri_results['median_spread']:.4f}")
        print(f"   Avg % dates with positive spread: {ri_results['avg_positive_spread_pct']:.1%}")
        
        print(f"\n   Window Details:")
        for w in ri_results['windows']:
            mark = "✅" if w['passed'] else "❌"
            print(f"   {mark} W{w['window_idx']:2d}: {w['period']}  "
                  f"Top5%={w['avg_top5_return']:.4f}  Bot20%={w['avg_bottom20_return']:.4f}  "
                  f"Spread={w['spread']:.4f}  (+{w['positive_spread_pct']:.0%})")
    
    if root_cause.get('summary'):
        print(f"\n📊 Root Cause Summary:")
        for k, v in root_cause['summary'].items():
            print(f"   {k}: {v}")
    
    if root_cause.get('root_causes'):
        print(f"\n🚨 Root Causes Identified:")
        for rc in root_cause['root_causes']:
            print(f"   [{rc['severity']}] {rc['cause']}: {rc['detail'][:120]}...")
    
    elapsed = time.time() - t0
    print(f"\n⏱️ Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
