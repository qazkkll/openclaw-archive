#!/usr/bin/env python3
"""
T2.1 IC/ICIR Analysis for Falcon V0.4.0
Optimized: groupby iterator + vectorized per-date IC.
"""

import pandas as pd
import numpy as np
from scipy.stats import rankdata
import json, os, time, warnings
warnings.filterwarnings('ignore')

FEATURES_PATH = 'data/falcon/features_v02.parquet'
TARGETS_PATH = 'data/falcon/targets_v04.parquet'
OUTPUT_PATH = 'data/falcon/v04_ic_analysis.json'

NON_FACTOR_COLS = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap']
TARGET_COL = 'fwd_ret_30d'
STRONG_ICIR = 0.1
WEAK_ICIR = 0.05
UNSTABLE_STD = 0.05
MIN_STOCKS = 20

def main():
    t0 = time.time()
    print("=" * 60)
    print("T2.1 IC/ICIR Analysis - Falcon V0.4.0")
    print("=" * 60)

    feat = pd.read_parquet(FEATURES_PATH)
    tgt = pd.read_parquet(TARGETS_PATH)
    feat['date'] = pd.to_datetime(feat['date'])
    tgt['date'] = pd.to_datetime(tgt['date'])
    merged = feat.merge(tgt[['ticker', 'date', TARGET_COL]], on=['ticker', 'date'], how='inner')
    print(f"Merged: {merged.shape}, dates: {merged['date'].nunique()}, tickers: {merged['ticker'].nunique()}")

    factor_cols = [c for c in merged.columns if c not in NON_FACTOR_COLS + [TARGET_COL]]
    print(f"Factor columns: {len(factor_cols)}")
    
    n_dates = merged['date'].nunique()
    
    # Pre-convert to numpy for speed - extract from sorted version
    merged_sorted = merged.sort_values('date').reset_index(drop=True)
    target_arr = merged_sorted[TARGET_COL].values.astype(np.float64)
    factor_arrs = {f: merged_sorted[f].values.astype(np.float64) for f in factor_cols}
    
    # Get date boundaries (start indices for each date in sorted order)
    date_groups = merged_sorted.groupby('date').indices
    
    # Initialize accumulators
    ic_sums = np.zeros(len(factor_cols))
    ic_sq_sums = np.zeros(len(factor_cols))
    ic_counts = np.zeros(len(factor_cols), dtype=int)
    
    print(f"Computing IC for {n_dates} dates...")
    
    for di, (date, indices) in enumerate(date_groups.items()):
        if (di + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Date {di+1}/{n_dates} [{elapsed:.0f}s]")
        
        # Get target values for this date
        t_vals = target_arr[indices]
        t_valid_mask = ~np.isnan(t_vals)
        n_valid = t_valid_mask.sum()
        
        if n_valid < MIN_STOCKS:
            continue
        
        # Rank target
        t_valid = t_vals[t_valid_mask]
        t_ranked = rankdata(t_valid)
        t_mean = t_ranked.mean()
        t_std = t_ranked.std(ddof=0)
        
        if t_std == 0:
            continue
        
        # For each factor, compute IC
        for fj, factor in enumerate(factor_cols):
            f_vals = factor_arrs[factor][indices]
            both_valid = t_valid_mask & ~np.isnan(f_vals)
            n_both = both_valid.sum()
            if n_both < MIN_STOCKS:
                continue
            
            f_valid = f_vals[both_valid]
            t_r = rankdata(t_vals[both_valid])
            
            f_ranked = rankdata(f_valid)
            f_mean = f_ranked.mean()
            f_std = f_ranked.std(ddof=0)
            
            if f_std == 0:
                continue
            
            n = len(f_ranked)
            t_r_std = t_r.std(ddof=0)
            if t_r_std == 0:
                continue
            
            corr = np.sum((f_ranked - f_mean) * (t_r - t_r.mean())) / (n * f_std * t_r_std)
            
            ic_sums[fj] += corr
            ic_sq_sums[fj] += corr * corr
            ic_counts[fj] += 1
    
    # Compute statistics
    print("\nComputing statistics...")
    results_list = []
    
    for fj, factor in enumerate(factor_cols):
        n = ic_counts[fj]
        if n == 0:
            continue
        
        ic_mean = ic_sums[fj] / n
        ic_sq_mean = ic_sq_sums[fj] / n
        ic_var = ic_sq_mean - ic_mean * ic_mean
        ic_std = np.sqrt(max(ic_var, 0))
        
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0
        coverage = n / n_dates
        
        results_list.append({
            'name': factor,
            'ic_mean': round(ic_mean, 6),
            'ic_std': round(ic_std, 6),
            'icir': round(icir, 6),
            't_stat': round(t_stat, 4),
            'coverage': round(coverage, 4),
            'n_dates': int(n),
            'abs_icir': round(abs(icir), 6)
        })

    results_list.sort(key=lambda x: x['abs_icir'], reverse=True)

    strong = [r['name'] for r in results_list if abs(r['icir']) > STRONG_ICIR]
    weak = [r['name'] for r in results_list if abs(r['icir']) < WEAK_ICIR]
    unstable = [r['name'] for r in results_list if r['ic_std'] > UNSTABLE_STD]

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total factors analyzed: {len(results_list)}")
    print(f"Strong factors (|ICIR| > {STRONG_ICIR}): {len(strong)}")
    print(f"Weak factors (|ICIR| < {WEAK_ICIR}): {len(weak)}")
    print(f"Unstable factors (IC std > {UNSTABLE_STD}): {len(unstable)}")

    print(f"\nTop 20 factors by |ICIR|:")
    print(f"{'-' * 70}")
    for i, r in enumerate(results_list[:20], 1):
        print(f"{i:2d}. {r['name']:35s} ICIR={r['icir']:+.4f}  IC={r['ic_mean']:+.4f}  t={r['t_stat']:+.2f}  cov={r['coverage']:.1%}")

    print(f"\nBottom 10 factors (weakest):")
    print(f"{'-' * 70}")
    for i, r in enumerate(results_list[-10:], 1):
        print(f"{i:2d}. {r['name']:35s} ICIR={r['icir']:+.4f}  IC={r['ic_mean']:+.4f}  t={r['t_stat']:+.2f}  cov={r['coverage']:.1%}")

    output = {
        'metadata': {
            'target': TARGET_COL,
            'min_stocks_per_day': MIN_STOCKS,
            'strong_threshold': STRONG_ICIR,
            'weak_threshold': WEAK_ICIR,
            'unstable_threshold': UNSTABLE_STD,
            'total_factors': len(results_list),
            'date_range': f"{merged['date'].min().date()} to {merged['date'].max().date()}",
            'n_dates': int(n_dates),
            'n_tickers': int(merged['ticker'].nunique()),
            'ic_method': 'Spearman rank correlation',
            'icir_formula': 'mean(IC) / std(IC)',
            't_stat_formula': 'mean(IC) / (std(IC) / sqrt(N))'
        },
        'factors': results_list,
        'strong_factors': strong,
        'weak_factors': weak,
        'unstable_factors': unstable
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"{'=' * 60}")

if __name__ == '__main__':
    main()
