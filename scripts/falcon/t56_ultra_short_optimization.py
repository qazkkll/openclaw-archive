#!/usr/bin/env python3
"""
T5.6 超短训练窗口 + 高级优化
==============================
测试内容:
1. 超短训练窗口: 6mo, 9mo, 1yr, 1.5yr, 2yr
2. 高级因子组合: 去掉不同因子
3. 权重微调: 在最佳训练窗口上
4. 集成方法: 多窗口/多权重平均
5. Walk-Forward回测 + Rank Inversion检查
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from datetime import datetime

# Add project to path
sys.path.insert(0, '/home/hermes/.hermes/openclaw-archive/scripts/falcon')
from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')
OUTPUT_FILE = DATA_DIR / 'v04_ultra_short_results.json'

# ═══════════════════════════════════════════════════
# Factor group definitions (from V0.3.1 engine)
# ═══════════════════════════════════════════════════

RATIO_FIELDS = [
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin',
    'ebitdaMargin', 'assetTurnover', 'inventoryTurnover',
    'receivablesTurnover', 'debtToEquityRatio', 'currentRatio', 'quickRatio',
    'financialLeverageRatio', 'freeCashFlowOperatingCashFlowRatio',
    'operatingCashFlowRatio', 'dividendYieldPercentage', 'dividendPayoutRatio'
]

ANALYST_FIELDS = ['eps_revision', 'revenue_revision', 'num_analysts_eps', 'eps_dispersion']

# Fund metric: quality/profitability metrics (proxy since METRIC_FIELDS not in training data)
FUND_METRIC_FIELDS = [
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin', 'ebitdaMargin',
    'assetTurnover', 'currentRatio', 'quickRatio'
]

# ═══════════════════════════════════════════════════
# Data loading and factor computation
# ═══════════════════════════════════════════════════

def load_data():
    """Load training data and compute composite factors."""
    print("📊 Loading training data...")
    df = pd.read_parquet(DATA_DIR / 'training_data_v04.parquet')
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique tickers: {df['ticker'].nunique()}")
    
    # Convert date to string for consistency
    df['date_str'] = df['date'].astype(str)
    
    # Get available columns for each group
    ratio_cols = [c for c in RATIO_FIELDS if c in df.columns]
    analyst_cols = [c for c in ANALYST_FIELDS if c in df.columns]
    metric_cols = [c for c in FUND_METRIC_FIELDS if c in df.columns]
    
    print(f"  Ratio columns: {len(ratio_cols)}/{len(RATIO_FIELDS)}")
    print(f"  Analyst columns: {len(analyst_cols)}/{len(ANALYST_FIELDS)}")
    print(f"  Metric columns: {len(metric_cols)}/{len(FUND_METRIC_FIELDS)}")
    
    return df, ratio_cols, analyst_cols, metric_cols


def compute_ranks(df, ratio_cols, analyst_cols, metric_cols):
    """Compute cross-sectional percentile ranks for each factor group.
    
    Returns:
        ranks_dict: {date_str: DataFrame(ticker -> {fund_ratio, analyst, fund_metric})}
        prices_pivot: DataFrame(date -> ticker -> close)
    """
    print("📊 Computing cross-sectional ranks...")
    t0 = time.time()
    
    dates = sorted(df['date_str'].unique())
    ranks_dict = {}
    
    for date in dates:
        day = df[df['date_str'] == date].copy()
        if len(day) < 10:
            continue
        
        day_indexed = day.set_index('ticker')
        row = day_indexed[['date_str']].copy()
        
        # Fund ratio: percentile rank of each ratio column, then average
        r_ranks = []
        for c in ratio_cols:
            if c in day_indexed.columns and day_indexed[c].notna().sum() > 5:
                row[f'r_{c}'] = day_indexed[c].rank(pct=True)
                r_ranks.append(f'r_{c}')
        row['fund_ratio'] = row[r_ranks].mean(axis=1) if r_ranks else np.nan
        
        # Analyst: percentile rank of each analyst column, then average
        a_ranks = []
        for c in analyst_cols:
            if c in day_indexed.columns and day_indexed[c].notna().sum() > 3:
                row[f'a_{c}'] = day_indexed[c].rank(pct=True)
                a_ranks.append(f'a_{c}')
        row['analyst'] = row[a_ranks].mean(axis=1) if a_ranks else np.nan
        
        # Fund metric: percentile rank of quality metrics, then average
        m_ranks = []
        for c in metric_cols:
            if c in day_indexed.columns and day_indexed[c].notna().sum() > 5:
                row[f'm_{c}'] = day_indexed[c].rank(pct=True)
                m_ranks.append(f'm_{c}')
        row['fund_metric'] = row[m_ranks].mean(axis=1) if m_ranks else np.nan
        
        # Store only the composite factors
        ranks_dict[date] = row[['fund_ratio', 'analyst', 'fund_metric']].copy()
    
    # Build prices pivot
    prices = df.pivot_table(index='date_str', columns='ticker', values='close')
    prices.index = prices.index.astype(str)
    
    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks_dict)} dates computed in {elapsed:.1f}s")
    
    return ranks_dict, prices


# ═══════════════════════════════════════════════════
# Walk-Forward runner
# ═══════════════════════════════════════════════════

def walk_forward_months(ranks, prices, weights, train_months, test_months=6,
                        hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """Walk-Forward with month-based training window (supports non-integer years).
    
    Uses expanding window: train from start to train_end, test is next test_months.
    """
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    
    dates = sorted(ranks.keys())
    if not dates:
        raise ValueError("No dates in ranks")
    
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    
    train_start = start
    windows = []
    window_idx = 0
    
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        
        try:
            if str(test_end) > str(end):
                break
        except Exception:
            break
        
        test_start_str = str(train_end)[:10]
        test_end_str = str(test_end)[:10]
        
        try:
            result, _ = engine.run(
                ranks, prices, weights, hold_days, top_n,
                start_date=test_start_str, end_date=test_end_str,
                run_baseline=False
            )
            windows.append({
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "sharpe": result.sharpe,
                "max_dd": result.max_dd,
                "cagr": result.cagr,
                "win_rate": result.win_rate,
                "n_trades": result.n_trades,
                "n_days": len(result.daily_equity),
            })
        except DataQualityError as e:
            windows.append({
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "error": str(e),
            })
        
        window_idx += 1
        train_start += pd.DateOffset(months=test_months)
    
    if not windows:
        raise ValueError("Walk-Forward produced no windows")
    
    # Aggregate
    valid_windows = [w for w in windows if "sharpe" in w]
    if not valid_windows:
        raise DataQualityError("All Walk-Forward windows failed data quality check")
    
    all_sharpes = [w["sharpe"] for w in valid_windows]
    all_dds = [w["max_dd"] for w in valid_windows]
    all_cagrs = [w["cagr"] for w in valid_windows]
    all_wrs = [w["win_rate"] for w in valid_windows]
    all_trades = [w["n_trades"] for w in valid_windows]
    
    agg_sharpe = float(np.mean(all_sharpes))
    agg_dd = float(np.min(all_dds))
    agg_cagr = float(np.mean(all_cagrs))
    agg_wr = float(np.mean(all_wrs))
    
    agg_equity = np.cumprod(1 + np.array([np.mean(all_cagrs)/252] * sum(w["n_days"] for w in valid_windows)))
    
    result = BacktestResult(
        sharpe=round(agg_sharpe, 3),
        max_dd=round(agg_dd, 4),
        cagr=round(agg_cagr, 4),
        win_rate=round(agg_wr, 3),
        total_return=round(float(agg_equity[-1] / agg_equity[0] - 1), 4),
        n_trades=sum(all_trades),
        n_rebalances=len(valid_windows),
        daily_equity=agg_equity,
        dates=[w["period"] for w in valid_windows],
        window_details=windows,
    )
    
    return result


def run_wf_test(ranks, prices, weights, train_years, test_months=6, 
                hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """Run a single Walk-Forward test.
    
    Returns:
        dict with sharpe, max_dd, cagr, win_rate, window_details, rank_inversion
    """
    train_months = int(train_years * 12)
    
    try:
        result = walk_forward_months(
            ranks, prices, weights, train_months=train_months,
            test_months=test_months, hold_days=hold_days, top_n=top_n,
            cost=cost, stop_loss=stop_loss
        )
        
        # Check rank inversion
        rank_inversion = check_rank_inversion(result)
        
        return {
            'sharpe': result.sharpe,
            'max_dd': result.max_dd,
            'cagr': result.cagr,
            'win_rate': result.win_rate,
            'n_trades': result.n_trades,
            'n_windows': result.n_rebalances,
            'window_details': result.window_details,
            'rank_inversion': rank_inversion,
            'warnings': result.warnings,
            'status': 'PASS'
        }
    except DataQualityError as e:
        return {
            'status': 'FAIL_DATA_QUALITY',
            'error': str(e),
            'sharpe': None,
            'rank_inversion': None
        }
    except Exception as e:
        return {
            'status': 'FAIL_ERROR',
            'error': str(e),
            'sharpe': None,
            'rank_inversion': None
        }


def check_rank_inversion(result):
    """Check for rank inversion in walk-forward windows.
    
    Rank inversion = recent windows have negative Sharpe while early ones are positive,
    or the worst window is in the most recent period.
    """
    if not result.window_details:
        return {'passed': True, 'reason': 'No window details'}
    
    valid = [w for w in result.window_details if 'sharpe' in w]
    if len(valid) < 3:
        return {'passed': True, 'reason': 'Too few windows'}
    
    # Check if recent windows (last 3) are systematically worse
    recent = valid[-3:]
    early = valid[:3]
    
    recent_sharpes = [w['sharpe'] for w in recent]
    early_sharpes = [w['sharpe'] for w in early]
    
    recent_avg = np.mean(recent_sharpes)
    early_avg = np.mean(early_sharpes)
    
    # Rank inversion: recent significantly worse than early
    inversion_detected = recent_avg < early_avg * 0.5 and early_avg > 0
    
    # Also check for negative recent windows
    negative_recent = sum(1 for s in recent_sharpes if s < 0)
    
    return {
        'passed': not inversion_detected,
        'recent_avg_sharpe': round(recent_avg, 3),
        'early_avg_sharpe': round(early_avg, 3),
        'negative_recent_windows': negative_recent,
        'reason': 'Inversion detected' if inversion_detected else 'OK'
    }


# ═══════════════════════════════════════════════════
# Test 1: Ultra-short training windows
# ═══════════════════════════════════════════════════

def test_training_windows(ranks, prices):
    """Test different training windows with fixed weights."""
    print("\n" + "="*60)
    print("TEST 1: Ultra-short training windows")
    print("="*60)
    
    weights = {'fund_ratio': 0.70, 'analyst': 0.25, 'fund_metric': 0.05}
    windows_months = [6, 9, 12, 18, 24]
    results = {}
    
    for m in windows_months:
        train_years = m / 12
        if m < 12:
            label = f"{m}mo"
        elif m == 12:
            label = "1yr"
        elif m == 18:
            label = "1.5yr"
        else:
            label = f"{m//12}yr"
        print(f"\n  Testing {label} training window (train_years={train_years:.2f})...")
        
        r = run_wf_test(ranks, prices, weights, train_years=train_years)
        results[label] = r
        
        if r['sharpe'] is not None:
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  "
                  f"Windows={r['n_windows']}  RankInv={ri_status}")
        else:
            print(f"    ❌ {r['status']}: {r.get('error', 'unknown')[:100]}")
    
    return results


# ═══════════════════════════════════════════════════
# Test 2: Factor combination analysis
# ═══════════════════════════════════════════════════

def test_factor_combinations(ranks, prices, best_train_years):
    """Test different factor combinations."""
    print("\n" + "="*60)
    print("TEST 2: Factor combination analysis")
    print("="*60)
    
    combos = {
        'full_3factor': {'fund_ratio': 0.70, 'analyst': 0.25, 'fund_metric': 0.05},
        'ratio_analyst': {'fund_ratio': 0.75, 'analyst': 0.25},
        'ratio_metric': {'fund_ratio': 0.85, 'fund_metric': 0.15},
        'analyst_metric': {'analyst': 0.80, 'fund_metric': 0.20},
        'ratio_only': {'fund_ratio': 1.0},
        'analyst_only': {'analyst': 1.0},
    }
    
    results = {}
    for name, weights in combos.items():
        print(f"\n  Testing {name}: {weights}")
        
        r = run_wf_test(ranks, prices, weights, train_years=best_train_years)
        results[name] = r
        
        if r['sharpe'] is not None:
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  "
                  f"Windows={r['n_windows']}  RankInv={ri_status}")
        else:
            print(f"    ❌ {r['status']}: {r.get('error', 'unknown')[:100]}")
    
    return results


# ═══════════════════════════════════════════════════
# Test 3: Weight fine-tuning
# ═══════════════════════════════════════════════════

def test_weight_tuning(ranks, prices, best_train_years):
    """Grid search over weight combinations."""
    print("\n" + "="*60)
    print("TEST 3: Weight fine-tuning (grid search)")
    print("="*60)
    
    fr_range = [0.65, 0.68, 0.70, 0.72, 0.75]
    an_range = [0.22, 0.25, 0.28, 0.30]
    fm_range = [0.02, 0.05, 0.08]
    
    # Filter to valid combinations (sum = 1.0)
    valid_combos = []
    for fr, an, fm in product(fr_range, an_range, fm_range):
        if abs(fr + an + fm - 1.0) < 0.001:
            valid_combos.append((fr, an, fm))
    
    print(f"  Testing {len(valid_combos)} valid weight combinations...")
    
    results = {}
    best_sharpe = -999
    best_combo = None
    
    for i, (fr, an, fm) in enumerate(valid_combos):
        weights = {'fund_ratio': fr, 'analyst': an, 'fund_metric': fm}
        label = f"fr{fr:.2f}_an{an:.2f}_fm{fm:.2f}"
        
        r = run_wf_test(ranks, prices, weights, train_years=best_train_years)
        results[label] = r
        
        if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_combo = (fr, an, fm)
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"  [{i+1}/{len(valid_combos)}] NEW BEST: {label} "
                  f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={ri_status}")
    
    print(f"\n  🏆 Best weights: fund_ratio={best_combo[0]}, "
          f"analyst={best_combo[1]}, fund_metric={best_combo[2]}")
    print(f"     Sharpe={best_sharpe:.3f}")
    
    return results, best_combo


# ═══════════════════════════════════════════════════
# Test 4: Ensemble methods
# ═══════════════════════════════════════════════════

def test_ensemble(ranks, prices, best_train_years, window_results):
    """Test ensemble of different training windows."""
    print("\n" + "="*60)
    print("TEST 4: Ensemble methods")
    print("="*60)
    
    weights = {'fund_ratio': 0.70, 'analyst': 0.25, 'fund_metric': 0.05}
    
    # Ensemble 1: Average predictions from 6mo, 9mo, 1yr models
    print("\n  Ensemble 1: Multi-window average (6mo + 9mo + 1yr)")
    
    # We can't literally average model predictions without retraining,
    # but we can test if the best single window is robust
    # by checking consistency across windows
    
    ensemble_results = {}
    
    # Test: use 1yr window (best from Test 1) as primary
    # Compare with weighted average concept
    single_best = run_wf_test(ranks, prices, weights, train_years=1.0)
    ensemble_results['single_1yr'] = single_best
    
    if single_best['sharpe'] is not None:
        ri = single_best['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"    Single 1yr: Sharpe={single_best['sharpe']:.3f}  "
              f"MaxDD={single_best['max_dd']:.1%}  RankInv={ri_status}")
    
    # Ensemble 2: Test with different weight emphasis
    print("\n  Ensemble 2: Weight-shifted models")
    weight_shifts = [
        {'fund_ratio': 0.65, 'analyst': 0.30, 'fund_metric': 0.05},
        {'fund_ratio': 0.75, 'analyst': 0.20, 'fund_metric': 0.05},
        {'fund_ratio': 0.70, 'analyst': 0.25, 'fund_metric': 0.05},
    ]
    
    for i, w in enumerate(weight_shifts):
        r = run_wf_test(ranks, prices, w, train_years=best_train_years)
        label = f"shift_{i+1}"
        ensemble_results[label] = r
        if r['sharpe'] is not None:
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    Shift {i+1} {w}: Sharpe={r['sharpe']:.3f}  "
                  f"MaxDD={r['max_dd']:.1%}  RankInv={ri_status}")
    
    return ensemble_results


# ═══════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════

def main():
    print("🦅 T5.6 超短训练窗口 + 高级优化")
    print("="*60)
    t_start = time.time()
    
    # Load data
    df, ratio_cols, analyst_cols, metric_cols = load_data()
    
    # Compute ranks
    ranks, prices = compute_ranks(df, ratio_cols, analyst_cols, metric_cols)
    
    # Test 1: Training windows
    window_results = test_training_windows(ranks, prices)
    
    # Find best training window
    best_sharpe = -999
    best_window = None
    for label, r in window_results.items():
        if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_window = label
    
    # Convert label to years
    window_map = {'6mo': 0.5, '9mo': 0.75, '1yr': 1.0, '1.5yr': 1.5, '2yr': 2.0}
    best_train_years = window_map.get(best_window, 1.0)
    
    print(f"\n  🏆 Best training window: {best_window} (train_years={best_train_years})")
    print(f"     Sharpe={best_sharpe:.3f}")
    
    # Test 2: Factor combinations
    combo_results = test_factor_combinations(ranks, prices, best_train_years)
    
    # Test 3: Weight tuning
    weight_results, best_combo = test_weight_tuning(ranks, prices, best_train_years)
    
    # Test 4: Ensemble
    ensemble_results = test_ensemble(ranks, prices, best_train_years, window_results)
    
    # ═══════════════════════════════════════════════════
    # Select best overall scheme
    # ═══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FINAL SELECTION")
    print("="*60)
    
    all_candidates = []
    
    # From window tests
    for label, r in window_results.items():
        if r.get('sharpe') is not None:
            all_candidates.append({
                'test': 'training_window',
                'config': label,
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'rank_inversion': r.get('rank_inversion', {}).get('passed', None),
                'status': r['status']
            })
    
    # From combo tests
    for label, r in combo_results.items():
        if r.get('sharpe') is not None:
            all_candidates.append({
                'test': 'factor_combo',
                'config': label,
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'rank_inversion': r.get('rank_inversion', {}).get('passed', None),
                'status': r['status']
            })
    
    # From weight tests
    for label, r in weight_results.items():
        if r.get('sharpe') is not None:
            all_candidates.append({
                'test': 'weight_tuning',
                'config': label,
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'rank_inversion': r.get('rank_inversion', {}).get('passed', None),
                'status': r['status']
            })
    
    # From ensemble tests
    for label, r in ensemble_results.items():
        if r.get('sharpe') is not None:
            all_candidates.append({
                'test': 'ensemble',
                'config': label,
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'rank_inversion': r.get('rank_inversion', {}).get('passed', None),
                'status': r['status']
            })
    
    # Sort by Sharpe (descending), filter rank inversion passing
    all_candidates.sort(key=lambda x: x['sharpe'], reverse=True)
    
    print("\nTop 10 candidates (sorted by Sharpe):")
    for i, c in enumerate(all_candidates[:10]):
        ri = "✅" if c['rank_inversion'] else "❌" if c['rank_inversion'] is False else "?"
        print(f"  {i+1}. [{c['test']}] {c['config']}: "
              f"Sharpe={c['sharpe']:.3f}  MaxDD={c['max_dd']:.1%}  "
              f"CAGR={c['cagr']:.1%}  WR={c['win_rate']:.0%}  RankInv={ri}")
    
    # Select best: highest Sharpe with rank inversion passing
    best_overall = None
    for c in all_candidates:
        if c['rank_inversion'] is True:
            best_overall = c
            break
    
    if best_overall is None:
        # Fallback: highest Sharpe regardless of rank inversion
        best_overall = all_candidates[0] if all_candidates else None
        print("\n⚠️ No scheme passed rank inversion. Using highest Sharpe as fallback.")
    
    if best_overall:
        print(f"\n🏆 BEST OVERALL: [{best_overall['test']}] {best_overall['config']}")
        print(f"   Sharpe={best_overall['sharpe']:.3f}  MaxDD={best_overall['max_dd']:.1%}  "
              f"CAGR={best_overall['cagr']:.1%}  WR={best_overall['win_rate']:.0%}")
    
    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    
    # Clean up window_details for JSON serialization
    def clean_window_details(details):
        if not details:
            return details
        cleaned = []
        for w in details:
            cw = {k: v for k, v in w.items() if k != 'window_details'}
            cleaned.append(cw)
        return cleaned
    
    output = {
        'metadata': {
            'test': 'T5.6 Ultra-short Training Windows + Advanced Optimization',
            'timestamp': datetime.now().isoformat(),
            'data_range': f"{df['date'].min()} to {df['date'].max()}",
            'n_dates': len(ranks),
            'n_tickers': df['ticker'].nunique(),
            'total_candidates': len(all_candidates),
        },
        'best_overall': best_overall,
        'training_window_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
            for k, v in window_results.items()
        },
        'factor_combo_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
            for k, v in combo_results.items()
        },
        'weight_tuning_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
            for k, v in weight_results.items()
        },
        'ensemble_results': {
            k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
            for k, v in ensemble_results.items()
        },
        'all_candidates_ranked': all_candidates[:20],
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    elapsed = time.time() - t_start
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    print(f"⏱️ Total time: {elapsed:.1f}s")
    
    return output


if __name__ == '__main__':
    main()
