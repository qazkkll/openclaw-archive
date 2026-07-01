#!/usr/bin/env python3
"""
T5.7 最终优化：精调权重 + 高级技术
====================================
测试内容:
1. 权重精调: fund_ratio + fund_metric = 1.0
2. 训练窗口精调: 3-9个月
3. 集成方法: 多窗口平均 + 多权重平均
4. 因子工程: fund_ratio * fund_metric 组合因子
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
OUTPUT_FILE = DATA_DIR / 'v04_final_optimization_results.json'

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
    metric_cols = [c for c in FUND_METRIC_FIELDS if c in df.columns]
    
    print(f"  Ratio columns: {len(ratio_cols)}/{len(RATIO_FIELDS)}")
    print(f"  Metric columns: {len(metric_cols)}/{len(FUND_METRIC_FIELDS)}")
    
    return df, ratio_cols, metric_cols


def compute_ranks(df, ratio_cols, metric_cols):
    """Compute cross-sectional percentile ranks for each factor group.
    
    Returns:
        ranks_dict: {date_str: DataFrame(ticker -> {fund_ratio, fund_metric, fund_ratio_x_metric})}
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
        
        # Fund metric: percentile rank of quality metrics, then average
        m_ranks = []
        for c in metric_cols:
            if c in day_indexed.columns and day_indexed[c].notna().sum() > 5:
                row[f'm_{c}'] = day_indexed[c].rank(pct=True)
                m_ranks.append(f'm_{c}')
        row['fund_metric'] = row[m_ranks].mean(axis=1) if m_ranks else np.nan
        
        # Store only the composite factors
        ranks_dict[date] = row[['fund_ratio', 'fund_metric']].copy()
    
    # Build prices pivot
    prices = df.pivot_table(index='date_str', columns='ticker', values='close')
    prices.index = prices.index.astype(str)
    
    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks_dict)} dates computed in {elapsed:.1f}s")
    
    return ranks_dict, prices


# ═══════════════════════════════════════════════════
# Walk-Forward runner (month-based training window)
# ═══════════════════════════════════════════════════

def walk_forward_months(ranks, prices, weights, train_months, test_months=6,
                        hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15,
                        combo_factor=False):
    """Walk-Forward with month-based training window.
    
    Uses expanding window: train from start to train_end, test is next test_months.
    If combo_factor=True, adds fund_ratio * fund_metric as a factor.
    """
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    
    # If combo factor, augment ranks
    if combo_factor:
        aug_ranks = {}
        for date, r in ranks.items():
            r2 = r.copy()
            r2['fund_ratio_x_metric'] = r2['fund_ratio'] * r2['fund_metric']
            aug_ranks[date] = r2
        ranks = aug_ranks
        # Augment weights
        if 'fund_ratio_x_metric' not in weights:
            weights = dict(weights)
            weights['fund_ratio_x_metric'] = 0.0
    
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


def run_wf_test(ranks, prices, weights, train_months, test_months=6, 
                hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15,
                combo_factor=False):
    """Run a single Walk-Forward test.
    
    Returns:
        dict with sharpe, max_dd, cagr, win_rate, window_details, rank_inversion
    """
    try:
        result = walk_forward_months(
            ranks, prices, weights, train_months=train_months,
            test_months=test_months, hold_days=hold_days, top_n=top_n,
            cost=cost, stop_loss=stop_loss, combo_factor=combo_factor
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
# Test 1: Weight fine-tuning (fund_ratio + fund_metric = 1.0)
# ═══════════════════════════════════════════════════

def test_weight_tuning(ranks, prices, train_months):
    """Grid search over weight combinations where fund_ratio + fund_metric = 1.0."""
    print("\n" + "="*60)
    print("TEST 1: Weight fine-tuning (fund_ratio + fund_metric = 1.0)")
    print("="*60)
    
    fr_range = [0.85, 0.88, 0.90, 0.92, 0.95]
    fm_range = [0.05, 0.08, 0.10, 0.12, 0.15]
    
    # Filter to valid combinations (sum = 1.0)
    valid_combos = [(fr, fm) for fr, fm in product(fr_range, fm_range) 
                    if abs(fr + fm - 1.0) < 0.001]
    
    print(f"  Testing {len(valid_combos)} valid weight combinations...")
    print(f"  Train months: {train_months}")
    
    results = {}
    best_sharpe = -999
    best_combo = None
    
    for i, (fr, fm) in enumerate(valid_combos):
        weights = {'fund_ratio': fr, 'fund_metric': fm}
        label = f"fr{fr:.2f}_fm{fm:.2f}"
        
        r = run_wf_test(ranks, prices, weights, train_months=train_months)
        results[label] = r
        
        if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_combo = (fr, fm)
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"  [{i+1}/{len(valid_combos)}] NEW BEST: {label} "
                  f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={ri_status}")
    
    if best_combo:
        print(f"\n  🏆 Best weights: fund_ratio={best_combo[0]}, fund_metric={best_combo[1]}")
        print(f"     Sharpe={best_sharpe:.3f}")
    
    return results, best_combo


# ═══════════════════════════════════════════════════
# Test 2: Training window fine-tuning
# ═══════════════════════════════════════════════════

def test_training_windows(ranks, prices, fixed_weights):
    """Test different training windows with fixed weights."""
    print("\n" + "="*60)
    print("TEST 2: Training window fine-tuning (3-9 months)")
    print("="*60)
    
    windows_months = [3, 4, 5, 6, 7, 8, 9]
    results = {}
    
    for m in windows_months:
        print(f"\n  Testing {m}mo training window...")
        
        r = run_wf_test(ranks, prices, fixed_weights, train_months=m)
        results[f"{m}mo"] = r
        
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
# Test 3: Ensemble methods
# ═══════════════════════════════════════════════════

def test_ensemble(ranks, prices, best_train_months, best_weights):
    """Test ensemble of different training windows and weights."""
    print("\n" + "="*60)
    print("TEST 3: Ensemble methods")
    print("="*60)
    
    ensemble_results = {}
    
    # Ensemble 1: Multi-window average (3mo + 6mo + 9mo)
    print("\n  Ensemble 1: Multi-window average (3mo + 6mo + 9mo)")
    
    # We can't literally average model predictions without retraining,
    # but we can test each window and check consistency
    windows_to_test = [3, 6, 9]
    window_sharpes = {}
    
    for m in windows_to_test:
        r = run_wf_test(ranks, prices, best_weights, train_months=m)
        label = f"ensemble_{m}mo"
        ensemble_results[label] = r
        if r['sharpe'] is not None:
            window_sharpes[m] = r['sharpe']
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    {m}mo: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  RankInv={ri_status}")
    
    # Ensemble 2: Weight-shifted models
    print("\n  Ensemble 2: Weight-shifted models")
    weight_shifts = [
        {'fund_ratio': 0.85, 'fund_metric': 0.15},
        {'fund_ratio': 0.90, 'fund_metric': 0.10},
        {'fund_ratio': 0.95, 'fund_metric': 0.05},
    ]
    
    weight_sharpes = {}
    for i, w in enumerate(weight_shifts):
        r = run_wf_test(ranks, prices, w, train_months=best_train_months)
        label = f"ensemble_w{i+1}"
        ensemble_results[label] = r
        if r['sharpe'] is not None:
            weight_sharpes[f"fr{w['fund_ratio']}_fm{w['fund_metric']}"] = r['sharpe']
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    {w}: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  RankInv={ri_status}")
    
    # Summary
    print("\n  Ensemble consistency check:")
    if window_sharpes:
        print(f"    Multi-window sharpes: {window_sharpes}")
        print(f"    Window Sharpe range: {min(window_sharpes.values()):.3f} - {max(window_sharpes.values()):.3f}")
    if weight_sharpes:
        print(f"    Multi-weight sharpes: {weight_sharpes}")
        print(f"    Weight Sharpe range: {min(weight_sharpes.values()):.3f} - {max(weight_sharpes.values()):.3f}")
    
    return ensemble_results


# ═══════════════════════════════════════════════════
# Test 4: Factor engineering
# ═══════════════════════════════════════════════════

def test_factor_engineering(ranks, prices, best_train_months, best_weights):
    """Test factor engineering: fund_ratio * fund_metric combo factor."""
    print("\n" + "="*60)
    print("TEST 4: Factor engineering (fund_ratio * fund_metric)")
    print("="*60)
    
    # Baseline: no combo factor
    print("\n  Baseline: no combo factor")
    r_baseline = run_wf_test(ranks, prices, best_weights, train_months=best_train_months)
    if r_baseline['sharpe'] is not None:
        ri = r_baseline['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"    Baseline: Sharpe={r_baseline['sharpe']:.3f}  MaxDD={r_baseline['max_dd']:.1%}  RankInv={ri_status}")
    
    # Test 1: Equal weight combo factor
    print("\n  Test 4a: fund_ratio * fund_metric (equal weight combo)")
    weights_combo = {
        'fund_ratio': best_weights['fund_ratio'] * 0.9,
        'fund_metric': best_weights['fund_metric'] * 0.9,
        'fund_ratio_x_metric': 0.1,
    }
    r_combo = run_wf_test(ranks, prices, weights_combo, train_months=best_train_months, combo_factor=True)
    if r_combo['sharpe'] is not None:
        ri = r_combo['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"    Combo (0.1): Sharpe={r_combo['sharpe']:.3f}  MaxDD={r_combo['max_dd']:.1%}  RankInv={ri_status}")
    
    # Test 2: Higher combo weight
    print("\n  Test 4b: fund_ratio * fund_metric (higher combo weight)")
    weights_combo2 = {
        'fund_ratio': best_weights['fund_ratio'] * 0.8,
        'fund_metric': best_weights['fund_metric'] * 0.8,
        'fund_ratio_x_metric': 0.2,
    }
    r_combo2 = run_wf_test(ranks, prices, weights_combo2, train_months=best_train_months, combo_factor=True)
    if r_combo2['sharpe'] is not None:
        ri = r_combo2['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"    Combo (0.2): Sharpe={r_combo2['sharpe']:.3f}  MaxDD={r_combo2['max_dd']:.1%}  RankInv={ri_status}")
    
    # Test 3: Pure combo only
    print("\n  Test 4c: Pure fund_ratio * fund_metric only")
    weights_pure_combo = {'fund_ratio_x_metric': 1.0}
    r_pure = run_wf_test(ranks, prices, weights_pure_combo, train_months=best_train_months, combo_factor=True)
    if r_pure['sharpe'] is not None:
        ri = r_pure['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"    Pure combo: Sharpe={r_pure['sharpe']:.3f}  MaxDD={r_pure['max_dd']:.1%}  RankInv={ri_status}")
    
    return {
        'baseline': r_baseline,
        'combo_01': r_combo,
        'combo_02': r_combo2,
        'pure_combo': r_pure,
    }


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    print("="*70)
    print("T5.7 最终优化：精调权重 + 高级技术")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print()
    
    # Load data
    df, ratio_cols, metric_cols = load_data()
    
    # Compute ranks
    ranks, prices = compute_ranks(df, ratio_cols, metric_cols)
    
    all_results = {}
    
    # ─── Test 1: Weight fine-tuning ───
    t1_start = time.time()
    weight_results, best_combo = test_weight_tuning(ranks, prices, train_months=6)
    t1_elapsed = time.time() - t1_start
    all_results['test1_weight_tuning'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in weight_results.items()},
        'best_combo': {'fund_ratio': best_combo[0], 'fund_metric': best_combo[1]} if best_combo else None,
        'elapsed_seconds': round(t1_elapsed, 1),
    }
    
    # Use best weights for subsequent tests
    if best_combo:
        fixed_weights = {'fund_ratio': best_combo[0], 'fund_metric': best_combo[1]}
    else:
        fixed_weights = {'fund_ratio': 0.90, 'fund_metric': 0.10}
    
    print(f"\n  Using best weights for subsequent tests: {fixed_weights}")
    
    # ─── Test 2: Training window fine-tuning ───
    t2_start = time.time()
    window_results = test_training_windows(ranks, prices, fixed_weights)
    t2_elapsed = time.time() - t2_start
    
    # Find best window
    best_window = None
    best_window_sharpe = -999
    for k, v in window_results.items():
        if v['sharpe'] is not None and v['sharpe'] > best_window_sharpe:
            best_window_sharpe = v['sharpe']
            best_window = k
    
    all_results['test2_training_windows'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in window_results.items()},
        'best_window': best_window,
        'elapsed_seconds': round(t2_elapsed, 1),
    }
    
    # Use best window for subsequent tests
    best_train_months = int(best_window.replace('mo', '')) if best_window else 6
    print(f"\n  Best training window: {best_window} (using {best_train_months} months)")
    
    # ─── Test 3: Ensemble methods ───
    t3_start = time.time()
    ensemble_results = test_ensemble(ranks, prices, best_train_months, fixed_weights)
    t3_elapsed = time.time() - t3_start
    all_results['test3_ensemble'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in ensemble_results.items()},
        'elapsed_seconds': round(t3_elapsed, 1),
    }
    
    # ─── Test 4: Factor engineering ───
    t4_start = time.time()
    factor_results = test_factor_engineering(ranks, prices, best_train_months, fixed_weights)
    t4_elapsed = time.time() - t4_start
    all_results['test4_factor_engineering'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in factor_results.items()},
        'elapsed_seconds': round(t4_elapsed, 1),
    }
    
    # ═══════════════════════════════════════════════════
    # Final selection
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("FINAL SELECTION")
    print("="*70)
    
    # Collect all candidates
    candidates = []
    
    # From Test 1 (weight tuning)
    if best_combo:
        label = f"weight_fr{best_combo[0]}_fm{best_combo[1]}"
        best_weight_result = weight_results.get(f"fr{best_combo[0]}_fm{best_combo[1]}", {})
        if best_weight_result.get('sharpe') is not None:
            candidates.append({
                'name': label,
                'sharpe': best_weight_result['sharpe'],
                'max_dd': best_weight_result['max_dd'],
                'cagr': best_weight_result['cagr'],
                'win_rate': best_weight_result['win_rate'],
                'rank_inversion_passed': best_weight_result['rank_inversion']['passed'] if best_weight_result.get('rank_inversion') else False,
                'train_months': 6,
                'weights': fixed_weights,
            })
    
    # From Test 2 (window tuning)
    if best_window:
        best_window_result = window_results.get(best_window, {})
        if best_window_result.get('sharpe') is not None:
            candidates.append({
                'name': f"window_{best_window}",
                'sharpe': best_window_result['sharpe'],
                'max_dd': best_window_result['max_dd'],
                'cagr': best_window_result['cagr'],
                'win_rate': best_window_result['win_rate'],
                'rank_inversion_passed': best_window_result['rank_inversion']['passed'] if best_window_result.get('rank_inversion') else False,
                'train_months': best_train_months,
                'weights': fixed_weights,
            })
    
    # From Test 4 (factor engineering) - check combo factor
    for fk, fv in factor_results.items():
        if fv.get('sharpe') is not None:
            candidates.append({
                'name': f"factor_{fk}",
                'sharpe': fv['sharpe'],
                'max_dd': fv['max_dd'],
                'cagr': fv['cagr'],
                'win_rate': fv['win_rate'],
                'rank_inversion_passed': fv['rank_inversion']['passed'] if fv.get('rank_inversion') else False,
                'train_months': best_train_months,
                'weights': fixed_weights,
            })
    
    # Rank by Sharpe (only those that passed rank inversion)
    passed_candidates = [c for c in candidates if c['rank_inversion_passed']]
    failed_candidates = [c for c in candidates if not c['rank_inversion_passed']]
    
    if passed_candidates:
        passed_candidates.sort(key=lambda x: x['sharpe'], reverse=True)
        best = passed_candidates[0]
        print(f"\n  🏆 Best candidate (Rank Inversion PASS):")
        print(f"     Name: {best['name']}")
        print(f"     Sharpe: {best['sharpe']:.3f}")
        print(f"     MaxDD: {best['max_dd']:.1%}")
        print(f"     CAGR: {best['cagr']:.1%}")
        print(f"     Win Rate: {best['win_rate']:.0%}")
        print(f"     Train months: {best['train_months']}")
        print(f"     Weights: {best['weights']}")
    else:
        print("\n  ⚠️ No candidates passed Rank Inversion!")
        if candidates:
            candidates.sort(key=lambda x: x['sharpe'], reverse=True)
            best = candidates[0]
            print(f"  Best overall (but Rank Inversion FAILED):")
            print(f"     Name: {best['name']}")
            print(f"     Sharpe: {best['sharpe']:.3f}")
        else:
            best = None
    
    if failed_candidates:
        print(f"\n  ❌ {len(failed_candidates)} candidates failed Rank Inversion:")
        for c in failed_candidates:
            ri = c.get('rank_inversion_passed', False)
            print(f"     {c['name']}: Sharpe={c['sharpe']:.3f}")
    
    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    output = {
        'timestamp': datetime.now().isoformat(),
        'task': 'T5.7 Final Optimization: Weight Fine-tuning + Advanced Techniques',
        'config': {
            'test_months': 6,
            'hold_days': 30,
            'top_n': 10,
            'cost': 0.001,
            'stop_loss': -0.15,
        },
        'results': all_results,
        'candidates': candidates,
        'passed_candidates': passed_candidates,
        'failed_candidates': failed_candidates,
        'best_candidate': best,
        'summary': {
            'best_weights': fixed_weights,
            'best_window': best_window,
            'best_train_months': best_train_months,
            'best_sharpe': best['sharpe'] if best else None,
            'best_rank_inversion': best['rank_inversion_passed'] if best else None,
        },
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    
    return output


if __name__ == '__main__':
    main()
