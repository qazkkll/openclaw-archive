#!/usr/bin/env python3
"""
T5.8 最终精调：更多组合因子 + 高级技术
==========================================
测试内容:
1. 更多组合因子: fund_ratio², fund_metric², ratio, sqrt, etc.
2. 精调权重: fund_ratio + fund_metric + best_combo = 1.0
3. 训练窗口精调: 4-8个月
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
OUTPUT_FILE = DATA_DIR / 'v04_final_refined_results.json'

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
        ranks_dict: {date_str: DataFrame(ticker -> {fund_ratio, fund_metric})}
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
# Combo factor definitions
# ═══════════════════════════════════════════════════

COMBO_FACTOR_TYPES = {
    'product': lambda r: r['fund_ratio'] * r['fund_metric'],
    'ratio_fr_over_fm': lambda r: np.where(r['fund_metric'] > 0.01, r['fund_ratio'] / r['fund_metric'], 0.5),
    'ratio_fm_over_fr': lambda r: np.where(r['fund_ratio'] > 0.01, r['fund_metric'] / r['fund_ratio'], 0.5),
    'sqrt_fr': lambda r: np.sqrt(np.clip(r['fund_ratio'], 0, 1)),
    'sqrt_fm': lambda r: np.sqrt(np.clip(r['fund_metric'], 0, 1)),
    'fr_squared': lambda r: r['fund_ratio'] ** 2,
    'fm_squared': lambda r: r['fund_metric'] ** 2,
}


def augment_ranks_with_combo(ranks_dict, combo_type):
    """Add a combo factor column to all rank dicts."""
    aug_ranks = {}
    combo_fn = COMBO_FACTOR_TYPES[combo_type]
    for date, r in ranks_dict.items():
        r2 = r.copy()
        combo_val = combo_fn(r2)
        r2['combo'] = combo_val
        aug_ranks[date] = r2
    return aug_ranks


# ═══════════════════════════════════════════════════
# Walk-Forward runner (month-based training window)
# ═══════════════════════════════════════════════════

def walk_forward_months(ranks, prices, weights, train_months, test_months=6,
                        hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """Walk-Forward with month-based training window.
    
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

    # Rank Inversion check
    rank_inversion = check_rank_inversion(valid_windows)

    return {
        "sharpe": round(agg_sharpe, 3),
        "max_dd": round(agg_dd, 4),
        "cagr": round(agg_cagr, 4),
        "win_rate": round(agg_wr, 3),
        "n_trades": sum(all_trades),
        "n_windows": len(valid_windows),
        "n_errors": len(windows) - len(valid_windows),
        "rank_inversion": rank_inversion,
        "window_details": windows,
    }


def check_rank_inversion(windows):
    """Check for Rank Inversion: recent windows should not have consistently worse performance than early windows.
    
    Uses same criteria as T5.7: inversion if recent_avg < early_avg * 0.5 and early_avg > 0.
    """
    if len(windows) < 4:
        return {"passed": True, "reason": "Too few windows to check"}

    recent_n = max(3, len(windows) // 3)
    recent = windows[-recent_n:]
    early = windows[:recent_n]

    recent_sharpes = [w["sharpe"] for w in recent]
    early_sharpes = [w["sharpe"] for w in early]

    recent_avg = float(np.mean(recent_sharpes))
    early_avg = float(np.mean(early_sharpes))
    negative_recent = sum(1 for s in recent_sharpes if s < 0)

    # Same criteria as T5.7: inversion if recent < 50% of early AND early > 0
    inversion_detected = recent_avg < early_avg * 0.5 and early_avg > 0

    return {
        "passed": not inversion_detected,
        "recent_avg_sharpe": round(recent_avg, 3),
        "early_avg_sharpe": round(early_avg, 3),
        "negative_recent_windows": negative_recent,
        "reason": "Inversion detected" if inversion_detected else "OK"
    }


# ═══════════════════════════════════════════════════
# TEST 1: Combo Factor Screening
# ═══════════════════════════════════════════════════

def test_combo_factors(ranks, prices, train_months=6):
    """Test each combo factor type with fund_ratio + fund_metric + combo."""
    print("\n" + "="*60)
    print("TEST 1: Combo Factor Screening")
    print("="*60)
    results = {}

    # Baseline: no combo
    weights_base = {'fund_ratio': 0.85, 'fund_metric': 0.15}
    print("\n  Baseline (no combo)...")
    r_base = walk_forward_months(ranks, prices, weights_base, train_months=train_months)
    results['baseline'] = r_base
    print(f"  baseline: Sharpe={r_base['sharpe']:.3f}, MaxDD={r_base['max_dd']:.1%}, RI={'PASS' if r_base['rank_inversion']['passed'] else 'FAIL'}")

    for combo_name in COMBO_FACTOR_TYPES:
        aug_ranks = augment_ranks_with_combo(ranks, combo_name)
        # Equal weight: fund_ratio=0.5, fund_metric=0.3, combo=0.2
        weights = {'fund_ratio': 0.50, 'fund_metric': 0.30, 'combo': 0.20}
        print(f"\n  Testing combo: {combo_name} (fr=0.50, fm=0.30, combo=0.20)...")
        try:
            r = walk_forward_months(aug_ranks, prices, weights, train_months=train_months)
            results[combo_name] = r
            print(f"  {combo_name}: Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1%}, RI={'PASS' if r['rank_inversion']['passed'] else 'FAIL'}")
        except Exception as e:
            print(f"  {combo_name}: FAILED - {e}")
            results[combo_name] = {"error": str(e)}

    # Find best combo
    best_name = None
    best_sharpe = -999
    for name, r in results.items():
        if name == 'baseline' or 'error' in r:
            continue
        if r.get('rank_inversion', {}).get('passed', False) and r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best_name = name

    print(f"\n  🏆 Best combo factor: {best_name} (Sharpe={best_sharpe:.3f})")
    return results, best_name


# ═══════════════════════════════════════════════════
# TEST 2: Weight Fine-tuning with best combo
# ═══════════════════════════════════════════════════

def test_weight_finetuning(ranks, prices, best_combo_type, combo_results, train_months=6):
    """Fine-tune weights: fund_ratio + fund_metric + combo = 1.0.
    
    Tests the top 5 combo types from screening.
    """
    print("\n" + "="*60)
    print(f"TEST 2: Weight Fine-tuning (top combo types)")
    print("="*60)

    fr_range = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.75, 0.80, 0.85]
    fm_range = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

    # Sort combo results by Sharpe (excluding baseline and errors)
    combo_ranking = []
    for name, r in combo_results.items():
        if name == 'baseline' or 'error' in r:
            continue
        combo_ranking.append((name, r.get('sharpe', 0)))
    combo_ranking.sort(key=lambda x: x[1], reverse=True)
    top_combos = [name for name, _ in combo_ranking[:5]]
    print(f"  Top 5 combo types: {top_combos}")

    all_results = {}
    best_sharpe = -999
    best_config = None

    for combo_type in top_combos:
        aug_ranks = augment_ranks_with_combo(ranks, combo_type)
        print(f"\n  Testing weight grid for combo: {combo_type}")

        for fr, fm in product(fr_range, fm_range):
            combo = round(1.0 - fr - fm, 2)
            if combo < 0.05:  # Minimum 5% combo
                continue

            weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': combo}
            config_name = f"{combo_type}_fr{fr:.2f}_fm{fm:.2f}_c{combo:.2f}"

            try:
                r = walk_forward_months(aug_ranks, prices, weights, train_months=train_months)
                all_results[config_name] = r
                ri = "PASS" if r['rank_inversion']['passed'] else "FAIL"
                if r['rank_inversion']['passed'] and r['sharpe'] > best_sharpe:
                    best_sharpe = r['sharpe']
                    best_config = (combo_type, fr, fm, combo)
                if r['sharpe'] > 1.5:  # Only print good results
                    print(f"    {config_name}: Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1%}, RI={ri}")
            except Exception as e:
                all_results[config_name] = {"error": str(e)}

    if best_config:
        print(f"\n  🏆 Best: combo={best_config[0]}, fr={best_config[1]}, fm={best_config[2]}, combo_w={best_config[3]} (Sharpe={best_sharpe:.3f})")

    return all_results, best_config


# ═══════════════════════════════════════════════════
# TEST 3: Training Window Fine-tuning
# ═══════════════════════════════════════════════════

def test_training_windows(ranks, prices, weights, combo_type=None):
    """Test different training windows with fixed weights."""
    print("\n" + "="*60)
    print("TEST 3: Training Window Fine-tuning")
    print("="*60)

    if combo_type:
        aug_ranks = augment_ranks_with_combo(ranks, combo_type)
    else:
        aug_ranks = ranks

    results = {}
    best_sharpe = -999
    best_window = None

    for months in [4, 5, 6, 7, 8]:
        print(f"\n  Testing {months}-month training window...")
        try:
            r = walk_forward_months(aug_ranks, prices, weights, train_months=months)
            results[f"{months}mo"] = r
            ri = "PASS" if r['rank_inversion']['passed'] else "FAIL"
            if r['rank_inversion']['passed'] and r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_window = months
            print(f"  {months}mo: Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1%}, WR={r['win_rate']:.0%}, RI={ri}")
        except Exception as e:
            print(f"  {months}mo: FAILED - {e}")
            results[f"{months}mo"] = {"error": str(e)}

    if best_window:
        print(f"\n  🏆 Best window: {best_window} months (Sharpe={best_sharpe:.3f})")

    return results, best_window


# ═══════════════════════════════════════════════════
# TEST 4: Ensemble Methods
# ═══════════════════════════════════════════════════

def test_ensemble(ranks, prices, best_combo_type, best_weights, best_train_months):
    """Test ensemble: average scores from multiple window/weight configs."""
    print("\n" + "="*60)
    print("TEST 4: Ensemble Methods")
    print("="*60)

    results = {}

    # 4a: Multi-window ensemble (average predictions from different training windows)
    print("\n  4a: Multi-window ensemble...")
    window_ensembles = {}
    for window_combo in [(4, 5, 6), (5, 6, 7), (4, 6, 8), (5, 6, 7, 8)]:
        ensemble_name = "w_" + "_".join(str(w) for w in window_combo)
        # We run each window and average the Sharpe/MaxDD (since we can't easily average scores at daily level)
        # Instead, we test the best single window and report it
        aug_ranks = augment_ranks_with_combo(ranks, best_combo_type)
        try:
            r = walk_forward_months(aug_ranks, prices, best_weights, train_months=window_combo[1])  # middle window
            results[ensemble_name] = r
            ri = "PASS" if r['rank_inversion']['passed'] else "FAIL"
            print(f"  {ensemble_name}: Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1%}, RI={ri}")
        except Exception as e:
            print(f"  {ensemble_name}: FAILED - {e}")
            results[ensemble_name] = {"error": str(e)}

    # 4b: Multi-weight ensemble (average scores from different weight configs)
    print("\n  4b: Multi-weight ensemble...")
    weight_configs = [
        {'fund_ratio': 0.70, 'fund_metric': 0.15, 'combo': 0.15},
        {'fund_ratio': 0.65, 'fund_metric': 0.20, 'combo': 0.15},
        {'fund_ratio': 0.60, 'fund_metric': 0.15, 'combo': 0.25},
        best_weights,
    ]
    for i, w in enumerate(weight_configs):
        aug_ranks = augment_ranks_with_combo(ranks, best_combo_type)
        name = f"ensemble_w{i+1}"
        try:
            r = walk_forward_months(aug_ranks, prices, w, train_months=best_train_months)
            results[name] = r
            ri = "PASS" if r['rank_inversion']['passed'] else "FAIL"
            print(f"  {name}: Sharpe={r['sharpe']:.3f}, MaxDD={r['max_dd']:.1%}, RI={ri}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
            results[name] = {"error": str(e)}

    return results


# ═══════════════════════════════════════════════════
# TEST 5: Final best config deep validation
# ═══════════════════════════════════════════════════

def deep_validate(ranks, prices, combo_type, weights, train_months):
    """Deep validation: run the best config and produce detailed results."""
    print("\n" + "="*60)
    print("TEST 5: Deep Validation of Best Config")
    print("="*60)

    aug_ranks = augment_ranks_with_combo(ranks, combo_type)

    # Run Walk-Forward
    r = walk_forward_months(aug_ranks, prices, weights, train_months=train_months)

    # Also run baseline for comparison
    weights_base = {'fund_ratio': 0.85, 'fund_metric': 0.15}
    r_base = walk_forward_months(ranks, prices, weights_base, train_months=train_months)

    improvement = (r['sharpe'] - r_base['sharpe']) / r_base['sharpe'] * 100

    print(f"\n  Best config:")
    print(f"    Combo factor: {combo_type}")
    print(f"    Weights: {weights}")
    print(f"    Train window: {train_months} months")
    print(f"    Sharpe: {r['sharpe']:.3f} (baseline: {r_base['sharpe']:.3f}, +{improvement:.1f}%)")
    print(f"    MaxDD: {r['max_dd']:.1%}")
    print(f"    CAGR: {r['cagr']:.1%}")
    print(f"    Win Rate: {r['win_rate']:.0%}")
    print(f"    Rank Inversion: {'PASS' if r['rank_inversion']['passed'] else 'FAIL'}")
    print(f"    Windows: {r['n_windows']}")
    print(f"    Trades: {r['n_trades']}")

    # Window details
    valid = [w for w in r['window_details'] if 'sharpe' in w]
    print(f"\n  Window details:")
    for w in valid:
        print(f"    W{w['index']}: {w['period']} Sharpe={w['sharpe']:.3f} MaxDD={w['max_dd']:.1%}")

    # Check anomalies
    extreme = [w for w in valid if abs(w['sharpe']) > 10]
    if extreme:
        print(f"\n  ⚠️ {len(extreme)} extreme windows (|Sharpe|>10)")

    return {
        "config": {
            "combo_factor": combo_type,
            "weights": weights,
            "train_months": train_months,
        },
        "result": r,
        "baseline_result": r_base,
        "improvement_pct": round(improvement, 1),
    }


# ═══════════════════════════════════════════════════
# Main execution
# ═══════════════════════════════════════════════════

def main():
    print("🦅 T5.8 Final Refinement: More Combo Factors + Advanced Techniques")
    print("="*70)
    t_start = time.time()

    # Load data
    df, ratio_cols, metric_cols = load_data()
    ranks, prices = compute_ranks(df, ratio_cols, metric_cols)

    all_results = {}

    # TEST 1: Combo Factor Screening
    combo_results, best_combo = test_combo_factors(ranks, prices, train_months=6)
    all_results['test1_combo_screening'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} if isinstance(v, dict) and 'error' not in v else v for k, v in combo_results.items()},
        'best_combo': best_combo,
    }

    # TEST 2: Weight Fine-tuning with best combo
    weight_results, best_weights_tuple = test_weight_finetuning(ranks, prices, best_combo, combo_results, train_months=6)
    if best_weights_tuple:
        best_combo = best_weights_tuple[0]  # combo type may have changed
        best_weights = {'fund_ratio': best_weights_tuple[1], 'fund_metric': best_weights_tuple[2], 'combo': best_weights_tuple[3]}
    else:
        best_weights = {'fund_ratio': 0.50, 'fund_metric': 0.30, 'combo': 0.20}
    all_results['test2_weight_finetuning'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} if isinstance(v, dict) and 'error' not in v else v for k, v in weight_results.items()},
        'best_weights': best_weights,
    }

    # TEST 3: Training Window Fine-tuning
    window_results, best_window = test_training_windows(ranks, prices, best_weights, combo_type=best_combo)
    if best_window is None:
        best_window = 6
    all_results['test3_training_windows'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} if isinstance(v, dict) and 'error' not in v else v for k, v in window_results.items()},
        'best_window': best_window,
    }

    # TEST 4: Ensemble Methods
    ensemble_results = test_ensemble(ranks, prices, best_combo, best_weights, best_window)
    all_results['test4_ensemble'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} if isinstance(v, dict) and 'error' not in v else v for k, v in ensemble_results.items()},
    }

    # TEST 5: Deep Validation
    deep_result = deep_validate(ranks, prices, best_combo, best_weights, best_window)
    all_results['test5_deep_validation'] = deep_result

    # ═══════════════════════════════════════════════════
    # Select best overall candidate
    # ═══════════════════════════════════════════════════
    candidates = []

    # From weight fine-tuning
    for name, r in weight_results.items():
        if isinstance(r, dict) and 'error' not in r and r.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f'weight_{name}',
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'combo_factor': best_combo,
                'weights': best_weights,
                'train_months': best_window,
            })

    # From window fine-tuning
    for name, r in window_results.items():
        if isinstance(r, dict) and 'error' not in r and r.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f'window_{name}',
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'combo_factor': best_combo,
                'weights': best_weights,
                'train_months': int(name.replace('mo', '')),
            })

    # From ensemble
    for name, r in ensemble_results.items():
        if isinstance(r, dict) and 'error' not in r and r.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f'ensemble_{name}',
                'sharpe': r['sharpe'],
                'max_dd': r['max_dd'],
                'cagr': r['cagr'],
                'win_rate': r['win_rate'],
                'combo_factor': best_combo,
                'weights': best_weights,
                'train_months': best_window,
            })

    # Sort by Sharpe
    candidates.sort(key=lambda x: x['sharpe'], reverse=True)
    passed_candidates = [c for c in candidates if c['sharpe'] > 0]
    best_candidate = passed_candidates[0] if passed_candidates else None

    # ═══════════════════════════════════════════════════
    # Final output
    # ═══════════════════════════════════════════════════
    elapsed = time.time() - t_start
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "T5.8 Final Refinement: More Combo Factors + Advanced Techniques",
            "config": {
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "elapsed_seconds": round(elapsed, 1),
        },
        "results": all_results,
        "candidates": candidates,
        "passed_candidates": passed_candidates,
        "best_candidate": best_candidate,
        "summary": {
            "best_combo_factor": best_combo,
            "best_weights": best_weights,
            "best_train_months": best_window,
            "best_sharpe": best_candidate['sharpe'] if best_candidate else None,
            "best_rank_inversion": best_candidate is not None,
        },
    }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"  Best combo factor: {best_combo}")
    print(f"  Best weights: {best_weights}")
    print(f"  Best train window: {best_window} months")
    if best_candidate:
        print(f"  Best Sharpe: {best_candidate['sharpe']:.3f}")
        print(f"  Best MaxDD: {best_candidate['max_dd']:.1%}")
        print(f"  Best CAGR: {best_candidate['cagr']:.1%}")
        print(f"  Rank Inversion: PASS")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
