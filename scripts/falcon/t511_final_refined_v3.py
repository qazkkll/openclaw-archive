#!/usr/bin/env python3
"""
T5.11 最终精调V3：更多组合因子 + 高级技术
============================================
测试内容:
1. 更多组合因子: sqrt, quartic root, log, cross products (3 new combos)
2. 精调权重(fund_ratio + fund_metric + combo = 1.0)
3. 训练窗口精调: 5, 5.5, 6, 6.5, 7 months
4. 集成方法: 多窗口平均 + 多权重平均
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
OUTPUT_FILE = DATA_DIR / 'v04_final_refined_v3_results.json'

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
    
    df['date_str'] = df['date'].astype(str)
    
    ratio_cols = [c for c in RATIO_FIELDS if c in df.columns]
    metric_cols = [c for c in FUND_METRIC_FIELDS if c in df.columns]
    
    print(f"  Ratio columns: {len(ratio_cols)}/{len(RATIO_FIELDS)}")
    print(f"  Metric columns: {len(metric_cols)}/{len(FUND_METRIC_FIELDS)}")
    
    return df, ratio_cols, metric_cols


def compute_ranks(df, ratio_cols, metric_cols):
    """Compute cross-sectional percentile ranks for each factor group."""
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

def augment_ranks_with_combo(ranks, combo_type):
    """Add combo factor to ranks dict.
    
    V3 includes 3 new combo types vs V2:
        - fr_x_log_fm: fund_ratio * log(fund_metric + 1)    [NEW]
        - fm_x_log_fr: fund_metric * log(fund_ratio + 1)    [NEW]
        - sqrt_fr_x_sqrt_fm: sqrt(fund_ratio) * sqrt(fm)    [NEW]
    """
    aug = {}
    for date, r in ranks.items():
        r2 = r.copy()
        fr = r2['fund_ratio'].fillna(0.5)
        fm = r2['fund_metric'].fillna(0.5)
        
        if combo_type == 'sqrt_fr':
            r2['combo'] = np.sqrt(fr.clip(0))
        elif combo_type == 'qrt_fm':
            r2['combo'] = np.power(fm.clip(0), 0.25)
        elif combo_type == 'log_fr':
            r2['combo'] = np.log1p(fr.clip(0))
        elif combo_type == 'log_fm':
            r2['combo'] = np.log1p(fm.clip(0))
        elif combo_type == 'fr_x_sqrt_fm':
            r2['combo'] = fr * np.sqrt(fm.clip(0))
        elif combo_type == 'fm_x_sqrt_fr':
            r2['combo'] = fm * np.sqrt(fr.clip(0))
        elif combo_type == 'fr_squared':
            r2['combo'] = fr ** 2
        elif combo_type == 'sqrt_fm':
            r2['combo'] = np.sqrt(fm.clip(0))
        elif combo_type == 'fr_x_log_fm':
            r2['combo'] = fr * np.log1p(fm.clip(0))
        elif combo_type == 'fm_x_log_fr':
            r2['combo'] = fm * np.log1p(fr.clip(0))
        elif combo_type == 'sqrt_fr_x_sqrt_fm':
            r2['combo'] = np.sqrt(fr.clip(0)) * np.sqrt(fm.clip(0))
        else:
            raise ValueError(f"Unknown combo_type: {combo_type}")
        
        aug[date] = r2
    return aug


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
        train_end = train_start + pd.DateOffset(days=int(train_months * 30.44))
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
                hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """Run a single Walk-Forward test."""
    try:
        result = walk_forward_months(
            ranks, prices, weights, train_months=train_months,
            test_months=test_months, hold_days=hold_days, top_n=top_n,
            cost=cost, stop_loss=stop_loss
        )
        
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
    """Check for rank inversion in walk-forward windows."""
    if not result.window_details:
        return {'passed': True, 'reason': 'No window details'}
    
    valid = [w for w in result.window_details if 'sharpe' in w]
    if len(valid) < 3:
        return {'passed': True, 'reason': 'Too few windows'}
    
    recent = valid[-3:]
    early = valid[:3]
    
    recent_sharpes = [w['sharpe'] for w in recent]
    early_sharpes = [w['sharpe'] for w in early]
    
    recent_avg = np.mean(recent_sharpes)
    early_avg = np.mean(early_sharpes)
    
    inversion_detected = recent_avg < early_avg * 0.5 and early_avg > 0
    negative_recent = sum(1 for s in recent_sharpes if s < 0)
    
    return {
        'passed': not inversion_detected,
        'recent_avg_sharpe': round(recent_avg, 3),
        'early_avg_sharpe': round(early_avg, 3),
        'negative_recent_windows': negative_recent,
        'reason': 'Inversion detected' if inversion_detected else 'OK'
    }


# ═══════════════════════════════════════════════════
# Test 1: Combo Factor Screening (V3: 11 combos total)
# ═══════════════════════════════════════════════════

def test_combo_screening(ranks, prices, train_months=6):
    """Test various combo factors to find the best one."""
    print("\n" + "="*60)
    print("TEST 1: Combo Factor Screening (V3: 11 combos)")
    print("="*60)
    
    # V3 combo list: all V2 combos + 3 new ones
    combo_types = [
        'sqrt_fr', 'qrt_fm', 'log_fr', 'log_fm',
        'fr_x_sqrt_fm', 'fm_x_sqrt_fr', 'fr_squared', 'sqrt_fm',
        # V3 NEW:
        'fr_x_log_fm', 'fm_x_log_fr', 'sqrt_fr_x_sqrt_fm'
    ]
    
    results = {}
    best_sharpe = -999
    best_combo = None
    
    # Baseline (no combo)
    print("\n  Testing baseline (no combo)...")
    r_base = run_wf_test(ranks, prices, 
                         {'fund_ratio': 0.70, 'fund_metric': 0.30},
                         train_months=train_months)
    results['baseline'] = r_base
    if r_base['sharpe'] is not None:
        ri = r_base['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"    Baseline: Sharpe={r_base['sharpe']:.3f}  MaxDD={r_base['max_dd']:.1%}  "
              f"CAGR={r_base['cagr']:.1%}  WR={r_base['win_rate']:.0%}  RankInv={ri_status}")
        if r_base['sharpe'] > best_sharpe:
            best_sharpe = r_base['sharpe']
            best_combo = 'baseline'
    
    # Test each combo type
    for combo_type in combo_types:
        print(f"\n  Testing {combo_type}...")
        aug_ranks = augment_ranks_with_combo(ranks, combo_type)
        
        # Test with combo weight = 0.15 (from T5.8 best)
        weights = {'fund_ratio': 0.70, 'fund_metric': 0.15, 'combo': 0.15}
        r = run_wf_test(aug_ranks, prices, weights, train_months=train_months)
        results[combo_type] = r
        
        if r['sharpe'] is not None:
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    {combo_type}: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={ri_status}")
            if r['sharpe'] > best_sharpe and ri['passed']:
                best_sharpe = r['sharpe']
                best_combo = combo_type
    
    if best_combo:
        print(f"\n  🏆 Best combo: {best_combo} (Sharpe={best_sharpe:.3f})")
    
    return results, best_combo


# ═══════════════════════════════════════════════════
# Test 2: Weight Fine-tuning
# ═══════════════════════════════════════════════════

def test_weight_finetuning(ranks, prices, best_combo, train_months=6):
    """Grid search over weight combinations where fund_ratio + fund_metric + combo = 1.0."""
    print("\n" + "="*60)
    print(f"TEST 2: Weight Fine-tuning (combo={best_combo})")
    print("="*60)
    
    fr_range = [0.65, 0.68, 0.70, 0.72, 0.75]
    fm_range = [0.12, 0.15, 0.18, 0.20]
    combo_range = [0.10, 0.12, 0.15, 0.18, 0.20]
    
    # Filter to valid combinations (sum = 1.0)
    valid_combos = [(fr, fm, c) for fr, fm, c in product(fr_range, fm_range, combo_range) 
                    if abs(fr + fm + c - 1.0) < 0.001]
    
    print(f"  Testing {len(valid_combos)} valid weight combinations...")
    print(f"  Train months: {train_months}")
    
    if best_combo == 'baseline':
        # No combo factor - but still test with combo placeholder to keep logic simple
        results = {}
        best_sharpe = -999
        best_weights = None
        
        for i, (fr, fm, c) in enumerate(valid_combos):
            weights = {'fund_ratio': fr, 'fund_metric': fm}
            label = f"fr{fr:.2f}_fm{fm:.2f}_c{c:.2f}"
            
            r = run_wf_test(ranks, prices, weights, train_months=train_months)
            results[label] = r
            
            if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_weights = (fr, fm, c)
                ri = r['rank_inversion']
                ri_status = "✅" if ri['passed'] else "❌"
                print(f"  [{i+1}/{len(valid_combos)}] NEW BEST: {label} "
                      f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                      f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={ri_status}")
    else:
        # With combo factor
        aug_ranks = augment_ranks_with_combo(ranks, best_combo)
        
        results = {}
        best_sharpe = -999
        best_weights = None
        
        for i, (fr, fm, c) in enumerate(valid_combos):
            weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c}
            label = f"fr{fr:.2f}_fm{fm:.2f}_c{c:.2f}"
            
            r = run_wf_test(aug_ranks, prices, weights, train_months=train_months)
            results[label] = r
            
            if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_weights = (fr, fm, c)
                ri = r['rank_inversion']
                ri_status = "✅" if ri['passed'] else "❌"
                print(f"  [{i+1}/{len(valid_combos)}] NEW BEST: {label} "
                      f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                      f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={ri_status}")
    
    if best_weights:
        print(f"\n  🏆 Best weights: fr={best_weights[0]}, fm={best_weights[1]}, combo={best_weights[2]}")
        print(f"     Sharpe={best_sharpe:.3f}")
    
    return results, best_weights


# ═══════════════════════════════════════════════════
# Test 3: Training Window Fine-tuning
# ═══════════════════════════════════════════════════

def test_training_windows(ranks, prices, best_weights, best_combo):
    """Test different training windows with fixed weights."""
    print("\n" + "="*60)
    print("TEST 3: Training Window Fine-tuning (5, 5.5, 6, 6.5, 7 months)")
    print("="*60)
    
    windows_months = [5, 5.5, 6, 6.5, 7]
    results = {}
    
    fr, fm, c = best_weights
    
    for m in windows_months:
        print(f"\n  Testing {m}mo training window...")
        
        if best_combo == 'baseline':
            weights = {'fund_ratio': fr, 'fund_metric': fm}
            r = run_wf_test(ranks, prices, weights, train_months=m)
        else:
            aug_ranks = augment_ranks_with_combo(ranks, best_combo)
            weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c}
            r = run_wf_test(aug_ranks, prices, weights, train_months=m)
        
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
# Test 4: Ensemble Methods
# ═══════════════════════════════════════════════════

def test_ensemble(ranks, prices, best_weights, best_combo, best_train_months):
    """Test ensemble of different training windows and weights."""
    print("\n" + "="*60)
    print("TEST 4: Ensemble Methods")
    print("="*60)
    
    fr, fm, c = best_weights
    ensemble_results = {}
    
    # Ensemble 1: Multi-window average (5mo + 6mo + 7mo)
    print("\n  Ensemble 1: Multi-window average (5mo + 6mo + 7mo)")
    
    windows_to_test = [5, 6, 7]
    window_sharpes = {}
    
    for m in windows_to_test:
        if best_combo == 'baseline':
            weights = {'fund_ratio': fr, 'fund_metric': fm}
            r = run_wf_test(ranks, prices, weights, train_months=m)
        else:
            aug_ranks = augment_ranks_with_combo(ranks, best_combo)
            weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c}
            r = run_wf_test(aug_ranks, prices, weights, train_months=m)
        
        label = f"ensemble_{m}mo"
        ensemble_results[label] = r
        if r['sharpe'] is not None:
            window_sharpes[m] = r['sharpe']
            ri = r['rank_inversion']
            ri_status = "✅" if ri['passed'] else "❌"
            print(f"    {m}mo: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  RankInv={ri_status}")
    
    # Ensemble 2: Weight-shifted models
    print("\n  Ensemble 2: Weight-shifted models")
    
    if best_combo == 'baseline':
        weight_shifts = [
            {'fund_ratio': 0.65, 'fund_metric': 0.35},
            {'fund_ratio': 0.70, 'fund_metric': 0.30},
            {'fund_ratio': 0.75, 'fund_metric': 0.25},
        ]
    else:
        weight_shifts = [
            {'fund_ratio': 0.65, 'fund_metric': 0.15, 'combo': 0.20},
            {'fund_ratio': 0.70, 'fund_metric': 0.15, 'combo': 0.15},
            {'fund_ratio': 0.75, 'fund_metric': 0.12, 'combo': 0.13},
        ]
    
    weight_sharpes = {}
    for i, w in enumerate(weight_shifts):
        if best_combo == 'baseline':
            r = run_wf_test(ranks, prices, w, train_months=best_train_months)
        else:
            aug_ranks = augment_ranks_with_combo(ranks, best_combo)
            r = run_wf_test(aug_ranks, prices, w, train_months=best_train_months)
        
        label = f"ensemble_w{i+1}"
        ensemble_results[label] = r
        if r['sharpe'] is not None:
            w_label = f"fr{w['fund_ratio']}_fm{w['fund_metric']}"
            if 'combo' in w:
                w_label += f"_c{w['combo']}"
            weight_sharpes[w_label] = r['sharpe']
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
# Test 5: Final Walk-Forward (best config)
# ═══════════════════════════════════════════════════

def test_final_wf(ranks, prices, best_weights, best_combo, best_train_months):
    """Run final Walk-Forward with best configuration."""
    print("\n" + "="*60)
    print(f"TEST 5: Final Walk-Forward (best config)")
    print("="*60)
    
    fr, fm, c = best_weights
    
    if best_combo == 'baseline':
        weights = {'fund_ratio': fr, 'fund_metric': fm}
        print(f"  Weights: {weights}")
        r = run_wf_test(ranks, prices, weights, train_months=best_train_months)
    else:
        aug_ranks = augment_ranks_with_combo(ranks, best_combo)
        weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c}
        print(f"  Weights: {weights}")
        print(f"  Combo: {best_combo}")
        r = run_wf_test(aug_ranks, prices, weights, train_months=best_train_months)
    
    if r['sharpe'] is not None:
        ri = r['rank_inversion']
        ri_status = "✅" if ri['passed'] else "❌"
        print(f"\n  🏆 Final result:")
        print(f"     Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
              f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={ri_status}")
        
        # Print window details
        if r.get('window_details'):
            print(f"\n  Window details:")
            for w in r['window_details']:
                if 'sharpe' in w:
                    print(f"    {w['period']}: Sharpe={w['sharpe']:.3f}  MaxDD={w['max_dd']:.1%}")
    
    return r


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    print("="*70)
    print("T5.11 最终精调V3：更多组合因子 + 高级技术")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}")
    print()
    
    # Load data
    df, ratio_cols, metric_cols = load_data()
    
    # Compute ranks
    ranks, prices = compute_ranks(df, ratio_cols, metric_cols)
    
    all_results = {}
    t_total_start = time.time()
    
    # ─── Test 1: Combo Factor Screening ───
    t1_start = time.time()
    combo_results, best_combo = test_combo_screening(ranks, prices, train_months=6)
    t1_elapsed = time.time() - t1_start
    all_results['test1_combo_screening'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in combo_results.items()},
        'best_combo': best_combo,
        'elapsed_seconds': round(t1_elapsed, 1),
    }
    
    # ─── Test 2: Weight Fine-tuning ───
    t2_start = time.time()
    weight_results, best_weights = test_weight_finetuning(ranks, prices, best_combo, train_months=6)
    t2_elapsed = time.time() - t2_start
    all_results['test2_weight_finetuning'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in weight_results.items()},
        'best_weights': {'fund_ratio': best_weights[0], 'fund_metric': best_weights[1], 'combo': best_weights[2]} if best_weights else None,
        'elapsed_seconds': round(t2_elapsed, 1),
    }
    
    # Use best weights for subsequent tests
    if not best_weights:
        best_weights = (0.70, 0.15, 0.15) if best_combo != 'baseline' else (0.70, 0.30, 0.0)
    print(f"\n  Using best weights for subsequent tests: {best_weights}")
    
    # ─── Test 3: Training Window Fine-tuning ───
    t3_start = time.time()
    window_results = test_training_windows(ranks, prices, best_weights, best_combo)
    t3_elapsed = time.time() - t3_start
    
    # Find best window
    best_window = None
    best_window_sharpe = -999
    for k, v in window_results.items():
        if v['sharpe'] is not None and v['sharpe'] > best_window_sharpe:
            ri = v.get('rank_inversion', {})
            if ri and ri.get('passed', True):
                best_window_sharpe = v['sharpe']
                best_window = k
    
    all_results['test3_training_windows'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in window_results.items()},
        'best_window': best_window,
        'elapsed_seconds': round(t3_elapsed, 1),
    }
    
    # Use best window for subsequent tests
    if best_window:
        best_train_months = float(best_window.replace('mo', ''))
    else:
        best_train_months = 6.0
    print(f"\n  Best training window: {best_window} (using {best_train_months} months)")
    
    # ─── Test 4: Ensemble Methods ───
    t4_start = time.time()
    ensemble_results = test_ensemble(ranks, prices, best_weights, best_combo, best_train_months)
    t4_elapsed = time.time() - t4_start
    all_results['test4_ensemble'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'} 
                    for k, v in ensemble_results.items()},
        'elapsed_seconds': round(t4_elapsed, 1),
    }
    
    # ─── Test 5: Final Walk-Forward ───
    t5_start = time.time()
    final_result = test_final_wf(ranks, prices, best_weights, best_combo, best_train_months)
    t5_elapsed = time.time() - t5_start
    all_results['test5_final_wf'] = {
        'result': {kk: vv for kk, vv in final_result.items() if kk != 'window_details'},
        'window_details': final_result.get('window_details', []),
        'elapsed_seconds': round(t5_elapsed, 1),
    }
    
    t_total_elapsed = time.time() - t_total_start
    
    # ═══════════════════════════════════════════════════
    # Final selection
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("FINAL SELECTION")
    print("="*70)
    
    # Collect all candidates
    candidates = []
    
    # From Test 1 (combo screening)
    for k, v in combo_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"combo_{k}",
                'sharpe': v['sharpe'],
                'max_dd': v['max_dd'],
                'cagr': v['cagr'],
                'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'),
                'rank_inversion_passed': True,
                'test': 'combo_screening',
            })
    
    # From Test 2 (weight tuning)
    for k, v in weight_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"weight_{k}",
                'sharpe': v['sharpe'],
                'max_dd': v['max_dd'],
                'cagr': v['cagr'],
                'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'),
                'rank_inversion_passed': True,
                'test': 'weight_tuning',
            })
    
    # From Test 3 (window tuning)
    for k, v in window_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"window_{k}",
                'sharpe': v['sharpe'],
                'max_dd': v['max_dd'],
                'cagr': v['cagr'],
                'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'),
                'rank_inversion_passed': True,
                'test': 'window_tuning',
            })
    
    # From Test 4 (ensemble)
    for k, v in ensemble_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"ensemble_{k}",
                'sharpe': v['sharpe'],
                'max_dd': v['max_dd'],
                'cagr': v['cagr'],
                'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'),
                'rank_inversion_passed': True,
                'test': 'ensemble',
            })
    
    # From Test 5 (final)
    if final_result.get('sharpe') is not None and final_result.get('rank_inversion', {}).get('passed', False):
        candidates.append({
            'name': 'final_best_config',
            'sharpe': final_result['sharpe'],
            'max_dd': final_result['max_dd'],
            'cagr': final_result['cagr'],
            'win_rate': final_result['win_rate'],
            'n_windows': final_result.get('n_windows'),
            'rank_inversion_passed': True,
            'test': 'final_wf',
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
        print(f"     Source: {best['test']}")
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
        for c in failed_candidates[:5]:
            print(f"     {c['name']}: Sharpe={c['sharpe']:.3f}")
    
    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'task': 'T5.11 Final Refinement V3: More Combo Factors + Advanced Techniques',
            'config': {
                'test_months': 6,
                'hold_days': 30,
                'top_n': 10,
                'cost': 0.001,
                'stop_loss': -0.15,
            },
            'v3_changes': 'Added 3 new combo types: fr_x_log_fm, fm_x_log_fr, sqrt_fr_x_sqrt_fm',
            'elapsed_seconds': round(t_total_elapsed, 1),
        },
        'results': all_results,
        'candidates': candidates,
        'passed_candidates': passed_candidates,
        'failed_candidates': failed_candidates,
        'best_candidate': best,
        'summary': {
            'best_combo': best_combo,
            'best_weights': {'fund_ratio': best_weights[0], 'fund_metric': best_weights[1], 'combo': best_weights[2]} if best_weights else None,
            'best_window': best_window,
            'best_train_months': best_train_months,
            'best_sharpe': best['sharpe'] if best else None,
            'best_rank_inversion': best['rank_inversion_passed'] if best else None,
            'v031_baseline_sharpe': 1.161,
            'improvement_pct': round((best['sharpe'] / 1.161 - 1) * 100, 1) if best and best['sharpe'] else None,
        },
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    print(f"End: {datetime.now().isoformat()}")
    print(f"Total elapsed: {t_total_elapsed:.1f}s")
    
    return output


if __name__ == '__main__':
    main()
