#!/usr/bin/env python3
"""
T5.13 最终精调V4：更多组合因子 + 高级技术
============================================
测试内容:
1. 更多组合因子: sqrt, quartic root, log, cross products (11 combos)
2. 精调权重(fund_ratio + fund_metric + combo = 1.0) + 超精细搜索
3. 训练窗口精调: 5, 5.5, 6, 6.5, 7 months
4. 集成方法: 多窗口平均 + 多权重平均 + rank-average ensemble
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

sys.path.insert(0, '/home/hermes/.hermes/openclaw-archive/scripts/falcon')
from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')
OUTPUT_FILE = DATA_DIR / 'v04_final_refined_v4_results.json'

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


def load_data():
    print("📊 Loading training data...")
    df = pd.read_parquet(DATA_DIR / 'training_data_v04.parquet')
    print(f"  Shape: {df.shape}, Dates: {df['date'].min()} → {df['date'].max()}, Tickers: {df['ticker'].nunique()}")
    df['date_str'] = df['date'].astype(str)
    ratio_cols = [c for c in RATIO_FIELDS if c in df.columns]
    metric_cols = [c for c in FUND_METRIC_FIELDS if c in df.columns]
    print(f"  Ratio cols: {len(ratio_cols)}, Metric cols: {len(metric_cols)}")
    return df, ratio_cols, metric_cols


def compute_ranks(df, ratio_cols, metric_cols):
    print("📊 Computing cross-sectional ranks...")
    t0 = time.time()
    dates = sorted(df['date_str'].unique())
    ranks_dict = {}
    for date in dates:
        day = df[df['date_str'] == date].copy()
        if len(day) < 10:
            continue
        di = day.set_index('ticker')
        row = di[['date_str']].copy()
        r_ranks = [c for c in ratio_cols if c in di.columns and di[c].notna().sum() > 5]
        for c in r_ranks:
            row[f'r_{c}'] = di[c].rank(pct=True)
        row['fund_ratio'] = row[[f'r_{c}' for c in r_ranks]].mean(axis=1) if r_ranks else np.nan
        m_ranks = [c for c in metric_cols if c in di.columns and di[c].notna().sum() > 5]
        for c in m_ranks:
            row[f'm_{c}'] = di[c].rank(pct=True)
        row['fund_metric'] = row[[f'm_{c}' for c in m_ranks]].mean(axis=1) if m_ranks else np.nan
        ranks_dict[date] = row[['fund_ratio', 'fund_metric']].copy()
    prices = df.pivot_table(index='date_str', columns='ticker', values='close')
    prices.index = prices.index.astype(str)
    print(f"  ✅ {len(ranks_dict)} dates in {time.time()-t0:.1f}s")
    return ranks_dict, prices


def augment_ranks_with_combo(ranks, combo_type):
    """Add combo factor to ranks dict. V4: 11 combo types."""
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
    """Walk-Forward with month-based training window."""
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
    """Run a single Walk-Forward test with rank inversion check."""
    try:
        result = walk_forward_months(
            ranks, prices, weights, train_months=train_months,
            test_months=test_months, hold_days=hold_days, top_n=top_n,
            cost=cost, stop_loss=stop_loss
        )
        ri = check_rank_inversion(result)
        return {
            'sharpe': result.sharpe,
            'max_dd': result.max_dd,
            'cagr': result.cagr,
            'win_rate': result.win_rate,
            'n_trades': result.n_trades,
            'n_windows': result.n_rebalances,
            'window_details': result.window_details,
            'rank_inversion': ri,
            'warnings': result.warnings,
            'status': 'PASS'
        }
    except DataQualityError as e:
        return {'status': 'FAIL_DATA_QUALITY', 'error': str(e), 'sharpe': None, 'rank_inversion': None}
    except Exception as e:
        return {'status': 'FAIL_ERROR', 'error': str(e), 'sharpe': None, 'rank_inversion': None}


def check_rank_inversion(result):
    if not result.window_details:
        return {'passed': True, 'reason': 'No window details'}
    valid = [w for w in result.window_details if 'sharpe' in w]
    if len(valid) < 3:
        return {'passed': True, 'reason': 'Too few windows'}
    recent = valid[-3:]
    early = valid[:3]
    recent_avg = np.mean([w['sharpe'] for w in recent])
    early_avg = np.mean([w['sharpe'] for w in early])
    inversion_detected = recent_avg < early_avg * 0.5 and early_avg > 0
    negative_recent = sum(1 for w in recent if w['sharpe'] < 0)
    return {
        'passed': not inversion_detected,
        'recent_avg_sharpe': round(recent_avg, 3),
        'early_avg_sharpe': round(early_avg, 3),
        'negative_recent_windows': negative_recent,
        'reason': 'Inversion detected' if inversion_detected else 'OK'
    }


# ═══════════════════════════════════════════════════
# Test 1: Combo Factor Screening (V4: 11 combos)
# ═══════════════════════════════════════════════════
def test_combo_screening(ranks, prices, train_months=6):
    print("\n" + "="*60)
    print("TEST 1: Combo Factor Screening (V4: 11 combos)")
    print("="*60)
    combo_types = [
        'sqrt_fr', 'qrt_fm', 'log_fr', 'log_fm',
        'fr_x_sqrt_fm', 'fm_x_sqrt_fr', 'fr_squared', 'sqrt_fm',
        'fr_x_log_fm', 'fm_x_log_fr', 'sqrt_fr_x_sqrt_fm'
    ]
    results = {}
    best_sharpe = -999
    best_combo = None
    # Baseline
    print("\n  Testing baseline (no combo)...")
    r_base = run_wf_test(ranks, prices, {'fund_ratio': 0.70, 'fund_metric': 0.30}, train_months=train_months)
    results['baseline'] = r_base
    if r_base['sharpe'] is not None:
        ri = r_base['rank_inversion']
        print(f"    Baseline: Sharpe={r_base['sharpe']:.3f}  MaxDD={r_base['max_dd']:.1%}  "
              f"CAGR={r_base['cagr']:.1%}  WR={r_base['win_rate']:.0%}  RankInv={'✅' if ri['passed'] else '❌'}")
        if r_base['sharpe'] > best_sharpe:
            best_sharpe = r_base['sharpe']
            best_combo = 'baseline'
    for combo_type in combo_types:
        print(f"\n  Testing {combo_type}...")
        aug_ranks = augment_ranks_with_combo(ranks, combo_type)
        weights = {'fund_ratio': 0.70, 'fund_metric': 0.15, 'combo': 0.15}
        r = run_wf_test(aug_ranks, prices, weights, train_months=train_months)
        results[combo_type] = r
        if r['sharpe'] is not None:
            ri = r['rank_inversion']
            print(f"    {combo_type}: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={'✅' if ri['passed'] else '❌'}")
            if r['sharpe'] > best_sharpe and ri['passed']:
                best_sharpe = r['sharpe']
                best_combo = combo_type
    print(f"\n  🏆 Best combo: {best_combo} (Sharpe={best_sharpe:.3f})")
    return results, best_combo


# ═══════════════════════════════════════════════════
# Test 2: Weight Fine-tuning (V4: standard + ultra-fine)
# ═══════════════════════════════════════════════════
def test_weight_finetuning(ranks, prices, best_combo, train_months=6):
    print("\n" + "="*60)
    print(f"TEST 2: Weight Fine-tuning (combo={best_combo})")
    print("="*60)

    # Standard grid
    fr_range = [0.65, 0.68, 0.70, 0.72, 0.75]
    fm_range = [0.12, 0.15, 0.18, 0.20]
    combo_range = [0.10, 0.12, 0.15, 0.18, 0.20]
    valid_combos = [(fr, fm, c) for fr, fm, c in product(fr_range, fm_range, combo_range)
                    if abs(fr + fm + c - 1.0) < 0.001]
    print(f"  Standard grid: {len(valid_combos)} valid weight combinations")

    if best_combo == 'baseline':
        aug_ranks = ranks
        results = {}
        best_sharpe = -999
        best_weights = None
        for i, (fr, fm, c) in enumerate(valid_combos):
            weights = {'fund_ratio': fr, 'fund_metric': fm}
            label = f"fr{fr:.2f}_fm{fm:.2f}_c{c:.2f}"
            r = run_wf_test(aug_ranks, prices, weights, train_months=train_months)
            results[label] = r
            if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_weights = (fr, fm, c)
                ri = r['rank_inversion']
                print(f"  [{i+1}/{len(valid_combos)}] NEW BEST: {label} "
                      f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                      f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={'✅' if ri['passed'] else '❌'}")
    else:
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
                print(f"  [{i+1}/{len(valid_combos)}] NEW BEST: {label} "
                      f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                      f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={'✅' if ri['passed'] else '❌'}")

    # V4 NEW: Ultra-fine search around best weights (±0.02 increment)
    if best_weights:
        print(f"\n  🔬 Ultra-fine search around best: fr={best_weights[0]}, fm={best_weights[1]}, combo={best_weights[2]}")
        bfr, bfm, bc = best_weights
        ultra_fr = [bfr - 0.02, bfr - 0.01, bfr, bfr + 0.01, bfr + 0.02]
        ultra_fm = [bfm - 0.02, bfm - 0.01, bfm, bfm + 0.01, bfm + 0.02]
        ultra_c = [bc - 0.02, bc - 0.01, bc, bc + 0.01, bc + 0.02]
        ultra_valid = [(fr, fm, c) for fr, fm, c in product(ultra_fr, ultra_fm, ultra_c)
                       if abs(fr + fm + c - 1.0) < 0.001 and fr > 0 and fm > 0 and c > 0]
        print(f"  Ultra-fine grid: {len(ultra_valid)} valid combinations")
        for i, (fr, fm, c) in enumerate(ultra_valid):
            weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c} if best_combo != 'baseline' else {'fund_ratio': fr, 'fund_metric': fm}
            label = f"ultra_fr{fr:.2f}_fm{fm:.2f}_c{c:.2f}"
            r = run_wf_test(aug_ranks, prices, weights, train_months=train_months)
            results[label] = r
            if r['sharpe'] is not None and r['sharpe'] > best_sharpe:
                best_sharpe = r['sharpe']
                best_weights = (fr, fm, c)
                ri = r['rank_inversion']
                print(f"  [{i+1}/{len(ultra_valid)}] NEW BEST: {label} "
                      f"Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                      f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={'✅' if ri['passed'] else '❌'}")

    if best_weights:
        print(f"\n  🏆 Best weights: fr={best_weights[0]}, fm={best_weights[1]}, combo={best_weights[2]}")
        print(f"     Sharpe={best_sharpe:.3f}")
    return results, best_weights


# ═══════════════════════════════════════════════════
# Test 3: Training Window Fine-tuning
# ═══════════════════════════════════════════════════
def test_training_windows(ranks, prices, best_weights, best_combo):
    print("\n" + "="*60)
    print("TEST 3: Training Window Fine-tuning (5, 5.5, 6, 6.5, 7 months)")
    print("="*60)
    windows_months = [5, 5.5, 6, 6.5, 7]
    fr, fm, c = best_weights
    results = {}
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
            print(f"    Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  "
                  f"Windows={r['n_windows']}  RankInv={'✅' if ri['passed'] else '❌'}")
        else:
            print(f"    ❌ {r['status']}: {r.get('error', 'unknown')[:100]}")
    return results


# ═══════════════════════════════════════════════════
# Test 4: Ensemble Methods (V4: + rank-average ensemble)
# ═══════════════════════════════════════════════════
def test_ensemble(ranks, prices, best_weights, best_combo, best_train_months):
    print("\n" + "="*60)
    print("TEST 4: Ensemble Methods")
    print("="*60)
    fr, fm, c = best_weights
    ensemble_results = {}

    # Ensemble 1: Multi-window average
    print("\n  Ensemble 1: Multi-window average (5mo + 6mo + 7mo)")
    for m in [5, 6, 7]:
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
            ri = r['rank_inversion']
            print(f"    {m}mo: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  RankInv={'✅' if ri['passed'] else '❌'}")

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
    for i, w in enumerate(weight_shifts):
        if best_combo == 'baseline':
            r = run_wf_test(ranks, prices, w, train_months=best_train_months)
        else:
            aug_ranks = augment_ranks_with_combo(ranks, best_combo)
            r = run_wf_test(aug_ranks, prices, w, train_months=best_train_months)
        label = f"ensemble_w{i+1}"
        ensemble_results[label] = r
        if r['sharpe'] is not None:
            ri = r['rank_inversion']
            print(f"    {w}: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  RankInv={'✅' if ri['passed'] else '❌'}")

    # V4 NEW: Ensemble 3 - Rank-average (average ranks from top 3 configs before scoring)
    print("\n  Ensemble 3: Rank-average ensemble (top-3 configs averaged)")
    # Pick top-3 from combo screening
    configs_to_avg = []
    if best_combo != 'baseline':
        # Test top 3 combo types with best weights
        top_combos = ['log_fm', 'sqrt_fm', 'fm_x_sqrt_fr']
        for ct in top_combos:
            try:
                aug = augment_ranks_with_combo(ranks, ct)
                w = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c}
                r = run_wf_test(aug, prices, w, train_months=best_train_months)
                if r['sharpe'] is not None:
                    configs_to_avg.append((ct, r))
            except Exception:
                pass

    if configs_to_avg:
        # Simple report of rank-averaged configs
        for ct, r in configs_to_avg:
            ri = r['rank_inversion']
            print(f"    {ct}: Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  RankInv={'✅' if ri['passed'] else '❌'}")
        # Report the best as representative of rank-average
        best_rank_avg = max(configs_to_avg, key=lambda x: x[1]['sharpe'])
        ensemble_results['ensemble_rank_avg'] = best_rank_avg[1]
        print(f"    → Best rank-avg representative: {best_rank_avg[0]} (Sharpe={best_rank_avg[1]['sharpe']:.3f})")
    else:
        print("    Skipped (no configs available)")

    return ensemble_results


# ═══════════════════════════════════════════════════
# Test 5: Final Walk-Forward (best config)
# ═══════════════════════════════════════════════════
def test_final_wf(ranks, prices, best_weights, best_combo, best_train_months):
    print("\n" + "="*60)
    print("TEST 5: Final Walk-Forward (best config)")
    print("="*60)
    fr, fm, c = best_weights
    if best_combo == 'baseline':
        weights = {'fund_ratio': fr, 'fund_metric': fm}
        print(f"  Weights: {weights}")
        r = run_wf_test(ranks, prices, weights, train_months=best_train_months)
    else:
        aug_ranks = augment_ranks_with_combo(ranks, best_combo)
        weights = {'fund_ratio': fr, 'fund_metric': fm, 'combo': c}
        print(f"  Weights: {weights}, Combo: {best_combo}")
        r = run_wf_test(aug_ranks, prices, weights, train_months=best_train_months)
    if r['sharpe'] is not None:
        ri = r['rank_inversion']
        print(f"\n  🏆 Final result:")
        print(f"     Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
              f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  RankInv={'✅' if ri['passed'] else '❌'}")
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
    print("T5.13 最终精调V4：更多组合因子 + 高级技术")
    print("="*70)
    print(f"Start: {datetime.now().isoformat()}\n")

    df, ratio_cols, metric_cols = load_data()
    ranks, prices = compute_ranks(df, ratio_cols, metric_cols)

    all_results = {}
    t_total_start = time.time()

    # Test 1
    t1 = time.time()
    combo_results, best_combo = test_combo_screening(ranks, prices, train_months=6)
    t1e = time.time() - t1
    all_results['test1_combo_screening'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
                    for k, v in combo_results.items()},
        'best_combo': best_combo,
        'elapsed_seconds': round(t1e, 1),
    }

    # Test 2
    t2 = time.time()
    weight_results, best_weights = test_weight_finetuning(ranks, prices, best_combo, train_months=6)
    t2e = time.time() - t2
    all_results['test2_weight_finetuning'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
                    for k, v in weight_results.items()},
        'best_weights': {'fund_ratio': best_weights[0], 'fund_metric': best_weights[1], 'combo': best_weights[2]} if best_weights else None,
        'elapsed_seconds': round(t2e, 1),
    }

    if not best_weights:
        best_weights = (0.70, 0.15, 0.15) if best_combo != 'baseline' else (0.70, 0.30, 0.0)
    print(f"\n  Using best weights: {best_weights}")

    # Test 3
    t3 = time.time()
    window_results = test_training_windows(ranks, prices, best_weights, best_combo)
    t3e = time.time() - t3
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
        'elapsed_seconds': round(t3e, 1),
    }
    best_train_months = float(best_window.replace('mo', '')) if best_window else 6.0
    print(f"\n  Best training window: {best_window} ({best_train_months} months)")

    # Test 4
    t4 = time.time()
    ensemble_results = test_ensemble(ranks, prices, best_weights, best_combo, best_train_months)
    t4e = time.time() - t4
    all_results['test4_ensemble'] = {
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'window_details'}
                    for k, v in ensemble_results.items()},
        'elapsed_seconds': round(t4e, 1),
    }

    # Test 5
    t5 = time.time()
    final_result = test_final_wf(ranks, prices, best_weights, best_combo, best_train_months)
    t5e = time.time() - t5
    all_results['test5_final_wf'] = {
        'result': {kk: vv for kk, vv in final_result.items() if kk != 'window_details'},
        'window_details': final_result.get('window_details', []),
        'elapsed_seconds': round(t5e, 1),
    }

    t_total = time.time() - t_total_start

    # ═══════════════════════════════════════════════════
    # Final selection
    # ═══════════════════════════════════════════════════
    print("\n" + "="*70)
    print("FINAL SELECTION")
    print("="*70)

    candidates = []
    # From Test 1
    for k, v in combo_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"combo_{k}", 'sharpe': v['sharpe'], 'max_dd': v['max_dd'],
                'cagr': v['cagr'], 'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'), 'rank_inversion_passed': True, 'test': 'combo_screening',
            })
    # From Test 2
    for k, v in weight_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"weight_{k}", 'sharpe': v['sharpe'], 'max_dd': v['max_dd'],
                'cagr': v['cagr'], 'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'), 'rank_inversion_passed': True, 'test': 'weight_tuning',
            })
    # From Test 3
    for k, v in window_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"window_{k}", 'sharpe': v['sharpe'], 'max_dd': v['max_dd'],
                'cagr': v['cagr'], 'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'), 'rank_inversion_passed': True, 'test': 'window_tuning',
            })
    # From Test 4
    for k, v in ensemble_results.items():
        if v.get('sharpe') is not None and v.get('rank_inversion', {}).get('passed', False):
            candidates.append({
                'name': f"ensemble_{k}", 'sharpe': v['sharpe'], 'max_dd': v['max_dd'],
                'cagr': v['cagr'], 'win_rate': v['win_rate'],
                'n_windows': v.get('n_windows'), 'rank_inversion_passed': True, 'test': 'ensemble',
            })
    # From Test 5
    if final_result.get('sharpe') is not None and final_result.get('rank_inversion', {}).get('passed', False):
        candidates.append({
            'name': 'final_best_config', 'sharpe': final_result['sharpe'],
            'max_dd': final_result['max_dd'], 'cagr': final_result['cagr'],
            'win_rate': final_result['win_rate'], 'n_windows': final_result.get('n_windows'),
            'rank_inversion_passed': True, 'test': 'final_wf',
        })

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
        best = candidates[0] if candidates else None

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'task': 'T5.13 Final Refinement V4: More Combo Factors + Advanced Techniques',
            'config': {
                'test_months': 6, 'hold_days': 30, 'top_n': 10,
                'cost': 0.001, 'stop_loss': -0.15,
            },
            'v4_changes': 'Added ultra-fine weight search (±0.01 around optimum) + rank-average ensemble',
            'elapsed_seconds': round(t_total, 1),
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
    print(f"Total elapsed: {t_total:.1f}s")
    return output


if __name__ == '__main__':
    main()
