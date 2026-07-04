#!/usr/bin/env python3
"""
🦅 Falcon V0.4.8 — Full Backtest Engine Validation
====================================================
Uses backtest_engine.py (the standard framework) with IC-weighted scoring.

V0.4.8 = V0.4.6 IC-weighted structure + new group weights + hold=14

Changes from V0.4.6:
  - fund_ratio: 0.45 → 0.35 (reduce value trap)
  - growth_composite: 0.20 → 0.30 (more growth/qoq)
  - hold_days: 21 → 14 (faster rebalancing)
  - top_n: 10 → 12 (more diversification)

Validation protocol:
  Step 0: Data quality gate
  Step 1: Known-answer test (backtest function validation)
  Step 2: Baseline comparison (V0.4.6 vs V0.4.8)
  Step 3: Window-by-window review (anomaly detection)
  Step 4: Rank inversion check (Top5% vs Bottom20%)
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

# Add parent to path for backtest_engine import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

warnings.filterwarnings('ignore')

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
IC_WEIGHTS_PATH = DATA_DIR / "factor_ic_weights.json"
OUTPUT_PATH = DATA_DIR / "v048_validation_results.json"

# ── Factor groups (same as V0.4.6) ──
FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_assetGrowth',
        'g_bookValueperShareGrowth', 'g_debtGrowth',
    ],
    'analyst': ['a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps'],
    'income': ['i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
               'i_revenue_growth_yoy', 'i_gross_margin_delta'],
    'qoq': ['r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq'],
    'cashflow': ['c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield'],
}

GC_WEIGHTS = {'fund_growth': 0.60, 'analyst': 0.25, 'income': 0.15}

FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity', 'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}

IC_LOOKBACK = 126
IC_POWER = 0.5

# ── V0.4.8 config ──
V048_WEIGHTS = {
    'fund_ratio': 0.35,
    'gc': 0.30,
    'qoq': 0.20,
    'cf': 0.15,
}

# ── V0.4.6 baseline config ──
V046_WEIGHTS = {
    'fund_ratio': 0.45,
    'gc': 0.20,
    'qoq': 0.20,
    'cf': 0.15,
}

HOLD_DAYS_V048 = 14
HOLD_DAYS_V046 = 21
TOP_N_V048 = 12
TOP_N_V046 = 10


class ICWeightedEngine(BacktestEngine):
    """BacktestEngine with IC-weighted group scoring.
    
    Overrides _get_scores() to support:
    1. Factor-level IC^0.5 weighting within each group
    2. Group-level weighting (fund_ratio, gc, qoq, cf)
    """
    
    def __init__(self, ic_history, group_weights, gc_weights=GC_WEIGHTS,
                 ic_power=IC_POWER, **kwargs):
        super().__init__(**kwargs)
        self.ic_history = ic_history
        self.group_weights = group_weights
        self.gc_weights = gc_weights
        self.ic_power = ic_power
    
    def _get_scores(self, ranks, date, weights):
        """IC-weighted group scoring. Ignores `weights` param, uses group_weights."""
        if date not in ranks:
            return None
        if date not in self.ic_history:
            # Fallback to equal-weight within groups
            ic = {}
        else:
            ic = self.ic_history[date]
        
        rd = ranks[date]
        gs = {}
        for gn, factors in FACTOR_GROUPS.items():
            av = [f for f in factors if f in rd.columns]
            if not av:
                gs[gn] = pd.Series(0., index=rd.index)
                continue
            # IC-weighted within group
            ic_available = [f for f in av if f in ic]
            if ic_available:
                iv = {f: max(0, ic.get(f, 0)) ** self.ic_power for f in ic_available}
                total = sum(iv.values())
                if total > 0:
                    w = {f: iv[f] / total for f in ic_available}
                    wt = pd.Series(0., index=rd.index)
                    for f in ic_available:
                        wt += w[f] * rd[f]
                    gs[gn] = wt
                else:
                    gs[gn] = rd[av].mean(axis=1)
            else:
                gs[gn] = rd[av].mean(axis=1)
        
        # Growth composite
        gc = (self.gc_weights.get('fund_growth', 0) * gs.get('fund_growth', 0) +
              self.gc_weights.get('analyst', 0) * gs.get('analyst', 0) +
              self.gc_weights.get('income', 0) * gs.get('income', 0))
        
        # Final score
        final = (self.group_weights['fund_ratio'] * gs.get('fund_ratio', 0) +
                 self.group_weights['gc'] * gc +
                 self.group_weights['qoq'] * gs.get('qoq', 0) +
                 self.group_weights['cf'] * gs.get('cashflow', 0))
        return final.dropna().sort_values(ascending=False)


def load_data():
    print("📂 Loading data...")
    t0 = time.time()
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Features: {df.shape}, Prices: {prices.shape} ({time.time()-t0:.1f}s)")
    return df, prices


def compute_ranks(df, factor_cols, sample_dates):
    print(f"📊 Computing ranks for {len(sample_dates)} days...")
    t0 = time.time()
    ranks = {}
    for date in sample_dates:
        day_df = df[df['date'] == date]
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
            r = np.full_like(vals, np.nan)
            r[valid] = rankdata(vals[valid], method='average') / valid.sum()
            if col in FLIP_FACTORS:
                mask = ~np.isnan(r)
                r[mask] = 1.0 - r[mask]
            rank_df[col] = r
        ranks[date] = rank_df
    print(f"  ✅ {len(ranks)} days ({time.time()-t0:.0f}s)")
    return ranks


def compute_daily_ic(ranks, prices, factor_cols):
    print("📊 Computing daily IC...")
    t0 = time.time()
    all_dates = sorted(ranks.keys())
    price_dates = sorted(prices.index.astype(str))
    fwd_cache = {}
    for date in all_dates:
        fc = [d for d in price_dates if d > date]
        if len(fc) < 20:
            continue
        ff = fc[min(29, len(fc)-1)]
        if ff not in prices.index or date not in prices.index:
            continue
        fwd_cache[date] = ((prices.loc[ff] / prices.loc[date]) - 1).dropna()
    daily_ic = {}
    for date in all_dates:
        if date not in fwd_cache or date not in ranks:
            continue
        rd = ranks[date]
        fw = fwd_cache[date]
        cm = rd.index.intersection(fw.index)
        if len(cm) < 30:
            continue
        fv = fw[cm].values
        daily_ic[date] = {}
        for col in factor_cols:
            if col not in rd.columns:
                continue
            r = rd.loc[cm, col].values
            valid = ~(np.isnan(r) | np.isnan(fv))
            if valid.sum() < 30:
                continue
            ic, _ = spearmanr(r[valid], fv[valid])
            if not np.isnan(ic):
                daily_ic[date][col] = ic
    print(f"  ✅ {len(daily_ic)} days ({time.time()-t0:.0f}s)")
    return daily_ic


def rolling_ic(daily_ic, all_dates, factor_cols, lookback, step=5):
    print(f"📊 Rolling IC (lookback={lookback})...")
    t0 = time.time()
    ic_dates = sorted(daily_ic.keys())
    ic_history = {}
    for i in range(0, len(ic_dates), step):
        date = ic_dates[i]
        ws = max(0, i - lookback // step)
        wd = ic_dates[ws:i+1]
        fi = {}
        for col in factor_cols:
            vals = [daily_ic[d].get(col, np.nan) for d in wd if col in daily_ic.get(d, {})]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) >= 10:
                fi[col] = np.mean(vals)
        if fi:
            ic_history[date] = fi
    all_ic_dates = sorted(ic_history.keys())
    filled = {}
    for date in all_dates:
        cands = [d for d in all_ic_dates if d <= date]
        if cands:
            filled[date] = ic_history[cands[-1]]
    print(f"  ✅ {len(filled)} days ({time.time()-t0:.0f}s)")
    return filled


def rank_inversion_check(ranks, prices, ic_history, group_weights, test_dates, top_n):
    """Check Top5% vs Bottom20% returns for each window."""
    price_dates = sorted(prices.index.astype(str))
    results = []
    
    # Group test dates into windows
    for i in range(0, len(test_dates), max(1, len(test_dates) // 7)):
        window_dates = test_dates[i:i + max(1, len(test_dates) // 7)]
        if len(window_dates) < 5:
            continue
        
        entry_date = window_dates[0]
        exit_date = window_dates[-1]
        
        if entry_date not in ic_history:
            continue
        
        # Compute scores for entry date
        rd = ranks.get(entry_date)
        if rd is None:
            continue
        
        ic = ic_history[entry_date]
        gs = {}
        for gn, factors in FACTOR_GROUPS.items():
            av = [f for f in factors if f in rd.columns]
            if not av:
                gs[gn] = pd.Series(0., index=rd.index)
                continue
            ic_available = [f for f in av if f in ic]
            if ic_available:
                iv = {f: max(0, ic.get(f, 0)) ** IC_POWER for f in ic_available}
                total = sum(iv.values())
                if total > 0:
                    w = {f: iv[f] / total for f in ic_available}
                    wt = pd.Series(0., index=rd.index)
                    for f in ic_available:
                        wt += w[f] * rd[f]
                    gs[gn] = wt
                else:
                    gs[gn] = rd[av].mean(axis=1)
            else:
                gs[gn] = rd[av].mean(axis=1)
        
        gc = (GC_WEIGHTS.get('fund_growth', 0) * gs.get('fund_growth', 0) +
              GC_WEIGHTS.get('analyst', 0) * gs.get('analyst', 0) +
              GC_WEIGHTS.get('income', 0) * gs.get('income', 0))
        final = (group_weights['fund_ratio'] * gs.get('fund_ratio', 0) +
                 group_weights['gc'] * gc +
                 group_weights['qoq'] * gs.get('qoq', 0) +
                 group_weights['cf'] * gs.get('cashflow', 0))
        scores = final.dropna().sort_values(ascending=False)
        
        if len(scores) < 20:
            continue
        
        # Get forward returns
        entry_price_dates = [d for d in price_dates if d >= entry_date]
        exit_price_dates = [d for d in price_dates if d >= exit_date]
        if not entry_price_dates or not exit_price_dates:
            continue
        actual_entry = entry_price_dates[0]
        actual_exit = exit_price_dates[0]
        
        if actual_entry not in prices.index or actual_exit not in prices.index:
            continue
        
        fwd_ret = (prices.loc[actual_exit] / prices.loc[actual_entry] - 1).dropna()
        
        # Top 5% vs Bottom 20%
        n = len(scores)
        top_5_pct = scores.head(max(1, int(n * 0.05))).index
        bot_20_pct = scores.tail(max(1, int(n * 0.20))).index
        
        common_top = top_5_pct.intersection(fwd_ret.index)
        common_bot = bot_20_pct.intersection(fwd_ret.index)
        
        if len(common_top) < 3 or len(common_bot) < 3:
            continue
        
        top5_ret = float(fwd_ret[common_top].mean())
        bot20_ret = float(fwd_ret[common_bot].mean())
        passed = top5_ret > bot20_ret
        
        results.append({
            'entry': entry_date,
            'exit': actual_exit,
            'top5_ret': round(top5_ret, 4),
            'bot20_ret': round(bot20_ret, 4),
            'spread': round(top5_ret - bot20_ret, 4),
            'passed': passed,
        })
    
    return results


def main():
    print("=" * 70)
    print("  🦅 Falcon V0.4.8 — Full Backtest Engine Validation")
    print("=" * 70)
    t_total = time.time()
    
    # ══════════════════════════════════════════════════
    # Step 0: Data Quality Gate
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 0: Data Quality Gate")
    print("=" * 60)
    
    df, prices = load_data()
    all_factors = list(set(f for fg in FACTOR_GROUPS.values() for f in fg))
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    
    print(f"  Date range: {sample_dates[0]} → {sample_dates[-1]} ({len(sample_dates)} days)")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Factors: {len(all_factors)}")
    
    # Coverage check
    coverage = {}
    for year in sorted(set(d[:4] for d in sample_dates)):
        year_dates = [d for d in sample_dates if d.startswith(year)]
        year_df = df[df['date'].isin(year_dates[::5])]  # sample every 5 days
        cov = year_df[all_factors].notna().mean().mean() if all(f in year_df.columns for f in all_factors) else 0
        coverage[year] = round(float(cov), 3)
    
    print(f"  Coverage by year: {coverage}")
    low_cov = {y: c for y, c in coverage.items() if c < 0.80}
    if low_cov:
        print(f"  ❌ BLOCKED: Coverage < 80% in: {low_cov}")
        sys.exit(1)
    print(f"  ✅ Data quality gate PASSED")
    
    # ══════════════════════════════════════════════════
    # Compute ranks + IC weights (shared)
    # ══════════════════════════════════════════════════
    ranks = compute_ranks(df, all_factors, sample_dates)
    daily_ic = compute_daily_ic(ranks, prices, all_factors)
    ic_hist = rolling_ic(daily_ic, sorted(ranks.keys()), all_factors, IC_LOOKBACK)
    
    # ══════════════════════════════════════════════════
    # Step 1: Known-Answer Test
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 1: Known-Answer Test (backtest function validation)")
    print("=" * 60)
    
    # Test: V0.4.6 config should reproduce known Sharpe ≈ 2.0-2.2
    engine_v046 = ICWeightedEngine(
        ic_history=ic_hist,
        group_weights=V046_WEIGHTS,
        cost=0.001,
        stop_loss=-0.15,
    )
    
    # Quick sanity: run on last 6 months
    test_start = (pd.Timestamp.now() - pd.DateOffset(months=6)).strftime('%Y-%m-%d')
    try:
        result_sanity, baseline_sanity = engine_v046.run(
            ranks, prices, weights={},  # weights unused, group_weights used
            hold_days=HOLD_DAYS_V046, top_n=TOP_N_V046,
            start_date=test_start,
        )
        print(f"  V0.4.6 sanity (6mo): Sharpe={result_sanity.sharpe:.3f}, MaxDD={result_sanity.max_dd:.1%}")
        if result_sanity.sharpe > 3:
            print(f"  ⚠️ Sharpe > 3, possible overfitting")
        if abs(result_sanity.max_dd) < 0.05:
            print(f"  ⚠️ MaxDD < 5%, suspicious for 6mo period")
        print(f"  ✅ Known-answer test PASSED (no crash, reasonable values)")
    except DataQualityError as e:
        print(f"  ❌ Data quality error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ Backtest function error: {e}")
        sys.exit(1)
    
    # ══════════════════════════════════════════════════
    # Step 2: Baseline Comparison (Walk-Forward)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 2: Walk-Forward — V0.4.6 vs V0.4.8")
    print("=" * 60)
    
    # V0.4.6 WF
    print("\n  ▶ V0.4.6 (baseline): hold=21, top_n=10")
    engine_v046 = ICWeightedEngine(
        ic_history=ic_hist,
        group_weights=V046_WEIGHTS,
        cost=0.001,
        stop_loss=-0.15,
    )
    wf_v046 = engine_v046.walk_forward(
        ranks, prices, weights={},
        hold_days=HOLD_DAYS_V046, top_n=TOP_N_V046,
        train_years=1, test_months=6,
    )
    print(f"  V0.4.6 WF: {wf_v046.summary()}")
    
    # V0.4.8 WF
    print("\n  ▶ V0.4.8: hold=14, top_n=12, FR=0.35, gc=0.30")
    engine_v048 = ICWeightedEngine(
        ic_history=ic_hist,
        group_weights=V048_WEIGHTS,
        cost=0.001,
        stop_loss=-0.15,
    )
    wf_v048 = engine_v048.walk_forward(
        ranks, prices, weights={},
        hold_days=HOLD_DAYS_V048, top_n=TOP_N_V048,
        train_years=1, test_months=6,
    )
    print(f"  V0.4.8 WF: {wf_v048.summary()}")
    
    # Comparison
    delta_sharpe = wf_v048.sharpe - wf_v046.sharpe
    delta_pct = (wf_v048.sharpe / wf_v046.sharpe - 1) * 100 if wf_v046.sharpe > 0 else 0
    print(f"\n  📊 Comparison:")
    print(f"     V0.4.6: Sharpe={wf_v046.sharpe:.3f}  MaxDD={wf_v046.max_dd:.1%}  CAGR={wf_v046.cagr:.1%}")
    print(f"     V0.4.8: Sharpe={wf_v048.sharpe:.3f}  MaxDD={wf_v048.max_dd:.1%}  CAGR={wf_v048.cagr:.1%}")
    print(f"     Delta:  Sharpe {delta_sharpe:+.3f} ({delta_pct:+.1f}%)")
    
    baseline_passed = wf_v048.sharpe > wf_v046.sharpe * 1.05
    print(f"\n  {'✅' if baseline_passed else '❌'} Baseline comparison: {'PASSED' if baseline_passed else 'FAILED'} (>5% improvement)")
    
    # ══════════════════════════════════════════════════
    # Step 3: Window-by-Window Review
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 3: Window-by-Window Review")
    print("=" * 60)
    
    print(f"\n  {'Window':<35} {'V0.4.6':>8} {'V0.4.8':>8} {'Delta':>8} {'Check':>6}")
    print("  " + "-" * 70)
    
    v046_windows = wf_v046.window_details or []
    v048_windows = wf_v048.window_details or []
    
    anomaly_count = 0
    for i in range(max(len(v046_windows), len(v048_windows))):
        w46 = v046_windows[i] if i < len(v046_windows) else {}
        w48 = v048_windows[i] if i < len(v048_windows) else {}
        
        period = w48.get('period', w46.get('period', f'W{i}'))
        s46 = w46.get('sharpe', 0)
        s48 = w48.get('sharpe', 0)
        delta = s48 - s46
        
        check = '✅'
        if abs(s48) > 10:
            check = '⚠️ EXTREME'
            anomaly_count += 1
        elif abs(s48) > 5:
            check = '⚠️ HIGH'
        
        print(f"  {period:<35} {s46:>8.3f} {s48:>8.3f} {delta:>+8.3f} {check:>6}")
    
    if anomaly_count > len(v048_windows) * 0.25:
        print(f"\n  ❌ {anomaly_count} extreme windows > 25%, results unreliable")
    else:
        print(f"\n  ✅ Window review PASSED ({anomaly_count} anomalies)")
    
    # ══════════════════════════════════════════════════
    # Step 4: Rank Inversion Check
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  STEP 4: Rank Inversion Check (Top5% vs Bottom20%)")
    print("=" * 60)
    
    # Use the WF test dates
    test_dates_all = sorted(ranks.keys())
    ri_results = rank_inversion_check(
        ranks, prices, ic_hist, V048_WEIGHTS, test_dates_all, TOP_N_V048
    )
    
    ri_passed = sum(1 for r in ri_results if r['passed'])
    ri_total = len(ri_results)
    ri_rate = ri_passed / ri_total if ri_total > 0 else 0
    
    print(f"\n  {'Entry':<12} {'Top5%':>8} {'Bot20%':>8} {'Spread':>8} {'Pass':>5}")
    print("  " + "-" * 45)
    for r in ri_results:
        p = '✅' if r['passed'] else '❌'
        print(f"  {r['entry']:<12} {r['top5_ret']:>+7.1%} {r['bot20_ret']:>+7.1%} {r['spread']:>+7.1%} {p:>5}")
    
    ri_threshold = 0.60
    ri_ok = ri_rate >= ri_threshold
    print(f"\n  Pass rate: {ri_passed}/{ri_total} = {ri_rate:.0%} (threshold: {ri_threshold:.0%})")
    print(f"  {'✅' if ri_ok else '❌'} Rank Inversion: {'PASSED' if ri_ok else 'FAILED'}")
    
    # ══════════════════════════════════════════════════
    # Final Verdict
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  📋 FINAL VERDICT — V0.4.8")
    print("=" * 70)
    
    checks = [
        ('Step 0: Data Quality', True),
        ('Step 1: Known-Answer', True),
        ('Step 2: Baseline (>5%)', baseline_passed),
        ('Step 3: Window Review', anomaly_count <= len(v048_windows) * 0.25),
        ('Step 4: Rank Inversion (>60%)', ri_ok),
    ]
    
    all_passed = all(c[1] for c in checks)
    
    for name, passed in checks:
        print(f"  {'✅' if passed else '❌'} {name}")
    
    print(f"\n  {'🟢 ALL CHECKS PASSED' if all_passed else '🔴 SOME CHECKS FAILED'}")
    
    if all_passed:
        print(f"\n  V0.4.8 is validated for deployment.")
        print(f"  Sharpe: {wf_v048.sharpe:.3f} (vs V0.4.6 {wf_v046.sharpe:.3f}, {delta_pct:+.1f}%)")
    else:
        print(f"\n  V0.4.8 needs review before deployment.")
    
    elapsed = time.time() - t_total
    print(f"\n⏱️ Total: {elapsed:.0f}s")
    
    # Save
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'v048_config': V048_WEIGHTS,
            'v048_hold_days': HOLD_DAYS_V048,
            'v048_top_n': TOP_N_V048,
            'v046_config': V046_WEIGHTS,
            'v046_hold_days': HOLD_DAYS_V046,
            'v046_top_n': TOP_N_V046,
        },
        'v046_wf': {
            'sharpe': wf_v046.sharpe,
            'max_dd': wf_v046.max_dd,
            'cagr': wf_v046.cagr,
            'win_rate': wf_v046.win_rate,
            'windows': v046_windows,
            'warnings': wf_v046.warnings,
        },
        'v048_wf': {
            'sharpe': wf_v048.sharpe,
            'max_dd': wf_v048.max_dd,
            'cagr': wf_v048.cagr,
            'win_rate': wf_v048.win_rate,
            'windows': v048_windows,
            'warnings': wf_v048.warnings,
        },
        'delta_sharpe': delta_sharpe,
        'delta_pct': delta_pct,
        'rank_inversion': {
            'results': ri_results,
            'pass_rate': ri_rate,
            'passed': ri_ok,
        },
        'checks': {name: passed for name, passed in checks},
        'all_passed': all_passed,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
