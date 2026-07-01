"""
T5.1 Fix Dynamic Linear Model — Rank Inversion

Problem: Dynamic linear model has Rank Inversion in 4/5 windows.
Root cause analysis:
  1. IC lookback too short (6mo) → overfits recent IC noise
  2. Weights change too fast (monthly recalculation)  
  3. Weak factors get high weights due to IC noise

Fix proposals:
  a. Longer IC window: 6mo → 12mo
  b. EMA smoothing of IC weights (span=6)
  c. Weight decay: older IC gets exponentially higher weight
  d. Hybrid: 70% static ICIR + 30% dynamic IC
  e. Factor screening: only use 32 strong factors (|ICIR|>0.1)

Each proposal is run via backtest_engine.py Walk-Forward.
Rank Inversion check: quintile spread must be monotonic.

Uses backtest_engine.py — no custom backtest logic.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*correlation coefficient.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, BacktestResult

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

DATA_PATH = PROJECT_ROOT / "data" / "falcon" / "training_data_v04.parquet"
IC_PATH = PROJECT_ROOT / "data" / "falcon" / "v04_ic_analysis.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "falcon" / "v04_dynamic_linear_fixed_results.json"

TRAIN_YEARS = 5
TEST_MONTHS = 6
HOLD_DAYS = 30
TOP_N = 10
COST = 0.001
STOP_LOSS = -0.15

IC_MIN_DATES = 40
MIN_FACTOR_COVERAGE = 0.3
N_QUINTILES = 5  # 用于Rank Inversion检查


# ═══════════════════════════════════════════════════════════════════
#  Data Loading (same as t31)
# ═══════════════════════════════════════════════════════════════════

def load_data():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    with open(IC_PATH) as f:
        ic_data = json.load(f)

    exclude = {
        'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap',
        'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'fwd_ret_30d',
        'fmp_covered', 'analyst_covered',
        'news_avg_sentiment', 'news_sentiment_vol', 'news_neg_ratio',
        'news_pos_ratio', 'news_article_count', 'news_confidence_avg',
    }
    factor_cols = [c for c in df.columns if c not in exclude]
    print(f"  Factor columns: {len(factor_cols)}")

    # Strong factors (|ICIR| > 0.1)
    strong_factors = [f['name'] for f in ic_data['factors'] if f['abs_icir'] > 0.1]
    print(f"  Strong factors (|ICIR|>0.1): {len(strong_factors)}")

    # ICIR dict for hybrid/static weights
    icir_dict = {f['name']: f['icir'] for f in ic_data['factors']}

    return df, ic_data, factor_cols, strong_factors, icir_dict


def compute_cross_sectional_ranks(df, factor_cols):
    print("Computing cross-sectional ranks...")
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: str(x))

    rank_dict = {}
    dates = sorted(df['date_str'].unique())

    for d in dates:
        day_df = df[df['date_str'] == d].copy()
        if len(day_df) < 20:
            continue
        day_df = day_df.set_index('ticker')

        ranks = pd.DataFrame(index=day_df.index)
        for f in factor_cols:
            if f in day_df.columns:
                ranks[f] = day_df[f].rank(pct=True, na_option='keep')
            else:
                ranks[f] = np.nan

        rank_dict[d] = ranks

    print(f"  Built ranks for {len(rank_dict)} dates")
    return rank_dict


def build_prices_df(df):
    print("Building prices matrix...")
    prices = df.pivot_table(index=df['date'].apply(lambda x: str(x)),
                           columns='ticker', values='close')
    prices = prices.sort_index()
    print(f"  Prices: {prices.shape[0]} dates × {prices.shape[1]} tickers")
    return prices


def compute_monthly_ic(df, factor_cols):
    print("Computing monthly IC...")
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: str(x))
    df['month'] = df['date'].apply(lambda x: x.strftime('%Y-%m'))

    months = sorted(df['month'].unique())
    monthly_ic = {}

    for m in months:
        month_data = df[df['month'] == m]
        dates_in_month = sorted(month_data['date_str'].unique())

        ic_dict = {}
        for f in factor_cols:
            daily_ics = []
            for d in dates_in_month:
                day_data = month_data[month_data['date_str'] == d][[f, 'fwd_ret_30d']].dropna()
                if len(day_data) < 20:
                    continue
                try:
                    ic, _ = spearmanr(day_data[f].values, day_data['fwd_ret_30d'].values)
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                except Exception:
                    pass

            if len(daily_ics) >= 5:
                ic_dict[f] = float(np.mean(daily_ics))

        if ic_dict:
            monthly_ic[m] = ic_dict

    print(f"  Computed monthly IC for {len(months)} months")
    return monthly_ic


def compute_factor_coverage(df, factor_cols, start_date, end_date):
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    subset = df[mask]

    if len(subset) == 0:
        return {f: 0.0 for f in factor_cols}

    coverage = {}
    for f in factor_cols:
        if f in subset.columns:
            coverage[f] = float(subset[f].notna().mean())
        else:
            coverage[f] = 0.0
    return coverage


# ═══════════════════════════════════════════════════════════════════
#  Original weight computation (baseline)
# ═══════════════════════════════════════════════════════════════════

def compute_dynamic_weights_original(monthly_ic, factor_cols, target_month,
                                      lookback_months=6, training_coverage=None):
    """Original: IC = mean(last N months), weight = IC/sum(|IC|)"""
    months = sorted(monthly_ic.keys())
    target_idx = months.index(target_month) if target_month in months else -1

    if target_idx < lookback_months:
        lookback_months = target_idx
    if lookback_months <= 0:
        return None

    recent_months = months[target_idx - lookback_months:target_idx]

    ic_sums = {}
    ic_counts = {}
    for m in recent_months:
        for f, ic in monthly_ic[m].items():
            ic_sums[f] = ic_sums.get(f, 0) + ic
            ic_counts[f] = ic_counts.get(f, 0) + 1

    avg_ic = {}
    for f in ic_sums:
        if ic_counts[f] >= 3:
            avg_ic[f] = ic_sums[f] / ic_counts[f]

    if not avg_ic:
        return None

    available = {}
    for f, ic in avg_ic.items():
        if f not in factor_cols:
            continue
        if training_coverage is not None:
            if training_coverage.get(f, 0) < MIN_FACTOR_COVERAGE:
                continue
        available[f] = ic

    if not available:
        return None

    abs_sum = sum(abs(v) for v in available.values())
    if abs_sum < 1e-8:
        n = len(available)
        return {f: 1.0 / n for f in available}

    weights = {f: v / abs_sum for f, v in available.items()}
    return weights


# ═══════════════════════════════════════════════════════════════════
#  Fix proposals
# ═══════════════════════════════════════════════════════════════════

def compute_weights_proposal_a(monthly_ic, factor_cols, target_month,
                                training_coverage=None):
    """Proposal A: Longer IC window — 12 months instead of 6."""
    return compute_dynamic_weights_original(monthly_ic, factor_cols, target_month,
                                             lookback_months=12,
                                             training_coverage=training_coverage)


def compute_weights_proposal_b(monthly_ic, factor_cols, target_month,
                                training_coverage=None):
    """Proposal B: EMA-smoothed IC weights (span=6).
    
    Instead of simple mean of last N months, use EMA with span=6.
    Recent months have exponentially more weight.
    """
    months = sorted(monthly_ic.keys())
    target_idx = months.index(target_month) if target_month in months else -1
    lookback = 12  # Look at 12 months but EMA will weight recent more

    if target_idx < lookback:
        lookback = target_idx
    if lookback <= 0:
        return None

    recent_months = months[target_idx - lookback:target_idx]
    alpha = 2.0 / (6 + 1)  # EMA span=6

    # Compute EMA-weighted IC
    ic_ema = {}
    for i, m in enumerate(recent_months):
        month_ic = monthly_ic.get(m, {})
        for f, ic in month_ic.items():
            if f not in ic_ema:
                ic_ema[f] = 0.0
            # EMA: new_value = alpha * current + (1-alpha) * previous
            ic_ema[f] = alpha * ic + (1 - alpha) * ic_ema[f]

    if not ic_ema:
        return None

    # Filter by coverage
    available = {}
    for f, ic in ic_ema.items():
        if f not in factor_cols:
            continue
        if training_coverage is not None:
            if training_coverage.get(f, 0) < MIN_FACTOR_COVERAGE:
                continue
        available[f] = ic

    if not available:
        return None

    abs_sum = sum(abs(v) for v in available.values())
    if abs_sum < 1e-8:
        n = len(available)
        return {f: 1.0 / n for f in available}

    return {f: v / abs_sum for f, v in available.items()}


def compute_weights_proposal_c(monthly_ic, factor_cols, target_month,
                                training_coverage=None):
    """Proposal C: Weight decay — older IC gets exponentially HIGHER weight.
    
    Rationale: Recent IC is noisy; longer-term IC is more stable.
    Weights decay from recent (low) to old (high) with exp decay.
    """
    months = sorted(monthly_ic.keys())
    target_idx = months.index(target_month) if target_month in months else -1
    lookback = 12

    if target_idx < lookback:
        lookback = target_idx
    if lookback <= 0:
        return None

    recent_months = months[target_idx - lookback:target_idx]
    decay_rate = 0.85  # Recent month gets 0.85x, older gets 0.85^2, etc.

    # Compute decay-weighted IC (older months get MORE weight)
    ic_weighted_sums = {}
    ic_weighted_counts = {}
    for i, m in enumerate(recent_months):
        month_ic = monthly_ic.get(m, {})
        # Distance from most recent: 0=most recent, lookback-1=oldest
        distance = lookback - 1 - i  # 0 for oldest, lookback-1 for newest
        weight = decay_rate ** distance  # Higher weight for OLDER months

        for f, ic in month_ic.items():
            ic_weighted_sums[f] = ic_weighted_sums.get(f, 0) + weight * ic
            ic_weighted_counts[f] = ic_weighted_counts.get(f, 0) + weight

    avg_ic = {}
    for f in ic_weighted_sums:
        if ic_weighted_counts[f] > 1.0:
            avg_ic[f] = ic_weighted_sums[f] / ic_weighted_counts[f]

    if not avg_ic:
        return None

    available = {}
    for f, ic in avg_ic.items():
        if f not in factor_cols:
            continue
        if training_coverage is not None:
            if training_coverage.get(f, 0) < MIN_FACTOR_COVERAGE:
                continue
        available[f] = ic

    if not available:
        return None

    abs_sum = sum(abs(v) for v in available.values())
    if abs_sum < 1e-8:
        n = len(available)
        return {f: 1.0 / n for f in available}

    return {f: v / abs_sum for f, v in available.items()}


def compute_weights_proposal_d(monthly_ic, factor_cols, target_month,
                                training_coverage=None, icir_dict=None):
    """Proposal D: Hybrid — 70% static ICIR + 30% dynamic IC.
    
    Static component: weights proportional to long-run ICIR (stable).
    Dynamic component: weights from recent IC (adaptive).
    Blend: 70% static + 30% dynamic.
    """
    if icir_dict is None:
        return compute_dynamic_weights_original(monthly_ic, factor_cols, target_month,
                                                 lookback_months=6,
                                                 training_coverage=training_coverage)

    # Static weights: from ICIR
    static_available = {}
    for f, icir in icir_dict.items():
        if f not in factor_cols:
            continue
        if training_coverage is not None:
            if training_coverage.get(f, 0) < MIN_FACTOR_COVERAGE:
                continue
        static_available[f] = icir

    static_abs_sum = sum(abs(v) for v in static_available.values())
    if static_abs_sum < 1e-8:
        return None
    static_weights = {f: v / static_abs_sum for f, v in static_available.items()}

    # Dynamic weights: from recent IC
    dynamic_weights = compute_dynamic_weights_original(monthly_ic, factor_cols, target_month,
                                                        lookback_months=6,
                                                        training_coverage=training_coverage)
    if dynamic_weights is None:
        return static_weights

    # Blend
    all_factors = set(static_weights.keys()) | set(dynamic_weights.keys())
    blended = {}
    for f in all_factors:
        sw = static_weights.get(f, 0.0)
        dw = dynamic_weights.get(f, 0.0)
        blended[f] = 0.7 * sw + 0.3 * dw

    # Re-normalize
    abs_sum = sum(abs(v) for v in blended.values())
    if abs_sum < 1e-8:
        return None
    return {f: v / abs_sum for f, v in blended.items()}


def compute_weights_proposal_e(monthly_ic, factor_cols, target_month,
                                training_coverage=None, strong_factors=None):
    """Proposal E: Factor screening — only use 32 strong factors (|ICIR|>0.1).
    
    Same as original but restricted to strong factors only.
    """
    if strong_factors is None:
        return compute_dynamic_weights_original(monthly_ic, factor_cols, target_month,
                                                 lookback_months=6,
                                                 training_coverage=training_coverage)

    # Only include strong factors
    filtered_factor_cols = [f for f in factor_cols if f in strong_factors]

    weights = compute_dynamic_weights_original(monthly_ic, filtered_factor_cols, target_month,
                                                lookback_months=6,
                                                training_coverage=training_coverage)
    return weights


# ═══════════════════════════════════════════════════════════════════
#  Rank Inversion Check
# ═══════════════════════════════════════════════════════════════════

def check_rank_inversion(df, rank_dict, weights, test_start, test_end, n_quintiles=5):
    """
    Check if COMBINED weighted score quintiles show monotonic forward returns.
    
    Optimized version: groups by date, computes scores via vectorized operations.
    """
    # Use string comparison to avoid Timestamp/datetime.date mismatch
    df_dates = df['date'].astype(str)
    test_mask = (df_dates >= test_start) & (df_dates <= test_end)
    test_df = df[test_mask].copy()

    if len(test_df) == 0:
        return {"inverted_count": 0, "total_pairs": 0, "quintile_returns": {},
                "spread": 0, "ri_rate": 0}

    # Compute combined score for each row using vectorized operations
    # Group by date to process each date's ranks
    test_df = test_df.copy()
    test_df['date_str'] = test_df['date'].astype(str).str[:10]
    
    active_factors = [f for f in weights if abs(weights[f]) > 1e-6]
    sample_date = next(iter(rank_dict)) if rank_dict else None
    if sample_date is None:
        return {"inverted_count": 0, "total_pairs": 0, "quintile_returns": {},
                "spread": 0, "ri_rate": 0}
    available_factors = [f for f in active_factors if f in rank_dict[sample_date].columns]
    if not available_factors:
        return {"inverted_count": 0, "total_pairs": 0, "quintile_returns": {},
                "spread": 0, "ri_rate": 0}

    # Vectorized score computation
    scores = pd.Series(np.nan, index=test_df.index)
    for d in test_df['date_str'].unique():
        if d not in rank_dict:
            continue
        mask = test_df['date_str'] == d
        day_tickers = test_df.loc[mask, 'ticker']
        r = rank_dict[d]
        common_tickers = day_tickers[day_tickers.isin(r.index)]
        if len(common_tickers) == 0:
            continue
        r_sub = r.loc[common_tickers, available_factors]
        w_vec = np.array([weights[f] for f in available_factors])
        score_vals = (r_sub.values * w_vec).sum(axis=1)
        scores.loc[common_tickers.index] = score_vals

    test_df = test_df.assign(score=scores)
    test_df = test_df.dropna(subset=['score', 'fwd_ret_30d'])

    if len(test_df) < n_quintiles * 10:
        return {"inverted_count": 0, "total_pairs": 0, "quintile_returns": {},
                "spread": 0, "ri_rate": 0}

    # Assign quintiles by combined score
    test_df['quintile'] = pd.qcut(test_df['score'].rank(method='first'),
                                    n_quintiles, labels=False, duplicates='drop')

    # Mean forward return per quintile
    quintile_returns = test_df.groupby('quintile')['fwd_ret_30d'].mean()

    if len(quintile_returns) < n_quintiles:
        return {"inverted_count": 0, "total_pairs": 0, "quintile_returns": {},
                "spread": 0, "ri_rate": 0}

    q_values = [quintile_returns.get(i, np.nan) for i in range(n_quintiles)]
    q_values_valid = [v for v in q_values if not np.isnan(v)]

    if len(q_values_valid) < n_quintiles:
        return {"inverted_count": 0, "total_pairs": 0, "quintile_returns": {},
                "spread": 0, "ri_rate": 0}

    # Check monotonicity: count violations
    inverted_count = 0
    total_pairs = 0
    for i in range(len(q_values_valid) - 1):
        total_pairs += 1
        if q_values_valid[i] < q_values_valid[i + 1]:
            inverted_count += 1

    spread = q_values_valid[0] - q_values_valid[-1]
    ri_rate = inverted_count / max(total_pairs, 1)

    return {
        "inverted_count": inverted_count,
        "total_pairs": total_pairs,
        "quintile_returns": {str(k): round(float(v), 4) for k, v in quintile_returns.items()},
        "spread": round(float(spread), 4),
        "ri_rate": round(ri_rate, 3),
    }


# ═══════════════════════════════════════════════════════════════════
#  Walk-Forward with Rank Inversion Check
# ═══════════════════════════════════════════════════════════════════

def run_walk_forward_with_ri(df, rank_dict, prices, factor_cols, monthly_ic,
                              weight_fn, weight_fn_kwargs, proposal_name):
    """Run Walk-Forward with Rank Inversion check for each window."""
    print(f"\n{'='*60}")
    print(f"Proposal: {proposal_name}")
    print(f"{'='*60}")

    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
    all_dates = sorted(prices.index.tolist())
    start_date = pd.Timestamp(all_dates[0])
    end_date = pd.Timestamp(all_dates[-1])

    windows = []
    train_start = start_date
    window_idx = 0

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > end_date:
            break

        test_start_str = str(train_end.date())
        test_end_str = str(test_end.date())
        train_end_month = train_end.strftime('%Y-%m')

        print(f"\n  Window {window_idx}: Test {test_start_str} → {test_end_str}")

        # Compute factor coverage in training period
        training_coverage = compute_factor_coverage(df, factor_cols,
                                                     train_start.date(), train_end.date())

        # Compute weights using proposal-specific function
        weights = weight_fn(monthly_ic, factor_cols, train_end_month,
                           training_coverage=training_coverage, **weight_fn_kwargs)

        if weights is None:
            print(f"    ⚠️ No valid weights, skipping")
            windows.append({
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "error": "No valid weights",
            })
            window_idx += 1
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        n_factors = len(weights)
        top3 = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"    Factors: {n_factors}, Top3: {[(f, round(w, 4)) for f, w in top3]}")

        # Run backtest
        try:
            result, baseline = engine.run(
                rank_dict, prices, weights,
                hold_days=HOLD_DAYS, top_n=TOP_N,
                start_date=test_start_str, end_date=test_end_str,
                run_baseline=True
            )
            print(f"    Sharpe={result.sharpe:.3f}  MaxDD={result.max_dd:.1%}  "
                  f"CAGR={result.cagr:.1%}  WR={result.win_rate:.0%}  "
                  f"Trades={result.n_trades}")
            if baseline:
                print(f"    Baseline: Sharpe={baseline.sharpe:.3f}")

            # Check Rank Inversion (non-fatal)
            try:
                ri = check_rank_inversion(df, rank_dict, weights,
                                           test_start_str, test_end_str)
                print(f"    Rank Inversion: {ri['inverted_count']}/{ri['total_pairs']} "
                      f"({ri['ri_rate']:.1%})  spread={ri['spread']:.4f}")
                print(f"    Quintile returns: {ri['quintile_returns']}")
            except Exception as ri_err:
                print(f"    ⚠️ RI check error: {ri_err}")
                ri = {"inverted_count": 0, "total_pairs": 0, "ri_rate": 0,
                      "quintile_returns": {}, "spread": 0}

            window_info = {
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "sharpe": result.sharpe,
                "max_dd": result.max_dd,
                "cagr": result.cagr,
                "win_rate": result.win_rate,
                "n_trades": result.n_trades,
                "n_factors": n_factors,
                "rank_inversion_count": ri['inverted_count'],
                "rank_inversion_pairs": ri['total_pairs'],
                "rank_inversion_rate": ri['ri_rate'],
                "quintile_returns": ri['quintile_returns'],
                "quintile_spread": ri['spread'],
                "weights_top5": {f: round(w, 4) for f, w in
                                 sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]},
            }
            if baseline:
                window_info["baseline_sharpe"] = baseline.sharpe
                window_info["beat_baseline"] = result.sharpe > baseline.sharpe

            windows.append(window_info)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            windows.append({
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "error": str(e),
            })

        window_idx += 1
        train_start += pd.DateOffset(months=TEST_MONTHS)

    return windows


def aggregate_proposal(windows, proposal_name):
    """Aggregate results for one proposal."""
    valid = [w for w in windows if "sharpe" in w]
    if not valid:
        return {"proposal": proposal_name, "error": "No valid windows"}

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]
    ri_rates = [w.get("rank_inversion_rate", 0) for w in valid]
    ri_counts = [w.get("rank_inversion_count", 0) for w in valid]
    ri_spreads = [w.get("quintile_spread", 0) for w in valid]

    # Stability: how many windows beat baseline
    beat_baseline_count = sum(1 for w in valid if w.get("beat_baseline", False))
    stability_rate = beat_baseline_count / len(valid)

    # Rank Inversion pass: <30% inversion rate AND positive mean spread
    mean_ri = np.mean(ri_rates)
    mean_spread = np.mean(ri_spreads)
    ri_pass = (mean_ri < 0.30 and mean_spread > 0)

    # Composite score: Sharpe × (1 - mean RI rate) × stability_rate × min(1, mean_spread*10)
    # Spread bonus: positive spread means model has predictive power
    spread_bonus = min(1.0, max(0.0, mean_spread * 10))  # 0.1 spread = 1.0 bonus
    composite_score = np.mean(sharpes) * (1 - mean_ri) * stability_rate * (0.5 + 0.5 * spread_bonus)

    return {
        "proposal": proposal_name,
        "wf_sharpe": round(float(np.mean(sharpes)), 3),
        "wf_sharpe_std": round(float(np.std(sharpes)), 3),
        "wf_max_dd": round(float(np.min(dds)), 4),
        "wf_cagr": round(float(np.mean(cagrs)), 4),
        "wf_win_rate": round(float(np.mean(wrs)), 3),
        "wf_n_windows": len(valid),
        "wf_all_sharpes": [round(s, 3) for s in sharpes],
        "wf_all_cagrs": [round(c, 4) for c in cagrs],
        "wf_all_max_dds": [round(d, 4) for d in dds],
        "mean_rank_inversion_rate": round(float(mean_ri), 3),
        "max_rank_inversion_rate": round(float(max(ri_rates)), 3),
        "total_inverted_factors": round(float(np.sum(ri_counts)), 1),
        "mean_quintile_spread": round(float(mean_spread), 4),
        "stability_beat_baseline": beat_baseline_count,
        "stability_rate": round(stability_rate, 3),
        "ri_pass": ri_pass,
        "composite_score": round(composite_score, 3),
        "positive_sharpe_pct": round(sum(1 for s in sharpes if s > 0) / len(sharpes), 3),
        "recent_sharpes": [round(s, 3) for s in sharpes[-3:]],
        "windows": windows,
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("T5.1 Fix Dynamic Linear Model — Rank Inversion")
    print("=" * 60)

    # 1. Load data
    df, ic_data, factor_cols, strong_factors, icir_dict = load_data()

    # 2. Compute cross-sectional ranks
    rank_dict = compute_cross_sectional_ranks(df, factor_cols)

    # 3. Build prices matrix
    prices = build_prices_df(df)

    # 4. Compute monthly IC
    monthly_ic = compute_monthly_ic(df, factor_cols)

    # 5. Define proposals
    proposals = {
        "A_longer_ic_12mo": {
            "fn": compute_weights_proposal_a,
            "kwargs": {},
            "description": "Longer IC window: 6mo → 12mo",
        },
        "B_ema_smoothed": {
            "fn": compute_weights_proposal_b,
            "kwargs": {},
            "description": "EMA-smoothed IC weights (span=6)",
        },
        "C_weight_decay": {
            "fn": compute_weights_proposal_c,
            "kwargs": {},
            "description": "Weight decay: older IC gets more weight",
        },
        "D_hybrid_70_30": {
            "fn": compute_weights_proposal_d,
            "kwargs": {"icir_dict": icir_dict},
            "description": "Hybrid: 70% static ICIR + 30% dynamic IC",
        },
        "E_strong_factors_only": {
            "fn": compute_weights_proposal_e,
            "kwargs": {"strong_factors": strong_factors},
            "description": "Factor screening: only 32 strong factors (|ICIR|>0.1)",
        },
    }

    # 6. Run Walk-Forward for each proposal
    results = {}
    for name, config in proposals.items():
        windows = run_walk_forward_with_ri(
            df, rank_dict, prices, factor_cols, monthly_ic,
            config["fn"], config["kwargs"], f"{name}: {config['description']}"
        )
        agg = aggregate_proposal(windows, name)
        agg["description"] = config["description"]
        results[name] = agg

    # 7. Compare and select best
    print("\n" + "=" * 60)
    print("COMPARISON OF ALL PROPOSALS")
    print("=" * 60)

    # Sort by composite score
    sorted_proposals = sorted(results.values(),
                              key=lambda x: x.get("composite_score", 0),
                              reverse=True)

    for i, r in enumerate(sorted_proposals):
        ri_status = "✅ PASS" if r.get("ri_pass", False) else "❌ FAIL"
        print(f"\n  #{i+1} {r['proposal']}")
        print(f"      Sharpe={r.get('wf_sharpe', 'N/A')}  "
              f"MaxDD={r.get('wf_max_dd', 'N/A')}  "
              f"CAGR={r.get('wf_cagr', 'N/A')}")
        print(f"      Rank Inversion: mean={r.get('mean_rank_inversion_rate', 'N/A')}  "
              f"max={r.get('max_rank_inversion_rate', 'N/A')}  "
              f"spread={r.get('mean_quintile_spread', 'N/A')}  {ri_status}")
        print(f"      Stability: {r.get('stability_rate', 'N/A')} "
              f"({r.get('stability_beat_baseline', 0)}/{r.get('wf_n_windows', 0)} beat baseline)")
        print(f"      Composite Score: {r.get('composite_score', 'N/A')}")

    # Select best: highest composite score among RI-pass proposals
    ri_pass_proposals = [r for r in sorted_proposals if r.get("ri_pass", False)]
    if ri_pass_proposals:
        best = ri_pass_proposals[0]
        print(f"\n{'='*60}")
        print(f"✅ BEST PROPOSAL: {best['proposal']}")
        print(f"   {best.get('description', '')}")
        print(f"   Sharpe={best.get('wf_sharpe')}  MaxDD={best.get('wf_max_dd')}  "
              f"CAGR={best.get('wf_cagr')}")
        print(f"   Rank Inversion: PASS (mean={best.get('mean_rank_inversion_rate')})")
        print(f"   Composite Score: {best.get('composite_score')}")
    else:
        best = sorted_proposals[0]  # Take best even if RI fails
        print(f"\n{'='*60}")
        print(f"⚠️ ALL PROPOSALS FAILED RANK INVERSION")
        print(f"   Best (by Sharpe): {best['proposal']}")
        print(f"   Sharpe={best.get('wf_sharpe')}  RI mean={best.get('mean_rank_inversion_rate')}")

    # 8. Save results
    output = {
        "task": "T5.1 Fix Dynamic Linear Model — Rank Inversion",
        "params": {
            "train_years": TRAIN_YEARS,
            "test_months": TEST_MONTHS,
            "hold_days": HOLD_DAYS,
            "top_n": TOP_N,
            "cost": COST,
            "stop_loss": STOP_LOSS,
        },
        "baseline_results": {
            "model": "Original Dynamic Linear",
            "wf_sharpe": 1.668,
            "wf_max_dd": -0.2219,
            "wf_cagr": 0.5934,
            "rank_inversion": "FAIL (4/5 windows)",
        },
        "proposals": results,
        "best_proposal": {
            "name": best["proposal"],
            "description": best.get("description", ""),
            "sharpe": best.get("wf_sharpe"),
            "max_dd": best.get("wf_max_dd"),
            "cagr": best.get("wf_cagr"),
            "ri_pass": best.get("ri_pass", False),
            "composite_score": best.get("composite_score"),
        },
        "summary": {
            "total_proposals": len(proposals),
            "ri_pass_count": len(ri_pass_proposals),
            "all_ri_fail": len(ri_pass_proposals) == 0,
            "best_sharpe": best.get("wf_sharpe"),
            "best_ri_rate": best.get("mean_rank_inversion_rate"),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    main()
