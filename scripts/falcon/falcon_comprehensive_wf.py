#!/usr/bin/env python3
"""
🦅 Falcon Comprehensive WF Optimization
==========================================
Tests all viable optimization paths against V0.4.6 baseline.

Configs:
  1. V0.4.6 baseline (FR=0.45, gc=0.20, qoq=0.20, cf=0.15)
  2. V0.4.9d (FR=0.40, gc=0.25, qoq=0.20, cf=0.15)
  3. V0.4.7e (FR=0.35, gc=0.20, qoq=0.25, cf=0.20)
  4. Industry neutralization (max 2 per sector)
  5. Dynamic position sizing (score dispersion)
  6. Trailing stop (-10%, -15%)

Walk-Forward: train=12mo, test=6mo, weekly rebalance, hold=7d
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')

# ── Paths ──
WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "comprehensive_wf_results.json"

# ── Factor Groups (same as V0.4.6) ──
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

# ── Configurations to test ──
CONFIGS = {
    'V0.4.6_baseline': {'fund_ratio': 0.45, 'gc': 0.20, 'qoq': 0.20, 'cf': 0.15},
    'V0.4.9d':         {'fund_ratio': 0.40, 'gc': 0.25, 'qoq': 0.20, 'cf': 0.15},
    'V0.4.7e':         {'fund_ratio': 0.35, 'gc': 0.20, 'qoq': 0.25, 'cf': 0.20},
    'FR0.35_gc0.30':   {'fund_ratio': 0.35, 'gc': 0.30, 'qoq': 0.20, 'cf': 0.15},
    'FR0.40_gc0.30_cf0.15': {'fund_ratio': 0.40, 'gc': 0.30, 'qoq': 0.15, 'cf': 0.15},
}

# ── TopN / Hold combinations ──
TOPN_GRID = [8, 10, 12, 15]
HOLD_GRID = [5, 7, 10]  # weekly, 10-day, etc.


def load_data():
    print("📂 Loading data...")
    t0 = time.time()
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    # Load sector mapping if available
    sector_map = {}
    sector_path = DATA_DIR / "sp500_sectors.json"
    if sector_path.exists():
        with open(sector_path) as f:
            sector_map = json.load(f)
    print(f"  ✅ Features: {df.shape}, Prices: {prices.shape}, Sectors: {len(sector_map)} ({time.time()-t0:.1f}s)")
    return df, prices, sector_map


def compute_ranks(df, factor_cols, sample_dates):
    print(f"📊 Computing cross-sectional ranks for {len(sample_dates)} days...")
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


def compute_scores_with_weights(ranks, ic_history, config, power=IC_POWER):
    """Compute scores with given weight configuration + IC weighting."""
    scores = {}
    for date in ranks:
        if date not in ic_history:
            continue
        rd = ranks[date]
        ic = ic_history[date]
        gs = {}
        for gn, factors in FACTOR_GROUPS.items():
            av = [f for f in factors if f in rd.columns and f in ic]
            if not av:
                gs[gn] = pd.Series(0., index=rd.index)
                continue
            iv = {f: max(0, ic[f]) ** power for f in av}
            total = sum(iv.values())
            if total <= 0:
                gs[gn] = rd[av].mean(axis=1)
            else:
                w = {f: iv[f] / total for f in av}
                wt = pd.Series(0., index=rd.index)
                for f in av:
                    wt += w[f] * rd[f]
                gs[gn] = wt
        gc = (GC_WEIGHTS.get('fund_growth', 0) * gs.get('fund_growth', 0) +
              GC_WEIGHTS.get('analyst', 0) * gs.get('analyst', 0) +
              GC_WEIGHTS.get('income', 0) * gs.get('income', 0))
        final = (config['fund_ratio'] * gs.get('fund_ratio', 0) +
                 config['gc'] * gc +
                 config['qoq'] * gs.get('qoq', 0) +
                 config['cf'] * gs.get('cashflow', 0))
        scores[date] = final.dropna().sort_values(ascending=False)
    return scores


def apply_industry_neutralization(scores, sector_map, max_per_sector=2):
    """Cap at max_per_sector stocks per GICS sector."""
    neutralized = {}
    for date, score_series in scores.items():
        picks = []
        sector_count = {}
        for ticker in score_series.index:
            sector = sector_map.get(ticker, 'Unknown')
            if sector_count.get(sector, 0) < max_per_sector:
                picks.append(ticker)
                sector_count[sector] = sector_count.get(sector, 0) + 1
            if len(picks) >= 20:  # cap at 20
                break
        neutralized[date] = score_series[picks] if picks else score_series.head(10)
    return neutralized


def apply_dynamic_sizing(scores, top_n=10):
    """Adjust position count based on score dispersion.
    High dispersion (top stocks far ahead) → use fewer stocks (more concentrated).
    Low dispersion (everything similar) → use more stocks (more diversified).
    """
    sized = {}
    for date, score_series in scores.items():
        if len(score_series) < top_n:
            sized[date] = score_series
            continue
        top_scores = score_series.head(top_n).values
        rest_scores = score_series.iloc[top_n:top_n*2].values if len(score_series) > top_n else top_scores
        gap = np.mean(top_scores) - np.mean(rest_scores) if len(rest_scores) > 0 else 0
        std = np.std(top_scores)
        # High gap or high std → concentrate
        if gap > 0.05 or std > 0.03:
            n = max(5, top_n - 3)
        elif gap < 0.01 and std < 0.01:
            n = min(15, top_n + 5)
        else:
            n = top_n
        sized[date] = score_series.head(n)
    return sized


def weekly_backtest(scores_dict, prices, top_n=10, hold_weeks=1,
                    stop_loss=None, label="", sector_map=None):
    """Walk-Forward backtest with configurable parameters."""
    all_dates = sorted(scores_dict.keys())
    price_dates = sorted(prices.index.astype(str))
    first_date = pd.Timestamp(all_dates[0])
    last_date = pd.Timestamp(all_dates[-1])

    # Build WF windows
    train_start = first_date
    windows = []
    while True:
        train_end = train_start + pd.DateOffset(months=12)
        test_end = train_end + pd.DateOffset(months=6)
        if test_end > last_date:
            break
        windows.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        train_start = train_start + pd.DateOffset(months=6)

    all_returns = []
    window_details = []
    yearly_returns_map = {}

    for wi, w in enumerate(windows):
        test_dates = [d for d in all_dates if w['test_start'] <= d <= w['test_end']]
        # Build rebalance dates based on hold_weeks
        rebal_dates = []
        prev_rebal = None
        for d in test_dates:
            dt = pd.Timestamp(d)
            week = dt.isocalendar()[1]
            year = dt.year
            if hold_weeks == 1:
                key = (year, week)
            else:
                key = (year, week // hold_weeks)
            if key != prev_rebal:
                rebal_dates.append(d)
                prev_rebal = key

        if len(rebal_dates) < 2:
            window_details.append({'window': wi, 'error': 'insufficient dates'})
            continue

        window_returns = []
        for i in range(len(rebal_dates) - 1):
            entry_date = rebal_dates[i]
            exit_date = rebal_dates[i + 1]
            if entry_date not in scores_dict:
                continue

            # Select stocks
            score_series = scores_dict[entry_date]
            if sector_map:
                # Industry neutralization
                picks = []
                sector_count = {}
                for ticker in score_series.index:
                    sector = sector_map.get(ticker, 'Unknown')
                    if sector_count.get(sector, 0) < 2:
                        picks.append(ticker)
                        sector_count[sector] = sector_count.get(sector, 0) + 1
                    if len(picks) >= top_n:
                        break
                top_stocks = picks
            else:
                top_stocks = score_series.head(top_n).index.tolist()

            entry_idx = [d for d in price_dates if d >= entry_date]
            exit_idx = [d for d in price_dates if d >= exit_date]
            if not entry_idx or not exit_idx:
                continue
            actual_entry, actual_exit = entry_idx[0], exit_idx[0]

            # If trailing stop, check daily
            if stop_loss is not None:
                # Find all trading days between entry and exit
                between = [d for d in price_dates if actual_entry < d <= actual_exit]
                period_returns = []
                for ticker in top_stocks:
                    if ticker not in prices.columns:
                        continue
                    p_in = prices.loc[actual_entry, ticker]
                    if pd.isna(p_in) or p_in <= 0:
                        continue
                    # Daily trailing stop check
                    peak = p_in
                    ret = 0.0
                    exited = False
                    for day in between:
                        if day not in prices.index:
                            continue
                        p_day = prices.loc[day, ticker]
                        if pd.isna(p_day):
                            continue
                        peak = max(peak, p_day)
                        dd = (p_day / peak) - 1
                        if dd <= stop_loss:
                            ret = (p_day / p_in) - 1
                            exited = True
                            break
                    if not exited:
                        p_out = prices.loc[actual_exit, ticker]
                        if pd.isna(p_out):
                            continue
                        ret = (p_out / p_in) - 1
                    ret -= 0.002  # cost
                    period_returns.append(ret)
            else:
                # Simple hold to exit
                period_returns = []
                for ticker in top_stocks:
                    if ticker not in prices.columns:
                        continue
                    p_in = prices.loc[actual_entry, ticker]
                    p_out = prices.loc[actual_exit, ticker]
                    if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
                        continue
                    ret = (p_out / p_in) - 1
                    if ret < -0.15:
                        ret = -0.15
                    ret -= 0.002
                    period_returns.append(ret)

            if period_returns:
                avg = np.mean(period_returns)
                window_returns.append(avg)
                all_returns.append(avg)
                year = entry_date[:4]
                if year not in yearly_returns_map:
                    yearly_returns_map[year] = []
                yearly_returns_map[year].append(avg)

        if window_returns:
            wr = np.array(window_returns)
            sharpe = np.sqrt(52) * wr.mean() / wr.std() if wr.std() > 0 else 0
            cum = np.cumprod(1 + wr)
            max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
            cagr = cum[-1] ** (52 / len(wr)) - 1
            window_details.append({
                'window': wi,
                'period': f"{w['test_start']} → {w['test_end']}",
                'sharpe': round(float(sharpe), 3),
                'cagr': round(float(cagr), 4),
                'max_dd': round(float(max_dd), 4),
                'win_rate': round(float((wr > 0).mean()), 3),
                'n_periods': len(window_returns),
            })

    # Overall
    all_wr = np.array(all_returns) if all_returns else np.array([0])
    cum = np.cumprod(1 + all_wr)
    total_sharpe = np.sqrt(52) * all_wr.mean() / all_wr.std() if all_wr.std() > 0 else 0
    total_max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    total_cagr = cum[-1] ** (52 / len(all_wr)) - 1 if len(all_wr) > 0 else 0

    # Yearly
    yearly_stats = {}
    for year in sorted(yearly_returns_map.keys()):
        yr = np.array(yearly_returns_map[year])
        if len(yr) < 2:
            continue
        yr_cum = np.cumprod(1 + yr)
        yearly_stats[year] = {
            'cagr': round(float(yr_cum[-1] ** (52 / len(yr)) - 1), 4),
            'sharpe': round(float(np.sqrt(52) * yr.mean() / yr.std() if yr.std() > 0 else 0), 3),
            'max_dd': round(float((yr_cum / np.maximum.accumulate(yr_cum) - 1).min()), 4),
            'win_rate': round(float((yr > 0).mean()), 3),
            'n_periods': len(yr),
        }

    # Window consistency
    valid_windows = [w for w in window_details if 'sharpe' in w]
    pos_windows = sum(1 for w in valid_windows if w['sharpe'] > 0)
    consistency = pos_windows / len(valid_windows) if valid_windows else 0

    return {
        'sharpe': round(float(total_sharpe), 3),
        'cagr': round(float(total_cagr), 4),
        'max_dd': round(float(total_max_dd), 4),
        'win_rate': round(float((all_wr > 0).mean()), 3),
        'total_return': round(float(cum[-1] - 1), 4),
        'n_periods': len(all_returns),
        'consistency': round(float(consistency), 3),
        'yearly': yearly_stats,
        'windows': window_details,
    }


def main():
    print("=" * 70)
    print("  🦅 Falcon Comprehensive WF Optimization")
    print("=" * 70)
    t_total = time.time()

    df, prices, sector_map = load_data()
    all_factors = list(set(f for fg in FACTOR_GROUPS.values() for f in fg))
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"\n📅 Range: {sample_dates[0]} → {sample_dates[-1]} ({len(sample_dates)} days)")

    # Compute ranks and IC (shared across all configs)
    ranks = compute_ranks(df, all_factors, sample_dates)
    daily_ic = compute_daily_ic(ranks, prices, all_factors)
    ic_hist = rolling_ic(daily_ic, sorted(ranks.keys()), all_factors, IC_LOOKBACK)

    results = {}

    # ══════════════════════════════════════════════════
    # Test 1: Weight configurations (FR/gc/qoq/cf variations)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: Weight Configurations")
    print("=" * 60)

    for name, config in CONFIGS.items():
        print(f"\n▶ {name}: FR={config['fund_ratio']}, gc={config['gc']}, qoq={config['qoq']}, cf={config['cf']}")
        scores = compute_scores_with_weights(ranks, ic_hist, config)
        bt = weekly_backtest(scores, prices, top_n=10, label=name)
        results[f"weight_{name}"] = {
            'type': 'weight',
            'config': config,
            'top_n': 10,
            **bt,
        }
        print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}  Consistency={bt['consistency']:.0%}")

    # ══════════════════════════════════════════════════
    # Test 2: TopN grid
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 2: TopN Grid (baseline weights)")
    print("=" * 60)

    baseline = CONFIGS['V0.4.6_baseline']
    base_scores = compute_scores_with_weights(ranks, ic_hist, baseline)

    for top_n in TOPN_GRID:
        print(f"\n▶ TopN={top_n}")
        bt = weekly_backtest(base_scores, prices, top_n=top_n, label=f"TopN{top_n}")
        results[f"topn_{top_n}"] = {
            'type': 'topn',
            'top_n': top_n,
            **bt,
        }
        print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Test 3: Hold period grid
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: Hold Period Grid")
    print("=" * 60)

    for hold_w in HOLD_GRID:
        print(f"\n▶ Hold={hold_w}w")
        bt = weekly_backtest(base_scores, prices, top_n=10, hold_weeks=hold_w, label=f"Hold{hold_w}w")
        results[f"hold_{hold_w}w"] = {
            'type': 'hold',
            'hold_weeks': hold_w,
            **bt,
        }
        print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Test 4: Industry Neutralization
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 4: Industry Neutralization")
    print("=" * 60)

    if sector_map:
        for max_sec in [2, 3]:
            print(f"\n▶ Max {max_sec} per sector")
            bt = weekly_backtest(base_scores, prices, top_n=10,
                                sector_map=sector_map, label=f"IndNeutral{max_sec}")
            results[f"industry_neutral_{max_sec}"] = {
                'type': 'industry_neutral',
                'max_per_sector': max_sec,
                **bt,
            }
            print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")
    else:
        print("  ⚠️ No sector mapping available, skipping")

    # ══════════════════════════════════════════════════
    # Test 5: Trailing Stop
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 5: Trailing Stop")
    print("=" * 60)

    for sl in [-0.10, -0.15, -0.20]:
        print(f"\n▶ Trailing Stop {sl:.0%}")
        bt = weekly_backtest(base_scores, prices, top_n=10,
                            stop_loss=sl, label=f"Trail{sl:.0%}")
        results[f"trailing_{abs(sl):.0%}"] = {
            'type': 'trailing_stop',
            'stop_loss': sl,
            **bt,
        }
        print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Test 6: Dynamic Position Sizing
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 6: Dynamic Position Sizing")
    print("=" * 60)

    dyn_scores = apply_dynamic_sizing(base_scores, top_n=10)
    bt = weekly_backtest(dyn_scores, prices, top_n=15, label="DynamicSizing")
    results["dynamic_sizing"] = {
        'type': 'dynamic_sizing',
        'top_n': 15,
        **bt,
    }
    print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Test 7: Combined best (weight + trailing + industry)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 7: Combined Best Configs")
    print("=" * 60)

    # Best weight + trailing stop
    best_weight_name = min(
        [k for k in results if k.startswith('weight_')],
        key=lambda k: results[k].get('sharpe', -999)
    )
    best_weight_config = CONFIGS[best_weight_name.replace('weight_', '')]
    best_weight_scores = compute_scores_with_weights(ranks, ic_hist, best_weight_config)

    for sl in [-0.10, -0.15]:
        print(f"\n▶ Best weight ({best_weight_name}) + Trailing {sl:.0%}")
        bt = weekly_backtest(best_weight_scores, prices, top_n=10,
                            stop_loss=sl, label=f"BestW+Trail{sl:.0%}")
        results[f"combo_{best_weight_name}_trail{abs(sl):.0%}"] = {
            'type': 'combo',
            'weight_config': best_weight_config,
            'stop_loss': sl,
            **bt,
        }
        print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # Best weight + industry neutral
    if sector_map:
        print(f"\n▶ Best weight ({best_weight_name}) + Industry max 2")
        bt = weekly_backtest(best_weight_scores, prices, top_n=10,
                            sector_map=sector_map, label="BestW+IndNeutral")
        results[f"combo_{best_weight_name}_indneutral"] = {
            'type': 'combo',
            'weight_config': best_weight_config,
            'industry_neutral': True,
            **bt,
        }
        print(f"  Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  📊 RESULTS SUMMARY")
    print("=" * 70)

    ranked = sorted(results.items(), key=lambda x: x[1].get('sharpe', -999), reverse=True)
    baseline_sharpe = results.get('weight_V0.4.6_baseline', {}).get('sharpe', 0)

    print(f"\n{'Config':<40} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'Consist':>8} {'vs Base':>8}")
    print("-" * 80)
    for name, r in ranked:
        delta = r.get('sharpe', 0) - baseline_sharpe
        marker = "✅" if delta > 0.05 else ("⚠️" if delta > 0 else "❌")
        print(f"{name:<40} {r.get('sharpe',0):>8.3f} {r.get('cagr',0):>7.1%} {r.get('max_dd',0):>7.1%} {r.get('consistency',0):>7.0%} {delta:>+7.3f} {marker}")

    # Find WF winners (Sharpe > baseline + 5%)
    winners = [(n, r) for n, r in ranked
               if r.get('sharpe', 0) > baseline_sharpe * 1.05
               and r.get('consistency', 0) >= 0.5]

    print(f"\n🏆 WF Winners (>5% improvement, consistency≥50%): {len(winners)}")
    for name, r in winners:
        print(f"  {name}: Sharpe={r['sharpe']:.3f} (+{(r['sharpe']/baseline_sharpe-1)*100:.1f}%)")

    if not winners:
        print("  ⚠️ No configuration beats baseline by >5% through WF")
        # Show closest
        close = [(n, r) for n, r in ranked if r.get('sharpe', 0) > baseline_sharpe]
        if close:
            print("  Closest candidates:")
            for name, r in close[:3]:
                print(f"    {name}: Sharpe={r['sharpe']:.3f} (+{(r['sharpe']/baseline_sharpe-1)*100:.1f}%)")

    elapsed = time.time() - t_total
    print(f"\n⏱️ Total: {elapsed:.0f}s")

    # Save
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'baseline': 'V0.4.6 (FR=0.45, gc=0.20, qoq=0.20, cf=0.15)',
            'ic_lookback': IC_LOOKBACK,
            'ic_power': IC_POWER,
            'wf_windows': 'train=12mo, test=6mo, step=6mo',
            'n_configs_tested': len(results),
        },
        'baseline_sharpe': baseline_sharpe,
        'results': results,
        'winners': [n for n, _ in winners] if winners else [],
        'ranked': [(n, r.get('sharpe',0), r.get('cagr',0), r.get('max_dd',0))
                   for n, r in ranked],
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
