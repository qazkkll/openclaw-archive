#!/usr/bin/env python3
"""
🦅 Falcon WF Optimization V2 — Proper Hold Period Testing
==========================================================
Fixes:1. Correct Sharpe/CAGR for different hold periods (not week-as-proxy)
2. Tests hold=7/14/21/30/42/63 days (matching 10yr validation sweet spots)
3. Tests trailing stops WITH proper daily checking
4. Tests combined best configs

WF: train=12mo, test=6mo, daily rebalance check, hold=N trading days
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "comprehensive_wf_v2_results.json"

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

CONFIGS = {
    'V0.4.6':   {'fund_ratio': 0.45, 'gc': 0.20, 'qoq': 0.20, 'cf': 0.15},
    'V0.4.9d':  {'fund_ratio': 0.40, 'gc': 0.25, 'qoq': 0.20, 'cf': 0.15},
    'V0.4.7e':  {'fund_ratio': 0.35, 'gc': 0.20, 'qoq': 0.25, 'cf': 0.20},
    'FR0.35_gc0.30': {'fund_ratio': 0.35, 'gc': 0.30, 'qoq': 0.20, 'cf': 0.15},
}


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


def compute_scores(ranks, ic_history, config, power=IC_POWER):
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


def daily_hold_backtest(scores_dict, prices, top_n=10, hold_days=21,
                         stop_loss=None, label=""):
    """Proper daily hold backtest with correct annualization.
    
    - Rebalances every `hold_days` trading days
    - Each rebalance picks top_n stocks
    - Returns are per-holding-period, annualized correctly
    - Trailing stop checked daily within each holding period
    """
    all_dates = sorted(scores_dict.keys())
    price_dates = sorted(prices.index.astype(str))
    first_date = pd.Timestamp(all_dates[0])
    last_date = pd.Timestamp(all_dates[-1])

    # Build WF windows: train=12mo, test=6mo, step=6mo
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

    all_period_returns = []
    window_details = []
    yearly_returns_map = {}

    for wi, w in enumerate(windows):
        test_dates = [d for d in all_dates if w['test_start'] <= d <= w['test_end']]
        if len(test_dates) < hold_days + 1:
            window_details.append({'window': wi, 'error': 'insufficient dates'})
            continue

        # Rebalance every hold_days trading days
        rebal_indices = list(range(0, len(test_dates), hold_days))
        window_returns = []

        for ri in range(len(rebal_indices) - 1):
            entry_idx = rebal_indices[ri]
            exit_idx = min(rebal_indices[ri + 1], len(test_dates) - 1)
            entry_date = test_dates[entry_idx]
            exit_date = test_dates[exit_idx]

            if entry_date not in scores_dict:
                continue

            # Select stocks
            score_series = scores_dict[entry_date]
            top_stocks = score_series.head(top_n).index.tolist()

            # Find actual price dates
            entry_price_dates = [d for d in price_dates if d >= entry_date]
            exit_price_dates = [d for d in price_dates if d >= exit_date]
            if not entry_price_dates or not exit_price_dates:
                continue
            actual_entry = entry_price_dates[0]
            actual_exit = exit_price_dates[0]

            # Get all trading days between entry and exit for trailing stop
            between = [d for d in price_dates if actual_entry < d <= actual_exit]

            period_returns = []
            for ticker in top_stocks:
                if ticker not in prices.columns:
                    continue
                p_in = prices.loc[actual_entry, ticker]
                if pd.isna(p_in) or p_in <= 0:
                    continue

                if stop_loss is not None:
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
                else:
                    # Simple hold
                    p_out = prices.loc[actual_exit, ticker]
                    if pd.isna(p_out):
                        continue
                    ret = (p_out / p_in) - 1

                # Cap extreme losses
                ret = max(ret, -0.50)
                ret -= 0.002  # round-trip cost
                period_returns.append(ret)

            if period_returns:
                avg = np.mean(period_returns)
                window_returns.append(avg)
                all_period_returns.append(avg)
                year = entry_date[:4]
                if year not in yearly_returns_map:
                    yearly_returns_map[year] = []
                yearly_returns_map[year].append(avg)

        if window_returns:
            wr = np.array(window_returns)
            # Correct annualization: periods_per_year = 252 / hold_days
            periods_per_year = 252 / hold_days
            sharpe = np.sqrt(periods_per_year) * wr.mean() / wr.std() if wr.std() > 0 else 0
            cum = np.cumprod(1 + wr)
            max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
            # CAGR: total return over years
            total_years = len(wr) / periods_per_year
            cagr = cum[-1] ** (1 / total_years) - 1 if total_years > 0 else 0
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
    all_wr = np.array(all_period_returns) if all_period_returns else np.array([0])
    periods_per_year = 252 / hold_days
    total_sharpe = np.sqrt(periods_per_year) * all_wr.mean() / all_wr.std() if all_wr.std() > 0 else 0
    cum = np.cumprod(1 + all_wr)
    total_max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    total_years = len(all_wr) / periods_per_year if periods_per_year > 0 else 1
    total_cagr = cum[-1] ** (1 / total_years) - 1 if total_years > 0 else 0

    # Yearly
    yearly_stats = {}
    for year in sorted(yearly_returns_map.keys()):
        yr = np.array(yearly_returns_map[year])
        if len(yr) < 2:
            continue
        yr_ppy = periods_per_year
        yr_cum = np.cumprod(1 + yr)
        yr_years = len(yr) / yr_ppy
        yearly_stats[year] = {
            'cagr': round(float(yr_cum[-1] ** (1 / yr_years) - 1), 4) if yr_years > 0 else 0,
            'sharpe': round(float(np.sqrt(yr_ppy) * yr.mean() / yr.std() if yr.std() > 0 else 0), 3),
            'max_dd': round(float((yr_cum / np.maximum.accumulate(yr_cum) - 1).min()), 4),
            'win_rate': round(float((yr > 0).mean()), 3),
            'n_periods': len(yr),
        }

    valid_windows = [w for w in window_details if 'sharpe' in w]
    pos_windows = sum(1 for w in valid_windows if w['sharpe'] > 0)
    consistency = pos_windows / len(valid_windows) if valid_windows else 0

    return {
        'sharpe': round(float(total_sharpe), 3),
        'cagr': round(float(total_cagr), 4),
        'max_dd': round(float(total_max_dd), 4),
        'win_rate': round(float((all_wr > 0).mean()), 3),
        'total_return': round(float(cum[-1] - 1), 4),
        'n_periods': len(all_period_returns),
        'hold_days': hold_days,
        'consistency': round(float(consistency), 3),
        'yearly': yearly_stats,
        'windows': window_details,
    }


def main():
    print("=" * 70)
    print("  🦅 Falcon WF Optimization V2 — Proper Hold Period")
    print("=" * 70)
    t_total = time.time()

    df, prices = load_data()
    all_factors = list(set(f for fg in FACTOR_GROUPS.values() for f in fg))
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"\n📅 Range: {sample_dates[0]} → {sample_dates[-1]} ({len(sample_dates)} days)")

    ranks = compute_ranks(df, all_factors, sample_dates)
    daily_ic = compute_daily_ic(ranks, prices, all_factors)
    ic_hist = rolling_ic(daily_ic, sorted(ranks.keys()), all_factors, IC_LOOKBACK)

    results = {}

    # ══════════════════════════════════════════════════
    # Test 1: Weight configs × hold_days grid
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: Weight × Hold Period Grid")
    print("=" * 60)

    hold_days_grid = [7, 14, 21, 30, 42, 63]
    top_n = 10

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n▶ Config: {cfg_name}")
        scores = compute_scores(ranks, ic_hist, cfg)
        for hd in hold_days_grid:
            bt = daily_hold_backtest(scores, prices, top_n=top_n, hold_days=hd,
                                     label=f"{cfg_name}_h{hd}")
            key = f"{cfg_name}_h{hd}"
            results[key] = {'type': 'weight_hold', 'config': cfg_name, **bt}
            print(f"  h{hd:>2}: Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}  N={bt['n_periods']}  Consist={bt['consistency']:.0%}")

    # ══════════════════════════════════════════════════
    # Test 2: TopN grid (baseline, hold=21)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 2: TopN Grid (baseline, hold=21)")
    print("=" * 60)

    base_scores = compute_scores(ranks, ic_hist, CONFIGS['V0.4.6'])
    for tn in [5, 8, 10, 12, 15, 20]:
        bt = daily_hold_backtest(base_scores, prices, top_n=tn, hold_days=21,
                                 label=f"TopN{tn}")
        results[f"topn_{tn}"] = {'type': 'topn', 'top_n': tn, **bt}
        print(f"  Top{tn:>2}: Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Test 3: Trailing Stop (baseline, hold=21, top10)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: Trailing Stop (hold=21)")
    print("=" * 60)

    for sl in [-0.08, -0.10, -0.15, -0.20, -0.25]:
        bt = daily_hold_backtest(base_scores, prices, top_n=10, hold_days=21,
                                 stop_loss=sl, label=f"Trail{abs(sl):.0%}")
        results[f"trail_{abs(sl):.0%}"] = {'type': 'trailing', 'stop_loss': sl, **bt}
        print(f"  SL {sl:.0%}: Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}  MaxDD={bt['max_dd']:.1%}")

    # ══════════════════════════════════════════════════
    # Test 4: Best combo (best weight + best hold + trailing)
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 4: Best Combos")
    print("=" * 60)

    # Find best weight config from Test 1
    weight_results = {k: v for k, v in results.items() if v['type'] == 'weight_hold'}
    best_key = max(weight_results, key=lambda k: weight_results[k]['sharpe'])
    best_cfg_name = weight_results[best_key]['config']
    best_hold = weight_results[best_key]['hold_days']
    print(f"  Best from grid: {best_key} (Sharpe={weight_results[best_key]['sharpe']:.3f})")

    best_scores = compute_scores(ranks, ic_hist, CONFIGS[best_cfg_name])

    # Best weight + best hold + trailing stops
    for sl in [-0.10, -0.15, -0.20]:
        bt = daily_hold_backtest(best_scores, prices, top_n=10, hold_days=best_hold,
                                 stop_loss=sl, label=f"Best+Trail{abs(sl):.0%}")
        results[f"combo_trail_{abs(sl):.0%}"] = {'type': 'combo', 'config': best_cfg_name,
                                                  'hold': best_hold, 'sl': sl, **bt}
        print(f"  {best_cfg_name} h{best_hold} SL{sl:.0%}: Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}")

    # Best weight + different hold periods
    for hd in [14, 30, 42]:
        bt = daily_hold_backtest(best_scores, prices, top_n=10, hold_days=hd,
                                 label=f"Best_h{hd}")
        results[f"combo_hold_{hd}"] = {'type': 'combo', 'config': best_cfg_name, 'hold': hd, **bt}
        print(f"  {best_cfg_name} h{hd}: Sharpe={bt['sharpe']:.3f}  CAGR={bt['cagr']:.1%}")

    # ══════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  📊 FULL RESULTS RANKED BY SHARPE")
    print("=" * 70)

    # Filter out configs with too few periods (< 15) for reliability
    reliable = {k: v for k, v in results.items() if v.get('n_periods', 0) >= 15}
    unreliable = {k: v for k, v in results.items() if v.get('n_periods', 0) < 15}

    ranked = sorted(reliable.items(), key=lambda x: x[1].get('sharpe', -999), reverse=True)
    baseline_key = 'V0.4.6_h21'
    baseline_sharpe = reliable.get(baseline_key, {}).get('sharpe', 2.185)

    print(f"\n{'Config':<45} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'N':>5} {'Cons':>5} {'vsBase':>7}")
    print("-" * 85)
    for name, r in ranked:
        delta = r.get('sharpe', 0) - baseline_sharpe
        marker = "✅" if delta > 0.1 else ("⚠️" if delta > 0 else "❌")
        print(f"{name:<45} {r.get('sharpe',0):>7.3f} {r.get('cagr',0):>6.1%} {r.get('max_dd',0):>6.1%} {r.get('n_periods',0):>5} {r.get('consistency',0):>4.0%} {delta:>+6.3f} {marker}")

    if unreliable:
        print(f"\n⚠️ Unreliable (<15 periods):")
        for name, r in sorted(unreliable.items(), key=lambda x: x[1].get('sharpe', -999), reverse=True):
            print(f"  {name}: Sharpe={r.get('sharpe',0):.3f} N={r.get('n_periods',0)}")

    # Winners
    winners = [(n, r) for n, r in ranked
               if r.get('sharpe', 0) > baseline_sharpe * 1.05
               and r.get('consistency', 0) >= 0.5]

    print(f"\n🏆 WF Winners (>5% improvement, consistency≥50%, N≥15):")
    if winners:
        for name, r in winners:
            pct = (r['sharpe'] / baseline_sharpe - 1) * 100
            print(f"  {name}: Sharpe={r['sharpe']:.3f} (+{pct:.1f}%) CAGR={r['cagr']:.1%} MaxDD={r['max_dd']:.1%}")
    else:
        print("  None found. Top 3 closest:")
        for name, r in ranked[:3]:
            pct = (r['sharpe'] / baseline_sharpe - 1) * 100
            print(f"  {name}: Sharpe={r['sharpe']:.3f} ({pct:+.1f}%)")

    elapsed = time.time() - t_total
    print(f"\n⏱️ Total: {elapsed:.0f}s")

    # Save
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'baseline': 'V0.4.6 h21',
            'baseline_sharpe': baseline_sharpe,
            'ic_lookback': IC_LOOKBACK,
            'ic_power': IC_POWER,
            'hold_days_grid': hold_days_grid,
            'n_configs_tested': len(results),
            'n_reliable': len(reliable),
            'n_unreliable': len(unreliable),
        },
        'results': results,
        'winners': [n for n, _ in winners] if winners else [],
        'ranked': [(n, r.get('sharpe',0), r.get('cagr',0), r.get('max_dd',0), r.get('n_periods',0))
                   for n, r in ranked],
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
