#!/usr/bin/env python3
"""
🦅 Falcon V0.4.6: 5-Year Weekly Walk-Forward
==============================================
V0.4.6 = V0.4.4因子结构 + IC加权（lookback=126天, power=0.5）
顶层权重不变: fund45 + gc20 + qoq20 + cf15
底层因子: 从等权 → 滚动IC^0.5加权

Walk-Forward: train=12个月, test=6个月, 每周一调仓, 持有7天
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
OUTPUT_PATH = DATA_DIR / "v046_5yr_weekly.json"

# V0.4.4/V0.4.6 共用因子组（V0.4.6=V0.4.4因子+IC加权）
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
MODEL_WEIGHTS = {'fund_ratio': 0.45, 'gc_baseline': 0.20, 'qoq': 0.20, 'cashflow': 0.15}

FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity', 'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}

# V0.4.6 参数
IC_LOOKBACK = 126
IC_POWER = 0.5


def load_data():
    print("📂 加载数据...")
    t0 = time.time()
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Features: {df.shape}, Prices: {prices.shape} ({time.time()-t0:.1f}s)")
    return df, prices


def compute_ranks(df, factor_cols, sample_dates):
    print(f"📊 计算 {len(sample_dates)} 天截面排名...")
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
    print(f"  ✅ {len(ranks)}天 ({time.time()-t0:.0f}s)")
    return ranks


def compute_daily_ic(ranks, prices, factor_cols):
    print(f"📊 计算日频IC...")
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
    print(f"  ✅ {len(daily_ic)}天 ({time.time()-t0:.0f}s)")
    return daily_ic


def rolling_ic(daily_ic, all_dates, factor_cols, lookback, step=5):
    print(f"📊 滚动IC (lookback={lookback})...")
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
    print(f"  ✅ {len(filled)}天 ({time.time()-t0:.0f}s)")
    return filled


def compute_equal_scores(ranks):
    print("📊 等权评分 (V0.4.4 baseline)...")
    t0 = time.time()
    scores = {}
    for date in ranks:
        rd = ranks[date]
        gs = {}
        for gn, factors in FACTOR_GROUPS.items():
            av = [f for f in factors if f in rd.columns]
            gs[gn] = rd[av].mean(axis=1) if av else pd.Series(0., index=rd.index)
        gc = (GC_WEIGHTS.get('fund_growth', 0) * gs.get('fund_growth', 0) +
              GC_WEIGHTS.get('analyst', 0) * gs.get('analyst', 0) +
              GC_WEIGHTS.get('income', 0) * gs.get('income', 0))
        final = (MODEL_WEIGHTS['fund_ratio'] * gs.get('fund_ratio', 0) +
                 MODEL_WEIGHTS['gc_baseline'] * gc +
                 MODEL_WEIGHTS['qoq'] * gs.get('qoq', 0) +
                 MODEL_WEIGHTS['cashflow'] * gs.get('cashflow', 0))
        scores[date] = final.dropna().sort_values(ascending=False)
    print(f"  ✅ {len(scores)}天 ({time.time()-t0:.0f}s)")
    return scores


def compute_ic_scores(ranks, ic_history, power):
    print(f"📊 IC加权评分 (power={power})...")
    t0 = time.time()
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
        final = (MODEL_WEIGHTS['fund_ratio'] * gs.get('fund_ratio', 0) +
                 MODEL_WEIGHTS['gc_baseline'] * gc +
                 MODEL_WEIGHTS['qoq'] * gs.get('qoq', 0) +
                 MODEL_WEIGHTS['cashflow'] * gs.get('cashflow', 0))
        scores[date] = final.dropna().sort_values(ascending=False)
    print(f"  ✅ {len(scores)}天 ({time.time()-t0:.0f}s)")
    return scores


def weekly_backtest(scores_dict, prices, label=""):
    """完整周频Walk-Forward回测，含按年+按窗口统计。"""
    print(f"\n🔄 {label} 周频WF回测...")
    t0 = time.time()
    all_dates = sorted(scores_dict.keys())
    price_dates = sorted(prices.index.astype(str))
    first_date = pd.Timestamp(all_dates[0])
    last_date = pd.Timestamp(all_dates[-1])

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

    all_weekly_returns = []
    window_details = []
    yearly_returns_map = {}

    for wi, w in enumerate(windows):
        test_dates = [d for d in all_dates if w['test_start'] <= d <= w['test_end']]
        weekly_dates = []
        prev_week = None
        for d in test_dates:
            dt = pd.Timestamp(d)
            week = dt.isocalendar()[1]
            year = dt.year
            if (year, week) != prev_week:
                weekly_dates.append(d)
                prev_week = (year, week)

        if len(weekly_dates) < 2:
            window_details.append({'window': wi, 'error': 'insufficient dates'})
            continue

        window_returns = []
        for i in range(len(weekly_dates) - 1):
            entry_date = weekly_dates[i]
            exit_date = weekly_dates[i + 1]
            if entry_date not in scores_dict:
                continue
            scores = scores_dict[entry_date]
            top_stocks = scores.head(10).index.tolist()

            entry_idx = [d for d in price_dates if d >= entry_date]
            exit_idx = [d for d in price_dates if d >= exit_date]
            if not entry_idx or not exit_idx:
                continue
            actual_entry, actual_exit = entry_idx[0], exit_idx[0]
            if actual_entry not in prices.index or actual_exit not in prices.index:
                continue

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
                all_weekly_returns.append(avg)
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
                'n_weeks': len(window_returns),
            })

    # 总体
    all_wr = np.array(all_weekly_returns) if all_weekly_returns else np.array([0])
    cum = np.cumprod(1 + all_wr)
    total_sharpe = np.sqrt(52) * all_wr.mean() / all_wr.std() if all_wr.std() > 0 else 0
    total_max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    total_cagr = cum[-1] ** (52 / len(all_wr)) - 1

    # 按年
    yearly_stats = {}
    for year in sorted(yearly_returns_map.keys()):
        yr = np.array(yearly_returns_map[year])
        if len(yr) < 2:
            continue
        yr_cum = np.cumprod(1 + yr)
        yearly_stats[year] = {
            'cagr': round(float(yr_cum[-1] ** (52 / len(yr)) - 1), 4),
            'total_return': round(float(yr_cum[-1] - 1), 4),
            'sharpe': round(float(np.sqrt(52) * yr.mean() / yr.std() if yr.std() > 0 else 0), 3),
            'max_dd': round(float((yr_cum / np.maximum.accumulate(yr_cum) - 1).min()), 4),
            'win_rate': round(float((yr > 0).mean()), 3),
            'n_weeks': len(yr),
        }

    elapsed = time.time() - t0
    print(f"  ✅ {elapsed:.0f}s | 周数={len(all_weekly_returns)} 交易={len(all_weekly_returns)*10}")

    return {
        'sharpe': round(float(total_sharpe), 3),
        'cagr': round(float(total_cagr), 4),
        'max_dd': round(float(total_max_dd), 4),
        'win_rate': round(float((all_wr > 0).mean()), 3),
        'total_return': round(float(cum[-1] - 1), 4),
        'n_weeks': len(all_weekly_returns),
        'yearly': yearly_stats,
        'windows': window_details,
    }


def main():
    print("=" * 70)
    print("  Falcon V0.4.4 vs V0.4.6: 5-Year Weekly Walk-Forward")
    print("=" * 70)
    t_total = time.time()

    df, prices = load_data()
    all_factors = list(set(f for fg in FACTOR_GROUPS.values() for f in fg))
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"\n📅 范围: {sample_dates[0]} → {sample_dates[-1]} ({len(sample_dates)}天)")

    # 截面排名
    ranks = compute_ranks(df, all_factors, sample_dates)

    # ── V0.4.4: 等权 ──
    print("\n" + "=" * 50)
    print("  V0.4.4: 等权 (baseline)")
    print("=" * 50)
    v044_scores = compute_equal_scores(ranks)
    v044 = weekly_backtest(v044_scores, prices, "V0.4.4")

    # ── 日频IC + 滚动IC ──
    daily_ic = compute_daily_ic(ranks, prices, all_factors)
    ic_hist = rolling_ic(daily_ic, sorted(ranks.keys()), all_factors, IC_LOOKBACK)

    # ── V0.4.6: IC加权 ──
    print("\n" + "=" * 50)
    print(f"  V0.4.6: IC加权 (lookback={IC_LOOKBACK}, power={IC_POWER})")
    print("=" * 50)
    v046_scores = compute_ic_scores(ranks, ic_hist, IC_POWER)
    v046 = weekly_backtest(v046_scores, prices, "V0.4.6")

    # ── 对比 ──
    print("\n" + "=" * 70)
    print("  📊 V0.4.4 vs V0.4.6")
    print("=" * 70)

    print(f"\n{'指标':<15} {'V0.4.4':>12} {'V0.4.6':>12} {'差异':>12}")
    print("-" * 51)
    print(f"{'Sharpe':<15} {v044['sharpe']:>12.3f} {v046['sharpe']:>12.3f} {v046['sharpe']-v044['sharpe']:>+12.3f}")
    print(f"{'CAGR':<15} {v044['cagr']:>12.1%} {v046['cagr']:>12.1%} {v046['cagr']-v044['cagr']:>+12.1%}")
    print(f"{'MaxDD':<15} {v044['max_dd']:>12.1%} {v046['max_dd']:>12.1%} {v046['max_dd']-v044['max_dd']:>+12.1%}")
    print(f"{'Win Rate':<15} {v044['win_rate']:>12.1%} {v046['win_rate']:>12.1%} {v046['win_rate']-v044['win_rate']:>+12.1%}")
    print(f"{'总收益':<15} {v044['total_return']:>12.1%} {v046['total_return']:>12.1%} {v046['total_return']-v044['total_return']:>+12.1%}")

    print(f"\n📅 按年:")
    print(f"{'年份':<6} {'V0.4.4':>8} {'V0.4.6':>8} {'V0.4.4 CAGR':>12} {'V0.4.6 CAGR':>12} {'赢者':>6}")
    print("-" * 52)
    all_years = sorted(set(list(v044.get('yearly', {}).keys()) + list(v046.get('yearly', {}).keys())))
    for year in all_years:
        v44s = v044.get('yearly', {}).get(year, {}).get('sharpe', 0)
        v46s = v046.get('yearly', {}).get(year, {}).get('sharpe', 0)
        v44c = v044.get('yearly', {}).get(year, {}).get('cagr', 0)
        v46c = v046.get('yearly', {}).get(year, {}).get('cagr', 0)
        winner = 'V0.4.6' if v46s > v44s else 'V0.4.4'
        print(f"{year:<6} {v44s:>8.3f} {v46s:>8.3f} {v44c:>12.1%} {v46c:>12.1%} {winner:>6}")

    winner = 'V0.4.6' if v046['sharpe'] > v044['sharpe'] else 'V0.4.4'
    print(f"\n🏆 总体赢家: {winner}")
    print(f"⏱️ 总耗时: {time.time()-t_total:.0f}秒")

    # 保存
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'v046_config': {
                'ic_lookback': IC_LOOKBACK,
                'ic_power': IC_POWER,
                'description': 'V0.4.4因子结构 + IC^0.5加权(lookback=126天)',
            },
        },
        'v044': v044,
        'v046': v046,
        'winner': winner,
    }
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 已保存: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
