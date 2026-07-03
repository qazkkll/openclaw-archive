#!/usr/bin/env python3
"""
🦅 IC加权稳健性验证
===================
测试3个维度的稳健性:
1. 不同lookback窗口 (126天/252天/504天)
2. 正则化 vs 原始IC (power=0.5 vs 1.0)
3. 滚动稳定性 (IC权重在时间上是否稳定)

输出: data/falcon/ic_weighting_robustness.json
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
OUTPUT_PATH = DATA_DIR / "ic_weighting_robustness.json"

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


def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    return df, prices


def compute_ranks(df, factor_cols, sample_dates):
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
    return ranks


def compute_daily_ic(ranks, prices, factor_cols):
    """预计算所有日期的单日IC。"""
    all_dates = sorted(ranks.keys())
    price_dates = sorted(prices.index.astype(str))
    
    fwd_cache = {}
    for date in all_dates:
        future_candidates = [d for d in price_dates if d > date]
        if len(future_candidates) < 20:
            continue
        future_date = future_candidates[min(29, len(future_candidates)-1)]
        if future_date not in prices.index or date not in prices.index:
            continue
        ret = (prices.loc[future_date] / prices.loc[date]) - 1
        fwd_cache[date] = ret.dropna()
    
    daily_ic = {}
    for date in all_dates:
        if date not in fwd_cache or date not in ranks:
            continue
        rank_df = ranks[date]
        fwd = fwd_cache[date]
        common = rank_df.index.intersection(fwd.index)
        if len(common) < 30:
            continue
        fwd_vals = fwd[common].values
        daily_ic[date] = {}
        for col in factor_cols:
            if col not in rank_df.columns:
                continue
            r = rank_df.loc[common, col].values
            valid = ~(np.isnan(r) | np.isnan(fwd_vals))
            if valid.sum() < 30:
                continue
            ic, _ = spearmanr(r[valid], fwd_vals[valid])
            if not np.isnan(ic):
                daily_ic[date][col] = ic
    
    return daily_ic


def rolling_ic_from_daily(daily_ic, all_dates, factor_cols, lookback, step=5):
    """从预计算的日频IC生成滚动均值。"""
    ic_dates = sorted(daily_ic.keys())
    
    # 每step天计算一次滚动均值
    ic_history = {}
    for i in range(0, len(ic_dates), step):
        date = ic_dates[i]
        window_start = max(0, i - lookback // step)
        window_dates = ic_dates[window_start:i+1]
        
        factor_ics = {}
        for col in factor_cols:
            vals = [daily_ic[d].get(col, np.nan) for d in window_dates if col in daily_ic.get(d, {})]
            vals = [v for v in vals if not np.isnan(v)]
            if len(vals) >= 10:
                factor_ics[col] = np.mean(vals)
        if factor_ics:
            ic_history[date] = factor_ics
    
    # 填充间隔
    all_ic_dates = sorted(ic_history.keys())
    filled = {}
    for date in all_dates:
        candidates = [d for d in all_ic_dates if d <= date]
        if candidates:
            filled[date] = ic_history[candidates[-1]]
    
    return filled


def compute_ic_weighted_scores(ranks, ic_history, factor_groups, gc_weights, model_weights, power=1.0):
    """IC加权评分，支持正则化power参数。
    
    power=1.0: 原始IC加权
    power=0.5: 平方根正则化（缩小强弱差距）
    power=0.3: 更强正则化
    """
    all_dates = sorted(ranks.keys())
    scores_dict = {}
    
    for date in all_dates:
        if date not in ic_history:
            continue
        df = ranks[date]
        ic = ic_history[date]
        
        group_scores = {}
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns and f in ic]
            if not available:
                group_scores[group_name] = pd.Series(0.0, index=df.index)
                continue
            
            # IC加权 + 正则化
            ic_values = {f: max(0, ic[f]) ** power for f in available}
            total = sum(ic_values.values())
            
            if total <= 0:
                group_scores[group_name] = df[available].mean(axis=1)
            else:
                weights = {f: ic_values[f] / total for f in available}
                weighted = pd.Series(0.0, index=df.index)
                for f in available:
                    weighted += weights[f] * df[f]
                group_scores[group_name] = weighted
        
        gc = (gc_weights.get('fund_growth', 0) * group_scores.get('fund_growth', 0) +
              gc_weights.get('analyst', 0) * group_scores.get('analyst', 0) +
              gc_weights.get('income', 0) * group_scores.get('income', 0))
        
        final = (model_weights['fund_ratio'] * group_scores.get('fund_ratio', 0) +
                 model_weights['gc_baseline'] * gc +
                 model_weights['qoq'] * group_scores.get('qoq', 0) +
                 model_weights['cashflow'] * group_scores.get('cashflow', 0))
        
        scores_dict[date] = final.dropna().sort_values(ascending=False)
    
    return scores_dict


def compute_equal_scores(ranks, factor_groups, gc_weights, model_weights):
    """等权baseline。"""
    all_dates = sorted(ranks.keys())
    scores_dict = {}
    for date in all_dates:
        df = ranks[date]
        group_scores = {}
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            group_scores[group_name] = df[available].mean(axis=1) if available else pd.Series(0.0, index=df.index)
        
        gc = (gc_weights.get('fund_growth', 0) * group_scores.get('fund_growth', 0) +
              gc_weights.get('analyst', 0) * group_scores.get('analyst', 0) +
              gc_weights.get('income', 0) * group_scores.get('income', 0))
        
        final = (model_weights['fund_ratio'] * group_scores.get('fund_ratio', 0) +
                 model_weights['gc_baseline'] * gc +
                 model_weights['qoq'] * group_scores.get('qoq', 0) +
                 model_weights['cashflow'] * group_scores.get('cashflow', 0))
        scores_dict[date] = final.dropna().sort_values(ascending=False)
    return scores_dict


def weekly_backtest(scores_dict, prices, top_n=10, cost=0.001, stop_loss=-0.15):
    """简化版周频回测，只返回核心指标。"""
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
            'test_start': train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        train_start = train_start + pd.DateOffset(months=6)
    
    all_returns = []
    yearly_map = {}
    
    for w in windows:
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
        
        for i in range(len(weekly_dates) - 1):
            entry_date = weekly_dates[i]
            exit_date = weekly_dates[i + 1]
            if entry_date not in scores_dict:
                continue
            scores = scores_dict[entry_date]
            top_stocks = scores.head(top_n).index.tolist()
            
            entry_idx = [d for d in price_dates if d >= entry_date]
            exit_idx = [d for d in price_dates if d >= exit_date]
            if not entry_idx or not exit_idx:
                continue
            actual_entry, actual_exit = entry_idx[0], exit_idx[0]
            if actual_entry not in prices.index or actual_exit not in prices.index:
                continue
            
            period_rets = []
            for ticker in top_stocks:
                if ticker not in prices.columns:
                    continue
                p_in = prices.loc[actual_entry, ticker]
                p_out = prices.loc[actual_exit, ticker]
                if pd.isna(p_in) or pd.isna(p_out) or p_in <= 0:
                    continue
                ret = (p_out / p_in) - 1
                if ret < stop_loss:
                    ret = stop_loss
                ret -= cost * 2
                period_rets.append(ret)
            
            if period_rets:
                avg = np.mean(period_rets)
                all_returns.append(avg)
                year = entry_date[:4]
                if year not in yearly_map:
                    yearly_map[year] = []
                yearly_map[year].append(avg)
    
    if not all_returns:
        return {'sharpe': 0, 'cagr': 0, 'max_dd': 0, 'win_rate': 0, 'total_return': 0, 'yearly': {}}
    
    wr = np.array(all_returns)
    cum = np.cumprod(1 + wr)
    sharpe = np.sqrt(52) * wr.mean() / wr.std() if wr.std() > 0 else 0
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    cagr = cum[-1] ** (52 / len(wr)) - 1
    
    yearly = {}
    for year, rets in sorted(yearly_map.items()):
        yr = np.array(rets)
        if len(yr) < 2:
            continue
        yr_cum = np.cumprod(1 + yr)
        yearly[year] = {
            'sharpe': round(float(np.sqrt(52) * yr.mean() / yr.std() if yr.std() > 0 else 0), 3),
            'cagr': round(float(yr_cum[-1] ** (52 / len(yr)) - 1), 4),
            'n_weeks': len(yr),
        }
    
    return {
        'sharpe': round(float(sharpe), 3),
        'cagr': round(float(cagr), 4),
        'max_dd': round(float(max_dd), 4),
        'win_rate': round(float((wr > 0).mean()), 3),
        'total_return': round(float(cum[-1] - 1), 4),
        'n_weeks': len(all_returns),
        'yearly': yearly,
    }


def check_ic_stability(daily_ic, factor_cols, window=63):
    """检查IC权重的时序稳定性。
    
    方法：计算每个因子的滚动IC，然后算IC的时间序列标准差。
    标准差小=稳定，标准差大=不稳定。
    """
    ic_dates = sorted(daily_ic.keys())
    
    stability = {}
    for col in factor_cols:
        ics = []
        for d in ic_dates:
            if col in daily_ic[d]:
                ics.append(daily_ic[d][col])
        
        if len(ics) < 50:
            continue
        
        ics = np.array(ics)
        stability[col] = {
            'ic_mean': round(float(np.mean(ics)), 5),
            'ic_std': round(float(np.std(ics)), 5),
            'ic_ir': round(float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0, 4),
            'pct_positive': round(float((ics > 0).mean()), 3),
            'n_obs': len(ics),
        }
    
    return stability


def main():
    print("=" * 70)
    print("  IC加权稳健性验证")
    print("=" * 70)
    
    t_total = time.time()
    df, prices = load_data()
    
    all_factors = []
    for factors in FACTOR_GROUPS.values():
        all_factors.extend(factors)
    all_factors = list(set(all_factors))
    
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"📅 范围: {sample_dates[0]} → {sample_dates[-1]}")
    
    # 计算截面排名
    print("\n📊 计算截面排名...")
    t0 = time.time()
    ranks = compute_ranks(df, all_factors, sample_dates)
    print(f"  ✅ {len(ranks)}天 ({time.time()-t0:.0f}s)")
    
    # 预计算日频IC
    print("\n📊 预计算日频IC...")
    t0 = time.time()
    daily_ic = compute_daily_ic(ranks, prices, all_factors)
    print(f"  ✅ {len(daily_ic)}天 ({time.time()-t0:.0f}s)")
    
    # ── 实验1: 不同lookback窗口 ──
    print("\n" + "=" * 50)
    print("  实验1: 不同lookback窗口")
    print("=" * 50)
    
    lookback_results = {}
    for lb in [126, 252, 504]:
        print(f"\n  lookback={lb}天...")
        ic_hist = rolling_ic_from_daily(daily_ic, sorted(ranks.keys()), all_factors, lb, step=5)
        scores = compute_ic_weighted_scores(ranks, ic_hist, FACTOR_GROUPS, GC_WEIGHTS, MODEL_WEIGHTS, power=1.0)
        result = weekly_backtest(scores, prices)
        lookback_results[lb] = result
        print(f"    Sharpe={result['sharpe']:.3f}  CAGR={result['cagr']:.1%}  MaxDD={result['max_dd']:.1%}")
    
    # ── 实验2: 正则化 ──
    print("\n" + "=" * 50)
    print("  实验2: 正则化 (power参数)")
    print("=" * 50)
    
    # 用252天lookback
    ic_hist_252 = rolling_ic_from_daily(daily_ic, sorted(ranks.keys()), all_factors, 252, step=5)
    
    reg_results = {}
    for power in [1.0, 0.5, 0.3, 0.0]:  # 0.0 = 等权
        if power == 0.0:
            scores = compute_equal_scores(ranks, FACTOR_GROUPS, GC_WEIGHTS, MODEL_WEIGHTS)
            label = "等权(0.0)"
        else:
            scores = compute_ic_weighted_scores(ranks, ic_hist_252, FACTOR_GROUPS, GC_WEIGHTS, MODEL_WEIGHTS, power=power)
            label = f"IC^{power}"
        result = weekly_backtest(scores, prices)
        reg_results[label] = result
        print(f"  power={power:.1f}  Sharpe={result['sharpe']:.3f}  CAGR={result['cagr']:.1%}  MaxDD={result['max_dd']:.1%}")
    
    # ── 实验3: IC稳定性 ──
    print("\n" + "=" * 50)
    print("  实验3: IC因子稳定性")
    print("=" * 50)
    
    stability = check_ic_stability(daily_ic, all_factors)
    
    # 按IC_IR排序，打印最稳定和最不稳定的
    sorted_stab = sorted(stability.items(), key=lambda x: -x[1]['ic_ir'])
    print(f"\n  最稳定 (IC_IR最高的5个):")
    for f, s in sorted_stab[:5]:
        print(f"    {f:<40} IC_IR={s['ic_ir']:>6.3f}  IC均值={s['ic_mean']:>8.5f}  正IC占比={s['pct_positive']:.0%}")
    
    print(f"\n  最不稳定 (IC_IR最低的5个):")
    for f, s in sorted_stab[-5:]:
        print(f"    {f:<40} IC_IR={s['ic_ir']:>6.3f}  IC均值={s['ic_mean']:>8.5f}  正IC占比={s['pct_positive']:.0%}")
    
    # ── 实验4: 按年看lookback=252的IC加权 vs 等权 ──
    print("\n" + "=" * 50)
    print("  实验4: 按年详细对比 (lookback=252)")
    print("=" * 50)
    
    eq_scores = compute_equal_scores(ranks, FACTOR_GROUPS, GC_WEIGHTS, MODEL_WEIGHTS)
    eq_result = weekly_backtest(eq_scores, prices)
    ic252_result = reg_results.get('IC^1.0', lookback_results[252])
    
    print(f"\n  {'年份':<6} {'等权':>8} {'IC^1.0':>8} {'IC^0.5':>8} {'IC^0.3':>8}")
    print(f"  {'-'*38}")
    
    all_years = sorted(set(list(eq_result.get('yearly', {}).keys())))
    for year in all_years:
        eq_s = eq_result.get('yearly', {}).get(year, {}).get('sharpe', 0)
        ic1_s = ic252_result.get('yearly', {}).get(year, {}).get('sharpe', 0)
        ic05_s = reg_results.get('IC^0.5', {}).get('yearly', {}).get(year, {}).get('sharpe', 0)
        ic03_s = reg_results.get('IC^0.3', {}).get('yearly', {}).get(year, {}).get('sharpe', 0)
        print(f"  {year:<6} {eq_s:>8.3f} {ic1_s:>8.3f} {ic05_s:>8.3f} {ic03_s:>8.3f}")
    
    # ── 总结 ──
    print(f"\n⏱️ 总耗时: {time.time()-t_total:.0f}秒")
    
    # 保存
    output = {
        'metadata': {'timestamp': datetime.now().isoformat()},
        'lookback_comparison': {str(k): v for k, v in lookback_results.items()},
        'regularization_comparison': {k: v for k, v in reg_results.items()},
        'ic_stability': stability,
        'best_config': None,
    }
    
    # 找最优配置
    best_sharpe = 0
    best_label = ''
    for label, result in list(lookback_results.items()) + list(reg_results.items()):
        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_label = str(label)
    output['best_config'] = {'label': best_label, 'sharpe': best_sharpe}
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"💾 已保存: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
