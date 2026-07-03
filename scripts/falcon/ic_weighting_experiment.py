#!/usr/bin/env python3
"""
🦅 Falcon IC加权 vs 等权: 5-Year Weekly Walk-Forward
=====================================================
对比底层因子等权 vs IC加权的差异。

原理:
  顶层权重不变: fund_ratio=45%, gc=20%, qoq=20%, cashflow=15%
  底层因子: 从等权 → 按滚动252天IC加权
  
  滚动IC计算:
    每个因子在每个日期，计算过去252天的截面IC(Spearman相关)
    IC均值作为权重基础
    负IC的因子权重设为0（不参与）
    正IC的因子按IC大小成比例分配权重

Walk-Forward: 在每个test窗口内，用train期间的IC来定权重
  - 确保不用未来数据
  - 每个窗口独立计算IC权重

输出: data/falcon/ic_weighting_comparison.json
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings('ignore')

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "ic_weighting_comparison.json"


# ═══════════════════════════════════════════════════
#  V0.4.4 因子组定义 (与生产一致)
# ═══════════════════════════════════════════════════

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
    'analyst': [
        'a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps',
    ],
    'income': [
        'i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
        'i_revenue_growth_yoy', 'i_gross_margin_delta',
    ],
    'qoq': [
        'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
        'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
    ],
    'cashflow': [
        'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
    ],
}

GC_WEIGHTS = {'fund_growth': 0.60, 'analyst': 0.25, 'income': 0.15}
MODEL_WEIGHTS = {'fund_ratio': 0.45, 'gc_baseline': 0.20, 'qoq': 0.20, 'cashflow': 0.15}

FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity',
    'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}


# ═══════════════════════════════════════════════════
#  数据加载 + 截面排名
# ═══════════════════════════════════════════════════

def load_data():
    print("📂 加载数据...")
    t0 = time.time()
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Features: {df.shape}, Prices: {price_pivot.shape} ({time.time()-t0:.1f}s)")
    return df, price_pivot


def compute_all_ranks(df, factor_cols, sample_dates):
    """计算所有日期的截面百分位排名（含翻转）。"""
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


# ═══════════════════════════════════════════════════
#  滚动IC计算
# ═══════════════════════════════════════════════════

def compute_rolling_ic(ranks, prices, factor_cols, lookback=252, step=5):
    """计算每个因子的滚动IC（每step天计算一次，节省时间）。
    
    IC = Spearman(因子截面排名, 前瞻30天收益)
    用过去lookback天的IC均值作为权重依据。
    """
    print(f"📊 计算滚动IC (lookback={lookback}天, step={step}天)...")
    t0 = time.time()
    
    all_dates = sorted(ranks.keys())
    price_dates = sorted(prices.index.astype(str))
    
    # 预计算所有日期的前瞻30天收益（向量化）
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
    
    # 预计算所有日期每个因子的截面IC（单日）
    daily_ic = {}  # {date: {factor: ic}}
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
    
    print(f"  日频IC计算完成: {len(daily_ic)}天 ({time.time()-t0:.0f}s)")
    
    # 滚动均值（每step天计算一次）
    ic_dates = sorted(daily_ic.keys())
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
    
    # 对step间隔之间的日期，复制最近的IC
    all_ic_dates = sorted(ic_history.keys())
    filled = {}
    for date in all_dates:
        # 找最近的已计算IC日期
        candidates = [d for d in all_ic_dates if d <= date]
        if candidates:
            filled[date] = ic_history[candidates[-1]]
    
    print(f"  ✅ {len(filled)}天IC完成 ({time.time()-t0:.0f}s)")
    return filled


# ═══════════════════════════════════════════════════
#  IC加权评分
# ═══════════════════════════════════════════════════

def compute_ic_weighted_scores(ranks, ic_history, factor_groups, gc_weights, model_weights):
    """用IC加权计算分数。
    
    对每个因子组内的因子，用IC均值作为权重（负IC的设为0）。
    组间权重（45%/20%/20%/15%）不变。
    """
    print("📊 计算IC加权分数...")
    t0 = time.time()
    
    all_dates = sorted(ranks.keys())
    scores_dict = {}
    weight_details = {}
    
    for date in all_dates:
        if date not in ic_history:
            continue
        
        df = ranks[date]
        ic = ic_history[date]
        
        group_scores = {}
        day_weights = {}
        
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns and f in ic]
            if not available:
                group_scores[group_name] = pd.Series(0.0, index=df.index)
                continue
            
            # IC加权: 只用正IC的因子
            ic_values = {f: max(0, ic[f]) for f in available}
            total_ic = sum(ic_values.values())
            
            if total_ic <= 0:
                # 所有IC都是负的，回退到等权
                group_scores[group_name] = df[available].mean(axis=1)
                day_weights[group_name] = {f: 1.0/len(available) for f in available}
            else:
                # IC加权
                weights = {f: ic_values[f] / total_ic for f in available}
                weighted = pd.Series(0.0, index=df.index)
                for f in available:
                    weighted += weights[f] * df[f]
                group_scores[group_name] = weighted
                day_weights[group_name] = weights
        
        # growth_composite
        gc = (gc_weights.get('fund_growth', 0) * group_scores.get('fund_growth', 0) +
              gc_weights.get('analyst', 0) * group_scores.get('analyst', 0) +
              gc_weights.get('income', 0) * group_scores.get('income', 0))
        
        final = (model_weights['fund_ratio'] * group_scores.get('fund_ratio', 0) +
                 model_weights['gc_baseline'] * gc +
                 model_weights['qoq'] * group_scores.get('qoq', 0) +
                 model_weights['cashflow'] * group_scores.get('cashflow', 0))
        
        scores_dict[date] = final.dropna().sort_values(ascending=False)
        weight_details[date] = day_weights
    
    print(f"  ✅ {len(scores_dict)}天 ({time.time()-t0:.0f}s)")
    return scores_dict, weight_details


def compute_equal_weighted_scores(ranks, factor_groups, gc_weights, model_weights):
    """等权评分（baseline）。"""
    print("📊 计算等权分数 (baseline)...")
    t0 = time.time()
    
    all_dates = sorted(ranks.keys())
    scores_dict = {}
    
    for date in all_dates:
        df = ranks[date]
        group_scores = {}
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                group_scores[group_name] = df[available].mean(axis=1)
            else:
                group_scores[group_name] = pd.Series(0.0, index=df.index)
        
        gc = (gc_weights.get('fund_growth', 0) * group_scores.get('fund_growth', 0) +
              gc_weights.get('analyst', 0) * group_scores.get('analyst', 0) +
              gc_weights.get('income', 0) * group_scores.get('income', 0))
        
        final = (model_weights['fund_ratio'] * group_scores.get('fund_ratio', 0) +
                 model_weights['gc_baseline'] * gc +
                 model_weights['qoq'] * group_scores.get('qoq', 0) +
                 model_weights['cashflow'] * group_scores.get('cashflow', 0))
        
        scores_dict[date] = final.dropna().sort_values(ascending=False)
    
    print(f"  ✅ {len(scores_dict)}天 ({time.time()-t0:.0f}s)")
    return scores_dict


# ═══════════════════════════════════════════════════
#  周频回测 (复用compare脚本的逻辑)
# ═══════════════════════════════════════════════════

def weekly_backtest(scores_dict, prices, top_n=10, cost=0.001, stop_loss=-0.15,
                    train_months=12, test_months=6):
    """周频Walk-Forward回测。"""
    all_dates = sorted(scores_dict.keys())
    price_dates = sorted(prices.index.astype(str))
    
    first_date = pd.Timestamp(all_dates[0])
    last_date = pd.Timestamp(all_dates[-1])
    
    train_start = first_date
    windows = []
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > last_date:
            break
        windows.append({
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': train_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })
        train_start = train_start + pd.DateOffset(months=test_months)
    
    all_weekly_returns = []
    window_details = []
    
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
            top_stocks = scores.head(top_n).index.tolist()
            
            entry_prices_idx = [d for d in price_dates if d >= entry_date]
            exit_prices_idx = [d for d in price_dates if d >= exit_date]
            if not entry_prices_idx or not exit_prices_idx:
                continue
            actual_entry = entry_prices_idx[0]
            actual_exit = exit_prices_idx[0]
            if actual_entry not in prices.index or actual_exit not in prices.index:
                continue
            
            period_returns = []
            for ticker in top_stocks:
                if ticker not in prices.columns:
                    continue
                p_entry = prices.loc[actual_entry, ticker]
                p_exit = prices.loc[actual_exit, ticker]
                if pd.isna(p_entry) or pd.isna(p_exit) or p_entry <= 0:
                    continue
                ret = (p_exit / p_entry) - 1
                if ret < stop_loss:
                    ret = stop_loss
                ret -= cost * 2
                period_returns.append(ret)
            
            if period_returns:
                avg_ret = np.mean(period_returns)
                window_returns.append(avg_ret)
                all_weekly_returns.append(avg_ret)
        
        if window_returns:
            wr = np.array(window_returns)
            sharpe = np.sqrt(52) * wr.mean() / wr.std() if wr.std() > 0 else 0
            cum = np.cumprod(1 + wr)
            max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
            cagr = cum[-1] ** (52 / len(wr)) - 1 if len(wr) > 0 else 0
            window_details.append({
                'window': wi,
                'period': f"{w['test_start']} → {w['test_end']}",
                'sharpe': round(float(sharpe), 3),
                'cagr': round(float(cagr), 4),
                'max_dd': round(float(max_dd), 4),
                'win_rate': round(float((wr > 0).mean()), 3),
                'n_weeks': len(window_returns),
            })
    
    all_wr = np.array(all_weekly_returns) if all_weekly_returns else np.array([0])
    total_sharpe = np.sqrt(52) * all_wr.mean() / all_wr.std() if all_wr.std() > 0 else 0
    cum = np.cumprod(1 + all_wr)
    total_max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    total_cagr = cum[-1] ** (52 / len(all_wr)) - 1 if len(all_wr) > 0 else 0
    
    # 按年统计
    yearly_returns_map = {}
    for wi, w in enumerate(windows):
        test_dates_in_window = [d for d in sorted(scores_dict.keys())
                                 if w['test_start'] <= d <= w['test_end']]
        weekly_in_window = []
        prev_week = None
        for d in test_dates_in_window:
            dt = pd.Timestamp(d)
            week = dt.isocalendar()[1]
            year = dt.year
            if (year, week) != prev_week:
                weekly_in_window.append(d)
                prev_week = (year, week)
        
        for i in range(len(weekly_in_window) - 1):
            entry_date = weekly_in_window[i]
            exit_date = weekly_in_window[i + 1]
            if entry_date not in scores_dict:
                continue
            scores = scores_dict[entry_date]
            top_stocks = scores.head(top_n).index.tolist()
            entry_prices_idx = [d for d in price_dates if d >= entry_date]
            exit_prices_idx = [d for d in price_dates if d >= exit_date]
            if not entry_prices_idx or not exit_prices_idx:
                continue
            actual_entry = entry_prices_idx[0]
            actual_exit = exit_prices_idx[0]
            if actual_entry not in prices.index or actual_exit not in prices.index:
                continue
            period_returns = []
            for ticker in top_stocks:
                if ticker not in prices.columns:
                    continue
                p_entry = prices.loc[actual_entry, ticker]
                p_exit = prices.loc[actual_exit, ticker]
                if pd.isna(p_entry) or pd.isna(p_exit) or p_entry <= 0:
                    continue
                ret = (p_exit / p_entry) - 1
                if ret < stop_loss:
                    ret = stop_loss
                ret -= cost * 2
                period_returns.append(ret)
            if period_returns:
                year = entry_date[:4]
                if year not in yearly_returns_map:
                    yearly_returns_map[year] = []
                yearly_returns_map[year].append(np.mean(period_returns))
    
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


# ═══════════════════════════════════════════════════
#  IC权重分析
# ═══════════════════════════════════════════════════

def analyze_ic_weights(ic_history, factor_groups):
    """分析IC权重的分布——哪些因子获得了更高权重。"""
    # 取最后一个日期的IC权重作为参考
    last_date = max(ic_history.keys())
    ic = ic_history[last_date]
    
    result = {}
    for group_name, factors in factor_groups.items():
        available = [f for f in factors if f in ic]
        if not available:
            continue
        
        ic_values = {f: ic[f] for f in available}
        positive = {f: v for f, v in ic_values.items() if v > 0}
        total = sum(positive.values())
        
        if total > 0:
            weights = {f: round(positive[f] / total, 4) for f in positive}
        else:
            weights = {f: round(1.0 / len(available), 4) for f in available}
        
        result[group_name] = {
            'ic_values': {f: round(ic_values[f], 5) for f in available},
            'ic_weights': weights,
            'n_positive': len(positive),
            'n_total': len(available),
        }
    
    return result


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Falcon: IC加权 vs 等权 (5-Year Weekly)")
    print("=" * 70)
    
    t_total = time.time()
    df, prices = load_data()
    
    # 收集所有因子
    all_factors = []
    for factors in FACTOR_GROUPS.values():
        all_factors.extend(factors)
    all_factors = list(set(all_factors))
    
    # 最近5年
    all_dates = sorted(df['date'].unique())
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"\n📅 范围: {sample_dates[0]} → {sample_dates[-1]} ({len(sample_dates)}天)")
    
    # 计算截面排名
    ranks = compute_all_ranks(df, all_factors, sample_dates)
    
    # ── 等权baseline ──
    print("\n" + "=" * 50)
    print("  等权 Baseline")
    print("=" * 50)
    eq_scores = compute_equal_weighted_scores(ranks, FACTOR_GROUPS, GC_WEIGHTS, MODEL_WEIGHTS)
    eq_result = weekly_backtest(eq_scores, prices)
    
    # ── 计算滚动IC ──
    print("\n" + "=" * 50)
    print("  滚动IC计算")
    print("=" * 50)
    ic_history = compute_rolling_ic(ranks, prices, all_factors, lookback=252)
    
    # ── IC加权 ──
    print("\n" + "=" * 50)
    print("  IC加权")
    print("=" * 50)
    ic_scores, weight_details = compute_ic_weighted_scores(
        ranks, ic_history, FACTOR_GROUPS, GC_WEIGHTS, MODEL_WEIGHTS)
    ic_result = weekly_backtest(ic_scores, prices)
    
    # ── IC权重分析 ──
    ic_weight_analysis = analyze_ic_weights(ic_history, FACTOR_GROUPS)
    
    # ── 对比 ──
    print("\n" + "=" * 70)
    print("  📊 对比结果")
    print("=" * 70)
    
    print(f"\n{'指标':<15} {'等权':>12} {'IC加权':>12} {'差异':>12}")
    print("-" * 51)
    print(f"{'Sharpe':<15} {eq_result['sharpe']:>12.3f} {ic_result['sharpe']:>12.3f} {ic_result['sharpe']-eq_result['sharpe']:>12.3f}")
    print(f"{'CAGR':<15} {eq_result['cagr']:>12.1%} {ic_result['cagr']:>12.1%} {ic_result['cagr']-eq_result['cagr']:>12.1%}")
    print(f"{'MaxDD':<15} {eq_result['max_dd']:>12.1%} {ic_result['max_dd']:>12.1%} {ic_result['max_dd']-eq_result['max_dd']:>12.1%}")
    print(f"{'Win Rate':<15} {eq_result['win_rate']:>12.1%} {ic_result['win_rate']:>12.1%} {ic_result['win_rate']-eq_result['win_rate']:>12.1%}")
    print(f"{'总收益':<15} {eq_result['total_return']:>12.1%} {ic_result['total_return']:>12.1%} {ic_result['total_return']-eq_result['total_return']:>12.1%}")
    
    # 按年对比
    print(f"\n📅 按年对比:")
    print(f"{'年份':<6} {'等权Sharpe':>12} {'IC加权Sharpe':>14} {'等权CAGR':>12} {'IC加权CAGR':>14} {'赢者':>6}")
    print("-" * 62)
    
    all_years = sorted(set(list(eq_result.get('yearly', {}).keys()) + list(ic_result.get('yearly', {}).keys())))
    for year in all_years:
        eq_y = eq_result.get('yearly', {}).get(year, {})
        ic_y = ic_result.get('yearly', {}).get(year, {})
        eq_s = eq_y.get('sharpe', 0)
        ic_s = ic_y.get('sharpe', 0)
        eq_c = eq_y.get('cagr', 0)
        ic_c = ic_y.get('cagr', 0)
        winner = 'IC' if ic_s > eq_s else '等权'
        print(f"{year:<6} {eq_s:>12.3f} {ic_s:>14.3f} {eq_c:>12.1%} {ic_c:>14.1%} {winner:>6}")
    
    winner = 'IC加权' if ic_result['sharpe'] > eq_result['sharpe'] else '等权'
    print(f"\n🏆 总体赢家: {winner}")
    
    # 打印IC权重分析
    print(f"\n📊 IC权重分布 (最后日期):")
    for group, data in ic_weight_analysis.items():
        print(f"\n  {group} ({data['n_positive']}/{data['n_total']}个因子正IC):")
        sorted_weights = sorted(data['ic_weights'].items(), key=lambda x: -x[1])
        for f, w in sorted_weights[:5]:
            ic_val = data['ic_values'].get(f, 0)
            print(f"    {f:<40} IC={ic_val:>8.5f}  权重={w:>6.1%}")
        if len(sorted_weights) > 5:
            print(f"    ... 还有{len(sorted_weights)-5}个因子")
    
    print(f"\n⏱️ 总耗时: {time.time()-t_total:.0f}秒")
    
    # 保存
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'range': f"{sample_dates[0]} → {sample_dates[-1]}",
            'frequency': 'weekly',
            'top_n': 10,
            'hold_days': 7,
            'ic_lookback': 252,
        },
        'equal_weighted': eq_result,
        'ic_weighted': ic_result,
        'ic_weight_analysis': ic_weight_analysis,
        'winner': winner,
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n💾 已保存: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
