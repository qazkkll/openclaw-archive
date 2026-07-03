#!/usr/bin/env python3
"""
🦅 Falcon V0.4.4 vs V0.4.5: 5-Year Weekly Backtest Comparison
================================================================
Walk-Forward回测，周频（每周一调仓），对比两个版本的表现。

V0.4.4配置:
  fund_ratio=0.45 + growth_composite=0.20 + qoq=0.20 + cashflow=0.15
  growth_composite: 0.60×fund_growth + 0.25×analyst + 0.15×income
  53因子, 翻转修正版

V0.4.5配置:
  fund_ratio=0.40 + growth_composite=0.30 + qoq=0.15 + cashflow=0.15
  growth_composite: 0.60×fund_growth + 0.40×analyst, income=0%
  43因子（移除9个负ICIR因子）

Walk-Forward: train=12个月, test=6个月, 每周一调仓, 持有7天
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "v044_vs_v045_5yr_weekly.json"

# ═══════════════════════════════════════════════════
#  V0.4.4 因子组定义 (53因子，翻转修正版)
# ═══════════════════════════════════════════════════

V044_FACTOR_GROUPS = {
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

V044_GC_WEIGHTS = {'fund_growth': 0.60, 'analyst': 0.25, 'income': 0.15}
V044_WEIGHTS = {'fund_ratio': 0.45, 'gc_baseline': 0.20, 'qoq': 0.20, 'cashflow': 0.15}

# V0.4.4 翻转修正版 (2026-07-03)
V044_FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity',
    'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}

# ═══════════════════════════════════════════════════
#  V0.4.5 因子组定义 (43因子，移除9个负ICIR因子)
# ═══════════════════════════════════════════════════

V045_FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_assetGrowth', 'g_bookValueperShareGrowth',
    ],
    'analyst': [
        'a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps',
    ],
    'income': [
        'i_gross_margin', 'i_ebitda_margin',
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

V045_GC_WEIGHTS = {'fund_growth': 0.60, 'analyst': 0.40, 'income': 0.00}
V045_WEIGHTS = {'fund_ratio': 0.40, 'gc_baseline': 0.30, 'qoq': 0.15, 'cashflow': 0.15}

# V0.4.5 翻转 (只保留使用中的因子)
V045_FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'c_capex_intensity',
    'a_eps_revision', 'a_revenue_revision',
}


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    print("📂 加载数据...")
    t0 = time.time()
    
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    print(f"  ✅ Features: {df.shape[0]}行 × {df.shape[1]}列, {df['ticker'].nunique()}只")
    
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")
    print(f"  ⏱️ 加载耗时: {time.time()-t0:.1f}秒")
    return df, price_pivot


# ═══════════════════════════════════════════════════
#  截面百分位排名
# ═══════════════════════════════════════════════════

def compute_ranks_for_dates(df, factor_cols, flip_factors, sample_dates):
    """计算指定日期的截面百分位排名。"""
    print(f"📊 计算 {len(sample_dates)} 天的截面排名...")
    t0 = time.time()
    
    ranks = {}
    for date in sample_dates:
        day_df = df[df['date'] == date].copy()
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
            
            ranks_raw = np.full_like(vals, np.nan)
            if valid.sum() > 0:
                ranks_raw[valid] = rankdata(vals[valid], method='average') / valid.sum()
            
            if col in flip_factors:
                mask = ~np.isnan(ranks_raw)
                ranks_raw[mask] = 1.0 - ranks_raw[mask]
            
            rank_df[col] = ranks_raw
        
        ranks[date] = rank_df
    
    print(f"  ✅ {len(ranks)}天排名计算完成 ({time.time()-t0:.0f}秒)")
    return ranks


def compute_group_scores(ranks, factor_groups, gc_weights, model_weights):
    """计算因子组分数和最终组合分数。"""
    print("📊 计算因子组分数...")
    t0 = time.time()
    
    all_dates = sorted(ranks.keys())
    scores_dict = {}
    
    for date in all_dates:
        df = ranks[date]
        
        # 计算各因子组分数
        group_scores = {}
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                group_scores[group_name] = df[available].mean(axis=1)
            else:
                group_scores[group_name] = pd.Series(0.0, index=df.index)
        
        # growth_composite
        gc = (gc_weights.get('fund_growth', 0) * group_scores.get('fund_growth', 0) +
              gc_weights.get('analyst', 0) * group_scores.get('analyst', 0) +
              gc_weights.get('income', 0) * group_scores.get('income', 0))
        
        # 最终分数
        final_score = (model_weights['fund_ratio'] * group_scores.get('fund_ratio', 0) +
                       model_weights['gc_baseline'] * gc +
                       model_weights['qoq'] * group_scores.get('qoq', 0) +
                       model_weights['cashflow'] * group_scores.get('cashflow', 0))
        
        scores_dict[date] = final_score.dropna().sort_values(ascending=False)
    
    print(f"  ✅ {len(scores_dict)}天分数计算完成 ({time.time()-t0:.0f}秒)")
    return scores_dict


# ═══════════════════════════════════════════════════
#  周频Walk-Forward回测
# ═══════════════════════════════════════════════════

def weekly_backtest(scores_dict, prices, top_n=10, cost=0.001, stop_loss=-0.15,
                    train_months=12, test_months=6):
    """周频Walk-Forward回测。
    
    每周一调仓，持有7天。
    """
    print(f"\n🔄 周频Walk-Forward回测 (train={train_months}mo, test={test_months}mo, top_n={top_n})")
    t0 = time.time()
    
    all_dates = sorted(scores_dict.keys())
    price_dates = sorted(prices.index.astype(str))
    
    # 确定Walk-Forward窗口
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
    
    print(f"  窗口数: {len(windows)}")
    
    # 每个窗口内做周频回测
    all_weekly_returns = []
    all_trades = []
    window_details = []
    
    for wi, w in enumerate(windows):
        # 在test期间，每周一调仓
        test_dates = [d for d in all_dates if w['test_start'] <= d <= w['test_end']]
        
        # 获取每周的第一个交易日（近似周一）
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
            window_details.append({'window': wi, 'period': f"{w['test_start']} → {w['test_end']}", 
                                   'error': 'insufficient weekly dates'})
            continue
        
        window_returns = []
        window_trades = []
        window_weekly_with_dates = []  # (date, return)
        
        for i in range(len(weekly_dates) - 1):
            entry_date = weekly_dates[i]
            exit_date = weekly_dates[i + 1]
            
            # 获取分数
            if entry_date not in scores_dict:
                continue
            scores = scores_dict[entry_date]
            top_stocks = scores.head(top_n).index.tolist()
            
            # 获取价格
            entry_prices_idx = [d for d in price_dates if d >= entry_date]
            exit_prices_idx = [d for d in price_dates if d >= exit_date]
            
            if not entry_prices_idx or not exit_prices_idx:
                continue
            
            actual_entry = entry_prices_idx[0]
            actual_exit = exit_prices_idx[0]
            
            if actual_entry not in prices.index or actual_exit not in prices.index:
                continue
            
            # 计算每只股票的收益
            period_returns = []
            for ticker in top_stocks:
                if ticker not in prices.columns:
                    continue
                p_entry = prices.loc[actual_entry, ticker]
                p_exit = prices.loc[actual_exit, ticker]
                
                if pd.isna(p_entry) or pd.isna(p_exit) or p_entry <= 0:
                    continue
                
                ret = (p_exit / p_entry) - 1
                
                # 止损检查
                if ret < stop_loss:
                    ret = stop_loss
                
                ret -= cost * 2  # 双边交易成本
                period_returns.append(ret)
                
                window_trades.append({
                    'entry': entry_date,
                    'exit': exit_date,
                    'ticker': ticker,
                    'return': round(ret, 4),
                })
            
            if period_returns:
                avg_ret = np.mean(period_returns)
                window_returns.append(avg_ret)
                all_weekly_returns.append(avg_ret)
                all_trades.extend(window_trades)
                window_weekly_with_dates.append((entry_date, avg_ret))
        
        if window_returns:
            wr = np.array(window_returns)
            sharpe = np.sqrt(52) * wr.mean() / wr.std() if wr.std() > 0 else 0
            cum = (1 + wr).cumprod()
            max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
            cagr = cum[-1] ** (52 / len(wr)) - 1 if len(wr) > 0 else 0
            win_rate = (wr > 0).mean()
            
            window_details.append({
                'window': wi,
                'period': f"{w['test_start']} → {w['test_end']}",
                'sharpe': round(sharpe, 3),
                'cagr': round(cagr, 4),
                'max_dd': round(max_dd, 4),
                'win_rate': round(win_rate, 3),
                'n_weeks': len(window_returns),
                'avg_weekly_ret': round(wr.mean(), 5),
            })
        else:
            window_details.append({'window': wi, 'period': f"{w['test_start']} → {w['test_end']}", 
                                   'error': 'no trades'})
    
    # 总体统计
    all_wr = np.array(all_weekly_returns) if all_weekly_returns else np.array([0])
    total_sharpe = np.sqrt(52) * all_wr.mean() / all_wr.std() if all_wr.std() > 0 else 0
    cum = (1 + all_wr).cumprod()
    total_max_dd = (cum / np.maximum.accumulate(cum) - 1).min()
    total_cagr = cum[-1] ** (52 / len(all_wr)) - 1 if len(all_wr) > 0 else 0
    total_win_rate = (all_wr > 0).mean()
    total_return = cum[-1] - 1
    
    # 按年统计 (从所有窗口的weekly returns with dates中提取)
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
        
        # 对每个window，重新计算周收益
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
        total_ret = yr_cum[-1] - 1
        cagr = yr_cum[-1] ** (52 / len(yr)) - 1
        sharpe = np.sqrt(52) * yr.mean() / yr.std() if yr.std() > 0 else 0
        yearly_stats[year] = {
            'cagr': round(float(cagr), 4),
            'total_return': round(float(total_ret), 4),
            'sharpe': round(float(sharpe), 3),
            'max_dd': round(float((yr_cum / np.maximum.accumulate(yr_cum) - 1).min()), 4),
            'win_rate': round(float((yr > 0).mean()), 3),
            'n_weeks': len(yr),
        }
    
    elapsed = time.time() - t0
    print(f"  ✅ 回测完成 ({elapsed:.0f}秒)")
    print(f"  总交易: {len(all_trades)}, 周数: {len(all_weekly_returns)}")
    
    return {
        'sharpe': round(total_sharpe, 3),
        'cagr': round(total_cagr, 4),
        'max_dd': round(total_max_dd, 4),
        'win_rate': round(total_win_rate, 3),
        'total_return': round(total_return, 4),
        'n_weeks': len(all_weekly_returns),
        'n_trades': len(all_trades),
        'yearly': yearly_stats,
        'windows': window_details,
    }


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Falcon V0.4.4 vs V0.4.5: 5-Year Weekly Walk-Forward")
    print("=" * 70)
    
    t_total = time.time()
    df, prices = load_data()
    
    # 获取所有可用日期
    all_dates = sorted(df['date'].unique())
    # 只取最近5年 (约2021-2026)
    five_years_ago = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    sample_dates = [d for d in all_dates if d >= five_years_ago]
    print(f"\n📅 回测范围: {sample_dates[0]} → {sample_dates[-1]} ({len(sample_dates)}天)")
    
    # 收集所有需要的因子列
    v044_factors = []
    for factors in V044_FACTOR_GROUPS.values():
        v044_factors.extend(factors)
    v044_factors = list(set(v044_factors))
    
    v045_factors = []
    for factors in V045_FACTOR_GROUPS.values():
        v045_factors.extend(factors)
    v045_factors = list(set(v045_factors))
    
    all_factors = list(set(v044_factors + v045_factors))
    
    # ── V0.4.4 ──
    print("\n" + "=" * 50)
    print("  V0.4.4: 53因子, 翻转修正版")
    print("=" * 50)
    
    v044_ranks = compute_ranks_for_dates(df, v044_factors, V044_FLIP_FACTORS, sample_dates)
    v044_scores = compute_group_scores(v044_ranks, V044_FACTOR_GROUPS, V044_GC_WEIGHTS, V044_WEIGHTS)
    v044_result = weekly_backtest(v044_scores, prices, top_n=10, cost=0.001, stop_loss=-0.15,
                                  train_months=12, test_months=6)
    
    # ── V0.4.5 ──
    print("\n" + "=" * 50)
    print("  V0.4.5: 43因子, 修剪版")
    print("=" * 50)
    
    v045_ranks = compute_ranks_for_dates(df, v045_factors, V045_FLIP_FACTORS, sample_dates)
    v045_scores = compute_group_scores(v045_ranks, V045_FACTOR_GROUPS, V045_GC_WEIGHTS, V045_WEIGHTS)
    v045_result = weekly_backtest(v045_scores, prices, top_n=10, cost=0.001, stop_loss=-0.15,
                                  train_months=12, test_months=6)
    
    # ── 对比 ──
    print("\n" + "=" * 70)
    print("  📊 对比结果")
    print("=" * 70)
    
    comparison = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'backtest_range': f"{sample_dates[0]} → {sample_dates[-1]}",
            'frequency': 'weekly',
            'top_n': 10,
            'hold_days': 7,
            'cost': 0.001,
            'stop_loss': -0.15,
            'train_months': 12,
            'test_months': 6,
        },
        'v044': v044_result,
        'v045': v045_result,
        'comparison': {
            'sharpe_diff': round(v044_result['sharpe'] - v045_result['sharpe'], 3),
            'cagr_diff': round(v044_result['cagr'] - v045_result['cagr'], 4),
            'max_dd_diff': round(v044_result['max_dd'] - v045_result['max_dd'], 4),
            'win_rate_diff': round(v044_result['win_rate'] - v045_result['win_rate'], 3),
            'winner': 'V0.4.4' if v044_result['sharpe'] > v045_result['sharpe'] else 'V0.4.5',
        },
    }
    
    # 打印对比表
    print(f"\n{'指标':<15} {'V0.4.4':>12} {'V0.4.5':>12} {'差异':>12}")
    print("-" * 51)
    print(f"{'Sharpe':<15} {v044_result['sharpe']:>12.3f} {v045_result['sharpe']:>12.3f} {comparison['comparison']['sharpe_diff']:>12.3f}")
    print(f"{'CAGR':<15} {v044_result['cagr']:>12.1%} {v045_result['cagr']:>12.1%} {comparison['comparison']['cagr_diff']:>12.1%}")
    print(f"{'MaxDD':<15} {v044_result['max_dd']:>12.1%} {v045_result['max_dd']:>12.1%} {comparison['comparison']['max_dd_diff']:>12.1%}")
    print(f"{'Win Rate':<15} {v044_result['win_rate']:>12.1%} {v045_result['win_rate']:>12.1%} {comparison['comparison']['win_rate_diff']:>12.1%}")
    print(f"{'总收益':<15} {v044_result['total_return']:>12.1%} {v045_result['total_return']:>12.1%} {v044_result['total_return']-v045_result['total_return']:>12.1%}")
    print(f"{'周数':<15} {v044_result['n_weeks']:>12} {v045_result['n_weeks']:>12}")
    
    # 按年对比
    print(f"\n📅 按年对比:")
    print(f"{'年份':<6} {'V0.4.4 CAGR':>12} {'V0.4.5 CAGR':>12} {'V0.4.4 Sharpe':>14} {'V0.4.5 Sharpe':>14} {'赢者':>6}")
    print("-" * 64)
    
    all_years = sorted(set(list(v044_result.get('yearly', {}).keys()) + list(v045_result.get('yearly', {}).keys())))
    for year in all_years:
        v44 = v044_result.get('yearly', {}).get(year, {})
        v45 = v045_result.get('yearly', {}).get(year, {})
        
        v44_cagr = v44.get('cagr', 'N/A')
        v45_cagr = v45.get('cagr', 'N/A')
        v44_sharpe = v44.get('sharpe', 'N/A')
        v45_sharpe = v45.get('sharpe', 'N/A')
        
        if isinstance(v44_cagr, (int, float)) and isinstance(v45_cagr, (int, float)):
            winner = 'V0.4.4' if v44_sharpe > v45_sharpe else 'V0.4.5'
            print(f"{year:<6} {v44_cagr:>12.1%} {v45_cagr:>12.1%} {v44_sharpe:>14.3f} {v45_sharpe:>14.3f} {winner:>6}")
        else:
            print(f"{year:<6} {'N/A':>12} {'N/A':>12} {'N/A':>14} {'N/A':>14}")
    
    print(f"\n🏆 总体赢家: {comparison['comparison']['winner']}")
    print(f"⏱️ 总耗时: {time.time()-t_total:.0f}秒")
    
    # 保存结果
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\n💾 结果已保存: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
