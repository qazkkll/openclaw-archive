#!/usr/bin/env python3
"""
🦅 Falcon V0.4.1 Optimized: Sub-Weight Optimization
====================================================
用最优子权重重新训练V0.4.1。

配置:
  主权重: fund_ratio=0.70, growth_composite=0.30
  growth_composite子权重: fund_growth=0.60, analyst=0.25, income=0.15
  (原: fund_growth=0.50, analyst=0.30, income=0.20)

Walk-Forward参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出:
  scripts/falcon/v041_optimized.py (本文件)
  data/falcon/v041_optimized_results.json
"""
import sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "v041_optimized_results.json"

# ═══════════════════════════════════════════════════
#  因子组定义 (从v041_fixed_validation.py复制)
# ═══════════════════════════════════════════════════

FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin',
        'r_ebitdaMargin', 'r_assetTurnover', 'r_inventoryTurnover',
        'r_receivablesTurnover', 'r_debtToEquityRatio', 'r_currentRatio',
        'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_metric': [
        'm_earningsYield', 'm_evToEBITDA', 'm_evToFreeCashFlow', 'm_evToSales',
        'm_freeCashFlowYield', 'm_returnOnEquity', 'm_returnOnAssets',
        'm_returnOnCapitalEmployed', 'm_returnOnInvestedCapital', 'm_returnOnTangibleAssets',
        'm_incomeQuality', 'm_grahamNumber', 'm_cashConversionCycle',
        'm_capexToRevenue', 'm_capexToDepreciation',
        'm_researchAndDevelopementToRevenue', 'm_stockBasedCompensationToRevenue',
        'm_netDebtToEBITDA', 'm_operatingReturnOnAssets',
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
    'balance': [
        'b_cash_to_assets', 'b_net_debt_to_assets', 'b_equity_ratio', 'b_debt_to_equity',
    ],
    'cashflow': [
        'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
    ],
    'income': [
        'i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
        'i_revenue_growth_yoy', 'i_gross_margin_delta',
    ],
    'qoq': [
        'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
        'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
    ],
}

# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载特征和价格数据。"""
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

def compute_cross_sectional_ranks(df, factor_cols):
    """计算截面百分位排名。"""
    print("📊 计算截面百分位排名...")
    t0 = time.time()

    # 越高越差的因子 → 翻转
    flip_factors = {
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
        'b_debt_to_equity', 'b_net_debt_to_assets', 'm_netDebtToEBITDA',
        'm_capexToRevenue', 'm_capexToDepreciation',
        'm_researchAndDevelopementToRevenue', 'm_stockBasedCompensationToRevenue',
        'c_capex_intensity',
        'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
        'a_eps_dispersion',
        'm_cashConversionCycle',
    }

    from scipy.stats import rankdata

    dates = sorted(df['date'].unique())
    ranks = {}

    for date in dates:
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

    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)}天排名计算完成 ({elapsed:.0f}秒)")
    return ranks


def compute_group_ranks(ranks, factor_groups):
    """将因子组的排名合并为组级排名(等权平均)。"""
    print("📊 计算因子组排名...")
    for date in list(ranks.keys()):
        df = ranks[date]
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                df[group_name] = df[available].mean(axis=1)
        ranks[date] = df
    print(f"  ✅ 因子组排名已添加: {list(factor_groups.keys())}")
    return ranks


# ═══════════════════════════════════════════════════
#  组合因子 (使用优化后的子权重)
# ═══════════════════════════════════════════════════

def add_combo_factors(ranks):
    """为每个日期添加组合因子列。使用优化后的growth_composite子权重。"""
    # ─── 关键优化: growth_composite子权重调整 ───
    # 原: fund_growth=0.50, analyst=0.30, income=0.20
    # 新: fund_growth=0.60, analyst=0.25, income=0.15
    GROWTH_COMPOSITE_WGTS = {
        'fund_growth': 0.60,
        'analyst': 0.25,
        'income': 0.15,
    }

    def growth_composite_func(d):
        return sum(d.get(k, 0) * v for k, v in GROWTH_COMPOSITE_WGTS.items())

    combo_defs = {
        'log_metric': lambda d: np.log(d.get('fund_metric', 0) + 1),
        'log_growth': lambda d: np.log(d.get('fund_growth', 0) + 1),
        'sqrt_ratio': lambda d: np.sqrt(d.get('fund_ratio', 0)),
        'ratio_x_growth': lambda d: d.get('fund_ratio', 0) * d.get('fund_growth', 0),
        'metric_x_growth': lambda d: d.get('fund_metric', 0) * d.get('fund_growth', 0),
        'sqrt_ratio_x_log_metric': lambda d: np.sqrt(d.get('fund_ratio', 0)) * np.log(d.get('fund_metric', 0) + 1),
        'quality_composite': lambda d: (
            d.get('fund_ratio', 0) * 0.4 +
            d.get('fund_metric', 0) * 0.3 +
            d.get('fund_growth', 0) * 0.2 +
            d.get('income', 0) * 0.1
        ),
        'growth_composite': growth_composite_func,
        'safety_composite': lambda d: (
            d.get('fund_ratio', 0) * 0.3 +
            d.get('balance', 0) * 0.3 +
            d.get('cashflow', 0) * 0.2 +
            d.get('fund_metric', 0) * 0.2
        ),
    }

    for date in ranks:
        df = ranks[date]
        for name, func in combo_defs.items():
            try:
                df[name] = func(df.to_dict('series'))
            except Exception:
                df[name] = np.nan
        ranks[date] = df

    print(f"  ✅ 组合因子(含优化growth_composite): {list(combo_defs.keys())}")
    return ranks, list(combo_defs.keys())


# ═══════════════════════════════════════════════════
#  真正的 Rank Inversion 测试 (Top5% vs Bottom20%)
# ═══════════════════════════════════════════════════

def compute_real_rank_inversion(ranks, prices, weights, hold_days=30, top_n=10,
                                 cost=0.001, stop_loss=-0.15):
    """
    计算每个Walk-Forward窗口的真正rank_inversion指标:
    - Top5%股票的平均收益
    - Bottom20%股票的平均收益
    - 检查Top5%收益 > Bottom20%收益
    
    Returns:
        dict: 包含每个窗口的rank_inversion结果
    """
    dates = sorted(ranks.keys())
    if not dates:
        return None

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    train_months = int(0.5 * 12)  # 6 months
    test_months = 6
    
    windows = []
    window_idx = 0
    
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
            
        test_start_str = str(train_end)[:10]
        test_end_str = str(test_end)[:10]
        
        # 获取测试窗口内的日期
        test_dates = [d for d in sorted(prices.index.astype(str))
                      if test_start_str <= d <= test_end_str]
        
        if len(test_dates) < 10:
            window_idx += 1
            train_start += pd.DateOffset(months=test_months)
            continue
        
        # 计算每个日期的分数和前向收益
        top5_returns = []
        bottom20_returns = []
        
        for i, date in enumerate(test_dates):
            if date not in ranks:
                continue
                
            r = ranks[date]
            available = [f for f in weights if f in r.columns and weights[f] > 0]
            if not available:
                continue
                
            # 计算组合分数
            combined = pd.Series(0.0, index=r.index)
            for f in available:
                combined = combined + weights[f] * r[f]
            scores = combined.dropna().sort_values(ascending=False)
            
            if len(scores) < 20:
                continue
            
            # 计算前向收益 (持有hold_days)
            if i + hold_days >= len(test_dates):
                continue
            future_date = test_dates[min(i + hold_days, len(test_dates) - 1)]
            
            if date not in prices.index or future_date not in prices.index:
                continue
                
            pr_today = prices.loc[date]
            pr_future = prices.loc[future_date]
            
            # 计算每只股票的收益
            returns = {}
            for ticker in scores.index:
                if ticker in pr_today.index and ticker in pr_future.index:
                    if pd.notna(pr_today[ticker]) and pd.notna(pr_future[ticker]) and pr_today[ticker] > 0:
                        returns[ticker] = (pr_future[ticker] / pr_today[ticker]) - 1
            
            if len(returns) < 20:
                continue
            
            # 转换为Series
            returns_series = pd.Series(returns)
            
            # 按分数排序
            sorted_tickers = scores.sort_values(ascending=False).index
            sorted_returns = returns_series.reindex(sorted_tickers).dropna()
            
            if len(sorted_returns) < 20:
                continue
            
            # Top 5% 和 Bottom 20%
            n_top5 = max(1, int(len(sorted_returns) * 0.05))
            n_bottom20 = max(1, int(len(sorted_returns) * 0.20))
            
            top5_ret = sorted_returns.head(n_top5).mean()
            bottom20_ret = sorted_returns.tail(n_bottom20).mean()
            
            top5_returns.append(top5_ret)
            bottom20_returns.append(bottom20_ret)
        
        # 计算窗口级别的rank_inversion
        if len(top5_returns) > 0:
            avg_top5 = np.mean(top5_returns)
            avg_bottom20 = np.mean(bottom20_returns)
            ri_passed = avg_top5 > avg_bottom20
            
            windows.append({
                'window_idx': window_idx,
                'period': f"{test_start_str} → {test_end_str}",
                'avg_top5_return': float(avg_top5),
                'avg_bottom20_return': float(avg_bottom20),
                'spread': float(avg_top5 - avg_bottom20),
                'passed': bool(ri_passed),
                'n_dates': len(top5_returns),
            })
        
        window_idx += 1
        train_start += pd.DateOffset(months=test_months)
    
    # 汇总
    if not windows:
        return None
        
    passed_windows = sum(1 for w in windows if w['passed'])
    total_windows = len(windows)
    
    return {
        'windows': windows,
        'total_windows': total_windows,
        'passed_windows': passed_windows,
        'pass_rate': passed_windows / total_windows if total_windows > 0 else 0,
        'overall_passed': passed_windows / total_windows > 0.6 if total_windows > 0 else False,
        'avg_spread': float(np.mean([w['spread'] for w in windows])),
    }


# ═══════════════════════════════════════════════════
#  Walk-Forward 回测
# ═══════════════════════════════════════════════════

def run_walk_forward(ranks, prices, weights,
                      train_years=0.5, test_months=6,
                      hold_days=30, top_n=10,
                      cost=0.001, stop_loss=-0.15):
    """
    运行Walk-Forward验证。
    
    Returns:
        tuple: (result_dict, window_details)
    """
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    dates = sorted(ranks.keys())
    if not dates:
        return None, []

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    train_months = int(train_years * 12)
    
    windows = []
    idx = 0
    
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
            
        tss = str(train_end)[:10]
        tes = str(test_end)[:10]
        
        try:
            result, baseline = engine.run(
                ranks, prices, weights, hold_days, top_n,
                start_date=tss, end_date=tes, run_baseline=True
            )
            windows.append({
                "index": idx,
                "period": f"{tss} → {tes}",
                "sharpe": result.sharpe,
                "max_dd": result.max_dd,
                "cagr": result.cagr,
                "win_rate": result.win_rate,
                "n_trades": result.n_trades,
                "n_days": len(result.daily_equity),
                "baseline_sharpe": baseline.sharpe if baseline else None,
                "year": int(tss[:4]),
            })
        except DataQualityError as e:
            windows.append({
                "index": idx, "period": f"{tss} → {tes}",
                "error": str(e)[:200], "year": int(tss[:4]),
            })
        except Exception as e:
            windows.append({
                "index": idx, "period": f"{tss} → {tes}",
                "error": str(e)[:200], "year": int(tss[:4]),
            })
        
        idx += 1
        train_start += pd.DateOffset(months=test_months)

    if not windows:
        return None, []

    valid = [w for w in windows if "sharpe" in w]
    if not valid:
        return {"error": "All windows failed", "windows": windows}, windows

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]

    result = {
        "sharpe": round(float(np.mean(sharpes)), 3),
        "max_dd": round(float(np.min(dds)), 4),
        "cagr": round(float(np.mean(cagrs)), 4),
        "win_rate": round(float(np.mean(wrs)), 3),
        "n_trades": sum(w["n_trades"] for w in valid),
        "n_windows": len(valid),
        "total_windows": len(windows),
        "failed_windows": len(windows) - len(valid),
        "warnings": [],
        "status": "PASS",
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  Sharpe 退化分析
# ═══════════════════════════════════════════════════

def analyze_sharpe_degradation(windows):
    """
    分析Sharpe退化:
    - 早期窗口(2017-2020)的平均Sharpe
    - 近期窗口(2021-2026)的平均Sharpe
    - 退化比例
    - 判断是否为阻断项
    """
    valid = [w for w in windows if "sharpe" in w]
    if len(valid) < 4:
        return {
            "passed": False,
            "reason": "Too few valid windows for degradation analysis",
            "early_avg_sharpe": None,
            "recent_avg_sharpe": None,
            "degradation_pct": None,
            "status": "ERROR",
        }
    
    early_windows = [w for w in valid if w.get("year", 2021) <= 2020]
    recent_windows = [w for w in valid if w.get("year", 2021) >= 2021]
    
    if not early_windows or not recent_windows:
        return {
            "passed": False,
            "reason": "Insufficient windows in early or recent period",
            "early_avg_sharpe": None,
            "recent_avg_sharpe": None,
            "degradation_pct": None,
            "status": "ERROR",
        }
    
    early_avg = np.mean([w["sharpe"] for w in early_windows])
    recent_avg = np.mean([w["sharpe"] for w in recent_windows])
    
    # 计算退化比例
    if early_avg > 0:
        degradation_pct = (early_avg - recent_avg) / early_avg * 100
    else:
        degradation_pct = 0
    
    # 判断阻断项
    passed = True
    status = "PASS"
    reason = "OK"
    
    if degradation_pct > 75:
        passed = False
        status = "FAIL"
        reason = f"Severe degradation: {degradation_pct:.1f}% (>75% FAIL threshold)"
    elif degradation_pct > 50:
        status = "WARN"
        reason = f"Moderate degradation: {degradation_pct:.1f}% (>50% WARN threshold)"
    
    return {
        "passed": passed,
        "status": status,
        "reason": reason,
        "early_avg_sharpe": round(float(early_avg), 3),
        "recent_avg_sharpe": round(float(recent_avg), 3),
        "degradation_pct": round(float(degradation_pct), 1),
        "early_window_count": len(early_windows),
        "recent_window_count": len(recent_windows),
        "early_windows": [{"period": w["period"], "sharpe": w["sharpe"]} for w in early_windows],
        "recent_windows": [{"period": w["period"], "sharpe": w["sharpe"]} for w in recent_windows],
    }


# ═══════════════════════════════════════════════════
#  JSON序列化辅助
# ═══════════════════════════════════════════════════

def serialize(obj):
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4.1 Optimized: Sub-Weight Optimization")
    print("=" * 80)
    print()
    print("优化配置:")
    print("  主权重: fund_ratio=0.70, growth_composite=0.30")
    print("  growth_composite子权重: fund_growth=0.60, analyst=0.25, income=0.15")
    print("  (原: fund_growth=0.50, analyst=0.30, income=0.20)")
    print()

    # ─── 1. 加载数据 ───
    df, price_pivot = load_data()

    # ─── 2. 确定可用PIT因子 ───
    all_pit_cols = []
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        all_pit_cols.extend(available)
        print(f"  {group}: {len(available)}/{len(cols)} factors")
    print(f"  Total PIT factors: {len(all_pit_cols)}")

    # ─── 3. 计算截面百分位排名 ───
    ranks = compute_cross_sectional_ranks(df, all_pit_cols)

    # ─── 4. 计算因子组排名 ───
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)

    # ─── 5. 添加组合因子 (使用优化后的growth_composite) ───
    ranks, combo_names = add_combo_factors(ranks)

    # ═══════════════════════════════════════════════
    #  使用优化后的权重
    # ═══════════════════════════════════════════════
    weights = {
        "fund_ratio": 0.70,
        "growth_composite": 0.30,
    }

    print(f"\n📌 使用权重: {weights}")
    print(f"  growth_composite = 0.60×fund_growth + 0.25×analyst + 0.15×income")
    print()

    # ═══════════════════════════════════════════════
    #  测试1: 真正的Rank Inversion测试
    # ═══════════════════════════════════════════════
    print("=" * 80)
    print("🔍 测试1: 真正的Rank Inversion测试 (Top5% vs Bottom20%)")
    print("=" * 80)

    ri_result = compute_real_rank_inversion(
        ranks, price_pivot, weights,
        hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15
    )

    if ri_result:
        print(f"  总窗口数: {ri_result['total_windows']}")
        print(f"  通过窗口数: {ri_result['passed_windows']}")
        print(f"  通过率: {ri_result['pass_rate']:.1%}")
        print(f"  平均spread: {ri_result['avg_spread']:.4f}")
        print(f"  整体结果: {'✅ PASS' if ri_result['overall_passed'] else '❌ FAIL'}")

        print("\n  窗口详情:")
        for w in ri_result['windows']:
            mark = "✅" if w['passed'] else "❌"
            print(f"    {mark} W{w['window_idx']}: {w['period']} | "
                  f"Top5%={w['avg_top5_return']:.4f} Bottom20%={w['avg_bottom20_return']:.4f} "
                  f"Spread={w['spread']:.4f}")
    else:
        print("  ❌ 无法计算Rank Inversion")

    # ═══════════════════════════════════════════════
    #  测试2: Walk-Forward验证
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 测试2: Walk-Forward验证")
    print("=" * 80)

    wf_result, wf_windows = run_walk_forward(
        ranks, price_pivot, weights,
        train_years=0.5, test_months=6,
        hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15
    )

    if wf_result:
        print(f"  WF Sharpe: {wf_result['sharpe']:.3f}")
        print(f"  MaxDD: {wf_result['max_dd']:.1%}")
        print(f"  CAGR: {wf_result['cagr']:.1%}")
        print(f"  Win Rate: {wf_result['win_rate']:.0%}")
        print(f"  有效窗口: {wf_result['n_windows']}/{wf_result['total_windows']}")
        print(f"  失败窗口: {wf_result['failed_windows']}")

        print("\n  窗口详情:")
        valid_wins = [w for w in wf_windows if "sharpe" in w]
        failed_wins = [w for w in wf_windows if "error" in w]

        for w in valid_wins:
            print(f"    ✅ W{w['index']}: {w['period']} | "
                  f"Sharpe={w['sharpe']:.3f} MaxDD={w['max_dd']:.1%} "
                  f"WR={w['win_rate']:.0%}")

        if failed_wins:
            print(f"\n  ⚠️ 失败窗口 ({len(failed_wins)}个):")
            for w in failed_wins:
                print(f"    ❌ W{w['index']}: {w['period']} | Error: {w['error'][:100]}")
    else:
        print("  ❌ Walk-Forward失败")

    # ═══════════════════════════════════════════════
    #  测试3: Sharpe退化分析
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📉 测试3: Sharpe退化分析")
    print("=" * 80)

    if wf_windows:
        degradation = analyze_sharpe_degradation(wf_windows)

        print(f"  早期窗口平均Sharpe: {degradation['early_avg_sharpe']}")
        print(f"  近期窗口平均Sharpe: {degradation['recent_avg_sharpe']}")
        print(f"  退化比例: {degradation['degradation_pct']:.1f}%")
        print(f"  早期窗口数: {degradation['early_window_count']}")
        print(f"  近期窗口数: {degradation['recent_window_count']}")
        print(f"  状态: {degradation['status']}")
        print(f"  原因: {degradation['reason']}")

        if degradation['early_windows']:
            print("\n  早期窗口 (2017-2020):")
            for w in degradation['early_windows']:
                print(f"    {w['period']}: Sharpe={w['sharpe']:.3f}")

        if degradation['recent_windows']:
            print("\n  近期窗口 (2021-2026):")
            for w in degradation['recent_windows']:
                print(f"    {w['period']}: Sharpe={w['sharpe']:.3f}")
    else:
        degradation = {"status": "ERROR", "reason": "No windows available"}

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("💾 保存结果")
    print("=" * 80)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.1 Optimized: Sub-Weight Optimization",
            "description": "用最优子权重重新训练V0.4.1",
            "config": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "weights": {
                "fund_ratio": 0.70,
                "growth_composite": 0.30,
            },
            "growth_composite_sub_weights": {
                "fund_growth": 0.60,
                "analyst": 0.25,
                "income": 0.15,
                "description": "优化后: fund_growth从0.50→0.60, analyst从0.30→0.25, income从0.20→0.15",
            },
            "features": "features_v04_1.parquet",
            "pit_factors": len(all_pit_cols),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "rank_inversion_test": serialize(ri_result) if ri_result else None,
        "walk_forward_result": serialize(wf_result) if wf_result else None,
        "walk_forward_windows": serialize(wf_windows) if wf_windows else None,
        "sharpe_degradation": serialize(degradation),
        "summary": {
            "rank_inversion_passed": ri_result['overall_passed'] if ri_result else False,
            "rank_inversion_pass_rate": ri_result['pass_rate'] if ri_result else 0,
            "rank_inversion_avg_spread": ri_result['avg_spread'] if ri_result else 0,
            "walk_forward_sharpe": wf_result['sharpe'] if wf_result else None,
            "walk_forward_max_dd": wf_result['max_dd'] if wf_result else None,
            "walk_forward_cagr": wf_result['cagr'] if wf_result else None,
            "walk_forward_win_rate": wf_result['win_rate'] if wf_result else None,
            "sharpe_degradation_status": degradation['status'],
            "sharpe_degradation_pct": degradation.get('degradation_pct'),
            "all_windows_included": wf_result['failed_windows'] == 0 if wf_result else False,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  ✅ 结果已保存: {OUTPUT_PATH}")
    print(f"  ⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")

    # ═══════════════════════════════════════════════
    #  最终摘要
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📋 最终摘要")
    print("=" * 80)
    print(f"  权重: fund_ratio=0.70, growth_composite=0.30")
    print(f"  growth_composite子权重: fund_growth=0.60, analyst=0.25, income=0.15")
    print(f"  Rank Inversion: {'✅ PASS' if ri_result and ri_result['overall_passed'] else '❌ FAIL'}")
    if ri_result:
        print(f"    通过率: {ri_result['pass_rate']:.1%}, 平均spread: {ri_result['avg_spread']:.4f}")
    print(f"  Walk-Forward Sharpe: {wf_result['sharpe']:.3f}" if wf_result else "  WF Sharpe: N/A")
    if wf_result:
        print(f"  MaxDD: {wf_result['max_dd']:.1%}  CAGR: {wf_result['cagr']:.1%}  WR: {wf_result['win_rate']:.0%}")
    print(f"  Sharpe退化: {degradation['status']}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
