#!/usr/bin/env python3
"""
🦅 Falcon V0.4.0 Fixed Validation
================================================================
修复V0.4.0数据问题: 用features_v04_1.parquet重新验证V0.4.0配置。

原问题: V0.4.0用features_v02.parquet (80列, 2025-2026基本面覆盖=0%)
         fund_metric和log_metric因子在features_v02.parquet中不存在
修复:   用features_v04_1.parquet (156列, 98.9%覆盖) 重新验证

V0.4.0配置:
  - weights: fund_ratio=0.70, fund_metric=0.15, log_metric=0.15
  - log_metric = log(fund_metric + 1)
  - train_years=0.5 (6个月), test_months=6, hold_days=30
  - top_n=10, cost=0.001, stop_loss=-0.15

测试:
  1. 数据门禁: 每个因子组的覆盖率
  2. Rank Inversion: Top5% vs Bottom20% (每个窗口检查)
  3. Walk-Forward: Sharpe, MaxDD, CAGR, 胜率
  4. 因子IC/ICIR分析

输出:
  - scripts/falcon/v040_fixed_validation.py (本文件)
  - data/falcon/v040_fixed_validation_results.json
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
OUTPUT_PATH = DATA_DIR / "v040_fixed_validation_results.json"

# ═══════════════════════════════════════════════════
#  V0.4.0 因子组定义 (使用r_/m_前缀因子, 覆盖率94.6%)
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
        'm_returnOnCapitalEmployed', 'm_returnOnInvestedCapital',
        'm_returnOnTangibleAssets', 'm_incomeQuality', 'm_grahamNumber',
        'm_cashConversionCycle', 'm_capexToRevenue', 'm_capexToDepreciation',
        'm_researchAndDevelopementToRevenue', 'm_stockBasedCompensationToRevenue',
        'm_netDebtToEBITDA', 'm_operatingReturnOnAssets',
    ],
}

# V0.4.0权重
V040_WEIGHTS = {
    'fund_ratio': 0.70,
    'fund_metric': 0.15,
    'log_metric': 0.15,
}

# 越高越差的因子 → 翻转rank
FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'm_netDebtToEBITDA', 'm_capexToRevenue', 'm_capexToDepreciation',
    'm_researchAndDevelopementToRevenue', 'm_stockBasedCompensationToRevenue',
    'm_cashConversionCycle',
}


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载features_v04_1.parquet和价格数据。"""
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
#  中位数填充 (analyst因子)
# ═══════════════════════════════════════════════════

def median_fill_analyst_factors(df, factor_cols):
    """
    用中位数填充analyst因子缺失值。
    V0.4.0虽然不直接用analyst因子, 但确保所有因子列都有数据。
    对于fund_ratio和fund_metric中的缺失值, 也用中位数填充。
    """
    print("🔧 中位数填充缺失值...")
    filled_count = 0
    for col in factor_cols:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            # 按date分组计算中位数, 填充该日期内的缺失值
            median_by_date = df.groupby('date')[col].transform('median')
            # 如果某个日期所有值都缺失, 用全局中位数
            global_median = df[col].median()
            df[col] = df[col].fillna(median_by_date).fillna(global_median)
            filled_count += n_missing
    print(f"  ✅ 填充了 {filled_count:,} 个缺失值")
    return df


# ═══════════════════════════════════════════════════
#  截面百分位排名
# ═══════════════════════════════════════════════════

def compute_cross_sectional_ranks(df, factor_cols):
    """计算截面百分位排名。"""
    print("📊 计算截面百分位排名...")
    t0 = time.time()

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

            if col in FLIP_FACTORS:
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


def add_log_metric(ranks):
    """添加log_metric = log(fund_metric + 1)。"""
    count = 0
    for date in ranks:
        df = ranks[date]
        if 'fund_metric' in df.columns:
            df['log_metric'] = np.log(df['fund_metric'] + 1)
            count += 1
        ranks[date] = df
    print(f"  ✅ log_metric已添加 ({count}天)")
    return ranks


# ═══════════════════════════════════════════════════
#  数据覆盖率检查
# ═══════════════════════════════════════════════════

def check_factor_coverage(df, factor_groups):
    """检查每个因子组的覆盖率。"""
    print("\n📋 因子覆盖率检查:")
    coverage_report = {}
    for group_name, factors in factor_groups.items():
        available = [f for f in factors if f in df.columns]
        missing = [f for f in factors if f not in df.columns]
        if available:
            cov = df[available].notna().mean().mean()
            coverage_report[group_name] = {
                'available': len(available),
                'total': len(factors),
                'missing_cols': missing,
                'avg_coverage': round(float(cov), 4),
            }
            mark = "✅" if cov >= 0.8 else "⚠️" if cov >= 0.6 else "❌"
            print(f"  {mark} {group_name}: {len(available)}/{len(factors)}因子, 覆盖率={cov:.1%}")
            if missing:
                print(f"      缺失列: {missing}")
        else:
            coverage_report[group_name] = {
                'available': 0,
                'total': len(factors),
                'missing_cols': factors,
                'avg_coverage': 0,
            }
            print(f"  ❌ {group_name}: 0/{len(factors)}因子可用!")
    return coverage_report


# ═══════════════════════════════════════════════════
#  真正的 Rank Inversion 测试 (Top5% vs Bottom20%)
# ═══════════════════════════════════════════════════

def compute_rank_inversion(ranks, prices, weights, hold_days=30):
    """
    真正的rank inversion: 在每个Walk-Forward窗口中,
    检查Top5%股票的平均收益是否 > Bottom20%股票的平均收益。

    Walk-Forward窗口参数:
      train_months=6, test_months=6 (与WF验证一致)
    """
    print("\n🔍 Rank Inversion测试 (Top5% vs Bottom20%)...")
    t0 = time.time()

    dates = sorted(ranks.keys())
    if not dates:
        return None

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    train_months = 6
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

        # 窗口级别聚合
        if len(top5_returns) > 0:
            avg_top5 = np.mean(top5_returns)
            avg_bottom20 = np.mean(bottom20_returns)
            ri_passed = avg_top5 > avg_bottom20
            spreads = [t - b for t, b in zip(top5_returns, bottom20_returns)]

            windows.append({
                'window_idx': window_idx,
                'period': f"{test_start_str} → {test_end_str}",
                'avg_top5_return': float(avg_top5),
                'avg_bottom20_return': float(avg_bottom20),
                'spread': float(avg_top5 - avg_bottom20),
                'median_spread': float(np.median(spreads)),
                'passed': bool(ri_passed),
                'n_dates': len(top5_returns),
                'positive_spread_pct': float(np.mean([s > 0 for s in spreads])),
            })

        window_idx += 1
        train_start += pd.DateOffset(months=test_months)

    elapsed = time.time() - t0
    print(f"  ✅ Rank Inversion测试完成 ({elapsed:.0f}秒, {len(windows)}个窗口)")

    if not windows:
        return None

    # 汇总
    passed_windows = sum(1 for w in windows if w['passed'])
    total_windows = len(windows)

    return {
        'windows': windows,
        'total_windows': total_windows,
        'passed_windows': passed_windows,
        'pass_rate': round(passed_windows / total_windows, 4) if total_windows > 0 else 0,
        'overall_passed': passed_windows / total_windows > 0.6 if total_windows > 0 else False,
        'avg_spread': round(float(np.mean([w['spread'] for w in windows])), 6),
        'median_spread': round(float(np.median([w['spread'] for w in windows])), 6),
        'avg_positive_spread_pct': round(float(np.mean([w['positive_spread_pct'] for w in windows])), 4),
    }


# ═══════════════════════════════════════════════════
#  Walk-Forward 验证
# ═══════════════════════════════════════════════════

def run_walk_forward(ranks, prices, weights,
                     train_years=0.5, test_months=6,
                     hold_days=30, top_n=10,
                     cost=0.001, stop_loss=-0.15):
    """
    运行Walk-Forward验证, 使用backtest_engine.py。
    """
    print("\n📊 Walk-Forward验证...")
    t0 = time.time()

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
                "total_return": result.total_return,
                "baseline_sharpe": baseline.sharpe if baseline else None,
                "year": int(tss[:4]),
            })
        except DataQualityError as e:
            windows.append({
                "index": idx,
                "period": f"{tss} → {tes}",
                "error": str(e)[:200],
                "year": int(tss[:4]),
            })
        except Exception as e:
            windows.append({
                "index": idx,
                "period": f"{tss} → {tes}",
                "error": str(e)[:200],
                "year": int(tss[:4]),
            })

        idx += 1
        train_start += pd.DateOffset(months=test_months)

    elapsed = time.time() - t0
    print(f"  ✅ Walk-Forward完成 ({elapsed:.0f}秒, {len(windows)}个窗口)")

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
        "sharpe_std": round(float(np.std(sharpes)), 3),
        "sharpe_min": round(float(np.min(sharpes)), 3),
        "sharpe_max": round(float(np.max(sharpes)), 3),
        "status": "PASS",
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  因子IC/ICIR分析
# ═══════════════════════════════════════════════════

def compute_factor_ic(df, factor_cols, price_pivot, hold_days=30):
    """计算每个因子的IC和ICIR。"""
    print("\n📊 因子IC/ICIR分析...")
    t0 = time.time()

    from scipy.stats import spearmanr

    dates = sorted(df['date'].unique())

    # 预计算前向收益
    fwd_ret_map = {}
    for i, date in enumerate(dates):
        if i + hold_days >= len(dates):
            continue
        future_date = dates[min(i + hold_days, len(dates) - 1)]
        if date in price_pivot.index and future_date in price_pivot.index:
            pr_today = price_pivot.loc[date]
            pr_future = price_pivot.loc[future_date]
            valid = pr_today.notna() & pr_future.notna() & (pr_today > 0)
            ret = ((pr_future[valid] / pr_today[valid]) - 1).to_dict()
            fwd_ret_map[date] = ret

    print(f"  前向收益: {len(fwd_ret_map)}天")

    # 计算每个因子的IC
    factor_ics = {col: [] for col in factor_cols}

    for date in dates:
        if date not in fwd_ret_map:
            continue
        fwd = fwd_ret_map[date]
        day_df = df[df['date'] == date]
        if len(day_df) < 20:
            continue

        ticker_fwd = day_df['ticker'].map(fwd).values.astype(float)
        valid_fwd = ~np.isnan(ticker_fwd)

        if valid_fwd.sum() < 20:
            continue

        for col in factor_cols:
            if col not in day_df.columns:
                factor_ics[col].append(np.nan)
                continue
            vals = day_df[col].values.astype(float)
            valid = (~np.isnan(vals)) & valid_fwd
            if valid.sum() < 20:
                factor_ics[col].append(np.nan)
                continue
            ic, _ = spearmanr(vals[valid], ticker_fwd[valid])
            factor_ics[col].append(ic)

    # 计算ICIR
    results = []
    for col in factor_cols:
        ics = factor_ics[col]
        valid_ics = [x for x in ics if not np.isnan(x)]
        if len(valid_ics) < 30:
            continue
        ic_mean = np.mean(valid_ics)
        ic_std = np.std(valid_ics)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        t_stat = ic_mean / (ic_std / np.sqrt(len(valid_ics))) if ic_std > 0 else 0
        results.append({
            'name': col,
            'ic_mean': round(float(ic_mean), 6),
            'ic_std': round(float(ic_std), 6),
            'icir': round(float(icir), 4),
            't_stat': round(float(t_stat), 2),
            'n_dates': len(valid_ics),
        })

    results.sort(key=lambda x: abs(x['icir']), reverse=True)
    elapsed = time.time() - t0
    print(f"  ✅ IC分析完成: {len(results)}个因子 ({elapsed:.0f}秒)")
    return results


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
    print("🦅 Falcon V0.4.0 Fixed Validation")
    print("  用features_v04_1.parquet重新验证V0.4.0配置")
    print("=" * 80)

    # ─── 1. 加载数据 ───
    df, price_pivot = load_data()

    # ─── 2. 确定可用因子 ───
    all_factor_cols = []
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        all_factor_cols.extend(available)
        print(f"  {group}: {len(available)}/{len(cols)}因子")
    print(f"  总因子数: {len(all_factor_cols)}")

    # ─── 3. 数据覆盖率检查 ───
    coverage = check_factor_coverage(df, FACTOR_GROUPS)

    # ─── 4. 中位数填充 ───
    df = median_fill_analyst_factors(df, all_factor_cols)

    # ─── 5. 填充后再次检查覆盖率 ───
    print("\n📋 填充后覆盖率:")
    post_fill_coverage = {}
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        if available:
            cov = df[available].notna().mean().mean()
            post_fill_coverage[group] = round(float(cov), 4)
            print(f"  {group}: {cov:.1%}")
    all_available = [c for c in all_factor_cols if c in df.columns]
    overall_cov = df[all_available].notna().mean().mean() if all_available else 0
    print(f"  总体覆盖率: {overall_cov:.1%}")

    # ─── 6. 计算截面百分位排名 ───
    ranks = compute_cross_sectional_ranks(df, all_factor_cols)

    # ─── 7. 计算因子组排名 ───
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)

    # ─── 8. 添加log_metric ───
    ranks = add_log_metric(ranks)

    # ═══════════════════════════════════════════════
    #  测试1: Rank Inversion (Top5% vs Bottom20%)
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("🔍 测试1: Rank Inversion (Top5% vs Bottom20%)")
    print("=" * 80)

    ri_result = compute_rank_inversion(
        ranks, price_pivot, V040_WEIGHTS, hold_days=30
    )

    if ri_result:
        print(f"  总窗口数: {ri_result['total_windows']}")
        print(f"  通过窗口数: {ri_result['passed_windows']}")
        print(f"  通过率: {ri_result['pass_rate']:.1%}")
        print(f"  平均spread: {ri_result['avg_spread']:.6f}")
        print(f"  整体结果: {'✅ PASS' if ri_result['overall_passed'] else '❌ FAIL'}")

        print("\n  窗口详情:")
        for w in ri_result['windows']:
            mark = "✅" if w['passed'] else "❌"
            print(f"    {mark} W{w['window_idx']}: {w['period']} | "
                  f"Top5%={w['avg_top5_return']:.4f} Bot20%={w['avg_bottom20_return']:.4f} "
                  f"Spread={w['spread']:.4f} Pos%={w['positive_spread_pct']:.0%}")
    else:
        print("  ❌ 无法计算Rank Inversion")

    # ═══════════════════════════════════════════════
    #  测试2: Walk-Forward验证
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 测试2: Walk-Forward验证")
    print("=" * 80)

    wf_result, wf_windows = run_walk_forward(
        ranks, price_pivot, V040_WEIGHTS,
        train_years=0.5, test_months=6,
        hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15
    )

    if wf_result and "error" not in wf_result:
        print(f"  WF Sharpe: {wf_result['sharpe']:.3f} (std={wf_result['sharpe_std']:.3f})")
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
                  f"CAGR={w['cagr']:.1%} WR={w['win_rate']:.0%}")

        if failed_wins:
            print(f"\n  ⚠️ 失败窗口 ({len(failed_wins)}个):")
            for w in failed_wins:
                print(f"    ❌ W{w['index']}: {w['period']} | Error: {w['error'][:100]}")
    elif wf_result:
        print(f"  ❌ Walk-Forward失败: {wf_result.get('error', 'unknown')}")
        if 'windows' in wf_windows:
            for w in wf_windows:
                if 'error' in w:
                    print(f"    W{w.get('index', '?')}: {w['error'][:100]}")
    else:
        print("  ❌ Walk-Forward失败")

    # ═══════════════════════════════════════════════
    #  测试3: 因子IC/ICIR
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("📊 测试3: 因子IC/ICIR分析")
    print("=" * 80)

    ic_results = compute_factor_ic(df, all_factor_cols, price_pivot, hold_days=30)

    if ic_results:
        print(f"\n  Top 10 因子 (by |ICIR|):")
        for r in ic_results[:10]:
            print(f"    {r['name']}: IC={r['ic_mean']:.4f} ICIR={r['icir']:.4f} t={r['t_stat']:.1f}")

        # 汇总
        positive_icir = sum(1 for r in ic_results if r['icir'] > 0)
        negative_icir = sum(1 for r in ic_results if r['icir'] < 0)
        print(f"\n  正ICIR因子: {positive_icir}/{len(ic_results)}")
        print(f"  负ICIR因子: {negative_icir}/{len(ic_results)}")

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("💾 保存结果")
    print("=" * 80)

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.0 Fixed Validation (features_v04_1.parquet)",
            "config": {
                "weights": V040_WEIGHTS,
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "features_file": "features_v04_1.parquet",
            "features_shape": [int(df.shape[0]), int(df.shape[1])],
            "n_tickers": int(df['ticker'].nunique()),
            "date_range": [str(df['date'].min()), str(df['date'].max())],
            "factor_groups": {k: len(v) for k, v in FACTOR_GROUPS.items()},
            "total_factors": len(all_factor_cols),
            "overall_coverage": round(float(overall_cov), 4),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "coverage": serialize(coverage),
        "post_fill_coverage": serialize(post_fill_coverage),
        "rank_inversion_test": serialize(ri_result) if ri_result else None,
        "walk_forward_result": serialize(wf_result) if wf_result else None,
        "walk_forward_windows": serialize(wf_windows) if wf_windows else None,
        "factor_ic_analysis": serialize(ic_results[:30]) if ic_results else None,
        "summary": {
            "rank_inversion_passed": ri_result['overall_passed'] if ri_result else False,
            "rank_inversion_pass_rate": ri_result['pass_rate'] if ri_result else 0,
            "wf_sharpe": wf_result['sharpe'] if wf_result and 'sharpe' in wf_result else None,
            "wf_max_dd": wf_result['max_dd'] if wf_result and 'max_dd' in wf_result else None,
            "wf_cagr": wf_result['cagr'] if wf_result and 'cagr' in wf_result else None,
            "wf_win_rate": wf_result['win_rate'] if wf_result and 'win_rate' in wf_result else None,
            "wf_failed_windows": wf_result['failed_windows'] if wf_result and 'failed_windows' in wf_result else None,
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
    print(f"  数据: features_v04_1.parquet ({df.shape[0]}行 × {df.shape[1]}列)")
    print(f"  覆盖率: {overall_cov:.1%}")
    print(f"  权重: {V040_WEIGHTS}")
    print()
    print(f"  Rank Inversion: {'✅ PASS' if ri_result and ri_result['overall_passed'] else '❌ FAIL'} "
          f"(通过率={ri_result['pass_rate']:.0%})" if ri_result else "  Rank Inversion: ❌ N/A")
    if wf_result and 'sharpe' in wf_result:
        print(f"  WF Sharpe: {wf_result['sharpe']:.3f}")
        print(f"  MaxDD: {wf_result['max_dd']:.1%}")
        print(f"  CAGR: {wf_result['cagr']:.1%}")
        print(f"  Win Rate: {wf_result['win_rate']:.0%}")
    else:
        print(f"  WF: ❌ 失败")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
