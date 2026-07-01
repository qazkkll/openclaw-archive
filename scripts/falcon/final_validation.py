#!/usr/bin/env python3
"""
🦅 Falcon Final Validation: V0.3.1 vs V0.4.0 vs V0.4.1
========================================================
一次性彻底解决analyst因子和features_v02问题:
  - 使用features_v04_1.parquet (156列)
  - analyst因子缺失值用中位数填充
  - 3个模型版本对比
  - 真正的rank inversion测试 (Top5% vs Bottom20%)
  - Walk-Forward验证
  - 找到最佳配置

Walk-Forward参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出:
  - scripts/falcon/final_validation.py (本脚本)
  - data/falcon/final_validation_results.json
  - data/falcon/best_model_config.json
"""
import sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════
WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
RESULTS_PATH = DATA_DIR / "final_validation_results.json"
CONFIG_PATH = DATA_DIR / "best_model_config.json"

# ═══════════════════════════════════════════════════
#  因子组定义
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

# 越高越差的因子 → 截面排名时翻转
FLIP_FACTORS = {
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


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载特征和价格数据。"""
    print("📂 加载数据...")
    t0 = time.time()

    # 加载 features_v04_1.parquet
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    print(f"  ✅ Features: {df.shape[0]}行 × {df.shape[1]}列, {df['ticker'].nunique()}只")

    # 加载价格
    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")

    print(f"  ⏱️ 加载耗时: {time.time()-t0:.1f}秒")
    return df, price_pivot


# ═══════════════════════════════════════════════════
#  analyst因子中位数填充
# ═══════════════════════════════════════════════════

def fill_analyst_missing(df, factor_groups):
    """
    在截面排名之前，用中位数填充analyst因子的缺失值。
    策略: 对每个日期，用该日期所有股票的中位数填充NaN。
    使用groupby加速（比逐行循环快10x+）。
    """
    print("🔧 填充analyst因子缺失值...")
    analyst_cols = []
    for col in FACTOR_GROUPS.get('analyst', []):
        if col in df.columns:
            analyst_cols.append(col)

    if not analyst_cols:
        print("  ⚠️ 无analyst因子列")
        return df

    # 检查填充前的覆盖率
    for col in analyst_cols:
        pct_before = df[col].notna().mean()
        print(f"  {col}: 填充前覆盖率 {pct_before:.1%}")

    # 使用groupby逐日期填充中位数（比逐行循环快10x+）
    filled_count = 0
    for col in analyst_cols:
        # 计算每个日期的中位数
        medians = df.groupby('date')[col].transform('median')
        # 用中位数填充NaN，但如果整列全NaN则用0
        nan_mask = df[col].isna()
        n_filled = nan_mask.sum()
        df[col] = df[col].fillna(medians)
        # 对于整个日期全NaN的情况，fillna不会填充，用0兜底
        df[col] = df[col].fillna(0.0)
        filled_count += n_filled

    print(f"  ✅ 填充了 {filled_count} 个NaN值 (中位数)")

    # 检查填充后的覆盖率
    for col in analyst_cols:
        pct_after = df[col].notna().mean()
        print(f"  {col}: 填充后覆盖率 {pct_after:.1%}")

    return df


# ═══════════════════════════════════════════════════
#  截面百分位排名
# ═══════════════════════════════════════════════════

def compute_cross_sectional_ranks(df, factor_cols):
    """计算截面百分位排名 (0-1, 越高越好)。"""
    print("📊 计算截面百分位排名...")
    t0 = time.time()

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
            ranks_raw[valid] = rankdata(vals[valid], method='average') / valid.sum()

            # 翻转: 越高越差的因子
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


def add_combo_factors(ranks):
    """为每个日期添加组合因子列。"""
    combo_defs = {
        'log_metric': lambda d: np.log(d.get('fund_metric', 0) + 1),
        'log_growth': lambda d: np.log(d.get('fund_growth', 0) + 1),
        'growth_composite': lambda d: (
            d.get('fund_growth', 0) * 0.5 +
            d.get('analyst', 0) * 0.3 +
            d.get('income', 0) * 0.2
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

    print(f"  ✅ 组合因子: {list(combo_defs.keys())}")
    return ranks, list(combo_defs.keys())


# ═══════════════════════════════════════════════════
#  真正的 Rank Inversion 测试
# ═══════════════════════════════════════════════════

def compute_real_rank_inversion(ranks, prices, weights, 
                                 hold_days=30, train_months=6, test_months=6):
    """
    在每个Walk-Forward窗口中计算:
    - Top5%股票的平均收益
    - Bottom20%股票的平均收益
    - 检查Top5%收益 > Bottom20%收益
    """
    dates = sorted(ranks.keys())
    if not dates:
        return None

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start

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
    """运行Walk-Forward，返回结果字典和窗口详情。"""
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
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  Sharpe 退化分析
# ═══════════════════════════════════════════════════

def analyze_sharpe_degradation(windows):
    """分析Sharpe退化: 早期 vs 近期窗口。"""
    valid = [w for w in windows if "sharpe" in w]
    if len(valid) < 4:
        return {
            "passed": False,
            "reason": "Too few valid windows",
            "status": "ERROR",
        }

    early_windows = [w for w in valid if w.get("year", 2021) <= 2020]
    recent_windows = [w for w in valid if w.get("year", 2021) >= 2021]

    if not early_windows or not recent_windows:
        return {
            "passed": False,
            "reason": "Insufficient windows in early or recent period",
            "status": "ERROR",
        }

    early_avg = np.mean([w["sharpe"] for w in early_windows])
    recent_avg = np.mean([w["sharpe"] for w in recent_windows])

    if early_avg > 0:
        degradation_pct = (early_avg - recent_avg) / early_avg * 100
    else:
        degradation_pct = 0

    passed = True
    status = "PASS"
    reason = "OK"

    if degradation_pct > 75:
        passed = False
        status = "FAIL"
        reason = f"Severe degradation: {degradation_pct:.1f}% (>75% FAIL)"
    elif degradation_pct > 50:
        status = "WARN"
        reason = f"Moderate degradation: {degradation_pct:.1f}% (>50% WARN)"

    return {
        "passed": passed,
        "status": status,
        "reason": reason,
        "early_avg_sharpe": round(float(early_avg), 3),
        "recent_avg_sharpe": round(float(recent_avg), 3),
        "degradation_pct": round(float(degradation_pct), 1),
        "early_window_count": len(early_windows),
        "recent_window_count": len(recent_windows),
    }


# ═══════════════════════════════════════════════════
#  JSON序列化
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
#  模型配置定义
# ═══════════════════════════════════════════════════

MODEL_CONFIGS = {
    "V0.3.1": {
        "description": "fund_ratio主导 + analyst + fund_metric",
        "weights": {
            "fund_ratio": 0.70,
            "analyst": 0.20,
            "fund_metric": 0.10,
        },
    },
    "V0.4.0": {
        "description": "fund_ratio + fund_metric + log(fm+1)",
        "weights": {
            "fund_ratio": 0.70,
            "fund_metric": 0.15,
            "log_metric": 0.15,
        },
    },
    "V0.4.1": {
        "description": "fund_ratio + growth_composite",
        "weights": {
            "fund_ratio": 0.70,
            "growth_composite": 0.30,
        },
    },
}


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon Final Validation: V0.3.1 vs V0.4.0 vs V0.4.1")
    print("=" * 80)

    # ─── 1. 加载数据 ───
    df, price_pivot = load_data()

    # ─── 2. analyst因子中位数填充 ───
    df = fill_analyst_missing(df, FACTOR_GROUPS)

    # ─── 3. 确定可用PIT因子 ───
    all_pit_cols = []
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        all_pit_cols.extend(available)
        print(f"  {group}: {len(available)}/{len(cols)} factors")
    print(f"  Total PIT factors: {len(all_pit_cols)}")

    # ─── 4. 计算截面百分位排名 ───
    ranks = compute_cross_sectional_ranks(df, all_pit_cols)

    # ─── 5. 计算因子组排名 ───
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)

    # ─── 6. 添加组合因子 ───
    ranks, combo_names = add_combo_factors(ranks)

    # ═══════════════════════════════════════════════
    #  逐个模型测试
    # ═══════════════════════════════════════════════
    all_results = {}

    for model_name, config in MODEL_CONFIGS.items():
        weights = config["weights"]
        print(f"\n{'=' * 80}")
        print(f"🔬 测试模型: {model_name} — {config['description']}")
        print(f"  权重: {weights}")
        print(f"{'=' * 80}")

        # 检查权重对应的因子组是否在ranks中
        # 使用中位日期检查（早期日期可能覆盖率不足）
        available_groups = set()
        all_rank_dates = sorted(ranks.keys())
        sample_date = all_rank_dates[len(all_rank_dates) // 2]
        for group_name in weights:
            if group_name in ranks[sample_date].columns:
                available_groups.add(group_name)
            else:
                print(f"  ⚠️ 因子组 '{group_name}' 不在ranks中，跳过")
        if not available_groups:
            print(f"  ❌ 无可用因子组，跳过 {model_name}")
            continue

        # 过滤权重: 只保留可用的因子组
        filtered_weights = {k: v for k, v in weights.items() if k in available_groups}
        # 归一化权重
        total_w = sum(filtered_weights.values())
        if total_w > 0:
            filtered_weights = {k: v / total_w for k, v in filtered_weights.items()}
        print(f"  有效权重: {filtered_weights}")

        # ─── Rank Inversion 测试 ───
        print(f"\n  🔍 Rank Inversion 测试 (Top5% vs Bottom20%)...")
        ri_result = compute_real_rank_inversion(
            ranks, price_pivot, filtered_weights,
            hold_days=30, train_months=6, test_months=6
        )

        if ri_result:
            print(f"  总窗口数: {ri_result['total_windows']}")
            print(f"  通过窗口数: {ri_result['passed_windows']}")
            print(f"  通过率: {ri_result['pass_rate']:.1%}")
            print(f"  平均spread: {ri_result['avg_spread']:.4f}")
            print(f"  整体结果: {'✅ PASS' if ri_result['overall_passed'] else '❌ FAIL'}")
        else:
            print(f"  ❌ 无法计算Rank Inversion")

        # ─── Walk-Forward 回测 ───
        print(f"\n  📊 Walk-Forward 回测...")
        wf_result, wf_windows = run_walk_forward(
            ranks, price_pivot, filtered_weights,
            train_years=0.5, test_months=6,
            hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15
        )

        if wf_result and "error" not in wf_result:
            print(f"  Sharpe: {wf_result['sharpe']:.3f} (std={wf_result['sharpe_std']:.3f})")
            print(f"  MaxDD: {wf_result['max_dd']:.1%}")
            print(f"  CAGR: {wf_result['cagr']:.1%}")
            print(f"  Win Rate: {wf_result['win_rate']:.0%}")
            print(f"  有效窗口: {wf_result['n_windows']}/{wf_result['total_windows']}")

            # 打印每个窗口
            valid_wins = [w for w in wf_windows if "sharpe" in w]
            for w in valid_wins:
                print(f"    W{w['index']}: {w['period']} | "
                      f"Sharpe={w['sharpe']:.3f} MaxDD={w['max_dd']:.1%} "
                      f"WR={w['win_rate']:.0%}")
        else:
            print(f"  ❌ Walk-Forward失败")
            if wf_result:
                print(f"  Error: {wf_result.get('error', 'unknown')}")

        # ─── Sharpe退化分析 ───
        degradation = analyze_sharpe_degradation(wf_windows) if wf_windows else {"status": "ERROR"}

        # ─── 汇总 ───
        model_summary = {
            "config": config,
            "rank_inversion": serialize(ri_result) if ri_result else None,
            "walk_forward": serialize(wf_result) if wf_result else None,
            "wf_windows": serialize(wf_windows) if wf_windows else None,
            "degradation": serialize(degradation),
        }
        all_results[model_name] = model_summary

        # 打印退化分析
        if degradation.get("status") != "ERROR":
            print(f"\n  📉 Sharpe退化: {degradation['status']}")
            print(f"    早期({degradation.get('early_window_count',0)}窗口): {degradation.get('early_avg_sharpe', 'N/A')}")
            print(f"    近期({degradation.get('recent_window_count',0)}窗口): {degradation.get('recent_avg_sharpe', 'N/A')}")
            print(f"    退化: {degradation.get('degradation_pct', 'N/A')}%")

    # ═══════════════════════════════════════════════
    #  对比与最佳配置
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("📊 模型对比总结")
    print(f"{'=' * 80}")

    comparison = {}
    for model_name, result in all_results.items():
        wf = result.get("walk_forward", {})
        ri = result.get("rank_inversion", {})
        deg = result.get("degradation", {})

        if wf and "error" not in wf:
            comparison[model_name] = {
                "sharpe": wf.get("sharpe"),
                "max_dd": wf.get("max_dd"),
                "cagr": wf.get("cagr"),
                "win_rate": wf.get("win_rate"),
                "n_windows": wf.get("n_windows"),
                "ri_pass_rate": ri.get("pass_rate", 0) if ri else 0,
                "ri_passed": ri.get("overall_passed", False) if ri else False,
                "degradation_status": deg.get("status", "ERROR"),
                "degradation_pct": deg.get("degradation_pct"),
            }
            print(f"\n  {model_name}:")
            print(f"    WF Sharpe:  {wf.get('sharpe', 'N/A'):.3f}")
            print(f"    MaxDD:      {wf.get('max_dd', 0):.1%}")
            print(f"    CAGR:       {wf.get('cagr', 0):.1%}")
            print(f"    Win Rate:   {wf.get('win_rate', 0):.0%}")
            print(f"    RI通过率:   {ri.get('pass_rate', 0):.1%}" if ri else "    RI: N/A")
            print(f"    RI通过:     {'✅' if ri and ri.get('overall_passed') else '❌'}" if ri else "    RI: N/A")
            print(f"    退化状态:   {deg.get('status', 'N/A')}")
        else:
            print(f"\n  {model_name}: ❌ 无效")

    # 找到最佳配置
    best_model = None
    best_score = -999

    for model_name, comp in comparison.items():
        # 评分公式: Sharpe * 0.4 + (1 + MaxDD) * 0.2 + RI通过率 * 0.2 + (1 if RI通过 else 0) * 0.2
        sharpe_score = comp.get("sharpe", 0) or 0
        dd_score = 1 + (comp.get("max_dd", 0) or 0)  # MaxDD是负数
        ri_score = comp.get("ri_pass_rate", 0)
        ri_bonus = 1.0 if comp.get("ri_passed") else 0.0

        # 如果RI不通过，严重惩罚
        if not comp.get("ri_passed", False):
            ri_score *= 0.3
            ri_bonus = 0.0

        # 如果退化严重，惩罚
        deg_penalty = 1.0
        if comp.get("degradation_status") == "FAIL":
            deg_penalty = 0.5
        elif comp.get("degradation_status") == "WARN":
            deg_penalty = 0.8

        score = (sharpe_score * 0.4 + dd_score * 0.2 + ri_score * 0.2 + ri_bonus * 0.2) * deg_penalty

        print(f"\n  {model_name} 评分: {score:.3f}")
        print(f"    Sharpe={sharpe_score:.3f}×0.4 + DD={dd_score:.3f}×0.2 + RI={ri_score:.3f}×0.2 + RI_bonus={ri_bonus:.1f}×0.2 × penalty={deg_penalty:.1f}")

        if score > best_score:
            best_score = score
            best_model = model_name

    print(f"\n{'=' * 80}")
    print(f"🏆 最佳模型: {best_model}")
    print(f"{'=' * 80}")

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    print(f"\n💾 保存结果...")

    # 确保目录存在
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 保存完整结果
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "Final Validation: V0.3.1 vs V0.4.0 vs V0.4.1",
            "features_file": "features_v04_1.parquet",
            "config": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "analyst_fill_method": "median_per_date",
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "models": all_results,
        "comparison": comparison,
        "best_model": best_model,
        "best_score": round(best_score, 3),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 完整结果: {RESULTS_PATH}")

    # 保存最佳配置
    if best_model and best_model in MODEL_CONFIGS:
        best_config = {
            "model": best_model,
            "config": MODEL_CONFIGS[best_model],
            "comparison": comparison.get(best_model, {}),
            "timestamp": datetime.now().isoformat(),
            "features_file": "features_v04_1.parquet",
            "backtest_params": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)
        print(f"  ✅ 最佳配置: {CONFIG_PATH}")

    # ═══════════════════════════════════════════════
    #  最终摘要
    # ═══════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("📋 最终摘要")
    print(f"{'=' * 80}")
    print(f"  数据: features_v04_1.parquet (156列, analyst中位数填充)")
    print(f"  模型数: {len(MODEL_CONFIGS)}")
    for model_name, comp in comparison.items():
        marker = "🏆" if model_name == best_model else "  "
        ri_mark = "✅" if comp.get("ri_passed") else "❌"
        print(f"  {marker} {model_name}: Sharpe={comp.get('sharpe', 'N/A')}, "
              f"RI={ri_mark}({comp.get('ri_pass_rate', 0):.0%}), "
              f"退化={comp.get('degradation_status', 'N/A')}")
    print(f"  ⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
