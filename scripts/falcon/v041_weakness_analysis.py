#!/usr/bin/env python3
"""
🦅 Falcon V0.4.1 Weakness Analysis
====================================
分析V0.4.1当前配置的弱点，找到最优配置。

分析维度:
1. 因子有效性分析: IC/ICIR + 因子相关性
2. 权重敏感性分析: fund_ratio + growth_composite
3. growth_composite子权重分析
4. 训练窗口分析: 3mo / 6mo / 12mo

红线: 必须用backtest_engine.py回测，不能自己实现回测逻辑。
"""
import sys, json, time, warnings, os
from pathlib import Path
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np
from scipy.stats import rankdata

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "v041_weakness_analysis.json"

# ═══════════════════════════════════════════════════
#  因子组定义 (与v041_optimization.py一致)
# ═══════════════════════════════════════════════════
EXCLUDE_COLS = {
    'ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap',
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'ma_cross_5_20', 'ma_cross_20_60',
    'price_position', 'ret1', 'ret5', 'ret10', 'ret20', 'ret30', 'ret60', 'ret90',
    'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
    'vol20', 'vol5', 'vol_ratio', 'vol_change', 'vol_regime',
    'rsi14', 'rsi_change', 'rsi_zone',
    'macd', 'macd_signal', 'macd_hist', 'macd_roc',
    'bb_std', 'bb_width', 'bb_pos',
    'ret_quality', 'range_ratio', 'avg_body', 'vwap_drift', 'dd_60', 'ud_vol_ratio', 'beta',
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin', 'ebitdaMargin',
    'assetTurnover', 'inventoryTurnover', 'receivablesTurnover',
    'debtToEquityRatio', 'currentRatio', 'quickRatio', 'financialLeverageRatio',
    'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
    'dividendYieldPercentage', 'dividendPayoutRatio',
    'eps_revision', 'revenue_revision', 'num_analysts_eps', 'num_analysts_rev',
    'eps_dispersion', 'fmp_covered', 'analyst_covered',
    'grossProfitMargin_qoq', 'netProfitMargin_qoq', 'operatingProfitMargin_qoq', 'ebitdaMargin_qoq',
}

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

# 越高越差的因子 → 翻转
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

# V0.4.1当前配置 (问题配置)
V041_CURRENT = {
    "fund_ratio": 0.70,
    "growth_composite": 0.30,
}
V041_GROWTH_COMPOSITE_SUB = {
    "fund_growth": 0.50,
    "analyst": 0.30,
    "income": 0.20,
}

# Walk-Forward 参数
WF_PARAMS = {
    "train_years": 0.5,
    "test_months": 6,
    "hold_days": 30,
    "top_n": 10,
    "cost": 0.001,
    "stop_loss": -0.15,
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


def add_combo_factors(ranks, growth_sub_weights=None):
    """为每个日期添加组合因子列。growth_composite子权重可调。"""
    if growth_sub_weights is None:
        growth_sub_weights = V041_GROWTH_COMPOSITE_SUB

    gw_fund = growth_sub_weights.get("fund_growth", 0.5)
    gw_analyst = growth_sub_weights.get("analyst", 0.3)
    gw_income = growth_sub_weights.get("income", 0.2)

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
        'growth_composite': lambda d: (
            d.get('fund_growth', 0) * gw_fund +
            d.get('analyst', 0) * gw_analyst +
            d.get('income', 0) * gw_income
        ),
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
    print(f"  ✅ 组合因子: {list(combo_defs.keys())}")
    return ranks, list(combo_defs.keys())


# ═══════════════════════════════════════════════════
#  IC/ICIR 分析 (第1步)
# ═══════════════════════════════════════════════════
def compute_ic_icir(df, factor_cols, target_col='fwd_ret_30d'):
    """计算每个因子的IC和ICIR (使用fwd_ret_30d作为target)。"""
    print("\n📊 计算因子IC/ICIR...")
    t0 = time.time()

    # 检查target是否存在
    if target_col not in df.columns:
        # 尝试构造30天前瞻收益
        print(f"  ⚠️ {target_col}不存在, 尝试构造...")
        # 构造: 同ticker下, 30个交易日后的close / 当前close - 1
        prices_df = pd.read_parquet(PRICES_PATH)
        prices_df['date'] = prices_df['date'].astype(str)
        price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
        dates = sorted(price_pivot.index.tolist())

        fwd_ret = {}
        for i, date in enumerate(dates):
            # 找30个交易日后的日期
            future_idx = min(i + 30, len(dates) - 1)
            if future_idx == i:
                continue
            future_date = dates[future_idx]
            for ticker in price_pivot.columns:
                p_now = price_pivot.loc[date, ticker] if ticker in price_pivot.columns else np.nan
                p_future = price_pivot.loc[future_date, ticker] if ticker in price_pivot.columns else np.nan
                if pd.notna(p_now) and pd.notna(p_future) and p_now > 0:
                    fwd_ret[(date, ticker)] = p_future / p_now - 1

        df[target_col] = df.apply(
            lambda r: fwd_ret.get((r['date'], r['ticker']), np.nan), axis=1
        )
        print(f"  ✅ 已构造{target_col}, 有效率: {df[target_col].notna().mean():.1%}")

    merged = df.dropna(subset=[target_col])
    n_dates = merged['date'].nunique()

    # 使用向量化方法计算IC
    merged_sorted = merged.sort_values('date').reset_index(drop=True)
    target_arr = merged_sorted[target_col].values.astype(np.float64)
    factor_arrs = {f: merged_sorted[f].values.astype(np.float64) for f in factor_cols if f in merged_sorted.columns}
    date_groups = merged_sorted.groupby('date').indices

    ic_sums = np.zeros(len(factor_arrs))
    ic_sq_sums = np.zeros(len(factor_arrs))
    ic_counts = np.zeros(len(factor_arrs), dtype=int)
    factor_list = list(factor_arrs.keys())

    MIN_STOCKS = 20

    for di, (date, indices) in enumerate(date_groups.items()):
        t_vals = target_arr[indices]
        t_valid_mask = ~np.isnan(t_vals)
        n_valid = t_valid_mask.sum()
        if n_valid < MIN_STOCKS:
            continue

        for fj, factor in enumerate(factor_list):
            f_vals = factor_arrs[factor][indices]
            both_valid = t_valid_mask & ~np.isnan(f_vals)
            n_both = both_valid.sum()
            if n_both < MIN_STOCKS:
                continue
            f_valid = f_vals[both_valid]
            t_r = rankdata(t_vals[both_valid])
            f_ranked = rankdata(f_valid)
            n = len(f_ranked)
            f_std = f_ranked.std(ddof=0)
            t_std = t_r.std(ddof=0)
            if f_std == 0 or t_std == 0:
                continue
            corr = np.sum((f_ranked - f_ranked.mean()) * (t_r - t_r.mean())) / (n * f_std * t_std)
            ic_sums[fj] += corr
            ic_sq_sums[fj] += corr * corr
            ic_counts[fj] += 1

    # 统计
    results = []
    for fj, factor in enumerate(factor_list):
        n = ic_counts[fj]
        if n == 0:
            continue
        ic_mean = ic_sums[fj] / n
        ic_sq_mean = ic_sq_sums[fj] / n
        ic_var = ic_sq_mean - ic_mean * ic_mean
        ic_std = np.sqrt(max(ic_var, 0))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 0 else 0.0
        coverage = n / n_dates

        results.append({
            'name': factor,
            'ic_mean': round(float(ic_mean), 6),
            'ic_std': round(float(ic_std), 6),
            'icir': round(float(icir), 6),
            't_stat': round(float(t_stat), 4),
            'coverage': round(float(coverage), 4),
            'n_dates': int(n),
        })

    results.sort(key=lambda x: abs(x['icir']), reverse=True)
    elapsed = time.time() - t0
    print(f"  ✅ IC/ICIR计算完成 ({elapsed:.0f}秒), {len(results)}个因子")
    return results


def compute_factor_correlations(ranks, factor_groups):
    """计算因子组间的相关性矩阵。"""
    print("\n📊 计算因子组相关性...")
    group_names = list(factor_groups.keys())
    # 取最近100个日期的平均排名相关性
    dates = sorted(ranks.keys())[-100:]
    corr_data = {}
    for g in group_names:
        vals = []
        for d in dates:
            if d in ranks and g in ranks[d].columns:
                vals.append(ranks[d][g].mean())
        corr_data[g] = vals

    if len(corr_data[group_names[0]]) < 10:
        print("  ⚠️ 日期不足, 跳过相关性计算")
        return {}

    corr_df = pd.DataFrame(corr_data)
    corr_matrix = corr_df.corr()
    print(f"  ✅ 相关性矩阵已计算")

    # 找高相关因子对
    high_corr_pairs = []
    for i, g1 in enumerate(group_names):
        for j, g2 in enumerate(group_names):
            if i < j and g1 in corr_matrix.columns and g2 in corr_matrix.columns:
                c = corr_matrix.loc[g1, g2]
                if abs(c) > 0.5:
                    high_corr_pairs.append({"pair": f"{g1}+{g2}", "corr": round(float(c), 3)})

    return {
        "matrix": {g: {g2: round(float(corr_matrix.loc[g, g2]), 3) for g2 in group_names if g2 in corr_matrix.columns} for g in group_names if g in corr_matrix.columns},
        "high_corr_pairs": sorted(high_corr_pairs, key=lambda x: abs(x['corr']), reverse=True),
    }


# ═══════════════════════════════════════════════════
#  Walk-Forward 辅助
# ═══════════════════════════════════════════════════
def check_rank_inversion(windows):
    """检查排名反转。"""
    valid = [w for w in windows if "sharpe" in w]
    if len(valid) < 2:
        return {"passed": False, "reason": "Too few windows"}
    recent = valid[-3:] if len(valid) >= 3 else valid
    early = valid[:3] if len(valid) >= 3 else valid
    recent_avg = np.mean([w["sharpe"] for w in recent])
    early_avg = np.mean([w["sharpe"] for w in early])
    neg_recent = sum(1 for w in recent if w["sharpe"] < 0)
    passed = True
    reason = "OK"
    if neg_recent >= 2:
        passed = False
        reason = f"Recent {neg_recent}/3 windows negative"
    elif early_avg > 0 and recent_avg < early_avg * 0.3:
        passed = False
        reason = f"Severe degradation: early={early_avg:.2f} → recent={recent_avg:.2f}"
    return {
        "passed": passed,
        "recent_avg_sharpe": round(float(recent_avg), 3),
        "early_avg_sharpe": round(float(early_avg), 3),
        "negative_recent_windows": neg_recent,
        "reason": reason,
    }


def run_wf(ranks, prices, weights, train_years=0.5, test_months=6,
           hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """运行Walk-Forward, 返回(result_dict, window_details)。"""
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    dates = sorted(ranks.keys())
    if not dates:
        return None, []

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    windows = []
    idx = 0
    train_months = int(train_years * 12)

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
                "index": idx, "period": f"{tss} → {tes}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades, "n_days": len(result.daily_equity),
                "baseline_sharpe": baseline.sharpe if baseline else None,
            })
        except (DataQualityError, Exception) as e:
            windows.append({"index": idx, "period": f"{tss} → {tes}", "error": str(e)[:200]})
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
    ri = check_rank_inversion(windows)

    result = {
        "sharpe": round(float(np.mean(sharpes)), 3),
        "max_dd": round(float(np.min(dds)), 4),
        "cagr": round(float(np.mean(cagrs)), 4),
        "win_rate": round(float(np.mean(wrs)), 3),
        "n_trades": sum(w["n_trades"] for w in valid),
        "n_windows": len(valid),
        "rank_inversion": ri,
        "warnings": [],
        "status": "PASS",
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  候选追踪器
# ═══════════════════════════════════════════════════
class CandidateTracker:
    def __init__(self):
        self.candidates = []

    def add(self, name, weights, res):
        if res and "sharpe" in res:
            self.candidates.append({
                "name": name,
                "sharpe": res["sharpe"],
                "max_dd": res["max_dd"],
                "cagr": res["cagr"],
                "win_rate": res["win_rate"],
                "n_windows": res["n_windows"],
                "rank_inversion_passed": res["rank_inversion"]["passed"],
                "weights": dict(weights),
            })

    def best(self):
        ri_passed = [c for c in self.candidates if c["rank_inversion_passed"]]
        pool = ri_passed if ri_passed else self.candidates
        if not pool:
            return None
        return max(pool, key=lambda x: x["sharpe"])


# ═══════════════════════════════════════════════════
#  分析1: 因子有效性
# ═══════════════════════════════════════════════════
def analysis1_factor_effectiveness(df, factor_cols, factor_groups):
    """因子有效性分析: IC/ICIR + 相关性。"""
    print(f"\n{'='*60}")
    print("📊 分析1: 因子有效性 (IC/ICIR)")
    print(f"{'='*60}")

    # 只计算PIT因子的IC/ICIR (排除技术因子加速)
    pit_cols = [c for c in factor_cols if c.startswith(('r_', 'm_', 'g_', 'a_', 'b_', 'c_', 'i_'))]
    print(f"  ℹ️  仅计算{len(pit_cols)}个PIT因子的IC/ICIR (排除技术因子)")
    ic_results = compute_ic_icir(df, pit_cols)

    # 找到最有效和最无效的因子
    top_10 = ic_results[:10] if len(ic_results) >= 10 else ic_results
    bottom_10 = ic_results[-10:] if len(ic_results) >= 10 else ic_results

    print(f"\n  🏆 Top 10 因子 (|ICIR|最高):")
    for i, r in enumerate(top_10, 1):
        print(f"    {i:2d}. {r['name']:40s} ICIR={r['icir']:+.4f}  IC={r['ic_mean']:+.4f}  t={r['t_stat']:+.2f}")

    print(f"\n  ❌ Bottom 10 因子 (|ICIR|最低):")
    for i, r in enumerate(bottom_10, 1):
        print(f"    {i:2d}. {r['name']:40s} ICIR={r['icir']:+.4f}  IC={r['ic_mean']:+.4f}  t={r['t_stat']:+.2f}")

    # 按因子组汇总
    group_icir = {}
    for group, cols in factor_groups.items():
        group_factors = [r for r in ic_results if r['name'] in cols]
        if group_factors:
            avg_icir = np.mean([abs(r['icir']) for r in group_factors])
            avg_ic = np.mean([r['ic_mean'] for r in group_factors])
            strong_count = sum(1 for r in group_factors if abs(r['icir']) > 0.05)
            group_icir[group] = {
                "avg_icir": round(float(avg_icir), 4),
                "avg_ic": round(float(avg_ic), 4),
                "strong_count": strong_count,
                "total_count": len(group_factors),
                "top_factor": group_factors[0]['name'] if group_factors else "",
            }
    print(f"\n  📊 因子组IC/ICIR汇总:")
    for g, v in sorted(group_icir.items(), key=lambda x: x[1]['avg_icir'], reverse=True):
        print(f"    {g:20s} avg_ICIR={v['avg_icir']:+.4f}  strong={v['strong_count']}/{v['total_count']}  top={v['top_factor']}")

    # 因子相关性
    corr_info = compute_factor_correlations(
        compute_group_ranks(
            compute_cross_sectional_ranks(df, factor_cols),
            factor_groups
        ),
        factor_groups
    )

    return {
        "ic_results": ic_results[:50],  # Top 50
        "group_icir": group_icir,
        "correlations": corr_info,
        "top_10": top_10,
        "bottom_10": bottom_10,
    }


# ═══════════════════════════════════════════════════
#  分析2: 权重敏感性 (fund_ratio + growth_composite)
# ═══════════════════════════════════════════════════
def analysis2_weight_sensitivity(ranks, prices, tracker):
    """测试fund_ratio从0.5到0.9, growth_composite从0.1到0.5。"""
    print(f"\n{'='*60}")
    print("📊 分析2: 权重敏感性 (fund_ratio + growth_composite)")
    print(f"{'='*60}")

    results = []
    for fr in np.arange(0.50, 0.91, 0.10):
        for gc in np.arange(0.10, 0.51, 0.10):
            # 确保权重和为1
            remainder = round(1.0 - fr - gc, 2)
            if remainder < 0.0 or remainder > 0.30:
                continue
            w = {"fund_ratio": round(fr, 2), "growth_composite": round(gc, 2)}
            if remainder > 0.01:
                w["fund_metric"] = remainder
            res, _ = run_wf(ranks, prices, w, **WF_PARAMS)
            if res:
                label = f"fr{fr:.2f}_gc{gc:.2f}"
                res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
                tracker.add(label, w, res)
                results.append({
                    "fund_ratio": round(fr, 2),
                    "growth_composite": round(gc, 2),
                    "remainder": remainder,
                    "sharpe": res["sharpe"],
                    "max_dd": res["max_dd"],
                    "cagr": res["cagr"],
                    "ri_passed": res["rank_inversion"]["passed"],
                })
                if res["sharpe"] > 1.5:
                    print(f"    fr={fr:.2f} gc={gc:.2f} → Sharpe={res['sharpe']:.3f} RI={'✅' if res['rank_inversion']['passed'] else '❌'}")

    # 找到最优
    valid = [r for r in results if r["ri_passed"]]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\n  🏆 最优权重: fund_ratio={best['fund_ratio']}, growth_composite={best['growth_composite']}, Sharpe={best['sharpe']:.3f}")
    else:
        best = max(results, key=lambda x: x["sharpe"]) if results else None
        print(f"\n  ⚠️ 无RI通过的配置, 取最高Sharpe")

    return results


# ═══════════════════════════════════════════════════
#  分析3: growth_composite子权重
# ═══════════════════════════════════════════════════
def analysis3_growth_sub_weights(ranks, prices, tracker):
    """测试fund_growth从0.3到0.7, analyst从0.1到0.5, income从0.1到0.3。"""
    print(f"\n{'='*60}")
    print("📊 分析3: growth_composite子权重")
    print(f"{'='*60}")

    results = []
    # 测试关键组合 (不重建ranks, 只改组合因子中的子权重)
    sub_combos = [
        (0.50, 0.30, 0.20),  # V0.4.1当前
        (0.60, 0.25, 0.15),
        (0.70, 0.20, 0.10),
        (0.40, 0.40, 0.20),
        (0.30, 0.50, 0.20),
        (0.60, 0.30, 0.10),
        (0.50, 0.40, 0.10),
        (0.70, 0.15, 0.15),
        (0.40, 0.30, 0.30),
        (0.55, 0.25, 0.20),
    ]
    for fg, an, inc in sub_combos:
        sub_w = {"fund_growth": round(fg, 2), "analyst": round(an, 2), "income": round(inc, 2)}
        w = {"fund_ratio": 0.70, "growth_composite": 0.30}
        # 重建ranks with新的子权重
        ranks_copy = {d: df.copy() for d, df in ranks.items()}
        ranks_copy = add_combo_factors(ranks_copy, sub_w)[0]

        res, _ = run_wf(ranks_copy, prices, w, **WF_PARAMS)
        if res:
            label = f"fg{fg:.2f}_an{an:.2f}_inc{inc:.2f}"
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(label, w, res)
            results.append({
                "fund_growth": round(fg, 2),
                "analyst": round(an, 2),
                "income": round(inc, 2),
                "sharpe": res["sharpe"],
                "max_dd": res["max_dd"],
                "ri_passed": res["rank_inversion"]["passed"],
            })
            if res["sharpe"] > 1.5:
                print(f"    fg={fg:.2f} an={an:.2f} inc={inc:.2f} → Sharpe={res['sharpe']:.3f} RI={'✅' if res['rank_inversion']['passed'] else '❌'}")

    valid = [r for r in results if r["ri_passed"]]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\n  🏆 最优子权重: fg={best['fund_growth']}, an={best['analyst']}, inc={best['income']}, Sharpe={best['sharpe']:.3f}")
    else:
        best = max(results, key=lambda x: x["sharpe"]) if results else None
        print(f"\n  ⚠️ 无RI通过的配置")

    return results


# ═══════════════════════════════════════════════════
#  分析4: 训练窗口
# ═══════════════════════════════════════════════════
def analysis4_training_windows(ranks, prices, tracker, base_weights):
    """测试3个月、6个月、12个月训练窗口。"""
    print(f"\n{'='*60}")
    print("📊 分析4: 训练窗口")
    print(f"{'='*60}")

    results = []
    for months in [3, 6, 12]:
        train_years = months / 12.0
        res, wins = run_wf(ranks, prices, base_weights, train_years=train_years, **{k: v for k, v in WF_PARAMS.items() if k != 'train_years'})
        if res:
            label = f"window_{months}mo"
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(label, base_weights, res)
            results.append({
                "months": months,
                "sharpe": res["sharpe"],
                "max_dd": res["max_dd"],
                "cagr": res["cagr"],
                "win_rate": res["win_rate"],
                "n_windows": res["n_windows"],
                "ri_passed": res["rank_inversion"]["passed"],
                "window_details": [w for w in (wins or []) if "sharpe" in w],
            })
            print(f"  {months:2d}个月: Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
                  f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}  "
                  f"RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}  "
                  f"Windows={res['n_windows']}")

    valid = [r for r in results if r["ri_passed"]]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\n  🏆 最优窗口: {best['months']}个月 (Sharpe={best['sharpe']:.3f})")
    else:
        best = max(results, key=lambda x: x["sharpe"]) if results else None

    return results


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4.1 Weakness Analysis")
    print("=" * 80)

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
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)
    ranks, combo_names = add_combo_factors(ranks)

    # ═══════════════════════════════════════════════
    #  分析1: 因子有效性
    # ═══════════════════════════════════════════════
    analysis1_results = analysis1_factor_effectiveness(df, all_pit_cols, FACTOR_GROUPS)

    # ═══════════════════════════════════════════════
    #  分析2: 权重敏感性
    # ═══════════════════════════════════════════════
    tracker = CandidateTracker()
    analysis2_results = analysis2_weight_sensitivity(ranks, price_pivot, tracker)

    # ═══════════════════════════════════════════════
    #  分析3: growth_composite子权重
    # ═══════════════════════════════════════════════
    analysis3_results = analysis3_growth_sub_weights(ranks, price_pivot, tracker)

    # ═══════════════════════════════════════════════
    #  分析4: 训练窗口
    # ═══════════════════════════════════════════════
    best_weights = {"fund_ratio": 0.70, "growth_composite": 0.30}  # 默认用V0.4.1配置
    if tracker.best():
        best_weights = tracker.best()["weights"]
        print(f"\n  📌 当前最佳权重: {best_weights}")
    analysis4_results = analysis4_training_windows(ranks, price_pivot, tracker, best_weights)

    # ═══════════════════════════════════════════════
    #  最终选择
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("🏆 最终选择")
    print(f"{'='*60}")

    final_best = tracker.best()
    if final_best:
        print(f"  Best: {final_best['name']}")
        print(f"  Sharpe: {final_best['sharpe']:.3f}")
        print(f"  MaxDD: {final_best['max_dd']:.1%}")
        print(f"  CAGR: {final_best['cagr']:.1%}")
        print(f"  Win Rate: {final_best['win_rate']:.0%}")
        print(f"  Rank Inversion: {'PASS' if final_best['rank_inversion_passed'] else 'FAIL'}")
        print(f"  Weights: {final_best['weights']}")
    else:
        print("  ❌ No valid candidates")

    # ═══════════════════════════════════════════════
    #  最终验证: 最优配置 vs V0.4.1当前配置
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("📊 最终对比: 最优 vs V0.4.1当前")
    print(f"{'='*60}")

    # V0.4.1当前配置
    res_current, wins_current = run_wf(ranks, price_pivot, V041_CURRENT, **WF_PARAMS)
    if res_current:
        print(f"\n  V0.4.1当前: fund_ratio=0.70 + growth_composite=0.30")
        print(f"    Sharpe={res_current['sharpe']:.3f}  MaxDD={res_current['max_dd']:.1%}  "
              f"CAGR={res_current['cagr']:.1%}  WR={res_current['win_rate']:.0%}")
        print(f"    RI={'PASS' if res_current['rank_inversion']['passed'] else 'FAIL'}  "
              f"RI_reason={res_current['rank_inversion']['reason']}")
        # 逐窗口
        valid_wins = [w for w in (wins_current or []) if "sharpe" in w]
        for w in valid_wins:
            mark = "✅" if w["sharpe"] > 0 else "❌"
            baseline_mark = ""
            if w.get("baseline_sharpe") is not None:
                baseline_mark = f"  baseline={w['baseline_sharpe']:.3f}"
            print(f"    {mark} W{w['index']}: {w['period']}  Sharpe={w['sharpe']:.3f}  "
                  f"MaxDD={w['max_dd']:.1%}  WR={w['win_rate']:.0%}{baseline_mark}")

    # 最优配置
    if final_best and final_best["name"] != "none":
        res_optimal, wins_optimal = run_wf(ranks, price_pivot, final_best["weights"], **WF_PARAMS)
        if res_optimal:
            print(f"\n  最优配置: {final_best['weights']}")
            print(f"    Sharpe={res_optimal['sharpe']:.3f}  MaxDD={res_optimal['max_dd']:.1%}  "
                  f"CAGR={res_optimal['cagr']:.1%}  WR={res_optimal['win_rate']:.0%}")
            print(f"    RI={'PASS' if res_optimal['rank_inversion']['passed'] else 'FAIL'}")
            valid_wins = [w for w in (wins_optimal or []) if "sharpe" in w]
            for w in valid_wins:
                mark = "✅" if w["sharpe"] > 0 else "❌"
                print(f"      {mark} W{w['index']}: {w['period']}  Sharpe={w['sharpe']:.3f}  "
                      f"MaxDD={w['max_dd']:.1%}  WR={w['win_rate']:.0%}")

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.1 Weakness Analysis",
            "current_config": {
                "fund_ratio": 0.70,
                "growth_composite": 0.30,
                "growth_sub_weights": V041_GROWTH_COMPOSITE_SUB,
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "features": "features_v04_1.parquet",
            "pit_factors": len(all_pit_cols),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "analysis1_factor_effectiveness": {
            "top_10_factors": analysis1_results["top_10"],
            "bottom_10_factors": analysis1_results["bottom_10"],
            "group_icir": analysis1_results["group_icir"],
            "correlations": analysis1_results["correlations"],
            "ic_results": analysis1_results["ic_results"],
        },
        "analysis2_weight_sensitivity": analysis2_results,
        "analysis3_growth_sub_weights": analysis3_results,
        "analysis4_training_windows": [
            {k: v for k, v in r.items() if k != "window_details"}
            for r in analysis4_results
        ],
        "current_v041_result": serialize(res_current) if res_current else None,
        "current_v041_windows": serialize(wins_current) if wins_current else None,
        "all_candidates": serialize(tracker.candidates),
        "final_best": serialize(final_best) if final_best else None,
        "recommendations": generate_recommendations(
            analysis1_results, analysis2_results, analysis3_results,
            analysis4_results, res_current, final_best
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {OUTPUT_PATH}")
    print(f"⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")

    # 最终摘要
    print(f"\n{'='*80}")
    print("📋 弱点分析摘要")
    print(f"{'='*80}")
    if res_current:
        ri_status = "PASS" if res_current["rank_inversion"]["passed"] else "FAIL"
        print(f"  V0.4.1当前 Sharpe={res_current['sharpe']:.3f}  MaxDD={res_current['max_dd']:.1%}  RI={ri_status}")
    if final_best and final_best["name"] != "none":
        ri_status = "PASS" if final_best["rank_inversion_passed"] else "FAIL"
        print(f"  最优配置  Sharpe={final_best['sharpe']:.3f}  MaxDD={final_best['max_dd']:.1%}  RI={ri_status}")
        print(f"  最优权重: {final_best['weights']}")
    print(f"{'='*80}")


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


def generate_recommendations(ic_analysis, weight_results, sub_weight_results, window_results, current_res, best_candidate):
    """生成优化建议。"""
    recs = []

    # 因子有效性建议
    if ic_analysis.get("group_icir"):
        sorted_groups = sorted(ic_analysis["group_icir"].items(), key=lambda x: x[1]["avg_icir"], reverse=True)
        weak_groups = [g for g, v in sorted_groups if v["avg_icir"] < 0.03]
        if weak_groups:
            recs.append({
                "type": "factor_effectiveness",
                "severity": "HIGH",
                "finding": f"弱因子组 (avg_ICIR<0.03): {weak_groups}",
                "recommendation": "降低这些因子组权重或移除",
            })

    # 权重敏感性建议
    if weight_results:
        valid_weights = [r for r in weight_results if r["ri_passed"]]
        if valid_weights:
            best_w = max(valid_weights, key=lambda x: x["sharpe"])
            recs.append({
                "type": "weight_sensitivity",
                "severity": "MEDIUM",
                "finding": f"最优fund_ratio={best_w['fund_ratio']}, growth_composite={best_w['growth_composite']}, Sharpe={best_w['sharpe']:.3f}",
                "recommendation": f"建议从fund_ratio=0.70调整到{best_w['fund_ratio']}, growth_composite从0.30调整到{best_w['growth_composite']}",
            })

    # 子权重建议
    if sub_weight_results:
        valid_sub = [r for r in sub_weight_results if r["ri_passed"]]
        if valid_sub:
            best_sub = max(valid_sub, key=lambda x: x["sharpe"])
            recs.append({
                "type": "sub_weight",
                "severity": "MEDIUM",
                "finding": f"最优子权重: fund_growth={best_sub['fund_growth']}, analyst={best_sub['analyst']}, income={best_sub['income']}",
                "recommendation": f"调整growth_composite子权重从(0.5/0.3/0.2)到({best_sub['fund_growth']}/{best_sub['analyst']}/{best_sub['income']})",
            })

    # 训练窗口建议
    if window_results:
        valid_windows = [r for r in window_results if r["ri_passed"]]
        if valid_windows:
            best_win = max(valid_windows, key=lambda x: x["sharpe"])
            recs.append({
                "type": "training_window",
                "severity": "LOW",
                "finding": f"最优训练窗口: {best_win['months']}个月 (Sharpe={best_win['sharpe']:.3f})",
                "recommendation": f"当前6个月可能不是最优, 建议测试{best_win['months']}个月",
            })

    # RI不通过建议
    if current_res and not current_res["rank_inversion"]["passed"]:
        recs.append({
            "type": "rank_inversion",
            "severity": "CRITICAL",
            "finding": f"Rank Inversion未通过: {current_res['rank_inversion']['reason']}",
            "recommendation": "RI不通过说明模型区分度不足, 需要重新设计因子组合或训练窗口",
        })

    return recs


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
