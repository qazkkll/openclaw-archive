#!/usr/bin/env python3
"""
🦅 Falcon V0.4.4: High ICIR Factor Group Expansion
===================================================
测试添加高ICIR因子组到growth_composite，寻找最优配置。

V0.4.3基准:
  fund_ratio=0.70 + growth_composite=0.30
  growth_composite: 0.60×fund_growth + 0.25×analyst + 0.15×income
  WF Sharpe: 2.007, RI=63.2%

未使用的高ICIR因子组:
  qoq:        ICIR=0.1917, 4/4强因子
  balance:    ICIR=0.1391, 3/4强因子
  cashflow:   ICIR=0.1312, 4/4强因子
  fund_metric: ICIR=0.1232, 17/19强因子

Walk-Forward 参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出: data/falcon/v044_factor_expansion_results.json
"""
import sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime
from itertools import product

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
OUTPUT_PATH = DATA_DIR / "v044_factor_expansion_results.json"

# ═══════════════════════════════════════════════════
#  因子组定义
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
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
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
#  组合因子: growth_composite的变体
# ═══════════════════════════════════════════════════

def add_growth_composite_variants(ranks):
    """为growth_composite添加不同高ICIR因子的变体。"""

    # V0.4.3 baseline growth_composite
    gc_baseline = lambda d: (
        d.get('fund_growth', 0) * 0.60 +
        d.get('analyst', 0) * 0.25 +
        d.get('income', 0) * 0.15
    )

    # Variant 1: + qoq (ICIR=0.1917)
    gc_qoq_light = lambda d: (
        d.get('fund_growth', 0) * 0.50 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('qoq', 0) * 0.15
    )
    gc_qoq_med = lambda d: (
        d.get('fund_growth', 0) * 0.45 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('qoq', 0) * 0.20
    )
    gc_qoq_heavy = lambda d: (
        d.get('fund_growth', 0) * 0.40 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('qoq', 0) * 0.25
    )

    # Variant 2: + balance (ICIR=0.1391)
    gc_bal_light = lambda d: (
        d.get('fund_growth', 0) * 0.50 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('balance', 0) * 0.15
    )
    gc_bal_med = lambda d: (
        d.get('fund_growth', 0) * 0.45 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('balance', 0) * 0.20
    )
    gc_bal_heavy = lambda d: (
        d.get('fund_growth', 0) * 0.40 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('balance', 0) * 0.25
    )

    # Variant 3: + cashflow (ICIR=0.1312)
    gc_cf_light = lambda d: (
        d.get('fund_growth', 0) * 0.50 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('cashflow', 0) * 0.15
    )
    gc_cf_med = lambda d: (
        d.get('fund_growth', 0) * 0.45 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('cashflow', 0) * 0.20
    )
    gc_cf_heavy = lambda d: (
        d.get('fund_growth', 0) * 0.40 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('cashflow', 0) * 0.25
    )

    # Variant 4: + fund_metric (ICIR=0.1232)
    gc_fm_light = lambda d: (
        d.get('fund_growth', 0) * 0.50 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('fund_metric', 0) * 0.15
    )
    gc_fm_med = lambda d: (
        d.get('fund_growth', 0) * 0.45 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('fund_metric', 0) * 0.20
    )
    gc_fm_heavy = lambda d: (
        d.get('fund_growth', 0) * 0.40 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('fund_metric', 0) * 0.25
    )

    variants = {
        'gc_baseline': gc_baseline,
        'gc_qoq_15': gc_qoq_light, 'gc_qoq_20': gc_qoq_med, 'gc_qoq_25': gc_qoq_heavy,
        'gc_bal_15': gc_bal_light, 'gc_bal_20': gc_bal_med, 'gc_bal_25': gc_bal_heavy,
        'gc_cf_15': gc_cf_light, 'gc_cf_20': gc_cf_med, 'gc_cf_25': gc_cf_heavy,
        'gc_fm_15': gc_fm_light, 'gc_fm_20': gc_fm_med, 'gc_fm_25': gc_fm_heavy,
    }

    for date in ranks:
        df = ranks[date]
        for name, func in variants.items():
            try:
                df[name] = func(df.to_dict('series'))
            except Exception:
                df[name] = np.nan
        ranks[date] = df

    print(f"  ✅ Growth composite variants: {len(variants)}")
    return ranks, list(variants.keys())


# ═══════════════════════════════════════════════════
#  Rank Inversion 检查
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


# ═══════════════════════════════════════════════════
#  Walk-Forward
# ═══════════════════════════════════════════════════

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
#  Candidate Tracker
# ═══════════════════════════════════════════════════

class CandidateTracker:
    """跟踪所有候选方案。"""
    def __init__(self):
        self.candidates = []

    def add(self, name, weights, res, category=""):
        if res and "sharpe" in res:
            self.candidates.append({
                "name": name,
                "category": category,
                "sharpe": res["sharpe"],
                "max_dd": res["max_dd"],
                "cagr": res["cagr"],
                "win_rate": res["win_rate"],
                "n_windows": res["n_windows"],
                "rank_inversion_passed": res["rank_inversion"]["passed"],
                "rank_inversion_detail": res["rank_inversion"],
                "weights": dict(weights),
            })

    def best(self):
        """返回最佳(RI通过, Sharpe最高)。"""
        ri_passed = [c for c in self.candidates if c["rank_inversion_passed"]]
        pool = ri_passed if ri_passed else self.candidates
        if not pool:
            return None
        return max(pool, key=lambda x: x["sharpe"])

    def best_in_category(self, category):
        """返回某类别的最佳。"""
        filtered = [c for c in self.candidates if c["category"] == category and c["rank_inversion_passed"]]
        if not filtered:
            filtered = [c for c in self.candidates if c["category"] == category]
        if not filtered:
            return None
        return max(filtered, key=lambda x: x["sharpe"])


# ═══════════════════════════════════════════════════
#  测试1: 单因子组添加到growth_composite
# ═══════════════════════════════════════════════════

def test1_gc_single_factor(ranks, prices, tracker, train_years=0.5):
    """测试1: 在growth_composite中添加单个高ICIR因子。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 1: Growth Composite + Single High ICIR Factor")
    print(f"{'='*60}")

    # V0.4.3 baseline
    w_baseline = {"fund_ratio": 0.70, "gc_baseline": 0.30}
    res, _ = run_wf(ranks, prices, w_baseline, train_years=train_years)
    if res:
        res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
        tracker.add("v043_baseline", w_baseline, res, category="baseline")
        print(f"  v043_baseline: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")

    # Test each variant
    factors = [
        ("qoq", 0.1917, ["gc_qoq_15", "gc_qoq_20", "gc_qoq_25"]),
        ("balance", 0.1391, ["gc_bal_15", "gc_bal_20", "gc_bal_25"]),
        ("cashflow", 0.1312, ["gc_cf_15", "gc_cf_20", "gc_cf_25"]),
        ("fund_metric", 0.1232, ["gc_fm_15", "gc_fm_20", "gc_fm_25"]),
    ]

    for factor_name, icir, variants in factors:
        print(f"\n  --- {factor_name} (ICIR={icir}) ---")
        for variant in variants:
            w = {"fund_ratio": 0.70, variant: 0.30}
            res, _ = run_wf(ranks, prices, w, train_years=train_years)
            if res:
                res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
                tracker.add(f"gc_{factor_name}_{variant.split('_')[-1]}", w, res, category=f"single_{factor_name}")
                print(f"    {variant}: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")


# ═══════════════════════════════════════════════════
#  测试2: 调整fund_ratio和growth_composite的权重
# ═══════════════════════════════════════════════════

def test2_weight_sweep(ranks, prices, tracker, train_years=0.5):
    """测试2: 对最优单因子变体，调整fund_ratio/gc的权重。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 2: Weight Sweep for Best Single Factor")
    print(f"{'='*60}")

    # Find best single factor variant from tracker
    best_per_factor = {}
    for cat_prefix in ["single_qoq", "single_balance", "single_cashflow", "single_fund_metric"]:
        best = tracker.best_in_category(cat_prefix)
        if best:
            best_per_factor[cat_prefix] = best
            print(f"  Best {cat_prefix}: {best['name']} Sharpe={best['sharpe']:.3f} weights={best['weights']}")

    if not best_per_factor:
        print("  ⚠️ No best single factor found, using defaults")
        best_per_factor = {
            "single_qoq": {"weights": {"fund_ratio": 0.70, "gc_qoq_20": 0.30}},
            "single_balance": {"weights": {"fund_ratio": 0.70, "gc_bal_20": 0.30}},
            "single_cashflow": {"weights": {"fund_ratio": 0.70, "gc_cf_20": 0.30}},
            "single_fund_metric": {"weights": {"fund_ratio": 0.70, "gc_fm_20": 0.30}},
        }

    for cat_prefix, best_info in best_per_factor.items():
        # Get the GC variant name from weights
        gc_variant = [k for k in best_info["weights"] if k.startswith("gc_")]
        if not gc_variant:
            continue
        gc_variant = gc_variant[0]

        print(f"\n  Sweeping weights for {gc_variant}...")
        for fr in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
            gc_w = round(1.0 - fr, 2)
            if gc_w < 0.15 or gc_w > 0.50:
                continue
            w = {"fund_ratio": fr, gc_variant: gc_w}
            res, _ = run_wf(ranks, prices, w, train_years=train_years)
            if res:
                res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
                label = f"sweep_{gc_variant}_fr{fr:.2f}"
                tracker.add(label, w, res, category=f"sweep_{cat_prefix}")
                if res["sharpe"] > 1.8:
                    print(f"    fr={fr:.2f} gc={gc_w:.2f}: Sharpe={res['sharpe']:.3f}")


# ═══════════════════════════════════════════════════
#  测试3: 多因子组合
# ═══════════════════════════════════════════════════

def test3_multi_factor_combos(ranks, prices, tracker, train_years=0.5):
    """测试3: fund_ratio + growth_composite + additional high ICIR factor。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 3: Multi-Factor Combinations (FR + GC + Factor)")
    print(f"{'='*60}")

    # Use the best single factor variants
    # Find best variants from tracker
    best_qoq = tracker.best_in_category("single_qoq")
    best_bal = tracker.best_in_category("single_balance")
    best_cf = tracker.best_in_category("single_cashflow")
    best_fm = tracker.best_in_category("single_fund_metric")

    # Default GC variant
    gc_default = "gc_qoq_20"  # fallback

    # Configs: fund_ratio + growth_composite + additional factor
    configs = [
        # FR + GC + qoq
        ("fr_gc_qoq", {
            "fund_ratio": 0.60, "gc_baseline": 0.25, "qoq": 0.15,
        }),
        ("fr_gc_qoq_2", {
            "fund_ratio": 0.55, "gc_baseline": 0.25, "qoq": 0.20,
        }),
        ("fr_gc_qoq_3", {
            "fund_ratio": 0.50, "gc_baseline": 0.25, "qoq": 0.25,
        }),
        ("fr_gc_qoq_4", {
            "fund_ratio": 0.65, "gc_baseline": 0.20, "qoq": 0.15,
        }),

        # FR + GC + balance
        ("fr_gc_bal", {
            "fund_ratio": 0.60, "gc_baseline": 0.25, "balance": 0.15,
        }),
        ("fr_gc_bal_2", {
            "fund_ratio": 0.55, "gc_baseline": 0.25, "balance": 0.20,
        }),
        ("fr_gc_bal_3", {
            "fund_ratio": 0.50, "gc_baseline": 0.25, "balance": 0.25,
        }),
        ("fr_gc_bal_4", {
            "fund_ratio": 0.65, "gc_baseline": 0.20, "balance": 0.15,
        }),

        # FR + GC + cashflow
        ("fr_gc_cf", {
            "fund_ratio": 0.60, "gc_baseline": 0.25, "cashflow": 0.15,
        }),
        ("fr_gc_cf_2", {
            "fund_ratio": 0.55, "gc_baseline": 0.25, "cashflow": 0.20,
        }),
        ("fr_gc_cf_3", {
            "fund_ratio": 0.50, "gc_baseline": 0.25, "cashflow": 0.25,
        }),
        ("fr_gc_cf_4", {
            "fund_ratio": 0.65, "gc_baseline": 0.20, "cashflow": 0.15,
        }),

        # FR + GC + fund_metric
        ("fr_gc_fm", {
            "fund_ratio": 0.60, "gc_baseline": 0.25, "fund_metric": 0.15,
        }),
        ("fr_gc_fm_2", {
            "fund_ratio": 0.55, "gc_baseline": 0.25, "fund_metric": 0.20,
        }),
        ("fr_gc_fm_3", {
            "fund_ratio": 0.50, "gc_baseline": 0.25, "fund_metric": 0.25,
        }),
        ("fr_gc_fm_4", {
            "fund_ratio": 0.65, "gc_baseline": 0.20, "fund_metric": 0.15,
        }),

        # 4-factor: FR + GC + qoq + balance
        ("fr_gc_qoq_bal", {
            "fund_ratio": 0.50, "gc_baseline": 0.20, "qoq": 0.15, "balance": 0.15,
        }),
        ("fr_gc_qoq_bal_2", {
            "fund_ratio": 0.45, "gc_baseline": 0.20, "qoq": 0.20, "balance": 0.15,
        }),

        # 4-factor: FR + GC + qoq + cashflow
        ("fr_gc_qoq_cf", {
            "fund_ratio": 0.50, "gc_baseline": 0.20, "qoq": 0.15, "cashflow": 0.15,
        }),
        ("fr_gc_qoq_cf_2", {
            "fund_ratio": 0.45, "gc_baseline": 0.20, "qoq": 0.20, "cashflow": 0.15,
        }),

        # 4-factor: FR + GC + balance + cashflow
        ("fr_gc_bal_cf", {
            "fund_ratio": 0.50, "gc_baseline": 0.20, "balance": 0.15, "cashflow": 0.15,
        }),
        ("fr_gc_bal_cf_2", {
            "fund_ratio": 0.45, "gc_baseline": 0.20, "balance": 0.20, "cashflow": 0.15,
        }),

        # 5-factor: FR + GC + qoq + balance + cashflow
        ("fr_gc_qoq_bal_cf", {
            "fund_ratio": 0.45, "gc_baseline": 0.20, "qoq": 0.15, "balance": 0.10, "cashflow": 0.10,
        }),
        ("fr_gc_qoq_bal_cf_2", {
            "fund_ratio": 0.40, "gc_baseline": 0.20, "qoq": 0.15, "balance": 0.15, "cashflow": 0.10,
        }),

        # Use best GC variants from TEST 1
        ("best_gc_qoq_fr70", {
            "fund_ratio": 0.70, "gc_qoq_20": 0.30,
        }),
        ("best_gc_bal_fr70", {
            "fund_ratio": 0.70, "gc_bal_20": 0.30,
        }),
        ("best_gc_cf_fr70", {
            "fund_ratio": 0.70, "gc_cf_20": 0.30,
        }),
        ("best_gc_fm_fr70", {
            "fund_ratio": 0.70, "gc_fm_20": 0.30,
        }),
    ]

    for name, w in configs:
        res, _ = run_wf(ranks, prices, w, train_years=train_years)
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(name, w, res, category="multi_factor")
            print(f"  {name}: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'} weights={w}")


# ═══════════════════════════════════════════════════
#  测试4: 最终验证
# ═══════════════════════════════════════════════════

def test4_final_validation(ranks, prices, weights, train_years=0.5):
    """最终Walk-Forward + 详细窗口。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 4: Final Walk-Forward (train={train_years:.2f}yr)")
    print(f"{'='*60}")
    print(f"  weights: {weights}")

    res, wins = run_wf(ranks, prices, weights, train_years=train_years)
    if res:
        print(f"  Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
              f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}")
        print(f"  Windows={res['n_windows']}  RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")

        valid_wins = [w for w in wins if "sharpe" in w]
        for w in valid_wins:
            ri_mark = "✅" if w["sharpe"] > 0 else "❌"
            print(f"    {ri_mark} W{w['index']}: {w['period']}  "
                  f"Sharpe={w['sharpe']:.3f}  MaxDD={w['max_dd']:.1%}  "
                  f"WR={w['win_rate']:.0%}  Trades={w['n_trades']}")
        return res, wins
    return None, []


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
    print("🦅 Falcon V0.4.4: High ICIR Factor Group Expansion")
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

    # ─── 4. 计算因子组排名 ───
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)

    # ─── 5. 添加growth composite variants ───
    ranks, gc_variants = add_growth_composite_variants(ranks)

    # ═══════════════════════════════════════════════
    #  追踪器
    # ═══════════════════════════════════════════════
    tracker = CandidateTracker()

    # ═══════════════════════════════════════════════
    #  TEST 1: 单因子组添加到growth_composite
    # ═══════════════════════════════════════════════
    test1_gc_single_factor(ranks, price_pivot, tracker, train_years=0.5)

    # ═══════════════════════════════════════════════
    #  TEST 2: 权重扫描
    # ═══════════════════════════════════════════════
    test2_weight_sweep(ranks, price_pivot, tracker, train_years=0.5)

    # ═══════════════════════════════════════════════
    #  TEST 3: 多因子组合
    # ═══════════════════════════════════════════════
    test3_multi_factor_combos(ranks, price_pivot, tracker, train_years=0.5)

    # ═══════════════════════════════════════════════
    #  最终选择
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"🏆 Final Selection")
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
        final_best = {"name": "none", "sharpe": 0, "weights": {}}

    # ═══════════════════════════════════════════════
    #  TEST 4: 最终验证
    # ═══════════════════════════════════════════════
    final_weights = final_best["weights"]
    final_result, final_windows = test4_final_validation(
        ranks, price_pivot, final_weights, train_years=0.5
    )

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.4 High ICIR Factor Group Expansion",
            "config": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "features": "features_v04_1.parquet",
            "pit_factors": len(all_pit_cols),
            "factor_groups": {k: len(v) for k, v in FACTOR_GROUPS.items()},
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "all_candidates": serialize(tracker.candidates),
        "final_best": serialize(final_best),
        "final_validation": {
            "result": serialize(final_result) if final_result else None,
            "window_details": serialize(final_windows),
        },
        "comparison_with_v043": {
            "v043_sharpe": 2.007,
            "v043_weights": {
                "fund_ratio": 0.70,
                "growth_composite": 0.30,
                "gc_composition": "0.60×fund_growth + 0.25×analyst + 0.15×income",
            },
            "v044_best_sharpe": final_result["sharpe"] if final_result else 0,
            "v044_best_weights": final_weights,
            "improvement_pct": round(
                (final_result["sharpe"] - 2.007) / 2.007 * 100, 1
            ) if final_result else 0,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved: {OUTPUT_PATH}")
    print(f"⏱️ Total time: {(time.time()-t0)/60:.1f} minutes")

    # Final summary
    print(f"\n{'='*80}")
    print(f"📋 Final Summary")
    print(f"{'='*80}")
    print(f"  Factors: {len(all_pit_cols)} PIT factors")
    print(f"  Best weights: {final_weights}")
    if final_result:
        print(f"  Final WF Sharpe: {final_result['sharpe']:.3f}")
        print(f"  Final WF MaxDD: {final_result['max_dd']:.1%}")
        print(f"  Rank Inversion: {'PASS' if final_result['rank_inversion']['passed'] else 'FAIL'}")
    print(f"  V0.4.3 baseline Sharpe: 2.007")
    if final_result:
        improvement = (final_result['sharpe'] - 2.007) / 2.007 * 100
        print(f"  Improvement vs V0.4.3: {improvement:+.1f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
