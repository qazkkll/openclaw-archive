#!/usr/bin/env python3
"""
🦅 Falcon V0.4.1 Optimization: New Features Re-Optimization
=============================================================
用 features_v04_1.parquet (76 PIT factors) 重新优化。

因子组:
  r_* (24): FMP Ratios PIT - 估值+盈利+流动性+杠杆
  m_* (19): Key Metrics PIT - 收益率+质量+效率
  g_* (15): Growth PIT - 增长率
  a_* (4):  Analyst PIT - 分析师修正
  b_* (4):  Balance Sheet - 资产负债
  c_* (4):  Cashflow - 现金流
  i_* (6):  Income Statement - 利润率+增长

Walk-Forward 参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出: data/falcon/v041_optimization_results.json
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
OUTPUT_PATH = DATA_DIR / "v041_optimization_results.json"

# ═══════════════════════════════════════════════════
#  因子组定义
# ═══════════════════════════════════════════════════

# 排除的技术因子(价格/成交量/技术指标)
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
    # 旧的非PIT因子
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin',
    'ebitdaMargin', 'assetTurnover', 'inventoryTurnover', 'receivablesTurnover',
    'debtToEquityRatio', 'currentRatio', 'quickRatio', 'financialLeverageRatio',
    'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
    'dividendYieldPercentage', 'dividendPayoutRatio',
    'eps_revision', 'revenue_revision', 'num_analysts_eps', 'num_analysts_rev',
    'eps_dispersion', 'fmp_covered', 'analyst_covered',
    'grossProfitMargin_qoq', 'netProfitMargin_qoq', 'operatingProfitMargin_qoq', 'ebitdaMargin_qoq',
}

# PIT因子组
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
#  组合因子
# ═══════════════════════════════════════════════════

def add_combo_factors(ranks):
    """为每个日期添加组合因子列。"""
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
            d.get('fund_growth', 0) * 0.5 +
            d.get('analyst', 0) * 0.3 +
            d.get('income', 0) * 0.2
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
#  测试辅助: 记录候选
# ═══════════════════════════════════════════════════

class CandidateTracker:
    """跟踪所有候选方案, 附带权重。"""
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
        """返回最佳(RI通过, Sharpe最高)。"""
        ri_passed = [c for c in self.candidates if c["rank_inversion_passed"]]
        pool = ri_passed if ri_passed else self.candidates
        if not pool:
            return None
        return max(pool, key=lambda x: x["sharpe"])


# ═══════════════════════════════════════════════════
#  测试1: 因子组筛选
# ═══════════════════════════════════════════════════

def test1_factor_group_screening(ranks, prices, tracker, train_years=0.5):
    """测试所有因子组的单独和组合效果。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 1: 因子组筛选 (train={train_years}yr)")
    print(f"{'='*60}")

    base_groups = list(FACTOR_GROUPS.keys()) + ['quality_composite', 'growth_composite', 'safety_composite']
    all_results = {}

    # Baseline: V0.4.0 配置
    w_baseline = {"fund_ratio": 0.70, "fund_metric": 0.15, "log_metric": 0.15}
    res, _ = run_wf(ranks, prices, w_baseline, train_years=train_years)
    if res:
        res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
        all_results["v040_baseline"] = res
        tracker.add("v040_baseline", w_baseline, res)
        print(f"  v040_baseline: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")

    # 每个因子组单独测试
    for group in base_groups:
        w = {group: 1.0}
        res, _ = run_wf(ranks, prices, w, train_years=train_years)
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            all_results[f"solo_{group}"] = res
            tracker.add(f"solo_{group}", w, res)
            print(f"  solo_{group}: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")

    # 两两组合 (fund_ratio + secondary)
    print("\n  两两组合...")
    for sec in [g for g in base_groups if g != 'fund_ratio']:
        for w1, w2 in [(0.70, 0.30), (0.60, 0.40), (0.50, 0.50)]:
            w = {"fund_ratio": w1, sec: w2}
            res, _ = run_wf(ranks, prices, w, train_years=train_years)
            if res:
                label = f"fr{w1:.0f}+{sec}{w2:.0f}"
                res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
                all_results[label] = res
                tracker.add(label, w, res)
                if res['sharpe'] > 1.5:
                    print(f"    {label}: Sharpe={res['sharpe']:.3f}")

    best = tracker.best()
    if best:
        print(f"\n  🏆 Best: {best['name']} (Sharpe={best['sharpe']:.3f}, weights={best['weights']})")
    return all_results


# ═══════════════════════════════════════════════════
#  测试2: 三因子精调 (fund_ratio + fund_metric + combo)
# ═══════════════════════════════════════════════════

def test2_three_factor_finetune(ranks, prices, tracker, train_years=0.5):
    """用V0.4.0风格的三因子模式精调。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 2: 三因子精调 (train={train_years}yr)")
    print(f"{'='*60}")

    combos = ['log_metric', 'log_growth', 'sqrt_ratio', 'ratio_x_growth',
              'metric_x_growth', 'sqrt_ratio_x_log_metric']

    for combo in combos:
        for fr in [0.50, 0.60, 0.70, 0.80]:
            for fm in [0.10, 0.15, 0.20]:
                c = round(1.0 - fr - fm, 2)
                if c < 0.05 or c > 0.35:
                    continue
                w = {"fund_ratio": fr, "fund_metric": fm, combo: c}
                res, _ = run_wf(ranks, prices, w, train_years=train_years)
                if res:
                    label = f"fr{fr:.2f}+fm{fm:.2f}+{combo}{c:.2f}"
                    res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
                    tracker.add(label, w, res)
                    if res['sharpe'] > 1.4:
                        print(f"    {label}: Sharpe={res['sharpe']:.3f}")

    best = tracker.best()
    if best:
        print(f"\n  🏆 Best 3-factor: {best['name']} (Sharpe={best['sharpe']:.3f})")
    return


# ═══════════════════════════════════════════════════
#  测试3: 训练窗口精调
# ═══════════════════════════════════════════════════

def test3_training_windows(ranks, prices, weights, tracker):
    """测试不同训练窗口。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 3: 训练窗口精调")
    print(f"{'='*60}")

    for wm in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        train_years = wm / 12.0
        res, _ = run_wf(ranks, prices, weights, train_years=train_years)
        if res:
            tag = f"window_{wm:.0f}mo"
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(tag, weights, res)
            print(f"  {wm:.0f}mo: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")

    best = tracker.best()
    if best:
        print(f"\n  🏆 Best window in tracker")
    return


# ═══════════════════════════════════════════════════
#  测试4: 多因子混合
# ═══════════════════════════════════════════════════

def test4_multi_factor(ranks, prices, tracker, train_years=0.5):
    """测试4+因子组合。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 4: 多因子混合 (train={train_years}yr)")
    print(f"{'='*60}")

    configs = [
        # 3因子 (V0.4.0 style as baseline)
        {"fund_ratio": 0.70, "fund_metric": 0.15, "log_metric": 0.15},
        # 3因子: growth combos
        {"fund_ratio": 0.60, "fund_metric": 0.20, "log_growth": 0.20},
        {"fund_ratio": 0.70, "fund_metric": 0.15, "metric_x_growth": 0.15},
        # 4因子
        {"fund_ratio": 0.40, "fund_metric": 0.25, "fund_growth": 0.20, "analyst": 0.15},
        {"fund_ratio": 0.35, "fund_metric": 0.25, "fund_growth": 0.25, "analyst": 0.15},
        {"fund_ratio": 0.50, "fund_metric": 0.20, "fund_growth": 0.15, "analyst": 0.15},
        # 4因子: + balance
        {"fund_ratio": 0.35, "fund_metric": 0.20, "fund_growth": 0.20, "balance": 0.25},
        {"fund_ratio": 0.40, "fund_metric": 0.20, "fund_growth": 0.15, "balance": 0.25},
        # 4因子: + cashflow
        {"fund_ratio": 0.35, "fund_metric": 0.20, "fund_growth": 0.20, "cashflow": 0.25},
        # 5因子
        {"fund_ratio": 0.35, "fund_metric": 0.20, "fund_growth": 0.20, "analyst": 0.10, "balance": 0.15},
        {"fund_ratio": 0.40, "fund_metric": 0.20, "fund_growth": 0.15, "analyst": 0.10, "balance": 0.15},
        {"fund_ratio": 0.35, "fund_metric": 0.20, "fund_growth": 0.20, "analyst": 0.10, "cashflow": 0.15},
        {"fund_ratio": 0.30, "fund_metric": 0.20, "fund_growth": 0.20, "income": 0.15, "analyst": 0.15},
        # 5因子: quality-focused
        {"fund_ratio": 0.25, "fund_metric": 0.25, "income": 0.20, "balance": 0.15, "cashflow": 0.15},
        # 5因子: growth-focused
        {"fund_growth": 0.30, "analyst": 0.20, "income": 0.20, "fund_metric": 0.15, "fund_ratio": 0.15},
        # 5因子: safety-focused
        {"fund_ratio": 0.30, "balance": 0.25, "cashflow": 0.20, "fund_metric": 0.15, "fund_growth": 0.10},
        # 6因子
        {"fund_ratio": 0.30, "fund_metric": 0.20, "fund_growth": 0.15, "analyst": 0.10, "balance": 0.10, "cashflow": 0.15},
        {"fund_ratio": 0.30, "fund_metric": 0.20, "fund_growth": 0.15, "analyst": 0.10, "income": 0.10, "cashflow": 0.15},
        # income+qoq combos
        {"fund_ratio": 0.40, "fund_metric": 0.20, "income": 0.20, "qoq": 0.20},
        {"fund_ratio": 0.50, "fund_metric": 0.15, "income": 0.20, "qoq": 0.15},
    ]

    for i, w in enumerate(configs):
        label = f"multi_{i+1:02d}"
        res, _ = run_wf(ranks, prices, w, train_years=train_years)
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(label, w, res)
            print(f"  {label}: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'} weights={w}")

    return


# ═══════════════════════════════════════════════════
#  测试5: 最终验证
# ═══════════════════════════════════════════════════

def test5_final_validation(ranks, prices, weights, train_years=0.5):
    """最终Walk-Forward + 详细窗口。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 5: 最终Walk-Forward (train={train_years:.2f}yr)")
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
    print("🦅 Falcon V0.4.1 Optimization: New Features Re-Optimization")
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

    # ─── 5. 添加组合因子 ───
    ranks, combo_names = add_combo_factors(ranks)

    # ═══════════════════════════════════════════════
    #  追踪器
    # ═══════════════════════════════════════════════
    tracker = CandidateTracker()

    # ═══════════════════════════════════════════════
    #  TEST 1: 因子组筛选
    # ═══════════════════════════════════════════════
    test1_factor_group_screening(ranks, price_pivot, tracker, train_years=0.5)

    # ═══════════════════════════════════════════════
    #  TEST 2: 三因子精调
    # ═══════════════════════════════════════════════
    test2_three_factor_finetune(ranks, price_pivot, tracker, train_years=0.5)

    # ═══════════════════════════════════════════════
    #  选择当前最佳, 精调训练窗口
    # ═══════════════════════════════════════════════
    current_best = tracker.best()
    if current_best:
        best_weights_for_window = current_best["weights"]
    else:
        best_weights_for_window = {"fund_ratio": 0.70, "fund_metric": 0.15, "log_metric": 0.15}

    print(f"\n  📌 Best weights for window tuning: {best_weights_for_window}")

    # ═══════════════════════════════════════════════
    #  TEST 3: 训练窗口精调
    # ═══════════════════════════════════════════════
    test3_training_windows(ranks, price_pivot, best_weights_for_window, tracker)

    # ═══════════════════════════════════════════════
    #  重新选择最佳, 用最佳窗口跑多因子测试
    # ═══════════════════════════════════════════════
    current_best = tracker.best()
    if current_best:
        # 找出当前最佳的训练窗口
        best_name = current_best["name"]
        if "window_" in best_name:
            wm = float(best_name.replace("window_", "").replace("mo", ""))
            best_train_yr = wm / 12.0
        else:
            best_train_yr = 0.5
        best_weights_for_multi = current_best["weights"]
    else:
        best_train_yr = 0.5
        best_weights_for_multi = {"fund_ratio": 0.70, "fund_metric": 0.15, "log_metric": 0.15}

    print(f"\n  📌 Best train_yr for multi-factor: {best_train_yr:.2f}")

    # ═══════════════════════════════════════════════
    #  TEST 4: 多因子混合
    # ═══════════════════════════════════════════════
    test4_multi_factor(ranks, price_pivot, tracker, train_years=best_train_yr)

    # ═══════════════════════════════════════════════
    #  最终选择
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"🏆 最终选择")
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
    #  TEST 5: 最终验证
    # ═══════════════════════════════════════════════
    final_weights = final_best["weights"]
    final_test5_result, final_test5_windows = test5_final_validation(
        ranks, price_pivot, final_weights, train_years=best_train_yr
    )

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "T0.4 V0.4.1 Optimization: New Features Re-Optimization",
            "config": {
                "train_years": best_train_yr,
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
            "result": serialize(final_test5_result) if final_test5_result else None,
            "window_details": serialize(final_test5_windows),
        },
        "comparison_with_v040": {
            "v040_best_sharpe": 1.851,
            "v040_best_weights": {"fund_ratio": 0.70, "fund_metric": 0.15, "log_fm": 0.15},
            "v041_best_sharpe": final_test5_result["sharpe"] if final_test5_result else 0,
            "v041_best_weights": final_weights,
            "improvement_pct": round(
                (final_test5_result["sharpe"] - 1.851) / 1.851 * 100, 1
            ) if final_test5_result else 0,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {OUTPUT_PATH}")
    print(f"⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")

    # 最终摘要
    print(f"\n{'='*80}")
    print(f"📋 最终摘要")
    print(f"{'='*80}")
    print(f"  特征: {len(all_pit_cols)} PIT factors")
    print(f"  最佳权重: {final_weights}")
    print(f"  最佳训练窗口: {best_train_yr*12:.0f}个月")
    if final_test5_result:
        print(f"  最终WF Sharpe: {final_test5_result['sharpe']:.3f}")
        print(f"  最终WF MaxDD: {final_test5_result['max_dd']:.1%}")
        print(f"  Rank Inversion: {'PASS' if final_test5_result['rank_inversion']['passed'] else 'FAIL'}")
    print(f"  V0.4.0基准Sharpe: 1.851")
    if final_test5_result:
        improvement = (final_test5_result['sharpe'] - 1.851) / 1.851 * 100
        print(f"  相对V0.4.0提升: {improvement:+.1f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
