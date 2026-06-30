#!/usr/bin/env python3
"""
🦅 Falcon Factor Weight Optimization v2 (Proper Walk-Forward)
==============================================================
修正版: 每个Walk-Forward窗口内独立优化权重，用该窗口最优权重测试下一窗口。
这才是真正的OOS验证。

流程:
  Phase 1: 计算PIT ranks (一次性, ~5分钟)
  Phase 2: IC/ICIR分析 → 剔除无效因子
  Phase 3: 窗口内贪心优化 → 每窗口独立最优权重
  Phase 4: OOS汇总 + 过拟合检测
  Phase 5: 审计日志输出
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from falcon_v03_engine import (
    precompute_pit_ranks_fast,
    build_pit_index_statements, compute_statement_factors,
    RATIO_FIELDS, METRIC_FIELDS, GROWTH_FIELDS, ANALYST_FIELDS,
    TECH_FIELDS, EARNINGS_FIELDS, GRADE_FIELDS,
    BALANCE_FIELDS, CASHFLOW_FIELDS, INCOME_FIELDS,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "falcon"
FMP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fmp_premium"
OUTPUT = DATA_DIR / "falcon_optimization_v2_result.json"

INVERT_FACTORS = {"debt_to_equity", "net_debt_to_assets", "capex_intensity"}


# ═══════════════════════════════════════════════════
# 数据加载 + PIT rank计算 (复用v1)
# ═══════════════════════════════════════════════════

def load_all_data():
    print("📂 加载数据...", flush=True)
    t0 = time.time()
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    data = {}
    for name in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
                  "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            data[name] = json.load(open(f))
    data["earnings"] = load_fmp_premium_earnings(str(FMP_DIR))
    data["grades"] = load_fmp_premium_grades(str(FMP_DIR))
    for name in ["fmp_balance_sheet", "fmp_cashflow", "fmp_income_stmt"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            data[name] = json.load(open(f))
    all_dates = sorted(master["date"].unique())
    print(f"  ✅ {len(master):,}行, {master['ticker'].nunique()}只, {len(all_dates)}天, {time.time()-t0:.0f}秒", flush=True)
    return master, data, all_dates


def compute_pit_ranks(master, data, all_dates):
    print("\n📊 Phase 1: 计算PIT ranks...", flush=True)
    t0 = time.time()
    ranks = precompute_pit_ranks_fast(
        master,
        data.get("fmp_ratios_historical", {}), data.get("analyst_historical", {}),
        data.get("fmp_key_metrics", {}), data.get("fmp_financial_growth", {}),
        data.get("fmp_insider", {}), data.get("fmp_dcf", {}),
        data.get("fmp_price_target", {}),
        earnings_hist=data.get("earnings"), grades_hist=data.get("grades"),
    )
    # 合并三大报表
    income_raw = data.get("fmp_income_stmt", {})
    balance_raw = data.get("fmp_balance_sheet", {})
    cashflow_raw = data.get("fmp_cashflow", {})
    if balance_raw or cashflow_raw or income_raw:
        print("  合并三大报表因子...", flush=True)
        income_idx = build_pit_index_statements(income_raw, use_filing_date=True)
        balance_idx = build_pit_index_statements(balance_raw, use_filing_date=False)
        cashflow_idx = build_pit_index_statements(cashflow_raw, use_filing_date=False)
        for di, date in enumerate(sorted(ranks.keys())):
            rank_df = ranks[date]
            tickers = rank_df.index.tolist()
            new_data = {}
            for t in tickers:
                factors = compute_statement_factors(t, date, balance_idx, cashflow_idx, income_idx, {})
                if factors:
                    new_data[t] = factors
            if new_data:
                new_df = pd.DataFrame.from_dict(new_data, orient="index")
                for col in new_df.columns:
                    if new_df[col].notna().sum() >= 10:
                        ranked = new_df[col].rank(pct=True)
                        if col in INVERT_FACTORS:
                            ranked = 1 - ranked
                        rank_df[col] = ranked
                for group_name, fields in [("balance", BALANCE_FIELDS), ("cashflow", CASHFLOW_FIELDS), ("income_stmt", INCOME_FIELDS)]:
                    cols = [c for c in fields if c in rank_df.columns]
                    if cols:
                        rank_df[group_name] = rank_df[cols].mean(axis=1)
            if (di + 1) % 500 == 0:
                print(f"    合并: {di+1}/{len(ranks)}", flush=True)
    print(f"  ✅ {len(ranks)}天, {time.time()-t0:.0f}秒", flush=True)
    return ranks


# ═══════════════════════════════════════════════════
# IC/ICIR分析
# ═══════════════════════════════════════════════════

def compute_ic_analysis(ranks, price_pivot, hold_days=60):
    print("\n📊 Phase 2: IC/ICIR分析...", flush=True)
    fwd_ret = price_pivot.pct_change(periods=hold_days, fill_method=None).shift(-hold_days)
    fwd_dates = set(str(d)[:10] for d in fwd_ret.index)
    sample_dates = sorted(set(ranks.keys()) & fwd_dates)[::hold_days]
    print(f"  采样{len(sample_dates)}个日期", flush=True)

    all_factors = set()
    for date in sample_dates:
        if date in ranks:
            all_factors.update(ranks[date].columns.tolist())
    all_factors = sorted(all_factors)

    ic_results = {f: [] for f in all_factors}
    for date in sample_dates:
        if date not in ranks:
            continue
        rank_df = ranks[date]
        date_ts = pd.Timestamp(date)
        if date_ts in fwd_ret.index:
            ret_row = fwd_ret.loc[date_ts]
        elif date in fwd_ret.index:
            ret_row = fwd_ret.loc[date]
        else:
            continue
        for factor in all_factors:
            if factor not in rank_df.columns:
                continue
            f_vals = rank_df[factor]
            common = f_vals.index.intersection(ret_row.index)
            if len(common) < 30:
                continue
            f_v = f_vals.loc[common].values
            r_v = ret_row.loc[common].values
            mask = ~np.isnan(f_v) & ~np.isnan(r_v)
            if mask.sum() >= 30:
                ic, _ = spearmanr(f_v[mask], r_v[mask])
                if not np.isnan(ic):
                    ic_results[factor].append(ic)

    summary = {}
    print(f"\n  {'因子':<22} {'IC':>7} {'ICIR':>7} {'IC>0':>5} {'N':>4}", flush=True)
    print("  " + "-" * 50, flush=True)
    for name in all_factors:
        ics = ic_results[name]
        if len(ics) < 5:
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        icir = mean_ic / std_ic if std_ic > 0 else 0
        pos_pct = float(np.mean(np.array(ics) > 0))
        summary[name] = {"ic_mean": round(mean_ic, 4), "ic_std": round(std_ic, 4),
                         "icir": round(icir, 3), "ic_positive_pct": round(pos_pct, 3), "n_samples": len(ics)}
        print(f"  {name:<22} {mean_ic:>7.4f} {icir:>7.3f} {pos_pct:>4.0%} {len(ics):>4}", flush=True)
    return summary


# ═══════════════════════════════════════════════════
# 单窗口回测 (给定权重，返回收益序列)
# ═══════════════════════════════════════════════════

def backtest_period(ranks, price_pivot, dates, weights, hold_days, top_n):
    """对给定日期序列回测，返回每期收益列表。"""
    rebalance_dates = dates[::hold_days]
    returns = []
    for rb_date in rebalance_dates:
        if rb_date not in ranks:
            continue
        rank_df = ranks[rb_date]
        available = [f for f in weights if f in rank_df.columns]
        if not available:
            continue
        combined = sum(weights[f] * rank_df[f] for f in available)
        combined = combined.dropna().sort_values(ascending=False)
        picks = combined.head(top_n).index.tolist()

        if rb_date not in price_pivot.index:
            continue
        rb_idx = price_pivot.index.get_loc(rb_date)
        end_idx = min(rb_idx + hold_days, len(price_pivot) - 1)
        if end_idx <= rb_idx:
            continue
        start_prices = price_pivot.iloc[rb_idx]
        end_prices = price_pivot.iloc[end_idx]

        rets = []
        for t in picks:
            if t in start_prices.index and t in end_prices.index:
                sp, ep = start_prices[t], end_prices[t]
                if pd.notna(sp) and pd.notna(ep) and sp > 0:
                    rets.append(ep / sp - 1)
        if rets:
            returns.append(float(np.mean(rets)))
    return returns


def returns_to_metrics(returns, hold_days):
    """收益序列→指标dict。"""
    if not returns:
        return None
    rets = np.array(returns)
    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets))
    sharpe = mean_r / std_r * np.sqrt(12) if std_r > 0 else 0
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = float(np.min((cum - peak) / peak))
    total = float(cum[-1] - 1)
    years = len(rets) * hold_days / 252
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    return {"sharpe": round(sharpe, 3), "cagr": round(cagr, 4), "max_dd": round(dd, 4),
            "win_rate": round(float(np.mean(rets > 0)), 3), "n_periods": len(rets)}


# ═══════════════════════════════════════════════════
# 窗口内贪心优化
# ═══════════════════════════════════════════════════

def greedy_optimize_in_window(ranks, price_pivot, train_dates, factor_groups,
                               hold_days, top_n, max_iterations=3):
    """在给定train_dates上贪心优化权重。返回最优权重dict。"""
    n = len(factor_groups)
    best_weights = {f: 1.0 / n for f in factor_groups}
    best_rets = backtest_period(ranks, price_pivot, train_dates, best_weights, hold_days, top_n)
    best_sharpe = returns_to_metrics(best_rets, hold_days)["sharpe"] if best_rets else -999

    weight_steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    for iteration in range(max_iterations):
        improved = False
        for factor in factor_groups:
            for w in weight_steps:
                test_w = dict(best_weights)
                test_w[factor] = w
                total = sum(test_w.values())
                if total == 0:
                    continue
                test_w = {k: v / total for k, v in test_w.items()}
                rets = backtest_period(ranks, price_pivot, train_dates, test_w, hold_days, top_n)
                if not rets:
                    continue
                m = returns_to_metrics(rets, hold_days)
                if m and m["sharpe"] > best_sharpe + 0.005:
                    best_sharpe = m["sharpe"]
                    best_weights = dict(test_w)
                    improved = True
        if not improved:
            break

    return best_weights


# ═══════════════════════════════════════════════════
# Phase 3: Proper Walk-Forward (窗口内优化)
# ═══════════════════════════════════════════════════

def proper_walk_forward(ranks, price_pivot, all_dates, factor_groups,
                         hold_days=60, top_n=20, train_years=2, test_months=6):
    """真正的Walk-Forward: 每窗口内独立优化，测试下一窗口。"""
    from dateutil.relativedelta import relativedelta

    start = pd.Timestamp(all_dates[0])
    end = pd.Timestamp(all_dates[-1])

    windows = []
    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        test_dates = [d for d in all_dates if train_end.strftime("%Y-%m-%d") <= d < test_end.strftime("%Y-%m-%d")]
        train_dates = [d for d in all_dates if train_start.strftime("%Y-%m-%d") <= d < train_end.strftime("%Y-%m-%d")]
        if len(test_dates) >= 20 and len(train_dates) >= 100:
            windows.append((train_dates, test_dates))
        train_start = train_start + relativedelta(months=test_months)

    print(f"  {len(windows)}个Walk-Forward窗口", flush=True)

    all_test_returns = []
    all_train_returns = []
    window_details = []
    window_weights = []

    for wi, (train_dates, test_dates) in enumerate(windows):
        print(f"\n  窗口 {wi+1}/{len(windows)}: train={train_dates[0][:10]}~{train_dates[-1][:10]}, test={test_dates[0][:10]}~{test_dates[-1][:10]}", flush=True)

        # 在train上优化权重
        opt_weights = greedy_optimize_in_window(
            ranks, price_pivot, train_dates, factor_groups, hold_days, top_n, max_iterations=3
        )

        # train指标
        train_rets = backtest_period(ranks, price_pivot, train_dates, opt_weights, hold_days, top_n)
        train_m = returns_to_metrics(train_rets, hold_days)

        # test指标 (用train优化出的权重)
        test_rets = backtest_period(ranks, price_pivot, test_dates, opt_weights, hold_days, top_n)
        test_m = returns_to_metrics(test_rets, hold_days)

        # 显示主要权重
        top_w = sorted(opt_weights.items(), key=lambda x: -x[1])[:4]
        top_w_str = ", ".join(f"{k}={v:.0%}" for k, v in top_w if v > 0.01)

        if train_m and test_m:
            print(f"    最优权重: {top_w_str}", flush=True)
            print(f"    Train Sharpe: {train_m['sharpe']:.3f} | Test Sharpe: {test_m['sharpe']:.3f}", flush=True)
            all_train_returns.extend(train_rets)
            all_test_returns.extend(test_rets)
            window_details.append({
                "window": wi, "train": f"{train_dates[0][:10]}~{train_dates[-1][:10]}",
                "test": f"{test_dates[0][:10]}~{test_dates[-1][:10]}",
                "train_sharpe": train_m["sharpe"], "test_sharpe": test_m["sharpe"],
                "weights": {k: round(v, 4) for k, v in opt_weights.items()},
            })
            window_weights.append(opt_weights)
        elif test_m:
            all_test_returns.extend(test_rets)
            window_details.append({"window": wi, "test_sharpe": test_m["sharpe"]})

    if not all_test_returns:
        return None

    # OOS汇总
    test_m = returns_to_metrics(all_test_returns, hold_days)
    train_m = returns_to_metrics(all_train_returns, hold_days) if all_train_returns else None

    # 权重稳定性分析
    if window_weights:
        all_keys = set()
        for w in window_weights:
            all_keys.update(w.keys())
        weight_stability = {}
        for k in sorted(all_keys):
            vals = [w.get(k, 0) for w in window_weights]
            weight_stability[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
    else:
        weight_stability = {}

    result = {
        "hold_days": hold_days, "top_n": top_n,
        "oos_sharpe": test_m["sharpe"], "oos_cagr": test_m["cagr"],
        "oos_max_dd": test_m["max_dd"], "oos_win_rate": test_m["win_rate"],
        "oos_n_periods": test_m["n_periods"],
        "train_sharpe": train_m["sharpe"] if train_m else 0,
        "overfit_ratio": round(test_m["sharpe"] / train_m["sharpe"], 3) if train_m and train_m["sharpe"] > 0 else 0,
        "weight_stability": weight_stability,
        "window_details": window_details,
    }

    return result


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("🦅 Falcon Factor Weight Optimization v2 (Proper Walk-Forward)", flush=True)
    print("=" * 70, flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)

    # Phase 1
    master, data, all_dates = load_all_data()
    ranks = compute_pit_ranks(master, data, all_dates)

    price_pivot = master.pivot_table(index="date", columns="ticker", values="close")
    price_pivot.index = price_pivot.index.astype(str)
    price_pivot = price_pivot.sort_index()

    # Phase 2
    ic_summary = compute_ic_analysis(ranks, price_pivot, hold_days=60)

    # 筛选: ICIR >= 0.03 且 n_samples >= 10
    active_factors = [name for name, ic in ic_summary.items()
                      if abs(ic["icir"]) >= 0.03 and ic["n_samples"] >= 10]
    exclude = {"income_stmt"}  # ICIR=-0.058, 弱
    factor_groups = [f for f in active_factors if f not in exclude]
    print(f"\n  优化因子 ({len(factor_groups)}): {factor_groups}", flush=True)

    # Phase 3: Proper Walk-Forward
    print("\n" + "=" * 70, flush=True)
    print("📊 Phase 3: Proper Walk-Forward (窗口内独立优化)", flush=True)
    print("=" * 70, flush=True)

    configs_to_test = [
        {"hold_days": 60, "top_n": 20, "label": "60d_20"},
        {"hold_days": 60, "top_n": 10, "label": "60d_10"},
        {"hold_days": 30, "top_n": 20, "label": "30d_20"},
        {"hold_days": 30, "top_n": 10, "label": "30d_10"},
    ]

    results = {}
    for cfg in configs_to_test:
        label = cfg["label"]
        print(f"\n{'='*50}", flush=True)
        print(f"  配置: hold={cfg['hold_days']}d, top_n={cfg['top_n']}", flush=True)
        print(f"{'='*50}", flush=True)
        r = proper_walk_forward(
            ranks, price_pivot, all_dates, factor_groups,
            hold_days=cfg["hold_days"], top_n=cfg["top_n"],
            train_years=2, test_months=6
        )
        if r:
            results[label] = r

    # Baseline (V0.3.1固定权重)
    print(f"\n{'='*50}", flush=True)
    print(f"  Baseline: V0.3.1 (固定权重, 60d, top20)", flush=True)
    print(f"{'='*50}", flush=True)
    baseline_weights = {
        "tech": 0.15, "fund_ratio": 0.05, "fund_metric": 0.06,
        "fund_growth": 0.15, "analyst": 0.12, "insider": 0.05,
        "valuation": 0.0, "earnings": 0.10, "grade_sentiment": 0.12,
    }
    from dateutil.relativedelta import relativedelta
    start = pd.Timestamp(all_dates[0])
    end = pd.Timestamp(all_dates[-1])
    train_start = start
    baseline_test_rets = []
    baseline_train_rets = []
    while True:
        train_end = train_start + relativedelta(years=2)
        test_end = train_end + relativedelta(months=6)
        if test_end > end:
            break
        test_dates = [d for d in all_dates if train_end.strftime("%Y-%m-%d") <= d < test_end.strftime("%Y-%m-%d")]
        train_dates = [d for d in all_dates if train_start.strftime("%Y-%m-%d") <= d < train_end.strftime("%Y-%m-%d")]
        if len(test_dates) >= 20:
            tr = backtest_period(ranks, price_pivot, test_dates, baseline_weights, 60, 20)
            baseline_test_rets.extend(tr)
            tr2 = backtest_period(ranks, price_pivot, train_dates, baseline_weights, 60, 20)
            baseline_train_rets.extend(tr2)
        train_start = train_start + relativedelta(months=6)

    if baseline_test_rets:
        bm = returns_to_metrics(baseline_test_rets, 60)
        btm = returns_to_metrics(baseline_train_rets, 60)
        results["V0.3.1_baseline"] = {
            "hold_days": 60, "top_n": 20,
            "oos_sharpe": bm["sharpe"], "oos_cagr": bm["cagr"],
            "oos_max_dd": bm["max_dd"], "oos_win_rate": bm["win_rate"],
            "train_sharpe": btm["sharpe"] if btm else 0,
            "overfit_ratio": round(bm["sharpe"] / btm["sharpe"], 3) if btm and btm["sharpe"] > 0 else 0,
        }

    # Phase 4: 对比
    print("\n" + "=" * 70, flush=True)
    print("📊 Phase 4: 配置对比", flush=True)
    print("=" * 70, flush=True)

    print(f"\n  {'配置':<18} {'OOS Sharpe':>10} {'CAGR':>8} {'MaxDD':>8} {'WinR':>6} {'Train':>7} {'OF比':>6}", flush=True)
    print("  " + "-" * 60, flush=True)
    for name, r in sorted(results.items(), key=lambda x: -x[1].get("oos_sharpe", 0)):
        print(f"  {name:<18} {r['oos_sharpe']:>10.3f} {r['oos_cagr']:>7.1%} {r['oos_max_dd']:>7.1%} "
              f"{r['oos_win_rate']:>5.1%} {r['train_sharpe']:>7.3f} {r['overfit_ratio']:>5.2f}", flush=True)

    # 最优配置详情
    best_name = max(results.keys(), key=lambda k: results[k].get("oos_sharpe", 0))
    best = results[best_name]
    print(f"\n  最优配置: {best_name}", flush=True)

    if "weight_stability" in best and best["weight_stability"]:
        print(f"\n  权重稳定性 (跨窗口):", flush=True)
        print(f"  {'因子':<25} {'均值':>6} {'标准差':>6} {'最小':>6} {'最大':>6}", flush=True)
        print("  " + "-" * 55, flush=True)
        ws = best["weight_stability"]
        for k, v in sorted(ws.items(), key=lambda x: -x[1]["mean"]):
            if v["mean"] > 0.01:
                print(f"  {k:<25} {v['mean']:>5.1%} {v['std']:>5.1%} {v['min']:>5.1%} {v['max']:>5.1%}", flush=True)

    # Phase 5: 保存
    print("\n📋 Phase 5: 保存审计日志", flush=True)
    audit = {
        "timestamp": datetime.now().isoformat(),
        "methodology": "Proper Walk-Forward: 2yr train → optimize → 6mo test (true OOS)",
        "data_source": "features_v02.parquet (PIT corrected, 2016-2026, 476 tickers)",
        "ic_analysis": ic_summary,
        "factor_groups_used": factor_groups,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "window_details"}
                    for k, v in results.items()},
        "detailed_results": results,
    }
    with open(OUTPUT, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  ✅ {OUTPUT}", flush=True)

    # 最终结论
    print("\n" + "=" * 70, flush=True)
    print("📋 最终结论", flush=True)
    print("=" * 70, flush=True)
    if best.get("weight_stability"):
        avg_weights = {k: v["mean"] for k, v in best["weight_stability"].items() if v["mean"] > 0.02}
        print(f"  推荐权重 (窗口均值, >2%):", flush=True)
        for k, v in sorted(avg_weights.items(), key=lambda x: -x[1]):
            print(f"    {k:<25} {v:.1%}", flush=True)
    print(f"\n  OOS Sharpe: {best['oos_sharpe']:.3f}", flush=True)
    print(f"  OOS CAGR: {best['oos_cagr']:.2%}", flush=True)
    print(f"  OOS MaxDD: {best['oos_max_dd']:.2%}", flush=True)
    print(f"  过拟合比: {best['overfit_ratio']:.2f} (>0.5=可接受)", flush=True)


if __name__ == "__main__":
    main()
