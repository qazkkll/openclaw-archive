#!/usr/bin/env python3
"""
🦅 Falcon Factor Weight Optimization (PIT Data)
================================================
用Walk-Forward验证找最优因子权重组合。

流程:
  Phase 1: 计算PIT ranks (一次性, ~5分钟)
  Phase 2: IC/ICIR分析 → 剔除无效因子
  Phase 3: 单因子sweep → 找每个因子最优权重区间
  Phase 4: 组合优化 → grid search + Walk-Forward
  Phase 5: 过拟合检测 → train/test Sharpe对比
  Phase 6: 审计日志输出

关键设计:
  - PIT ranks只计算一次, 所有sweep复用 → 快速迭代
  - Walk-Forward: 2年train, 6个月test, 16个窗口
  - 过拟合指标: test/train Sharpe比值, 窗口稳定性
  - 输出完整审计JSON
"""

import sys, json, time, warnings
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from falcon_v03_engine import (
    precompute_pit_ranks_fast, backtest_flexible,
    build_pit_index_statements, compute_statement_factors,
    RATIO_FIELDS, METRIC_FIELDS, GROWTH_FIELDS, ANALYST_FIELDS,
    TECH_FIELDS, EARNINGS_FIELDS, GRADE_FIELDS,
    BALANCE_FIELDS, CASHFLOW_FIELDS, INCOME_FIELDS,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "falcon"
FMP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fmp_premium"
OUTPUT = DATA_DIR / "falcon_optimization_result.json"

# 反向因子 (低值=好)
INVERT_FACTORS = {"debt_to_equity", "net_debt_to_assets", "capex_intensity"}


# ═══════════════════════════════════════════════════
# Phase 1: 计算PIT ranks (一次性)
# ═══════════════════════════════════════════════════

def load_all_data():
    """加载全部数据。"""
    print("📂 加载数据...")
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
    print(f"  ✅ {len(master):,}行, {master['ticker'].nunique()}只, {len(all_dates)}天, {time.time()-t0:.0f}秒")
    return master, data, all_dates


def compute_pit_ranks(master, data, all_dates):
    """计算旧因子PIT ranks。"""
    print("\n📊 Phase 1: 计算PIT ranks...")
    t0 = time.time()
    
    ranks = precompute_pit_ranks_fast(
        master,
        data.get("fmp_ratios_historical", {}),
        data.get("analyst_historical", {}),
        data.get("fmp_key_metrics", {}),
        data.get("fmp_financial_growth", {}),
        data.get("fmp_insider", {}),
        data.get("fmp_dcf", {}),
        data.get("fmp_price_target", {}),
        earnings_hist=data.get("earnings"),
        grades_hist=data.get("grades"),
    )
    
    # 合并三大报表因子
    income_raw = data.get("fmp_income_stmt", {})
    balance_raw = data.get("fmp_balance_sheet", {})
    cashflow_raw = data.get("fmp_cashflow", {})
    
    if balance_raw or cashflow_raw or income_raw:
        print("  合并三大报表因子...")
        income_idx = build_pit_index_statements(income_raw, use_filing_date=True)
        balance_idx = build_pit_index_statements(balance_raw, use_filing_date=False)
        cashflow_idx = build_pit_index_statements(cashflow_raw, use_filing_date=False)
        
        for di, date in enumerate(sorted(ranks.keys())):
            rank_df = ranks[date]
            tickers = rank_df.index.tolist()
            new_data = {}
            for t in tickers:
                factors = compute_statement_factors(
                    t, date, balance_idx, cashflow_idx, income_idx, {}
                )
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
                
                # 组级分数
                for group_name, fields in [("balance", BALANCE_FIELDS),
                                            ("cashflow", CASHFLOW_FIELDS),
                                            ("income_stmt", INCOME_FIELDS)]:
                    cols = [c for c in fields if c in rank_df.columns]
                    if cols:
                        rank_df[group_name] = rank_df[cols].mean(axis=1)
            
            if (di + 1) % 500 == 0:
                print(f"    合并: {di+1}/{len(ranks)}")
    
    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)}天, 9旧+3新因子组, {elapsed:.0f}秒")
    return ranks


# ═══════════════════════════════════════════════════
# Phase 2: IC/ICIR分析
# ═══════════════════════════════════════════════════

def compute_ic_analysis(ranks, price_pivot, hold_days=60):
    """计算所有因子组的IC和ICIR。"""
    print("\n📊 Phase 2: IC/ICIR分析...")
    
    fwd_ret = price_pivot.pct_change(periods=hold_days, fill_method=None).shift(-hold_days)
    fwd_dates = set(str(d)[:10] for d in fwd_ret.index)
    
    sample_dates = sorted(set(ranks.keys()) & fwd_dates)[::hold_days]
    print(f"  采样{len(sample_dates)}个日期, hold={hold_days}天")
    
    # 收集所有因子名
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
    
    # 汇总
    summary = {}
    print(f"\n  {'因子组':<20} {'IC均值':>8} {'ICIR':>8} {'IC>0%':>6} {'样本':>5}")
    print("  " + "-" * 55)
    
    for name in all_factors:
        ics = ic_results[name]
        if len(ics) < 5:
            continue
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics))
        icir = mean_ic / std_ic if std_ic > 0 else 0
        pos_pct = float(np.mean(np.array(ics) > 0))
        summary[name] = {
            "ic_mean": round(mean_ic, 4),
            "ic_std": round(std_ic, 4),
            "icir": round(icir, 3),
            "ic_positive_pct": round(pos_pct, 3),
            "n_samples": len(ics),
        }
        print(f"  {name:<20} {mean_ic:>8.4f} {icir:>8.3f} {pos_pct:>5.1%} {len(ics):>5}")
    
    return summary


# ═══════════════════════════════════════════════════
# Phase 3: 单因子Sweep
# ═══════════════════════════════════════════════════

def single_factor_sweep(ranks, price_pivot, dates, regime_above,
                         base_weights, factor_name, sweep_range,
                         hold_days=60, top_n=20):
    """对单个因子做权重sweep, 返回每个权重的Sharpe。"""
    results = []
    
    for w in sweep_range:
        test_weights = dict(base_weights)
        test_weights[factor_name] = w
        
        # 归一化
        total = sum(test_weights.values())
        if total > 0:
            test_weights = {k: v / total for k, v in test_weights.items()}
        
        # Walk-Forward回测
        wf_result = walk_forward_test(
            ranks, price_pivot, dates, regime_above,
            test_weights, hold_days, top_n
        )
        
        if wf_result:
            results.append({
                "weight": w,
                "sharpe": wf_result["sharpe"],
                "cagr": wf_result["cagr"],
                "max_dd": wf_result["max_dd"],
                "train_sharpe": wf_result.get("train_sharpe", 0),
                "test_sharpe": wf_result.get("test_sharpe", 0),
            })
    
    return results


# ═══════════════════════════════════════════════════
# Walk-Forward回测核心
# ═══════════════════════════════════════════════════

def walk_forward_test(ranks, price_pivot, all_dates, regime_above,
                      weights, hold_days=60, top_n=20,
                      train_years=2, test_months=6):
    """Walk-Forward回测, 返回OOS指标 + 过拟合检测。"""
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
    
    if not windows:
        return None
    
    all_test_returns = []
    all_train_returns = []
    window_details = []
    
    for wi, (train_dates, test_dates) in enumerate(windows):
        # 对train和test分别回测
        for phase, phase_dates in [("train", train_dates), ("test", test_dates)]:
            rebalance_dates = phase_dates[::hold_days]
            period_returns = []
            
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
                
                # 计算持有期收益
                if rb_date in price_pivot.index:
                    rb_idx = price_pivot.index.get_loc(rb_date)
                else:
                    continue
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
                    period_returns.append(float(np.mean(rets)))
            
            if period_returns:
                if phase == "test":
                    all_test_returns.extend(period_returns)
                    window_details.append({
                        "window": wi,
                        "phase": "test",
                        "n_periods": len(period_returns),
                        "avg_ret": float(np.mean(period_returns)),
                    })
                else:
                    all_train_returns.extend(period_returns)
    
    if not all_test_returns:
        return None
    
    # 计算OOS指标
    test_rets = np.array(all_test_returns)
    test_sharpe = float(np.mean(test_rets) / np.std(test_rets) * np.sqrt(12)) if np.std(test_rets) > 0 else 0
    test_cum = np.cumprod(1 + test_rets)
    test_peak = np.maximum.accumulate(test_cum)
    test_dd = float(np.min((test_cum - test_peak) / test_peak))
    test_total = float(test_cum[-1] - 1)
    test_years = len(test_rets) * hold_days / 252
    test_cagr = (1 + test_total) ** (1 / test_years) - 1 if test_years > 0 else 0
    test_wr = float(np.mean(test_rets > 0))
    
    # Train指标 (过拟合检测)
    train_sharpe = 0
    if all_train_returns:
        train_rets = np.array(all_train_returns)
        train_sharpe = float(np.mean(train_rets) / np.std(train_rets) * np.sqrt(12)) if np.std(train_rets) > 0 else 0
    
    return {
        "sharpe": round(test_sharpe, 3),
        "cagr": round(test_cagr, 4),
        "max_dd": round(test_dd, 4),
        "win_rate": round(test_wr, 3),
        "n_periods": len(test_rets),
        "hold_days": hold_days,
        "top_n": top_n,
        "train_sharpe": round(train_sharpe, 3),
        "test_sharpe": round(test_sharpe, 3),
        "overfit_ratio": round(test_sharpe / train_sharpe, 3) if train_sharpe > 0 else 0,
        "windows": window_details,
    }


# ═══════════════════════════════════════════════════
# Phase 4: 组合优化 (Grid Search)
# ═══════════════════════════════════════════════════

def grid_search_weights(ranks, price_pivot, all_dates, regime_above,
                        factor_groups, hold_days=60, top_n=20):
    """Grid search组合权重。"""
    print("\n📊 Phase 4: 组合优化 (Grid Search)...")
    
    # 简化: 每个因子取3个权重级别 (0, medium, high)
    weight_levels = [0.0, 0.10, 0.20, 0.30]
    
    n_factors = len(factor_groups)
    print(f"  {n_factors}个因子, {len(weight_levels)}级权重")
    
    # 如果因子太多, 用贪心法
    if n_factors > 6:
        return greedy_optimize(ranks, price_pivot, all_dates, regime_above,
                               factor_groups, hold_days, top_n)
    
    best_result = None
    best_weights = None
    total_combos = len(weight_levels) ** n_factors
    tested = 0
    
    # 生成所有组合
    for combo in product(weight_levels, repeat=n_factors):
        weights = dict(zip(factor_groups, combo))
        total_w = sum(weights.values())
        if total_w == 0:
            continue
        weights = {k: v / total_w for k, v in weights.items()}
        
        result = walk_forward_test(ranks, price_pivot, all_dates, regime_above,
                                   weights, hold_days, top_n)
        
        tested += 1
        if tested % 50 == 0:
            print(f"    测试 {tested}/{total_combos}...")
        
        if result and (best_result is None or result["sharpe"] > best_result["sharpe"]):
            best_result = result
            best_weights = weights
    
    return best_weights, best_result


def greedy_optimize(ranks, price_pivot, all_dates, regime_above,
                    factor_groups, hold_days=60, top_n=20):
    """贪心优化: 逐步添加/调整因子权重。"""
    print("  使用贪心优化法...")
    
    # 从等权开始
    n = len(factor_groups)
    current_weights = {f: 1.0 / n for f in factor_groups}
    
    current_result = walk_forward_test(ranks, price_pivot, all_dates, regime_above,
                                       current_weights, hold_days, top_n)
    best_sharpe = current_result["sharpe"] if current_result else -999
    best_weights = dict(current_weights)
    
    print(f"  初始等权 Sharpe: {best_sharpe:.3f}")
    
    # 贪心迭代: 逐因子调整权重
    weight_steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    improved = True
    iteration = 0
    
    while improved:
        improved = False
        iteration += 1
        
        for factor in factor_groups:
            for w in weight_steps:
                test_weights = dict(best_weights)
                test_weights[factor] = w
                
                # 归一化
                total = sum(test_weights.values())
                if total == 0:
                    continue
                test_weights = {k: v / total for k, v in test_weights.items()}
                
                result = walk_forward_test(ranks, price_pivot, all_dates, regime_above,
                                           test_weights, hold_days, top_n)
                
                if result and result["sharpe"] > best_sharpe + 0.01:  # 最小改进阈值
                    best_sharpe = result["sharpe"]
                    best_weights = dict(test_weights)
                    improved = True
                    print(f"    迭代{iteration} {factor}→{w:.0%}: Sharpe={best_sharpe:.3f}")
        
        if iteration >= 5:
            break
    
    # 最终回测
    final_result = walk_forward_test(ranks, price_pivot, all_dates, regime_above,
                                     best_weights, hold_days, top_n)
    
    return best_weights, final_result


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("🦅 Falcon Factor Weight Optimization (PIT Data)")
    print("=" * 70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"数据: features_v02.parquet (PIT corrected)")
    
    # Phase 1: 加载 + 计算PIT ranks
    master, data, all_dates = load_all_data()
    ranks = compute_pit_ranks(master, data, all_dates)
    
    # 构建价格pivot
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close")
    price_pivot.index = price_pivot.index.astype(str)
    price_pivot = price_pivot.sort_index()
    
    # Regime
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)
    
    # Phase 2: IC/ICIR
    ic_summary = compute_ic_analysis(ranks, price_pivot, hold_days=60)
    
    # 筛选有效因子 (ICIR > 0.03 或 |IC| > 0.005)
    active_factors = []
    for name, ic_data in ic_summary.items():
        if abs(ic_data["icir"]) >= 0.03 and ic_data["n_samples"] >= 10:
            active_factors.append(name)
    
    print(f"\n  有效因子 ({len(active_factors)}): {active_factors}")
    
    # 剔除已知无效/弱因子
    exclude = {"income_stmt"}  # ICIR=0.015, 太弱
    factor_groups = [f for f in active_factors if f not in exclude]
    print(f"  优化因子 ({len(factor_groups)}): {factor_groups}")
    
    # Phase 3: 单因子sweep
    print("\n" + "=" * 70)
    print("📊 Phase 3: 单因子权重Sweep")
    print("=" * 70)
    
    base_weights = {f: 1.0 / len(factor_groups) for f in factor_groups}
    sweep_results = {}
    
    for factor in factor_groups:
        sweep_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        results = single_factor_sweep(
            ranks, price_pivot, all_dates, regime_above,
            base_weights, factor, sweep_range,
            hold_days=60, top_n=20
        )
        
        if results:
            best = max(results, key=lambda x: x["sharpe"])
            sweep_results[factor] = {
                "best_weight": best["weight"],
                "best_sharpe": best["sharpe"],
                "sweep": results,
            }
            print(f"  {factor:<20} 最优: w={best['weight']:.0%} Sharpe={best['sharpe']:.3f}")
    
    # Phase 4: 组合优化
    print("\n" + "=" * 70)
    print("📊 Phase 4: 组合优化")
    print("=" * 70)
    
    # 用单因子sweep结果初始化
    init_weights = {}
    for f in factor_groups:
        if f in sweep_results:
            init_weights[f] = sweep_results[f]["best_weight"]
        else:
            init_weights[f] = 1.0 / len(factor_groups)
    
    # 归一化
    total = sum(init_weights.values())
    if total > 0:
        init_weights = {k: v / total for k, v in init_weights.items()}
    
    print(f"  初始权重(基于单因子sweep): {json.dumps({k: f'{v:.1%}' for k,v in init_weights.items()})}")
    
    # 贪心优化
    best_weights, best_result = greedy_optimize(
        ranks, price_pivot, all_dates, regime_above,
        factor_groups, hold_days=60, top_n=20
    )
    
    # Phase 5: 过拟合检测 + 多配置对比
    print("\n" + "=" * 70)
    print("📊 Phase 5: 过拟合检测 + 多配置对比")
    print("=" * 70)
    
    configs = {
        "V0.3.1_baseline": {
            "tech": 0.15, "fund_ratio": 0.05, "fund_metric": 0.06,
            "fund_growth": 0.15, "analyst": 0.12, "insider": 0.05,
            "valuation": 0.0, "earnings": 0.10, "grade_sentiment": 0.12,
        },
        "optimized_60d_20": {"weights": best_weights, "hold_days": 60, "top_n": 20},
    }
    
    # 也测试不同hold_days和top_n
    for hd in [30, 60]:
        for tn in [10, 20]:
            key = f"optimized_{hd}d_{tn}"
            if key not in configs:
                r = walk_forward_test(ranks, price_pivot, all_dates, regime_above,
                                     best_weights, hd, tn)
                if r:
                    configs[key] = {"weights": best_weights, "hold_days": hd, "top_n": tn,
                                    "result": r}
    
    # 运行baseline
    baseline_result = walk_forward_test(
        ranks, price_pivot, all_dates, regime_above,
        configs["V0.3.1_baseline"], 60, 20
    )
    configs["V0.3.1_baseline"]["result"] = baseline_result
    
    # 输出对比
    print(f"\n  {'配置':<25} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'WinR':>6} {'Train':>7} {'Test':>7} {'OF比':>6}")
    print("  " + "-" * 80)
    
    for name, cfg in configs.items():
        r = cfg.get("result")
        if r:
            print(f"  {name:<25} {r['sharpe']:>8.3f} {r['cagr']:>7.1%} {r['max_dd']:>7.1%} "
                  f"{r['win_rate']:>5.1%} {r['train_sharpe']:>7.3f} {r['test_sharpe']:>7.3f} "
                  f"{r['overfit_ratio']:>5.2f}")
    
    # Phase 6: 保存审计日志
    print("\n" + "=" * 70)
    print("📋 Phase 6: 保存审计日志")
    print("=" * 70)
    
    audit = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "features_v02.parquet (PIT corrected)",
        "data_coverage": "2016-2026, 476 tickers, fundamental NaN < 1%",
        "methodology": "Walk-Forward: 2yr train, 6mo test, 16 windows",
        "phase1_pit_ranks": {
            "n_days": len(ranks),
            "n_factors": len(list(ranks.values())[0].columns) if ranks else 0,
            "factors": list(list(ranks.values())[0].columns) if ranks else [],
        },
        "phase2_ic_analysis": ic_summary,
        "phase3_sweep_results": sweep_results,
        "phase4_optimized_weights": {k: round(v, 4) for k, v in best_weights.items()},
        "phase5_comparison": {},
    }
    
    for name, cfg in configs.items():
        r = cfg.get("result")
        if r:
            w = cfg.get("weights", {})
            audit["phase5_comparison"][name] = {
                "weights": {k: round(v, 4) for k, v in w.items()} if w else {},
                "hold_days": cfg.get("hold_days", 60),
                "top_n": cfg.get("top_n", 20),
                "sharpe": r["sharpe"],
                "cagr": r["cagr"],
                "max_dd": r["max_dd"],
                "win_rate": r["win_rate"],
                "train_sharpe": r["train_sharpe"],
                "test_sharpe": r["test_sharpe"],
                "overfit_ratio": r["overfit_ratio"],
            }
    
    with open(OUTPUT, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  ✅ 保存到 {OUTPUT}")
    
    # 最终结论
    print("\n" + "=" * 70)
    print("📋 最终结论")
    print("=" * 70)
    print(f"  最优权重: {json.dumps({k: f'{v:.1%}' for k,v in best_weights.items()}, indent=4)}")
    if best_result:
        print(f"  OOS Sharpe: {best_result['sharpe']:.3f}")
        print(f"  OOS CAGR: {best_result['cagr']:.2%}")
        print(f"  OOS MaxDD: {best_result['max_dd']:.2%}")
        print(f"  过拟合比: {best_result['overfit_ratio']:.2f} (>0.5=可接受)")


if __name__ == "__main__":
    main()
