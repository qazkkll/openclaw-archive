#!/usr/bin/env python3
"""
🦅 Falcon V0.3.2 Hybrid Validation
=====================================
在V0.3.1基础上增量添加新因子，对比多种权重方案。

测试方案:
  A. V0.3.1 (旧): fund_ratio=70, analyst=20, fund_metric=10
  B. V0.3.2_full (全数据驱动): fund_growth=26.7, cashflow=13.9, ...
  C. V0.3.2_hybrid (混合): V0.3.1核心 + 新因子增量
  D. V0.3.2_hybrid_60d: 同C但60天调仓
"""
import sys, json, time, warnings
from pathlib import Path
import pandas as pd
import numpy as np

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

INVERT_FACTORS = {"debt_to_equity", "net_debt_to_assets", "capex_intensity"}


def load_data():
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


def compute_all_ranks(master, data, all_dates):
    """计算旧因子 + 新因子 PIT rank。"""
    print("\n📊 计算PIT rank...")
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
        earnings_hist=data.get("earnings", {}),
        grades_hist=data.get("grades", {}),
    )
    print(f"  旧因子: {len(ranks)}天, {time.time()-t0:.0f}秒")

    # 新因子
    t0 = time.time()
    income_raw = data.get("fmp_income_stmt", {})
    balance_raw = data.get("fmp_balance_sheet", {})
    cashflow_raw = data.get("fmp_cashflow", {})
    income_idx = build_pit_index_statements(income_raw, use_filing_date=True)
    balance_idx = build_pit_index_statements(balance_raw, use_filing_date=False)
    cashflow_idx = build_pit_index_statements(cashflow_raw, use_filing_date=False)

    new_factor_names = BALANCE_FIELDS + CASHFLOW_FIELDS + INCOME_FIELDS
    for date in sorted(ranks.keys()):
        rank_df = ranks[date]
        tickers = rank_df.index.tolist()
        new_data = {}
        for t in tickers:
            factors = compute_statement_factors(t, date, balance_idx, cashflow_idx, income_idx, {})
            if factors:
                new_data[t] = factors
        if new_data:
            new_df = pd.DataFrame.from_dict(new_data, orient="index")
            valid_cols = [c for c in new_df.columns if new_df[c].notna().sum() >= 10]
            if valid_cols:
                ranked_new = new_df[valid_cols].rank(pct=True)
                for col in valid_cols:
                    if col in INVERT_FACTORS:
                        ranked_new[col] = 1 - ranked_new[col]
                    rank_df[col] = ranked_new[col]

        # 组级分数
        for group_name, fields in [("balance", BALANCE_FIELDS),
                                    ("cashflow", CASHFLOW_FIELDS),
                                    ("income_stmt", INCOME_FIELDS)]:
            cols = [c for c in fields if c in rank_df.columns]
            if cols:
                rank_df[group_name] = rank_df[cols].mean(axis=1)

    print(f"  新因子: {time.time()-t0:.0f}秒")
    return ranks


def walk_forward_windows(dates, train_years=2, test_months=6):
    from dateutil.relativedelta import relativedelta
    windows = []
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        test_dates = [d for d in dates
                      if train_end.strftime("%Y-%m-%d") <= d < test_end.strftime("%Y-%m-%d")]
        if len(test_dates) >= 50:
            windows.append(test_dates)
        train_start = train_start + relativedelta(months=test_months)
    return windows


def safe_backtest(ranks, price_pivot, dates, regime_above, weights, hold_days, top_n):
    try:
        result = backtest_flexible(
            ranks, price_pivot, dates, regime_above,
            weights=weights, strategy="fixed",
            params={"hold_days": hold_days, "cost": 0.001, "stop_loss": -0.15},
            top_n=top_n,
        )
        if result and result.get("sharpe") is not None:
            return result
    except Exception:
        pass
    return None


def get_regime(price_pivot):
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    return (mkt_price > mkt_ma200).astype(int)


def main():
    print("🦅 Falcon V0.3.2 Hybrid Validation")
    print("=" * 80)
    t_total = time.time()

    master, data, all_dates = load_data()
    ranks = compute_all_ranks(master, data, all_dates)

    # Verify
    sample_date = list(ranks.keys())[500]
    sample_df = ranks[sample_date]
    print(f"\n🔍 Ranks: {sample_df.shape}, columns={list(sample_df.columns)}")

    price_pivot = master.pivot(index="date", columns="ticker", values="close")
    price_pivot.index = price_pivot.index.astype(str)
    regime_above = get_regime(price_pivot)
    windows = walk_forward_windows(all_dates)
    print(f"📅 Walk-Forward: {len(windows)} 窗口")

    # ═══════════════════════════════════════════
    # 权重方案
    # ═══════════════════════════════════════════

    # A. V0.3.1 (旧)
    w_v031 = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10,
              "earnings": 0.0, "grade_sentiment": 0.0}

    # B. V0.3.2 full (全ICIR驱动)
    w_full = {"fund_growth": 0.267, "cashflow": 0.139, "grade_sentiment": 0.130,
              "analyst": 0.121, "earnings": 0.117, "balance": 0.086,
              "fund_metric": 0.075, "insider": 0.052, "income_stmt": 0.013}

    # C. V0.3.2 hybrid (V0.3.1核心 + 新因子增量)
    # 保留V0.3.1的稳定性骨架(fund_ratio降到45%), 加入ICIR>0.1的新因子
    w_hybrid = {
        "fund_ratio": 0.40,       # 旧核心 (降权, IC负但提供稳定性)
        "fund_growth": 0.10,      # 新发现的强因子 (控权防过拟合)
        "analyst": 0.12,          # 旧, ICIR=0.133
        "fund_metric": 0.06,      # 旧, ICIR=0.082
        "earnings": 0.08,         # 旧, ICIR=0.129
        "grade_sentiment": 0.08,  # 旧, ICIR=0.143
        "cashflow": 0.08,         # 新, ICIR=0.153
        "balance": 0.04,          # 新, ICIR=0.094
        "insider": 0.02,          # 旧, ICIR=0.057
        "valuation": 0.0,         # IC负, 去掉
        "tech": 0.0,              # IC负, 去掉
        "income_stmt": 0.0,       # ICIR太低(0.015), 去掉
    }

    # D. V0.3.2 hybrid 去掉fund_ratio
    w_hybrid_nfr = {
        "fund_growth": 0.20,
        "analyst": 0.15,
        "fund_metric": 0.08,
        "earnings": 0.10,
        "grade_sentiment": 0.12,
        "cashflow": 0.15,
        "balance": 0.08,
        "insider": 0.05,
        "valuation": 0.0,
        "tech": 0.0,
        "income_stmt": 0.0,
        "fund_ratio": 0.07,       # 最小保留
    }

    configs = {
        "A. V0.3.1 (旧)": {"weights": w_v031, "hold": 30, "top": 10},
        "B. V0.3.2_full": {"weights": w_full, "hold": 30, "top": 10},
        "C. V0.3.2_hybrid": {"weights": w_hybrid, "hold": 30, "top": 10},
        "D. V0.3.2_hybrid_nfr": {"weights": w_hybrid_nfr, "hold": 30, "top": 10},
        "E. V0.3.2_hybrid_60d": {"weights": w_hybrid, "hold": 60, "top": 10},
        "F. V0.3.2_full_60d": {"weights": w_full, "hold": 60, "top": 10},
    }

    # ═══════════════════════════════════════════
    # Walk-Forward全量对比
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 Walk-Forward对比回测 (所有方案)")
    print("=" * 80)

    all_results = {}
    for name, cfg in configs.items():
        sharpes, dds, rets, wrs = [], [], [], []
        for test_dates in windows:
            r = safe_backtest(ranks, price_pivot, test_dates, regime_above,
                            cfg["weights"], cfg["hold"], cfg["top"])
            if r:
                sharpes.append(r["sharpe"])
                dds.append(r["dd"])
                rets.append(r["ret"])
                wrs.append(r["wr"])

        if sharpes:
            pos_rate = np.mean([1 for s in sharpes if s > 0]) * 100
            result = {
                "sharpe": np.mean(sharpes), "dd": np.mean(dds),
                "ret": np.mean(rets), "wr": np.mean(wrs),
                "positive_rate": pos_rate, "n_windows": len(sharpes),
                "sharpe_std": np.std(sharpes),
            }
            all_results[name] = result
            print(f"\n  {name}:")
            print(f"    OOS Sharpe: {result['sharpe']:.3f} ± {result['sharpe_std']:.3f} (正率={pos_rate:.0f}%)")
            print(f"    OOS DD:     {result['dd']:.1f}%")
            print(f"    OOS Ret:    {result['ret']:.1f}%")
            print(f"    OOS WR:     {result['wr']:.1f}%")

    # ═══════════════════════════════════════════
    # 近期窗口详细对比 (2024+)
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 近期窗口详细对比 (2024+)")
    print("=" * 80)

    recent = [w for w in windows if w[0] >= "2024-01-01"]
    for test_dates in recent:
        label = f"{test_dates[0]}~{test_dates[-1]}"
        print(f"\n  {label}:")
        for name, cfg in configs.items():
            r = safe_backtest(ranks, price_pivot, test_dates, regime_above,
                            cfg["weights"], cfg["hold"], cfg["top"])
            s = r["sharpe"] if r else None
            s_str = f"{s:.3f}" if s is not None else "N/A"
            print(f"    {name}: Sharpe={s_str}")

    # ═══════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════

    elapsed = time.time() - t_total
    print("\n" + "=" * 80)
    print("📋 最终对比总结")
    print("=" * 80)

    print(f"\n{'方案':<30} {'Sharpe':>8} {'DD':>8} {'Ret':>8} {'WR':>8} {'正率':>6}")
    print("-" * 70)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]["sharpe"]):
        print(f"{name:<30} {r['sharpe']:>8.3f} {r['dd']:>7.1f}% {r['ret']:>7.1f}% {r['wr']:>7.1f}% {r['positive_rate']:>5.0f}%")

    # 找最佳方案
    best = max(all_results.items(), key=lambda x: x[1]["sharpe"])
    baseline = all_results.get("A. V0.3.1 (旧)")
    if baseline:
        delta = best[1]["sharpe"] - baseline["sharpe"]
        pct = (delta / abs(baseline["sharpe"]) * 100) if baseline["sharpe"] != 0 else 0
        print(f"\n🏆 最佳方案: {best[0]}")
        print(f"   Sharpe: {best[1]['sharpe']:.3f} (vs V0.3.1: {delta:+.3f}, {pct:+.1f}%)")
        print(f"   DD: {best[1]['dd']:.1f}% (vs V0.3.1: {best[1]['dd']-baseline['dd']:+.1f}%)")

    print(f"\n  总耗时: {elapsed/60:.1f}分钟")

    # Save
    output = {
        "results": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                        for kk, vv in v.items()}
                   for k, v in all_results.items()},
        "best": best[0] if best else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(DATA_DIR / "v032_hybrid_validation.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  结果已保存: {DATA_DIR / 'v032_hybrid_validation.json'}")


if __name__ == "__main__":
    main()
