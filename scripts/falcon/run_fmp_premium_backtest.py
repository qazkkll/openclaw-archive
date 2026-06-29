#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — FMP Premium因子增量回测 (优化版)
对比有无 earnings + grade_sentiment 因子组的效果差异
Fixed 63天调仓, Top 20, 生产配置权重
"""
import sys, time, json
from pathlib import Path

PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))

import pandas as pd, numpy as np
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = PROJECT_ROOT / "data" / "falcon"
FMP_PREMIUM_DIR = PROJECT_ROOT / "data" / "fmp_premium"


def load_all_data():
    """加载所有数据源。"""
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    data = {}
    for name in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
                  "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            with open(f) as fh:
                data[name] = json.load(fh)
            print(f"  ✅ {name}: {len(data[name])} 只")
        else:
            data[name] = {}
            print(f"  ❌ {name}: 不存在")
    return master, data


def main():
    t0 = time.time()
    print("=" * 110)
    print("🦅 Falcon V0.3 — FMP Premium因子增量回测 (Fixed 63d, Top 20)")
    print("=" * 110)

    # ── 1. 加载数据 ──
    print("\n📂 Step 1: 加载数据...")
    master, data = load_all_data()

    n_tickers = master["ticker"].nunique()
    print(f"\n📊 Master: {len(master)} 行, {n_tickers} 只")
    
    # 限制: 取日期覆盖最多的200只，且只用2023-2024年数据 (回测效率)
    tick_counts = master.groupby("ticker")["date"].nunique().sort_values(ascending=False)
    top200 = tick_counts.head(200).index.tolist()
    master = master[master["ticker"].isin(top200)]
    # 限制日期范围
    master = master[master["date"] >= "2023-01-01"]
    print(f"⚠️ 优化: 200只 × 2023-2024 ({len(master)} 行, {master['ticker'].nunique()} 只)")

    # 加载FMP Premium数据
    print("\n📂 Step 1b: 加载FMP Premium数据...")
    earnings_hist = load_fmp_premium_earnings(FMP_PREMIUM_DIR)
    grades_hist = load_fmp_premium_grades(FMP_PREMIUM_DIR)

    universe_tickers = set(master["ticker"].unique())
    earnings_overlap = universe_tickers & set(earnings_hist.keys())
    grades_overlap = universe_tickers & set(grades_hist.keys())
    print(f"\n📊 FMP Premium覆盖率:")
    print(f"  earnings: {len(earnings_overlap)}/{len(universe_tickers)} ({100*len(earnings_overlap)/len(universe_tickers):.0f}%)")
    print(f"  grades:   {len(grades_overlap)}/{len(universe_tickers)} ({100*len(grades_overlap)/len(universe_tickers):.0f}%)")

    # ── 2. 预计算PIT rank (无premium) ──
    print("\n📊 Step 2: 预计算PIT rank (基准: 7因子组)...")
    t1 = time.time()
    ranks_base = precompute_pit_ranks(
        master, data["fmp_ratios_historical"], data["analyst_historical"],
        data["fmp_key_metrics"], data["fmp_financial_growth"],
        data["fmp_insider"], data["fmp_dcf"], data["fmp_price_target"]
    )
    print(f"  ⏱️ {time.time()-t1:.1f}秒, {len(ranks_base)} 天")

    # ── 3. 预计算PIT rank (有premium) ──
    print("\n📊 Step 3: 预计算PIT rank (全量: 9因子组)...")
    t1 = time.time()
    ranks_full = precompute_pit_ranks(
        master, data["fmp_ratios_historical"], data["analyst_historical"],
        data["fmp_key_metrics"], data["fmp_financial_growth"],
        data["fmp_insider"], data["fmp_dcf"], data["fmp_price_target"],
        earnings_hist=earnings_hist, grades_hist=grades_hist
    )
    print(f"  ⏱️ {time.time()-t1:.1f}秒, {len(ranks_full)} 天")

    # ── 4. 新因子rank分布 ──
    print("\n📊 Step 4: 新因子rank分布检查...")
    sample_dates = sorted(ranks_full.keys())[:5]
    for d in sample_dates:
        r = ranks_full[d]
        for col in ["earnings", "grade_sentiment"]:
            if col in r.columns:
                v = r[col]
                print(f"  {d} {col:20} mean={v.mean():.3f} std={v.std():.3f} min={v.min():.3f} max={v.max():.3f} non-0.5={((v > 0.01) & (v < 0.99)).sum()}/{len(v)}")

    # ── 5. 价格矩阵 + regime ──
    print("\n📊 Step 5: 构建价格矩阵...")
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    all_dates = sorted(ranks_base.keys())
    split_idx = int(len(all_dates) * 0.7)
    is_dates = all_dates[:split_idx]
    oos_dates = all_dates[split_idx:]
    print(f"  IS: {is_dates[0]} → {is_dates[-1]} ({len(is_dates)} 天)")
    print(f"  OOS: {oos_dates[0]} → {oos_dates[-1]} ({len(oos_dates)} 天)")

    # ── 6. 权重配置 ──
    weight_configs = {
        "baseline_70_20_10": {
            "tech": 0.0, "fund_ratio": 0.70, "fund_metric": 0.10,
            "fund_growth": 0.0, "analyst": 0.20, "insider": 0.0, "valuation": 0.0,
        },
        "earnings_0_grades_0": {
            "tech": 0.0, "fund_ratio": 0.70, "fund_metric": 0.10,
            "fund_growth": 0.0, "analyst": 0.20, "insider": 0.0, "valuation": 0.0,
            "earnings": 0.0, "grade_sentiment": 0.0,
        },
        "earnings_5_grades_5": {
            "tech": 0.0, "fund_ratio": 0.60, "fund_metric": 0.10,
            "fund_growth": 0.0, "analyst": 0.15, "insider": 0.0, "valuation": 0.0,
            "earnings": 0.05, "grade_sentiment": 0.05,
        },
        "earnings_10_grades_10": {
            "tech": 0.0, "fund_ratio": 0.55, "fund_metric": 0.10,
            "fund_growth": 0.0, "analyst": 0.15, "insider": 0.0, "valuation": 0.0,
            "earnings": 0.10, "grade_sentiment": 0.10,
        },
        "earnings_10_grades_5": {
            "tech": 0.0, "fund_ratio": 0.60, "fund_metric": 0.10,
            "fund_growth": 0.0, "analyst": 0.15, "insider": 0.0, "valuation": 0.0,
            "earnings": 0.10, "grade_sentiment": 0.05,
        },
        "earnings_5_grades_10": {
            "tech": 0.0, "fund_ratio": 0.60, "fund_metric": 0.10,
            "fund_growth": 0.0, "analyst": 0.10, "insider": 0.0, "valuation": 0.0,
            "earnings": 0.05, "grade_sentiment": 0.10,
        },
    }

    # ── 7. 运行回测 ──
    print("\n" + "=" * 110)
    print("📊 Step 7: 回测对比")
    print("=" * 110)

    params = {"hold_days": 63, "stop_loss": -0.15, "bear_alloc": 0.50}

    results_table = []
    for cfg_name, weights in weight_configs.items():
        # IS
        is_res = backtest_flexible(ranks_base, price_pivot, is_dates, regime_above, weights, "fixed", params, 20) if cfg_name == "baseline_70_20_10" else None
        if is_res is None:
            is_res = backtest_flexible(ranks_full, price_pivot, is_dates, regime_above, weights, "fixed", params, 20)
        # OOS
        oos_res = backtest_flexible(ranks_base, price_pivot, oos_dates, regime_above, weights, "fixed", params, 20) if cfg_name == "baseline_70_20_10" else None
        if oos_res is None:
            oos_res = backtest_flexible(ranks_full, price_pivot, oos_dates, regime_above, weights, "fixed", params, 20)
        
        results_table.append((cfg_name, is_res, oos_res))

    # 打印结果表
    print(f"\n{'配置':30} | {'IS Sharpe':>10} {'IS DD':>8} {'IS WR':>7} {'IS Tr':>5} | {'OOS Sharpe':>10} {'OOS DD':>8} {'OOS WR':>7} {'OOS Tr':>5}")
    print("-" * 115)

    for name, is_res, oos_res in results_table:
        is_s = f"{is_res['sharpe']:10.3f} {is_res['dd']:7.1f}% {is_res['wr']:6.0f}% {is_res['trades']:5}" if is_res else "       N/A"
        oos_s = f"{oos_res['sharpe']:10.3f} {oos_res['dd']:7.1f}% {oos_res['wr']:6.0f}% {oos_res['trades']:5}" if oos_res else "       N/A"
        print(f"  {name:28} | {is_s} | {oos_s}")

    # 增量分析
    base_is = results_table[0][1]  # baseline IS
    base_oos = results_table[0][2]  # baseline OOS
    full_10_10_is = results_table[3][1]  # earnings_10_grades_10 IS
    full_10_10_oos = results_table[3][2]  # earnings_10_grades_10 OOS

    if base_is and full_10_10_is:
        print(f"\n📊 增量 delta (基准 → earnings 10% + grades 10%):")
        print(f"  IS:  Sharpe {base_is['sharpe']:.3f} → {full_10_10_is['sharpe']:.3f} (Δ={full_10_10_is['sharpe']-base_is['sharpe']:+.3f})")
        print(f"       DD     {base_is['dd']:.1f}% → {full_10_10_is['dd']:.1f}% (Δ={full_10_10_is['dd']-base_is['dd']:+.1f}%)")
        print(f"       WR     {base_is['wr']:.0f}% → {full_10_10_is['wr']:.0f}% (Δ={full_10_10_is['wr']-base_is['wr']:+.0f}%)")
        print(f"       Trades {base_is['trades']} → {full_10_10_is['trades']}")
    if base_oos and full_10_10_oos:
        print(f"\n  OOS: Sharpe {base_oos['sharpe']:.3f} → {full_10_10_oos['sharpe']:.3f} (Δ={full_10_10_oos['sharpe']-base_oos['sharpe']:+.3f})")
        print(f"       DD     {base_oos['dd']:.1f}% → {full_10_10_oos['dd']:.1f}% (Δ={full_10_10_oos['dd']-base_oos['dd']:+.1f}%)")
        print(f"       WR     {base_oos['wr']:.0f}% → {full_10_10_oos['wr']:.0f}% (Δ={full_10_10_oos['wr']-base_oos['wr']:+.0f}%)")
        print(f"       Trades {base_oos['trades']} → {full_10_10_oos['trades']}")

    print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
