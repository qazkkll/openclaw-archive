#!/usr/bin/env python3
"""
🦅 Falcon V0.3 vs V0.3.1 对比回测
===================================
在相同数据上对比两个版本的性能差异。
使用Walk-Forward方法论（OOS评估）。
"""
import sys, json, time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from falcon_v03_engine import (
    precompute_pit_ranks_fast, backtest_flexible,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "falcon"
FMP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fmp_premium"


# ═══════════════════════════════════════════════════
# 版本配置
# ═══════════════════════════════════════════════════

VERSIONS = {
    "V0.3": {
        "weights": {"fund_ratio": 0.56, "analyst": 0.16, "fund_metric": 0.08, "earnings": 0.20, "grade_sentiment": 0.0},
        "hold_days": 30,
        "top_n": 5,
    },
    "V0.3.1": {
        "weights": {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0},
        "hold_days": 90,
        "top_n": 10,
    },
}


def walk_forward_windows(dates, train_years=2, test_months=6):
    """生成walk-forward滚动窗口。"""
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

        train_dates = [d for d in dates if train_start.strftime("%Y-%m-%d") <= d < train_end.strftime("%Y-%m-%d")]
        test_dates = [d for d in dates if train_end.strftime("%Y-%m-%d") <= d < test_end.strftime("%Y-%m-%d")]
        if len(train_dates) >= 200 and len(test_dates) >= 50:
            windows.append((train_dates, test_dates))
        train_start = train_start + relativedelta(months=test_months)

    return windows


def run_backtest(ranks, price_pivot, dates, regime_above, weights, hold_days, top_n):
    """安全运行backtest。"""
    try:
        result = backtest_flexible(
            ranks, price_pivot, dates, regime_above,
            weights=weights,
            strategy="fixed",
            params={"hold_days": hold_days, "cost": 0.001, "stop_loss": -0.15},
            top_n=top_n,
        )
        if result is None or result.get("sharpe") is None:
            return None
        return result
    except Exception:
        return None


def main():
    t_start = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.3 vs V0.3.1 对比回测 (Walk-Forward)")
    print("=" * 80)

    # ── 加载数据 ──
    print("\n📂 加载数据...")
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)

    data = {}
    for name in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
                  "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            data[name] = json.load(open(f))

    earnings_all = load_fmp_premium_earnings(str(FMP_DIR))
    grades_all = load_fmp_premium_grades(str(FMP_DIR))
    data["earnings"] = earnings_all
    data["grades"] = grades_all

    all_dates = sorted(master["date"].unique())
    n_tickers = master["ticker"].nunique()
    print(f"  ✅ {len(master):,}行, {n_tickers}只, {len(all_dates)}天")

    # ── PIT rank (全局一次) ──
    print("\n📊 计算PIT rank (bisect加速)...")
    t_pit = time.time()
    ranks = precompute_pit_ranks_fast(
        master,
        data.get("fmp_ratios_historical", {}),
        data.get("analyst_historical", {}),
        data.get("fmp_key_metrics", {}),
        data.get("fmp_financial_growth", {}),
        data.get("fmp_insider", {}),
        data.get("fmp_dcf", {}),
        data.get("fmp_price_target", {}),
        earnings_hist=earnings_all,
        grades_hist=grades_all,
    )
    print(f"  ✅ PIT rank: {len(ranks)}天, {time.time()-t_pit:.0f}秒")

    # ── 价格矩阵 + regime ──
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    # ── Walk-Forward窗口 ──
    windows = walk_forward_windows(all_dates, train_years=2, test_months=6)
    print(f"\n📊 Walk-Forward: {len(windows)}个窗口")

    # ── 对比回测 ──
    results = {v: {"oos_sharpes": [], "oos_dds": [], "oos_wrs": [], "oos_rets": []} for v in VERSIONS}

    for i, (train_dates, test_dates) in enumerate(windows):
        print(f"\n  窗口 {i+1}/{len(windows)}: {test_dates[0]}~{test_dates[-1]}")

        for ver_name, ver_cfg in VERSIONS.items():
            oos = run_backtest(
                ranks, price_pivot, test_dates, regime_above,
                ver_cfg["weights"], ver_cfg["hold_days"], ver_cfg["top_n"],
            )
            if oos:
                results[ver_name]["oos_sharpes"].append(oos["sharpe"])
                results[ver_name]["oos_dds"].append(oos.get("dd", 0) / 100)  # dd是百分比
                results[ver_name]["oos_wrs"].append(oos.get("wr", 0) / 100)  # wr是百分比
                results[ver_name]["oos_rets"].append(oos.get("ret", 0) / 100)  # ret是百分比
                print(f"    {ver_name}: Sharpe={oos['sharpe']:.3f} DD={oos.get('dd',0):.1f}% WR={oos.get('wr',0):.1f}% Ret={oos.get('ret',0):.1f}% Trades={oos.get('trades',0)}")

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("📊 V0.3 vs V0.3.1 对比汇总")
    print("=" * 80)

    for ver_name in VERSIONS:
        r = results[ver_name]
        cfg = VERSIONS[ver_name]
        sharpes = r["oos_sharpes"]
        dds = r["oos_dds"]
        wrs = r["oos_wrs"]
        rets = r["oos_rets"]

        print(f"\n{'─'*50}")
        print(f"🏷️  {ver_name}")
        print(f"{'─'*50}")
        print(f"  权重: fund={cfg['weights']['fund_ratio']:.2f} ana={cfg['weights']['analyst']:.2f} met={cfg['weights']['fund_metric']:.2f} earn={cfg['weights']['earnings']:.2f}")
        print(f"  调仓: {cfg['hold_days']}天 | Top-N: {cfg['top_n']}")
        print(f"  窗口数: {len(sharpes)}")
        if sharpes:
            print(f"  OOS Sharpe: mean={np.mean(sharpes):.3f} median={np.median(sharpes):.3f} std={np.std(sharpes):.3f}")
            print(f"  OOS DD:     mean={np.mean(dds)*100:.1f}%")
            print(f"  OOS WR:     mean={np.mean(wrs)*100:.1f}%")
            print(f"  OOS Ret:    mean={np.mean(rets)*100:.1f}%")
            positive = sum(1 for s in sharpes if s > 0)
            print(f"  正Sharpe窗口: {positive}/{len(sharpes)} ({positive/len(sharpes)*100:.0f}%)")

    # ── 提升幅度 ──
    if results["V0.3"]["oos_sharpes"] and results["V0.3.1"]["oos_sharpes"]:
        v03_sharpe = np.mean(results["V0.3"]["oos_sharpes"])
        v031_sharpe = np.mean(results["V0.3.1"]["oos_sharpes"])
        v03_dd = np.mean(results["V0.3"]["oos_dds"])
        v031_dd = np.mean(results["V0.3.1"]["oos_dds"])
        v03_wr = np.mean(results["V0.3"]["oos_wrs"])
        v031_wr = np.mean(results["V0.3.1"]["oos_wrs"])
        v03_ret = np.mean(results["V0.3"]["oos_rets"])
        v031_ret = np.mean(results["V0.3.1"]["oos_rets"])

        print(f"\n{'='*50}")
        print(f"📈 V0.3.1 vs V0.3 提升幅度")
        print(f"{'='*50}")
        sharpe_chg = (v031_sharpe/v03_sharpe - 1)*100 if v03_sharpe != 0 else float('inf')
        dd_chg = (v03_dd - v031_dd)/v03_dd*100 if v03_dd != 0 else 0
        wr_chg = (v031_wr/v03_wr - 1)*100 if v03_wr != 0 else 0
        ret_chg = (v031_ret/v03_ret - 1)*100 if v03_ret != 0 else 0

        print(f"  OOS Sharpe: {v03_sharpe:.3f} → {v031_sharpe:.3f} ({sharpe_chg:+.1f}%)")
        print(f"  OOS DD:     {v03_dd*100:.1f}% → {v031_dd*100:.1f}% ({dd_chg:+.1f}%{'改善' if v031_dd < v03_dd else '恶化'})")
        print(f"  OOS WR:     {v03_wr*100:.1f}% → {v031_wr*100:.1f}% ({wr_chg:+.1f}%)")
        print(f"  OOS Ret:    {v03_ret*100:.1f}% → {v031_ret*100:.1f}% ({ret_chg:+.1f}%)")

    print(f"\n⏱️ 总耗时: {(time.time()-t_start)/60:.1f}分钟")

    # 保存结果
    out = {}
    for ver_name in VERSIONS:
        r = results[ver_name]
        out[ver_name] = {
            "oos_sharpe_mean": float(np.mean(r["oos_sharpes"])) if r["oos_sharpes"] else None,
            "oos_sharpe_median": float(np.median(r["oos_sharpes"])) if r["oos_sharpes"] else None,
            "oos_dd_mean": float(np.mean(r["oos_dds"])) if r["oos_dds"] else None,
            "oos_wr_mean": float(np.mean(r["oos_wrs"])) if r["oos_wrs"] else None,
            "oos_ret_mean": float(np.mean(r["oos_rets"])) if r["oos_rets"] else None,
            "n_windows": len(r["oos_sharpes"]),
        }
    with open(DATA_DIR / "v03_vs_v031_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 结果已保存: {DATA_DIR / 'v03_vs_v031_comparison.json'}")


if __name__ == "__main__":
    main()
