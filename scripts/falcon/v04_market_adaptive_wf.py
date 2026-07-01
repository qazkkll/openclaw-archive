#!/usr/bin/env python3
"""
🦅 Falcon V0.4 — 市场状态自适应 + 更长训练窗口 Walk-Forward (FMP PIT版)
================================================================
使用FMP JSON文件 + PIT索引计算rank，与V0.3.1相同的数据源。
然后用backtest_engine.py回测。

方案:
  A: V0.3.1 baseline (2yr train, fund_ratio=0.70, analyst=0.20, fund_metric=0.10)
  B: V0.3.1 optim (2yr train, fund_ratio=0.75, analyst=0.20, fund_metric=0.05)
  C: 市场状态自适应 (2yr train, VIX-based dynamic weights)
  D: V0.3.1 baseline (5yr train)
  E: V0.3.1 optim (5yr train)
  F: 市场状态自适应 (5yr train)
  G: V0.3.1 baseline (8yr train)
  H: V0.3.1 optim (8yr train)
  I: 市场状态自适应 (8yr train)

Walk-Forward参数:
  - test_months=6
  - hold_days=30, top_n=10
  - cost=0.001, stop_loss=-0.15

红线: 必须用backtest_engine.py回测
"""
import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError
from falcon_v03_engine import (
    precompute_pit_ranks_fast,
    RATIO_FIELDS, ANALYST_FIELDS, METRIC_FIELDS,
    GROWTH_FIELDS,
)

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
OUTPUT_PATH = DATA_DIR / "v04_market_adaptive_results.json"

# ═══════════════════════════════════════════════════
#  市场状态
# ═══════════════════════════════════════════════════
REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_RANGE = "range"

WEIGHTS_REGIME = {
    REGIME_BULL: {"fund_ratio": 0.80, "analyst": 0.15, "fund_metric": 0.05},
    REGIME_BEAR: {"fund_ratio": 0.60, "analyst": 0.30, "fund_metric": 0.10},
    REGIME_RANGE: {"fund_ratio": 0.75, "analyst": 0.20, "fund_metric": 0.05},
}
WEIGHTS_V031_BASE = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10}
WEIGHTS_V031_OPTIM = {"fund_ratio": 0.75, "analyst": 0.20, "fund_metric": 0.05}


def load_all_data():
    """加载FMP JSON数据。"""
    print("📂 加载FMP数据...")
    files = {
        "fmp_ratios_historical": DATA_DIR / "fmp_ratios_historical.json",
        "analyst_historical": DATA_DIR / "analyst_historical.json",
        "fmp_key_metrics": DATA_DIR / "fmp_key_metrics.json",
        "fmp_financial_growth": DATA_DIR / "fmp_financial_growth.json",
        "fmp_insider": DATA_DIR / "fmp_insider.json",
        "fmp_dcf": DATA_DIR / "fmp_dcf.json",
        "fmp_price_target": DATA_DIR / "fmp_price_target.json",
    }
    data = {}
    for name, path in files.items():
        if path.exists():
            with open(path) as f:
                data[name] = json.load(f)
            print(f"  ✅ {name}: {len(data[name])} tickers")
        else:
            print(f"  ⚠️ {name}: NOT FOUND")
            data[name] = {}
    return data


def build_price_pivot(master):
    """从master构建价格矩阵。"""
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    print(f"  ✅ Price matrix: {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")
    return price_pivot


def compute_regime(vix_path):
    """计算市场状态。"""
    vix = pd.read_parquet(vix_path)
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date").sort_index()

    regime = pd.Series(index=vix.index, dtype=str)
    for d in vix.index:
        v = vix.loc[d, "close"]
        if pd.isna(v):
            regime[d] = REGIME_RANGE
        elif v < 20:
            regime[d] = REGIME_BULL
        elif v > 25:
            regime[d] = REGIME_BEAR
        else:
            regime[d] = REGIME_RANGE

    counts = regime.value_counts()
    print(f"  ✅ 市场状态: {dict(counts)}")
    return regime


# ═══════════════════════════════════════════════════
#  Walk-Forward
# ═══════════════════════════════════════════════════
def wf_fixed(ranks, price_pivot, weights, train_years, test_months=6,
             hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15, label=""):
    """固定权重 Walk-Forward。"""
    print(f"\n{'='*60}")
    print(f"📊 WF: {label} (train={train_years}yr)")
    print(f"{'='*60}")

    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    dates = sorted(ranks.keys())
    if not dates:
        return {"label": label, "error": "No dates", "windows": []}

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    windows = []
    idx = 0

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
        tss = str(train_end)[:10]
        tes = str(test_end)[:10]
        try:
            result, baseline = engine.run(
                ranks, price_pivot, weights, hold_days, top_n,
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

    return _agg(label, windows)


def wf_regime(ranks, price_pivot, regime, train_years, test_months=6,
              hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15, label=""):
    """市场状态自适应 Walk-Forward。"""
    print(f"\n{'='*60}")
    print(f"📊 WF: {label} (train={train_years}yr, regime-adaptive)")
    print(f"{'='*60}")

    dates = sorted(ranks.keys())
    if not dates:
        return {"label": label, "error": "No dates", "windows": []}

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    windows = []
    idx = 0

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
        tss = str(train_end)[:10]
        tes = str(test_end)[:10]
        test_dates = [d for d in dates if tss <= d <= tes]
        if len(test_dates) < 20:
            windows.append({"index": idx, "period": f"{tss} → {tes}", "error": f"Too few: {len(test_dates)}"})
            idx += 1
            train_start += pd.DateOffset(months=test_months)
            continue

        # Regime distribution in test window
        regime_days = {}
        for d in test_dates:
            d_ts = pd.Timestamp(d)
            if d_ts in regime.index:
                r = regime[d_ts]
            else:
                mask = regime.index <= d_ts
                r = regime.loc[mask].iloc[-1] if mask.any() else REGIME_RANGE
            regime_days[r] = regime_days.get(r, 0) + 1

        total = sum(regime_days.values())
        combined = {"fund_ratio": 0.0, "analyst": 0.0, "fund_metric": 0.0}
        for r, cnt in regime_days.items():
            w = cnt / total
            for k in combined:
                combined[k] += w * WEIGHTS_REGIME[r][k]

        try:
            engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
            result, baseline = engine.run(
                ranks, price_pivot, combined, hold_days, top_n,
                start_date=tss, end_date=tes, run_baseline=True
            )
            windows.append({
                "index": idx, "period": f"{tss} → {tes}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades, "n_days": len(result.daily_equity),
                "baseline_sharpe": baseline.sharpe if baseline else None,
                "regime_dist": {r: round(c/total, 3) for r, c in regime_days.items()},
                "eff_weights": {k: round(v, 3) for k, v in combined.items()},
            })
        except (DataQualityError, Exception) as e:
            windows.append({"index": idx, "period": f"{tss} → {tes}", "error": str(e)[:200]})

        idx += 1
        train_start += pd.DateOffset(months=test_months)

    return _agg(label, windows)


def _agg(label, windows):
    if not windows:
        return {"label": label, "error": "No windows", "windows": []}
    valid = [w for w in windows if "sharpe" in w]
    if not valid:
        return {"label": label, "error": "All failed", "windows": windows}

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]
    bls = [w["baseline_sharpe"] for w in valid if w.get("baseline_sharpe") is not None]

    recent = valid[-3:] if len(valid) >= 3 else valid
    recent_neg = sum(1 for w in recent if w["sharpe"] < 0)
    extreme = [w for w in valid if abs(w["sharpe"]) > 10]

    out = {
        "label": label,
        "sharpe": round(float(np.mean(sharpes)), 3),
        "sharpe_std": round(float(np.std(sharpes)), 3),
        "max_dd": round(float(np.min(dds)), 4),
        "cagr": round(float(np.mean(cagrs)), 4),
        "win_rate": round(float(np.mean(wrs)), 3),
        "n_windows": len(valid),
        "n_errors": len(windows) - len(valid),
        "window_details": windows,
        "rank_inversion_check": {
            "recent_3_sharpes": [w["sharpe"] for w in recent],
            "recent_3_negative": recent_neg,
            "passed": recent_neg < 2,
        },
        "extreme_windows": len(extreme),
    }
    if bls:
        out["avg_baseline_sharpe"] = round(float(np.mean(bls)), 3)
    return out


def select_best(results, v031_sharpe=1.161):
    print(f"\n{'='*60}")
    print(f"🏆 最佳方案 (V0.3.1基准: Sharpe={v031_sharpe})")
    print(f"{'='*60}")

    candidates = []
    for r in results:
        if r.get("n_windows", 0) == 0:
            print(f"  ❌ {r['label']}: {r.get('error', '?')}")
            continue
        ri = r.get("rank_inversion_check", {})
        ri_pass = ri.get("passed", False)
        beats = r["sharpe"] > v031_sharpe
        extreme = r.get("extreme_windows", 0)
        nw = r.get("n_windows", 1)
        tag = "✅" if (ri_pass and beats) else ("⚠️" if beats else "❌")

        print(f"\n  {tag} {r['label']}")
        print(f"    Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}")
        print(f"    Windows={nw}  Extreme={extreme}  RI={'PASS' if ri_pass else 'FAIL'}")
        if "avg_baseline_sharpe" in r:
            print(f"    Baseline Sharpe={r['avg_baseline_sharpe']:.3f}")

        candidates.append({
            "label": r["label"], "sharpe": r["sharpe"],
            "max_dd": r["max_dd"], "cagr": r["cagr"],
            "win_rate": r["win_rate"],
            "ri_passed": ri_pass, "beats_v031": beats,
        })

    valid = [c for c in candidates if c["ri_passed"] and c["beats_v031"]]
    if valid:
        best = max(valid, key=lambda x: x["sharpe"])
        print(f"\n  🏆 最佳: {best['label']} (Sharpe={best['sharpe']:.3f})")
        return best

    if candidates:
        best = max(candidates, key=lambda x: x["sharpe"])
        print(f"\n  ⚠️ 无方案满足RI+beat V0.3.1。最佳Sharpe: {best['label']} ({best['sharpe']:.3f})")
        return best
    return None


def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4 — 市场状态自适应 + 更长训练窗口 (FMP PIT版)")
    print("=" * 80)

    # Load FMP data
    data = load_all_data()

    # Load master (training data) for price pivot
    print("\n📂 加载master数据...")
    master = pd.read_parquet(WORKSPACE / "data" / "falcon" / "training_data_v04.parquet")
    master["date"] = master["date"].astype(str)
    print(f"  ✅ Master: {len(master)}行, {master['ticker'].nunique()}只")

    # Compute PIT ranks using FMP JSON (same as V0.3.1)
    print("\n📊 预计算PIT Rank (FMP JSON, bisect加速)...")
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
    )
    print(f"  ✅ PIT Rank: {len(ranks)}天 ({time.time()-t_pit:.0f}秒)")

    # Price pivot
    price_pivot = build_price_pivot(master)

    # Market regime
    print("\n📊 计算市场状态...")
    regime = compute_regime(WORKSPACE / "data" / "us" / "vix_10y.parquet")

    # Run all schemes
    results = []

    # A: V0.3.1 baseline, 2yr train
    results.append(wf_fixed(ranks, price_pivot, WEIGHTS_V031_BASE,
        train_years=2, label="A_v031_base_2yr"))

    # B: V0.3.1 optim, 2yr train
    results.append(wf_fixed(ranks, price_pivot, WEIGHTS_V031_OPTIM,
        train_years=2, label="B_v031_optim_2yr"))

    # C: Regime adaptive, 2yr train
    results.append(wf_regime(ranks, price_pivot, regime,
        train_years=2, label="C_regime_2yr"))

    # D: V0.3.1 baseline, 5yr train
    results.append(wf_fixed(ranks, price_pivot, WEIGHTS_V031_BASE,
        train_years=5, label="D_v031_base_5yr"))

    # E: V0.3.1 optim, 5yr train
    results.append(wf_fixed(ranks, price_pivot, WEIGHTS_V031_OPTIM,
        train_years=5, label="E_v031_optim_5yr"))

    # F: Regime adaptive, 5yr train
    results.append(wf_regime(ranks, price_pivot, regime,
        train_years=5, label="F_regime_5yr"))

    # G: V0.3.1 baseline, 8yr train
    results.append(wf_fixed(ranks, price_pivot, WEIGHTS_V031_BASE,
        train_years=8, label="G_v031_base_8yr"))

    # H: V0.3.1 optim, 8yr train
    results.append(wf_fixed(ranks, price_pivot, WEIGHTS_V031_OPTIM,
        train_years=8, label="H_v031_optim_8yr"))

    # I: Regime adaptive, 8yr train
    results.append(wf_regime(ranks, price_pivot, regime,
        train_years=8, label="I_regime_8yr"))

    # Select best
    best = select_best(results, v031_sharpe=1.161)

    # Save
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": "FMP JSON + PIT indexing (same as V0.3.1)",
            "v031_benchmark_sharpe": 1.161,
            "params": {"hold_days": 30, "top_n": 10, "cost": 0.001, "stop_loss": -0.15},
        },
        "results": [],
        "best_scheme": best,
    }
    for r in results:
        serial = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                serial[k] = float(v)
            elif isinstance(v, np.ndarray):
                serial[k] = v.tolist()
            else:
                serial[k] = v
        output["results"].append(serial)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已保存: {OUTPUT_PATH}")
    print(f"⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
