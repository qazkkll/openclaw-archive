#!/usr/bin/env python3
"""
🦅 Falcon V0.4 — T5.5 细粒度优化：训练窗口 + 权重网格搜索
================================================================
1. 训练窗口网格搜索 (1yr, 1.5yr, 2yr, 2.5yr, 3yr, 4yr, 5yr)
2. 权重网格搜索 (在最佳训练窗口上)
3. 因子组合测试 (2因子 vs 3因子)
4. 集成方法 (不同窗口/权重的模型平均)
5. 对每个方案跑Walk-Forward + Rank Inversion检查

Walk-Forward参数:
  test_months=6, hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15

红线: 必须用backtest_engine.py回测
"""
import sys, json, time, warnings
from pathlib import Path
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError
from falcon_v03_engine import (
    precompute_pit_ranks_fast,
)

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
OUTPUT_PATH = DATA_DIR / "v04_fine_grained_results.json"

# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════
#  Walk-Forward核心
# ═══════════════════════════════════════════════════
def run_wf(ranks, price_pivot, weights, train_years, test_months=6,
           hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """运行Walk-Forward，返回聚合结果。
    
    train_years可以是浮点数(如1.5)，会转换为月份。
    """
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    dates = sorted(ranks.keys())
    if not dates:
        return {"error": "No dates"}

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    windows = []
    idx = 0
    train_months_total = int(train_years * 12)  # 转为月份，避免非整数year

    while True:
        train_end = train_start + pd.DateOffset(months=train_months_total)
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

    return _agg(windows)


def _agg(windows):
    """聚合窗口结果。"""
    if not windows:
        return {"error": "No windows"}
    valid = [w for w in windows if "sharpe" in w]
    if not valid:
        return {"error": "All failed", "windows": windows}

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]
    bls = [w["baseline_sharpe"] for w in valid if w.get("baseline_sharpe") is not None]

    recent = valid[-3:] if len(valid) >= 3 else valid
    recent_neg = sum(1 for w in recent if w["sharpe"] < 0)
    extreme = [w for w in valid if abs(w["sharpe"]) > 10]

    out = {
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


# ═══════════════════════════════════════════════════
#  1. 训练窗口网格搜索
# ═══════════════════════════════════════════════════
def grid_search_train_window(ranks, price_pivot):
    """测试不同训练窗口，固定权重 fund_ratio=0.75, analyst=0.20, fund_metric=0.05。"""
    print("\n" + "=" * 70)
    print("📊 1. 训练窗口网格搜索")
    print("=" * 70)

    weights = {"fund_ratio": 0.75, "analyst": 0.20, "fund_metric": 0.05}
    train_windows = [1, 1.5, 2, 2.5, 3, 4, 5]
    results = []

    for tw in train_windows:
        label = f"train_{tw}yr"
        print(f"\n  ⏳ {label} (fund_ratio=0.75, analyst=0.20, fund_metric=0.05)...")
        r = run_wf(ranks, price_pivot, weights, train_years=tw)
        r["label"] = label
        r["train_years"] = tw
        r["weights"] = weights.copy()
        results.append(r)
        if "sharpe" in r:
            print(f"    → Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  "
                  f"Windows={r['n_windows']}  RI={'PASS' if r.get('rank_inversion_check', {}).get('passed') else 'FAIL'}")
        else:
            print(f"    → ERROR: {r.get('error', '?')}")

    return results


# ═══════════════════════════════════════════════════
#  2. 权重网格搜索
# ═══════════════════════════════════════════════════
def grid_search_weights(ranks, price_pivot, best_train_years):
    """在最佳训练窗口上搜索最佳权重组合。"""
    print("\n" + "=" * 70)
    print(f"📊 2. 权重网格搜索 (train={best_train_years}yr)")
    print("=" * 70)

    fund_ratios = [0.65, 0.70, 0.75, 0.80, 0.85]
    analysts = [0.10, 0.15, 0.20, 0.25, 0.30]
    fund_metrics = [0.05, 0.10, 0.15]

    # 约束: fund_ratio + analyst + fund_metric = 1.0
    valid_combos = []
    for fr, an, fm in product(fund_ratios, analysts, fund_metrics):
        if abs(fr + an + fm - 1.0) < 1e-6:
            valid_combos.append((fr, an, fm))

    print(f"  有效权重组合: {len(valid_combos)} 个 (满足 sum=1.0)")
    results = []

    for i, (fr, an, fm) in enumerate(valid_combos):
        weights = {"fund_ratio": fr, "analyst": an, "fund_metric": fm}
        label = f"w_{fr:.2f}_{an:.2f}_{fm:.2f}"
        if (i + 1) % 5 == 0 or i == 0:
            print(f"\n  ⏳ [{i+1}/{len(valid_combos)}] {label}...")

        r = run_wf(ranks, price_pivot, weights, train_years=best_train_years)
        r["label"] = label
        r["weights"] = weights.copy()
        r["train_years"] = best_train_years
        results.append(r)

        if "sharpe" in r and ((i + 1) % 5 == 0 or i == 0):
            print(f"    → Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  "
                  f"RI={'PASS' if r.get('rank_inversion_check', {}).get('passed') else 'FAIL'}")

    return results


# ═══════════════════════════════════════════════════
#  3. 因子组合测试
# ═══════════════════════════════════════════════════
def test_factor_combinations(ranks, price_pivot, train_years):
    """测试去掉某个因子组的效果。"""
    print("\n" + "=" * 70)
    print(f"📊 3. 因子组合测试 (train={train_years}yr)")
    print("=" * 70)

    combos = [
        ("fund_ratio+analyst (去掉fund_metric)", {"fund_ratio": 0.85, "analyst": 0.15}),
        ("fund_ratio+fund_metric (去掉analyst)", {"fund_ratio": 0.90, "fund_metric": 0.10}),
        ("analyst+fund_metric (去掉fund_ratio)", {"analyst": 0.80, "fund_metric": 0.20}),
        ("fund_ratio only", {"fund_ratio": 1.0}),
        ("analyst only", {"analyst": 1.0}),
        ("fund_metric only", {"fund_metric": 1.0}),
    ]

    results = []
    for label, weights in combos:
        print(f"\n  ⏳ {label}: {weights}")
        r = run_wf(ranks, price_pivot, weights, train_years=train_years)
        r["label"] = label
        r["weights"] = weights.copy()
        r["train_years"] = train_years
        results.append(r)
        if "sharpe" in r:
            print(f"    → Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd']:.1%}  "
                  f"CAGR={r['cagr']:.1%}  WR={r['win_rate']:.0%}  "
                  f"RI={'PASS' if r.get('rank_inversion_check', {}).get('passed') else 'FAIL'}")
        else:
            print(f"    → ERROR: {r.get('error', '?')}")

    return results


# ═══════════════════════════════════════════════════
#  4. 集成方法
# ═══════════════════════════════════════════════════
def test_ensemble(ranks, price_pivot):
    """测试集成方法：不同训练窗口的模型平均。"""
    print("\n" + "=" * 70)
    print("📊 4. 集成方法")
    print("=" * 70)

    # 不同训练窗口的模型平均
    train_years_list = [1, 2, 3]
    base_weights = {"fund_ratio": 0.75, "analyst": 0.20, "fund_metric": 0.05}

    results = []

    # 4a: 多窗口权重平均 (对每个日期，用多个训练窗口的分数加权平均)
    # 注意: 这里用一个技巧——用较短窗口的权重做加权组合
    # 简化实现: 对每组参数分别跑WF，然后取平均
    print("\n  4a: 多训练窗口结果平均 (1yr+2yr+3yr)")
    wf_results = []
    for tw in train_years_list:
        r = run_wf(ranks, price_pivot, base_weights, train_years=tw)
        wf_results.append(r)

    # 对齐窗口并平均
    all_window_sharpes = []
    for wr in wf_results:
        if "window_details" in wr:
            valid = [w for w in wr["window_details"] if "sharpe" in w]
            all_window_sharpes.append([w["sharpe"] for w in valid])

    if all_window_sharpes and all(len(s) == len(all_window_sharpes[0]) for s in all_window_sharpes):
        avg_sharpes = [float(np.mean([s[i] for s in all_window_sharpes]))
                       for i in range(len(all_window_sharpes[0]))]
        ensemble = {
            "label": "ensemble_1yr_2yr_3yr",
            "method": "average_window_sharpes",
            "sharpe": round(float(np.mean(avg_sharpes)), 3),
            "sharpe_std": round(float(np.std(avg_sharpes)), 3),
            "n_windows": len(avg_sharpes),
            "individual_results": [
                {"train_years": tw, "sharpe": wr.get("sharpe", 0)}
                for tw, wr in zip(train_years_list, wf_results)
            ],
        }
        # Rank inversion check on ensemble
        recent = avg_sharpes[-3:] if len(avg_sharpes) >= 3 else avg_sharpes
        ensemble["rank_inversion_check"] = {
            "recent_3_sharpes": [round(s, 3) for s in recent],
            "recent_3_negative": sum(1 for s in recent if s < 0),
            "passed": sum(1 for s in recent if s < 0) < 2,
        }
        results.append(ensemble)
        print(f"    → Ensemble Sharpe={ensemble['sharpe']:.3f}  "
              f"RI={'PASS' if ensemble['rank_inversion_check']['passed'] else 'FAIL'}")
    else:
        print("    → 无法对齐窗口，跳过")

    # 4b: 不同权重组合平均
    print("\n  4b: 不同权重组合平均 (0.75+0.20+0.05 vs 0.70+0.20+0.10)")
    weight_sets = [
        {"fund_ratio": 0.75, "analyst": 0.20, "fund_metric": 0.05},
        {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10},
        {"fund_ratio": 0.80, "analyst": 0.15, "fund_metric": 0.05},
    ]
    wf_results_w = []
    for ws in weight_sets:
        r = run_wf(ranks, price_pivot, ws, train_years=2)
        wf_results_w.append(r)

    all_window_sharpes_w = []
    for wr in wf_results_w:
        if "window_details" in wr:
            valid = [w for w in wr["window_details"] if "sharpe" in w]
            all_window_sharpes_w.append([w["sharpe"] for w in valid])

    if all_window_sharpes_w and all(len(s) == len(all_window_sharpes_w[0]) for s in all_window_sharpes_w):
        avg_sharpes_w = [float(np.mean([s[i] for s in all_window_sharpes_w]))
                         for i in range(len(all_window_sharpes_w[0]))]
        ensemble_w = {
            "label": "ensemble_3weights_2yr",
            "method": "average_weight_sharpes",
            "sharpe": round(float(np.mean(avg_sharpes_w)), 3),
            "sharpe_std": round(float(np.std(avg_sharpes_w)), 3),
            "n_windows": len(avg_sharpes_w),
            "individual_results": [
                {"weights": ws, "sharpe": wr.get("sharpe", 0)}
                for ws, wr in zip(weight_sets, wf_results_w)
            ],
        }
        recent = avg_sharpes_w[-3:] if len(avg_sharpes_w) >= 3 else avg_sharpes_w
        ensemble_w["rank_inversion_check"] = {
            "recent_3_sharpes": [round(s, 3) for s in recent],
            "recent_3_negative": sum(1 for s in recent if s < 0),
            "passed": sum(1 for s in recent if s < 0) < 2,
        }
        results.append(ensemble_w)
        print(f"    → Ensemble Sharpe={ensemble_w['sharpe']:.3f}  "
              f"RI={'PASS' if ensemble_w['rank_inversion_check']['passed'] else 'FAIL'}")
    else:
        print("    → 无法对齐窗口，跳过")

    return results


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4 — T5.5 细粒度优化")
    print("   训练窗口网格搜索 + 权重网格搜索 + 因子组合 + 集成方法")
    print("=" * 80)

    # Load data
    data = load_all_data()

    print("\n📂 加载master数据...")
    master = pd.read_parquet(DATA_DIR / "training_data_v04.parquet")
    master["date"] = master["date"].astype(str)
    print(f"  ✅ Master: {len(master)}行, {master['ticker'].nunique()}只")

    # Compute PIT ranks
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

    # ── 1. 训练窗口网格搜索 ──
    tw_results = grid_search_train_window(ranks, price_pivot)

    # 找最佳训练窗口
    valid_tw = [r for r in tw_results if "sharpe" in r and r.get("rank_inversion_check", {}).get("passed")]
    if valid_tw:
        best_tw = max(valid_tw, key=lambda x: x["sharpe"])
    else:
        # Fallback: 用所有有sharpe的
        valid_tw = [r for r in tw_results if "sharpe" in r]
        best_tw = max(valid_tw, key=lambda x: x["sharpe"]) if valid_tw else None

    best_train_years = best_tw["train_years"] if best_tw else 2
    print(f"\n  🏆 最佳训练窗口: {best_tw['label']} (Sharpe={best_tw['sharpe']:.3f})" if best_tw else "  ⚠️ 无有效结果，默认用2yr")

    # ── 2. 权重网格搜索 ──
    w_results = grid_search_weights(ranks, price_pivot, best_train_years)

    # ── 3. 因子组合测试 ──
    fc_results = test_factor_combinations(ranks, price_pivot, best_train_years)

    # ── 4. 集成方法 ──
    ens_results = test_ensemble(ranks, price_pivot)

    # ── 5. 对每个方案跑Walk-Forward + Rank Inversion检查 ──
    V031_SHARPE = 1.161  # V0.3.1 baseline
    all_candidates = []
    for r in tw_results + w_results + fc_results + ens_results:
        if "sharpe" not in r:
            continue
        ri = r.get("rank_inversion_check", {})
        ri_pass = ri.get("passed", False)
        beats = r["sharpe"] > V031_SHARPE
        extreme = r.get("extreme_windows", 0)
        nw = r.get("n_windows", 1)

        # 排除异常窗口太多的
        if extreme > nw * 0.25 and nw > 0:
            status = "EXCLUDED"
        elif ri_pass and beats:
            status = "✅"
        elif beats:
            status = "⚠️"
        else:
            status = "❌"

        tag = f"  {status} {r.get('label', '?')}"
        print(f"\n{tag}")
        # 安全打印所有可用指标
        parts = [f"Sharpe={r['sharpe']:.3f}"]
        if 'max_dd' in r: parts.append(f"MaxDD={r['max_dd']:.1%}")
        if 'cagr' in r: parts.append(f"CAGR={r['cagr']:.1%}")
        if 'win_rate' in r: parts.append(f"WR={r['win_rate']:.0%}")
        parts.append(f"Windows={nw}  RI={'PASS' if ri_pass else 'FAIL'}")
        print(f"    {'  '.join(parts)}")
        if "weights" in r:
            print(f"    Weights: {r['weights']}")
        if "train_years" in r:
            print(f"    Train: {r['train_years']}yr")

        all_candidates.append({
            "label": r.get("label", "?"),
            "sharpe": r["sharpe"],
            "max_dd": r.get("max_dd"),
            "cagr": r.get("cagr"),
            "win_rate": r.get("win_rate"),
            "ri_passed": ri_pass,
            "beats_v031": beats,
            "extreme_windows": extreme,
            "n_windows": nw,
            "weights": r.get("weights"),
            "train_years": r.get("train_years"),
            "excluded": extreme > nw * 0.25 and nw > 0,
        })

    # 选择最佳: RI通过 + beat V0.3.1 + Sharpe最高 + 不被排除
    valid_candidates = [c for c in all_candidates
                        if c["ri_passed"] and c["beats_v031"] and not c["excluded"]]
    if valid_candidates:
        best = max(valid_candidates, key=lambda x: x["sharpe"])
        print(f"\n  🏆 最终最佳: {best['label']} (Sharpe={best['sharpe']:.3f})")
    else:
        # 尝试只要 beat V0.3.1
        beats_candidates = [c for c in all_candidates if c["beats_v031"] and not c["excluded"]]
        if beats_candidates:
            best = max(beats_candidates, key=lambda x: x["sharpe"])
            print(f"\n  ⚠️ 无RI+beat组合。最佳beats: {best['label']} (Sharpe={best['sharpe']:.3f})")
        else:
            best = max(all_candidates, key=lambda x: x["sharpe"]) if all_candidates else None
            print(f"\n  ❌ 无方案beat V0.3.1。最佳Sharpe: {best['label'] if best else '?'}")

    # ═══════════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════════
    def serialize(obj):
        """JSON序列化辅助。"""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": "T5.5 Fine-Grained Optimization",
            "v031_benchmark_sharpe": V031_SHARPE,
            "params": {
                "hold_days": 30, "top_n": 10, "cost": 0.001,
                "stop_loss": -0.15, "test_months": 6,
            },
            "total_runtime_minutes": round((time.time() - t0) / 60, 1),
        },
        "sections": {
            "1_train_window_grid": _serialize_results(tw_results),
            "2_weight_grid": _serialize_results(w_results),
            "3_factor_combinations": _serialize_results(fc_results),
            "4_ensemble": _serialize_results(ens_results),
        },
        "best_scheme": best,
        "all_candidates": all_candidates,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=serialize)

    print(f"\n✅ 结果已保存: {OUTPUT_PATH}")
    print(f"⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")


def _serialize_results(results):
    """序列化结果列表。"""
    out = []
    for r in results:
        serial = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                serial[k] = float(v)
            elif isinstance(v, np.ndarray):
                serial[k] = v.tolist()
            elif isinstance(v, pd.Timestamp):
                serial[k] = str(v)
            elif isinstance(v, dict):
                serial[k] = {kk: serialize(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                serial[k] = [serialize(x) for x in v]
            else:
                serial[k] = v
        out.append(serial)
    return out


def serialize(obj):
    """通用序列化。"""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(x) for x in obj]
    return obj


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
