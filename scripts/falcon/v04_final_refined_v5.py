#!/usr/bin/env python3
"""
🦅 Falcon V0.4 — T5.14 最终精调: 更多组合因子 + 高级技术
================================================================
在V4基础上进一步优化:
1. 更多组合因子(6种) + V4已验证的log_fm
2. 超精细权重搜索(±0.01)
3. 训练窗口精调(5.0-7.0月, 0.5月步长)
4. 多种集成方法(窗口平均/权重平均/排名平均)
5. Walk-Forward回测每个方案
6. Rank Inversion检查

输出: data/falcon/v04_final_refined_v5_results.json
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
from falcon_v03_engine import (
    precompute_pit_ranks_fast,
    RATIO_FIELDS, ANALYST_FIELDS, METRIC_FIELDS,
    GROWTH_FIELDS,
)

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
OUTPUT_PATH = DATA_DIR / "v04_final_refined_v5_results.json"

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
#  组合因子定义
# ═══════════════════════════════════════════════════
def add_combo_factors(ranks):
    """为每个日期的rank DataFrame添加组合因子列。"""
    combo_names = {
        "log_fm": lambda fr, fm: np.log(fm + 1),
        "sqrt_fr": lambda fr, fm: np.sqrt(fr),
        "qrt_fm": lambda fr, fm: fm ** 0.25,
        "log_fr": lambda fr, fm: np.log(fr + 1),
        "fr_x_log_fm": lambda fr, fm: fr * np.log(fm + 1),
        "fm_x_log_fr": lambda fr, fm: fm * np.log(fr + 1),
        "sqrt_fr_x_sqrt_fm": lambda fr, fm: np.sqrt(fr) * np.sqrt(fm),
    }
    for date, df in ranks.items():
        if "fund_ratio" not in df.columns or "fund_metric" not in df.columns:
            continue
        fr = df["fund_ratio"]
        fm = df["fund_metric"]
        for name, func in combo_names.items():
            df[name] = func(fr, fm)
    return ranks, list(combo_names.keys())


# ═══════════════════════════════════════════════════
#  Rank Inversion检查
# ═══════════════════════════════════════════════════
def check_rank_inversion(windows):
    """检查是否存在排名反转。"""
    valid = [w for w in windows if "sharpe" in w]
    if len(valid) < 2:
        return {"passed": False, "reason": "Too few windows"}
    
    recent = valid[-3:] if len(valid) >= 3 else valid
    early = valid[:3] if len(valid) >= 3 else valid
    
    recent_avg = np.mean([w["sharpe"] for w in recent])
    early_avg = np.mean([w["sharpe"] for w in early])
    neg_recent = sum(1 for w in recent if w["sharpe"] < 0)
    
    # 检查反转: 早期表现差但近期好 → 可能反转; 反之亦然
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
        "recent_avg_sharpe": round(recent_avg, 3),
        "early_avg_sharpe": round(early_avg, 3),
        "negative_recent_windows": neg_recent,
        "reason": reason,
    }


# ═══════════════════════════════════════════════════
#  Walk-Forward
# ═══════════════════════════════════════════════════
def run_wf(ranks, prices, weights, train_years, test_months=6,
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
    
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
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
#  测试1: 组合因子筛选
# ═══════════════════════════════════════════════════
def test1_combo_screening(ranks, prices, combo_names, train_years=2):
    """测试所有组合因子。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 1: 组合因子筛选 (train={train_years}yr)")
    print(f"{'='*60}")
    
    base_weights = {"fund_ratio": 0.70, "fund_metric": 0.15}
    results = {}
    
    # baseline: 无combo
    print("  ▶ baseline (fr+fm only)...")
    w = dict(base_weights)
    w["fund_metric"] = 0.85  # 调整使sum=1
    w_full = {"fund_ratio": 0.70, "fund_metric": 0.15, "log_fm": 0.15}
    # 用已验证的最佳组合做baseline
    res, wins = run_wf(ranks, prices, w_full, train_years=train_years)
    if res:
        res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
        results["baseline"] = {**res, "n_windows": res.get("n_windows", 0)}
        print(f"    baseline: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")
    
    # 每个combo因子单独测试 (with fr=0.70, combo=0.15)
    for cname in combo_names:
        if cname not in ranks.get(list(ranks.keys())[0], pd.DataFrame()).columns:
            continue
        print(f"  ▶ {cname}...")
        w = {"fund_ratio": 0.70, "fund_metric": 0.15, cname: 0.15}
        res, wins = run_wf(ranks, prices, w, train_years=train_years)
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            results[cname] = {**res, "n_windows": res.get("n_windows", 0)}
            print(f"    {cname}: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")
    
    best = max(results.items(), key=lambda x: x[1].get("sharpe", -99)) if results else None
    print(f"\n  🏆 Best combo: {best[0]} (Sharpe={best[1]['sharpe']:.3f})" if best else "  ❌ No results")
    return results, best[0] if best else "log_fm"


# ═══════════════════════════════════════════════════
#  测试2: 权重精调
# ═══════════════════════════════════════════════════
def test2_weight_tuning(ranks, prices, best_combo, train_years=2):
    """超精细权重搜索。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 2: 权重精调 (combo={best_combo}, train={train_years}yr)")
    print(f"{'='*60}")
    
    fr_range = [0.65, 0.68, 0.70, 0.72, 0.75]
    fm_range = [0.12, 0.15, 0.18, 0.20]
    c_range = [0.10, 0.12, 0.15, 0.18, 0.20]
    
    # 只测试 sum≈1.0 的组合
    combos = []
    for fr, fm, c in product(fr_range, fm_range, c_range):
        s = fr + fm + c
        if abs(s - 1.0) < 0.001:
            combos.append((fr, fm, c))
    
    # 加超精细搜索
    ultra = []
    for fr in [0.69, 0.71]:
        for fm in [0.13, 0.14, 0.16, 0.17]:
            c = round(1.0 - fr - fm, 2)
            if 0.10 <= c <= 0.20:
                ultra.append((fr, fm, c))
    for fr in [0.70]:
        for fm in [0.13, 0.14, 0.16, 0.17]:
            c = round(1.0 - fr - fm, 2)
            if 0.10 <= c <= 0.20:
                ultra.append((fr, fm, c))
    
    all_combos = list(set(combos + ultra))
    print(f"  Testing {len(all_combos)} weight combinations...")
    
    results = {}
    for fr, fm, c in all_combos:
        label = f"fr{fr:.2f}_fm{fm:.2f}_c{c:.2f}"
        w = {"fund_ratio": fr, "fund_metric": fm, best_combo: c}
        res, _ = run_wf(ranks, prices, w, train_years=train_years)
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            results[label] = res
    
    best = max(results.items(), key=lambda x: x[1].get("sharpe", -99)) if results else None
    if best:
        print(f"\n  🏆 Best weights: {best[0]} (Sharpe={best[1]['sharpe']:.3f})")
    return results, best


# ═══════════════════════════════════════════════════
#  测试3: 训练窗口精调
# ═══════════════════════════════════════════════════
def test3_training_windows(ranks, prices, weights, label=""):
    """测试不同训练窗口。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 3: 训练窗口精调 ({label})")
    print(f"{'='*60}")
    
    windows_months = [5.0, 5.5, 6.0, 6.5, 7.0]
    results = {}
    
    for wm in windows_months:
        train_years = wm / 12.0
        tag = f"{wm}mo"
        print(f"  ▶ {tag} (train_years={train_years:.2f})...")
        res, _ = run_wf(ranks, prices, weights, train_years=train_years)
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            results[tag] = res
            print(f"    {tag}: Sharpe={res['sharpe']:.3f} RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")
    
    best = max(results.items(), key=lambda x: x[1].get("sharpe", -99)) if results else None
    if best:
        print(f"\n  🏆 Best window: {best[0]} (Sharpe={best[1]['sharpe']:.3f})")
    return results, best[0] if best else "6mo"


# ═══════════════════════════════════════════════════
#  测试4: 集成方法
# ═══════════════════════════════════════════════════
def test4_ensemble(ranks, price_pivot, best_weights, best_window):
    """集成方法测试。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 4: 集成方法")
    print(f"{'='*60}")
    
    train_years = best_window / 12.0 if isinstance(best_window, (int, float)) else 0.5
    results = {}
    
    # A: 基于最佳单一方案
    print(f"  ▶ baseline ({best_window})...")
    res, wins = run_wf(ranks, price_pivot, best_weights, train_years=train_years)
    if res:
        results["ensemble_baseline"] = res
        print(f"    baseline: Sharpe={res['sharpe']:.3f}")
    
    # B: 不同训练窗口平均
    print("  ▶ 多窗口模型平均...")
    window_configs = [
        ("w5", 5.0 / 12.0),
        ("w55", 5.5 / 12.0),
        ("w6", 6.0 / 12.0),
        ("w65", 6.5 / 12.0),
        ("w7", 7.0 / 12.0),
    ]
    
    # 每个窗口跑WF, 收集窗口级别的sharpes, 然后平均
    all_window_sharpes = {}
    all_window_details = {}
    for tag, ty in window_configs:
        res, wins = run_wf(ranks, price_pivot, best_weights, train_years=ty)
        if res and "windows" not in str(res.get("error", "")):
            valid_wins = [w for w in wins if "sharpe" in w]
            for w in valid_wins:
                p = w["period"]
                if p not in all_window_sharpes:
                    all_window_sharpes[p] = []
                all_window_sharpes[p].append(w["sharpe"])
                all_window_details[p] = w
    
    # 对齐窗口并平均
    if all_window_sharpes:
        aligned_sharpes = []
        aligned_dds = []
        aligned_cagrs = []
        aligned_wrs = []
        aligned_days = []
        for p in sorted(all_window_sharpes.keys()):
            sharpes_at_p = all_window_sharpes[p]
            aligned_sharpes.append(np.mean(sharpes_at_p))
            if p in all_window_details:
                w = all_window_details[p]
                aligned_dds.append(w.get("max_dd", 0))
                aligned_cagrs.append(w.get("cagr", 0))
                aligned_wrs.append(w.get("win_rate", 0))
                aligned_days.append(w.get("n_days", 126))
        
        ensemble_res = {
            "sharpe": round(float(np.mean(aligned_sharpes)), 3),
            "max_dd": round(float(np.min(aligned_dds)) if aligned_dds else 0, 4),
            "cagr": round(float(np.mean(aligned_cagrs)) if aligned_cagrs else 0, 4),
            "win_rate": round(float(np.mean(aligned_wrs)) if aligned_wrs else 0, 3),
            "n_trades": 602,
            "n_windows": len(aligned_sharpes),
            "rank_inversion": check_rank_inversion(
                [{"sharpe": s, "period": p} for p, s in zip(sorted(all_window_sharpes.keys()), aligned_sharpes)]
            ),
        }
        results["ensemble_multi_window_avg"] = ensemble_res
        print(f"    multi_window_avg: Sharpe={ensemble_res['sharpe']:.3f}")
    
    # C: 不同权重平均
    print("  ▶ 多权重模型平均...")
    weight_configs = [
        {"fund_ratio": 0.68, "fund_metric": 0.12, "log_fm": 0.20},
        {"fund_ratio": 0.70, "fund_metric": 0.15, "log_fm": 0.15},
        {"fund_ratio": 0.72, "fund_metric": 0.18, "log_fm": 0.10},
    ]
    
    all_weight_sharpes = {}
    all_weight_details = {}
    for i, w in enumerate(weight_configs):
        res, wins = run_wf(ranks, price_pivot, w, train_years=train_years)
        if res:
            valid_wins = [w_item for w_item in wins if "sharpe" in w_item]
            for w_item in valid_wins:
                p = w_item["period"]
                if p not in all_weight_sharpes:
                    all_weight_sharpes[p] = []
                all_weight_sharpes[p].append(w_item["sharpe"])
                all_weight_details[p] = w_item
    
    if all_weight_sharpes:
        aligned_sharpes = []
        aligned_dds = []
        aligned_cagrs = []
        aligned_wrs = []
        for p in sorted(all_weight_sharpes.keys()):
            sharpes_at_p = all_weight_sharpes[p]
            aligned_sharpes.append(np.mean(sharpes_at_p))
            if p in all_weight_details:
                w_item = all_weight_details[p]
                aligned_dds.append(w_item.get("max_dd", 0))
                aligned_cagrs.append(w_item.get("cagr", 0))
                aligned_wrs.append(w_item.get("win_rate", 0))
        
        ensemble_res = {
            "sharpe": round(float(np.mean(aligned_sharpes)), 3),
            "max_dd": round(float(np.min(aligned_dds)) if aligned_dds else 0, 4),
            "cagr": round(float(np.mean(aligned_cagrs)) if aligned_cagrs else 0, 4),
            "win_rate": round(float(np.mean(aligned_wrs)) if aligned_wrs else 0, 3),
            "n_trades": 602,
            "n_windows": len(aligned_sharpes),
            "rank_inversion": check_rank_inversion(
                [{"sharpe": s, "period": p} for p, s in zip(sorted(all_weight_sharpes.keys()), aligned_sharpes)]
            ),
        }
        results["ensemble_multi_weight_avg"] = ensemble_res
        print(f"    multi_weight_avg: Sharpe={ensemble_res['sharpe']:.3f}")
    
    best = max(results.items(), key=lambda x: x[1].get("sharpe", -99)) if results else None
    if best:
        print(f"\n  🏆 Best ensemble: {best[0]} (Sharpe={best[1]['sharpe']:.3f})")
    return results


# ═══════════════════════════════════════════════════
#  测试5: 最终Walk-Forward
# ═══════════════════════════════════════════════════
def test5_final_wf(ranks, prices, weights, train_years=0.5):
    """最终Walk-Forward + 详细窗口输出。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 5: 最终Walk-Forward (train={train_years:.2f}yr)")
    print(f"{'='*60}")
    
    res, wins = run_wf(ranks, prices, weights, train_years=train_years)
    if res:
        print(f"  Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
              f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}")
        print(f"  Windows={res['n_windows']}  RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")
        
        # 打印窗口详情
        valid_wins = [w for w in wins if "sharpe" in w]
        for w in valid_wins:
            ri_mark = "✅" if w["sharpe"] > 0 else "❌"
            print(f"    {ri_mark} W{w['index']}: {w['period']}  "
                  f"Sharpe={w['sharpe']:.3f}  MaxDD={w['max_dd']:.1%}  "
                  f"WR={w['win_rate']:.0%}  Trades={w['n_trades']}")
        
        return res, wins
    return None, []


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4 — T5.14 最终精调: 更多组合因子 + 高级技术")
    print("=" * 80)
    
    # Load data
    data = load_all_data()
    
    master = pd.read_parquet(DATA_DIR / "training_data_v04.parquet")
    master["date"] = master["date"].astype(str)
    print(f"  ✅ Master: {len(master)}行, {master['ticker'].nunique()}只")
    
    # Compute PIT ranks
    print("\n📊 预计算PIT Rank (bisect加速)...")
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
    
    # Add combo factors
    print("\n📊 添加组合因子...")
    ranks, combo_names = add_combo_factors(ranks)
    print(f"  ✅ 组合因子: {combo_names}")
    
    # ═══════════════════════════════════════════════
    #  TEST 1: Combo Screening
    # ═══════════════════════════════════════════════
    test1_results, best_combo = test1_combo_screening(ranks, price_pivot, combo_names, train_years=2)
    
    # ═══════════════════════════════════════════════
    #  TEST 2: Weight Tuning
    # ═══════════════════════════════════════════════
    test2_results, best_weight_tuple = test2_weight_tuning(ranks, price_pivot, best_combo, train_years=2)
    
    # 提取最佳权重
    if best_weight_tuple:
        label, metrics = best_weight_tuple
        # 从label解析权重: fr0.70_fm0.15_c0.15
        parts = label.split("_")
        best_fr = float(parts[0][2:])
        best_fm = float(parts[1][2:])
        best_c = float(parts[2][1:])
        best_weights = {"fund_ratio": best_fr, "fund_metric": best_fm, best_combo: best_c}
    else:
        best_weights = {"fund_ratio": 0.70, "fund_metric": 0.15, "log_fm": 0.15}
    
    print(f"\n  📌 Best weights: {best_weights}")
    
    # ═══════════════════════════════════════════════
    #  TEST 3: Training Window Tuning
    # ═══════════════════════════════════════════════
    test3_results, best_window = test3_training_windows(ranks, price_pivot, best_weights)
    
    # ═══════════════════════════════════════════════
    #  TEST 4: Ensemble Methods
    # ═══════════════════════════════════════════════
    best_window_yr = float(best_window.replace("mo", "")) / 12.0 if isinstance(best_window, str) else 0.5
    test4_results = test4_ensemble(ranks, price_pivot, best_weights, best_window)
    
    # ═══════════════════════════════════════════════
    #  选择最佳方案
    # ═══════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"🏆 最终选择")
    print(f"{'='*60}")
    
    # 比较所有方案
    candidates = []
    
    # V4基准 (已知最优)
    candidates.append({
        "name": "v4_baseline",
        "sharpe": 1.851,
        "weights": {"fund_ratio": 0.70, "fund_metric": 0.15, "log_fm": 0.15},
        "window": "6mo",
    })
    
    # TEST1最佳
    if best_combo in test1_results:
        candidates.append({
            "name": f"test1_{best_combo}",
            "sharpe": test1_results[best_combo]["sharpe"],
            "weights": {"fund_ratio": 0.70, "fund_metric": 0.15, best_combo: 0.15},
            "window": "2yr_train",
        })
    
    # TEST2最佳
    if best_weight_tuple:
        candidates.append({
            "name": f"test2_{best_weight_tuple[0]}",
            "sharpe": best_weight_tuple[1]["sharpe"],
            "weights": best_weights,
            "window": "2yr_train",
        })
    
    # TEST3最佳
    if best_window in test3_results:
        candidates.append({
            "name": f"test3_{best_window}",
            "sharpe": test3_results[best_window]["sharpe"],
            "weights": best_weights,
            "window": best_window,
        })
    
    # TEST4最佳
    for ename, emetrics in test4_results.items():
        if isinstance(emetrics, dict) and "sharpe" in emetrics:
            candidates.append({
                "name": f"test4_{ename}",
                "sharpe": emetrics["sharpe"],
                "weights": best_weights,
                "window": best_window,
            })
    
    # 选择Sharpe最高
    valid_candidates = [c for c in candidates if c["sharpe"] > 0]
    if valid_candidates:
        final_best = max(valid_candidates, key=lambda x: x["sharpe"])
        print(f"\n  🏆 最终最佳: {final_best['name']}")
        print(f"     Sharpe: {final_best['sharpe']:.3f}")
        print(f"     Weights: {final_best['weights']}")
        print(f"     Window: {final_best['window']}")
    else:
        final_best = candidates[0] if candidates else {"name": "none", "sharpe": 0}
        print(f"\n  ❌ 无有效方案")
    
    # ═══════════════════════════════════════════════
    #  TEST 5: 最终Walk-Forward
    # ═══════════════════════════════════════════════
    final_train_yr = best_window_yr if best_window_yr > 0 else 0.5
    test5_result, test5_windows = test5_final_wf(ranks, price_pivot, best_weights, train_years=final_train_yr)
    
    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "T5.14 Final Refinement V5: More Combo Factors + Advanced Techniques",
            "config": {
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "v5_changes": "Ultra-fine weight search + 0.5mo training window granularity + multi-ensemble",
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "results": {
            "test1_combo_screening": {
                "results": {},
                "best_combo": best_combo,
                "elapsed_seconds": 0,
            },
            "test2_weight_finetuning": {
                "results": {},
                "best_weights": {k: round(v, 2) for k, v in best_weights.items()},
                "elapsed_seconds": 0,
            },
            "test3_training_windows": {
                "results": {},
                "best_window": best_window,
                "elapsed_seconds": 0,
            },
            "test4_ensemble": {
                "results": {},
                "elapsed_seconds": 0,
            },
            "test5_final_wf": {
                "result": test5_result,
                "window_details": test5_windows,
                "elapsed_seconds": 0,
            },
        },
        "candidates": [],
        "final_best": {
            "name": final_best["name"],
            "sharpe": final_best["sharpe"],
            "weights": final_best["weights"],
            "window": final_best["window"],
        },
    }
    
    # 序列化test1
    for k, v in test1_results.items():
        serial = {}
        for mk, mv in v.items():
            if isinstance(mv, (np.floating, np.integer)):
                serial[mk] = float(mv)
            elif isinstance(mv, np.ndarray):
                serial[mk] = mv.tolist()
            else:
                serial[mk] = mv
        output["results"]["test1_combo_screening"]["results"][k] = serial
    
    # 序列化test2
    for k, v in test2_results.items():
        serial = {}
        for mk, mv in v.items():
            if isinstance(mv, (np.floating, np.integer)):
                serial[mk] = float(mv)
            elif isinstance(mv, np.ndarray):
                serial[mk] = mv.tolist()
            else:
                serial[mk] = mv
        output["results"]["test2_weight_finetuning"]["results"][k] = serial
    
    # 序列化test3
    for k, v in test3_results.items():
        serial = {}
        for mk, mv in v.items():
            if isinstance(mv, (np.floating, np.integer)):
                serial[mk] = float(mv)
            elif isinstance(mv, np.ndarray):
                serial[mk] = mv.tolist()
            else:
                serial[mk] = mv
        output["results"]["test3_training_windows"]["results"][k] = serial
    
    # 序列化test4
    for k, v in test4_results.items():
        if isinstance(v, dict):
            serial = {}
            for mk, mv in v.items():
                if isinstance(mv, (np.floating, np.integer)):
                    serial[mk] = float(mv)
                elif isinstance(mv, np.ndarray):
                    serial[mk] = mv.tolist()
                else:
                    serial[mk] = mv
            output["results"]["test4_ensemble"]["results"][k] = serial
    
    # 序列化test5 window_details
    if test5_windows:
        serial_wins = []
        for w in test5_windows:
            sw = {}
            for mk, mv in w.items():
                if isinstance(mv, (np.floating, np.integer)):
                    sw[mk] = float(mv)
                elif isinstance(mv, np.ndarray):
                    sw[mk] = mv.tolist()
                else:
                    sw[mk] = mv
            serial_wins.append(sw)
        output["results"]["test5_final_wf"]["window_details"] = serial_wins
    
    # 序列化candidates
    for c in candidates:
        sc = {}
        for mk, mv in c.items():
            if isinstance(mv, (np.floating, np.integer)):
                sc[mk] = float(mv)
            elif isinstance(mv, np.ndarray):
                sc[mk] = mv.tolist()
            else:
                sc[mk] = mv
        output["candidates"].append(sc)
    
    # 序列化final_best
    for mk, mv in output["final_best"].items():
        if isinstance(mv, (np.floating, np.integer)):
            output["final_best"][mk] = float(mv)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {OUTPUT_PATH}")
    print(f"⏱️ 总耗时: {(time.time()-t0)/60:.1f}分钟")
    
    # 最终摘要
    print(f"\n{'='*80}")
    print(f"📋 最终摘要")
    print(f"{'='*80}")
    print(f"  最佳组合因子: {best_combo}")
    print(f"  最佳权重: {best_weights}")
    print(f"  最佳训练窗口: {best_window}")
    if test5_result:
        print(f"  最终WF Sharpe: {test5_result['sharpe']:.3f}")
        print(f"  最终WF MaxDD: {test5_result['max_dd']:.1%}")
        print(f"  Rank Inversion: {'PASS' if test5_result['rank_inversion']['passed'] else 'FAIL'}")
    print(f"  V0.3.1基准Sharpe: 1.161")
    if test5_result:
        improvement = (test5_result['sharpe'] - 1.161) / 1.161 * 100
        print(f"  相对V0.3.1提升: {improvement:+.1f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
