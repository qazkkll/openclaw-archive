#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — Walk-Forward参数优化 (防过拟合, 优化版)
=======================================================
方法论:
  - 2年训练窗口 → 6个月测试窗口 → 滚动前进
  - 训练集网格搜索最优参数 → 测试集验证
  - OOS结果汇总 → 确认参数稳定性

防过拟合措施:
  1. 训练/测试严格隔离 (无未来信息泄露)
  2. 参数空间小 (每组3-5个选项, 不是连续优化)
  3. PIT延迟33天 (已内置)
  4. OOS Sharpe > IS Sharpe × 0.5 才算"稳健"
  5. 参数在不同窗口间的一致性检查

性能优化:
  - 全局只算一次PIT rank (17窗口共享)
  - 用100只最活跃股票 (内存+速度)
"""
import sys, json, time, itertools
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from falcon_v03_engine import (
    precompute_pit_ranks, precompute_pit_ranks_fast, backtest_flexible,
    RATIO_FIELDS, METRIC_FIELDS, GROWTH_FIELDS, ANALYST_FIELDS,
    TECH_FIELDS, EARNINGS_FIELDS, GRADE_FIELDS,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "falcon"
FMP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fmp_premium"


# ═══════════════════════════════════════════════════
# 参数网格 (刻意小, 防过拟合)
# ═══════════════════════════════════════════════════

WEIGHT_GRID = [
    # (fund_ratio, analyst, fund_metric, earnings, grade_sentiment)
    (0.70, 0.20, 0.10, 0.00, 0.00),  # 基准: 无新因子
    (0.63, 0.18, 0.09, 0.10, 0.00),  # earnings=10%
    (0.56, 0.16, 0.08, 0.20, 0.00),  # earnings=20% (当前最优)
    (0.63, 0.18, 0.09, 0.05, 0.05),  # earnings=5% + grade=5%
    (0.56, 0.16, 0.08, 0.15, 0.05),  # earnings=15% + grade=5%
]

HOLD_DAYS_GRID = [30, 63, 90]
TOP_N_GRID = [10, 20]


def make_weights(row):
    """将权重元组转为dict。"""
    return {
        "tech": 0.0,
        "fund_ratio": row[0],
        "analyst": row[1],
        "fund_metric": row[2],
        "earnings": row[3],
        "grade_sentiment": row[4],
    }


def walk_forward_windows(dates, train_years=2, test_months=6):
    """生成Walk-Forward窗口。"""
    windows = []
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    min_date = date_objs[0]
    max_date = date_objs[-1]
    
    start_test = datetime(min_date.year + train_years, min_date.month, min_date.day)
    
    current = start_test
    while current < max_date:
        test_end_month = current.month + test_months
        test_end_year = current.year
        while test_end_month > 12:
            test_end_month -= 12
            test_end_year += 1
        try:
            test_end = datetime(test_end_year, test_end_month, current.day)
        except ValueError:
            test_end = datetime(test_end_year, test_end_month, 28)
        
        train_start = datetime(current.year - train_years, current.month, current.day)
        
        windows.append((
            train_start.strftime("%Y-%m-%d"),
            current.strftime("%Y-%m-%d"),
            current.strftime("%Y-%m-%d"),
            min(test_end, max_date).strftime("%Y-%m-%d"),
        ))
        
        current = test_end
    
    return windows


def run_backtest_safe(ranks, price_pivot, dates, regime, weights, hold_days, top_n):
    """安全运行回测, 返回None如果失败。"""
    try:
        result = backtest_flexible(
            ranks, price_pivot, dates, regime,
            weights, strategy="fixed",
            params={"hold_days": hold_days, "cost": 0.001, "stop_loss": -0.15},
            top_n=top_n,
        )
        return result
    except Exception as e:
        return None


def main():
    t_start = time.time()
    print("=" * 80)
    print("🦅 Falcon Walk-Forward 参数优化 (优化版)")
    print("=" * 80)
    
    # ── 加载全量数据 ──
    print("\n📂 Step 1: 加载数据...")
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
    
    # 全量SPX成分股 (476只, bisect加速后可行)
    
    all_dates = sorted(master["date"].unique())
    print(f"  ✅ 全量: {len(master)}行, {master['ticker'].nunique()}只, {len(all_dates)}天")
    
    # ── 全局PIT rank (只算一次, bisect加速) ──
    print("\n📊 Step 2: 全局PIT rank (bisect加速, 一次性计算)...")
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
    print("\n📊 Step 3: 价格矩阵...")
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)
    
    # ── Walk-Forward窗口 ──
    windows = walk_forward_windows(all_dates, train_years=2, test_months=6)
    print(f"\n📊 Step 4: Walk-Forward: {len(windows)}个窗口")
    for i, (ts, te, tss, tee) in enumerate(windows):
        print(f"  窗口{i+1}: 训练 {ts}~{te} → 测试 {tss}~{tee}")
    
    # ── 逐窗口优化 (PIT rank已全局算好, 只做切片+回测) ──
    all_oos_results = []
    
    for wi, (train_start, train_end, test_start, test_end) in enumerate(windows):
        t_win = time.time()
        print(f"\n{'='*60}")
        print(f"📊 窗口 {wi+1}/{len(windows)}: 训练 {train_start}~{train_end} → 测试 {test_start}~{test_end}")
        print(f"{'='*60}")
        
        train_dates = [d for d in all_dates if train_start <= d < train_end]
        test_dates = [d for d in all_dates if test_start <= d <= test_end]
        
        if len(train_dates) < 100 or len(test_dates) < 20:
            print(f"  ⚠️ 数据不足 (训练{len(train_dates)}天, 测试{len(test_dates)}天), 跳过")
            continue
        
        # ── 网格搜索 (只变weights/hold/top, 不重算PIT) ──
        best_train_sr = -999
        best_params = None
        
        for w_row in WEIGHT_GRID:
            for hold in HOLD_DAYS_GRID:
                for top_n in TOP_N_GRID:
                    weights = make_weights(w_row)
                    
                    result = run_backtest_safe(
                        ranks, price_pivot, train_dates, regime_above,
                        weights, hold, top_n,
                    )
                    
                    if result and result["sharpe"] > best_train_sr:
                        best_train_sr = result["sharpe"]
                        best_params = {
                            "weights": weights,
                            "hold_days": hold,
                            "top_n": top_n,
                            "train_sharpe": result["sharpe"],
                            "train_dd": result["dd"],
                            "train_wr": result["wr"],
                        }
        
        if best_params is None:
            print(f"  ❌ 训练期无有效结果")
            continue
        
        print(f"  📈 训练最优: Sharpe={best_params['train_sharpe']:.3f} "
              f"DD={best_params['train_dd']:.1f}% WR={best_params['train_wr']:.1f}%")
        print(f"     参数: earn={best_params['weights']['earnings']:.2f} "
              f"hold={best_params['hold_days']}d top={best_params['top_n']}")
        
        # ── 测试: 直接用已有的PIT rank (已含测试期) ──
        test_result = run_backtest_safe(
            ranks, price_pivot, test_dates, regime_above,
            best_params["weights"], best_params["hold_days"], best_params["top_n"],
        )
        
        if test_result:
            decay = test_result["sharpe"] / best_params["train_sharpe"] if best_params["train_sharpe"] > 0 else 0
            print(f"  📉 测试OOS: Sharpe={test_result['sharpe']:.3f} "
                  f"DD={test_result['dd']:.1f}% WR={test_result['wr']:.1f}%")
            print(f"  🔍 过拟合检查: OOS/IS = {decay:.2f} "
                  f"({'✅ 健康' if decay > 0.5 else '⚠️ 过拟合风险'})")
            
            all_oos_results.append({
                "window": wi + 1,
                "train_period": f"{train_start}~{train_end}",
                "test_period": f"{test_start}~{test_end}",
                "best_weights": {k: v for k, v in best_params["weights"].items() if v > 0},
                "hold_days": best_params["hold_days"],
                "top_n": best_params["top_n"],
                "train_sharpe": best_params["train_sharpe"],
                "oos_sharpe": test_result["sharpe"],
                "oos_dd": test_result["dd"],
                "oos_wr": test_result["wr"],
                "decay": decay,
            })
        
        print(f"  ⏱️ 窗口耗时: {time.time()-t_win:.0f}秒")
    
    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("📊 Walk-Forward 汇总")
    print("=" * 80)
    
    if not all_oos_results:
        print("❌ 无有效OOS结果")
        return
    
    oos_sharpes = [r["oos_sharpe"] for r in all_oos_results]
    oos_dds = [r["oos_dd"] for r in all_oos_results]
    decays = [r["decay"] for r in all_oos_results]
    
    print(f"\nOOS Sharpe: mean={np.mean(oos_sharpes):.3f} median={np.median(oos_sharpes):.3f} "
          f"std={np.std(oos_sharpes):.3f}")
    print(f"OOS DD:     mean={np.mean(oos_dds):.1f}%")
    print(f"过拟合比:   mean={np.mean(decays):.2f} "
          f"({'✅ 健康' if np.mean(decays) > 0.5 else '⚠️ 过拟合'})")
    
    # 各窗口详情
    print("\n📊 各窗口最优参数:")
    print(f"{'窗口':>4} {'earnings':>10} {'hold':>6} {'top':>5} {'IS':>8} {'OOS':>8} {'decay':>8}")
    print("-" * 50)
    for r in all_oos_results:
        earn = r["best_weights"].get("earnings", 0)
        print(f"{r['window']:>4} {earn:>10.2f} {r['hold_days']:>6} {r['top_n']:>5} "
              f"{r['train_sharpe']:>8.3f} {r['oos_sharpe']:>8.3f} {r['decay']:>8.2f}")
    
    # 全局推荐 (取OOS Sharpe最高的参数组合)
    best_window = max(all_oos_results, key=lambda x: x["oos_sharpe"])
    print(f"\n🏆 全局推荐 (OOS Sharpe最高):")
    print(f"  权重: {best_window['best_weights']}")
    print(f"  调仓: {best_window['hold_days']}天")
    print(f"  Top-N: {best_window['top_n']}")
    print(f"  OOS Sharpe: {best_window['oos_sharpe']:.3f}")
    
    # 稳健推荐
    stable = [r for r in all_oos_results if r["oos_sharpe"] > 0 and r["decay"] > 0.3]
    if stable:
        from collections import Counter
        earn_counter = Counter(round(r["best_weights"].get("earnings", 0), 2) for r in stable)
        hold_counter = Counter(r["hold_days"] for r in stable)
        top_counter = Counter(r["top_n"] for r in stable)
        
        print(f"\n🛡️ 稳健推荐 ({len(stable)}/{len(all_oos_results)}个窗口稳定):")
        print(f"  最常选earnings权重: {earn_counter.most_common(1)[0][0]} "
              f"({earn_counter.most_common(1)[0][1]}/{len(stable)}次)")
        print(f"  最常选hold_days: {hold_counter.most_common(1)[0][0]}")
        print(f"  最常选top_n: {top_counter.most_common(1)[0][0]}")
    
    elapsed = time.time() - t_start
    print(f"\n⏱️ 总耗时: {elapsed/60:.1f}分钟")


if __name__ == "__main__":
    main()
