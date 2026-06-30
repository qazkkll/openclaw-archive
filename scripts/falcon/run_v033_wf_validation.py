#!/usr/bin/env python3
"""
🦅 Falcon V0.3.3 Walk-Forward 验证
与 V0.3.1 对比: Sharpe, CAGR, MaxDD, WinRate, Trades
使用 backtest_engine.py 的 BacktestEngine（红线：不能自己实现回测）
"""
import sys, time, warnings, json
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from falcon_v03_engine import (
    precompute_pit_ranks_fast,
    RATIO_FIELDS, METRIC_FIELDS, GROWTH_FIELDS, ANALYST_FIELDS,
    TECH_FIELDS, EARNINGS_FIELDS, GRADE_FIELDS,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades
from backtest_engine import BacktestEngine, DataQualityError

DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "falcon"
FMP_DIR = SCRIPT_DIR.parent.parent / "data" / "fmp_premium"


# ═══════════════════════════════════════════════════
# V0.3.1 vs V0.3.3 权重
# ═══════════════════════════════════════════════════

V031_WEIGHTS = {
    "fund_ratio": 0.70,
    "analyst": 0.20,
    "fund_metric": 0.10,
    "fund_growth": 0.0,
    "cashflow": 0.0,
    "balance": 0.0,
    "income_stmt": 0.0,
    "grade_sentiment": 0.0,
    "earnings": 0.0,
    "insider": 0.0,
    "tech": 0.0,
    "valuation": 0.0,
}

V033_WEIGHTS = {
    "fund_ratio": 0.55,
    "analyst": 0.15,
    "fund_metric": 0.10,
    "earnings": 0.15,
    "composite": 0.05,
    "fund_growth": 0.0,
    "cashflow": 0.0,
    "balance": 0.0,
    "income_stmt": 0.0,
    "grade_sentiment": 0.0,
    "insider": 0.0,
    "tech": 0.0,
    "valuation": 0.0,
}


def load_data():
    """加载全部数据。"""
    print("📂 加载数据...")
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
    print(f"  ✅ {len(master):,}行, {master['ticker'].nunique()}只, {len(all_dates)}天")
    return master, data, all_dates


def compute_pit_ranks(master, data, all_dates):
    """计算全局PIT rank。"""
    print("\n📊 计算PIT rank (bisect加速)...")
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
    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)}天, {elapsed:.0f}秒")
    return ranks


def add_composite_factor(ranks):
    """为V0.3.3添加composite因子: fund_ratio × analyst (简化版组合因子)。"""
    print("\n🔧 添加composite因子 (fund_ratio × analyst)...")
    composite_count = 0
    for date, df in ranks.items():
        if "fund_ratio" in df.columns and "analyst" in df.columns:
            # composite = fund_ratio rank × analyst rank (两者都是0-1的percentile rank)
            df["composite"] = df["fund_ratio"] * df["analyst"]
            composite_count += 1
    print(f"  ✅ {composite_count}天添加了composite因子")
    return ranks


def get_prices(master):
    """从master构建价格矩阵。"""
    print("\n📊 构建价格矩阵...")
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    print(f"  ✅ {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")
    return price_pivot


def run_walk_forward(ranks, prices, weights, version_name):
    """运行Walk-Forward并返回结果。"""
    print(f"\n{'='*70}")
    print(f"🚀 运行 {version_name} Walk-Forward...")
    print(f"{'='*70}")
    t0 = time.time()

    engine = BacktestEngine(cost=0.001, stop_loss=-0.15)

    try:
        result = engine.walk_forward(
            ranks, prices, weights,
            hold_days=30,
            top_n=10,
            train_years=2,
            test_months=6,
        )
        elapsed = time.time() - t0
        print(f"  ✅ {version_name} 完成 ({elapsed:.0f}秒)")
        return result
    except DataQualityError as e:
        print(f"  ❌ {version_name} DataQualityError: {e}")
        return None
    except Exception as e:
        print(f"  ❌ {version_name} 异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison(v031_result, v033_result):
    """打印V0.3.1 vs V0.3.3对比表格。"""
    print(f"\n{'='*70}")
    print(f"📊 V0.3.1 vs V0.3.3 Walk-Forward 对比")
    print(f"{'='*70}")
    print(f"参数: train_years=2, test_months=6, hold_days=30, top_n=10, cost=0.1%, SL=-15%")
    print(f"{'-'*70}")

    def fmt_val(val, fmt=".3f"):
        if val is None:
            return "N/A"
        return f"{val:{fmt}}"

    # 表头
    header = f"{'指标':<20} {'V0.3.1':>15} {'V0.3.3':>15} {'Δ (V033-V031)':>15}"
    print(header)
    print(f"{'-'*70}")

    # Sharpe
    s1 = v031_result.sharpe if v031_result else None
    s3 = v033_result.sharpe if v033_result else None
    delta_s = (s3 - s1) if (s1 is not None and s3 is not None) else None
    print(f"{'Sharpe Ratio':<20} {fmt_val(s1):>15} {fmt_val(s3):>15} {fmt_val(delta_s):>15}")

    # CAGR
    c1 = v031_result.cagr if v031_result else None
    c3 = v033_result.cagr if v033_result else None
    delta_c = (c3 - c1) if (c1 is not None and c3 is not None) else None
    print(f"{'CAGR':<20} {fmt_val(c1, '.1%'):>15} {fmt_val(c3, '.1%'):>15} {fmt_val(delta_c, '.1%'):>15}")

    # MaxDD
    d1 = v031_result.max_dd if v031_result else None
    d3 = v033_result.max_dd if v033_result else None
    delta_d = (d3 - d1) if (d1 is not None and d3 is not None) else None
    print(f"{'Max Drawdown':<20} {fmt_val(d1, '.1%'):>15} {fmt_val(d3, '.1%'):>15} {fmt_val(delta_d, '.1%'):>15}")

    # Win Rate
    w1 = v031_result.win_rate if v031_result else None
    w3 = v033_result.win_rate if v033_result else None
    delta_w = (w3 - w1) if (w1 is not None and w3 is not None) else None
    print(f"{'Win Rate':<20} {fmt_val(w1, '.0%'):>15} {fmt_val(w3, '.0%'):>15} {fmt_val(delta_w, '.0%'):>15}")

    # Total Trades
    t1 = v031_result.n_trades if v031_result else None
    t3 = v033_result.n_trades if v033_result else None
    delta_t = (t3 - t1) if (t1 is not None and t3 is not None) else None
    print(f"{'Total Trades':<20} {str(t1):>15} {str(t3):>15} {str(delta_t):>15}")

    # Rebalances
    r1 = v031_result.n_rebalances if v031_result else None
    r3 = v033_result.n_rebalances if v033_result else None
    print(f"{'Rebalances':<20} {str(r1):>15} {str(r3):>15}")

    # Total Return
    tr1 = v031_result.total_return if v031_result else None
    tr3 = v033_result.total_return if v033_result else None
    print(f"{'Total Return':<20} {fmt_val(tr1, '.1%'):>15} {fmt_val(tr3, '.1%'):>15}")

    print(f"{'-'*70}")

    # 判定
    if s1 is not None and s3 is not None:
        if s3 > s1:
            print(f"\n✅ V0.3.3 Sharpe ({s3:.3f}) > V0.3.1 ({s1:.3f}) — 提升 {(s3-s1):.3f}")
        elif s3 < s1:
            print(f"\n⚠️ V0.3.3 Sharpe ({s3:.3f}) < V0.3.1 ({s1:.3f}) — 退步 {(s1-s3):.3f}")
        else:
            print(f"\n➡️ V0.3.3 Sharpe = V0.3.1 Sharpe = {s3:.3f}")

    # Warnings
    if v031_result and v031_result.warnings:
        print(f"\n  ⚠️ V0.3.1 warnings: {'; '.join(v031_result.warnings)}")
    if v033_result and v033_result.warnings:
        print(f"  ⚠️ V0.3.3 warnings: {'; '.join(v033_result.warnings)}")

    # 逐窗口明细
    print(f"\n{'='*70}")
    print(f"📋 逐窗口明细")
    print(f"{'='*70}")

    for label, result in [("V0.3.1", v031_result), ("V0.3.3", v033_result)]:
        if result and result.window_details:
            print(f"\n  {label}:")
            for w in result.window_details:
                if "error" in w:
                    print(f"    W{w['index']}: {w['period']} ❌ {w['error'][:60]}")
                else:
                    print(f"    W{w['index']}: {w['period']}  "
                          f"Sharpe={w['sharpe']:.2f}  DD={w['max_dd']:.1%}  "
                          f"WR={w['win_rate']:.0%}  Trades={w['n_trades']}")


def main():
    t_start = time.time()
    print("=" * 70)
    print("🦅 Falcon V0.3.3 Walk-Forward 验证 (与 V0.3.1 对比)")
    print("=" * 70)

    # 1. 加载数据
    master, data, all_dates = load_data()

    # 2. 计算PIT ranks
    ranks = compute_pit_ranks(master, data, all_dates)

    # 3. 构建价格矩阵
    prices = get_prices(master)

    # 4. V0.3.1 Walk-Forward (不含composite)
    v031_result = run_walk_forward(ranks, prices, V031_WEIGHTS, "V0.3.1")

    # 5. 为V0.3.3添加composite因子 (在V0.3.1之后, 不影响V0.3.1结果)
    # composite = fund_ratio rank × analyst rank (简化版组合因子)
    add_composite_factor(ranks)

    # V0.3.3 Walk-Forward (含composite)
    v033_result = run_walk_forward(ranks, prices, V033_WEIGHTS, "V0.3.3")

    # 6. 对比结果
    print_comparison(v031_result, v033_result)

    elapsed = time.time() - t_start
    print(f"\n⏱️ 总耗时: {elapsed/60:.1f}分钟")

    # 保存结果
    results = {}
    if v031_result:
        results["v031"] = {
            "sharpe": v031_result.sharpe,
            "cagr": v031_result.cagr,
            "max_dd": v031_result.max_dd,
            "win_rate": v031_result.win_rate,
            "n_trades": v031_result.n_trades,
            "total_return": v031_result.total_return,
            "window_details": v031_result.window_details,
        }
    if v033_result:
        results["v033"] = {
            "sharpe": v033_result.sharpe,
            "cagr": v033_result.cagr,
            "max_dd": v033_result.max_dd,
            "win_rate": v033_result.win_rate,
            "n_trades": v033_result.n_trades,
            "total_return": v033_result.total_return,
            "window_details": v033_result.window_details,
        }
    out_path = DATA_DIR / "v033_walk_forward_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
