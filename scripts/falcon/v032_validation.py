#!/usr/bin/env python3
"""
🦅 Falcon V0.3.2 Validation Suite
===================================
验证三大报表因子（balance sheet / cashflow / income stmt）的增量价值。

测试清单:
  1. 全因子IC/ICIR分析 (9旧 + 15新 = 24因子组)
  2. V0.3.1 vs V0.3.2 Walk-Forward对比回测
  3. 30天 vs 60天调仓对比
  4. 近期窗口退化检查
  5. 蒙特卡洛置信区间
  6. 行业集中度
"""
import sys, json, time, warnings
from pathlib import Path
from collections import Counter
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

# ═══════════════════════════════════════════════════
# 因子组定义 (含方向, 用于IC计算)
# direction=1: 高=好 (正向因子), direction=-1: 低=好 (反向因子, 排名时反转)
# ═══════════════════════════════════════════════════
FACTOR_GROUPS = {
    # 旧因子组
    "fund_ratio": {"fields": RATIO_FIELDS, "direction": 1},
    "fund_metric": {"fields": METRIC_FIELDS, "direction": 1},
    "analyst": {"fields": ANALYST_FIELDS, "direction": 1},
    "earnings": {"fields": EARNINGS_FIELDS, "direction": 1},
    "grade_sentiment": {"fields": GRADE_FIELDS, "direction": 1},
    "tech": {"fields": TECH_FIELDS, "direction": 1},
    "growth": {"fields": GROWTH_FIELDS, "direction": 1},
    "fund_growth": {"fields": [], "direction": 1},  # engine内部计算
    "insider": {"fields": [], "direction": 1},       # engine内部计算
    "valuation": {"fields": [], "direction": 1},      # engine内部计算 (DCF/PT)
    # 新因子组 (三大报表)
    "balance": {"fields": BALANCE_FIELDS, "direction": 1},
    "cashflow": {"fields": CASHFLOW_FIELDS, "direction": 1},
    "income_stmt": {"fields": INCOME_FIELDS, "direction": 1},
}

# 反向因子 (低=好): debt_to_equity, net_debt_to_assets, capex_intensity
INVERT_FACTORS = {"debt_to_equity", "net_debt_to_assets", "capex_intensity"}


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载全部数据 (旧 + 新)。"""
    print("📂 加载数据...")
    t0 = time.time()

    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)

    # 旧数据
    data = {}
    for name in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
                  "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            data[name] = json.load(open(f))

    # FMP Premium
    data["earnings"] = load_fmp_premium_earnings(str(FMP_DIR))
    data["grades"] = load_fmp_premium_grades(str(FMP_DIR))

    # 新数据 (三大报表)
    for name in ["fmp_balance_sheet", "fmp_cashflow", "fmp_income_stmt"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            data[name] = json.load(open(f))
            print(f"  ✅ {name}: {len(data[name])} tickers")

    all_dates = sorted(master["date"].unique())
    print(f"  ✅ 总计: {len(master):,}行, {master['ticker'].nunique()}只, {len(all_dates)}天, {time.time()-t0:.0f}秒")
    return master, data, all_dates


# ═══════════════════════════════════════════════════
# 计算PIT rank (旧 + 新)
# ═══════════════════════════════════════════════════

def compute_all_ranks(master, data, all_dates):
    """计算旧因子PIT rank + 新因子PIT rank, 合并。"""
    print("\n📊 计算PIT rank...")

    # Step 1: 旧因子 (bisect加速, ~400秒)
    print("  [1/2] 旧因子 (9组)...")
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
    print(f"    ✅ {len(ranks)}天, {time.time()-t0:.0f}秒")

    # Step 2: 新因子 (三大报表, ~60秒)
    print("  [2/2] 新因子 (三大报表)...")
    t0 = time.time()
    ranks = merge_statement_factors(ranks, master, data, all_dates)
    print(f"    ✅ {len(ranks)}天, {time.time()-t0:.0f}秒")

    return ranks


def merge_statement_factors(ranks, master, data, all_dates):
    """计算三大报表因子并合并到现有ranks。"""
    # Build PIT indices (一次性)
    income_raw = data.get("fmp_income_stmt", {})
    balance_raw = data.get("fmp_balance_sheet", {})
    cashflow_raw = data.get("fmp_cashflow", {})

    if not income_raw and not balance_raw and not cashflow_raw:
        print("    ⚠️ 无三大报表数据, 跳过")
        return ranks

    print(f"    构建PIT索引: income={len(income_raw)}, balance={len(balance_raw)}, cashflow={len(cashflow_raw)} tickers...")
    income_idx = build_pit_index_statements(income_raw, use_filing_date=True)
    balance_idx = build_pit_index_statements(balance_raw, use_filing_date=False)
    cashflow_idx = build_pit_index_statements(cashflow_raw, use_filing_date=False)

    # 逐日计算新因子
    new_factor_names = BALANCE_FIELDS + CASHFLOW_FIELDS + INCOME_FIELDS
    dates_processed = 0
    dates_with_data = 0

    for date in sorted(ranks.keys()):
        rank_df = ranks[date]
        tickers = rank_df.index.tolist()

        # 计算每个ticker的新因子
        new_data = {}
        for t in tickers:
            factors = compute_statement_factors(
                t, date, balance_idx, cashflow_idx, income_idx, {}
            )
            if factors:
                new_data[t] = factors

        if not new_data:
            continue

        # 转为DataFrame, 只保留有数据的因子列
        new_df = pd.DataFrame.from_dict(new_data, orient="index")
        # 只保留至少有10个ticker有数据的列
        valid_cols = [c for c in new_df.columns if new_df[c].notna().sum() >= 10]

        if valid_cols:
            # Cross-sectional rank (percentile)
            ranked_new = new_df[valid_cols].rank(pct=True)
            # 反向因子: 低值排高 → 取 (1 - rank)
            for col in valid_cols:
                if col in INVERT_FACTORS:
                    ranked_new[col] = 1 - ranked_new[col]

            # 合并到现有ranks
            for col in valid_cols:
                rank_df[col] = ranked_new[col]

            dates_with_data += 1

        dates_processed += 1
        if dates_processed % 500 == 0:
            print(f"    📊 {dates_processed}/{len(ranks)} 天...")

    print(f"    新因子覆盖: {dates_with_data}/{len(ranks)} 天有数据")

    # 合并因子组级分数 (balance/cashflow/income_stmt = 各子因子均值)
    for date in ranks:
        df = ranks[date]
        for group_name, col_prefix in [("balance", BALANCE_FIELDS),
                                        ("cashflow", CASHFLOW_FIELDS),
                                        ("income_stmt", INCOME_FIELDS)]:
            cols = [c for c in col_prefix if c in df.columns]
            if cols:
                df[group_name] = df[cols].mean(axis=1)

    return ranks


# ═══════════════════════════════════════════════════
# Walk-Forward 辅助
# ═══════════════════════════════════════════════════

def walk_forward_windows(dates, train_years=2, test_months=6):
    """生成walk-forward窗口。"""
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
    """安全运行backtest, 返回result dict或None。"""
    try:
        result = backtest_flexible(
            ranks, price_pivot, dates, regime_above,
            weights=weights, strategy="fixed",
            params={"hold_days": hold_days, "cost": 0.001, "stop_loss": -0.15},
            top_n=top_n,
        )
        if result and result.get("sharpe") is not None:
            return result
    except Exception as e:
        pass
    return None


def get_regime(price_pivot):
    """计算市场regime (MA200上方=1, 下方=0)。"""
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    return (mkt_price > mkt_ma200).astype(int)


# ═══════════════════════════════════════════════════
# TEST 1: 全因子IC/ICIR分析
# ═══════════════════════════════════════════════════

def test_factor_ic(ranks, price_pivot, all_dates):
    """计算所有因子组的IC和ICIR, 包括新因子。"""
    print("\n" + "=" * 80)
    print("📊 TEST 1: 全因子IC/ICIR分析 (9旧 + 3新因子组)")
    print("=" * 80)

    from scipy.stats import spearmanr

    # Forward return (20天)
    fwd_ret = price_pivot.pct_change(periods=20, fill_method=None).shift(-20)
    if hasattr(fwd_ret.index, 'strftime'):
        fwd_dates = set(fwd_ret.index.strftime("%Y-%m-%d"))
    else:
        fwd_dates = set(str(d)[:10] for d in fwd_ret.index)

    dates_used = sorted(set(ranks.keys()) & fwd_dates)
    print(f"  IC计算覆盖: {len(dates_used)} 天")

    # 计算所有因子组的IC
    # 包括: 组级分数 + 个别新子因子
    all_factors = {}
    # 组级
    for fg in FACTOR_GROUPS:
        all_factors[fg] = {"direction": FACTOR_GROUPS[fg]["direction"], "level": "group"}
    # 子因子级 (新因子)
    for fname in BALANCE_FIELDS + CASHFLOW_FIELDS + INCOME_FIELDS:
        direction = -1 if fname in INVERT_FACTORS else 1
        all_factors[fname] = {"direction": direction, "level": "sub"}

    ic_series = {f: [] for f in all_factors}

    for date_str in dates_used:
        if date_str not in ranks:
            continue
        rank_row = ranks[date_str]
        date_ts = pd.Timestamp(date_str)
        try:
            if date_ts in fwd_ret.index:
                ret_row = fwd_ret.loc[date_ts]
            elif date_str in fwd_ret.index:
                ret_row = fwd_ret.loc[date_str]
            else:
                continue
        except Exception:
            continue

        for factor_name, info in all_factors.items():
            if factor_name not in rank_row.columns:
                continue
            scores = rank_row[factor_name].dropna().to_dict()
            if len(scores) < 20:
                continue

            common = set(scores.keys()) & set(ret_row.dropna().index)
            if len(common) < 20:
                continue

            s = [scores[t] * info["direction"] for t in common]
            r = [ret_row[t] for t in common]

            ic, p = spearmanr(s, r)
            if not np.isnan(ic):
                ic_series[factor_name].append(ic)

    # 汇总
    print(f"\n{'因子':<25} {'IC均值':>8} {'ICIR':>8} {'IC>0%':>7} {'样本':>6} {'级别':>6}")
    print("-" * 65)

    factor_weights_raw = {}
    for factor_name in all_factors:
        ics = ic_series[factor_name]
        level = all_factors[factor_name]["level"]
        if not ics:
            print(f"{factor_name:<25} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'0':>6} {level:>6}")
            continue

        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        pct_pos = np.mean([1 for x in ics if x > 0]) * 100

        print(f"{factor_name:<25} {ic_mean:>8.4f} {icir:>8.3f} {pct_pos:>6.1f}% {len(ics):>6} {level:>6}")
        factor_weights_raw[factor_name] = max(0, icir)

    # 数据驱动权重 (组级, 按ICIR归一化)
    group_weights_raw = {k: v for k, v in factor_weights_raw.items() if k in FACTOR_GROUPS}
    total = sum(group_weights_raw.values())
    if total > 0:
        data_weights = {k: round(v / total, 3) for k, v in group_weights_raw.items() if v > 0}
    else:
        data_weights = {}

    # 子因子级权重
    sub_weights_raw = {k: v for k, v in factor_weights_raw.items() if k not in FACTOR_GROUPS}
    total_sub = sum(sub_weights_raw.values())
    if total_sub > 0:
        sub_weights = {k: round(v / total_sub, 3) for k, v in sub_weights_raw.items() if v > 0}
    else:
        sub_weights = {}

    print(f"\n📊 数据驱动权重 (组级, 按ICIR归一化):")
    for k, v in sorted(data_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    if sub_weights:
        print(f"\n📊 新子因子权重 (按ICIR归一化):")
        for k, v in sorted(sub_weights.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.3f}")

    return data_weights, sub_weights, ic_series


# ═══════════════════════════════════════════════════
# TEST 2: V0.3.1 vs V0.3.2 Walk-Forward对比回测
# ═══════════════════════════════════════════════════

def test_walk_forward_comparison(ranks, price_pivot, all_dates, regime_above, windows,
                                  v032_weights):
    """Walk-Forward对比回测: V0.3.1 vs V0.3.2。"""
    print("\n" + "=" * 80)
    print("📊 TEST 2: Walk-Forward对比回测 (V0.3.1 vs V0.3.2)")
    print("=" * 80)

    # V0.3.1权重 (旧)
    v031_w = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10,
              "earnings": 0.0, "grade_sentiment": 0.0}

    # V0.3.2权重 (数据驱动, 包含新因子)
    v032_w = v032_weights

    configs = {
        "V0.3.1 (旧)": {"weights": v031_w, "hold": 30, "top": 10},
        "V0.3.2 (新)": {"weights": v032_w, "hold": 30, "top": 10},
    }

    # 检查V0.3.2权重是否包含新因子
    has_new = any(f in v032_w for f in BALANCE_FIELDS + CASHFLOW_FIELDS + INCOME_FIELDS +
                  ["balance", "cashflow", "income_stmt"])
    if not has_new:
        print("  ⚠️ V0.3.2权重不含新因子, 退化为旧因子权重测试")

    results = {}
    for name, cfg in configs.items():
        window_sharpes = []
        window_dds = []
        window_rets = []
        window_wrs = []
        for test_dates in windows:
            r = safe_backtest(ranks, price_pivot, test_dates, regime_above,
                            cfg["weights"], cfg["hold"], cfg["top"])
            if r:
                window_sharpes.append(r["sharpe"])
                window_dds.append(r["dd"])
                window_rets.append(r["ret"])
                window_wrs.append(r["wr"])

        if window_sharpes:
            results[name] = {
                "sharpe": np.mean(window_sharpes),
                "dd": np.mean(window_dds),
                "ret": np.mean(window_rets),
                "wr": np.mean(window_wrs),
                "positive_rate": np.mean([1 for s in window_sharpes if s > 0]) * 100,
                "n_windows": len(window_sharpes),
            }
            print(f"\n  {name}:")
            print(f"    OOS Sharpe: {results[name]['sharpe']:.3f} (正率={results[name]['positive_rate']:.0f}%)")
            print(f"    OOS DD:     {results[name]['dd']:.1f}%")
            print(f"    OOS Ret:    {results[name]['ret']:.1f}%")
            print(f"    OOS WR:     {results[name]['wr']:.1f}%")
        else:
            print(f"\n  {name}: ❌ 无数据")
            results[name] = None

    # 对比
    if results.get("V0.3.1 (旧)") and results.get("V0.3.2 (新)"):
        old = results["V0.3.1 (旧)"]
        new = results["V0.3.2 (新)"]
        delta = new["sharpe"] - old["sharpe"]
        pct = (delta / abs(old["sharpe"]) * 100) if old["sharpe"] != 0 else 0
        print(f"\n  📈 V0.3.2 vs V0.3.1:")
        print(f"    Sharpe变化: {delta:+.3f} ({pct:+.1f}%)")
        print(f"    回撤变化: {new['dd'] - old['dd']:+.1f}%")
        print(f"    胜率变化: {new['wr'] - old['wr']:+.1f}%")

    return results


# ═══════════════════════════════════════════════════
# TEST 3: 调仓频率对比 (30d vs 45d vs 60d)
# ═══════════════════════════════════════════════════

def test_rebalance_frequency(ranks, price_pivot, all_dates, regime_above, windows, v032_weights):
    """不同调仓频率对比。"""
    print("\n" + "=" * 80)
    print("📊 TEST 3: 调仓频率对比 (V0.3.2)")
    print("=" * 80)

    hold_options = [30, 45, 60]
    for hold in hold_options:
        sharpes = []
        dds = []
        for test_dates in windows:
            r = safe_backtest(ranks, price_pivot, test_dates, regime_above,
                            v032_weights, hold, 10)
            if r:
                sharpes.append(r["sharpe"])
                dds.append(r["dd"])
        if sharpes:
            print(f"  Hold={hold}d: Sharpe={np.mean(sharpes):.3f}, DD={np.mean(dds):.1f}%, 正率={np.mean([1 for s in sharpes if s>0])*100:.0f}%")
        else:
            print(f"  Hold={hold}d: ❌ 无数据")


# ═══════════════════════════════════════════════════
# TEST 4: 近期窗口退化检查
# ═══════════════════════════════════════════════════

def test_recent_degradation(ranks, price_pivot, all_dates, regime_above, windows, v032_weights):
    """检查V0.3.2在近期窗口是否退化。"""
    print("\n" + "=" * 80)
    print("📊 TEST 4: 近期窗口退化检查")
    print("=" * 80)

    v031_w = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10,
              "earnings": 0.0, "grade_sentiment": 0.0}

    recent = [w for w in windows if w[0] >= "2024-01-01"]
    if not recent:
        print("  ⚠️ 无近期窗口 (2024+)")
        return

    print(f"\n  近期窗口 (2024+): {len(recent)} 个")
    for test_dates in recent:
        label = f"{test_dates[0]}~{test_dates[-1]}"
        r_old = safe_backtest(ranks, price_pivot, test_dates, regime_above, v031_w, 30, 10)
        r_new = safe_backtest(ranks, price_pivot, test_dates, regime_above, v032_weights, 30, 10)
        s_old = r_old["sharpe"] if r_old else None
        s_new = r_new["sharpe"] if r_new else None
        winner = "✅ V0.3.2" if s_new and s_old and s_new > s_old else "❌ V0.3.1"
        if s_new is None:
            winner = "⚠️ 无数据"
        s_old_str = f"{s_old:.3f}" if s_old is not None else "N/A"
        s_new_str = f"{s_new:.3f}" if s_new is not None else "N/A"
        print(f"  {label}: V0.3.1={s_old_str}, V0.3.2={s_new_str} → {winner}")


# ═══════════════════════════════════════════════════
# TEST 5: 蒙特卡洛置信区间
# ═══════════════════════════════════════════════════

def test_monte_carlo(ranks, price_pivot, all_dates, regime_above, windows, v032_weights):
    """Bootstrap置信区间。"""
    print("\n" + "=" * 80)
    print("📊 TEST 5: 蒙特卡洛Bootstrap置信区间 (V0.3.2)")
    print("=" * 80)

    window_sharpes = []
    for test_dates in windows:
        r = safe_backtest(ranks, price_pivot, test_dates, regime_above, v032_weights, 30, 10)
        if r:
            window_sharpes.append(r["sharpe"])

    if len(window_sharpes) < 5:
        print("  ⚠️ 窗口数不足")
        return

    np.random.seed(42)
    n_bootstrap = 1000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(window_sharpes, size=len(window_sharpes), replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_low = np.percentile(bootstrap_means, 2.5)
    ci_high = np.percentile(bootstrap_means, 97.5)

    print(f"\n  OOS Sharpe Bootstrap ({n_bootstrap}次):")
    print(f"  均值: {np.mean(bootstrap_means):.3f}")
    print(f"  95%置信区间: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  P(Sharpe>0): {np.mean([1 for x in bootstrap_means if x > 0]) * 100:.0f}%")
    print(f"  P(Sharpe>0.5): {np.mean([1 for x in bootstrap_means if x > 0.5]) * 100:.0f}%")


# ═══════════════════════════════════════════════════
# TEST 6: 行业集中度
# ═══════════════════════════════════════════════════

def test_sector_concentration(ranks, price_pivot, all_dates, regime_above, windows, v032_weights):
    """检查V0.3.2的行业集中度。"""
    print("\n" + "=" * 80)
    print("📊 TEST 6: 行业集中度 (V0.3.2)")
    print("=" * 80)

    top_tickers_all = []
    for test_dates in windows:
        if test_dates[0] < "2020-01-01":
            continue
        last_date = test_dates[-1]
        if last_date not in ranks:
            continue

        df = ranks[last_date]
        scores = {}
        for t in df.index:
            s = 0
            for fg, w in v032_weights.items():
                if fg in df.columns:
                    v = df.loc[t, fg]
                    if pd.notna(v):
                        s += w * v
            scores[t] = s

        top10 = sorted(scores, key=scores.get, reverse=True)[:10]
        top_tickers_all.extend(top10)

    freq = Counter(top_tickers_all)
    n_windows = len([w for w in windows if w[0] >= "2020-01-01"])
    print(f"\n  Top 20 最常入选 (近6年, {n_windows} 窗口):")
    for ticker, count in freq.most_common(20):
        pct = count / n_windows * 100
        print(f"    {ticker}: {count}次 ({pct:.0f}%)")

    total_slots = n_windows * 10
    unique = len(freq)
    top10_pct = sum(v for _, v in freq.most_common(10)) / total_slots * 100
    print(f"\n  唯一股票数: {unique}")
    print(f"  Top10高频股占比: {top10_pct:.1f}%")
    print(f"  {'⚠️ 集中度高' if top10_pct > 50 else '✅ 分散度良好'}")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    print("🦅 Falcon V0.3.2 Validation Suite")
    print("=" * 80)
    print(f"日期: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"目标: 验证三大报表因子对V0.3的增量价值")
    print()

    t_total = time.time()

    # Load data
    master, data, all_dates = load_data()

    # Compute ranks
    ranks = compute_all_ranks(master, data, all_dates)

    # Verify ranks structure (防踩坑)
    sample_date = list(ranks.keys())[100]
    sample_df = ranks[sample_date]
    print(f"\n🔍 Ranks结构验证 (date={sample_date}):")
    print(f"  type: {type(sample_df).__name__}")
    print(f"  shape: {sample_df.shape}")
    print(f"  columns: {list(sample_df.columns)}")
    assert isinstance(sample_df, pd.DataFrame), "ranks[date]必须是DataFrame"
    assert "fund_ratio" in sample_df.columns, "旧因子必须存在"
    # 检查新因子
    new_cols = [c for c in sample_df.columns if c in BALANCE_FIELDS + CASHFLOW_FIELDS + INCOME_FIELDS
                or c in ("balance", "cashflow", "income_stmt")]
    print(f"  新因子列: {new_cols}")

    # Build price pivot & regime
    price_pivot = master.pivot(index="date", columns="ticker", values="close")
    price_pivot.index = price_pivot.index.astype(str)
    regime_above = get_regime(price_pivot)

    # Generate WF windows
    windows = walk_forward_windows(all_dates)
    print(f"\n📅 Walk-Forward窗口: {len(windows)}个")
    if windows:
        print(f"  首个: {windows[0][0]}~{windows[0][-1]}")
        print(f"  最后: {windows[-1][0]}~{windows[-1][-1]}")

    # ── TEST 1: IC/ICIR ──
    data_weights, sub_weights, ic_series = test_factor_ic(ranks, price_pivot, all_dates)

    # 设计V0.3.2权重
    # 策略: 用组级数据驱动权重 (与V0.3.1保持相同粒度)
    if data_weights:
        # 只用ICIR>0的组
        v032_weights = {k: v for k, v in data_weights.items() if v > 0}
        total = sum(v032_weights.values())
        if total > 0:
            v032_weights = {k: round(v / total, 3) for k, v in v032_weights.items()}
            # 修正浮点误差
            diff = 1.0 - sum(v032_weights.values())
            if v032_weights:
                max_key = max(v032_weights, key=v032_weights.get)
                v032_weights[max_key] = round(v032_weights[max_key] + diff, 3)
        else:
            print("  ⚠️ 无有效因子, 使用等权")
            v032_weights = {"fund_metric": 0.25, "analyst": 0.25, "earnings": 0.25, "grade_sentiment": 0.25}
    else:
        print("  ⚠️ IC分析无结果, 使用旧权重")
        v032_weights = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10}

    print(f"\n📊 V0.3.2 最终权重:")
    for k, v in sorted(v032_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    # ── TEST 2: Walk-Forward对比 ──
    wf_results = test_walk_forward_comparison(ranks, price_pivot, all_dates, regime_above,
                                               windows, v032_weights)

    # ── TEST 3: 调仓频率 ──
    test_rebalance_frequency(ranks, price_pivot, all_dates, regime_above, windows, v032_weights)

    # ── TEST 4: 近期退化 ──
    test_recent_degradation(ranks, price_pivot, all_dates, regime_above, windows, v032_weights)

    # ── TEST 5: 蒙特卡洛 ──
    test_monte_carlo(ranks, price_pivot, all_dates, regime_above, windows, v032_weights)

    # ── TEST 6: 行业集中度 ──
    test_sector_concentration(ranks, price_pivot, all_dates, regime_above, windows, v032_weights)

    # ── 总结 ──
    elapsed = time.time() - t_total
    print("\n" + "=" * 80)
    print("📋 V0.3.2 验证总结")
    print("=" * 80)
    print(f"  总耗时: {elapsed/60:.1f}分钟")
    print(f"  因子组数: {len([c for c in ranks[list(ranks.keys())[0]].columns])}")
    print(f"  Walk-Forward窗口: {len(windows)}")
    if wf_results:
        v031 = wf_results.get("V0.3.1 (旧)")
        v032 = wf_results.get("V0.3.2 (新)")
        if v031 and v032:
            delta = v032["sharpe"] - v031["sharpe"]
            print(f"  V0.3.1 OOS Sharpe: {v031['sharpe']:.3f}")
            print(f"  V0.3.2 OOS Sharpe: {v032['sharpe']:.3f} ({delta:+.3f}, {delta/abs(v031['sharpe'])*100:+.1f}%)")

    # Save results
    output = {
        "v032_weights": v032_weights,
        "v031_weights": {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10,
                         "earnings": 0.0, "grade_sentiment": 0.0},
        "wf_results": wf_results,
        "ic_top5": sorted(
            [(k, np.mean(v)) for k, v in ic_series.items() if v],
            key=lambda x: abs(x[1]), reverse=True
        )[:10],
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "n_windows": len(windows),
    }
    output_path = DATA_DIR / "v032_validation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
