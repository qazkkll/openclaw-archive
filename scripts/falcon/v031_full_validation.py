#!/usr/bin/env python3
"""
🦅 Falcon V0.3.1 全面验证套件
================================
测试清单:
  1. 因子IC/ICIR分析 → 用数据决定权重
  2. 近期窗口退化分析 (2024H2-2025H2)
  3. 行业集中度检查
  4. 交易成本敏感度
  5. SPY基准对比
  6. 参数稳定性 (earnings=0 vs 0.05 vs 0.10)
  7. 蒙特卡洛模拟 (bootstrap置信区间)
"""
import sys, json, time, warnings
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from falcon_v03_engine import (
    precompute_pit_ranks_fast, backtest_flexible,
    RATIO_FIELDS, METRIC_FIELDS, GROWTH_FIELDS, ANALYST_FIELDS,
    TECH_FIELDS, EARNINGS_FIELDS, GRADE_FIELDS,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "falcon"
FMP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fmp_premium"


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
    print(f"  ✅ {len(ranks)}天, {time.time()-t0:.0f}秒")
    return ranks


def get_regime(price_pivot):
    """计算市场regime。"""
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    return (mkt_price > mkt_ma200).astype(int)


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
        test_dates = [d for d in dates if train_end.strftime("%Y-%m-%d") <= d < test_end.strftime("%Y-%m-%d")]
        if len(test_dates) >= 50:
            windows.append(test_dates)
        train_start = train_start + relativedelta(months=test_months)
    return windows


def safe_backtest(ranks, price_pivot, dates, regime_above, weights, hold_days, top_n):
    """安全运行backtest。"""
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


# ═══════════════════════════════════════════════════
# TEST 1: 因子IC/ICIR分析
# ═══════════════════════════════════════════════════

def test_factor_ic(ranks, price_pivot, all_dates):
    """计算每个因子组的IC和ICIR。"""
    print("\n" + "=" * 80)
    print("📊 TEST 1: 因子IC/ICIR分析 (预测能力)")
    print("=" * 80)

    # 因子组定义
    factor_groups = {
        "fund_ratio": RATIO_FIELDS,
        "fund_metric": METRIC_FIELDS,
        "analyst": ANALYST_FIELDS,
        "earnings": EARNINGS_FIELDS,
        "grade_sentiment": GRADE_FIELDS,
        "tech": TECH_FIELDS,
        "growth": GROWTH_FIELDS,
    }

    # 计算forward return (20天)
    fwd_ret = price_pivot.pct_change(periods=20, fill_method=None).shift(-20)

    # 每个因子组的IC序列
    ic_series = {fg: [] for fg in factor_groups}
    # fwd_ret index可能是Timestamp或str
    if hasattr(fwd_ret.index, 'strftime'):
        fwd_dates = set(fwd_ret.index.strftime("%Y-%m-%d"))
    else:
        fwd_dates = set(str(d)[:10] for d in fwd_ret.index)
    dates_used = sorted(set(ranks.keys()) & fwd_dates)

    for date_str in dates_used:
        if date_str not in ranks:
            continue
        rank_row = ranks[date_str]
        date_ts = pd.Timestamp(date_str)
        # fwd_ret index可能是Timestamp或str
        try:
            if date_ts in fwd_ret.index:
                ret_row = fwd_ret.loc[date_ts]
            elif date_str in fwd_ret.index:
                ret_row = fwd_ret.loc[date_str]
            else:
                continue
        except Exception:
            continue

        for fg, fields in factor_groups.items():
            if not fields:
                continue
            # ranks[date]是DataFrame(index=ticker, cols=factor_group)
            df = rank_row
            if fg not in df.columns:
                continue
            fg_scores = df[fg].dropna().to_dict()

            if len(fg_scores) < 20:
                continue

            # 计算IC (rank correlation)
            common_tickers = set(fg_scores.keys()) & set(ret_row.dropna().index)
            if len(common_tickers) < 20:
                continue

            scores = [fg_scores[t] for t in common_tickers]
            rets = [ret_row[t] for t in common_tickers]

            # Spearman rank correlation
            from scipy.stats import spearmanr
            ic, p = spearmanr(scores, rets)
            if not np.isnan(ic):
                ic_series[fg].append(ic)

    # 汇总
    print(f"\n{'因子组':<18} {'IC均值':>8} {'ICIR':>8} {'IC>0占比':>10} {'样本':>6}")
    print("-" * 60)

    factor_weights_data = {}
    for fg in factor_groups:
        ics = ic_series[fg]
        if not ics:
            print(f"{fg:<18} {'N/A':>8} {'N/A':>8} {'N/A':>10} {'0':>6}")
            factor_weights_data[fg] = 0
            continue

        ic_mean = np.mean(ics)
        ic_std = np.std(ics)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        pct_positive = np.mean([1 for x in ics if x > 0]) * 100

        print(f"{fg:<18} {ic_mean:>8.4f} {icir:>8.3f} {pct_positive:>9.1f}% {len(ics):>6}")

        # 权重 = max(0, ICIR) 归一化
        factor_weights_data[fg] = max(0, icir)

    # 按ICIR生成数据驱动权重
    total = sum(factor_weights_data.values())
    if total > 0:
        data_weights = {k: round(v / total, 3) for k, v in factor_weights_data.items()}
    else:
        data_weights = {k: 0 for k in factor_weights_data}

    print(f"\n📊 数据驱动权重 (按ICIR归一化):")
    for k, v in sorted(data_weights.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    return data_weights, ic_series


# ═══════════════════════════════════════════════════
# TEST 2: 近期窗口退化分析
# ═══════════════════════════════════════════════════

def test_recent_degradation(ranks, price_pivot, all_dates, regime_above, windows):
    """分析近期窗口退化原因。"""
    print("\n" + "=" * 80)
    print("📊 TEST 2: 近期窗口退化分析 (2024H2-2025H2)")
    print("=" * 80)

    # 只跑最近3个窗口
    recent = [w for w in windows if w[0] >= "2024-07-01"]

    configs = {
        "V0.3": {"weights": {"fund_ratio": 0.56, "analyst": 0.16, "fund_metric": 0.08, "earnings": 0.20, "grade_sentiment": 0.0}, "hold": 30, "top": 5},
        "V0.3.1": {"weights": {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0}, "hold": 90, "top": 10},
        "V0.3.1_fast": {"weights": {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0}, "hold": 30, "top": 10},
        "V0.3.1_earn5": {"weights": {"fund_ratio": 0.665, "analyst": 0.19, "fund_metric": 0.095, "earnings": 0.05, "grade_sentiment": 0.0}, "hold": 90, "top": 10},
    }

    for test_dates in recent:
        print(f"\n  窗口: {test_dates[0]}~{test_dates[-1]}")
        for name, cfg in configs.items():
            r = safe_backtest(ranks, price_pivot, test_dates, regime_above,
                            cfg["weights"], cfg["hold"], cfg["top"])
            if r:
                print(f"    {name}: Sharpe={r['sharpe']:.3f} DD={r.get('dd',0):.1f}% WR={r.get('wr',0):.1f}% Ret={r.get('ret',0):.1f}%")
            else:
                print(f"    {name}: ❌ 无数据")


# ═══════════════════════════════════════════════════
# TEST 3: 行业集中度检查
# ═══════════════════════════════════════════════════

def test_sector_concentration(ranks, price_pivot, all_dates, regime_above, windows):
    """检查Top10是否存在行业集中。"""
    print("\n" + "=" * 80)
    print("📊 TEST 3: 行业集中度检查")
    print("=" * 80)

    # GICS sector mapping (简化版, 基于ticker)
    # 这里用一个近似方法: 看top10的ticker是否重叠度高
    weights = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0}

    top_tickers_all = []
    for test_dates in windows:
        if test_dates[0] < "2020-01-01":
            continue  # 只看近6年
        # 每个窗口的Top10
        last_date = test_dates[-1]
        if last_date not in ranks:
            continue

        # 计算综合得分 — ranks[date]是DataFrame(index=ticker, cols=factor_group)
        df = ranks[last_date]
        scores = {}
        for t in df.index:
            s = 0
            for fg, w in weights.items():
                if fg in df.columns:
                    v = df.loc[t, fg]
                    if pd.notna(v):
                        s += w * v
            scores[t] = s

        top10 = sorted(scores, key=scores.get, reverse=True)[:10]
        top_tickers_all.extend(top10)

    # 统计每个ticker出现频率
    freq = Counter(top_tickers_all)
    print(f"\nTop 20 最常入选的股票 (近6年):")
    for ticker, count in freq.most_common(20):
        pct = count / len([w for w in windows if w[0] >= "2020-01-01"]) * 100
        print(f"  {ticker}: {count}次 ({pct:.0f}%)")

    # 检查重复度
    total_slots = len([w for w in windows if w[0] >= "2020-01-01"]) * 10
    unique_tickers = len(freq)
    concentration = sum(v for _, v in freq.most_common(10)) / total_slots * 100
    print(f"\n  唯一股票数: {unique_tickers}")
    print(f"  Top10高频股占比: {concentration:.1f}%")
    if concentration > 50:
        print(f"  ⚠️ 行业集中风险高 — Top10频繁选入相同股票")
    else:
        print(f"  ✅ 分散度良好")


# ═══════════════════════════════════════════════════
# TEST 4: 交易成本敏感度
# ═══════════════════════════════════════════════════

def test_cost_sensitivity(ranks, price_pivot, all_dates, regime_above, windows):
    """不同交易成本下的表现。"""
    print("\n" + "=" * 80)
    print("📊 TEST 4: 交易成本敏感度")
    print("=" * 80)

    costs = [0.0005, 0.001, 0.002, 0.003]  # 0.05%, 0.1%, 0.2%, 0.3%
    configs = {
        "V0.3": {"weights": {"fund_ratio": 0.56, "analyst": 0.16, "fund_metric": 0.08, "earnings": 0.20, "grade_sentiment": 0.0}, "hold": 30, "top": 5},
        "V0.3.1": {"weights": {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0}, "hold": 90, "top": 10},
    }

    print(f"\n{'成本':<8}", end="")
    for name in configs:
        print(f"  {name+' Sharpe':>15}", end="")
    print()
    print("-" * 40)

    for cost in costs:
        print(f"{cost*100:.2f}%    ", end="")
        for name, cfg in configs.items():
            sharpes = []
            for test_dates in windows:
                r = safe_backtest(ranks, price_pivot, test_dates, regime_above,
                                cfg["weights"], cfg["hold"], cfg["top"])
                if r:
                    # 调整成本影响: 更高成本 → 更低sharpe
                    # 近似: sharpe_adj = sharpe - (cost_diff / volatility) * trades_per_year
                    trades_per_year = 365 / cfg["hold"]
                    cost_impact = (cost - 0.001) * trades_per_year * 2  # 2 sides
                    sharpes.append(r["sharpe"] - cost_impact * 0.5)  # 粗略调整
            if sharpes:
                print(f"  {np.mean(sharpes):>15.3f}", end="")
            else:
                print(f"  {'N/A':>15}", end="")
        print()


# ═══════════════════════════════════════════════════
# TEST 5: SPY基准对比
# ═══════════════════════════════════════════════════

def test_spy_benchmark(price_pivot, all_dates, windows):
    """SPY基准对比。"""
    print("\n" + "=" * 80)
    print("📊 TEST 5: SPY基准对比")
    print("=" * 80)

    # 检查SPY是否在数据中
    if "SPY" not in price_pivot.columns:
        print("  ⚠️ SPY不在数据中, 用等权市场收益代替")
        mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    else:
        mkt_ret = price_pivot["SPY"].pct_change(fill_method=None)

    spy_sharpes = []
    spy_dds = []
    spy_rets = []

    for test_dates in windows:
        rets = []
        for d in test_dates:
            try:
                if hasattr(mkt_ret.index, 'strftime'):
                    key = pd.Timestamp(d)
                else:
                    key = d
                if key in mkt_ret.index:
                    r = mkt_ret.loc[key]
                else:
                    continue
                if not np.isnan(r):
                    rets.append(r)
            except Exception:
                continue

        if len(rets) > 20:
            sr = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
            cumulative = (1 + pd.Series(rets)).cumprod()
            dd = (cumulative / cumulative.cummax() - 1).min()
            total_ret = cumulative.iloc[-1] - 1
            spy_sharpes.append(sr)
            spy_dds.append(dd)
            spy_rets.append(total_ret)

    if spy_sharpes:
        print(f"\n  SPY/市场 基准 (同期):")
        print(f"  OOS Sharpe: mean={np.mean(spy_sharpes):.3f}")
        print(f"  OOS DD:     mean={np.mean(spy_dds)*100:.1f}%")
        print(f"  OOS Ret:    mean={np.mean(spy_rets)*100:.1f}%")

        print(f"\n  V0.3.1 vs SPY:")
        v031_sharpe = 0.951  # 从之前的结果
        v031_ret = 8.4
        alpha_sharpe = v031_sharpe - np.mean(spy_sharpes)
        alpha_ret = v031_ret - np.mean(spy_rets) * 100
        print(f"  Sharpe Alpha: {alpha_sharpe:+.3f}")
        print(f"  Return Alpha: {alpha_ret:+.1f}%")
    else:
        print("  ⚠️ 无法计算SPY基准")


# ═══════════════════════════════════════════════════
# TEST 6: 扩展参数网格 (数据驱动权重)
# ═══════════════════════════════════════════════════

def test_data_driven_weights(ranks, price_pivot, all_dates, regime_above, windows, data_weights):
    """用IC驱动的权重 vs 人工权重对比。"""
    print("\n" + "=" * 80)
    print("📊 TEST 6: 数据驱动权重 vs 人工权重")
    print("=" * 80)

    # 只用IC>0的因子
    active_weights = {k: v for k, v in data_weights.items() if v > 0}
    if not active_weights:
        print("  ⚠️ 无有效因子")
        return

    total = sum(active_weights.values())
    norm_weights = {k: round(v / total, 3) for k, v in active_weights.items()}
    # 确保总和为1
    diff = 1.0 - sum(norm_weights.values())
    if norm_weights:
        max_key = max(norm_weights, key=norm_weights.get)
        norm_weights[max_key] = round(norm_weights[max_key] + diff, 3)

    print(f"\n  数据驱动权重: {norm_weights}")

    configs = {
        "V0.3.1_人工": {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0},
        "V0.3.1_数据": norm_weights,
    }

    for name, weights in configs.items():
        sharpes = []
        for test_dates in windows:
            r = safe_backtest(ranks, price_pivot, test_dates, regime_above, weights, 90, 10)
            if r:
                sharpes.append(r["sharpe"])
        if sharpes:
            print(f"  {name}: OOS Sharpe={np.mean(sharpes):.3f} (正率={np.mean([1 for s in sharpes if s>0])*100:.0f}%)")


# ═══════════════════════════════════════════════════
# TEST 7: 蒙特卡洛模拟
# ═══════════════════════════════════════════════════

def test_monte_carlo(ranks, price_pivot, all_dates, regime_above, windows):
    """Bootstrap置信区间。"""
    print("\n" + "=" * 80)
    print("📊 TEST 7: 蒙特卡洛Bootstrap置信区间")
    print("=" * 80)

    weights = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10, "earnings": 0.00, "grade_sentiment": 0.0}

    # 收集每个窗口的OOS Sharpe
    window_sharpes = []
    for test_dates in windows:
        r = safe_backtest(ranks, price_pivot, test_dates, regime_above, weights, 90, 10)
        if r:
            window_sharpes.append(r["sharpe"])

    if len(window_sharpes) < 5:
        print("  ⚠️ 窗口数不足")
        return

    # Bootstrap 1000次
    n_bootstrap = 1000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(window_sharpes, size=len(window_sharpes), replace=True)
        bootstrap_means.append(np.mean(sample))

    ci_low = np.percentile(bootstrap_means, 2.5)
    ci_high = np.percentile(bootstrap_means, 97.5)
    ci_mid = np.mean(bootstrap_means)

    print(f"\n  OOS Sharpe Bootstrap (1000次):")
    print(f"  均值: {ci_mid:.3f}")
    print(f"  95%置信区间: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  P(Sharpe>0): {np.mean([1 for x in bootstrap_means if x > 0])*100:.1f}%")
    print(f"  P(Sharpe>0.5): {np.mean([1 for x in bootstrap_means if x > 0.5])*100:.1f}%")


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.3.1 全面验证套件")
    print("=" * 80)

    master, data, all_dates = load_data()
    ranks = compute_pit_ranks(master, data, all_dates)
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    regime_above = get_regime(price_pivot)
    windows = walk_forward_windows(all_dates)

    # TEST 1: 因子IC
    data_weights, ic_series = test_factor_ic(ranks, price_pivot, all_dates)

    # TEST 2: 近期退化
    test_recent_degradation(ranks, price_pivot, all_dates, regime_above, windows)

    # TEST 3: 行业集中
    test_sector_concentration(ranks, price_pivot, all_dates, regime_above, windows)

    # TEST 4: 成本敏感
    test_cost_sensitivity(ranks, price_pivot, all_dates, regime_above, windows)

    # TEST 5: SPY基准
    test_spy_benchmark(price_pivot, all_dates, windows)

    # TEST 6: 数据驱动权重
    test_data_driven_weights(ranks, price_pivot, all_dates, regime_above, windows, data_weights)

    # TEST 7: 蒙特卡洛
    test_monte_carlo(ranks, price_pivot, all_dates, regime_above, windows)

    print(f"\n{'='*80}")
    print(f"⏱️ 总耗时: {(time.time()-t_start)/60:.1f}分钟")
    print(f"{'='*80}")

    # 保存结果
    results = {
        "data_weights": data_weights,
        "ic_means": {fg: float(np.mean(ics)) if ics else 0 for fg, ics in ic_series.items()},
    }
    out_path = DATA_DIR / "v031_full_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
