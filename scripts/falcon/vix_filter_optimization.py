#!/usr/bin/env python3
"""
🦅 VIX动态过滤最优解搜索
===========================
在V0.3.2基础上，测试VIX过滤策略，找最优组合。

策略逻辑：
  当VIX高于阈值 → 减仓或不买入（保护下行期）
  当VIX低于阈值 → 正常买入

测试维度：
  1. 静态阈值：VIX > X 则不买入
  2. 比例缩放：仓位 = min(1, target/VIX)
  3. 双阈值：VIX > 高阈值=空仓，介于低高之间=半仓
  4. VIX趋势：VIX > MA(X) 则不买入
"""
import sys, json, warnings, time
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings('ignore')
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
INV = {'debt_to_equity', 'net_debt_to_assets', 'capex_intensity'}

# V0.3.2权重
V032_W = {
    'fund_growth': 0.15, 'cashflow': 0.12, 'analyst': 0.12,
    'grade_sentiment': 0.12, 'earnings': 0.10, 'balance': 0.08,
    'fund_metric': 0.06, 'insider': 0.05, 'fund_ratio': 0.05,
}

V031_W = {'fund_ratio': 0.70, 'analyst': 0.20, 'fund_metric': 0.10}


def load_all_data():
    """加载所有数据。"""
    print("📂 加载数据...")
    t0 = time.time()
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    data = {}
    for n in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
              "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{n}.json"
        if f.exists():
            data[n] = json.load(open(f))
    data["earnings"] = load_fmp_premium_earnings(str(FMP_DIR))
    data["grades"] = load_fmp_premium_grades(str(FMP_DIR))
    for n in ["fmp_balance_sheet", "fmp_cashflow", "fmp_income_stmt"]:
        f = DATA_DIR / f"{n}.json"
        if f.exists():
            data[n] = json.load(open(f))
    all_dates = sorted(master["date"].unique())

    # VIX
    vix_raw = pd.read_parquet(DATA_DIR.parent / "us" / "vix_10y.parquet")
    vix = vix_raw[("Close", "^VIX")].copy()
    vix.index = vix.index.strftime("%Y-%m-%d")
    vix_dict = vix.to_dict()

    # VIX MA
    vix_ma20 = vix.rolling(20, min_periods=10).mean()
    vix_ma60 = vix.rolling(60, min_periods=30).mean()
    vix_ma200 = vix.rolling(200, min_periods=100).mean()

    # VIX百分位 (滚动252天)
    vix_pctile = vix.rolling(252, min_periods=100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    print(f"  数据: {len(master):,}行, {master['ticker'].nunique()}只, {len(all_dates)}天, {time.time()-t0:.0f}秒")
    print(f"  VIX: {len(vix_dict)}天, 范围={min(vix_dict.values()):.1f}~{max(vix_dict.values()):.1f}")

    return master, data, all_dates, vix_dict, vix_ma20.to_dict(), vix_ma60.to_dict(), vix_ma200.to_dict(), vix_pctile.to_dict()


def compute_ranks(master, data):
    """计算PIT rank（旧+新）。"""
    print("\n📊 计算PIT rank...")
    t0 = time.time()
    ranks = precompute_pit_ranks_fast(
        master,
        data.get("fmp_ratios_historical", {}), data.get("analyst_historical", {}),
        data.get("fmp_key_metrics", {}), data.get("fmp_financial_growth", {}),
        data.get("fmp_insider", {}), data.get("fmp_dcf", {}),
        data.get("fmp_price_target", {}),
        earnings_hist=data.get("earnings", {}), grades_hist=data.get("grades", {}),
    )
    # 新因子
    income_idx = build_pit_index_statements(data.get("fmp_income_stmt", {}), use_filing_date=True)
    balance_idx = build_pit_index_statements(data.get("fmp_balance_sheet", {}), use_filing_date=False)
    cashflow_idx = build_pit_index_statements(data.get("fmp_cashflow", {}), use_filing_date=False)
    for date in sorted(ranks.keys()):
        df = ranks[date]
        tk = df.index.tolist()
        nd = {}
        for t in tk:
            f = compute_statement_factors(t, date, balance_idx, cashflow_idx, income_idx, {})
            if f:
                nd[t] = f
        if nd:
            ndf = pd.DataFrame.from_dict(nd, orient="index")
            vc = [c for c in ndf.columns if ndf[c].notna().sum() >= 10]
            if vc:
                rn = ndf[vc].rank(pct=True)
                for c in vc:
                    if c in INV:
                        rn[c] = 1 - rn[c]
                    df[c] = rn[c]
        for gn, fs in [("balance", BALANCE_FIELDS), ("cashflow", CASHFLOW_FIELDS), ("income_stmt", INCOME_FIELDS)]:
            cols = [c for c in fs if c in df.columns]
            if cols:
                df[gn] = df[cols].mean(axis=1)
    print(f"  完成: {len(ranks)}天, {time.time()-t0:.0f}秒")
    return ranks


def walk_forward_windows(dates, train_years=2, test_months=6):
    from dateutil.relativedelta import relativedelta
    windows = []
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    ts = start
    while True:
        te = ts + relativedelta(years=train_years)
        tte = te + relativedelta(months=test_months)
        if tte > end:
            break
        td = [d for d in dates if te.strftime("%Y-%m-%d") <= d < tte.strftime("%Y-%m-%d")]
        if len(td) >= 50:
            windows.append(td)
        ts += relativedelta(months=test_months)
    return windows


def get_regime(price_pivot):
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    return (mkt_price > mkt_ma200).astype(int)


def backtest_with_vix_filter(ranks, price_pivot, test_dates, regime_above, weights,
                              hold_days, top_n, vix_dict, vix_filter_fn):
    """
    带VIX过滤的回测。

    vix_filter_fn(date_str) -> float in [0, 1]
      0 = 不买入, 1 = 全仓, 0.5 = 半仓
    """
    # 先检查窗口中VIX过滤的天数
    active_dates = []
    for d in test_dates:
        scale = vix_filter_fn(d)
        if scale > 0:  # 至少有一定仓位才纳入
            active_dates.append(d)

    if len(active_dates) < 10:
        return None

    # 如果过滤后活动天数少于原始窗口的30%，跳过
    if len(active_dates) < len(test_dates) * 0.3:
        return None

    try:
        result = backtest_flexible(
            ranks, price_pivot, active_dates, regime_above,
            weights=weights, strategy="fixed",
            params={"hold_days": hold_days, "cost": 0.001, "stop_loss": -0.15},
            top_n=top_n,
        )
        if result and result.get("sharpe") is not None:
            return result
    except Exception:
        pass
    return None


def main():
    print("🦅 VIX动态过滤最优解搜索")
    print("=" * 80)
    t_total = time.time()

    master, data, all_dates, vix_dict, vix_ma20, vix_ma60, vix_ma200, vix_pctile = load_all_data()
    ranks = compute_ranks(master, data)
    price_pivot = master.pivot(index="date", columns="ticker", values="close")
    price_pivot.index = price_pivot.index.astype(str)
    regime_above = get_regime(price_pivot)
    windows = walk_forward_windows(all_dates)
    print(f"\n📅 Walk-Forward: {len(windows)} 窗口")

    # ═══════════════════════════════════════════
    # 定义VIX过滤策略
    # ═══════════════════════════════════════════

    strategies = {}

    # 基准（无过滤）
    strategies["无过滤_60d"] = {"fn": lambda d: 1.0, "hold": 60, "w": V032_W}
    strategies["无过滤_30d"] = {"fn": lambda d: 1.0, "hold": 30, "w": V032_W}
    strategies["V031_无过滤_30d"] = {"fn": lambda d: 1.0, "hold": 30, "w": V031_W}

    # 静态阈值：VIX > X 则不买入
    for thresh in [15, 18, 20, 22, 25, 30]:
        strategies[f"静态_{thresh}"] = {
            "fn": lambda d, t=thresh: 0.0 if vix_dict.get(d, 20) > t else 1.0,
            "hold": 60, "w": V032_W
        }

    # 比例缩放：仓位 = min(1, target/VIX)
    for target in [12, 15, 18, 20]:
        strategies[f"比例_{target}"] = {
            "fn": lambda d, t=target: min(1.0, t / max(vix_dict.get(d, 20), 1)),
            "hold": 60, "w": V032_W
        }

    # 双阈值：>高=空仓, 低~高=半仓, <低=全仓
    for low, high in [(15, 25), (18, 25), (18, 30), (20, 30), (15, 30)]:
        strategies[f"双阈值_{low}_{high}"] = {
            "fn": lambda d, l=low, h=high: (
                0.0 if vix_dict.get(d, 20) > h else
                0.5 if vix_dict.get(d, 20) > l else
                1.0
            ),
            "hold": 60, "w": V032_W
        }

    # VIX趋势：VIX > MA(X) 则不买入
    for ma_dict, ma_name in [(vix_ma20, "MA20"), (vix_ma60, "MA60"), (vix_ma200, "MA200")]:
        strategies[f"趋势_{ma_name}"] = {
            "fn": lambda d, m=ma_dict: 0.0 if vix_dict.get(d, 20) > m.get(d, 20) else 1.0,
            "hold": 60, "w": V032_W
        }

    # VIX百分位：>X%百分位则不买入
    for pct in [0.5, 0.6, 0.7, 0.8]:
        strategies[f"百分位_{int(pct*100)}"] = {
            "fn": lambda d, p=pct: 0.0 if vix_pctile.get(d, 0.5) > p else 1.0,
            "hold": 60, "w": V032_W
        }

    # ═══════════════════════════════════════════
    # 全量回测
    # ═══════════════════════════════════════════

    print(f"\n📊 测试 {len(strategies)} 个策略组合...")
    results = {}

    for name, cfg in strategies.items():
        sharpes, dds, rets, wrs = [], [], [], []
        active_window_dates = []  # 记录有效窗口日期
        for test_dates in windows:
            r = backtest_with_vix_filter(
                ranks, price_pivot, test_dates, regime_above,
                cfg["w"], cfg["hold"], 10,
                vix_dict, cfg["fn"]
            )
            if r:
                sharpes.append(r["sharpe"])
                dds.append(r["dd"])
                rets.append(r["ret"])
                wrs.append(r["wr"])
                active_window_dates.append(test_dates)

        if sharpes:
            pos_rate = np.mean([1 for s in sharpes if s > 0]) * 100
            # 近期窗口（修复：用日期匹配而非索引）
            recent_sharpes = []
            for td, s_val in zip(active_window_dates, sharpes):
                if td[0] >= "2024-01-01":
                    recent_sharpes.append(s_val)

            results[name] = {
                "sharpe": np.mean(sharpes),
                "dd": np.mean(dds),
                "ret": np.mean(rets),
                "wr": np.mean(wrs),
                "pos": pos_rate,
                "recent": np.mean(recent_sharpes) if recent_sharpes else None,
                "n": len(sharpes),
                "std": np.std(sharpes),
            }

    # ═══════════════════════════════════════════
    # 排序输出
    # ═══════════════════════════════════════════

    print("\n" + "=" * 90)
    print("📊 全量结果排名（按夏普比率）")
    print("=" * 90)
    print(f"\n{'策略':<25} {'夏普':>7} {'波动':>7} {'回撤':>7} {'收益':>7} {'胜率':>6} {'近期':>7} {'正率':>5}")
    print("-" * 90)

    for name, r in sorted(results.items(), key=lambda x: -x[1]["sharpe"]):
        recent_str = f"{r['recent']:.3f}" if r["recent"] is not None else "N/A"
        print(f"{name:<25} {r['sharpe']:>7.3f} {r['std']:>7.3f} {r['dd']:>6.1f}% {r['ret']:>6.1f}% {r['wr']:>5.1f}% {recent_str:>7} {r['pos']:>4.0f}%")

    # ═══════════════════════════════════════════
    # 最优解分析
    # ═══════════════════════════════════════════

    baseline = results.get("无过滤_60d")
    best = max(results.items(), key=lambda x: x[1]["sharpe"])

    print("\n" + "=" * 90)
    print("📋 最优解分析")
    print("=" * 90)

    if baseline:
        print(f"\n基准 (无过滤60天): 夏普={baseline['sharpe']:.3f}, 回撤={baseline['dd']:.1f}%")
        if baseline.get("recent"):
            print(f"  近期: {baseline['recent']:.3f}")

    print(f"\n最优: {best[0]}")
    print(f"  夏普: {best[1]['sharpe']:.3f} (vs基准: {best[1]['sharpe'] - baseline['sharpe']:+.3f})")
    print(f"  回撤: {best[1]['dd']:.1f}% (vs基准: {best[1]['dd'] - baseline['dd']:+.1f}%)")
    if best[1].get("recent") and baseline.get("recent"):
        print(f"  近期: {best[1]['recent']:.3f} (vs基准: {best[1]['recent'] - baseline['recent']:+.3f})")

    # 找近期最强
    recent_best = max(
        ((n, r) for n, r in results.items() if r.get("recent") is not None),
        key=lambda x: x[1]["recent"],
        default=None,
    )
    if recent_best:
        print(f"\n近期最强: {recent_best[0]}")
        print(f"  近期夏普: {recent_best[1]['recent']:.3f}")
        print(f"  全期夏普: {recent_best[1]['sharpe']:.3f}")

    # 找综合最优（全期+近期加权）
    composite_best = max(
        results.items(),
        key=lambda x: x[1]["sharpe"] * 0.6 + (x[1].get("recent") or 0) * 0.4,
    )
    print(f"\n综合最优 (60%全期+40%近期): {composite_best[0]}")
    print(f"  全期: {composite_best[1]['sharpe']:.3f}, 近期: {composite_best[1].get('recent', 'N/A')}")

    # 保存
    output = {
        "results": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                        for kk, vv in v.items()}
                   for k, v in results.items()},
        "best_overall": best[0],
        "best_recent": recent_best[0] if recent_best else None,
        "best_composite": composite_best[0],
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(DATA_DIR / "vix_filter_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {DATA_DIR / 'vix_filter_results.json'}")
    print(f"总耗时: {(time.time() - t_total) / 60:.1f}分钟")


if __name__ == "__main__":
    main()
