#!/usr/bin/env python3
"""
Falcon Walk-Forward Validation (PIT Data)
==========================================
用features_v02_pit.parquet自包含运行Walk-Forward验证。
不依赖任何外部JSON文件。

因子组:
  - fund_ratio: PE/PB/PS/PFCF/EV (低=便宜→反向rank)
  - profitability: gross/net/operating/ebitda margins (高=好)
  - efficiency: asset/inventory/receivables turnover (高=好)
  - leverage: debt/equity, financial leverage (低=好→反向rank)
  - liquidity: current/quick ratio (高=好)
  - cashflow: operating cashflow ratio, FCF/OCF (高=好)
  - dividend: yield + payout (适度=好)
  - tech: 用价格因子(momentum, vol, RSI等)
"""

import sys, time, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

DATA_DIR = Path("data/falcon")
FEATURES_PIT = DATA_DIR / "features_v02_pit.parquet"
OUTPUT = DATA_DIR / "pit_walkforward_result.json"

# ═══════════════════════════════════════════════════
# 因子定义 (col_name, direction: 1=高好, -1=低好)
# ═══════════════════════════════════════════════════
FACTOR_GROUPS = {
    "fund_ratio": {
        "cols": ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
                 "priceToFreeCashFlowRatio", "enterpriseValueMultiple"],
        "direction": -1,  # 低估值=好
    },
    "profitability": {
        "cols": ["grossProfitMargin", "netProfitMargin", "operatingProfitMargin", "ebitdaMargin"],
        "direction": 1,
    },
    "efficiency": {
        "cols": ["assetTurnover", "inventoryTurnover", "receivablesTurnover"],
        "direction": 1,
    },
    "leverage": {
        "cols": ["debtToEquityRatio", "financialLeverageRatio"],
        "direction": -1,  # 低杠杆=好
    },
    "liquidity": {
        "cols": ["currentRatio", "quickRatio"],
        "direction": 1,
    },
    "cashflow": {
        "cols": ["freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio"],
        "direction": 1,
    },
    "dividend": {
        "cols": ["dividendYieldPercentage", "dividendPayoutRatio"],
        "direction": 1,
    },
    "tech_momentum": {
        "cols": ["ret20", "ret60", "momentum_6m"],
        "direction": 1,
    },
    "tech_quality": {
        "cols": ["vol20", "rsi14", "bb_width"],
        "direction": -1,  # 低波动=好
    },
}


def load_features():
    """加载PIT features数据。"""
    print("📂 加载 features_v02_pit.parquet...")
    t0 = time.time()
    df = pd.read_parquet(FEATURES_PIT)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    print(f"  ✅ {len(df):,}行, {df['ticker'].nunique()}只, "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}, {time.time()-t0:.0f}秒")
    return df


def compute_daily_scores(df, date):
    """计算某一天所有ticker的因子分数。"""
    day = df[df["date"] == date].copy()
    if len(day) < 50:
        return None

    scores = pd.DataFrame(index=day.index)
    scores["ticker"] = day["ticker"].values

    group_scores = {}

    for group_name, group_def in FACTOR_GROUPS.items():
        available_cols = [c for c in group_def["cols"] if c in day.columns]
        if not available_cols:
            continue

        # 取可用列
        sub = day[available_cols].copy()

        # 处理异常值: winsorize at 1%/99%
        for col in available_cols:
            q01 = sub[col].quantile(0.01)
            q99 = sub[col].quantile(0.99)
            sub[col] = sub[col].clip(q01, q99)

        # Cross-sectional rank (percentile)
        ranked = sub.rank(pct=True)

        # 方向调整
        if group_def["direction"] == -1:
            ranked = 1 - ranked

        # 组内均值作为组分数
        group_score = ranked.mean(axis=1)
        scores[group_name] = group_score.values
        group_scores[group_name] = group_score

    return scores


def walk_forward_backtest(df, weights, hold_days=60, top_n=20, train_years=2, test_months=6):
    """Walk-Forward回测。"""
    dates = sorted(df["date"].unique())
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])

    # 生成窗口
    from dateutil.relativedelta import relativedelta
    windows = []
    train_start = start
    while True:
        train_end = train_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(months=test_months)
        if test_end > end:
            break
        test_dates = [d for d in dates if train_end <= d < test_end]
        if len(test_dates) >= 20:
            windows.append(test_dates)
        train_start = train_start + relativedelta(months=test_months)

    if not windows:
        print("  ❌ 没有有效窗口")
        return None

    print(f"  {len(windows)}个Walk-Forward窗口, 每窗口~{test_months}个月")

    # 构建价格pivot
    price_pivot = df.pivot_table(index="date", columns="ticker", values="close")
    price_pivot = price_pivot.sort_index()

    # 逐窗口回测
    all_returns = []
    window_results = []

    for wi, test_dates in enumerate(windows):
        # 每hold_days调仓
        rebalance_dates = test_dates[::hold_days]
        if not rebalance_dates:
            continue

        window_ret = []
        for rb_date in rebalance_dates:
            scores = compute_daily_scores(df, rb_date)
            if scores is None:
                continue

            # 综合分数
            available_groups = [g for g in weights if g in scores.columns]
            if not available_groups:
                continue

            total_score = sum(scores[g].values * weights[g] for g in available_groups)
            scores["total"] = total_score

            # 选top_n
            top = scores.nlargest(top_n, "total")
            selected_tickers = top["ticker"].tolist()

            # 计算持有期收益
            rb_idx = price_pivot.index.get_loc(rb_date)
            end_idx = min(rb_idx + hold_days, len(price_pivot) - 1)
            if end_idx <= rb_idx:
                continue

            start_prices = price_pivot.iloc[rb_idx]
            end_prices = price_pivot.iloc[end_idx]

            # 等权组合
            rets = []
            for t in selected_tickers:
                if t in start_prices.index and t in end_prices.index:
                    sp = start_prices[t]
                    ep = end_prices[t]
                    if pd.notna(sp) and pd.notna(ep) and sp > 0:
                        rets.append(ep / sp - 1)

            if rets:
                avg_ret = np.mean(rets)
                window_ret.append(avg_ret)
                all_returns.append(avg_ret)

        if window_ret:
            window_results.append({
                "window": wi,
                "start": str(test_dates[0])[:10],
                "end": str(test_dates[-1])[:10],
                "n_rebalances": len(window_ret),
                "avg_ret": float(np.mean(window_ret)),
                "cum_ret": float(np.prod([1 + r for r in window_ret]) - 1),
            })

    if not all_returns:
        return None

    # 计算指标
    all_returns = np.array(all_returns)
    mean_ret = float(np.mean(all_returns))
    std_ret = float(np.std(all_returns))
    sharpe = mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0  # annualized

    # 最大回撤 (按累积收益)
    cum = np.cumprod(1 + all_returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(np.min(dd))

    # CAGR
    total_periods = len(all_returns)
    total_ret = float(cum[-1] - 1)
    years = total_periods * hold_days / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    win_rate = float(np.mean(all_returns > 0))

    result = {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "total_return": round(total_ret, 4),
        "win_rate": round(win_rate, 3),
        "n_periods": total_periods,
        "hold_days": hold_days,
        "top_n": top_n,
        "mean_ret_per_period": round(mean_ret, 4),
        "std_ret_per_period": round(std_ret, 4),
        "windows": window_results,
    }

    return result


def run_ic_analysis(df, weights, hold_days=60):
    """IC/ICIR分析。"""
    print("\n📊 IC/ICIR分析...")
    dates = sorted(df["date"].unique())

    # 构建价格pivot
    price_pivot = df.pivot_table(index="date", columns="ticker", values="close")
    price_pivot = price_pivot.sort_index()

    # Forward return
    fwd_ret = price_pivot.pct_change(periods=hold_days, fill_method=None).shift(-hold_days)

    ic_results = {g: [] for g in FACTOR_GROUPS}
    ic_results["composite"] = []

    # 每60天采样一次
    sample_dates = dates[::hold_days]
    print(f"  采样{len(sample_dates)}个日期, hold={hold_days}天")

    for date in sample_dates:
        if date not in fwd_ret.index:
            continue

        scores = compute_daily_scores(df, date)
        if scores is None:
            continue

        fwd = fwd_ret.loc[date]
        # 对齐
        common_tickers = scores["ticker"].tolist()
        fwd_vals = [fwd.get(t, np.nan) for t in common_tickers]

        for group_name in FACTOR_GROUPS:
            if group_name in scores.columns:
                grp_vals = scores[group_name].values
                mask = ~np.isnan(grp_vals) & ~np.isnan(np.array(fwd_vals, dtype=float))
                if mask.sum() >= 30:
                    ic, _ = spearmanr(grp_vals[mask], np.array(fwd_vals, dtype=float)[mask])
                    if not np.isnan(ic):
                        ic_results[group_name].append(ic)

        # Composite
        available_groups = [g for g in weights if g in scores.columns]
        if available_groups:
            total = sum(scores[g].values * weights[g] for g in available_groups)
            mask = ~np.isnan(total) & ~np.isnan(np.array(fwd_vals, dtype=float))
            if mask.sum() >= 30:
                ic, _ = spearmanr(total[mask], np.array(fwd_vals, dtype=float)[mask])
                if not np.isnan(ic):
                    ic_results["composite"].append(ic)

    # 汇总
    print(f"\n{'因子组':<20} {'IC均值':>8} {'ICIR':>8} {'样本数':>6}")
    print("-" * 50)
    summary = {}
    for name, ics in ic_results.items():
        if ics:
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            icir = mean_ic / std_ic if std_ic > 0 else 0
            summary[name] = {
                "ic_mean": round(float(mean_ic), 4),
                "ic_std": round(float(std_ic), 4),
                "icir": round(float(icir), 3),
                "n_samples": len(ics),
            }
            print(f"  {name:<18} {mean_ic:>8.4f} {icir:>8.3f} {len(ics):>6}")

    return summary


def main():
    print("=" * 60)
    print("🦅 Falcon Walk-Forward Validation (PIT Data)")
    print("=" * 60)

    # 1. 加载数据
    df = load_features()

    # 2. V0.3.2权重
    weights = {
        "fund_ratio": 0.05,
        "profitability": 0.20,
        "efficiency": 0.10,
        "leverage": 0.10,
        "liquidity": 0.05,
        "cashflow": 0.15,
        "dividend": 0.05,
        "tech_momentum": 0.20,
        "tech_quality": 0.10,
    }

    print(f"\n📊 权重: {weights}")

    # 3. IC分析
    ic_summary = run_ic_analysis(df, weights, hold_days=60)

    # 4. Walk-Forward回测
    print("\n" + "=" * 60)
    print("📊 Walk-Forward回测 (2年训练, 6个月测试)")
    print("=" * 60)

    for hold_days in [30, 60]:
        for top_n in [10, 20]:
            print(f"\n--- hold_days={hold_days}, top_n={top_n} ---")
            result = walk_forward_backtest(
                df, weights, hold_days=hold_days, top_n=top_n,
                train_years=2, test_months=6
            )
            if result:
                print(f"  Sharpe: {result['sharpe']:.3f}")
                print(f"  CAGR: {result['cagr']:.2%}")
                print(f"  MaxDD: {result['max_dd']:.2%}")
                print(f"  Win Rate: {result['win_rate']:.1%}")
                print(f"  Periods: {result['n_periods']}")

    # 5. 保存结果
    import json
    output = {
        "timestamp": datetime.now().isoformat(),
        "data": "features_v02_pit.parquet",
        "ic_analysis": ic_summary,
        "weights": weights,
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✅ 结果保存到 {OUTPUT}")


if __name__ == "__main__":
    main()
