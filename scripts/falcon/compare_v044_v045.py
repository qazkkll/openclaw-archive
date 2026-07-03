#!/usr/bin/env python3
"""V0.4.4 vs V0.4.5 每日对比：评分/排名/Top10/实际收益"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
FEATURES_PATH = WORKSPACE / "data/falcon/features_v04_1.parquet"
PRICES_PATH = WORKSPACE / "data/falcon/us_prices_daily.parquet"

# ─── V0.4.4 配置 ───
V044_FLIP = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'c_capex_intensity',
    'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'a_eps_revision', 'a_revenue_revision',
}
V044_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_assetGrowth',
        'g_bookValueperShareGrowth', 'g_debtGrowth',
    ],
    'analyst': ['a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps'],
    'income': ['i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
               'i_revenue_growth_yoy', 'i_gross_margin_delta'],
    'qoq': ['r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq'],
    'cashflow': ['c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield'],
}
V044_WEIGHTS = {"fund_ratio": 0.45, "growth_composite": 0.20, "qoq": 0.20, "cashflow": 0.15}
V044_GC = {"fund_growth": 0.60, "analyst": 0.25, "income": 0.15}

# ─── V0.4.5 配置 ───
V045_FLIP = {
    'r_priceToEarningsRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'c_capex_intensity',
    'a_eps_revision', 'a_revenue_revision',
}
V045_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_assetGrowth', 'g_bookValueperShareGrowth',
    ],
    'analyst': ['a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps'],
    'income': ['i_gross_margin', 'i_ebitda_margin',
               'i_revenue_growth_yoy', 'i_gross_margin_delta'],
    'qoq': ['r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq'],
    'cashflow': ['c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield'],
}
V045_WEIGHTS = {"fund_ratio": 0.40, "growth_composite": 0.30, "qoq": 0.15, "cashflow": 0.15}
V045_GC = {"fund_growth": 0.60, "analyst": 0.40, "income": 0.00}


def group_score(day, cols, flip_set):
    avail = [c for c in cols if c in day.columns and day[c].notna().sum() > 5]
    if not avail:
        return pd.Series(0.5, index=day.index)
    ranks = pd.DataFrame(index=day.index)
    for c in avail:
        r = day[c].rank(pct=True)
        if c in flip_set:
            r = 1 - r
        ranks[c] = r
    return ranks.mean(axis=1)


def score_day(day, weights, gc_weights, groups, flip_set):
    day = day.copy()
    day.index = day["ticker"].values
    scores = {}
    scores["fund_ratio"] = group_score(day, groups["fund_ratio"], flip_set)
    fg = group_score(day, groups["fund_growth"], flip_set)
    an = group_score(day, groups["analyst"], flip_set)
    inc = group_score(day, groups["income"], flip_set)
    scores["growth_composite"] = gc_weights["fund_growth"]*fg + gc_weights["analyst"]*an + gc_weights["income"]*inc
    scores["qoq"] = group_score(day, groups["qoq"], flip_set)
    scores["cashflow"] = group_score(day, groups["cashflow"], flip_set)
    composite = sum(weights[f]*scores[f] for f in weights)
    return composite


def main():
    print("📊 加载数据...")
    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = features["date"].astype(str)
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = prices["date"].astype(str)
    prices = prices.sort_values(["ticker", "date"])

    # 计算前瞻收益 (当日→5日后, 10日后, 20日后)
    prices_close = prices[["ticker", "date", "close"]].copy()
    prices_close["fwd_5d"] = prices_close.groupby("ticker")["close"].transform(lambda x: x.shift(-5)/x - 1)
    prices_close["fwd_10d"] = prices_close.groupby("ticker")["close"].transform(lambda x: x.shift(-10)/x - 1)
    prices_close["fwd_20d"] = prices_close.groupby("ticker")["close"].transform(lambda x: x.shift(-20)/x - 1)

    # 最近2个月日期
    all_dates = sorted(features["date"].unique())
    cutoff = (datetime.now() - timedelta(days=65)).strftime("%Y-%m-%d")
    recent_dates = [d for d in all_dates if d >= cutoff]
    print(f"  日期范围: {recent_dates[0]} → {recent_dates[-1]} ({len(recent_dates)}天)")

    # 计算S&P 500基准收益 (等权)
    sp500_path = WORKSPACE / "data/sp500_symbols.json"
    import json
    with open(sp500_path) as f:
        data = json.load(f)
        spx_tickers = set(data) if isinstance(data, list) else set(data.get("tickers", []))

    results = []
    for i, date in enumerate(recent_dates):
        if i % 10 == 0:
            print(f"  处理 {date} ({i+1}/{len(recent_dates)})...")
        day = features[features["date"] == date].copy()
        if len(day) < 50:
            continue

        # V0.4.4 评分
        s44 = score_day(day, V044_WEIGHTS, V044_GC, V044_GROUPS, V044_FLIP)
        # V0.4.5 评分
        s45 = score_day(day, V045_WEIGHTS, V045_GC, V045_GROUPS, V045_FLIP)

        # 合并
        df = pd.DataFrame({
            "ticker": day["ticker"].values,
            "close": day["close"].values,
            "v044_score": s44.values,
            "v045_score": s45.values,
        }, index=day.index)

        # 前瞻收益
        day_prices = prices_close[prices_close["date"] == date].set_index("ticker")
        df = df.join(day_prices[["fwd_5d", "fwd_10d", "fwd_20d"]], on="ticker")

        # 排名
        df["v044_rank"] = df["v044_score"].rank(ascending=False, method="min")
        df["v045_rank"] = df["v045_score"].rank(ascending=False, method="min")

        # S&P 500 subset
        df_spx = df[df["ticker"].isin(spx_tickers)].copy()
        spx_bench = df_spx["fwd_10d"].mean() if df_spx["fwd_10d"].notna().any() else np.nan

        # Top-10 前瞻收益 (先去掉NaN评分)
        df_spx_valid = df_spx.dropna(subset=["v044_score", "v045_score"])
        top10_44 = df_spx_valid.nsmallest(10, "v044_rank")
        top10_45 = df_spx_valid.nsmallest(10, "v045_rank")

        # 共同Top-10
        common = set(top10_44["ticker"]) & set(top10_45["ticker"])

        results.append({
            "date": date,
            "n_stocks": len(df_spx),
            # Top-10 平均评分
            "v044_top10_avg_score": round(top10_44["v044_score"].mean(), 4),
            "v045_top10_avg_score": round(top10_45["v045_score"].mean(), 4),
            # Top-10 前瞻收益
            "v044_top10_fwd5d": round(top10_44["fwd_5d"].mean()*100, 2) if top10_44["fwd_5d"].notna().any() else None,
            "v045_top10_fwd5d": round(top10_45["fwd_5d"].mean()*100, 2) if top10_45["fwd_5d"].notna().any() else None,
            "v044_top10_fwd10d": round(top10_44["fwd_10d"].mean()*100, 2) if top10_44["fwd_10d"].notna().any() else None,
            "v045_top10_fwd10d": round(top10_45["fwd_10d"].mean()*100, 2) if top10_45["fwd_10d"].notna().any() else None,
            "v044_top10_fwd20d": round(top10_44["fwd_20d"].mean()*100, 2) if top10_44["fwd_20d"].notna().any() else None,
            "v045_top10_fwd20d": round(top10_45["fwd_20d"].mean()*100, 2) if top10_45["fwd_20d"].notna().any() else None,
            # S&P 500 基准
            "sp500_fwd10d": round(spx_bench*100, 2) if not np.isnan(spx_bench) else None,
            # 超额收益
            "v044_alpha_10d": round((top10_44["fwd_10d"].mean() - spx_bench)*100, 2) if top10_44["fwd_10d"].notna().any() and not np.isnan(spx_bench) else None,
            "v045_alpha_10d": round((top10_45["fwd_10d"].mean() - spx_bench)*100, 2) if top10_45["fwd_10d"].notna().any() and not np.isnan(spx_bench) else None,
            # Top-10 差异
            "common_top10": len(common),
            "v044_top10": ",".join(top10_44["ticker"].tolist()),
            "v045_top10": ",".join(top10_45["ticker"].tolist()),
        })

    rdf = pd.DataFrame(results)
    rdf.to_csv(WORKSPACE / "data/falcon/v044_vs_v045_comparison.csv", index=False)

    # ─── 输出结果 ───
    print("\n" + "="*80)
    print("V0.4.4 vs V0.4.5 每日对比（最近2个月）")
    print("="*80)

    # 最近2周重点
    recent_2w = rdf.tail(10)
    print(f"\n📅 最近2周 ({recent_2w.iloc[0]['date']} → {recent_2w.iloc[-1]['date']})")
    print("-"*80)
    print(f"{'日期':12s} {'V044分数':>8s} {'V045分数':>8s} {'V044_10d':>8s} {'V045_10d':>8s} {'SPX_10d':>8s} {'V044α':>6s} {'V045α':>6s} {'共同':>4s}")
    print("-"*80)
    for _, r in recent_2w.iterrows():
        v44a = f"{r['v044_alpha_10d']:+.1f}" if r['v044_alpha_10d'] is not None else "N/A"
        v45a = f"{r['v045_alpha_10d']:+.1f}" if r['v045_alpha_10d'] is not None else "N/A"
        spx = f"{r['sp500_fwd10d']:+.1f}" if r['sp500_fwd10d'] is not None else "N/A"
        v44f = f"{r['v044_top10_fwd10d']:+.1f}" if r['v044_top10_fwd10d'] is not None else "N/A"
        v45f = f"{r['v045_top10_fwd10d']:+.1f}" if r['v045_top10_fwd10d'] is not None else "N/A"
        print(f"{r['date']:12s} {r['v044_top10_avg_score']:8.4f} {r['v045_top10_avg_score']:8.4f} {v44f:>8s} {v45f:>8s} {spx:>8s} {v44a:>6s} {v45a:>6s} {int(r['common_top10']):>4d}")

    # 最近2周Top-10 详细
    latest = rdf.iloc[-1]
    print(f"\n📊 最新一天 Top-10 对比 ({latest['date']})")
    print("-"*80)
    top44 = latest["v044_top10"].split(",")
    top45 = latest["v045_top10"].split(",")
    print(f"{'排名':>4s} {'V0.4.4':>8s} {'V0.4.5':>8s} {'相同?':>6s}")
    for i in range(10):
        same = "✅" if top44[i] == top45[i] else "❌"
        print(f"{i+1:>4d} {top44[i]:>8s} {top45[i]:>8s} {same:>6s}")

    # 汇总统计
    print(f"\n📈 2个月汇总统计")
    print("-"*60)

    valid = rdf.dropna(subset=["v044_top10_fwd10d", "v045_top10_fwd10d", "sp500_fwd10d"])
    if len(valid) > 0:
        v44_avg = valid["v044_top10_fwd10d"].mean()
        v45_avg = valid["v045_top10_fwd10d"].mean()
        spx_avg = valid["sp500_fwd10d"].mean()
        v44_alpha = valid["v044_alpha_10d"].mean()
        v45_alpha = valid["v045_alpha_10d"].mean()

        # 胜率
        v44_win = (valid["v044_top10_fwd10d"] > valid["sp500_fwd10d"]).mean()
        v45_win = (valid["v045_top10_fwd10d"] > valid["sp500_fwd10d"]).mean()
        v45_better = (valid["v045_top10_fwd10d"] > valid["v044_top10_fwd10d"]).mean()

        print(f"  SPX基准均值: {spx_avg:+.2f}%")
        print(f"  V0.4.4 Top-10均值: {v44_avg:+.2f}% (超额: {v44_alpha:+.2f}%)")
        print(f"  V0.4.5 Top-10均值: {v45_avg:+.2f}% (超额: {v45_alpha:+.2f}%)")
        print(f"  V0.4.4 胜率(vs SPX): {v44_win:.1%}")
        print(f"  V0.4.5 胜率(vs SPX): {v45_win:.1%}")
        print(f"  V0.4.5赢V0.4.4: {v45_better:.1%}")
        print(f"  Top-10平均重合: {valid['common_top10'].mean():.1f}只")

    # 最近2周单独统计
    print(f"\n📉 最近2周统计")
    print("-"*60)
    r2w = rdf.tail(10).dropna(subset=["v044_top10_fwd10d", "v045_top10_fwd10d", "sp500_fwd10d"])
    if len(r2w) > 0:
        print(f"  SPX基准均值: {r2w['sp500_fwd10d'].mean():+.2f}%")
        print(f"  V0.4.4 Top-10均值: {r2w['v044_top10_fwd10d'].mean():+.2f}%")
        print(f"  V0.4.5 Top-10均值: {r2w['v045_top10_fwd10d'].mean():+.2f}%")
        print(f"  V0.4.4超额: {r2w['v044_alpha_10d'].mean():+.2f}%")
        print(f"  V0.4.5超额: {r2w['v045_alpha_10d'].mean():+.2f}%")

    # 按月统计
    print(f"\n📅 按月统计")
    print("-"*60)
    rdf["month"] = rdf["date"].str[:7]
    for month, grp in rdf.groupby("month"):
        m = grp.dropna(subset=["v044_top10_fwd10d", "v045_top10_fwd10d", "sp500_fwd10d"])
        if len(m) > 0:
            print(f"  {month}: SPX={m['sp500_fwd10d'].mean():+.2f}% | "
                  f"V044={m['v044_top10_fwd10d'].mean():+.2f}%({m['v044_alpha_10d'].mean():+.2f}%) | "
                  f"V045={m['v045_top10_fwd10d'].mean():+.2f}%({m['v045_alpha_10d'].mean():+.2f}%)")

    print(f"\n✅ 完整数据已保存: data/falcon/v044_vs_v045_comparison.csv")


if __name__ == "__main__":
    main()
