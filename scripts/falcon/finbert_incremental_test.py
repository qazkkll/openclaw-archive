#!/usr/bin/env python3
"""
FinBERT 增量测试
================
测试情绪因子加入 Falcon 后 Sharpe 是否提升。
对比: 纯基本面 vs 基本面+情绪 (权重 0%~20%)
"""
import sys
import json
import time
import glob
from pathlib import Path

import pandas as pd
import numpy as np

FALCON_DIR = Path(__file__).resolve().parent.parent / "falcon"
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
FINBERT_DIR = PROJECT_ROOT / "data" / "finbert_sentiment"

sys.path.insert(0, str(FALCON_DIR))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible


def load_sentiment_data():
    """加载所有 FinBERT 情绪数据，返回 ticker -> date -> avg_sentiment。"""
    files = sorted(FINBERT_DIR.rglob("*.parquet"))
    if not files:
        print("❌ 无 FinBERT 数据")
        return {}

    all_dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            all_dfs.append(df)
        except:
            continue

    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True, errors="coerce")
    combined = combined.dropna(subset=["published_at", "sentiment"])
    combined["date"] = combined["published_at"].dt.strftime("%Y-%m-%d")

    # 按 ticker+date 聚合: 7天滚动均值 × 置信度加权
    combined["weighted_sent"] = combined["sentiment"] * combined["confidence"]
    daily = combined.groupby(["ticker", "date"]).agg(
        avg_sent=("sentiment", "mean"),
        weighted_sent=("weighted_sent", "mean"),
        count=("sentiment", "count"),
    ).reset_index()

    # 转为 dict: ticker -> {date -> sentiment}
    sent_dict = {}
    for _, row in daily.iterrows():
        t = row["ticker"]
        if t not in sent_dict:
            sent_dict[t] = {}
        sent_dict[t][row["date"]] = row["weighted_sent"]

    print(f"📊 情绪数据: {len(sent_dict)} tickers, {len(daily)} ticker-dates")
    return sent_dict


def add_sentiment_to_ranks(ranks_dict, sent_dict, lookback=7):
    """给 ranks_dict 的每一天添加情绪因子。"""
    dates = sorted(ranks_dict.keys())
    enriched = 0

    for date in dates:
        rank_df = ranks_dict[date]
        sentiments = {}

        for ticker in rank_df.index:
            # 过去 lookback 天的情绪均值
            ticker_sent = sent_dict.get(ticker, {})
            recent = []
            for d_offset in range(lookback):
                d = pd.Timestamp(date) - pd.Timedelta(days=d_offset)
                d_str = d.strftime("%Y-%m-%d")
                if d_str in ticker_sent:
                    recent.append(ticker_sent[d_str])

            if recent:
                sentiments[ticker] = np.mean(recent)

        if len(sentiments) > 10:
            sent_series = pd.Series(sentiments)
            rank_df["sentiment"] = sent_series.rank(pct=True)
            enriched += 1
        else:
            rank_df["sentiment"] = 0.5

        ranks_dict[date] = rank_df

    print(f"📊 情绪因子加入: {enriched}/{len(dates)} 天有足够数据")
    return ranks_dict


def main():
    t0 = time.time()
    print("=" * 80)
    print("FinBERT 增量测试 — 情绪因子对 Falcon Sharpe 的影响")
    print("=" * 80)

    # 加载 Falcon 数据
    print("\n📊 加载 SPX 数据...")
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)

    data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_historical.json"),
        ("analyst_historical", "analyst_historical.json"),
        ("fmp_key_metrics", "fmp_key_metrics.json"),
        ("fmp_financial_growth", "fmp_financial_growth.json"),
    ]:
        f = DATA_DIR / fname
        data[name] = json.load(open(f)) if f.exists() else {}

    print(f"  ✅ {master['ticker'].nunique()} 只")

    # 加载情绪数据
    sent_dict = load_sentiment_data()
    if not sent_dict:
        print("❌ 无情绪数据，退出")
        return

    # 预计算 rank (基础版，无情绪)
    print("\n📊 预计算基础 PIT rank...")
    ranks_dict = precompute_pit_ranks(
        master, data["fmp_ratios_historical"], data["analyst_historical"],
        data["fmp_key_metrics"], data["fmp_financial_growth"],
        {}, {}, {}
    )

    # 加入情绪因子
    print("\n📊 加入情绪因子...")
    ranks_dict = add_sentiment_to_ranks(ranks_dict, sent_dict)

    # 价格矩阵 + regime
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    # 测试日期范围 (用有情绪数据的日期)
    all_dates = sorted(ranks_dict.keys())
    test_dates = [d for d in all_dates if "2024" in d or "2023" in d]

    # ── 对比测试 ──
    configs = [
        ("纯基本面 (基准)", {"fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1}),
        ("+情绪5%", {"fund_ratio": 0.665, "analyst": 0.19, "fund_metric": 0.095, "sentiment": 0.05}),
        ("+情绪10%", {"fund_ratio": 0.63, "analyst": 0.18, "fund_metric": 0.09, "sentiment": 0.10}),
        ("+情绪15%", {"fund_ratio": 0.595, "analyst": 0.17, "fund_metric": 0.085, "sentiment": 0.15}),
        ("+情绪20%", {"fund_ratio": 0.56, "analyst": 0.16, "fund_metric": 0.08, "sentiment": 0.20}),
    ]

    params = {"hold_days": 30, "stop_loss": -0.15, "bear_alloc": 0.50}
    results = []

    print(f"\n{'='*80}")
    print(f"📊 增量测试 (2023-2024, {len(test_dates)} 天)")
    print(f"{'='*80}")
    print(f"\n{'配置':20} {'Sharpe':>8} {'MaxDD':>8} {'Return':>8} {'WR':>8} {'Trades':>8}")
    print("-" * 60)

    for name, weights in configs:
        res = backtest_flexible(ranks_dict, price_pivot, test_dates, regime_above,
                                weights, "fixed", params, top_n=5)
        if res:
            print(f"{name:20} {res['sharpe']:8.3f} {res['dd']:7.1f}% {res['ret']:7.0f}% "
                  f"{res['wr']:7.1f}% {res['trades']:8d}")
            results.append({"config": name, **res})
        else:
            print(f"{name:20}  FAILED")

    # 结论
    if len(results) >= 2:
        baseline = results[0]["sharpe"]
        best = max(results[1:], key=lambda x: x["sharpe"])
        improvement = (best["sharpe"] - baseline) / baseline * 100 if baseline > 0 else 0

        print(f"\n{'='*80}")
        print(f"📊 结论")
        print(f"{'='*80}")
        print(f"  基准 Sharpe: {baseline:.3f}")
        print(f"  最优配置: {best['config']} (Sharpe={best['sharpe']:.3f})")
        print(f"  提升: {improvement:+.1f}%")

        if improvement > 5:
            print(f"  ✅ 情绪因子有增量，建议加入")
        elif improvement > 0:
            print(f"  ⚠️ 情绪因子增量微弱，可选加入")
        else:
            print(f"  ❌ 情绪因子无增量或有害，不建议加入")

    # 保存
    out = {"results": results, "baseline_sharpe": results[0]["sharpe"] if results else 0}
    with open(DATA_DIR / "finbert_incremental_test.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n⏱️ {time.time()-t0:.0f}秒")
    print(f"📁 结果: data/falcon/finbert_incremental_test.json")


if __name__ == "__main__":
    main()
