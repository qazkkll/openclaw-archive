#!/usr/bin/env python3
"""
🦅 Falcon 全链路现实性审计
从理想回测 → 实盘现实, 逐层叠加成本, 找到真实alpha
"""
import pandas as pd, numpy as np, json, time
from pathlib import Path

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")


def futu_cost_per_share(price):
    """Futu美股每股成本: 佣金$0.0049 + 平台费$0.005, 最低$1.99/笔"""
    per_share = 0.0049 + 0.005
    min_fee = 1.99
    return max(min_fee, per_share * 100) / (price * 100)


def futu_sell_extra():
    """卖出额外: SEC fee 0.00278%"""
    return 0.0000278


def precompute_pit_ranks(master, fmp_hist, ana_hist, metrics_hist, growth_hist):
    """PIT rank: 只用Ratios+Metrics+Growth+Analyst (100%覆盖, 无稀疏问题)"""
    print("📊 预计算PIT rank (4组因子, 100%覆盖)...")
    dates = sorted(master["date"].unique())
    ranks_dict = {}

    RATIO_FIELDS = ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
                    "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
                    "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
                    "ebitdaMargin", "assetTurnover", "inventoryTurnover",
                    "receivablesTurnover", "debtToEquityRatio", "currentRatio",
                    "quickRatio", "financialLeverageRatio",
                    "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
                    "dividendYieldPercentage", "dividendPayoutRatio"]
    METRIC_FIELDS = ["earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
                     "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
                     "returnOnCapitalEmployed", "returnOnInvestedCapital",
                     "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
                     "cashConversionCycle", "capexToRevenue",
                     "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
                     "netDebtToEBITDA", "operatingReturnOnAssets"]
    GROWTH_FIELDS = ["revenueGrowth", "grossProfitGrowth", "ebitgrowth",
                     "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
                     "freeCashFlowGrowth", "fiveYRevenueGrowthPerShare",
                     "threeYRevenueGrowthPerShare", "assetGrowth", "bookValueperShareGrowth"]
    ANALYST_FIELDS = ["eps_revision", "revenue_revision", "eps_dispersion"]
    TECH_FIELDS = ["rsi14", "macd_hist", "momentum_1m", "vol20", "bb_pos",
                   "ma_align", "ret_quality", "dd_60", "ud_vol_ratio"]

    for date in dates:
        day = master[master["date"] == date].copy()
        if len(day) < 10:
            continue
        day.index = day["ticker"].values
        row = day[["ticker"]].copy()

        # Tech
        tech_r = []
        for f in TECH_FIELDS:
            if f in day.columns and day[f].notna().sum() > 5:
                row[f"t_{f}"] = day[f].rank(pct=True)
                tech_r.append(f"t_{f}")
        row["tech"] = row[tech_r].mean(axis=1) if tech_r else 0.5

        # Ratios
        for f in RATIO_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                for q in fmp_hist.get(t, []):
                    if q.get("date", "") <= date:
                        vals[t] = q.get(f)
            if len(vals) > 10:
                row[f"r_{f}"] = pd.Series(vals).rank(pct=True)

        # Metrics
        for f in METRIC_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                for q in metrics_hist.get(t, []):
                    if q.get("date", "") <= date:
                        vals[t] = q.get(f)
            if len(vals) > 10:
                row[f"m_{f}"] = pd.Series(vals).rank(pct=True)

        # Growth
        for f in GROWTH_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                for q in growth_hist.get(t, []):
                    if q.get("date", "") <= date:
                        vals[t] = q.get(f)
            if len(vals) > 10:
                row[f"g_{f}"] = pd.Series(vals).rank(pct=True)

        # Analyst
        for f in ANALYST_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                for q in ana_hist.get(t, []):
                    if q.get("date", "") <= date:
                        vals[t] = q.get(f)
            if len(vals) > 5:
                row[f"a_{f}"] = pd.Series(vals).rank(pct=True)

        r_cols = [c for c in row.columns if c.startswith("r_")]
        m_cols = [c for c in row.columns if c.startswith("m_")]
        g_cols = [c for c in row.columns if c.startswith("g_")]
        a_cols = [c for c in row.columns if c.startswith("a_")]

        row["fund"] = row[r_cols].mean(axis=1) if r_cols else 0.5
        row["metric"] = row[m_cols].mean(axis=1) if m_cols else 0.5
        row["growth"] = row[g_cols].mean(axis=1) if g_cols else 0.5
        row["analyst"] = row[a_cols].mean(axis=1) if a_cols else 0.5

        ranks_dict[date] = row.set_index("ticker")[["tech", "fund", "metric", "growth", "analyst"]]

    print(f"✅ PIT rank: {len(ranks_dict)} 天")
    return ranks_dict


def backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                       dates, regime_above, weights, execution_mode,
                       stop_loss=-0.15, bear_alloc=0.50, hold_days=30, top_n=5,
                       gap_filter=0.05):
    """
    执行模式:
      "ideal"    — T close买入/卖出 (当前回测, 0滑点)
      "open"     — T+1 open市价单
      "vwap"     — T+1 VWAP
      "vwap_filtered" — T+1 VWAP + 跳空>gap_filter放弃
      "limit"    — T+1 目标价: 跳空小→open, 跳空大→(open+close)/2, 太大→放弃
    """
    cash = 100000.0
    portfolio = {}
    values = []
    trades = []
    skipped_gaps = 0
    rebalance_count = 0

    def get_scores(date):
        if date not in ranks_dict:
            return None
        r = ranks_dict[date]
        combined = sum(w * r[f] for f, w in weights.items() if f in r.columns)
        return combined.dropna().sort_values(ascending=False)

    def exec_price(ticker, date_i, action="buy"):
        """获取执行价格。"""
        if date_i >= len(dates):
            return None
        date = dates[date_i]

        if execution_mode == "ideal":
            if date in price_pivot.index and ticker in price_pivot.columns:
                p = price_pivot.loc[date, ticker]
                return p if not pd.isna(p) else None

        elif execution_mode == "open":
            if date in open_pivot.index and ticker in open_pivot.columns:
                p = open_pivot.loc[date, ticker]
                return p if not pd.isna(p) else None

        elif execution_mode == "vwap":
            if date in vwap_pivot.index and ticker in vwap_pivot.columns:
                p = vwap_pivot.loc[date, ticker]
                if not pd.isna(p):
                    return p
            if date in open_pivot.index and ticker in open_pivot.columns:
                p = open_pivot.loc[date, ticker]
                return p if not pd.isna(p) else None

        elif execution_mode == "vwap_filtered":
            # 检查跳空
            prev_date = dates[date_i - 1] if date_i > 0 else None
            if prev_date and prev_date in price_pivot.index and date in open_pivot.index:
                if ticker in price_pivot.columns and ticker in open_pivot.columns:
                    prev_close = price_pivot.loc[prev_date, ticker]
                    today_open = open_pivot.loc[date, ticker]
                    if not pd.isna(prev_close) and not pd.isna(today_open) and prev_close > 0:
                        gap = abs((today_open - prev_close) / prev_close)
                        if gap > gap_filter:
                            return None  # 跳空太大, 放弃
            if date in vwap_pivot.index and ticker in vwap_pivot.columns:
                p = vwap_pivot.loc[date, ticker]
                if not pd.isna(p):
                    return p
            if date in open_pivot.index and ticker in open_pivot.columns:
                p = open_pivot.loc[date, ticker]
                return p if not pd.isna(p) else None

        elif execution_mode == "limit":
            prev_date = dates[date_i - 1] if date_i > 0 else None
            if prev_date and prev_date in price_pivot.index and date in open_pivot.index:
                if ticker in price_pivot.columns and ticker in open_pivot.columns:
                    prev_close = price_pivot.loc[prev_date, ticker]
                    today_open = open_pivot.loc[date, ticker]
                    if not pd.isna(prev_close) and not pd.isna(today_open) and prev_close > 0:
                        gap = (today_open - prev_close) / prev_close
                        if abs(gap) > 0.10:
                            return None  # 跳空>10%, 放弃
                        elif abs(gap) > 0.03:
                            # 跳空3-10%, 用open和prev_close的中间价(limit order)
                            return (today_open + prev_close) / 2
                        else:
                            # 跳空小, 用open买
                            return today_open
            if date in open_pivot.index and ticker in open_pivot.columns:
                p = open_pivot.loc[date, ticker]
                return p if not pd.isna(p) else None

        return None

    for i, date in enumerate(dates):
        if date not in price_pivot.index:
            continue
        pr = price_pivot.loc[date]
        above = regime_above.loc[date] if date in regime_above.index else 1
        alloc = bear_alloc if above == 0 else 1.0

        # ── 止损: 用当天收盘价判断, 用当天收盘价执行(止损是市价单) ──
        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in pr and not pd.isna(pr[t]):
                pnl = (pr[t] - ep) / ep
                if pnl <= stop_loss:
                    sell_p = exec_price(t, i, "sell")
                    if sell_p is None:
                        sell_p = pr[t]  # fallback to close
                    futu_sell = futu_cost_per_share(sell_p) + futu_sell_extra()
                    cash += sh * sell_p * (1 - futu_sell)
                    trades.append({"pnl": (sell_p - ep) / ep, "reason": "止损", "date": date})
                    to_close.append(t)
        for t in to_close:
            del portfolio[t]

        # ── 调仓: 固定hold_days ──
        sell_tickers = []
        for t, (ei, ep, sh) in list(portfolio.items()):
            if (i - ei) >= hold_days:
                sell_tickers.append(t)

        for t in sell_tickers:
            if t in portfolio:
                ei, ep, sh = portfolio.pop(t)
                sell_p = exec_price(t, i, "sell")
                if sell_p is None:
                    sell_p = pr.get(t, ep)
                if not pd.isna(sell_p):
                    futu_sell = futu_cost_per_share(sell_p) + futu_sell_extra()
                    cash += sh * sell_p * (1 - futu_sell)
                    pnl = (sell_p - ep) / ep
                    trades.append({"pnl": pnl, "reason": "调仓", "date": date})

        # ── 买入: 信号日T, 执行日T+1 ──
        if len(portfolio) == 0 and cash > 100:
            scores = get_scores(date)
            if scores is not None:
                deploy = cash * alloc
                reserve = cash - deploy
                picks = scores.head(top_n).index.tolist()

                # 执行日是下一个交易日
                exec_i = i + 1
                if exec_i < len(dates):
                    actual_buys = 0
                    per = deploy / len(picks) if picks else 0
                    for t in picks:
                        buy_p = exec_price(t, exec_i, "buy")
                        if buy_p is not None and buy_p > 0:
                            futu_buy = futu_cost_per_share(buy_p)
                            sh = (per * (1 - futu_buy)) / buy_p
                            portfolio[t] = (exec_i, buy_p, sh)
                            actual_buys += 1
                        else:
                            skipped_gaps += 1

                    if actual_buys > 0:
                        cash = reserve + (len(picks) - actual_buys) * per
                    else:
                        cash = deploy + reserve  # 全部跳空, 不买
                    rebalance_count += 1

        # ── 净值: 用收盘价估值 ──
        pv = cash
        for t, (_, ep, sh) in portfolio.items():
            p = pr.get(t, ep)
            pv += sh * (p if not pd.isna(p) else ep)
        values.append(pv)

    if len(values) < 20:
        return None

    v = np.array(values, dtype=np.float64)
    rets = np.diff(v) / np.where(v[:-1] > 0, v[:-1], 1)
    std = np.std(rets)
    if std == 0:
        return None

    sr = np.mean(rets) / std * np.sqrt(252)
    tr = (v[-1] / v[0] - 1) * 100
    pk = np.maximum.accumulate(v)
    dd = ((pk - v) / pk).max() * 100
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / len(trades) * 100 if trades else 0

    return {
        "sharpe": round(sr, 3), "dd": round(dd, 2), "ret": round(tr, 2),
        "wr": round(wr, 1), "trades": len(trades), "rebalances": rebalance_count,
        "skipped_gaps": skipped_gaps,
    }


def main():
    t0 = time.time()

    # ── 加载数据 ──
    print("📂 加载数据...")
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)

    with open(DATA_DIR / "fmp_ratios_historical.json") as f:
        fmp_hist = json.load(f)
    with open(DATA_DIR / "analyst_historical.json") as f:
        ana_hist = json.load(f)
    with open(DATA_DIR / "fmp_key_metrics.json") as f:
        metrics_hist = json.load(f)
    with open(DATA_DIR / "fmp_financial_growth.json") as f:
        growth_hist = json.load(f)

    print(f"  {len(master)} 行, {master['ticker'].nunique()} 只")

    # ── 预计算PIT rank ──
    ranks_dict = precompute_pit_ranks(master, fmp_hist, ana_hist, metrics_hist, growth_hist)

    # ── 价格矩阵 ──
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    open_pivot = master.pivot_table(index="date", columns="ticker", values="open").sort_index()
    vwap_pivot = master.pivot_table(index="date", columns="ticker", values="vwap").sort_index()

    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])
    bear_dates = sorted([d for d in ranks_dict if "2022" in d])

    # ── 权重组合 ──
    weight_configs = {
        "Fund70_Ana20_Met10": {"fund": 0.7, "analyst": 0.2, "metric": 0.1},
        "Fund50_Met20_Ana20_Gr10": {"fund": 0.5, "metric": 0.2, "analyst": 0.2, "growth": 0.1},
        "Fund60_Ana20_Gr20": {"fund": 0.6, "analyst": 0.2, "growth": 0.2},
        "Fund40_Ana30_Gr20_Met10": {"fund": 0.4, "analyst": 0.3, "growth": 0.2, "metric": 0.1},
    }

    # ── 执行模式 ──
    exec_modes = ["ideal", "open", "vwap", "vwap_filtered", "limit"]

    # ═══════════════════════════════════════════════════
    # ROUND 1: 5种执行模式 × 4种权重 (固定hold=30, SL=-15%)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("ROUND 1: 执行模式现实性对比 (Hold=30, SL=-15%, Bear=50%)")
    print(f"{'='*130}")

    header = f"{'权重':30} {'执行模式':15} | {'牛SR':7} {'DD':6} {'Ret':7} | {'熊SR':7} {'DD':6} {'WR':5} {'Tr':5} {'跳空跳过':8}"
    print(header)
    print("-" * 130)

    for w_name, weights in weight_configs.items():
        for mode in exec_modes:
            bull = backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                                      bull_dates, regime_above, weights, mode,
                                      stop_loss=-0.15, bear_alloc=0.50, hold_days=30)
            bear = backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                                      bear_dates, regime_above, weights, mode,
                                      stop_loss=-0.15, bear_alloc=0.50, hold_days=30)
            if bull and bear:
                b = f"{bull['sharpe']:7.3f} {bull['dd']:5.1f}% {bull['ret']:6.0f}%"
                r = f"{bear['sharpe']:7.3f} {bear['dd']:5.1f}% {bear['wr']:4.0f}% {bear['trades']:5} {bear['skipped_gaps']:8}"
                passed = "✅" if bear["dd"] <= 28 and bear["wr"] >= 42 else "❌"
                print(f"{w_name:30} {mode:15} | {b} | {r} {passed}")

    # ═══════════════════════════════════════════════════
    # ROUND 2: 最现实模式(limit) × 多种hold × 多种SL
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("ROUND 2: limit模式网格搜索 (最现实执行)")
    print(f"{'='*130}")

    best_weights = {"fund": 0.7, "analyst": 0.2, "metric": 0.1}
    results = []

    for hold in [10, 20, 30, 60]:
        for sl in [-0.10, -0.15, -0.20, -0.25]:
            for bear_alloc in [0.30, 0.50]:
                for gap_filter in [0.03, 0.05, 0.08, 0.10]:
                    bull = backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                                              bull_dates, regime_above, best_weights, "limit",
                                              stop_loss=sl, bear_alloc=bear_alloc, hold_days=hold,
                                              gap_filter=gap_filter)
                    bear = backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                                              bear_dates, regime_above, best_weights, "limit",
                                              stop_loss=sl, bear_alloc=bear_alloc, hold_days=hold,
                                              gap_filter=gap_filter)
                    if bull and bear:
                        passed = bear["dd"] <= 28 and bear["wr"] >= 42
                        results.append({
                            "hold": hold, "sl": sl, "bear_alloc": bear_alloc,
                            "gap_filter": gap_filter,
                            "bull_sr": bull["sharpe"], "bull_dd": bull["dd"], "bull_ret": bull["ret"],
                            "bear_sr": bear["sharpe"], "bear_dd": bear["dd"], "bear_wr": bear["wr"],
                            "bear_trades": bear["trades"], "bear_skipped": bear["skipped_gaps"],
                            "passed": passed,
                        })

    rdf = pd.DataFrame(results)
    passed = rdf[rdf["passed"]].sort_values("bull_sr", ascending=False)

    print(f"\n通过: {len(passed)}/{len(rdf)}")
    if len(passed) > 0:
        print(f"\n{'Hold':5} {'SL':5} {'Bear':5} {'Gap':5} | {'牛SR':7} {'DD':6} {'Ret':7} | {'熊SR':7} {'DD':6} {'WR':5} {'Tr':5} {'Skip':5}")
        for _, r in passed.head(10).iterrows():
            print(f"{r['hold']:5.0f} {r['sl']:.0%} {r['bear_alloc']:.0%} {r['gap_filter']:.0%} | "
                  f"{r['bull_sr']:7.3f} {r['bull_dd']:5.1f}% {r['bull_ret']:6.0f}% | "
                  f"{r['bear_sr']:7.3f} {r['bear_dd']:5.1f}% {r['bear_wr']:4.0f}% {r['bear_trades']:5.0f} {r['bear_skipped']:5.0f}")
    else:
        rdf["gap"] = (rdf["bear_dd"] - 28).clip(lower=0) + (42 - rdf["bear_wr"]).clip(lower=0)
        closest = rdf.sort_values("gap").head(10)
        print(f"\n最接近通过:")
        for _, r in closest.iterrows():
            dd_ok = "✅" if r["bear_dd"] <= 28 else f"❌+{r['bear_dd']-28:.0f}"
            wr_ok = "✅" if r["bear_wr"] >= 42 else f"❌-{42-r['bear_wr']:.0f}"
            print(f"  H={r['hold']:.0f} SL={r['sl']:.0%} Bear={r['bear_alloc']:.0%} Gap={r['gap_filter']:.0%} → "
                  f"Bull_SR={r['bull_sr']:.3f} DD={dd_ok} WR={wr_ok} Skip={r['bear_skipped']:.0f}")

    # ═══════════════════════════════════════════════════
    # ROUND 3: alpha衰减分析
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*130}")
    print("ROUND 3: Alpha衰减分析 (理想→现实)")
    print(f"{'='*130}")

    configs = [
        ("理想(无成本)", "ideal", -0.15, 0.50, 30, 0.05),
        ("T+1 Open", "open", -0.15, 0.50, 30, 0.05),
        ("T+1 VWAP", "vwap", -0.15, 0.50, 30, 0.05),
        ("VWAP+5%过滤", "vwap_filtered", -0.15, 0.50, 30, 0.05),
        ("Limit订单", "limit", -0.15, 0.50, 30, 0.05),
        ("Limit+严过滤", "limit", -0.15, 0.50, 30, 0.03),
    ]

    print(f"\n{'场景':20} | {'牛SR':7} {'DD':6} {'Ret':7} | {'熊SR':7} {'DD':6} {'WR':5} | {'Sharpe衰减':10}")
    ideal_bull_sr = None
    for label, mode, sl, ba, hold, gf in configs:
        bull = backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                                  bull_dates, regime_above, best_weights, mode,
                                  stop_loss=sl, bear_alloc=ba, hold_days=hold, gap_filter=gf)
        bear = backtest_realistic(ranks_dict, price_pivot, open_pivot, vwap_pivot,
                                  bear_dates, regime_above, best_weights, mode,
                                  stop_loss=sl, bear_alloc=ba, hold_days=hold, gap_filter=gf)
        if bull and bear:
            if ideal_bull_sr is None:
                ideal_bull_sr = bull["sharpe"]
            decay = (1 - bull["sharpe"] / ideal_bull_sr) * 100 if ideal_bull_sr else 0
            print(f"{label:20} | {bull['sharpe']:7.3f} {bull['dd']:5.1f}% {bull['ret']:6.0f}% | "
                  f"{bear['sharpe']:7.3f} {bear['dd']:5.1f}% {bear['wr']:4.0f}% | {decay:+.1f}%")

    print(f"\n⏱️ {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
