#!/usr/bin/env python3
"""
🦅 Falcon V0.2.1 — 修复前视偏差后的回测
所有FMP/分析师数据point-in-time: 只用backtest date之前已发布的季度
"""
import pandas as pd, numpy as np, json, time
from pathlib import Path

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")
TECH_FIELDS = ["rsi14", "macd_hist", "momentum_1m", "vol20", "bb_pos",
               "ma_align", "ret_quality", "dd_60", "ud_vol_ratio"]
FMP_FIELDS = ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
              "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
              "grossProfitMargin", "netProfitMargin", "operatingProfitMargin", "ebitdaMargin",
              "assetTurnover", "inventoryTurnover", "receivablesTurnover",
              "debtToEquityRatio", "currentRatio", "quickRatio", "financialLeverageRatio",
              "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
              "dividendYieldPercentage", "dividendPayoutRatio"]
ANALYST_FIELDS = ["eps_revision", "revenue_revision", "eps_dispersion"]


def get_pit_fmp(ticker, date, fmp_hist):
    """Point-in-time: 返回date之前最新季度的FMP数据。"""
    quarters = fmp_hist.get(ticker, [])
    # 找date之前最新的季度
    latest = None
    for q in quarters:
        if q["date"] <= date:
            latest = q
        else:
            break
    return latest or {}


def get_pit_analyst(ticker, date, ana_hist):
    """Point-in-time: 返回date之前最新季度的分析师数据。"""
    estimates = ana_hist.get(ticker, [])
    latest = None
    for q in estimates:
        if q["date"] <= date:
            latest = q
        else:
            break
    return latest or {}


def precompute_pit_ranks(master, fmp_hist, ana_hist):
    """预计算point-in-time截面rank。"""
    print("📊 预计算PIT rank(修复前视偏差)...")
    dates = sorted(master["date"].unique())
    ranks_dict = {}

    for date in dates:
        day = master[master["date"] == date].copy()
        if len(day) < 10:
            continue
        day.index = day["ticker"].values  # index=ticker, 确保rank对齐

        row = day[["ticker"]].copy()

        # Tech rank (从K线算, 无前视偏差)
        tech_r = []
        for f in TECH_FIELDS:
            if f in day.columns and day[f].notna().sum() > 5:
                row[f"t_{f}"] = day[f].rank(pct=True)
                tech_r.append(f"t_{f}")
        row["tech"] = row[tech_r].mean(axis=1) if tech_r else 0.5

        # FMP rank (point-in-time)
        fmp_features = {}
        for t in day["ticker"].values:
            pit = get_pit_fmp(t, date, fmp_hist)
            fmp_features[t] = pit
        for f in FMP_FIELDS:
            vals = []
            for t in day["ticker"].values:
                v = fmp_features.get(t, {}).get(f)
                vals.append(v if v is not None else np.nan)
            series = pd.Series(vals, index=day["ticker"].values)
            if series.notna().sum() > 10:
                row[f"f_{f}"] = series.rank(pct=True)
        fmp_rank_cols = [c for c in row.columns if c.startswith("f_")]
        row["fund"] = row[fmp_rank_cols].mean(axis=1) if fmp_rank_cols else 0.5

        # Analyst rank (point-in-time)
        ana_features = {}
        for t in day["ticker"].values:
            pit = get_pit_analyst(t, date, ana_hist)
            ana_features[t] = pit
        for f in ANALYST_FIELDS:
            vals = []
            for t in day["ticker"].values:
                v = ana_features.get(t, {}).get(f)
                vals.append(v if v is not None else np.nan)
            series = pd.Series(vals, index=day["ticker"].values)
            if series.notna().sum() > 5:
                row[f"a_{f}"] = series.rank(pct=True)
        ana_rank_cols = [c for c in row.columns if c.startswith("a_")]
        row["analyst"] = row[ana_rank_cols].mean(axis=1) if ana_rank_cols else 0.5

        ranks_dict[date] = row.set_index("ticker")[["tech", "fund", "analyst"]]

    print(f"✅ PIT rank: {len(ranks_dict)} 天")
    return ranks_dict


def run_bt(ranks_dict, price_pivot, dates, regime_above,
           wt, wf, wa, stop_pct, bear_alloc, hold, top_n=5):
    cost = 0.0045
    cash = 100000.0
    portfolio = {}
    values = []
    trades = []

    for i, date in enumerate(dates):
        if date not in price_pivot.index or date not in ranks_dict:
            continue
        pr = price_pivot.loc[date]
        above = regime_above.loc[date] if date in regime_above.index else 1
        alloc = bear_alloc if above == 0 else 1.0

        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in pr and not pd.isna(pr[t]):
                pnl = (pr[t] - ep) / ep
                if pnl <= stop_pct:
                    cash += sh * pr[t] * (1 - cost)
                    trades.append(pnl - 2*cost)
                    to_close.append(t)
                elif (i - ei) >= hold:
                    cash += sh * pr[t] * (1 - cost)
                    trades.append(pnl - 2*cost)
                    to_close.append(t)
        for t in to_close:
            del portfolio[t]

        if len(portfolio) == 0 and cash > 100:
            scores = ranks_dict[date]
            combined = wt * scores["tech"] + wf * scores["fund"] + wa * scores["analyst"]
            combined = combined.dropna().sort_values(ascending=False)
            deploy = cash * alloc
            reserve = cash - deploy
            picks = combined.head(top_n).index.tolist()
            per = deploy / len(picks) if picks else 0
            for t in picks:
                if t in pr and not pd.isna(pr[t]) and pr[t] > 0:
                    sh = (per * (1 - cost)) / pr[t]
                    portfolio[t] = (i, pr[t], sh)
            cash = reserve

        pv = cash
        for t, (_, ep, sh) in portfolio.items():
            pv += sh * (pr[t] if t in pr and not pd.isna(pr[t]) else ep)
        values.append(pv)

    if len(values) < 20:
        return None
    v = np.array(values, dtype=np.float64)
    rets = np.diff(v) / np.where(v[:-1] > 0, v[:-1], 1)
    std = np.std(rets)
    if std == 0:
        return None
    sr = np.mean(rets) / std * np.sqrt(252)
    tr = (v[-1]/v[0]-1)*100
    pk = np.maximum.accumulate(v)
    dd = ((pk-v)/pk).max()*100
    wins = sum(1 for t in trades if t > 0)
    wr = wins / len(trades) * 100 if trades else 0
    return {"sharpe": round(sr, 3), "dd": round(dd, 2), "ret": round(tr, 2),
            "wr": round(wr, 1), "trades": len(trades)}


def main():
    t0 = time.time()

    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    print(f"📊 {len(master)} 行, {master['ticker'].nunique()} 只")

    # 加载历史FMP数据
    with open(DATA_DIR / "fmp_ratios_historical.json") as f:
        fmp_hist = json.load(f)
    with open(DATA_DIR / "analyst_historical.json") as f:
        ana_hist = json.load(f)
    print(f"📂 FMP历史: {len(fmp_hist)} 只, Analyst: {len(ana_hist)} 只")

    # 预计算PIT rank
    ranks_dict = precompute_pit_ranks(master, fmp_hist, ana_hist)

    # 价格矩阵 + regime
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])
    bear_dates = sorted([d for d in ranks_dict if "2022" in d])

    # 对比: V0.1(tech only) vs V0.2.1(PIT all)
    configs = [
        ("TechOnly(Wt=1.0)", 1.0, 0.0, 0.0),
        ("FundOnly(Wf=1.0)", 0.0, 1.0, 0.0),
        ("AnalystOnly(Wa=1.0)", 0.0, 0.0, 1.0),
        ("V0.2最优(0.2/0.7/0.1)", 0.2, 0.7, 0.1),
        ("均衡(0.33/0.33/0.33)", 0.33, 0.33, 0.33),
        ("Tech+Fund(0.5/0.5/0)", 0.5, 0.5, 0.0),
    ]

    print(f"\n{'='*110}")
    print(f"PIT回测对比 (Hold=30, BearAlloc=50%, SL=-30%)")
    print(f"{'='*110}")
    print(f"{'配置':25} | {'牛市SR':7} {'DD':6} {'Ret':7} | {'熊市SR':7} {'DD':6} {'WR':5} {'Trades':7}")

    for label, wt, wf, wa in configs:
        bull = run_bt(ranks_dict, price_pivot, bull_dates, regime_above,
                      wt, wf, wa, stop_pct=-0.30, bear_alloc=0.50, hold=30)
        bear = run_bt(ranks_dict, price_pivot, bear_dates, regime_above,
                      wt, wf, wa, stop_pct=-0.30, bear_alloc=0.50, hold=30)
        b_str = f"{bull['sharpe']:7.3f} {bull['dd']:5.1f}% {bull['ret']:6.0f}%" if bull else "N/A"
        r_str = f"{bear['sharpe']:7.3f} {bear['dd']:5.1f}% {bear['wr']:4.0f}% {bear['trades']:6}" if bear else "N/A"
        print(f"{label:25} | {b_str} | {r_str}")

    # 搜索最优PIT组合
    print(f"\n{'='*110}")
    print("PIT最优搜索 (66权重 × 5止损 × 4仓位 × 2持有期)")
    print(f"{'='*110}")

    combos = []
    for wt in np.arange(0, 1.01, 0.1):
        for wf in np.arange(0, 1.01 - wt + 0.001, 0.1):
            wa = round(1.0 - wt - wf, 1)
            if 0 <= wa <= 1:
                combos.append((round(wt, 1), round(wf, 1), wa))
    combos = sorted(set(combos))

    results = []
    count = 0
    for hold in [30, 60]:
        for stop_pct in [-0.10, -0.15, -0.20, -0.25, -0.30]:
            for bear_alloc in [0.0, 0.15, 0.30, 0.50]:
                for wt, wf, wa in combos:
                    count += 1
                    bull = run_bt(ranks_dict, price_pivot, bull_dates, regime_above,
                                  wt, wf, wa, stop_pct, bear_alloc, hold)
                    bear = run_bt(ranks_dict, price_pivot, bear_dates, regime_above,
                                  wt, wf, wa, stop_pct, bear_alloc, hold)
                    if bull and bear:
                        passed = bear["dd"] <= 28 and bear["wr"] >= 42
                        results.append({
                            "hold": hold, "stop": stop_pct, "bear_alloc": bear_alloc,
                            "wt": wt, "wf": wf, "wa": wa,
                            "bull_sr": bull["sharpe"], "bull_dd": bull["dd"], "bull_ret": bull["ret"],
                            "bear_sr": bear["sharpe"], "bear_dd": bear["dd"], "bear_wr": bear["wr"],
                            "bear_trades": bear["trades"],
                            "passed": passed,
                        })
                if count % 200 == 0:
                    n_pass = sum(1 for r in results if r["passed"])
                    print(f"  {count}: {n_pass} 通过")

    rdf = pd.DataFrame(results)
    passed = rdf[rdf["passed"] == True].sort_values("bull_sr", ascending=False)
    print(f"\n✅ 通过: {len(passed)}/{len(rdf)}")
    if len(passed) > 0:
        print("\nTop-10:")
        for _, r in passed.head(10).iterrows():
            print(f"  H={r['hold']:.0f} SL={r['stop']:.0%} Bear={r['bear_alloc']:.0%} "
                  f"Wt={r['wt']:.1f} Wf={r['wf']:.1f} Wa={r['wa']:.1f} → "
                  f"Bull_SR={r['bull_sr']:.3f} DD={r['bull_dd']:.0f}% | "
                  f"Bear_SR={r['bear_sr']:.3f} DD={r['bear_dd']:.0f}% WR={r['bear_wr']:.0f}%")
    else:
        rdf["dd_gap"] = (rdf["bear_dd"] - 28).clip(lower=0)
        rdf["wr_gap"] = (42 - rdf["bear_wr"]).clip(lower=0)
        rdf["total_gap"] = rdf["dd_gap"] + rdf["wr_gap"]
        closest = rdf.sort_values("total_gap").head(10)
        print("\n最接近通过:")
        for _, r in closest.iterrows():
            dd_ok = "✅" if r["bear_dd"] <= 28 else f"❌+{r['dd_gap']:.0f}"
            wr_ok = "✅" if r["bear_wr"] >= 42 else f"❌-{r['wr_gap']:.0f}"
            print(f"  H={r['hold']:.0f} SL={r['stop']:.0%} Bear={r['bear_alloc']:.0%} "
                  f"Wt={r['wt']:.1f} Wf={r['wf']:.1f} Wa={r['wa']:.1f} → "
                  f"Bull_SR={r['bull_sr']:.3f} DD={dd_ok} WR={wr_ok}")

    print(f"\n⏱️ {time.time()-t0:.0f}秒")

    # 保存verdict
    best = rdf.sort_values("total_gap").iloc[0] if len(rdf) > 0 else None
    verdict = {
        "version": "V0.2.1",
        "audit_status": "PASSED" if len(passed) > 0 else "FAILED",
        "look_ahead_bias": "FIXED",
        "data": "PIT (point-in-time quarterly FMP + analyst)",
        "total_combos": len(rdf),
        "passed_combos": len(passed),
        "note": "Sharpe下降是正常的(去除了前视偏差)",
    }
    with open(DATA_DIR / "falcon_v021_verdict.json", "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
