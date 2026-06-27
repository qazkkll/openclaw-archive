#!/usr/bin/env python3
"""Falcon V0.2 — 最优组合搜索(修正动态止损)"""
import pandas as pd, numpy as np
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

master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
master["date"] = master["date"].astype(str)

dates_all = sorted(master["date"].unique())
ranks_dict = {}
vol_dict = {}
for date in dates_all:
    day = master[master["date"] == date].copy()
    if len(day) < 10:
        continue
    row = day[["ticker"]].copy()
    for f in TECH_FIELDS:
        if f in day.columns and day[f].notna().sum() > 5:
            row[f"t_{f}"] = day[f].rank(pct=True)
    row["tech"] = row[[c for c in row.columns if c.startswith("t_")]].mean(axis=1)
    for f in FMP_FIELDS:
        if f in day.columns and day[f].notna().sum() > 10:
            row[f"f_{f}"] = day[f].rank(pct=True)
    row["fund"] = row[[c for c in row.columns if c.startswith("f_")]].mean(axis=1)
    for f in ANALYST_FIELDS:
        if f in day.columns and day[f].notna().sum() > 5:
            row[f"a_{f}"] = day[f].rank(pct=True)
    row["analyst"] = row[[c for c in row.columns if c.startswith("a_")]].mean(axis=1)
    ranks_dict[date] = row.set_index("ticker")[["tech", "fund", "analyst"]]
    if "vol20" in day.columns:
        vol_dict[date] = day.set_index("ticker")["vol20"]

price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
mkt_price = (1 + mkt_ret).cumprod()
mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
regime_above = (mkt_price > mkt_ma200).astype(int)

bear_dates = sorted([d for d in ranks_dict if "2022" in d])
bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])


def run_bt(dates, wt, wf, wa, stop_pct, bear_alloc, hold, top_n=5):
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
        vols = vol_dict.get(date, pd.Series(dtype=float))

        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in pr and not pd.isna(pr[t]):
                pnl = (pr[t] - ep) / ep
                # Dynamic stop: stop_pct scaled by stock's vol20 relative to market average
                tv = vols.get(t, 0.25) if not vols.empty else 0.25
                # Normalize: avg vol ~25% → stop = stop_pct; high vol → wider stop
                sl = stop_pct * (tv / 0.25)
                sl = max(-0.40, min(-0.05, sl))  # cap [-40%, -5%]
                if pnl <= sl:
                    cash += sh * pr[t] * (1 - cost)
                    trades.append({"pnl": pnl, "reason": "止损"})
                    to_close.append(t)
                elif (i - ei) >= hold:
                    cash += sh * pr[t] * (1 - cost)
                    trades.append({"pnl": pnl, "reason": "到期"})
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
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / len(trades) * 100 if trades else 0
    return {"sharpe": round(sr, 3), "dd": round(dd, 2), "ret": round(tr, 2),
            "wr": round(wr, 1), "trades": len(trades), "wins": wins}


# Grid: weights × stop × bear_alloc × hold
weight_combos = []
for wt in np.arange(0, 1.01, 0.1):
    for wf in np.arange(0, 1.01 - wt + 0.001, 0.1):
        wa = round(1.0 - wt - wf, 1)
        if 0 <= wa <= 1:
            weight_combos.append((round(wt, 1), round(wf, 1), wa))
weight_combos = sorted(set(weight_combos))

print(f"🔍 搜索: {len(weight_combos)} 权重 × 5止损 × 4仓位 × 2持有期")

results = []
count = 0
for hold in [30, 60]:
    for stop_pct in [-0.10, -0.15, -0.20, -0.25, -0.30]:
        for bear_alloc in [0.0, 0.15, 0.30, 0.50]:
            for wt, wf, wa in weight_combos:
                count += 1
                bull = run_bt(bull_dates, wt, wf, wa, stop_pct, bear_alloc, hold)
                bear = run_bt(bear_dates, wt, wf, wa, stop_pct, bear_alloc, hold)
                if bull and bear:
                    passed = bear["dd"] <= 28 and bear["wr"] >= 42
                    results.append({
                        "hold": hold, "stop": stop_pct, "bear_alloc": bear_alloc,
                        "wt": wt, "wf": wf, "wa": wa,
                        "bull_sr": bull["sharpe"], "bull_dd": bull["dd"], "bull_ret": bull["ret"],
                        "bear_sr": bear["sharpe"], "bear_dd": bear["dd"], "bear_wr": bear["wr"],
                        "bear_trades": bear["trades"], "bear_wins": bear["wins"],
                        "passed": passed,
                    })
            if count % 100 == 0:
                n_pass = sum(1 for r in results if r["passed"])
                print(f"  {count} 组合, {n_pass} 通过")

# Sort by bull_sharpe among passed, or by bear_wr among all
rdf = pd.DataFrame(results)
passed = rdf[rdf["passed"] == True].sort_values("bull_sr", ascending=False)
print(f"\n{'='*100}")
print(f"✅ 通过熊市压测(DD≤28%, WR≥42%): {len(passed)} 组合")
if len(passed) > 0:
    print(f"\nTop-10 通过:")
    for _, r in passed.head(10).iterrows():
        print(f"  H={r['hold']:.0f} SL={r['stop']:.0%} Bear={r['bear_alloc']:.0%} Wt={r['wt']:.1f} Wf={r['wf']:.1f} Wa={r['wa']:.1f} → Bull_SR={r['bull_sr']:.3f} Bull_DD={r['bull_dd']:.0f}% Bear_SR={r['bear_sr']:.3f} Bear_DD={r['bear_dd']:.0f}% Bear_WR={r['bear_wr']:.0f}%")
else:
    # Show closest to passing
    rdf["dd_gap"] = rdf["bear_dd"] - 28
    rdf["wr_gap"] = 42 - rdf["bear_wr"]
    rdf["total_gap"] = rdf["dd_gap"].clip(lower=0) + rdf["wr_gap"].clip(lower=0)
    closest = rdf.sort_values("total_gap").head(10)
    print(f"\n最接近通过的10组:")
    for _, r in closest.iterrows():
        dd_ok = "✅" if r["bear_dd"] <= 28 else f"❌+{r['dd_gap']:.0f}"
        wr_ok = "✅" if r["bear_wr"] >= 42 else f"❌-{r['wr_gap']:.0f}"
        print(f"  H={r['hold']:.0f} SL={r['stop']:.0%} Bear={r['bear_alloc']:.0%} Wt={r['wt']:.1f} Wf={r['wf']:.1f} Wa={r['wa']:.1f} → Bull_SR={r['bull_sr']:.3f} DD={dd_ok} WR={wr_ok} Bear_SR={r['bear_sr']:.3f}")

print(f"\n⏱️ 总搜索: {count} 组合")
