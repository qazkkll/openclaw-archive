#!/usr/bin/env python3
"""
🦅 Falcon V0.2 — 市场状态感知版
核心: 熊市(MA200下方)自动缩仓
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


def precompute(master):
    dates = sorted(master["date"].unique())
    ranks_dict = {}
    for date in dates:
        day = master[master["date"] == date].copy()
        if len(day) < 10:
            continue
        row = day[["ticker"]].copy()
        tech_r, fmp_r, ana_r = [], [], []
        for f in TECH_FIELDS:
            if f in day.columns and day[f].notna().sum() > 5:
                row[f"t_{f}"] = day[f].rank(pct=True)
                tech_r.append(f"t_{f}")
        row["tech"] = row[tech_r].mean(axis=1) if tech_r else 0.5
        for f in FMP_FIELDS:
            if f in day.columns and day[f].notna().sum() > 10:
                row[f"f_{f}"] = day[f].rank(pct=True)
                fmp_r.append(f"f_{f}")
        row["fund"] = row[fmp_r].mean(axis=1) if fmp_r else 0.5
        for f in ANALYST_FIELDS:
            if f in day.columns and day[f].notna().sum() > 5:
                row[f"a_{f}"] = day[f].rank(pct=True)
                ana_r.append(f"a_{f}")
        row["analyst"] = row[ana_r].mean(axis=1) if ana_r else 0.5
        ranks_dict[date] = row.set_index("ticker")[["tech", "fund", "analyst"]]
    return ranks_dict


def compute_regime(master):
    """计算市场regime: 等权市场MA200 + 波动率。"""
    price_p = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_p.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    mkt_vol60 = mkt_ret.rolling(60, min_periods=30).std() * np.sqrt(252)
    regime = pd.DataFrame({
        "above_ma200": (mkt_price > mkt_ma200).astype(int),
        "vol60": mkt_vol60,
    }, index=mkt_price.index)
    return regime


def backtest(ranks_dict, price_pivot, dates, regime, wt, wf, wa,
             top_n=5, hold_days=60, stop_loss=-0.10, cost=0.0045,
             bear_alloc=0.3, vol_scale=True):
    """
    bear_alloc: 熊市(MA200下)仓位比例
    vol_scale: 高波动时进一步缩仓
    """
    cash = 100000.0
    cash_deployed = 100000.0  # 总资金
    portfolio = {}
    values = []
    trades = []
    skipped = 0

    for i, date in enumerate(dates):
        if date not in price_pivot.index or date not in ranks_dict:
            continue
        pr = price_pivot.loc[date]

        # regime
        r = regime.loc[date] if date in regime.index else None
        is_bear = r is not None and r["above_ma200"] == 0
        vol = r["vol60"] if r is not None and not pd.isna(r.get("vol60", np.nan)) else 0.15

        # 动态仓位: 熊市缩仓, 高波动进一步缩
        alloc = bear_alloc if is_bear else 1.0
        if vol_scale and vol > 0.25:
            alloc *= 0.5  # 极高波动再砍半

        # 止损/到期
        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in pr and not pd.isna(pr[t]):
                pnl = (pr[t] - ep) / ep
                if pnl <= stop_loss:
                    cash += sh * pr[t] * (1 - cost)
                    trades.append(pnl - 2*cost)
                    to_close.append(t)
                elif (i - ei) >= hold_days:
                    cash += sh * pr[t] * (1 - cost)
                    trades.append(pnl - 2*cost)
                    to_close.append(t)
        for t in to_close:
            del portfolio[t]

        # 轮换(只有非持仓时)
        if len(portfolio) == 0 and cash > 100:
            scores = ranks_dict[date]
            combined = wt * scores["tech"] + wf * scores["fund"] + wa * scores["analyst"]
            combined = combined.dropna().sort_values(ascending=False)

            # alloc决定投入比例
            deploy = cash * alloc
            reserve = cash - deploy

            picks = combined.head(top_n).index.tolist()
            per = deploy / len(picks) if picks else 0
            for t in picks:
                if t in pr and not pd.isna(pr[t]) and pr[t] > 0:
                    sh = (per * (1 - cost)) / pr[t]
                    portfolio[t] = (i, pr[t], sh)
            cash = reserve  # 未部署的现金保留

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

    sharpe = np.mean(rets) / std * np.sqrt(252)
    total_ret = (v[-1] / v[0] - 1) * 100
    peak = np.maximum.accumulate(v)
    max_dd = ((peak - v) / peak).max() * 100
    win = sum(1 for t in trades if t > 0)
    win_rate = win / len(trades) * 100 if trades else 0

    return {"sharpe": round(sharpe, 3), "max_dd": round(max_dd, 2),
            "total_return": round(total_ret, 2), "win_rate": round(win_rate, 1),
            "trades": len(trades)}


def main():
    t0 = time.time()

    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    print(f"📊 {len(master)} 行, {master['ticker'].nunique()} 只")

    ranks_dict = precompute(master)
    regime = compute_regime(master)
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()

    bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])
    bear_dates = sorted([d for d in ranks_dict if "2022" in d])
    full_dates = sorted(ranks_dict.keys())

    print(f"📊 {len(ranks_dict)} 天, regime: {len(regime)} 天")
    print(f"   熊市天数(2022 MA200下方): {(regime.loc[regime.index.str.startswith('2022'), 'above_ma200'] == 0).sum()}")

    # Grid: 权重 × hold × bear_alloc
    combos = []
    for wt in np.arange(0, 1.01, 0.1):
        for wf in np.arange(0, 1.01 - wt + 0.001, 0.1):
            wa = round(1.0 - wt - wf, 1)
            if 0 <= wa <= 1:
                combos.append((round(wt, 1), round(wf, 1), wa))
    combos = sorted(set(combos))

    for hold in [30, 60]:
        for bear_alloc in [0.0, 0.15, 0.30, 0.50]:
            for vol_scale in [True, False]:
                results = []
                for wt, wf, wa in combos:
                    r = backtest(ranks_dict, price_pivot, bull_dates, regime,
                                 wt, wf, wa, top_n=5, hold_days=hold,
                                 stop_loss=-0.10, cost=0.0045,
                                 bear_alloc=bear_alloc, vol_scale=vol_scale)
                    if r:
                        results.append({"Wt": wt, "Wf": wf, "Wa": wa, **r})

                if not results:
                    continue

                results.sort(key=lambda x: x["sharpe"], reverse=True)
                best = results[0]

                # 熊市压测
                bear = backtest(ranks_dict, price_pivot, bear_dates, regime,
                                best["Wt"], best["Wf"], best["Wa"],
                                top_n=5, hold_days=hold,
                                stop_loss=-0.10, cost=0.0045,
                                bear_alloc=bear_alloc, vol_scale=vol_scale)

                vs = "VOL" if vol_scale else "noV"
                ba = f"{bear_alloc:.0%}"
                status = ""
                if bear:
                    passed = bear["max_dd"] <= 28 and bear["win_rate"] >= 42
                    status = "✅" if passed else f"❌ DD={bear['max_dd']:.0f}% WR={bear['win_rate']:.0f}%"
                else:
                    status = "❌ NO DATA"

                print(f"H={hold} BearAlloc={ba} {vs}: Bull_SR={best['sharpe']:.3f} DD={best['max_dd']:.0f}% Ret={best['total_return']:.0f}% | Bear: {status} | Wt={best['Wt']:.1f} Wf={best['Wf']:.1f} Wa={best['Wa']:.1f}")

    print(f"\n⏱️ {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
