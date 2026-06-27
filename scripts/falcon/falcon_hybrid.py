#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — 简单Hybrid: SPX+R2K 固定比例加权
不做网格搜索, 不搜参数, 0过拟合风险
"""
import pandas as pd, numpy as np, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")


def compute_tech_features(df):
    df = df.sort_values("date").copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_hist"] = macd - signal
    df["momentum_1m"] = df["close"].pct_change(20)
    df["vol20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - sma20) / std20.replace(0, np.nan)
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    ma60 = df["close"].rolling(60).mean()
    df["ma_align"] = ((ma5 > ma20).astype(float) + (ma20 > ma60).astype(float)) / 2
    daily_ret = df["close"].pct_change()
    df["ret_quality"] = (daily_ret > 0).rolling(20).mean()
    peak60 = df["close"].rolling(60).max()
    df["dd_60"] = (df["close"] - peak60) / peak60
    up_vol = df["volume"].where(daily_ret > 0, 0).rolling(20).sum()
    dn_vol = df["volume"].where(daily_ret < 0, 0).rolling(20).sum()
    df["ud_vol_ratio"] = up_vol / dn_vol.replace(0, np.nan)
    return df


def get_equity_curve(ranks_dict, price_pivot, dates, regime_above, weights, strategy, params):
    """返回日净值序列 (供混合用)。"""
    from falcon_v03_engine import futu_cost
    cash = 100000.0
    portfolio = {}
    values = []
    stop_loss = params.get("stop_loss", -0.15)
    bear_alloc = params.get("bear_alloc", 0.50)

    def get_scores(date):
        if date not in ranks_dict:
            return None
        r = ranks_dict[date]
        combined = sum(w * r[f] for f, w in weights.items() if f in r.columns)
        return combined.dropna().sort_values(ascending=False)

    for i, date in enumerate(dates):
        if date not in price_pivot.index:
            values.append(cash + sum(ep * sh for _, (_, ep, sh) in portfolio.items()))
            continue
        pr = price_pivot.loc[date]
        above = regime_above.loc[date] if date in regime_above.index else 1
        alloc = bear_alloc if above == 0 else 1.0

        # 止损
        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in pr and not pd.isna(pr[t]):
                pnl = (pr[t] - ep) / ep
                if pnl <= stop_loss:
                    cash += sh * pr[t] * (1 - futu_cost(pr[t], "sell"))
                    to_close.append(t)
        for t in to_close:
            del portfolio[t]

        # 调仓
        if strategy == "fixed":
            hold_days = params.get("hold_days", 30)
            sell_tickers = [t for t, (ei, _, _) in portfolio.items() if (i - ei) >= hold_days]
            for t in sell_tickers:
                if t in portfolio and t in pr and not pd.isna(pr[t]):
                    _, ep, sh = portfolio.pop(t)
                    cash += sh * pr[t] * (1 - futu_cost(pr[t], "sell"))

            if len(portfolio) == 0 and cash > 100:
                scores = get_scores(date)
                if scores is not None:
                    deploy = cash * alloc
                    picks = scores.head(5).index.tolist()
                    per = deploy / len(picks) if picks else 0
                    for t in picks:
                        if t in pr and not pd.isna(pr[t]) and pr[t] > 0:
                            sh = (per * (1 - futu_cost(pr[t], "buy"))) / pr[t]
                            portfolio[t] = (i, pr[t], sh)
                    cash = cash - deploy

        pv = cash
        for t, (_, ep, sh) in portfolio.items():
            pv += sh * (pr[t] if t in pr and not pd.isna(pr[t]) else ep)
        values.append(pv)

    return np.array(values, dtype=np.float64)


def curve_metrics(v, label=""):
    """从净值序列计算指标。"""
    rets = np.diff(v) / np.where(v[:-1] > 0, v[:-1], 1)
    std = np.std(rets)
    if std == 0:
        return None
    sr = np.mean(rets) / std * np.sqrt(252)
    tr = (v[-1] / v[0] - 1) * 100
    pk = np.maximum.accumulate(v)
    dd = ((pk - v) / pk).max() * 100
    return {"label": label, "sharpe": round(sr, 3), "dd": round(dd, 1), "ret": round(tr, 1)}


def main():
    t0 = time.time()
    print("=" * 100)
    print("🦅 Falcon V0.3 — 简单Hybrid (SPX 80/20/50 + R2K)")
    print("=" * 100)

    # ── 加载SPX ──
    print("\n📊 加载 SPX...")
    spx_master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    spx_master["date"] = spx_master["date"].astype(str)
    spx_data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_historical.json"),
        ("analyst_historical", "analyst_historical.json"),
        ("fmp_key_metrics", "fmp_key_metrics.json"),
        ("fmp_financial_growth", "fmp_financial_growth.json"),
    ]:
        f = DATA_DIR / fname
        spx_data[name] = json.load(open(f)) if f.exists() else {}
    spx_data["fmp_insider"] = {}
    spx_data["fmp_dcf"] = {}
    spx_data["fmp_price_target"] = {}
    print(f"  ✅ {spx_master['ticker'].nunique()} 只")

    # ── 加载R2K ──
    print("📊 加载 R2K...")
    with open(DATA_DIR / "russell_prices.json") as f:
        prices_raw = json.load(f)
    rows = []
    for ticker, bars in prices_raw.items():
        if not isinstance(bars, list) or len(bars) < 100:
            continue
        for bar in bars:
            rows.append({"ticker": ticker, "date": bar["date"], "open": bar["open"],
                        "high": bar["high"], "low": bar["low"], "close": bar["close"],
                        "volume": bar.get("volume", 0)})
    r2k_master = pd.DataFrame(rows)
    r2k_master["date"] = r2k_master["date"].astype(str)
    tech_dfs = []
    for ticker, group in r2k_master.groupby("ticker"):
        if len(group) < 60:
            continue
        tech_dfs.append(compute_tech_features(group))
    r2k_master = pd.concat(tech_dfs, ignore_index=True)
    r2k_data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_russell.json"),
        ("analyst_historical", "fmp_analyst_russell.json"),
        ("fmp_key_metrics", "fmp_metrics_russell.json"),
        ("fmp_financial_growth", "fmp_growth_russell.json"),
    ]:
        f = DATA_DIR / fname
        r2k_data[name] = json.load(open(f)) if f.exists() else {}
    r2k_data["fmp_insider"] = {}
    r2k_data["fmp_dcf"] = {}
    r2k_data["fmp_price_target"] = {}
    tickers_with_fmp = set()
    for name in ["fmp_ratios_historical", "fmp_key_metrics", "fmp_financial_growth"]:
        for t, v in r2k_data.get(name, {}).items():
            if v and len(v) > 0:
                tickers_with_fmp.add(t)
    r2k_master = r2k_master[r2k_master["ticker"].isin(tickers_with_fmp)]
    print(f"  ✅ {r2k_master['ticker'].nunique()} 只")

    # ── 预计算rank ──
    print("\n📊 预计算 SPX rank...")
    spx_ranks = precompute_pit_ranks(spx_master, spx_data["fmp_ratios_historical"],
        spx_data["analyst_historical"], spx_data["fmp_key_metrics"],
        spx_data["fmp_financial_growth"], spx_data["fmp_insider"],
        spx_data["fmp_dcf"], spx_data["fmp_price_target"])

    print("📊 预计算 R2K rank...")
    r2k_ranks = precompute_pit_ranks(r2k_master, r2k_data["fmp_ratios_historical"],
        r2k_data["analyst_historical"], r2k_data["fmp_key_metrics"],
        r2k_data["fmp_financial_growth"], r2k_data["fmp_insider"],
        r2k_data["fmp_dcf"], r2k_data["fmp_price_target"])

    # ── 价格矩阵 ──
    spx_pp = spx_master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    r2k_pp = r2k_master.pivot_table(index="date", columns="ticker", values="close").sort_index()

    # ── Regime (分别用各自proxy) ──
    spx_ret = spx_pp.pct_change(fill_method=None).mean(axis=1)
    spx_price = (1 + spx_ret).cumprod()
    spx_ma200 = spx_price.rolling(200, min_periods=100).mean()
    spx_regime = (spx_price > spx_ma200).astype(int)

    r2k_ret = r2k_pp.pct_change(fill_method=None).mean(axis=1)
    r2k_price = (1 + r2k_ret).cumprod()
    r2k_ma200 = r2k_price.rolling(200, min_periods=100).mean()
    r2k_regime = (r2k_price > r2k_ma200).astype(int)

    bull_dates = sorted(set(d for d in spx_ranks if "2023" in d or "2024" in d) & set(d for d in r2k_ranks if "2023" in d or "2024" in d))
    bear_dates = sorted(set(d for d in spx_ranks if "2022" in d) & set(d for d in r2k_ranks if "2022" in d))
    all_dates = sorted(set(bull_dates) | set(bear_dates))

    # ── 各自最优配置 ──
    spx_weights = {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1}  # Fund+Ana
    spx_params = {"hold_days": 30, "stop_loss": -0.15, "bear_alloc": 0.50}

    r2k_weights = {"tech": 0.0, "fund_ratio": 0.5, "fund_metric": 0.3, "fund_growth": 0.2}  # Pure_Fund
    r2k_params = {"hold_days": 10, "stop_loss": -0.15, "bear_alloc": 0.30}

    # ── 生成净值曲线 ──
    print("\n📊 生成净值曲线...")

    spx_bull_v = get_equity_curve(spx_ranks, spx_pp, bull_dates, spx_regime, spx_weights, "fixed", spx_params)
    spx_bear_v = get_equity_curve(spx_ranks, spx_pp, bear_dates, spx_regime, spx_weights, "fixed", spx_params)
    r2k_bull_v = get_equity_curve(r2k_ranks, r2k_pp, bull_dates, r2k_regime, r2k_weights, "fixed", r2k_params)
    r2k_bear_v = get_equity_curve(r2k_ranks, r2k_pp, bear_dates, r2k_regime, r2k_weights, "fixed", r2k_params)

    spx_all_v = np.concatenate([spx_bear_v, spx_bull_v])
    r2k_all_v = np.concatenate([r2k_bear_v, r2k_bull_v])

    # ── Hybrid混合 ──
    print(f"\n{'='*100}")
    print("📊 简单Hybrid结果 (固定比例, 0参数搜索)")
    print(f"{'='*100}")

    configs = [
        ("纯 SPX", 1.0, 0.0),
        ("纯 R2K", 0.0, 1.0),
        ("Hybrid 90/10", 0.9, 0.1),
        ("Hybrid 80/20", 0.8, 0.2),
        ("Hybrid 70/30", 0.7, 0.3),
        ("Hybrid 50/50", 0.5, 0.5),
    ]

    print(f"\n{'配置':16} | {'全样本':24} | {'牛市(23-24)':24} | {'熊市(22)':24}")
    print(f"{'':16} | {'SR':>6} {'DD':>6} {'Ret':>7} | {'SR':>6} {'DD':>6} {'Ret':>7} | {'SR':>6} {'DD':>6} {'Ret':>7}")

    for label, w_spx, w_r2k in configs:
        all_v = w_spx * spx_all_v + w_r2k * r2k_all_v
        bull_v = w_spx * spx_bull_v + w_r2k * r2k_bull_v
        bear_v = w_spx * spx_bear_v + w_r2k * r2k_bear_v

        m_all = curve_metrics(all_v, label)
        m_bull = curve_metrics(bull_v, label)
        m_bear = curve_metrics(bear_v, label)

        if m_all and m_bull and m_bear:
            print(f"{label:16} | {m_all['sharpe']:6.3f} {m_all['dd']:5.1f}% {m_all['ret']:6.0f}% | "
                  f"{m_bull['sharpe']:6.3f} {m_bull['dd']:5.1f}% {m_bull['ret']:6.0f}% | "
                  f"{m_bear['sharpe']:6.3f} {m_bear['dd']:5.1f}% {m_bear['ret']:6.0f}%")

    # ── 相关性分析 ──
    print(f"\n{'='*100}")
    print("📊 SPX vs R2K 日收益相关性")
    print(f"{'='*100}")

    spx_daily = np.diff(spx_all_v) / spx_all_v[:-1]
    r2k_daily = np.diff(r2k_all_v) / r2k_all_v[:-1]
    min_len = min(len(spx_daily), len(r2k_daily))
    corr = np.corrcoef(spx_daily[:min_len], r2k_daily[:min_len])[0, 1]
    print(f"  全样本相关性: {corr:.3f}")

    spx_bull_daily = np.diff(spx_bull_v) / spx_bull_v[:-1]
    r2k_bull_daily = np.diff(r2k_bull_v) / r2k_bull_v[:-1]
    min_bull = min(len(spx_bull_daily), len(r2k_bull_daily))
    corr_bull = np.corrcoef(spx_bull_daily[:min_bull], r2k_bull_daily[:min_bull])[0, 1]

    spx_bear_daily = np.diff(spx_bear_v) / spx_bear_v[:-1]
    r2k_bear_daily = np.diff(r2k_bear_v) / r2k_bear_v[:-1]
    min_bear = min(len(spx_bear_daily), len(r2k_bear_daily))
    corr_bear = np.corrcoef(spx_bear_daily[:min_bear], r2k_bear_daily[:min_bear])[0, 1]

    print(f"  牛市相关性:   {corr_bull:.3f}")
    print(f"  熊市相关性:   {corr_bear:.3f}")

    if corr < 0.7:
        print(f"  → 相关性低, Hybrid有分散化价值 ✅")
    else:
        print(f"  → 相关性高, Hybrid分散化效果有限 ⚠️")

    print(f"\n⏱️ {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
