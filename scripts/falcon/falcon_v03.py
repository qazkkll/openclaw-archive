#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — 统一回测框架
支持 S&P 500 / Russell 2000 / 两者混合
用法:
  python3 falcon_v03.py --universe spx     # 只跑大盘
  python3 falcon_v03.py --universe r2k     # 只跑小盘
  python3 falcon_v03.py --universe both    # 分别跑+Hybrid对比
"""
import pandas as pd, numpy as np, json, time, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")

# ═══════════════════════════════════════════════════
# 技术特征计算 (R2K需要, SPX已有预计算)
# ═══════════════════════════════════════════════════
def compute_tech_features(df):
    """从OHLCV计算技术特征。"""
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


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════
def load_spx():
    """加载S&P 500数据 (预计算特征)。"""
    print("📊 加载 S&P 500...")
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
    data["fmp_insider"] = {}
    data["fmp_dcf"] = {}
    data["fmp_price_target"] = {}
    n = master["ticker"].nunique()
    print(f"  ✅ {n} 只, {len(master)} 行")
    return master, data, n


def load_r2k():
    """加载Russell 2000数据 (从OHLCV算技术特征)。"""
    print("📊 加载 Russell 2000...")
    with open(DATA_DIR / "russell_prices.json") as f:
        prices_raw = json.load(f)

    rows = []
    for ticker, bars in prices_raw.items():
        if not isinstance(bars, list) or len(bars) < 100:
            continue
        for bar in bars:
            rows.append({
                "ticker": ticker, "date": bar["date"],
                "open": bar["open"], "high": bar["high"],
                "low": bar["low"], "close": bar["close"],
                "volume": bar.get("volume", 0),
            })
    master = pd.DataFrame(rows)
    master["date"] = master["date"].astype(str)

    # 计算技术特征
    tech_dfs = []
    for i, (ticker, group) in enumerate(master.groupby("ticker")):
        if len(group) < 60:
            continue
        tech_dfs.append(compute_tech_features(group))
    master = pd.concat(tech_dfs, ignore_index=True)

    # 过滤: 只保留有FMP数据的
    data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_russell.json"),
        ("analyst_historical", "fmp_analyst_russell.json"),
        ("fmp_key_metrics", "fmp_metrics_russell.json"),
        ("fmp_financial_growth", "fmp_growth_russell.json"),
    ]:
        f = DATA_DIR / fname
        data[name] = json.load(open(f)) if f.exists() else {}
    data["fmp_insider"] = {}
    data["fmp_dcf"] = {}
    data["fmp_price_target"] = {}

    tickers_with_fmp = set()
    for name in ["fmp_ratios_historical", "fmp_key_metrics", "fmp_financial_growth"]:
        for t, v in data.get(name, {}).items():
            if v and len(v) > 0:
                tickers_with_fmp.add(t)
    master = master[master["ticker"].isin(tickers_with_fmp)]
    n = master["ticker"].nunique()
    print(f"  ✅ {n} 只, {len(master)} 行 (有FMP数据)")
    return master, data, n


def run_universe(name, master, data, n_tickers):
    """跑一个universe的全量回测。"""
    ranks_dict = precompute_pit_ranks(
        master, data["fmp_ratios_historical"], data["analyst_historical"],
        data["fmp_key_metrics"], data["fmp_financial_growth"],
        data["fmp_insider"], data["fmp_dcf"], data["fmp_price_target"]
    )

    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)

    bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])
    bear_dates = sorted([d for d in ranks_dict if "2022" in d])

    below_pct = (1 - regime_above.loc[bear_dates]).mean() if bear_dates else 0
    print(f"  牛市{len(bull_dates)}天, 熊市{len(bear_dates)}天, 2022 below MA200: {below_pct:.0%}")

    # ── 权重组合 ──
    weight_configs = {
        "Fund+Ana(V0.2)": {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1},
        "Pure_Fund":      {"tech": 0.0, "fund_ratio": 0.5, "fund_metric": 0.3, "fund_growth": 0.2},
        "Full_FMP":       {"fund_ratio": 0.3, "fund_metric": 0.2, "fund_growth": 0.15, "analyst": 0.1, "insider": 0.1, "valuation": 0.1, "tech": 0.05},
        "Analyst_Heavy":  {"analyst": 0.5, "fund_ratio": 0.3, "fund_metric": 0.2},
    }

    # ── 策略组合 ──
    strategy_configs = {
        "Fixed_10d": {"strategy": "fixed", "params": {"hold_days": 10}},
        "Fixed_30d": {"strategy": "fixed", "params": {"hold_days": 30}},
        "Signal_5d": {"strategy": "signal", "params": {"check_every": 5, "rank_threshold": 0.5}},
        "Signal_10d": {"strategy": "signal", "params": {"check_every": 10, "rank_threshold": 0.4}},
    }

    results = []
    total = len(weight_configs) * len(strategy_configs) * 2 * 2
    done = 0

    for w_name, weights in weight_configs.items():
        for s_name, s_config in strategy_configs.items():
            for sl in [-0.10, -0.15]:
                for bear_alloc in [0.30, 0.50]:
                    params = dict(s_config["params"])
                    params["stop_loss"] = sl
                    params["bear_alloc"] = bear_alloc

                    bull = backtest_flexible(ranks_dict, price_pivot, bull_dates, regime_above,
                                            weights, s_config["strategy"], params)
                    bear = backtest_flexible(ranks_dict, price_pivot, bear_dates, regime_above,
                                            weights, s_config["strategy"], params)
                    done += 1
                    if bull and bear:
                        passed = bear["dd"] <= 28 and bear["wr"] >= 42
                        results.append({
                            "universe": name, "weight": w_name, "strategy": s_name,
                            "stop_loss": sl, "bear_alloc": bear_alloc,
                            "bull_sr": bull["sharpe"], "bull_dd": bull["dd"], "bull_ret": bull["ret"],
                            "bear_sr": bear["sharpe"], "bear_dd": bear["dd"], "bear_wr": bear["wr"],
                            "bear_trades": bear["trades"], "passed": passed,
                        })

    rdf = pd.DataFrame(results)
    passed = rdf[rdf["passed"]].sort_values("bull_sr", ascending=False)

    print(f"\n{'='*100}")
    print(f"📊 {name} ({n_tickers}只): {len(passed)}/{len(rdf)} 通过 Falcon协议")
    print(f"{'='*100}")

    if len(passed) > 0:
        print(f"\n🏆 Top-5:")
        print(f"{'权重':18} {'策略':10} {'SL':5} {'Bear':5} | {'牛SR':7} {'DD':6} {'Ret':6} | {'熊SR':7} {'DD':6} {'WR':5}")
        for _, r in passed.head(5).iterrows():
            print(f"{r['weight']:18} {r['strategy']:10} {r['stop_loss']:.0%} {r['bear_alloc']:.0%} | "
                  f"{r['bull_sr']:7.3f} {r['bull_dd']:5.1f}% {r['bull_ret']:5.0f}% | "
                  f"{r['bear_sr']:7.3f} {r['bear_dd']:5.1f}% {r['bear_wr']:4.0f}%")

    return rdf, ranks_dict, price_pivot, regime_above, bull_dates, bear_dates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", choices=["spx", "r2k", "both"], default="both")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 100)
    print("🦅 Falcon V0.3 — 统一回测框架")
    print("=" * 100)

    all_results = {}

    if args.universe in ("spx", "both"):
        master, data, n = load_spx()
        rdf, ranks, pp, regime, bull, bear = run_universe("SPX", master, data, n)
        all_results["SPX"] = {"rdf": rdf, "ranks": ranks, "pp": pp, "regime": regime, "bull": bull, "bear": bear}

    if args.universe in ("r2k", "both"):
        master, data, n = load_r2k()
        rdf, ranks, pp, regime, bull, bear = run_universe("R2K", master, data, n)
        all_results["R2K"] = {"rdf": rdf, "ranks": ranks, "pp": pp, "regime": regime, "bull": bull, "bear": bear}

    # ═══════════════════════════════════════════════════
    # 双universe对比
    # ═══════════════════════════════════════════════════
    if len(all_results) == 2:
        print(f"\n{'='*100}")
        print("📊 SPX vs R2K 对比 (各universe最优)")
        print(f"{'='*100}")

        for uname, res in all_results.items():
            rdf = res["rdf"]
            passed = rdf[rdf["passed"]]
            if len(passed) > 0:
                best = passed.sort_values("bull_sr", ascending=False).iloc[0]
                print(f"\n  {uname} 最优:")
                print(f"    {best['weight']} + {best['strategy']}, SL={best['stop_loss']:.0%}, Bear={best['bear_alloc']:.0%}")
                print(f"    牛市: SR={best['bull_sr']:.3f}, DD={best['bull_dd']:.1f}%, Ret={best['bull_ret']:.0f}%")
                print(f"    熊市: SR={best['bear_sr']:.3f}, DD={best['bear_dd']:.1f}%, WR={best['bear_wr']:.0f}%")

        # ── OOS检查: 2024 H2 样本外 ──
        print(f"\n{'='*100}")
        print("📊 OOS验证: 2024 H2 样本外测试")
        print(f"{'='*100}")

        for uname, res in all_results.items():
            oos_dates = sorted([d for d in res["ranks"] if d >= "2024-07-01"])
            if len(oos_dates) < 20:
                print(f"  {uname}: 数据不足 ({len(oos_dates)}天)")
                continue

            # 用各universe最优权重
            rdf = res["rdf"]
            passed = rdf[rdf["passed"]]
            if len(passed) == 0:
                passed = rdf.sort_values("bull_sr", ascending=False).head(1)

            best = passed.sort_values("bull_sr", ascending=False).iloc[0]
            weights = {
                "Fund+Ana(V0.2)": {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1},
                "Pure_Fund":      {"tech": 0.0, "fund_ratio": 0.5, "fund_metric": 0.3, "fund_growth": 0.2},
                "Full_FMP":       {"fund_ratio": 0.3, "fund_metric": 0.2, "fund_growth": 0.15, "analyst": 0.1, "insider": 0.1, "valuation": 0.1, "tech": 0.05},
                "Analyst_Heavy":  {"analyst": 0.5, "fund_ratio": 0.3, "fund_metric": 0.2},
            }[best["weight"]]

            strategy_map = {
                "Fixed_10d": ("fixed", {"hold_days": 10}),
                "Fixed_30d": ("fixed", {"hold_days": 30}),
                "Signal_5d": ("signal", {"check_every": 5, "rank_threshold": 0.5}),
                "Signal_10d": ("signal", {"check_every": 10, "rank_threshold": 0.4}),
            }
            strat_name, strat_params = strategy_map[best["strategy"]]
            strat_params["stop_loss"] = best["stop_loss"]
            strat_params["bear_alloc"] = 1.0  # OOS期间不区分牛熊

            oos = backtest_flexible(res["ranks"], res["pp"], oos_dates, res["regime"],
                                    weights, strat_name, strat_params)
            if oos:
                is_pass = "✅" if oos["dd"] <= 28 and oos["wr"] >= 42 else "❌"
                print(f"  {uname} OOS (2024H2): SR={oos['sharpe']:.3f}, DD={oos['dd']:.1f}%, "
                      f"WR={oos['wr']:.0f}%, Ret={oos['ret']:.0f}% {is_pass}")
            else:
                print(f"  {uname} OOS: 无足够数据")

    print(f"\n⏱️ {time.time()-t0:.0f}秒")

    # 保存
    if all_results:
        combined = pd.concat([r["rdf"] for r in all_results.values()], ignore_index=True)
        combined.to_csv(DATA_DIR / "falcon_v03_unified.csv", index=False)
        print(f"💾 保存: data/falcon/falcon_v03_unified.csv")


if __name__ == "__main__":
    main()
