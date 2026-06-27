#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — Russell 2000 小盘股回测
复用V0.3引擎, 只换数据源
"""
import pandas as pd, numpy as np, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")


def compute_tech_features(df):
    """从OHLCV计算技术特征 (与V0.2一致)。"""
    df = df.sort_values("date").copy()
    
    # RSI 14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    
    # MACD histogram
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    df["macd_hist"] = macd - signal
    
    # Momentum 1-month (20 days)
    df["momentum_1m"] = df["close"].pct_change(20)
    
    # Volatility 20-day
    df["vol20"] = df["close"].pct_change().rolling(20).std() * np.sqrt(252)
    
    # Bollinger Band position
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_pos"] = (df["close"] - sma20) / std20.replace(0, np.nan)
    
    # MA alignment (5 > 20 > 60 → 1.0)
    ma5 = df["close"].rolling(5).mean()
    ma20 = df["close"].rolling(20).mean()
    ma60 = df["close"].rolling(60).mean()
    df["ma_align"] = ((ma5 > ma20).astype(float) + (ma20 > ma60).astype(float)) / 2
    
    # Return quality (fraction of positive days in last 20)
    daily_ret = df["close"].pct_change()
    df["ret_quality"] = (daily_ret > 0).rolling(20).mean()
    
    # Drawdown 60-day
    peak60 = df["close"].rolling(60).max()
    df["dd_60"] = (df["close"] - peak60) / peak60
    
    # Up/Down volume ratio
    up_vol = df["volume"].where(daily_ret > 0, 0).rolling(20).sum()
    dn_vol = df["volume"].where(daily_ret < 0, 0).rolling(20).sum()
    df["ud_vol_ratio"] = up_vol / dn_vol.replace(0, np.nan)
    
    return df


def load_russell_data():
    """加载Russell 2000数据。"""
    print("📊 加载Russell数据...")
    
    # 1. 价格数据
    with open(DATA_DIR / "russell_prices.json") as f:
        prices_raw = json.load(f)
    
    print(f"  价格: {len(prices_raw)} 只")
    
    # 转成master dataframe
    rows = []
    for ticker, bars in prices_raw.items():
        if not isinstance(bars, list) or len(bars) < 100:
            continue
        for bar in bars:
            rows.append({
                "ticker": ticker,
                "date": bar["date"],
                "open": bar["open"],
                "high": bar["high"],
                "low": bar["low"],
                "close": bar["close"],
                "volume": bar.get("volume", 0),
            })
    
    master = pd.DataFrame(rows)
    master["date"] = master["date"].astype(str)
    print(f"  行数: {len(master)}, 只数: {master['ticker'].nunique()}")
    
    # 2. 按ticker计算技术特征
    print("  计算技术特征...")
    tech_dfs = []
    for i, (ticker, group) in enumerate(master.groupby("ticker")):
        if len(group) < 60:
            continue
        tech_dfs.append(compute_tech_features(group))
        if (i+1) % 100 == 0:
            print(f"    {i+1}/{master['ticker'].nunique()}...")
    
    master = pd.concat(tech_dfs, ignore_index=True)
    print(f"  ✅ 技术特征完成: {len(master)} 行")
    
    # 3. 加载FMP数据
    data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_russell.json"),
        ("fmp_key_metrics", "fmp_metrics_russell.json"),
        ("fmp_financial_growth", "fmp_growth_russell.json"),
        ("analyst_historical", "fmp_analyst_russell.json"),
    ]:
        f = DATA_DIR / fname
        if f.exists():
            with open(f) as fh:
                data[name] = json.load(fh)
            print(f"  ✅ {name}: {len(data[name])} 只")
        else:
            data[name] = {}
            print(f"  ❌ {name}: 不存在")
    
    # Russell没有insider, dcf, price_target
    data["fmp_insider"] = {}
    data["fmp_dcf"] = {}
    data["fmp_price_target"] = {}
    
    return master, data


def main():
    t0 = time.time()
    print("=" * 100)
    print("🦅 Falcon V0.3 — Russell 2000 小盘股回测")
    print("=" * 100)
    
    master, data = load_russell_data()
    n_tickers = master["ticker"].nunique()
    print(f"\n📊 {len(master)} 行, {n_tickers} 只")
    
    # 过滤: 只保留有FMP数据的ticker
    tickers_with_fmp = set()
    for name in ["fmp_ratios_historical", "fmp_key_metrics", "fmp_financial_growth"]:
        for t, v in data.get(name, {}).items():
            if v and len(v) > 0:
                tickers_with_fmp.add(t)
    
    master = master[master["ticker"].isin(tickers_with_fmp)]
    n_filtered = master["ticker"].nunique()
    print(f"  过滤后: {n_filtered} 只 (有FMP基本面数据)")
    
    # 预计算PIT rank
    ranks_dict = precompute_pit_ranks(
        master, 
        data["fmp_ratios_historical"], 
        data["analyst_historical"],
        data["fmp_key_metrics"], 
        data["fmp_financial_growth"],
        data["fmp_insider"], 
        data["fmp_dcf"], 
        data["fmp_price_target"]
    )
    
    # 价格矩阵 + regime
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)
    
    bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])
    bear_dates = sorted([d for d in ranks_dict if "2022" in d])
    
    print(f"\n  牛市天数: {len(bull_dates)}, 熊市天数: {len(bear_dates)}")
    print(f"  2022 regime below MA200: {(1-regime_above.loc[bear_dates]).mean():.0%}")
    
    # ═══════════════════════════════════════════════════
    # 与S&P 500完全相同的权重组合
    # ═══════════════════════════════════════════════════
    weight_configs = {
        "V0.2(Fund+Ana)": {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1},
        "Pure_Fund": {"tech": 0.0, "fund_ratio": 0.5, "fund_metric": 0.3, "fund_growth": 0.2},
        "Full_FMP": {"fund_ratio": 0.3, "fund_metric": 0.2, "fund_growth": 0.15, "analyst": 0.1, "insider": 0.1, "valuation": 0.1, "tech": 0.05},
        "Analyst_Heavy": {"analyst": 0.5, "fund_ratio": 0.3, "fund_metric": 0.2},
    }
    
    strategy_configs = {
        "Fixed_10d": {"strategy": "fixed", "params": {"hold_days": 10}},
        "Fixed_30d": {"strategy": "fixed", "params": {"hold_days": 30}},
        "Signal_5d": {"strategy": "signal", "params": {"check_every": 5, "rank_threshold": 0.5}},
        "Signal_10d": {"strategy": "signal", "params": {"check_every": 10, "rank_threshold": 0.4}},
    }
    
    results = []
    total = len(weight_configs) * len(strategy_configs) * 2 * 2  # 2 SL × 2 bear_alloc
    done = 0
    
    print(f"\n🔍 测试 {len(weight_configs)} 权重 × {len(strategy_configs)} 策略 = {total} 组合")
    print(f"{'='*120}")
    
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
                            "weight": w_name, "strategy": s_name,
                            "stop_loss": sl, "bear_alloc": bear_alloc,
                            "bull_sr": bull["sharpe"], "bull_dd": bull["dd"], "bull_ret": bull["ret"],
                            "bear_sr": bear["sharpe"], "bear_dd": bear["dd"], "bear_wr": bear["wr"],
                            "bear_trades": bear["trades"],
                            "passed": passed,
                        })
                    
                    if done % 10 == 0:
                        n_pass = sum(1 for r in results if r["passed"])
                        print(f"  {done}/{total}: {n_pass} 通过")
    
    rdf = pd.DataFrame(results)
    passed = rdf[rdf["passed"] == True].sort_values("bull_sr", ascending=False)
    
    print(f"\n{'='*120}")
    print(f"📊 Russell 2000 结果: {len(passed)}/{len(rdf)} 通过 Falcon协议")
    print(f"{'='*120}")
    
    if len(passed) > 0:
        print(f"\n🏆 Top-10 (按牛市Sharpe排序):")
        for _, r in passed.head(10).iterrows():
            print(f"  {r['weight']:20} {r['strategy']:12} SL={r['stop_loss']:.0%} Bear={r['bear_alloc']:.0%} | "
                  f"Bull_SR={r['bull_sr']:.3f} DD={r['bull_dd']:.0f}% Ret={r['bull_ret']:.0f}% | "
                  f"Bear_SR={r['bear_sr']:.3f} DD={r['bear_dd']:.0f}% WR={r['bear_wr']:.0f}%")
    
    # 全量结果（即使没通过也显示）
    rdf_sorted = rdf.sort_values("bull_sr", ascending=False)
    print(f"\n📊 全量结果 Top-15:")
    print(f"{'权重':20} {'策略':12} {'SL':5} {'Bear':5} | {'牛SR':7} {'DD':6} {'Ret':7} | {'熊SR':7} {'DD':6} {'WR':5}")
    for _, r in rdf_sorted.head(15).iterrows():
        mark = "✅" if r["passed"] else "❌"
        print(f"{r['weight']:20} {r['strategy']:12} {r['stop_loss']:.0%} {r['bear_alloc']:.0%} | "
              f"{r['bull_sr']:7.3f} {r['bull_dd']:5.1f}% {r['bull_ret']:6.0f}% | "
              f"{r['bear_sr']:7.3f} {r['bear_dd']:5.1f}% {r['bear_wr']:4.0f}% {mark}")
    
    # 对比S&P 500
    print(f"\n{'='*120}")
    print("📊 大盘 vs 小盘 对比 (V0.2权重, Signal_5d, SL=-15%, Bear=50%)")
    print(f"{'='*120}")
    
    base_weights = {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1}
    base_params = {"check_every": 5, "rank_threshold": 0.5, "stop_loss": -0.15, "bear_alloc": 0.50}
    
    bull = backtest_flexible(ranks_dict, price_pivot, bull_dates, regime_above,
                            base_weights, "signal", base_params)
    bear = backtest_flexible(ranks_dict, price_pivot, bear_dates, regime_above,
                            base_weights, "signal", base_params)
    
    if bull and bear:
        print(f"  Russell 2000 小盘:")
        print(f"    牛市: Sharpe={bull['sharpe']:.3f}, DD={bull['dd']:.1f}%, Ret={bull['ret']:.0f}%")
        print(f"    熊市: Sharpe={bear['sharpe']:.3f}, DD={bear['dd']:.1f}%, WR={bear['wr']:.0f}%")
        print(f"    交易数: 牛={len(rdf)}, 熊={bear['trades']}")
        print(f"\n  S&P 500 大盘 (之前结果):")
        print(f"    牛市: Sharpe=2.349, DD=7%, Ret=+206%")
        print(f"    熊市: Sharpe=0.080, DD=9.7%, WR=50%")
    
    print(f"\n⏱️ {time.time()-t0:.0f}秒")
    
    # 保存
    rdf.to_csv(DATA_DIR / "falcon_v03_russell_results.csv", index=False)
    print(f"💾 保存: data/falcon/falcon_v03_russell_results.csv")


if __name__ == "__main__":
    main()
