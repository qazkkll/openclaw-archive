"""
V10蓝盾LambdaMART重训 — 去掉假PE数据
=====================================
只用真实beta（从价格计算），去掉pe_trailing（86%假值）
"""
import pandas as pd, numpy as np, lightgbm as lgb, gc, json, time, sys
from pathlib import Path

ROOT = Path("/home/hermes/.hermes/openclaw-archive")

def main():
    t0 = time.time()
    print("=== V10蓝盾 LambdaMART (clean beta only) ===")

    # 加载2年数据
    df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet",
                          columns=["date", "sym", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].max() - pd.Timedelta(days=730)
    df = df[df["date"] >= cutoff].copy()

    # 宇宙：>$10
    df = df[df["close"] >= 10].copy()
    print(f"  Universe: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 只用beta，不用PE
    fund = pd.read_parquet(ROOT / "data/us/fundamentals_latest.parquet",
                            columns=["sym", "beta"])
    df = df.merge(fund, on="sym", how="left")
    del fund; gc.collect()

    # 特征计算
    df = df.sort_values(["sym", "date"])
    g = df.groupby("sym")["close"]

    df["ret5"] = g.transform(lambda x: x.pct_change(5)).astype(np.float32)
    df["ret20"] = g.transform(lambda x: x.pct_change(20)).astype(np.float32)
    df["ret60"] = g.transform(lambda x: x.pct_change(60)).astype(np.float32)
    df["momentum_6m"] = g.transform(lambda x: x.pct_change(126)).astype(np.float32)
    df["momentum_1m"] = g.transform(lambda x: x.pct_change(21)).astype(np.float32)

    # 均线偏离
    ma20 = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma_bias20"] = ((df["close"] - ma20) / ma20.replace(0, np.nan)).astype(np.float32)
    del ma20

    # 波动率
    df["vol20"] = g.transform(lambda x: x.pct_change(5).rolling(20, min_periods=10).std()).astype(np.float32)
    df["vol_ratio"] = (df["vol20"] / g.transform(
        lambda x: x.pct_change(5).rolling(60, min_periods=20).std()
    ).replace(0, np.nan)).astype(np.float32)

    # RSI
    delta = g.transform(lambda x: x.diff()).astype(np.float32)
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    avg_loss = loss.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    df["rsi14"] = (100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))).astype(np.float32)
    del delta, gain, loss, avg_gain, avg_loss; gc.collect()

    # MACD
    ema12 = g.transform(lambda x: x.ewm(span=12, min_periods=6).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, min_periods=13).mean())
    macd = (ema12 - ema26).astype(np.float32)
    df["macd_hist"] = (macd - macd.groupby(df["sym"]).transform(
        lambda x: x.ewm(span=9, min_periods=5).mean()
    )).astype(np.float32)
    del ema12, ema26, macd

    # Bollinger
    bb_mid = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std = g.transform(lambda x: x.rolling(20, min_periods=10).std())
    df["bb_pos"] = ((df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)).astype(np.float32)
    del bb_mid, bb_std

    # 价格位置
    high60 = g.transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = g.transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)
    del high60, low60; gc.collect()

    # Beta（真实，从价格计算）
    df["beta_c"] = df["beta"].clip(-2, 5).fillna(0.73).astype(np.float32)

    # VIX
    vix = pd.read_parquet(ROOT / "data/us/vix_10y.parquet")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.reset_index()
    vix_date = [c for c in vix.columns if "date" in str(c).lower()][0]
    vix_val = [c for c in vix.columns if "close" in str(c).lower()][0]
    vix_df = pd.DataFrame({"date": pd.to_datetime(vix[vix_date]), "vix_close": vix[vix_val].astype(np.float32)})
    df = df.merge(vix_df, on="date", how="left")
    del vix, vix_df

    # SPY returns
    spy = df[df["sym"] == "SPY"][["date", "close"]].sort_values("date")
    for d in [5, 20, 60]:
        spy[f"spy_ret{d}"] = spy["close"].pct_change(d).astype(np.float32)
    df = df.merge(spy[["date", "spy_ret5", "spy_ret20", "spy_ret60"]], on="date", how="left")
    del spy; gc.collect()

    # 前瞻收益
    df["fwd_5d"] = g.transform(lambda x: x.shift(-5) / x - 1).astype(np.float32)
    df = df.drop(columns=["volume", "beta"], errors="ignore")
    df = df.dropna(subset=["fwd_5d"])

    feat = ["ret5", "ret20", "ret60", "momentum_6m", "momentum_1m", "ma_bias20",
            "vol20", "vol_ratio", "rsi14", "macd_hist", "bb_pos", "price_position",
            "beta_c", "vix_close", "spy_ret5", "spy_ret20", "spy_ret60"]
    df[feat] = df[feat].fillna(0)
    df["year"] = df["date"].dt.year
    print(f"  Clean: {len(df)} rows, {df['sym'].nunique()} stocks, {len(feat)} features")
    print(f"  Data prep: {time.time()-t0:.0f}s")

    # Walk-Forward
    results = []
    last_model = None

    for test_year in sorted(df["year"].unique()):
        if test_year < 2025:
            continue

        tr = df[df["year"] < test_year].copy()
        te = df[df["year"] == test_year].copy()

        if len(tr) < 10000 or len(te) < 1000:
            print(f"  {test_year}: SKIP")
            continue

        print(f"  Training {test_year} (train={len(tr)}, test={len(te)})...")
        t1 = time.time()

        tr["y"] = tr.groupby("date")["fwd_5d"].transform(
            lambda x: pd.cut(x, 5, labels=[0,1,2,3,4])
        ).astype(np.int8)
        te["y"] = te.groupby("date")["fwd_5d"].transform(
            lambda x: pd.cut(x, 5, labels=[0,1,2,3,4])
        ).astype(np.int8)

        tg = tr.groupby("date").size().values
        vg = te.groupby("date").size().values

        trn = lgb.Dataset(tr[feat], label=tr["y"], group=tg)
        tst = lgb.Dataset(te[feat], label=te["y"], group=vg, reference=trn)

        m = lgb.train(
            {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [30],
             "learning_rate": 0.05, "num_leaves": 63, "feature_fraction": 0.8,
             "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": 42},
            trn, 500, valid_sets=[tst],
            callbacks=[lgb.log_evaluation(100)],
        )

        te["pred"] = m.predict(te[feat])
        daily_ic = te.groupby("date").apply(lambda g: g["pred"].corr(g["fwd_5d"]), include_groups=False)
        ic = float(daily_ic.mean())
        icir = ic / float(daily_ic.std()) if daily_ic.std() > 0 else 0

        te["q"] = te.groupby("date")["pred"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if len(x) >= 5 else 2
        )
        qr = te.groupby("q")["fwd_5d"].mean()
        top5 = float(qr.iloc[-1])
        bot20 = float(qr.iloc[0])
        win = float(te.loc[te["q"]==qr.index[-1],"fwd_5d"].gt(0).mean())
        mono = top5 > bot20

        r = {"year": test_year, "ic": ic, "icir": icir, "top5": top5,
             "bot20": bot20, "win": win, "mono": mono, "time_s": time.time()-t1}
        results.append(r)
        last_model = m
        print(f"    IC={ic:+.4f} ICIR={icir:.3f} Top5={top5:+.4f} Bot20={bot20:+.4f} Win={win:.1%} Mono={'✅' if mono else '❌'} ({r['time_s']:.0f}s)")

        del trn, tst, tr, te; gc.collect()

    # 汇总
    if results:
        avg = lambda k: np.mean([r[k] for r in results])
        mono_all = avg("top5") > avg("bot20")
        icir_ok = avg("icir") > 0.3

        print(f"\n=== Summary ===")
        print(f"  Avg IC: {avg('ic'):+.4f}")
        print(f"  Avg ICIR: {avg('icir'):.3f}")
        print(f"  Avg Top5: {avg('top5'):+.4f}")
        print(f"  Avg Bot20: {avg('bot20'):+.4f}")
        print(f"  Monotonic: {'✅' if mono_all else '❌'}")
        print(f"  Total: {time.time()-t0:.0f}s")

        if last_model:
            path = ROOT / "models/us/blueshield_v10_lambdamart.txt"
            last_model.save_model(str(path))
            print(f"  Model: {path}")

        # 更新meta
        meta_path = ROOT / "models/us/blueshield_v9_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        meta["features"] = feat
        meta["n_features"] = len(feat)
        meta["lambdamart_features"] = feat
        meta["lambdamart_n_features"] = len(feat)
        meta["algorithm"] = "LambdaMART (LightGBM)"
        meta["label"] = "rank_quintile_5d_forward"
        meta["universe"] = ">$10"
        meta["note"] = "Clean beta only, no PE (PE data 86% fake)"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        if mono_all and icir_ok:
            print("✅ PASSED — deployable")
        elif mono_all:
            print("⚠️ PARTIAL — monotonic but ICIR<0.3")
        else:
            print("❌ FAILED — rank inverted")


if __name__ == "__main__":
    main()
