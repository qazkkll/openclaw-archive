"""
LambdaMART GPU训练 — 与CPU版完全相同的特征和数据处理
50x加速，特征一致
"""
import pandas as pd, numpy as np, lightgbm as lgb, gc, json, time, sys
from pathlib import Path

ROOT = Path("/home/hermes/.hermes/openclaw-archive")

GPU_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [30],
    "learning_rate": 0.05, "num_leaves": 63, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1, "seed": 42,
    "device": "cuda",
}

def main():
    model_type = sys.argv[1] if len(sys.argv) > 1 else "blueshield_v10"
    print(f"=== GPU LambdaMART Training: {model_type} ===")
    t0 = time.time()

    # 1. 加载全量数据
    print("Loading full dataset...")
    df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet",
                          columns=["date", "sym", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Full: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 2. 取最近2年
    cutoff = df["date"].max() - pd.Timedelta(days=730)
    df = df[df["date"] >= cutoff].copy()
    print(f"  2yr subset: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 3. 宇宙过滤
    if "blueshield" in model_type:
        df = df[df["close"] >= 10].copy()
    else:
        df = df[(df["close"] >= 1) & (df["close"] <= 10)].copy()
    print(f"  Universe: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 4. 基本面
    fund = pd.read_parquet(ROOT / "data/us/fundamentals_latest.parquet",
                            columns=["sym", "beta"])
    df = df.merge(fund, on="sym", how="left")
    del fund; gc.collect()

    # 5. 特征计算（与CPU版完全一致）
    print("Computing features (full)...")
    df = df.sort_values(["sym", "date"])
    g = df.groupby("sym")["close"]

    # 动量
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
    df["vol20"] = df.groupby("sym")["ret5"].transform(
        lambda x: x.rolling(20, min_periods=10).std()
    ).astype(np.float32)
    df["vol_ratio"] = (df["vol20"] / df.groupby("sym")["ret5"].transform(
        lambda x: x.rolling(60, min_periods=20).std()
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
    high60 = df.groupby("sym")["close"].transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = df.groupby("sym")["close"].transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)
    del high60, low60; gc.collect()

    # 基本面
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

    # 清理
    df = df.drop(columns=["volume", "beta"], errors="ignore")
    gc.collect()

    # 6. 清理
    df = df.dropna(subset=["fwd_5d"])
    feat = ["ret5", "ret20", "ret60", "momentum_6m", "momentum_1m", "ma_bias20",
            "vol20", "vol_ratio", "rsi14", "macd_hist", "bb_pos", "price_position",
            "beta_c", "vix_close", "spy_ret5", "spy_ret20", "spy_ret60"]
    df[feat] = df[feat].fillna(0)
    df["year"] = df["date"].dt.year

    print(f"  Clean: {len(df)} rows, {df['sym'].nunique()} stocks")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
    print(f"  Data prep: {time.time() - t0:.0f}s")

    # 7. Walk-Forward训练
    results = {}
    last_model = None

    for test_year in sorted(df["year"].unique()):
        if test_year < 2024:
            continue
        tr = df[df["year"] < test_year]
        te = df[df["year"] == test_year]
        if len(tr) < 10000 or len(te) < 10000:
            print(f"  {test_year}: SKIP (tr={len(tr)}, te={len(te)})")
            continue

        # LambdaMART标签
        tr = tr.copy()
        te = te.copy()
        tr["y"] = tr.groupby("date")["fwd_5d"].transform(
            lambda x: pd.cut(x, 5, labels=[0, 1, 2, 3, 4])
        ).astype(np.int8)
        te["y"] = te.groupby("date")["fwd_5d"].transform(
            lambda x: pd.cut(x, 5, labels=[0, 1, 2, 3, 4])
        ).astype(np.int8)

        tg = tr.groupby("date").size().values
        vg = te.groupby("date").size().values

        trn = lgb.Dataset(tr[feat], label=tr["y"], group=tg)
        tst = lgb.Dataset(te[feat], label=te["y"], group=vg, reference=trn)

        print(f"  Training {test_year} (train={len(tr)}, test={len(te)})...")
        ts = time.time()
        m = lgb.train(GPU_PARAMS, trn, 300, valid_sets=[tst],
                      callbacks=[lgb.log_evaluation(50)])

        # 评估
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
        mono = bool(qr.is_monotonic_increasing)
        win = float((daily_ic > 0).mean())
        elapsed = time.time() - ts

        print(f"    IC={ic:+.4f} ICIR={icir:.3f} Top5={top5:+.4f} Bot20={bot20:+.4f} Win={win:.1%} Mono={'OK' if mono else 'FAIL'} ({elapsed:.0f}s)")
        results[str(test_year)] = {"ic": ic, "icir": icir, "top5": top5, "bot20": bot20, "mono": mono, "win": win, "time_s": elapsed}
        last_model = m

    # Save
    if last_model:
        model_path = ROOT / f"models/us/{model_type}_lambdamart.txt"
        last_model.save_model(str(model_path))
        report = {"model": model_type, "features": feat, "n_features": len(feat),
                  "results": results, "gpu": True, "time_total_s": time.time()-t0}
        with open(ROOT / f"data/lambdamart_{model_type}_full.json", "w") as f:
            json.dump(report, f, indent=2)

    # Summary
    ics = [v["icir"] for v in results.values()]
    avg_icir = np.mean(ics) if ics else 0
    avg_mono = all(v["mono"] for v in results.values()) if results else False
    print(f"\n=== Summary ===")
    for yr, v in results.items():
        print(f"  {yr}: IC={v['ic']:+.4f} ICIR={v['icir']:.3f} Top5={v['top5']:+.4f} Bot20={v['bot20']:+.4f} Mono={'OK' if v['mono'] else 'FAIL'}")
    print(f"  Avg ICIR: {avg_icir:.3f}")
    print(f"  Monotonic: {'OK' if avg_mono else 'FAIL'}")
    print(f"  Total: {time.time()-t0:.0f}s (GPU)")
    status = "PASSED" if avg_icir >= 0.3 else ("PARTIAL" if avg_mono else "FAILED")
    print(f"  {status}")

if __name__ == "__main__":
    main()
