"""
LambdaMART重训（内存优化版）
============================
关键优化：
1. 只取最近3年数据（不加载10年）
2. float32（省50%内存）
3. 采样3000只股票（OOM-safe）
4. 特征计算增量式（不保留中间列）
"""

import sys, json, time, argparse, gc
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb


def load_recent_data(years=3):
    """只加载最近N年数据"""
    print(f"Loading last {years} years of data...")
    df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet")
    df["date"] = pd.to_datetime(df["date"])

    cutoff = df["date"].max() - pd.Timedelta(days=365 * years)
    df = df[df["date"] >= cutoff].copy()
    print(f"  After date filter: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 降精度
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = df[c].astype(np.float32)

    return df


def compute_features_light(df):
    """轻量特征计算（不保留中间列）"""
    print("Computing features (light)...")
    t0 = time.time()
    df = df.sort_values(["sym", "date"]).copy()

    g = df.groupby("sym")["close"]

    # 核心特征（只保留模型需要的）
    df["ma20"] = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma_bias20"] = ((df["close"] - df["ma20"]) / df["ma20"].replace(0, np.nan)).astype(np.float32)
    df.drop(columns=["ma20"], inplace=True)

    df["ret1"] = g.transform(lambda x: x.pct_change(1)).astype(np.float32)
    df["ret5"] = g.transform(lambda x: x.pct_change(5)).astype(np.float32)
    df["ret20"] = g.transform(lambda x: x.pct_change(20)).astype(np.float32)
    df["ret60"] = g.transform(lambda x: x.pct_change(60)).astype(np.float32)
    df["momentum_6m"] = g.transform(lambda x: x.pct_change(126)).astype(np.float32)
    df["momentum_1m"] = g.transform(lambda x: x.pct_change(21)).astype(np.float32)

    df["vol20"] = df.groupby("sym")["ret1"].transform(lambda x: x.rolling(20, min_periods=10).std()).astype(np.float32)
    df["vol5"] = df.groupby("sym")["ret1"].transform(lambda x: x.rolling(5, min_periods=3).std()).astype(np.float32)
    df["vol_ratio"] = (df["vol5"] / df["vol20"].replace(0, np.nan)).astype(np.float32)

    # RSI
    delta = df.groupby("sym")["close"].transform(lambda x: x.diff()).astype(np.float32)
    gain = delta.where(delta > 0, 0).astype(np.float32)
    loss = (-delta).where(delta < 0, 0).astype(np.float32)
    avg_gain = gain.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    avg_loss = loss.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    df["rsi14"] = (100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))).astype(np.float32)
    del delta, gain, loss, avg_gain, avg_loss
    gc.collect()

    # MACD
    ema12 = df.groupby("sym")["close"].transform(lambda x: x.ewm(span=12, min_periods=6).mean())
    ema26 = df.groupby("sym")["close"].transform(lambda x: x.ewm(span=26, min_periods=13).mean())
    df["macd"] = (ema12 - ema26).astype(np.float32)
    df["macd_signal"] = df.groupby("sym")["macd"].transform(lambda x: x.ewm(span=9, min_periods=5).mean()).astype(np.float32)
    df["macd_hist"] = (df["macd"] - df["macd_signal"]).astype(np.float32)
    del ema12, ema26

    # Bollinger
    bb_mid = df.groupby("sym")["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std = df.groupby("sym")["close"].transform(lambda x: x.rolling(20, min_periods=10).std())
    df["bb_width"] = (2 * bb_std / bb_mid.replace(0, np.nan)).astype(np.float32)
    df["bb_pos"] = ((df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)).astype(np.float32)
    del bb_mid, bb_std

    # 价格位置
    high60 = df.groupby("sym")["high"].transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = df.groupby("sym")["low"].transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)
    del high60, low60

    # 成交量
    vol_ma20 = df.groupby("sym")["volume"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["vol_ratio_ma"] = (df["volume"] / vol_ma20.replace(0, np.nan)).astype(np.float32)
    del vol_ma20

    # 基本面
    fund = pd.read_parquet(ROOT / "data/us/fundamentals_latest.parquet")
    df = df.merge(fund, on="sym", how="left")
    df["pe_log"] = np.log1p(df["pe_trailing"].clip(-0.99, None).fillna(0)).astype(np.float32)
    df["beta"] = df["beta"].clip(-2, 5).fillna(0.73).astype(np.float32)
    df["div_yield"] = df["div_yield"].fillna(0).astype(np.float32)

    # 前瞻收益
    df["fwd_5d"] = df.groupby("sym")["close"].transform(lambda x: x.shift(-5) / x - 1).astype(np.float32)

    # VIX
    vix = pd.read_parquet(ROOT / "data/us/vix_10y.parquet")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.reset_index()
    vix_date = [c for c in vix.columns if "date" in str(c).lower()][0]
    vix_val = [c for c in vix.columns if "close" in str(c).lower()][0]
    vix_df = pd.DataFrame({"date": pd.to_datetime(vix[vix_date]), "vix_close": vix[vix_val].astype(np.float32)})
    df = df.merge(vix_df, on="date", how="left")

    # SPY
    spy = df[df["sym"] == "SPY"][["date", "close"]].sort_values("date")
    for d in [1, 5, 20, 60]:
        spy[f"spy_ret{d}"] = spy["close"].pct_change(d).astype(np.float32)
    df = df.merge(spy[["date"] + [f"spy_ret{d}" for d in [1, 5, 20, 60]]], on="date", how="left")

    print(f"  Features done in {time.time() - t0:.0f}s, memory: {df.memory_usage(deep=True).sum() / 1e6:.0f}MB")
    return df


FEATURE_COLS = [
    "ma_bias20", "ret1", "ret5", "ret20", "ret60",
    "momentum_6m", "momentum_1m", "vol20", "vol5", "vol_ratio",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pos", "price_position", "vol_ratio_ma",
    "vix_close", "spy_ret1", "spy_ret5", "spy_ret20", "spy_ret60",
    "pe_log", "beta", "div_yield",
]


def train_and_evaluate(df, universe_filter, model_name):
    """Walk-Forward LambdaMART训练"""
    print(f"\n=== LambdaMART Training: {model_name} ({universe_filter}) ===")

    # 宇宙过滤
    if universe_filter == ">$10":
        df = df[df["close"] >= 10].copy()
    elif universe_filter == "$1-$10":
        df = df[(df["close"] >= 1) & (df["close"] <= 10)].copy()

    print(f"  Universe: {len(df)} rows, {df['sym'].nunique()} stocks")

    # 清理
    df = df.dropna(subset=["fwd_5d"])
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df["year"] = df["date"].dt.year

    # 排除最新30天（没有fwd_5d）
    max_date = df["date"].max()
    df = df[df["date"] < max_date - pd.Timedelta(days=10)]

    years = sorted(df["year"].unique())
    print(f"  Years: {years}")

    results = []
    last_model = None

    for test_year in years:
        if test_year < 2021:  # 只用最近几年
            continue

        train_df = df[df["year"] < test_year]
        test_df = df[df["year"] == test_year]

        if len(train_df) < 5000 or len(test_df) < 500:
            print(f"  {test_year}: SKIP (train={len(train_df)}, test={len(test_df)})")
            continue

        X_train = train_df[FEATURE_COLS]
        X_test = test_df[FEATURE_COLS]

        # LambdaMART需要整数标签：每天内按fwd_5d排序分成5组(0-4)
        def make_rank_labels(sub_df):
            return sub_df.groupby("date")["fwd_5d"].transform(
                lambda x: pd.qcut(x, 5, labels=[0, 1, 2, 3, 4], duplicates="drop") if len(x) >= 5 else 2
            ).astype(int)

        y_train = make_rank_labels(train_df)
        y_test = make_rank_labels(test_df)

        # 分组
        train_groups = train_df.groupby("date").size().values
        test_groups = test_df.groupby("date").size().values

        train_data = lgb.Dataset(X_train, label=y_train, group=train_groups)
        test_data = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=train_data)

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [30],
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, train_data, num_boost_round=300,
            valid_sets=[test_data],
            callbacks=[lgb.log_evaluation(0)],
        )

        # 评估
        test_df = test_df.copy()
        test_df["pred"] = model.predict(X_test)

        # IC
        daily_ic = test_df.groupby("date").apply(
            lambda g: g["pred"].corr(g["fwd_5d"]), include_groups=False
        )
        ic = daily_ic.mean()
        icir = ic / daily_ic.std() if daily_ic.std() > 0 else 0

        # 排名分析
        test_df["q"] = test_df.groupby("date")["pred"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if len(x) >= 5 else 2
        )
        qr = test_df.groupby("q")["fwd_5d"].agg(["mean", "count"])
        top_ret = qr.iloc[-1]["mean"] if len(qr) >= 2 else 0
        bot_ret = qr.iloc[0]["mean"] if len(qr) >= 2 else 0

        # Top30 spread
        top30 = test_df.groupby("date").apply(
            lambda g: g.nlargest(min(30, len(g)), "pred")["fwd_5d"].mean(), include_groups=False
        )
        avg30 = test_df.groupby("date").apply(
            lambda g: g.sample(min(30, len(g)), random_state=42)["fwd_5d"].mean(), include_groups=False
        )
        spread = (top30 - avg30).mean()

        # 胜率
        top_mask = test_df["q"] == qr.index[-1]
        win = (test_df.loc[top_mask, "fwd_5d"] > 0).mean() if top_mask.sum() > 0 else 0

        r = {"year": test_year, "ic": ic, "icir": icir, "top5": top_ret, "bot20": bot_ret,
             "spread": spread, "win": win, "n": len(test_df)}
        results.append(r)
        last_model = model

        print(f"  {test_year}: IC={ic:+.4f} ICIR={icir:.3f} Top5%={top_ret:+.4f} "
              f"Bot20%={bot_ret:+.4f} Spread={spread:+.4f} Win={win:.1%}")

        del train_data, test_data, X_train, X_test
        gc.collect()

    return results, last_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["blueshield_v10", "arrow_v12"])
    parser.add_argument("--sample", type=int, default=3000, help="Sample N stocks to avoid OOM")
    args = parser.parse_args()

    # 加载数据（最近3年）
    df = load_recent_data(years=3)

    # 采样
    all_syms = df["sym"].unique()
    if len(all_syms) > args.sample:
        np.random.seed(42)
        sampled = np.random.choice(all_syms, args.sample, replace=False)
        # 保留SPY（需要计算spy_ret）
        if "SPY" not in sampled:
            sampled = np.append(sampled, "SPY")
        df = df[df["sym"].isin(sampled)].copy()
        print(f"  Sampled to {len(sampled)} stocks, {len(df)} rows")

    # 特征计算
    df = compute_features_light(df)
    gc.collect()

    # 宇宙
    if "blueshield" in args.model:
        universe = ">$10"
    else:
        universe = "$1-$10"

    # 训练
    results, model = train_and_evaluate(df, universe, args.model)

    if not results:
        print("\n❌ No results — not enough data")
        return

    # 汇总
    avg = lambda k: np.mean([r[k] for r in results])
    monotonic = avg("top5") > avg("bot20")

    print(f"\n=== {args.model} Summary ===")
    print(f"  Avg IC:     {avg('ic'):+.4f}")
    print(f"  Avg ICIR:   {avg('icir'):.3f}")
    print(f"  Avg Top5%:  {avg('top5'):+.4f}")
    print(f"  Avg Bot20%: {avg('bot20'):+.4f}")
    print(f"  Avg Spread: {avg('spread'):+.4f}")
    print(f"  Avg Win:    {avg('win'):.1%}")
    print(f"  Monotonic:  {'✅' if monotonic else '❌'}")

    # 保存
    if model:
        path = ROOT / f"models/us/{args.model}_lambdamart.txt"
        model.save_model(str(path))
        print(f"\n  Model: {path}")

    report = {
        "model": args.model, "timestamp": datetime.now().isoformat(),
        "algorithm": "LambdaMART", "universe": universe,
        "sample_size": args.sample, "features": FEATURE_COLS,
        "results": results,
        "summary": {k: round(avg(k), 4) for k in ["ic", "icir", "top5", "bot20", "spread", "win"]},
        "rank_monotonic": monotonic,
    }
    rpath = ROOT / f"data/lambdamart_{args.model}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {rpath}")

    if monotonic and avg("icir") > 0.3:
        print(f"\n✅ PASSED")
    elif monotonic:
        print(f"\n⚠️ PARTIAL (monotonic but ICIR<0.3)")
    else:
        print(f"\n❌ FAILED (rank inverted)")


if __name__ == "__main__":
    main()
