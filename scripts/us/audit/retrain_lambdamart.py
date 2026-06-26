"""
LambdaMART完整重训 — 基于改进的基本面数据
============================================
目标：修复排名反转问题，用真实beta数据重训V10/V12模型
输出：新模型文件 + 验证报告

用法：
  python scripts/us/audit/retrain_lambdamart.py --model blueshield_v10
  python scripts/us/audit/retrain_lambdamart.py --model arrow_v12
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime


def load_data():
    """加载价格+基本面+宏观数据"""
    print("Loading data...")
    df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet")
    df["date"] = pd.to_datetime(df["date"])

    # 基本面
    fund = pd.read_parquet(ROOT / "data/us/fundamentals_latest.parquet")
    df = df.merge(fund, on="sym", how="left")

    # VIX
    vix = pd.read_parquet(ROOT / "data/us/vix_10y.parquet")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.reset_index()
    vix_date = [c for c in vix.columns if "date" in c.lower() or "Date" in c][0]
    vix_val = [c for c in vix.columns if "close" in c.lower() or "Close" in c][0]
    vix_df = pd.DataFrame({"date": pd.to_datetime(vix[vix_date]), "vix_close": vix[vix_val].astype(float)})
    df = df.merge(vix_df, on="date", how="left")

    # SPY returns
    spy = df[df["sym"] == "SPY"][["date", "close"]].sort_values("date")
    for d in [1, 5, 20, 60]:
        spy[f"spy_ret{d}"] = spy["close"].pct_change(d)
    df = df.merge(spy[["date"] + [f"spy_ret{d}" for d in [1, 5, 20, 60]]], on="date", how="left")

    print(f"Loaded: {len(df)} rows, {df['sym'].nunique()} stocks")
    return df


def compute_features(df):
    """计算特征"""
    print("Computing features...")
    t0 = time.time()

    df = df.sort_values(["sym", "date"])

    # 前瞻收益（用于训练目标）
    df["fwd_5d"] = df.groupby("sym")["close"].transform(lambda x: x.shift(-5) / x - 1)

    # 技术特征
    g = df.groupby("sym")["close"]
    df["ma5"] = g.transform(lambda x: x.rolling(5).mean())
    df["ma20"] = g.transform(lambda x: x.rolling(20).mean())
    df["ma60"] = g.transform(lambda x: x.rolling(60).mean())
    df["ma_bias20"] = (df["close"] - df["ma20"]) / df["ma20"]
    df["ret1"] = g.transform(lambda x: x.pct_change(1))
    df["ret5"] = g.transform(lambda x: x.pct_change(5))
    df["ret20"] = g.transform(lambda x: x.pct_change(20))
    df["ret60"] = g.transform(lambda x: x.pct_change(60))
    df["momentum_6m"] = g.transform(lambda x: x.pct_change(126))
    df["momentum_1m"] = g.transform(lambda x: x.pct_change(21))

    # 波动率
    df["vol20"] = df.groupby("sym")["ret1"].transform(lambda x: x.rolling(20).std())
    df["vol5"] = df.groupby("sym")["ret1"].transform(lambda x: x.rolling(5).std())
    df["vol_ratio"] = df["vol5"] / df["vol20"].replace(0, np.nan)

    # RSI
    delta = df.groupby("sym")["close"].transform(lambda x: x.diff())
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.groupby(df["sym"]).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(df["sym"]).transform(lambda x: x.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df.groupby("sym")["close"].transform(lambda x: x.ewm(span=12).mean())
    ema26 = df.groupby("sym")["close"].transform(lambda x: x.ewm(span=26).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df.groupby("sym")["macd"].transform(lambda x: x.ewm(span=9).mean())
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    bb_mid = df.groupby("sym")["close"].transform(lambda x: x.rolling(20).mean())
    bb_std = df.groupby("sym")["close"].transform(lambda x: x.rolling(20).std())
    df["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
    df["bb_pos"] = (df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)

    # 价格位置
    high60 = df.groupby("sym")["high"].transform(lambda x: x.rolling(60).max())
    low60 = df.groupby("sym")["low"].transform(lambda x: x.rolling(60).min())
    df["price_position"] = (df["close"] - low60) / (high60 - low60).replace(0, np.nan)

    # 成交量
    df["vol_ma20"] = df.groupby("sym")["volume"].transform(lambda x: x.rolling(20).mean())
    df["vol_ratio_ma"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # 基本面特征
    df["pe_log"] = np.log1p(df["pe_trailing"].clip(-0.99, None))
    df["beta"] = df["beta"].clip(-2, 5)
    df["div_yield"] = df["div_yield"].fillna(0)

    # 宏观特征已在load_data中添加

    print(f"Features computed in {time.time() - t0:.0f}s")
    return df


def get_feature_cols(model_type="blueshield"):
    """获取特征列"""
    tech = [
        "ma_bias20", "ret1", "ret5", "ret20", "ret60",
        "momentum_6m", "momentum_1m", "vol20", "vol5", "vol_ratio",
        "rsi14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_pos", "price_position", "vol_ratio_ma",
    ]
    macro = ["vix_close", "spy_ret1", "spy_ret5", "spy_ret20", "spy_ret60"]
    fundamental = ["pe_log", "beta", "div_yield"]
    return tech + macro + fundamental


def walk_forward_train(df, feature_cols, model_type="blueshield", universe_filter=None):
    """
    Walk-Forward训练，使用LambdaMART目标
    """
    print(f"\n=== Walk-Forward LambdaMART Training ({model_type}) ===")

    # 应用宇宙过滤
    if universe_filter == ">$10":
        df = df[df["close"] >= 10].copy()
        print(f"Universe: >$10, {df['sym'].nunique()} stocks")
    elif universe_filter == "$1-$10":
        df = df[(df["close"] >= 1) & (df["close"] <= 10)].copy()
        print(f"Universe: $1-$10, {df['sym'].nunique()} stocks")

    # 清理
    df = df.dropna(subset=["fwd_5d"])
    df = df.dropna(subset=feature_cols, how="all")
    df[feature_cols] = df[feature_cols].fillna(0)

    # 时间分割
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    print(f"Years: {years}")

    # Walk-Forward: 训练到Y-1，测试Y
    all_results = []
    models = {}

    for test_year in years:
        if test_year < 2019:  # 需要至少3年训练数据
            continue

        train_mask = df["year"] < test_year
        test_mask = df["year"] == test_year

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 10000 or len(test_df) < 1000:
            continue

        X_train = train_df[feature_cols]
        y_train = train_df["fwd_5d"]
        X_test = test_df[feature_cols]
        y_test = test_df["fwd_5d"]

        # 创建分组（每天的股票数）
        train_groups = train_df.groupby("date").size().values
        test_groups = test_df.groupby("date").size().values

        # LambdaMART训练
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
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[test_data],
            callbacks=[lgb.log_evaluation(0)],
        )

        # 预测
        test_df = test_df.copy()
        test_df["pred_score"] = model.predict(X_test)

        # 计算IC
        daily_ic = test_df.groupby("date").apply(
            lambda g: g["pred_score"].corr(g["fwd_5d"]), include_groups=False
        )
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = ic_mean / ic_std if ic_std > 0 else 0

        # 排名分析
        test_df["group"] = test_df.groupby("date")["pred_score"].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
        )
        group_ret = test_df.groupby("group")["fwd_5d"].agg(["mean", "count"])

        top_ret = group_ret.iloc[-1]["mean"] if len(group_ret) >= 2 else 0
        bot_ret = group_ret.iloc[0]["mean"] if len(group_ret) >= 2 else 0

        # Top30 spread
        top30 = test_df.groupby("date").apply(
            lambda g: g.nlargest(30, "pred_score")["fwd_5d"].mean(), include_groups=False
        )
        all30 = test_df.groupby("date").apply(
            lambda g: g.sample(min(30, len(g)))["fwd_5d"].mean(), include_groups=False
        )
        spread = (top30 - all30).mean()

        # 胜率
        top5_mask = test_df["group"] == group_ret.index[-1]
        win_rate = (test_df.loc[top5_mask, "fwd_5d"] > 0).mean() if top5_mask.sum() > 0 else 0

        result = {
            "year": test_year,
            "ic": ic_mean,
            "icir": icir,
            "top5_ret": top_ret,
            "bot20_ret": bot_ret,
            "top30_spread": spread,
            "win_rate": win_rate,
            "n_test": len(test_df),
        }
        all_results.append(result)
        models[test_year] = model

        print(f"  {test_year}: IC={ic_mean:+.4f} ICIR={icir:.3f} "
              f"Top5%={top_ret:+.4f} Bot20%={bot_ret:+.4f} "
              f"Spread={spread:+.4f} Win={win_rate:.1%}")

    return all_results, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["blueshield_v10", "arrow_v12"])
    args = parser.parse_args()

    # 加载数据
    df = load_data()
    df = compute_features(df)

    # 获取特征
    feature_cols = get_feature_cols(args.model)

    # 选择宇宙过滤
    if "blueshield" in args.model:
        universe = ">$10"
    else:
        universe = "$1-$10"

    # Walk-Forward训练
    results, models = walk_forward_train(df, feature_cols, args.model, universe)

    # 汇总
    print(f"\n=== {args.model} Summary ===")
    if results:
        avg_ic = np.mean([r["ic"] for r in results])
        avg_icir = np.mean([r["icir"] for r in results])
        avg_top = np.mean([r["top5_ret"] for r in results])
        avg_bot = np.mean([r["bot20_ret"] for r in results])
        avg_spread = np.mean([r["top30_spread"] for r in results])
        avg_win = np.mean([r["win_rate"] for r in results])

        monotonic = avg_top > avg_bot

        print(f"  Avg IC: {avg_ic:+.4f}")
        print(f"  Avg ICIR: {avg_icir:.3f}")
        print(f"  Avg Top5%: {avg_top:+.4f}")
        print(f"  Avg Bot20%: {avg_bot:+.4f}")
        print(f"  Avg Spread: {avg_spread:+.4f}")
        print(f"  Avg Win Rate: {avg_win:.1%}")
        print(f"  Rank Monotonic: {'✅ YES' if monotonic else '❌ NO'}")

        # 保存最新模型
        latest_model = list(models.values())[-1]
        model_path = ROOT / f"models/us/{args.model}_lambdamart.txt"
        latest_model.save_model(str(model_path))
        print(f"\n  Model saved: {model_path}")

        # 保存报告
        report = {
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "algorithm": "LambdaMART",
            "universe": universe,
            "features": feature_cols,
            "n_features": len(feature_cols),
            "results": results,
            "summary": {
                "avg_ic": avg_ic,
                "avg_icir": avg_icir,
                "avg_top5_ret": avg_top,
                "avg_bot20_ret": avg_bot,
                "avg_spread": avg_spread,
                "avg_win_rate": avg_win,
                "rank_monotonic": monotonic,
            },
        }
        report_path = ROOT / f"data/audit_{args.model}_lambdamart_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report saved: {report_path}")

        # 最终判定
        if monotonic and avg_icir > 0.3:
            print(f"\n✅ {args.model} LambdaMART PASSED — rank monotonic, ICIR>0.3")
        elif monotonic:
            print(f"\n⚠️ {args.model} LambdaMART PARTIAL — rank monotonic but ICIR<0.3")
        else:
            print(f"\n❌ {args.model} LambdaMART FAILED — rank still inverted")


if __name__ == "__main__":
    main()
