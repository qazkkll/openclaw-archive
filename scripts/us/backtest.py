"""
LambdaMART回测 (OOS: 2025-01-01起)
====================================
只用out-of-sample数据，对齐production.json配置。
输出: data/backtest/{model}_oos.json

用法:
  python3 backtest.py                  # 跑两个模型
  python3 backtest.py arrow_v12        # 只跑V12
"""
import sys, json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(ROOT / "scripts" / "us"))

with open(ROOT / "config" / "central_config.json") as f:
    CFG = json.load(f)
FEATURES = CFG["features"]["lambdamart_v10_v12"]


def compute_features(df):
    """17个特征，与训练一致"""
    df = df.sort_values(["sym", "date"])
    g = df.groupby("sym")["close"]

    df["ret5"] = g.transform(lambda x: x.pct_change(5)).astype(np.float32)
    df["ret20"] = g.transform(lambda x: x.pct_change(20)).astype(np.float32)
    df["ret60"] = g.transform(lambda x: x.pct_change(60)).astype(np.float32)
    df["momentum_6m"] = g.transform(lambda x: x.pct_change(126)).astype(np.float32)
    df["momentum_1m"] = g.transform(lambda x: x.pct_change(21)).astype(np.float32)

    ma20 = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma_bias20"] = ((df["close"] - ma20) / ma20.replace(0, np.nan)).astype(np.float32)

    df["vol20"] = g.transform(lambda x: x.rolling(20, min_periods=10).std()).astype(np.float32)
    vol60 = g.transform(lambda x: x.rolling(60, min_periods=20).std())
    df["vol_ratio"] = (df["vol20"] / vol60.replace(0, np.nan)).astype(np.float32)

    delta = g.transform(lambda x: x.diff()).astype(np.float32)
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    avg_loss = loss.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    df["rsi14"] = (100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))).astype(np.float32)

    ema12 = g.transform(lambda x: x.ewm(span=12, min_periods=6).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, min_periods=13).mean())
    macd = (ema12 - ema26).astype(np.float32)
    df["macd_hist"] = (macd - macd.groupby(df["sym"]).transform(
        lambda x: x.ewm(span=9, min_periods=5).mean())).astype(np.float32)

    bb_mid = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std = g.transform(lambda x: x.rolling(20, min_periods=10).std())
    df["bb_pos"] = ((df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)).astype(np.float32)

    high60 = g.transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = g.transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)

    fund = pd.read_parquet(ROOT / "data/us/fundamentals_latest.parquet", columns=["sym", "beta"])
    df = df.merge(fund, on="sym", how="left")
    df["beta_c"] = df["beta"].clip(-2, 5).fillna(0.73).astype(np.float32)

    vix = pd.read_parquet(ROOT / "data/us/vix_10y.parquet")
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.reset_index()
    vix_date = [c for c in vix.columns if "date" in str(c).lower()][0]
    vix_val = [c for c in vix.columns if "close" in str(c).lower()][0]
    vix_df = pd.DataFrame({"date": pd.to_datetime(vix[vix_date]),
                           "vix_close": vix[vix_val].astype(np.float32)})
    df = df.merge(vix_df, on="date", how="left")

    spy = df[df["sym"] == "SPY"][["date", "close"]].sort_values("date")
    for d in [5, 20, 60]:
        spy[f"spy_ret{d}"] = spy["close"].pct_change(d).astype(np.float32)
    df = df.merge(spy[["date", "spy_ret5", "spy_ret20", "spy_ret60"]], on="date", how="left")

    df[FEATURES] = df[FEATURES].fillna(0)
    return df


def backtest_model(model_type):
    mc = CFG["models"][model_type]
    top_n = mc["top_n"]
    hold_days = mc["hold_days"]

    print(f"\n{'='*45}")
    print(f"  {model_type}  |  Top-{top_n}  |  {hold_days}天持有  |  OOS 2025+")
    print(f"{'='*45}")
    t0 = time.time()

    # 1. 加载 + 宇宙过滤 + OOS
    print("  加载数据...")
    df = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet",
                         columns=["date", "sym", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])

    # OOS: 2025-01-01起
    df = df[df["date"] >= "2025-01-01"].copy()

    # 宇宙过滤
    from universe_filter import filter_green_arrow, filter_blue_shield
    if "arrow" in model_type:
        df = filter_green_arrow(df)
    else:
        df = filter_blue_shield(df)
    print(f"  宇宙: {df['sym'].nunique()}只, {df['date'].nunique()}天")

    # 2. 特征
    print("  计算特征...")
    df = compute_features(df)

    # 3. 预测
    print("  模型预测...")
    model = lgb.Booster(model_file=str(ROOT / mc["model_file"]))
    df["pred"] = model.predict(df[FEATURES])

    # 4. 信号: 百分位🟢🟢(top 5%)取top_n
    df["rank_pct"] = df.groupby("date")["pred"].rank(pct=True, ascending=True)
    green2 = df[df["rank_pct"] >= 0.95].copy()
    green2["daily_rank"] = green2.groupby("date")["pred"].rank(ascending=False, method="first")
    green2 = green2[green2["daily_rank"] <= top_n]
    print(f"  🟢🟢信号: {green2['date'].nunique()}天有, 平均{green2.groupby('date').size().mean():.0f}只/天")

    # 5. 构建组合
    price = df.pivot_table(index="date", columns="sym", values="close")
    weights = pd.DataFrame(0.0, index=price.index, columns=price.columns)
    for date, group in green2.groupby("date"):
        syms = [s for s in group["sym"] if s in weights.columns]
        if syms:
            weights.loc[date, syms] = 1.0 / len(syms)
    weights = weights.replace(0, np.nan).ffill(limit=hold_days).fillna(0)

    # 6. 收益
    returns = price.pct_change().fillna(0)
    port_ret = (weights.shift(1) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    port_ret -= turnover * 0.002  # 0.1%+0.1%
    equity = (1 + port_ret).cumprod() * 100000

    # 统计
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_days = len(port_ret)
    ann_ret = ((1 + total_ret / 100) ** (252 / n_days) - 1) * 100
    sharpe = (port_ret.mean() - 0.05 / 252) / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
    max_dd = ((equity / equity.cummax()) - 1).min() * 100
    active = port_ret[port_ret != 0]
    win_rate = (active > 0).mean() * 100 if len(active) > 0 else 0

    # SPY基准
    spy = pd.read_parquet(ROOT / "data/us/us_hist_full_10y.parquet", columns=["date", "sym", "close"])
    spy = spy[(spy["sym"] == "SPY") & (spy["date"] >= "2025-01-01")].sort_values("date")
    spy_ret = (spy["close"].iloc[-1] / spy["close"].iloc[0] - 1) * 100
    spy_dd = ((spy["close"] / spy["close"].cummax()) - 1).min() * 100

    elapsed = time.time() - t0

    print(f"\n  📊 结果")
    print(f"  总收益:     {total_ret:+.1f}%")
    print(f"  年化收益:   {ann_ret:+.1f}%")
    print(f"  夏普比率:   {sharpe:.3f}")
    print(f"  最大回撤:   {max_dd:.1f}%")
    print(f"  胜率:       {win_rate:.0f}%")
    print(f"  SPY同期:    {spy_ret:+.1f}% (回撤{spy_dd:.1f}%)")
    print(f"  超额:       {total_ret - spy_ret:+.1f}%")
    print(f"  耗时:       {elapsed:.0f}s")

    # 保存
    result = {
        "model": model_type,
        "period": "2025-01-01 ~ OOS",
        "top_n": top_n,
        "hold_days": hold_days,
        "signal": "🟢🟢 (top 5%)",
        "universe_filter": True,
        "total_return_pct": round(total_ret, 1),
        "annualized_return_pct": round(ann_ret, 1),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 1),
        "win_rate_pct": round(win_rate, 0),
        "spy_return_pct": round(spy_ret, 1),
        "excess_return_pct": round(total_ret - spy_ret, 1),
        "universe_size": int(df["sym"].nunique()),
        "backtest_days": n_days,
    }
    out = ROOT / f"data/backtest/{model_type}_oos.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {out}")
    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        backtest_model(sys.argv[1])
    else:
        backtest_model("arrow_v12")
        backtest_model("blueshield_v10")
