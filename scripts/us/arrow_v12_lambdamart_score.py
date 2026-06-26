"""
LambdaMART评分脚本 — V12绿箭
=============================
17个特征，$1-$10宇宙，5天前瞻排名
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

# 导入宇宙过滤器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universe_filter import filter_green_arrow

# === 代码级契约：启动时校验manifest ===
from contract.manifest import validate_before_scoring
_manifest = validate_before_scoring("arrow_v12")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT.endswith("scripts"):
    ROOT = os.path.dirname(ROOT)

MODEL_PATH = os.path.join(ROOT, "models/us/arrow_v12_lambdamart.txt")
VIX_PATH = os.path.join(ROOT, "data/us/vix_10y.parquet")
FUND_PATH = os.path.join(ROOT, "data/us/fundamentals_latest.parquet")
OUTPUT_DIR = os.path.join(ROOT, "data/us")

FEATURES = [
    "ret5", "ret20", "ret60", "momentum_6m", "momentum_1m",
    "ma_bias20", "vol20", "vol_ratio", "rsi14", "macd_hist",
    "bb_pos", "price_position", "beta_c", "vix_close",
    "spy_ret5", "spy_ret20", "spy_ret60",
]


def compute_features(df, fund, vix_close, spy_rets):
    """计算17个特征"""
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

    # MACD
    ema12 = g.transform(lambda x: x.ewm(span=12, min_periods=6).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, min_periods=13).mean())
    macd = (ema12 - ema26).astype(np.float32)
    df["macd_hist"] = (macd - macd.groupby(df["sym"]).transform(
        lambda x: x.ewm(span=9, min_periods=5).mean()
    )).astype(np.float32)

    # Bollinger
    bb_mid = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    bb_std = g.transform(lambda x: x.rolling(20, min_periods=10).std())
    df["bb_pos"] = ((df["close"] - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)).astype(np.float32)

    # 价格位置
    high60 = g.transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = g.transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)

    # Beta
    df = df.merge(fund[["sym", "beta"]], on="sym", how="left")
    df["beta_c"] = df["beta"].clip(-2, 5).fillna(0.73).astype(np.float32)

    # VIX
    df["vix_close"] = vix_close

    # SPY returns
    for d in [5, 20, 60]:
        df[f"spy_ret{d}"] = spy_rets.get(d, 0.0)

    return df


def load_macro_data():
    """加载VIX和SPY数据"""
    # VIX
    vix = pd.read_parquet(VIX_PATH)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix = vix.reset_index()
    vix_date = [c for c in vix.columns if "date" in str(c).lower()][0]
    vix_val = [c for c in vix.columns if "close" in str(c).lower()][0]
    latest_vix = float(vix.sort_values(vix_date)[vix_val].iloc[-1])

    # SPY (用历史数据的最新值)
    spy = pd.read_parquet(os.path.join(ROOT, "data/us/us_hist_full_10y.parquet"),
                           columns=["date", "sym", "close"])
    spy = spy[spy["sym"] == "SPY"].sort_values("date").tail(200)
    spy_rets = {}
    for d in [5, 20, 60]:
        spy_rets[d] = float(spy["close"].pct_change(d).iloc[-1]) if len(spy) > d else 0.0

    return latest_vix, spy_rets


def main():
    print("=== LambdaMART V12 绿箭评分 ===")

    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        sys.exit(1)

    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"   模型: {os.path.basename(MODEL_PATH)} ({model.num_feature()} 特征)")

    # 加载宏观数据
    vix_close, spy_rets = load_macro_data()
    print(f"   VIX: {vix_close:.1f}")

    # 加载价格数据（最近200天）
    print("   加载价格数据...")
    df = pd.read_parquet(os.path.join(ROOT, "data/us/us_hist_full_10y.parquet"),
                          columns=["date", "sym", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    cutoff = latest_date - pd.Timedelta(days=400)  # 需要历史数据计算特征
    df = df[df["date"] >= cutoff].copy()

    # 宇宙过滤：$1-$10 + 质量过滤
    latest_prices = df[df["date"] == latest_date].set_index("sym")["close"]
    # 计算20日日均美元成交量（流动性指标）
    df["dollar_vol"] = (df["close"] * df["volume"]).astype(np.float32)
    df["dollar_vol_20d"] = df.groupby("sym")["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    ).astype(np.float32)
    # 用最新一天的流动性数据过滤
    latest_dv = df[df["date"] == latest_date].set_index("sym")["dollar_vol_20d"]
    temp = pd.DataFrame({"close": latest_prices, "dollar_vol_20d": latest_dv}).reset_index()
    temp.columns = ["sym", "close", "dollar_vol_20d"]
    filtered = filter_green_arrow(temp, min_price=1.0, min_dollar_vol=500_000)
    universe_syms = filtered["sym"].values
    df = df[df["sym"].isin(universe_syms)].copy()
    print(f"   宇宙: {len(universe_syms)} 只 ($1-$10, 排除权证/SPAC/杠杆/低流动性)")

    # 加载基本面
    fund = pd.read_parquet(FUND_PATH, columns=["sym", "beta"])

    # 计算特征
    print("   计算特征...")
    df = compute_features(df, fund, vix_close, spy_rets)
    df[FEATURES] = df[FEATURES].fillna(0)

    # 只评最新日
    latest = df[df["date"] == latest_date].copy()
    print(f"   评分: {len(latest)} 只股票")

    # 预测
    latest["score"] = model.predict(latest[FEATURES])
    latest = latest.sort_values("score", ascending=False)

    # 信号分配
    n = len(latest)
    latest["rank_pct"] = np.arange(1, n + 1) / n

    def signal_label(pct):
        if pct <= 0.05:
            return "🟢🟢精品买入"
        elif pct <= 0.10:
            return "🟢强信号"
        elif pct <= 0.20:
            return "🟡观察"
        else:
            return "⚪无信号"

    latest["signal"] = latest["rank_pct"].apply(signal_label)

    # Top20 picks
    top20 = latest.head(20)

    # 输出
    picks = []
    for _, r in top20.iterrows():
        picks.append({
            "sym": r["sym"], "close": float(r["close"]), "score": float(r["score"]),
            "rank_pct": float(r["rank_pct"]), "signal": r["signal"],
            "pct_rank": float(r["rank_pct"]) * 100,
        })

    out = {
        "timestamp": datetime.now().isoformat(),
        "date": str(latest_date.date()),
        "model": "arrow_v12_lambdamart",
        "version": "V12-LambdaMART",
        "universe": "$1-$10",
        "total": len(latest),
        "picks": picks,
        "vix": vix_close,
    }
    out_path = os.path.join(OUTPUT_DIR, f"arrow_v12_scored_{latest_date.strftime('%Y%m%d')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n   保存: {out_path}")
    print("✅ 完成")


if __name__ == "__main__":
    main()
