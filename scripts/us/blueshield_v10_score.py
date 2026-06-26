"""
BlueShield V10 Quantile 评分脚本
=================================
43维特征，>$10宇宙，中位数分位数回归 (ICIR=0.7335)

旧模型文件: models/us/blueshield_lgb_v9_quantile_lgb.txt
新框架包装: manifest校验, universe_filter, central_config
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb

# 导入宇宙过滤器
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from universe_filter import filter_blue_shield

# === 代码级契约：启动时校验manifest ===
from contract.manifest import validate_before_scoring
_manifest = validate_before_scoring("blueshield_v10")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT.endswith("scripts"):
    ROOT = os.path.dirname(ROOT)

MODEL_PATH = os.path.join(ROOT, "models/us/blueshield_lgb_v9_quantile_lgb.txt")
FUND_PATH = os.path.join(ROOT, "data/us/fundamentals_latest.parquet")
OUTPUT_DIR = os.path.join(ROOT, "data/us")

# 43 features from meta file
FEATURES = [
    "ma5", "ma20", "ma60", "ma_bias20", "ma_align", "price_position",
    "ret1", "ret5", "ret20", "ret60", "momentum_6m", "momentum_1m",
    "mom_divergence", "trend_accel", "vol20", "vol5", "vol_ratio", "vol_change",
    "rsi14", "rsi_change", "macd", "macd_signal", "macd_hist",
    "bb_std", "bb_width", "bb_pos", "ret_quality",
    "range_ratio", "avg_body", "vwap_drift",
    "ret_10d", "ret_30d", "ret_90d",
    "vol_regime", "ma_cross_5_20", "ma_cross_20_60",
    "rsi_zone", "macd_roc", "dd_60", "ud_vol_ratio",
    "pe_log", "div_yield", "beta",
]


def compute_features(df, fund):
    """
    计算全部43个特征
    基于 blueshield_v10_meta.json 和 precompute_scores.py 的定义
    """
    df = df.sort_values(["sym", "date"]).copy()

    # --- Moving Averages ---
    g = df.groupby("sym")["close"]
    df["ma5"] = g.transform(lambda x: x.rolling(5, min_periods=3).mean()).astype(np.float32)
    df["ma20"] = g.transform(lambda x: x.rolling(20, min_periods=10).mean()).astype(np.float32)
    df["ma60"] = g.transform(lambda x: x.rolling(60, min_periods=30).mean()).astype(np.float32)

    # --- MA Bias & Alignment ---
    df["ma_bias20"] = ((df["close"] - df["ma20"]) / df["ma20"].replace(0, np.nan)).astype(np.float32)
    df["ma_align"] = (
        (df["close"] > df["ma5"]).astype(int) + (df["ma5"] > df["ma20"]).astype(int)
    ).astype(np.float32)

    # --- Price Position ---
    high60 = g.transform(lambda x: x.rolling(60, min_periods=30).max())
    low60 = g.transform(lambda x: x.rolling(60, min_periods=30).min())
    df["price_position"] = ((df["close"] - low60) / (high60 - low60).replace(0, np.nan)).astype(np.float32)

    # --- Returns ---
    df["ret1"] = g.transform(lambda x: x.pct_change(1)).astype(np.float32)
    df["ret5"] = g.transform(lambda x: x.pct_change(5)).astype(np.float32)
    df["ret20"] = g.transform(lambda x: x.pct_change(20)).astype(np.float32)
    df["ret60"] = g.transform(lambda x: x.pct_change(60)).astype(np.float32)

    # --- Momentum ---
    df["momentum_6m"] = g.transform(lambda x: x.pct_change(126)).astype(np.float32)
    df["momentum_1m"] = g.transform(lambda x: x.pct_change(21)).astype(np.float32)
    df["mom_divergence"] = (df["momentum_1m"] - df["ret20"]).astype(np.float32)

    # --- Trend Acceleration ---
    ret5_lag5 = g.transform(lambda x: x.pct_change(5).shift(5))
    df["trend_accel"] = (df["ret5"] - ret5_lag5).astype(np.float32)

    # --- Volatility ---
    dr = g.transform(lambda x: x.pct_change(1))
    df["vol20"] = dr.groupby(df["sym"]).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    ).astype(np.float32)
    df["vol5"] = dr.groupby(df["sym"]).transform(
        lambda x: x.rolling(5, min_periods=3).std()
    ).astype(np.float32)
    df["vol_ratio"] = (df["vol20"] / df.groupby("sym")["vol20"].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    ).replace(0, np.nan)).astype(np.float32)
    df["vol_change"] = (df["vol20"] / df.groupby("sym")["vol20"].transform(
        lambda x: x.shift(20)
    ).replace(0, np.nan)).astype(np.float32)

    # --- RSI ---
    delta = df.groupby("sym")["close"].transform(lambda x: x.diff()).astype(np.float32)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    avg_loss = loss.groupby(df["sym"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
    df["rsi14"] = (100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))).astype(np.float32)
    df["rsi_change"] = df.groupby("sym")["rsi14"].transform(lambda x: x.diff(5)).astype(np.float32)

    # --- MACD ---
    ema12 = g.transform(lambda x: x.ewm(span=12, min_periods=6).mean())
    ema26 = g.transform(lambda x: x.ewm(span=26, min_periods=13).mean())
    df["macd"] = (ema12 - ema26).astype(np.float32)
    df["macd_signal"] = df.groupby("sym")["macd"].transform(
        lambda x: x.ewm(span=9, min_periods=5).mean()
    ).astype(np.float32)
    df["macd_hist"] = (df["macd"] - df["macd_signal"]).astype(np.float32)

    # --- Bollinger Bands ---
    bb_mid = g.transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["bb_std"] = g.transform(lambda x: x.rolling(20, min_periods=10).std()).astype(np.float32)
    df["bb_width"] = (2 * df["bb_std"] / bb_mid.replace(0, np.nan)).astype(np.float32)
    df["bb_pos"] = ((df["close"] - bb_mid) / (2 * df["bb_std"]).replace(0, np.nan)).astype(np.float32)

    # --- Return Quality ---
    df["ret_quality"] = (df["ret20"] / df["vol20"].replace(0, np.nan)).astype(np.float32)

    # --- Range Ratio (approximate: 2*vol5 since we don't have H/L in this parquet) ---
    df["range_ratio"] = (2 * df["vol5"]).astype(np.float32)

    # --- Avg Body (approximate: abs daily return) ---
    df["avg_body"] = df["ret1"].abs().astype(np.float32)

    # --- VWAP Drift (approximate: ret5 / vol5) ---
    df["vwap_drift"] = (df["ret5"] / df["vol5"].replace(0, np.nan)).astype(np.float32)

    # --- Extended Returns ---
    df["ret_10d"] = g.transform(lambda x: x.pct_change(10)).astype(np.float32)
    df["ret_30d"] = g.transform(lambda x: x.pct_change(30)).astype(np.float32)
    df["ret_90d"] = g.transform(lambda x: x.pct_change(90)).astype(np.float32)

    # --- Volatility Regime (vol20 / vol60) ---
    vol60 = dr.groupby(df["sym"]).transform(
        lambda x: x.rolling(60, min_periods=30).std()
    )
    df["vol_regime"] = (df["vol20"] / vol60.replace(0, np.nan)).astype(np.float32)

    # --- MA Crosses ---
    df["ma_cross_5_20"] = (df["ma5"] > df["ma20"]).astype(np.float32)
    df["ma_cross_20_60"] = (df["ma20"] > df["ma60"]).astype(np.float32)

    # --- RSI Zone (discretized) ---
    df["rsi_zone"] = (df["rsi14"] // 10).astype(np.float32)

    # --- MACD ROC ---
    df["macd_roc"] = df.groupby("sym")["macd_hist"].transform(
        lambda x: x.diff(5)
    ).astype(np.float32)

    # --- 60-day Drawdown ---
    rolling_max = g.transform(lambda x: x.rolling(60, min_periods=30).max())
    df["dd_60"] = ((df["close"] / rolling_max) - 1).astype(np.float32)

    # --- Up/Down Volume Ratio (positive return std / negative return std over 20d) ---
    pos_ret = dr.clip(lower=0)
    neg_ret = (-dr).clip(lower=0)
    pos_std = pos_ret.groupby(df["sym"]).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    neg_std = neg_ret.groupby(df["sym"]).transform(
        lambda x: x.rolling(20, min_periods=10).std()
    )
    df["ud_vol_ratio"] = (pos_std / neg_std.replace(0, np.nan)).astype(np.float32)

    # --- Fundamentals ---
    fund_cols = fund[["sym", "pe_trailing", "div_yield", "beta"]].copy()
    df = df.merge(fund_cols, on="sym", how="left")

    # pe_log: log(pe_trailing), fillna with median log or 0
    pe_positive = df["pe_trailing"].clip(lower=0.01)
    df["pe_log"] = np.log(pe_positive).astype(np.float32)
    pe_median = df["pe_log"].median()
    df["pe_log"] = df["pe_log"].fillna(pe_median if np.isfinite(pe_median) else 0).astype(np.float32)

    # div_yield: fillna 0 (known bad data)
    df["div_yield"] = df["div_yield"].fillna(0).astype(np.float32)

    # beta: fillna with median
    df["beta"] = df["beta"].clip(-3, 8).fillna(df["beta"].median()).astype(np.float32)

    # drop helper columns from fund merge
    for col in ["pe_trailing"]:
        if col in df.columns and col not in FEATURES:
            df.drop(columns=[col], inplace=True, errors="ignore")

    return df


def main():
    print("=== BlueShield V10 Quantile 评分 ===")

    # 加载模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型不存在: {MODEL_PATH}")
        sys.exit(1)

    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"   模型: {os.path.basename(MODEL_PATH)} ({model.num_feature()} 特征)")
    print(f"   类型: 分位数回归 (quantile median, alpha=0.5)")
    print(f"   OOS: IC=0.0743, ICIR=0.7335")

    # 加载价格数据
    print("   加载价格数据...")
    df = pd.read_parquet(os.path.join(ROOT, "data/us/us_hist_full_10y.parquet"))
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    cutoff = latest_date - pd.Timedelta(days=400)  # need history for rolling features
    df = df[df["date"] >= cutoff].copy()
    print(f"   日期范围: {cutoff.date()} → {latest_date.date()}")

    # 宇宙过滤：>$10 + 流动性
    latest_prices = df[df["date"] == latest_date].set_index("sym")["close"]
    df["dollar_vol"] = (df["close"] * df["volume"]).astype(np.float32)
    df["dollar_vol_20d"] = df.groupby("sym")["dollar_vol"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    ).astype(np.float32)
    latest_dv = df[df["date"] == latest_date].set_index("sym")["dollar_vol_20d"]
    temp = pd.DataFrame({"sym": latest_prices.index, "close": latest_prices.values,
                          "dollar_vol_20d": latest_dv.values}).reset_index(drop=True)
    filtered = filter_blue_shield(temp, min_price=10.0, min_dollar_vol=5_000_000)
    universe_syms = set(filtered["sym"].values)
    df = df[df["sym"].isin(universe_syms)].copy()
    print(f"   宇宙: {len(universe_syms)} 只 (>$10, 排除杠杆ETF/SPAC/低流动性)")

    # 加载基本面
    fund = pd.read_parquet(FUND_PATH)
    fund = fund[["sym", "pe_trailing", "div_yield", "beta"]].copy()

    # 计算特征
    print("   计算43个特征...")
    df = compute_features(df, fund)
    df[FEATURES] = df[FEATURES].fillna(0)

    # 只评最新日
    latest = df[df["date"] == latest_date].copy()
    print(f"   评分: {len(latest)} 只股票")

    # 预测（quantile regression 输出连续值，越大越好）
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

    # Top picks
    top_n = 20
    top_picks = latest.head(top_n)

    # 输出
    picks = []
    for _, r in top_picks.iterrows():
        picks.append({
            "sym": r["sym"],
            "close": float(r["close"]),
            "score": float(r["score"]),
            "rank_pct": float(r["rank_pct"]),
            "signal": r["signal"],
            "pct_rank": float(r["rank_pct"]) * 100,
        })

    out = {
        "timestamp": datetime.now().isoformat(),
        "date": str(latest_date.date()),
        "model": "blueshield_v10_quantile",
        "version": "V10-Quantile",
        "universe": ">$10",
        "total": len(latest),
        "picks": picks,
    }
    out_path = os.path.join(OUTPUT_DIR, f"blueshield_v10_quantile_scored_{latest_date.strftime('%Y%m%d')}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n   保存: {out_path}")

    # 打印摘要
    print(f"\n📊 Top 10:")
    for i, p in enumerate(picks[:10], 1):
        print(f"   {i:2d}. {p['sym']:6s}  ${p['close']:8.2f}  score={p['score']:.4f}  {p['signal']}")
    print(f"\n✅ 完成 — {len(latest)} 只股票评分, top1-5 signal: {sum(1 for p in picks[:5] if '🟢' in p['signal'])} green")


if __name__ == "__main__":
    main()
