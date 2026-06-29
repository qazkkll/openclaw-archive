#!/usr/bin/env python3
"""
grades_historical buy_ratio IC 验证
与现有grade_sentiment因子对比，检查是否提供增量信息
"""
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
PREMIUM_DIR = PROJECT_ROOT / "data" / "fmp_premium"

print("加载数据...")
master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
master["date"] = master["date"].astype(str)
dates = sorted(master["date"].unique())

# 构建价格矩阵
price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()

# 加载grades_historical（月度评级分布）
print("加载grades_historical...")
grades_hist = {}
raw_dir = PREMIUM_DIR / "data" / "raw"
for f in raw_dir.glob("grades_historical_symbol-*.json"):
    ticker = f.stem.split("symbol-")[1].split("_")[0]
    data = json.load(open(f))
    if isinstance(data, list) and data:
        grades_hist[ticker] = data
print(f"  覆盖: {len(grades_hist)} 只")

# 加载grades（单个评级变动，用于构造grade_sentiment作为对照）
print("加载grades...")
grades_individual = {}
for f in raw_dir.glob("grades_symbol-*.json"):
    ticker = f.stem.split("symbol-")[1].split("_")[0]
    data = json.load(open(f))
    if isinstance(data, list) and data:
        grades_individual[ticker] = data
print(f"  覆盖: {len(grades_individual)} 只")

def get_buy_ratio(ticker, date_str):
    """从grades_historical获取date之前最近的buy_ratio"""
    records = grades_hist.get(ticker, [])
    if not records:
        return None
    # 找date之前最近的月度记录
    prior = [r for r in records if r.get("date", "") <= date_str]
    if not prior:
        return None
    latest = max(prior, key=lambda r: r["date"])
    strong_buy = latest.get("analystRatingsStrongBuy", 0) or 0
    buy = latest.get("analystRatingsBuy", 0) or 0
    hold = latest.get("analystRatingsHold", 0) or 0
    sell = latest.get("analystRatingsSell", 0) or 0
    strong_sell = latest.get("analystRatingsStrongSell", 0) or 0
    total = strong_buy + buy + hold + sell + strong_sell
    if total == 0:
        return None
    return (strong_buy + buy) / total

def get_buy_ratio_trend(ticker, date_str):
    """buy_ratio的90天变化趋势"""
    records = grades_hist.get(ticker, [])
    if not records:
        return None
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
    prior = [r for r in records if start <= r.get("date", "") <= date_str]
    if len(prior) < 2:
        return None
    # 计算buy_ratio的变化
    def calc_br(r):
        sb = r.get("analystRatingsStrongBuy", 0) or 0
        b = r.get("analystRatingsBuy", 0) or 0
        h = r.get("analystRatingsHold", 0) or 0
        s = r.get("analystRatingsSell", 0) or 0
        ss = r.get("analystRatingsStrongSell", 0) or 0
        t = sb + b + h + s + ss
        return (sb + b) / t if t > 0 else None
    
    brs = [calc_br(r) for r in sorted(prior, key=lambda x: x["date"])]
    brs = [b for b in brs if b is not None]
    if len(brs) < 2:
        return None
    return brs[-1] - brs[0]  # 90天变化

def get_grade_sentiment_proxy(ticker, date_str):
    """用grades数据构造grade_sentiment代理（升级-降级比例）"""
    records = grades_individual.get(ticker, [])
    if not records:
        return None
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
    recent = [r for r in records if start <= r.get("date", "") <= date_str]
    if not recent:
        return None
    upgrades = sum(1 for r in recent if r.get("action") in ["upgrade", "initiate"])
    downgrades = sum(1 for r in recent if r.get("action") == "downgrade")
    total = len(recent)
    if total == 0:
        return None
    return (upgrades - downgrades) / total

# Walk-Forward IC计算
HOLD_DAYS = 60
rebal_dates = dates[::HOLD_DAYS]
print(f"调仓日: {len(rebal_dates)} 个")

ics_buy_ratio = []
ics_trend = []
ics_sentiment = []

for i, rebal_date in enumerate(rebal_dates[:-1]):
    next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
    if not next_rebal:
        continue
    
    br_vals = {}
    trend_vals = {}
    sent_vals = {}
    fwd_ret = {}
    
    tickers = master[master["date"] == rebal_date]["ticker"].unique()
    for ticker in tickers:
        br = get_buy_ratio(ticker, rebal_date)
        trend = get_buy_ratio_trend(ticker, rebal_date)
        sent = get_grade_sentiment_proxy(ticker, rebal_date)
        
        if br is not None:
            br_vals[ticker] = br
        if trend is not None:
            trend_vals[ticker] = trend
        if sent is not None:
            sent_vals[ticker] = sent
        
        try:
            p0 = price_pivot.loc[rebal_date, ticker]
            p1 = price_pivot.loc[next_rebal, ticker]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret[ticker] = (p1 - p0) / p0
        except:
            continue
    
    # IC for buy_ratio
    common_br = set(br_vals.keys()) & set(fwd_ret.keys())
    if len(common_br) >= 30:
        fr = pd.Series({t: br_vals[t] for t in common_br}).rank(pct=True)
        rr = pd.Series({t: fwd_ret[t] for t in common_br}).rank(pct=True)
        ic = fr.corr(rr, method="spearman")
        ics_buy_ratio.append({"date": rebal_date, "ic": ic, "n": len(common_br)})
    
    # IC for trend
    common_tr = set(trend_vals.keys()) & set(fwd_ret.keys())
    if len(common_tr) >= 30:
        fr = pd.Series({t: trend_vals[t] for t in common_tr}).rank(pct=True)
        rr = pd.Series({t: fwd_ret[t] for t in common_tr}).rank(pct=True)
        ic = fr.corr(rr, method="spearman")
        ics_trend.append({"date": rebal_date, "ic": ic, "n": len(common_tr)})
    
    # IC for sentiment proxy
    common_sent = set(sent_vals.keys()) & set(fwd_ret.keys())
    if len(common_sent) >= 30:
        fr = pd.Series({t: sent_vals[t] for t in common_sent}).rank(pct=True)
        rr = pd.Series({t: fwd_ret[t] for t in common_sent}).rank(pct=True)
        ic = fr.corr(rr, method="spearman")
        ics_sentiment.append({"date": rebal_date, "ic": ic, "n": len(common_sent)})

def print_stats(name, ics):
    df = pd.DataFrame(ics)
    if len(df) == 0:
        print(f"\n{name}: 无数据")
        return
    mean_ic = df['ic'].mean()
    std_ic = df['ic'].std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    t_stat = mean_ic / (std_ic / np.sqrt(len(df))) if std_ic > 0 else 0
    win = (df['ic'] > 0).mean()
    print(f"\n=== {name} ===")
    print(f"样本: {len(df)} 个调仓窗口")
    print(f"平均IC: {mean_ic:.4f}")
    print(f"IC标准差: {std_ic:.4f}")
    print(f"ICIR: {icir:.4f}")
    print(f"t-stat: {t_stat:.3f}")
    print(f"IC>0 胜率: {win:.1%}")
    return icir

icir_br = print_stats("buy_ratio (StrongBuy+Buy比例)", ics_buy_ratio)
icir_trend = print_stats("buy_ratio_90d变化趋势", ics_trend)
icir_sent = print_stats("grade_sentiment代理 (升级-降级)", ics_sentiment)

print("\n=== 结论 ===")
print(f"buy_ratio ICIR: {icir_br:.4f}")
print(f"buy_ratio趋势 ICIR: {icir_trend:.4f}")
print(f"grade_sentiment代理 ICIR: {icir_sent:.4f}")

# 交叉相关性
if ics_buy_ratio and ics_trend:
    df_br = pd.DataFrame(ics_buy_ratio).set_index("date")
    df_tr = pd.DataFrame(ics_trend).set_index("date")
    common_dates = set(df_br.index) & set(df_tr.index)
    if len(common_dates) > 5:
        corr = df_br.loc[list(common_dates), "ic"].corr(df_tr.loc[list(common_dates), "ic"])
        print(f"\nbuy_ratio vs 趋势 IC相关性: {corr:.3f}")
        if abs(corr) > 0.7:
            print("⚠️ 高度相关，不建议同时使用")
        else:
            print("✅ 相关性不高，可考虑组合使用")
