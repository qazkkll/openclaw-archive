#!/usr/bin/env python3
"""
🦅 Project Falcon V0.2 — 全特征版
===================================
核心变更: 从7个粗糙因子 → 60+因子(FMP全部65字段+43技术+FinBERT+分析师)
权重逻辑: 花钱买的(FMP/Massive)占主导权重
"""

import sys, os, json, time, argparse, warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

from dotenv import load_dotenv
load_dotenv(ENV_PATH)

MASSIVE_KEY = os.environ.get("MASSIVE_API_KEY", "")
FMP_KEY = os.environ.get("FMP_API_KEY", "")

DATA_DIR = PROJECT_ROOT / "data" / "falcon"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════
# 特征定义
# ═══════════════════════════════════════════════════

# FMP Ratios — 选最有alpha潜力的字段(避免冗余)
FMP_VALUATION = ['priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
                 'priceToFreeCashFlowRatio', 'enterpriseValueMultiple']
FMP_PROFIT = ['grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin', 'ebitdaMargin']
FMP_EFFICIENCY = ['assetTurnover', 'inventoryTurnover', 'receivablesTurnover']
FMP_LEVERAGE = ['debtToEquityRatio', 'currentRatio', 'quickRatio', 'financialLeverageRatio']
FMP_CASHFLOW = ['freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio']
FMP_DIVIDEND = ['dividendYieldPercentage', 'dividendPayoutRatio']

FMP_ALL_FIELDS = FMP_VALUATION + FMP_PROFIT + FMP_EFFICIENCY + FMP_LEVERAGE + FMP_CASHFLOW + FMP_DIVIDEND


# ═══════════════════════════════════════════════════
# Phase 1: 数据拉取
# ═══════════════════════════════════════════════════

def load_sp500():
    f = PROJECT_ROOT / "config" / "universe_sp500.csv"
    df = pd.read_csv(f)
    return sorted(df["ticker"].dropna().unique().tolist())


def fetch_daily(ticker, start, end):
    import urllib.request
    url = (f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
           f"/range/1/day/{start}/{end}"
           f"?adjusted=true&sort=asc&limit=50000&apiKey={MASSIVE_KEY}")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception:
        return pd.DataFrame()
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                            "v": "volume", "vw": "vwap", "t": "timestamp_ms"})
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
    df["ticker"] = ticker
    return df[["ticker", "date", "open", "high", "low", "close", "volume", "vwap"]]


def fetch_fmp_ratios(ticker):
    """拉取FMP全部65个财务比率。"""
    import urllib.request
    url = f"https://financialmodelingprep.com/stable/ratios?symbol={ticker}&period=quarter&limit=8&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception:
        return {}
    if not isinstance(data, list) or not data:
        return {}
    latest = data[0]
    # 提取所有alpha字段
    result = {}
    for field in FMP_ALL_FIELDS:
        result[field] = latest.get(field)
    # 额外: 计算趋势(最新vs上一季度)
    if len(data) >= 2:
        prev = data[1]
        for field in FMP_PROFIT:
            curr_val = latest.get(field)
            prev_val = prev.get(field)
            if curr_val is not None and prev_val is not None and prev_val != 0:
                result[f"{field}_qoq"] = (curr_val - prev_val) / abs(prev_val)
    return result


def fetch_fmp_analyst(ticker):
    """拉取FMP分析师估计。"""
    import urllib.request
    url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol={ticker}&period=quarter&limit=8&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception:
        return {}
    if not isinstance(data, list) or len(data) < 2:
        return {}
    # 最新vs上一季度 → 修正动量
    latest = data[0]
    prev = data[1]
    result = {}
    # EPS修正
    eps_now = latest.get("epsAvg")
    eps_prev = prev.get("epsAvg")
    if eps_now and eps_prev and eps_prev != 0:
        result["eps_revision"] = (eps_now - eps_prev) / abs(eps_prev)
    # 收入修正
    rev_now = latest.get("revenueAvg")
    rev_prev = prev.get("revenueAvg")
    if rev_now and rev_prev and rev_prev != 0:
        result["revenue_revision"] = (rev_now - rev_prev) / abs(rev_prev)
    # 分析师覆盖度
    result["num_analysts_eps"] = latest.get("numAnalystsEps", 0)
    result["num_analysts_rev"] = latest.get("numAnalystsRevenue", 0)
    # EPS范围(分歧度)
    eps_low = latest.get("epsLow")
    eps_high = latest.get("epsHigh")
    if eps_now and eps_now != 0 and eps_low is not None and eps_high is not None:
        result["eps_dispersion"] = (eps_high - eps_low) / abs(eps_now)
    return result


def fetch_fmp_news(ticker, start, end):
    import urllib.request
    articles = []
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    current = s
    while current < e:
        slice_end = min(current + timedelta(days=180), e)
        url = (f"https://financialmodelingprep.com/stable/news/stock?"
               f"symbols={ticker}&from={current.strftime('%Y-%m-%d')}&to={slice_end.strftime('%Y-%m-%d')}"
               f"&limit=100&apikey={FMP_KEY}")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            if isinstance(data, list):
                for a in data:
                    articles.append({"ticker": ticker, "title": a.get("title", ""),
                                     "text": a.get("text", ""), "published_at": a.get("publishedDate", "")})
        except Exception:
            pass
        current = slice_end
        time.sleep(0.05)
    return articles


# ═══════════════════════════════════════════════════
# 特征计算
# ═══════════════════════════════════════════════════

def compute_v10_tech_features(df):
    """V10的43个技术特征，全部从K线计算。"""
    df = df.sort_values("date").copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # ── 均线 ──
    for w in [5, 20, 60]:
        df[f"ma{w}"] = c.rolling(w, min_periods=w).mean()
    df["ma_bias20"] = (c - df["ma20"]) / df["ma20"]
    # MA排列: ma5 > ma20 > ma60 = 1, 否则逐步递减
    df["ma_align"] = ((df["ma5"] > df["ma20"]).astype(float) +
                      (df["ma20"] > df["ma60"]).astype(float)) / 2
    df["ma_cross_5_20"] = (df["ma5"] > df["ma20"]).astype(float)
    df["ma_cross_20_60"] = (df["ma20"] > df["ma60"]).astype(float)
    df["price_position"] = (c - l.rolling(60).min()) / (h.rolling(60).max() - l.rolling(60).min() + 1e-10)

    # ── 收益率 ──
    for w in [1, 5, 10, 20, 30, 60, 90]:
        df[f"ret{w}"] = c.pct_change(w)
    df["momentum_6m"] = c.pct_change(126)
    df["momentum_1m"] = c.pct_change(21)
    df["mom_divergence"] = df["momentum_1m"] - df["momentum_6m"]
    df["trend_accel"] = df["ret20"] - df["ret60"]

    # ── 波动率 ──
    daily_ret = c.pct_change()
    df["vol20"] = daily_ret.rolling(20, min_periods=15).std() * np.sqrt(252)
    df["vol5"] = daily_ret.rolling(5, min_periods=3).std() * np.sqrt(252)
    df["vol_ratio"] = df["vol5"] / df["vol20"].replace(0, np.nan)
    df["vol_change"] = df["vol20"].pct_change(20)
    df["vol_regime"] = (df["vol20"] > df["vol20"].rolling(60).mean()).astype(float)

    # ── RSI ──
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=10).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    df["rsi_change"] = df["rsi14"].diff(5)
    df["rsi_zone"] = (df["rsi14"] > 70).astype(float) - (df["rsi14"] < 30).astype(float)

    # ── MACD ──
    ema12 = c.ewm(span=12, min_periods=10).mean()
    ema26 = c.ewm(span=26, min_periods=20).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, min_periods=7).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_roc"] = df["macd_hist"].diff(5)

    # ── 布林带 ──
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_std"] = bb_std
    df["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
    df["bb_pos"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # ── 质量指标 ──
    df["ret_quality"] = daily_ret.rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
    body = abs(c - o)
    df["range_ratio"] = (h - l) / c.replace(0, np.nan)
    df["avg_body"] = body.rolling(20).mean() / c.replace(0, np.nan)
    df["vwap_drift"] = (c - df["vwap"]) / df["vwap"].replace(0, np.nan) if "vwap" in df.columns else 0

    # ── 回撤 ──
    peak_60 = c.rolling(60, min_periods=20).max()
    df["dd_60"] = (c - peak_60) / peak_60.replace(0, np.nan)

    # ── 涨跌量比 ──
    up_vol = v.where(daily_ret > 0, 0).rolling(20).sum()
    dn_vol = v.where(daily_ret < 0, 0).rolling(20).sum()
    df["ud_vol_ratio"] = up_vol / dn_vol.replace(0, np.nan)

    # ── Beta(从价格算) ──
    # 需要SPY，先跳过，后面单独算

    return df


def compute_beta(stock_ret, spy_ret, window=504):
    """从价格计算Beta: cov(stock, spy) / var(spy)。"""
    cov = stock_ret.rolling(window, min_periods=252).cov(spy_ret)
    var = spy_ret.rolling(window, min_periods=252).var()
    beta = cov / var.replace(0, np.nan)
    return beta.clip(-2, 5)


def winsorize_series(s, lower=0.01, upper=0.99):
    """Winsorize极端值。"""
    ql = s.quantile(lower)
    qu = s.quantile(upper)
    return s.clip(ql, qu)


def rank_normalize(s):
    """截面rank标准化到[0,1]。"""
    return s.rank(pct=True)


# ═══════════════════════════════════════════════════
# Phase 1: 全量数据准备
# ═══════════════════════════════════════════════════

def prepare_all_data(tickers, start, end):
    """拉取全量数据: K线 + FMP全部特征 + 分析师 + 新闻。"""
    kline_dir = DATA_DIR / "klines"
    kline_dir.mkdir(exist_ok=True)

    # ── K线 ──
    print(f"📥 拉取 {len(tickers)} 只日K线 ({start} → {end})...")
    all_klines = {}

    def _fetch(t):
        cached = kline_dir / f"{t}.parquet"
        if cached.exists():
            return t, pd.read_parquet(cached)
        df = fetch_daily(t, start, end)
        if not df.empty:
            df.to_parquet(cached, index=False)
        return t, df

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_fetch, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            t, df = f.result()
            done += 1
            if not df.empty:
                all_klines[t] = df
            if done % 100 == 0:
                print(f"  {done}/{len(tickers)}...")

    # 过滤完整3年
    valid = []
    for t, df in all_klines.items():
        ds = df["date"].astype(str)
        if (ds <= "2022-02-01").any() and (ds >= "2024-11-01").any() and len(df) >= 600:
            valid.append(t)
    print(f"✅ K线: {len(valid)} 只完整3年")

    # ── FMP全部特征 ──
    fmp_file = DATA_DIR / "fmp_full.json"
    if fmp_file.exists():
        with open(fmp_file) as f:
            fmp_data = json.load(f)
        print(f"📂 FMP缓存: {len(fmp_data)} 只")
    else:
        print(f"📥 拉取 {len(valid)} 只FMP全量特征(65字段)...")
        fmp_data = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(fetch_fmp_ratios, t): t for t in valid}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    fmp_data[t] = f.result()
                except:
                    fmp_data[t] = {}
        with open(fmp_file, "w") as f:
            json.dump(fmp_data, f)
        has_data = sum(1 for v in fmp_data.values() if v)
        print(f"✅ FMP: {has_data}/{len(valid)} 只有数据")

    # ── 分析师估计 ──
    analyst_file = DATA_DIR / "analyst.json"
    if analyst_file.exists():
        with open(analyst_file) as f:
            analyst_data = json.load(f)
        print(f"📂 分析师缓存: {len(analyst_data)} 只")
    else:
        print(f"📥 拉取 {len(valid)} 只分析师估计...")
        analyst_data = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(fetch_fmp_analyst, t): t for t in valid}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    analyst_data[t] = f.result()
                except:
                    analyst_data[t] = {}
        with open(analyst_file, "w") as f:
            json.dump(analyst_data, f)
        has_data = sum(1 for v in analyst_data.values() if v)
        print(f"✅ 分析师: {has_data}/{len(valid)} 只有数据")

    # ── 计算全部特征 ──
    print("📊 计算全部特征(43技术 + FMP + 分析师)...")
    feat_file = DATA_DIR / "features_v02.parquet"

    # 先算SPY beta
    spy_kline = all_klines.get("SPY")
    if spy_kline is not None:
        spy_kline = spy_kline.sort_values("date")
        spy_ret = spy_kline["close"].pct_change()
        spy_ret.index = spy_kline["date"].values
    else:
        spy_ret = None

    all_feats = []
    for t in valid:
        df = all_klines[t].copy()

        # 43个技术特征
        df = compute_v10_tech_features(df)

        # Beta
        if spy_ret is not None:
            stock_ret = df["close"].pct_change()
            stock_ret.index = df["date"].values
            # 对齐日期
            common_dates = stock_ret.index.intersection(spy_ret.index)
            if len(common_dates) > 252:
                sr = stock_ret.reindex(common_dates)
                spr = spy_ret.reindex(common_dates)
                beta = compute_beta(sr, spr)
                beta.index = common_dates
                df["beta"] = df["date"].map(beta)
            else:
                df["beta"] = np.nan
        else:
            df["beta"] = np.nan

        # FMP全部特征(合并到每日行 — 季度数据forward-fill)
        fmp = fmp_data.get(t, {})
        for field in FMP_ALL_FIELDS:
            df[field] = fmp.get(field)
        # QoQ趋势
        for field in FMP_PROFIT:
            df[f"{field}_qoq"] = fmp.get(f"{field}_qoq")

        # 分析师特征
        analyst = analyst_data.get(t, {})
        for field in ["eps_revision", "revenue_revision", "num_analysts_eps",
                       "num_analysts_rev", "eps_dispersion"]:
            df[field] = analyst.get(field)

        # 标记是否FMP覆盖
        df["fmp_covered"] = 1 if fmp else 0
        df["analyst_covered"] = 1 if analyst else 0

        all_feats.append(df)

    master = pd.concat(all_feats, ignore_index=True)
    master.to_parquet(feat_file, index=False)
    print(f"✅ 特征矩阵: {len(master)} 行, {len(valid)} 只, {len(master.columns)} 列")
    return valid


# ═══════════════════════════════════════════════════
# Phase 2: 综合评分
# ═══════════════════════════════════════════════════

def compute_composite_score(master, date, w_tech, w_fund, w_analyst):
    """
    计算每日综合得分。
    有FMP覆盖的股票: tech + fund(全部FMP) + analyst
    无覆盖的: 仅tech(降权)
    """
    day = master[master["date"].astype(str) == date].copy()
    if day.empty:
        return pd.DataFrame()

    # ── 技术得分: rank normalize → [0,1] ──
    tech_fields = ["rsi14", "macd_hist", "momentum_1m", "vol20", "bb_pos",
                   "ma_align", "ret_quality", "dd_60", "ud_vol_ratio"]
    for f in tech_fields:
        if f in day.columns:
            day[f"_rank"] = rank_normalize(day[f].fillna(day[f].median()))
    rank_cols = [f"_rank" for f in tech_fields if f in day.columns]
    day["tech_score"] = day[rank_cols].mean(axis=1)

    # ── 基本面得分: FMP全部特征rank ──
    fund_fields = [f for f in FMP_ALL_FIELDS if f in day.columns and day[f].notna().sum() > 10]
    for f in fund_fields:
        day[f"_frank"] = rank_normalize(day[f].fillna(day[f].median()))
    frank_cols = [f"_frank" for f in fund_fields]
    if frank_cols:
        day["fund_score"] = day[frank_cols].mean(axis=1)
    else:
        day["fund_score"] = 0.5

    # ── 分析师得分 ──
    analyst_fields = ["eps_revision", "revenue_revision", "eps_dispersion"]
    for f in analyst_fields:
        if f in day.columns and day[f].notna().sum() > 5:
            day[f"_arank"] = rank_normalize(day[f].fillna(day[f].median()))
    arank_cols = [f"_arank" for f in analyst_fields if f"_arank" in day.columns]
    if arank_cols:
        day["analyst_score"] = day[arank_cols].mean(axis=1)
    else:
        day["analyst_score"] = 0.5

    # ── 综合得分 ──
    # FMP覆盖的: 全权重
    # 无覆盖的: 只有tech, fund=0.5(中性)
    has_fmp = day["fmp_covered"] == 1
    day.loc[has_fmp, "composite"] = (
        w_tech * day.loc[has_fmp, "tech_score"] +
        w_fund * day.loc[has_fmp, "fund_score"] +
        w_analyst * day.loc[has_fmp, "analyst_score"]
    )
    day.loc[~has_fmp, "composite"] = (
        w_tech * day.loc[~has_fmp, "tech_score"] +
        (1 - w_tech) * 0.5  # 无FMP数据的，基本面和分析师用中性值
    )

    return day[["ticker", "composite", "tech_score", "fund_score", "analyst_score", "fmp_covered"]].dropna(subset=["composite"])


# ═══════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════

def vectorized_backtest(master, start, end, wt, wf, wa,
                        top_n=5, hold_days=10, stop_loss=-0.10, cost=0.0045):
    """向量化回测。"""
    dates_in_range = sorted(master[master["date"].astype(str).between(start, end)]["date"].astype(str).unique())
    if len(dates_in_range) < hold_days + 10:
        return None

    price_pivot = master[master["date"].astype(str).between(start, end)].pivot_table(
        index="date", columns="ticker", values="close").sort_index()

    cash = 100000.0
    portfolio = {}
    values = []
    trades = []

    for i, date in enumerate(dates_in_range):
        if date not in price_pivot.index:
            continue
        prices = price_pivot.loc[date]

        # 止损/到期
        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in prices and not pd.isna(prices[t]):
                pnl = (prices[t] - ep) / ep
                if pnl <= stop_loss:
                    cash += sh * prices[t] * (1 - cost)
                    trades.append({"pnl": pnl - 2*cost, "reason": "止损"})
                    to_close.append(t)
                elif (i - ei) >= hold_days:
                    cash += sh * prices[t] * (1 - cost)
                    trades.append({"pnl": pnl - 2*cost, "reason": "到期"})
                    to_close.append(t)
        for t in to_close:
            del portfolio[t]

        # 轮换
        if len(portfolio) == 0 and cash > 0:
            scores = compute_composite_score(master, date, wt, wf, wa)
            if scores.empty:
                continue
            scores = scores.sort_values("composite", ascending=False)
            picks = scores.head(top_n)
            per = cash / len(picks) if len(picks) > 0 else 0
            for _, row in picks.iterrows():
                t = row["ticker"]
                if t in prices and not pd.isna(prices[t]) and prices[t] > 0:
                    sh = (per * (1 - cost)) / prices[t]
                    portfolio[t] = (i, prices[t], sh)
            cash = 0.0

        pv = cash
        for t, (_, ep, sh) in portfolio.items():
            pv += sh * (prices[t] if t in prices and not pd.isna(prices[t]) else ep)
        values.append(pv)

    if len(values) < 20:
        return None

    v = np.array(values, dtype=np.float64)
    rets = np.zeros(len(v) - 1)
    for j in range(1, len(v)):
        rets[j-1] = (v[j] - v[j-1]) / v[j-1] if v[j-1] > 0 else 0

    ret_std = np.std(rets)
    if ret_std == 0 or np.isnan(ret_std):
        return None

    sharpe = np.mean(rets) / ret_std * np.sqrt(252)
    total_return = (v[-1] / v[0] - 1) * 100
    peak = np.maximum.accumulate(v)
    max_dd = ((peak - v) / peak).max() * 100
    win_trades = [t for t in trades if t["pnl"] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "max_dd": round(float(max_dd), 2),
        "total_return": round(float(total_return), 2),
        "win_rate": round(float(win_rate), 1),
        "trades": len(trades),
    }


# ═══════════════════════════════════════════════════
# Grid Search + Stress Test + Verdict
# ═══════════════════════════════════════════════════

def grid_search(master, start, end, cost, top_n, hold_days, stop_loss):
    step = 0.05
    combos = []
    for wt in np.arange(0, 1.01, step):
        for wf in np.arange(0, 1.01 - wt + 0.001, step):
            wa = round(1.0 - wt - wf, 2)
            if 0 <= wa <= 1:
                combos.append((round(wt, 2), round(wf, 2), wa))
    combos = list(set(combos))
    combos.sort()
    print(f"🔍 网格搜索: {len(combos)} 组, {start} → {end}")

    results = []
    for i, (wt, wf, wa) in enumerate(combos):
        res = vectorized_backtest(master, start, end, wt, wf, wa, top_n, hold_days, stop_loss, cost)
        if res:
            results.append({"Wt": wt, "Wf": wf, "Wa": wa, **res})
        if (i+1) % 50 == 0:
            best = max(results, key=lambda x: x["sharpe"]) if results else None
            print(f"  {i+1}/{len(combos)}: best={best['sharpe'] if best else 'N/A'}")

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="🦅 Falcon V0.2")
    parser.add_argument("--phase", default="all", choices=["data", "search", "all"])
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--stop-loss", type=float, default=-0.10)
    parser.add_argument("--cost", type=float, default=0.0045)
    parser.add_argument("--invert", action="store_true", help="反转信号方向")
    args = parser.parse_args()

    t0 = time.time()
    tickers = load_sp500()
    print(f"🦅 Falcon V0.2 | {len(tickers)} 只 | top_n={args.top_n} hold={args.hold_days} SL={args.stop_loss} cost={args.cost*100}%")
    print(f"   特征: 43技术 + 20 FMP + 5分析师 + 覆盖标记")

    feat_file = DATA_DIR / "features_v02.parquet"
    if args.phase in ("data", "all") or not feat_file.exists():
        valid = prepare_all_data(tickers, "2022-01-01", "2024-12-31")
        if args.phase == "data":
            return

    master = pd.read_parquet(feat_file)
    master["date"] = master["date"].astype(str)
    n_tickers = master["ticker"].nunique()
    fmp_cov = master.groupby("ticker")["fmp_covered"].first().sum()
    print(f"📊 加载: {len(master)} 行, {n_tickers} 只, FMP覆盖: {fmp_cov}")

    if args.phase == "data":
        return

    # 测试多种hold_days
    for hold in [10, 20, 30, 60]:
        print(f"\n{'='*60}")
        print(f"🔍 Hold={hold}天, Top-N={args.top_n}, Cost={args.cost*100}%")
        print(f"{'='*60}")

        bull = grid_search(master, "2023-01-01", "2024-12-31", args.cost,
                           args.top_n, hold, args.stop_loss)
        if not bull:
            print("❌ 无有效结果")
            continue

        # Top-5
        print(f"\n🏆 牛市Top-5:")
        for r in bull[:5]:
            print(f"   Wt={r['Wt']:.2f} Wf={r['Wf']:.2f} Wa={r['Wa']:.2f} → Sharpe={r['sharpe']:.3f} DD={r['max_dd']:.1f}% WR={r['win_rate']:.0f}%")

        # 熊市压测Top-1
        best = bull[0]
        bear = vectorized_backtest(master, "2022-01-01", "2022-12-31",
                                   best["Wt"], best["Wf"], best["Wa"],
                                   args.top_n, hold, args.stop_loss, args.cost)
        if bear:
            print(f"🐻 熊市: Sharpe={bear['sharpe']:.3f} DD={bear['max_dd']:.1f}% WR={bear['win_rate']:.0f}%")
            passed_bear = bear["max_dd"] <= 28 and bear["win_rate"] >= 42
            print(f"   {'✅ 通过' if passed_bear else '❌ 失败'}")
        else:
            print("🐻 熊市: 无结果")

    elapsed = time.time() - t0
    print(f"\n⏱️ {elapsed:.0f}秒")


if __name__ == "__main__":
    main()
