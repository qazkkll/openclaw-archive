#!/usr/bin/env python3
"""
Project Mercurius — Night 1+2 全栈闭环
========================================
Step 1: 生成 Universe
Step 2: 拉取日K + 基本面 + 新闻 (Massive + FMP)
Step 3: 特征工程 + FinBERT 打标
Step 4: Walk-Forward 回测 + 归因
Step 5: 权重优化（Hermes 介入）
"""

import sys, os, json, time, glob, argparse, hashlib
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

from dotenv import load_dotenv
load_dotenv(ENV_PATH)

# ═══════════════════════════════════════════════════
# Step 1: Universe 生成
# ═══════════════════════════════════════════════════

def generate_universe(mode="scored20"):
    """从 scored JSON 提取 ticker，生成 universe CSV。"""
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    
    tickers = set()
    
    # 读取 signals/us/ 下的 scored 文件
    signals_dir = PROJECT_ROOT / "signals" / "us"
    if signals_dir.exists():
        for f in signals_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            picks = data.get("picks", [])
            for p in picks:
                t = p.get("ticker") or p.get("sym") or p.get("symbol", "")
                if t:
                    tickers.add(t.upper())
    
    # 也读 output/ 下的
    output_dir = PROJECT_ROOT / "output"
    for fname in ["shield_scores.json", "arrow_scores.json"]:
        f = output_dir / fname
        if f.exists():
            with open(f) as fh:
                data = json.load(fh)
            picks = data.get("picks", data.get("scores", []))
            for p in picks:
                t = p.get("ticker") or p.get("sym") or p.get("symbol", "")
                if t:
                    tickers.add(t.upper())
    
    tickers = sorted(tickers)
    
    # 加 SPY 和 QQQ 作为基准
    for bench in ["SPY", "QQQ"]:
        if bench not in tickers:
            tickers.append(bench)
    tickers = sorted(tickers)
    
    out_file = config_dir / f"universe_{mode}.csv"
    with open(out_file, "w") as f:
        f.write("ticker\n")
        for t in tickers:
            f.write(f"{t}\n")
    
    print(f"✅ Universe: {len(tickers)} tickers → {out_file}")
    return tickers


def load_universe(mode="scored20"):
    """加载 universe。"""
    f = PROJECT_ROOT / "config" / f"universe_{mode}.csv"
    if not f.exists():
        return generate_universe(mode)
    df = pd.read_csv(f)
    return sorted(df["ticker"].dropna().unique().tolist())


# ═══════════════════════════════════════════════════
# Step 2: 数据拉取
# ═══════════════════════════════════════════════════

def fetch_massive_daily(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """从 Massive 拉取日K线。"""
    import urllib.request
    
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
        f"/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        return pd.DataFrame()
    
    results = data.get("results", [])
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vw": "vwap", "t": "timestamp_ms", "n": "transactions"
    })
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
    df["ticker"] = ticker
    
    cols = ["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"]
    return df[[c for c in cols if c in df.columns]]


def fetch_fmp_ratios(ticker: str, api_key: str) -> pd.DataFrame:
    """从 FMP 拉取财务比率。"""
    import urllib.request
    
    url = f"https://financialmodelingprep.com/stable/ratios?symbol={ticker}&period=quarter&limit=12&apikey={api_key}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except:
        return pd.DataFrame()
    
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    df["ticker"] = ticker
    return df


def fetch_fmp_analyst(ticker: str, api_key: str) -> pd.DataFrame:
    """从 FMP 拉取分析师评级。"""
    import urllib.request
    
    url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol={ticker}&period=quarter&limit=12&apikey={api_key}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except:
        return pd.DataFrame()
    
    if not isinstance(data, list) or not data:
        return pd.DataFrame()
    
    return pd.DataFrame(data)


def fetch_fmp_news(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """从 FMP 拉取新闻（按半年切片）。"""
    import urllib.request
    
    articles = []
    # 半年切片
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    
    current = s
    while current < e:
        slice_end = min(current + timedelta(days=180), e)
        url = (
            f"https://financialmodelingprep.com/stable/news/stock?"
            f"symbols={ticker}&from={current.strftime('%Y-%m-%d')}&to={slice_end.strftime('%Y-%m-%d')}"
            f"&limit=100&apikey={api_key}"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
            if isinstance(data, list):
                for a in data:
                    articles.append({
                        "ticker": ticker,
                        "title": a.get("title", ""),
                        "text": a.get("text", ""),
                        "published_at": a.get("publishedDate", ""),
                        "publisher": a.get("publisher", ""),
                        "source": a.get("site", ""),
                    })
        except:
            pass
        
        current = slice_end
        time.sleep(0.08)
    
    return pd.DataFrame(articles) if articles else pd.DataFrame()


def fetch_massive_news(ticker: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    """从 Massive 拉取新闻。"""
    import urllib.request
    
    articles = []
    url = (
        f"https://api.polygon.io/v2/reference/news?"
        f"ticker={ticker}&published_utc.gte={start}&published_utc.lt={end}"
        f"&limit=100&order=desc&sort=published_utc&apiKey={api_key}"
    )
    
    page = 0
    while url and page < 5:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
        except:
            break
        
        for a in data.get("results", []):
            articles.append({
                "ticker": ticker,
                "title": a.get("title", ""),
                "text": a.get("description", ""),
                "published_at": a.get("published_utc", ""),
                "publisher": a.get("publisher", {}).get("name", ""),
                "source": a.get("article_url", ""),
            })
        
        next_url = data.get("next_url")
        if next_url:
            url = f"{next_url}&apiKey={api_key}"
            page += 1
            time.sleep(0.08)
        else:
            break
    
    return pd.DataFrame(articles) if articles else pd.DataFrame()


def step2_fetch_all(tickers: list, start: str = "2023-01-01", end: str = "2024-12-31"):
    """并行拉取所有数据。"""
    massive_key = os.environ.get("MASSIVE_API_KEY", "")
    fmp_key = os.environ.get("FMP_API_KEY", "")
    
    total = len(tickers)
    print(f"\n{'='*60}")
    print(f"📥 Step 2: 拉取数据 ({total} tickers, {start} → {end})")
    print(f"{'='*60}")
    
    # ── 2a: Massive 日K线 ──
    print(f"\n  📈 Massive 日K线...")
    kline_dir = PROJECT_ROOT / "data" / "raw" / "massive" / "daily"
    kline_dir.mkdir(parents=True, exist_ok=True)
    
    fetched = 0
    failed = []
    for i, ticker in enumerate(tickers):
        if i % 20 == 0 and i > 0:
            print(f"    {i}/{total} tickers...")
        
        out_file = kline_dir / f"{ticker}.parquet"
        if out_file.exists():
            fetched += 1
            continue  # 幂等跳过
        
        df = fetch_massive_daily(ticker, start, end, massive_key)
        if not df.empty:
            df.to_parquet(out_file, index=False)
            fetched += 1
        else:
            failed.append(ticker)
        
        time.sleep(0.08)  # 800/min limit
    
    print(f"  ✅ 日K线: {fetched}/{total} 成功, {len(failed)} 失败")
    if failed:
        with open(PROJECT_ROOT / "data" / "missing_kline.log", "w") as f:
            f.write("\n".join(failed))
    
    # ── 2b: FMP 基本面 ──
    print(f"\n  📊 FMP 基本面...")
    ratios_dir = PROJECT_ROOT / "data" / "raw" / "fmp" / "ratios"
    ratios_dir.mkdir(parents=True, exist_ok=True)
    
    analyst_dir = PROJECT_ROOT / "data" / "raw" / "fmp" / "analyst"
    analyst_dir.mkdir(parents=True, exist_ok=True)
    
    fetched_ratios = 0
    fetched_analyst = 0
    for i, ticker in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            print(f"    {i}/{total} tickers...")
        
        # Ratios
        out_r = ratios_dir / f"{ticker}.parquet"
        if not out_r.exists():
            df_r = fetch_fmp_ratios(ticker, fmp_key)
            if not df_r.empty:
                df_r.to_parquet(out_r, index=False)
                fetched_ratios += 1
            time.sleep(0.08)
        else:
            fetched_ratios += 1
        
        # Analyst
        out_a = analyst_dir / f"{ticker}.parquet"
        if not out_a.exists():
            df_a = fetch_fmp_analyst(ticker, fmp_key)
            if not df_a.empty:
                df_a.to_parquet(out_a, index=False)
                fetched_analyst += 1
            time.sleep(0.08)
        else:
            fetched_analyst += 1
    
    print(f"  ✅ Ratios: {fetched_ratios}/{total}, Analyst: {fetched_analyst}/{total}")
    
    # ── 2c: FMP + Massive 新闻 ──
    print(f"\n  📰 新闻...")
    news_dir = PROJECT_ROOT / "data" / "raw" / "fmp" / "news"
    news_dir.mkdir(parents=True, exist_ok=True)
    
    massive_news_dir = PROJECT_ROOT / "data" / "raw" / "massive" / "news"
    massive_news_dir.mkdir(parents=True, exist_ok=True)
    
    fetched_news = 0
    for i, ticker in enumerate(tickers):
        if i % 20 == 0 and i > 0:
            print(f"    {i}/{total} tickers...")
        
        out_fmp = news_dir / f"{ticker}.parquet"
        out_massive = massive_news_dir / f"{ticker}.parquet"
        
        # FMP news
        if not out_fmp.exists():
            df_n = fetch_fmp_news(ticker, start, end, fmp_key)
            if not df_n.empty:
                df_n.to_parquet(out_fmp, index=False)
            time.sleep(0.08)
        
        # Massive news
        if not out_massive.exists():
            df_m = fetch_massive_news(ticker, start, end, massive_key)
            if not df_m.empty:
                df_m.to_parquet(out_massive, index=False)
            time.sleep(0.08)
        
        fetched_news += 1
    
    print(f"  ✅ 新闻: {fetched_news}/{total} tickers")
    
    # ── 完整性校验 ──
    print(f"\n  🔍 完整性校验...")
    missing = []
    low_quality = []
    for ticker in tickers:
        kfile = kline_dir / f"{ticker}.parquet"
        if not kfile.exists():
            missing.append(ticker)
            continue
        df = pd.read_parquet(kfile)
        trading_days = len(df)
        # 2023+2024 约 504 个交易日
        if trading_days < 480:  # 504 * 0.95
            low_quality.append((ticker, trading_days))
    
    print(f"  完整: {total - len(missing) - len(low_quality)}/{total}")
    if missing:
        print(f"  缺失: {len(missing)} — {missing[:10]}")
    if low_quality:
        print(f"  低质量: {len(low_quality)} — {low_quality[:5]}")
    
    return {
        "kline_ok": fetched,
        "ratios_ok": fetched_ratios,
        "analyst_ok": fetched_analyst,
        "news_ok": fetched_news,
        "missing": missing,
        "low_quality": low_quality,
    }


# ═══════════════════════════════════════════════════
# Step 3: 特征工程 + FinBERT
# ═══════════════════════════════════════════════════

_finbert_model = None
_finbert_tokenizer = None

def load_finbert():
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        print("  🧠 加载 FinBERT...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return _finbert_tokenizer, _finbert_model


def compute_technical(kline_file: Path) -> pd.DataFrame:
    """计算技术指标。"""
    df = pd.read_parquet(kline_file)
    if len(df) < 60:
        return pd.DataFrame()
    
    df = df.sort_values("date").copy()
    df["date"] = pd.to_datetime(df["date"])
    c = df["close"]
    
    # MA
    df["ma20"] = c.rolling(20).mean()
    df["ma60"] = c.rolling(60).mean()
    
    # RSI(14)
    delta = c.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    
    # MACD(12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Bollinger Bands(20,2)
    df["bb_mid"] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pos"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # 动量
    df["ret_5d"] = c.pct_change(5)
    df["ret_20d"] = c.pct_change(20)
    
    # 波动率
    df["vol_20d"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    
    # 成交量特征
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    
    return df


def compute_sentiment(ticker: str) -> pd.DataFrame:
    """FinBERT 打标新闻，按日聚合。"""
    import torch
    
    news_files = []
    for d in [PROJECT_ROOT / "data" / "raw" / "fmp" / "news",
              PROJECT_ROOT / "data" / "raw" / "massive" / "news"]:
        f = d / f"{ticker}.parquet"
        if f.exists():
            news_files.append(f)
    
    if not news_files:
        return pd.DataFrame()
    
    # 合并新闻
    dfs = []
    for f in news_files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except:
            pass
    
    if not dfs:
        return pd.DataFrame()
    
    news = pd.concat(dfs, ignore_index=True)
    
    # 去重
    news["title_norm"] = news["title"].str.lower().str.strip()
    news = news.drop_duplicates(subset=["title_norm"])
    news = news[news["title"].str.len() > 10]
    
    if news.empty:
        return pd.DataFrame()
    
    # FinBERT 打分
    tokenizer, model = load_finbert()
    
    texts = (news["title"].fillna("") + ". " + news["text"].fillna("")).tolist()
    sentiments = []
    confidences = []
    
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for j in range(len(batch)):
            pos, neg, neu = probs[j].tolist()
            sentiments.append(round(pos - neg, 4))
            confidences.append(round(max(probs[j]).item(), 4))
    
    news["sentiment"] = sentiments
    news["confidence"] = confidences
    news["published_at"] = pd.to_datetime(news["published_at"], utc=True, errors="coerce")
    news["date"] = news["published_at"].dt.date
    
    # 按日聚合
    daily = news.groupby("date").agg(
        daily_avg_sentiment=("sentiment", "mean"),
        sentiment_volatility=("sentiment", "std"),
        news_count=("sentiment", "count"),
        avg_confidence=("confidence", "mean"),
    ).reset_index()
    daily["ticker"] = ticker
    daily["date"] = pd.to_datetime(daily["date"])
    daily["sentiment_volatility"] = daily["sentiment_volatility"].fillna(0)
    
    return daily


def compute_fundamental_score(ticker: str) -> pd.DataFrame:
    """基于 FMP Ratios 计算基本面评分。"""
    ratios_file = PROJECT_ROOT / "data" / "raw" / "fmp" / "ratios" / f"{ticker}.parquet"
    if not ratios_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(ratios_file)
    except:
        return pd.DataFrame()
    
    if df.empty:
        return pd.DataFrame()
    
    # 提取关键指标
    result = pd.DataFrame()
    result["date"] = pd.to_datetime(df.get("date", df.get("fillingDate", "")))
    result["ticker"] = ticker
    
    # PE (越低越好，但不能为负)
    pe = pd.to_numeric(df.get("priceEarningsRatio", df.get("peRatio", None)), errors="coerce")
    if pe is not None:
        result["pe"] = pe.clip(-100, 200)
    
    # ROE (越高越好)
    roe = pd.to_numeric(df.get("returnOnEquity", df.get("roe", None)), errors="coerce")
    if roe is not None:
        result["roe"] = roe
    
    # 营收增长
    rev_growth = pd.to_numeric(df.get("revenueGrowth", df.get("growthRevenue", None)), errors="coerce")
    if rev_growth is not None:
        result["revenue_growth"] = rev_growth
    
    return result.dropna(subset=["date"])


def step3_features(tickers: list):
    """特征工程全量执行。"""
    print(f"\n{'='*60}")
    print(f"🔧 Step 3: 特征工程 + FinBERT ({len(tickers)} tickers)")
    print(f"{'='*60}")
    
    tech_dir = PROJECT_ROOT / "data" / "features" / "technical"
    sent_dir = PROJECT_ROOT / "data" / "features" / "sentiment"
    fund_dir = PROJECT_ROOT / "data" / "features" / "fundamental"
    
    tech_ok = sent_ok = fund_ok = 0
    
    # ── 技术指标 ──
    print(f"\n  📈 技术指标...")
    for i, ticker in enumerate(tickers):
        if i % 20 == 0 and i > 0:
            print(f"    {i}/{len(tickers)}...")
        
        out = tech_dir / f"{ticker}.parquet"
        if out.exists():
            tech_ok += 1
            continue
        
        kline = PROJECT_ROOT / "data" / "raw" / "massive" / "daily" / f"{ticker}.parquet"
        if not kline.exists():
            continue
        
        df = compute_technical(kline)
        if not df.empty:
            df.to_parquet(out, index=False)
            tech_ok += 1
    
    print(f"  ✅ 技术指标: {tech_ok}/{len(tickers)}")
    
    # ── FinBERT 情绪 ──
    print(f"\n  🧠 FinBERT 情绪打标...")
    for i, ticker in enumerate(tickers):
        if i % 20 == 0 and i > 0:
            print(f"    {i}/{len(tickers)}...")
        
        out = sent_dir / f"{ticker}.parquet"
        if out.exists():
            sent_ok += 1
            continue
        
        df = compute_sentiment(ticker)
        if not df.empty:
            df.to_parquet(out, index=False)
            sent_ok += 1
        else:
            sent_ok += 1  # 无新闻也算处理过
    
    print(f"  ✅ 情绪: {sent_ok}/{len(tickers)}")
    
    # ── 基本面评分 ──
    print(f"\n  📊 基本面评分...")
    for ticker in tickers:
        out = fund_dir / f"{ticker}.parquet"
        if out.exists():
            fund_ok += 1
            continue
        
        df = compute_fundamental_score(ticker)
        if not df.empty:
            df.to_parquet(out, index=False)
            fund_ok += 1
        else:
            fund_ok += 1
    
    print(f"  ✅ 基本面: {fund_ok}/{len(tickers)}")
    
    return {"tech": tech_ok, "sentiment": sent_ok, "fundamental": fund_ok}


# ═══════════════════════════════════════════════════
# Step 4: 回测 + 归因
# ═══════════════════════════════════════════════════

def generate_signals(tickers: list, weights: dict = None) -> pd.DataFrame:
    """加载三大特征，生成日频买卖信号。"""
    if weights is None:
        weights = {"tech": 0.3, "fund": 0.4, "sent": 0.3}
    
    tech_dir = PROJECT_ROOT / "data" / "features" / "technical"
    sent_dir = PROJECT_ROOT / "data" / "features" / "sentiment"
    fund_dir = PROJECT_ROOT / "data" / "features" / "fundamental"
    
    all_signals = []
    
    for ticker in tickers:
        tech_file = tech_dir / f"{ticker}.parquet"
        sent_file = sent_dir / f"{ticker}.parquet"
        fund_file = fund_dir / f"{ticker}.parquet"
        
        if not tech_file.exists():
            continue
        
        # 技术信号
        tech = pd.read_parquet(tech_file)
        tech["date"] = pd.to_datetime(tech["date"])
        tech = tech.set_index("date")
        
        # 技术得分: RSI + MACD + BB + 动量
        tech_score = pd.Series(0.0, index=tech.index)
        
        # RSI: 超卖(30以下)买, 超买(70以上)卖
        if "rsi14" in tech.columns:
            rsi = tech["rsi14"].fillna(50)
            tech_score += np.where(rsi < 30, 0.3, np.where(rsi > 70, -0.3, 0))
        
        # MACD: 金叉买, 死叉卖
        if "macd_hist" in tech.columns:
            macd = tech["macd_hist"].fillna(0)
            macd_prev = macd.shift(1).fillna(0)
            tech_score += np.where((macd > 0) & (macd_prev <= 0), 0.3,
                          np.where((macd < 0) & (macd_prev >= 0), -0.3, 0))
        
        # BB: 下轨买, 上轨卖
        if "bb_pos" in tech.columns:
            bb = tech["bb_pos"].fillna(0.5)
            tech_score += np.where(bb < 0.1, 0.2, np.where(bb > 0.9, -0.2, 0))
        
        # 动量: 20日正向
        if "ret_20d" in tech.columns:
            ret = tech["ret_20d"].fillna(0)
            tech_score += np.where(ret > 0.05, 0.2, np.where(ret < -0.05, -0.2, 0))
        
        # 情绪信号
        sent_score = pd.Series(0.0, index=tech.index)
        if sent_file.exists():
            sent = pd.read_parquet(sent_file)
            sent["date"] = pd.to_datetime(sent["date"])
            sent = sent.set_index("date")
            if "daily_avg_sentiment" in sent.columns:
                sent_aligned = sent["daily_avg_sentiment"].reindex(tech.index).fillna(0)
                sent_score = sent_aligned.clip(-1, 1)
        
        # 基本面信号
        fund_score = pd.Series(0.0, index=tech.index)
        if fund_file.exists():
            fund = pd.read_parquet(fund_file)
            fund["date"] = pd.to_datetime(fund["date"])
            fund = fund.set_index("date")
            # forward fill 季度数据到日频
            fund = fund.reindex(tech.index, method="ffill")
            
            if "roe" in fund.columns:
                roe = fund["roe"].fillna(0)
                fund_score += np.where(roe > 0.15, 0.3, np.where(roe < 0, -0.3, 0))
            if "pe" in fund.columns:
                pe = fund["pe"].fillna(20)
                fund_score += np.where((pe > 0) & (pe < 15), 0.2, np.where(pe > 50, -0.2, 0))
        
        # 综合信号
        composite = (
            weights["tech"] * tech_score +
            weights["fund"] * fund_score +
            weights["sent"] * sent_score
        )
        
        # 标准化到 -1 ~ +1
        if composite.std() > 0:
            composite = (composite - composite.mean()) / composite.std()
            composite = composite.clip(-1, 1)
        
        signal_df = pd.DataFrame({
            "date": tech.index,
            "ticker": ticker,
            "close": tech["close"].values,
            "signal": composite.values,
            "tech_score": tech_score.values,
            "fund_score": fund_score.values,
            "sent_score": sent_score.values,
            "rsi": tech["rsi14"].values if "rsi14" in tech.columns else np.nan,
            "macd_hist": tech["macd_hist"].values if "macd_hist" in tech.columns else np.nan,
            "volume": tech["volume"].values if "volume" in tech.columns else np.nan,
        })
        all_signals.append(signal_df)
    
    if not all_signals:
        return pd.DataFrame()
    
    return pd.concat(all_signals, ignore_index=True)


def walk_forward_backtest(signals: pd.DataFrame, train_start="2023-01-01", train_end="2023-12-31",
                          test_start="2024-01-01", test_end="2024-12-31",
                          top_n=5, hold_days=10, stop_loss=-0.10) -> dict:
    """Walk-Forward 回测：训练期学习阈值，验证期执行。"""
    
    # 训练期：学习最优信号阈值
    train = signals[(signals["date"] >= train_start) & (signals["date"] <= train_end)]
    test = signals[(signals["date"] >= test_start) & (signals["date"] <= test_end)]
    
    if train.empty or test.empty:
        return {"error": "Insufficient data"}
    
    # 训练期：找最优买入阈值（信号强度 top quantile）
    train_returns = []
    for ticker in train["ticker"].unique():
        tdf = train[train["ticker"] == ticker].sort_values("date")
        if len(tdf) < 20:
            continue
        tdf["fwd_ret"] = tdf["close"].pct_change(hold_days).shift(-hold_days)
        train_returns.append(tdf)
    
    if not train_returns:
        return {"error": "No training data"}
    
    train_all = pd.concat(train_returns, ignore_index=True)
    train_all = train_all.dropna(subset=["fwd_ret", "signal"])
    
    # 找最优阈值：信号 > threshold 的平均收益
    best_threshold = 0.0
    best_sharpe = -999
    for threshold in np.arange(-0.5, 0.6, 0.1):
        mask = train_all["signal"] > threshold
        if mask.sum() < 10:
            continue
        rets = train_all.loc[mask, "fwd_ret"]
        sharpe = rets.mean() / rets.std() * np.sqrt(252 / hold_days) if rets.std() > 0 else 0
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_threshold = threshold
    
    # ── 验证期执行 ──
    trades = []
    dates = sorted(test["date"].unique())
    
    portfolio_value = 100000.0
    cash = 100000.0
    positions = {}  # ticker -> {qty, entry_price, entry_date}
    daily_values = []
    
    for date in dates:
        day_data = test[test["date"] == date]
        
        # 更新持仓市值
        total_value = cash
        for ticker, pos in list(positions.items()):
            row = day_data[day_data["ticker"] == ticker]
            if not row.empty:
                current_price = row.iloc[0]["close"]
                total_value += pos["qty"] * current_price
        
        daily_values.append({"date": date, "value": total_value})
        
        # 卖出逻辑：持有超 hold_days 或止损
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            days_held = (date - pos["entry_date"]).days
            
            row = day_data[day_data["ticker"] == ticker]
            if row.empty:
                continue
            
            current_price = row.iloc[0]["close"]
            pnl = (current_price - pos["entry_price"]) / pos["entry_price"]
            
            should_sell = False
            reason = ""
            
            if days_held >= hold_days:
                should_sell = True
                reason = "到期"
            elif pnl <= stop_loss:
                should_sell = True
                reason = "止损"
            
            if should_sell:
                cash += pos["qty"] * current_price
                trades.append({
                    "ticker": ticker,
                    "side": "SELL",
                    "date": str(date.date()) if hasattr(date, 'date') else str(date),
                    "price": current_price,
                    "qty": pos["qty"],
                    "pnl_pct": round(pnl * 100, 2),
                    "reason": reason,
                    "days_held": days_held,
                })
                del positions[ticker]
        
        # 买入逻辑：信号 > threshold，买 top_n
        buy_candidates = day_data[
            (day_data["signal"] > best_threshold) &
            (~day_data["ticker"].isin(positions.keys()))
        ].nlargest(top_n, "signal")
        
        if len(positions) < top_n and not buy_candidates.empty:
            per_stock = cash / max(top_n - len(positions), 1) * 0.95
            
            for _, row in buy_candidates.iterrows():
                if len(positions) >= top_n:
                    break
                if cash < per_stock:
                    break
                
                ticker = row["ticker"]
                price = row["close"]
                if price <= 0:
                    continue
                
                qty = int(per_stock / price)
                if qty <= 0:
                    continue
                
                cash -= qty * price
                positions[ticker] = {
                    "qty": qty,
                    "entry_price": price,
                    "entry_date": date,
                }
                trades.append({
                    "ticker": ticker,
                    "side": "BUY",
                    "date": str(date.date()) if hasattr(date, 'date') else str(date),
                    "price": price,
                    "qty": qty,
                })
    
    # ── 计算绩效指标 ──
    values = pd.DataFrame(daily_values)
    if values.empty:
        return {"error": "No trades executed"}
    
    values["returns"] = values["value"].pct_change()
    
    # Sharpe
    mean_ret = values["returns"].mean()
    std_ret = values["returns"].std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
    
    # Max Drawdown
    peak = values["value"].cummax()
    drawdown = (values["value"] - peak) / peak
    max_dd = drawdown.min()
    
    # Win Rate
    sell_trades = [t for t in trades if t["side"] == "SELL"]
    wins = [t for t in sell_trades if t.get("pnl_pct", 0) > 0]
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0
    
    # Profit Factor
    gross_profit = sum(t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) > 0)
    gross_loss = abs(sum(t["pnl_pct"] for t in sell_trades if t.get("pnl_pct", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    
    # Total Return
    total_return = (values["value"].iloc[-1] / values["value"].iloc[0] - 1) * 100
    
    result = {
        "train_period": f"{train_start} → {train_end}",
        "test_period": f"{test_start} → {test_end}",
        "best_threshold": round(best_threshold, 2),
        "train_sharpe": round(best_sharpe, 3),
        "total_trades": len(trades),
        "sell_trades": len(sell_trades),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "total_return_pct": round(total_return, 2),
        "final_value": round(values["value"].iloc[-1], 2),
        "trades": trades[-20:],  # 最近20笔
        "daily_values_sample": values.iloc[::20].to_dict("records"),  # 每20天采样
    }
    
    return result


def loss_attribution(trades: list, signals: pd.DataFrame) -> dict:
    """亏损归因分析。"""
    sell_trades = [t for t in trades if t["side"] == "SELL"]
    losing = [t for t in sell_trades if t.get("pnl_pct", 0) < 0]
    winning = [t for t in sell_trades if t.get("pnl_pct", 0) > 0]
    
    if not losing:
        return {"losing_trades": 0, "note": "No losing trades"}
    
    # 亏损交易的特征分析
    loss_rsi = []
    loss_signal = []
    loss_tickers = set()
    
    for t in losing:
        ticker = t["ticker"]
        date = t["date"]
        loss_tickers.add(ticker)
        
        sig_row = signals[
            (signals["ticker"] == ticker) & 
            (signals["date"].astype(str).str[:10] == date[:10])
        ]
        if not sig_row.empty:
            row = sig_row.iloc[0]
            if not np.isnan(row.get("rsi", np.nan)):
                loss_rsi.append(row["rsi"])
            loss_signal.append(row["signal"])
    
    # 高RSI追涨分析
    high_rsi_losses = sum(1 for r in loss_rsi if r > 70) if loss_rsi else 0
    rsi_pct = high_rsi_losses / len(loss_rsi) * 100 if loss_rsi else 0
    
    # 信号强度分析
    strong_signal_losses = sum(1 for s in loss_signal if s > 0.3) if loss_signal else 0
    
    # 分析
    attribution = {
        "total_losing": len(losing),
        "total_winning": len(winning),
        "loss_rate": round(len(losing) / len(sell_trades) * 100, 1),
        "avg_loss_pct": round(np.mean([t["pnl_pct"] for t in losing]), 2),
        "avg_win_pct": round(np.mean([t["pnl_pct"] for t in winning]), 2) if winning else 0,
        "high_rsi_loss_pct": round(rsi_pct, 1),
        "strong_signal_losses": strong_signal_losses,
        "losing_tickers": list(loss_tickers)[:20],
        "loss_by_reason": {},
    }
    
    # 按原因分类
    for t in losing:
        reason = t.get("reason", "unknown")
        if reason not in attribution["loss_by_reason"]:
            attribution["loss_by_reason"][reason] = {"count": 0, "total_pnl": 0}
        attribution["loss_by_reason"][reason]["count"] += 1
        attribution["loss_by_reason"][reason]["total_pnl"] += t.get("pnl_pct", 0)
    
    return attribution


def step4_backtest(tickers: list, weights: dict = None):
    """回测 + 归因。"""
    print(f"\n{'='*60}")
    print(f"📊 Step 4: Walk-Forward 回测")
    print(f"{'='*60}")
    
    if weights is None:
        weights = {"tech": 0.3, "fund": 0.4, "sent": 0.3}
    
    print(f"  权重: Tech={weights['tech']}, Fund={weights['fund']}, Sent={weights['sent']}")
    
    # 生成信号
    print(f"  生成信号...")
    signals = generate_signals(tickers, weights)
    if signals.empty:
        print("  ❌ 无信号数据")
        return None
    
    print(f"  信号: {len(signals)} 行, {signals['ticker'].nunique()} tickers")
    
    # 回测
    print(f"  回测中...")
    result = walk_forward_backtest(signals, top_n=5, hold_days=10, stop_loss=-0.10)
    
    if "error" in result:
        print(f"  ❌ {result['error']}")
        return None
    
    # 归因
    print(f"  归因分析...")
    attribution = loss_attribution(result.get("trades", []), signals)
    
    # 合并报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "weights": weights,
        "performance": {k: v for k, v in result.items() if k not in ["trades", "daily_values_sample"]},
        "attribution": attribution,
        "recent_trades": result.get("trades", []),
    }
    
    # 保存
    report_file = PROJECT_ROOT / "data" / "backtest" / "attribution_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # 打印摘要
    p = report["performance"]
    a = report["attribution"]
    
    print(f"\n  {'='*50}")
    print(f"  📊 回测结果")
    print(f"  {'='*50}")
    print(f"  Sharpe:        {p['sharpe_ratio']}")
    print(f"  Max Drawdown:  {p['max_drawdown_pct']}%")
    print(f"  Win Rate:      {p['win_rate']}%")
    print(f"  Profit Factor: {p['profit_factor']}")
    print(f"  Total Return:  {p['total_return_pct']}%")
    print(f"  总交易:        {p['total_trades']} 笔")
    print(f"  最优阈值:      {p['best_threshold']}")
    print(f"\n  归因:")
    print(f"  亏损交易:      {a['total_losing']}/{a['total_losing']+a['total_winning']}")
    print(f"  平均亏损:      {a['avg_loss_pct']}%")
    print(f"  高RSI亏损占比: {a.get('high_rsi_loss_pct', 0)}%")
    print(f"  报告:          {report_file}")
    
    return report


# ═══════════════════════════════════════════════════
# Step 5: 权重优化（输出报告供 Hermes 决策）
# ═══════════════════════════════════════════════════

def step5_generate_optimization_prompt(report: dict) -> str:
    """生成 Hermes 决策 Prompt。"""
    p = report["performance"]
    a = report["attribution"]
    w = report["weights"]
    
    prompt = f"""你是 Project Mercurius 首席风控官。基于 2023-2024 年日频回测归因报告决策：

**当前权重**: Tech={w['tech']}, Fund={w['fund']}, Sent={w['sent']}

**绩效指标**:
- Sharpe: {p['sharpe_ratio']}
- Max Drawdown: {p['max_drawdown_pct']}%
- Win Rate: {p['win_rate']}%
- Profit Factor: {p['profit_factor']}
- Total Return: {p['total_return_pct']}%

**亏损归因**:
- 亏损交易: {a['total_losing']}/{a['total_losing']+a['total_winning']}
- 平均亏损: {a['avg_loss_pct']}%
- 高RSI(>70)追涨导致亏损占比: {a.get('high_rsi_loss_pct', 0)}%
- 亏损按原因: {json.dumps(a.get('loss_by_reason', {}), ensure_ascii=False)}

**约束**: 单次权重调整幅度 ≤ ±0.05, 任何因子权重 ∈ [0.1, 0.6]

**规则**: 若 Sharpe > 0.8 且 MaxDD < 15%, 输出「维持现状」。否则提出调整方案。

请输出调整后的权重 + 1句理由。格式：
```json
{{"tech": X, "fund": Y, "sent": Z, "reason": "..."}}
```"""
    
    return prompt


def save_weights(weights: dict):
    """保存权重到 config/active_weights.yaml。"""
    config_dir = PROJECT_ROOT / "config"
    config_dir.mkdir(exist_ok=True)
    
    w_file = config_dir / "active_weights.yaml"
    with open(w_file, "w") as f:
        f.write(f"# Project Mercurius Active Weights\n")
        f.write(f"# Updated: {datetime.now().isoformat()}\n")
        f.write(f"tech: {weights['tech']}\n")
        f.write(f"fund: {weights['fund']}\n")
        f.write(f"sent: {weights['sent']}\n")
    
    print(f"  💾 权重已保存: {w_file}")


def archive_run(report: dict):
    """归档本次运行。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = PROJECT_ROOT / "archive" / f"{ts}_baseline"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制报告
    report_file = archive_dir / "attribution_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # 复制权重
    weights_file = PROJECT_ROOT / "config" / "active_weights.yaml"
    if weights_file.exists():
        import shutil
        shutil.copy2(weights_file, archive_dir / "active_weights.yaml")
    
    print(f"  📦 归档: {archive_dir}")


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Project Mercurius Night 1+2")
    sub = parser.add_subparsers(dest="command")
    
    # universe
    p_uni = sub.add_parser("universe", help="生成 universe")
    p_uni.add_argument("--mode", default="scored20", choices=["scored20", "v12_1032"])
    
    # fetch
    p_fetch = sub.add_parser("fetch", help="拉取数据")
    p_fetch.add_argument("--start", default="2023-01-01")
    p_fetch.add_argument("--end", default="2024-12-31")
    p_fetch.add_argument("--mode", default="scored20")
    
    # features
    p_feat = sub.add_parser("features", help="特征工程")
    p_feat.add_argument("--mode", default="scored20")
    
    # backtest
    p_bt = sub.add_parser("backtest", help="回测+归因")
    p_bt.add_argument("--mode", default="scored20")
    
    # full (all steps)
    p_full = sub.add_parser("full", help="全流程")
    p_full.add_argument("--start", default="2023-01-01")
    p_full.add_argument("--end", default="2024-12-31")
    p_full.add_argument("--mode", default="scored20")
    
    args = parser.parse_args()
    
    if args.command == "universe":
        generate_universe(args.mode)
    
    elif args.command == "fetch":
        tickers = load_universe(args.mode)
        step2_fetch_all(tickers, args.start, args.end)
    
    elif args.command == "features":
        tickers = load_universe(args.mode)
        step3_features(tickers)
    
    elif args.command == "backtest":
        tickers = load_universe(args.mode)
        report = step4_backtest(tickers)
        if report:
            prompt = step5_generate_optimization_prompt(report)
            print(f"\n  {'='*50}")
            print(f"  🤖 Hermes 决策 Prompt")
            print(f"  {'='*50}")
            print(prompt)
    
    elif args.command == "full":
        print("🚀 Project Mercurius — Night 1+2 全流程")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1
        tickers = generate_universe(args.mode)
        
        # Step 2
        step2_fetch_all(tickers, args.start, args.end)
        
        # Step 3
        step3_features(tickers)
        
        # Step 4
        report = step4_backtest(tickers)
        
        elapsed = time.time() - start_time
        print(f"\n  ⏱️ 总耗时: {elapsed/60:.1f} 分钟")
        
        if report:
            prompt = step5_generate_optimization_prompt(report)
            print(f"\n{'='*60}")
            print(f"🤖 Step 5: Hermes 决策")
            print(f"{'='*60}")
            print(prompt)
            
            # 保存 prompt 供 Hermes 读取
            prompt_file = PROJECT_ROOT / "data" / "backtest" / "optimization_prompt.txt"
            with open(prompt_file, "w") as f:
                f.write(prompt)
            
            # 归档
            archive_run(report)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
