#!/usr/bin/env python3
"""
FinBERT 离线情绪打标流水线
==========================
从 Massive(Polygon) + FMP 拉取新闻，用 FinBERT 打情绪分，存 Parquet。

路径: data/finbert_sentiment/year={yyyy}/month={mm}/ticker={sym}.parquet
Schema: ticker, published_at, title, text, source, publisher, sentiment, confidence

用法:
    # 测试：对 AAPL 2024-01 打标
    python3 scripts/falcons/finbert_pipeline.py score --tickers AAPL --start 2024-01-01 --end 2024-02-01

    # 全量：当前交易宇宙最近6个月
    python3 scripts/falcons/finbert_pipeline.py backfill --months 6

    # 增量：昨天的新闻（给 cron 用）
    python3 scripts/falcons/finbert_pipeline.py daily

    # 状态：检查覆盖率
    python3 scripts/falcons/finbert_pipeline.py status
"""

import sys
import os
import json
import time
import argparse
import glob
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "finbert_sentiment"
CONFIG_PATH = PROJECT_ROOT / "config" / "central_config.json"
ENV_PATH = PROJECT_ROOT / ".env"

# 加载 .env
from dotenv import load_dotenv
load_dotenv(ENV_PATH)

# 允许的新闻源（按 Mercurius 规范）
ALLOWED_PUBLISHERS = {
    "benzinga", "prnewswire", "businesswire", "seekingalpha",
    "reuters", "bloomberg", "associated press", "cnbc",
    "marketwatch", "yahoo finance", "investors business daily",
    "barrons", "the wall street journal", "dow jones",
    "zacks investment research", "seeking alpha",
    "the motley fool", "investopedia", "streetinsider",
}


def get_api_keys():
    """读取 API keys。"""
    return {
        "massive": os.environ.get("MASSIVE_API_KEY", ""),
        "fmp": os.environ.get("FMP_API_KEY", ""),
    }


def load_universe():
    """加载 Falcon 交易宇宙 (SPX + R2K)。"""
    tickers = set()
    
    # 优先: Falcon 特征矩阵 (SPX 476只)
    features_file = PROJECT_ROOT / "data" / "falcon" / "features_v02.parquet"
    if features_file.exists():
        import pandas as pd
        df = pd.read_parquet(features_file, columns=["ticker"])
        tickers.update(df["ticker"].unique())
    
    # 其次: Russell 2000 (691只)
    russell_file = PROJECT_ROOT / "data" / "falcon" / "russell_prices.json"
    if russell_file.exists():
        import json as _json
        with open(russell_file) as f:
            prices = _json.load(f)
        tickers.update(prices.keys())
    
    # 回退: scored 文件
    if not tickers:
        for pattern in [str(PROJECT_ROOT / "data" / "us" / "*scored*.json")]:
            for f in sorted(glob.glob(pattern)):
                with open(f) as fh:
                    d = json.load(fh)
                if isinstance(d, dict) and "picks" in d:
                    tickers.update(p["sym"] for p in d["picks"])
    
    if not tickers:
        print("⚠️ 无法加载交易宇宙，使用默认列表")
        tickers = {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"}
    
    return sorted(tickers)


# ── 新闻抓取 ──

def fetch_massive_news(ticker: str, start: str, end: str, key: str) -> list:
    """从 Massive (Polygon) 拉取新闻。返回 [{title, text, published_at, publisher, source}]"""
    import urllib.request

    articles = []
    url = (
        f"https://api.polygon.io/v2/reference/news?"
        f"ticker={ticker}&published_utc.gte={start}&published_utc.lt={end}"
        f"&limit=100&order=desc&sort=published_utc&apiKey={key}"
    )

    page = 0
    while url and page < 10:  # 最多10页（1000篇/ticker/月）
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read())
        except Exception as e:
            if "429" in str(e):
                time.sleep(12)  # 限流，等12秒
                continue
            break

        for a in data.get("results", []):
            pub_name = a.get("publisher", {}).get("name", "").lower()
            title = a.get("title", "")
            desc = a.get("description", "")

            if not title:
                continue

            articles.append({
                "ticker": ticker,
                "title": title,
                "text": desc or title,
                "published_at": a.get("published_utc", ""),
                "publisher": pub_name,
                "source": "massive",
            })

        # 分页
        next_url = data.get("next_url")
        if next_url:
            url = f"{next_url}&apiKey={key}"
            page += 1
            time.sleep(0.08)  # 800/min limit
        else:
            break

    return articles


def fetch_fmp_news(ticker: str, start: str, end: str, key: str) -> list:
    """从 FMP 拉取新闻。返回 [{title, text, published_at, publisher, source}]"""
    import urllib.request

    articles = []
    url = (
        f"https://financialmodelingprep.com/stable/news/stock?"
        f"symbols={ticker}&from={start}&to={end}&limit=100&apikey={key}"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        if "429" in str(e):
            time.sleep(10)
        return articles

    if not isinstance(data, list):
        return articles

    for a in data:
        pub_name = a.get("publisher", "").lower()
        title = a.get("title", "")
        text = a.get("text", "")

        if not title:
            continue

        articles.append({
            "ticker": ticker,
            "title": title,
            "text": text or title,
            "published_at": a.get("publishedDate", ""),
            "publisher": pub_name,
            "source": "fmp",
        })

    return articles


def fetch_news(tickers: list, start: str, end: str) -> pd.DataFrame:
    """从两个源拉取新闻，去重。"""
    keys = get_api_keys()
    all_articles = []
    total = len(tickers)

    for i, ticker in enumerate(tickers):
        if i % 50 == 0:
            print(f"  📥 抓取新闻: {i}/{total} tickers...")

        # Massive
        if keys["massive"]:
            arts = fetch_massive_news(ticker, start, end, keys["massive"])
            all_articles.extend(arts)

        # FMP
        if keys["fmp"]:
            arts = fetch_fmp_news(ticker, start, end, keys["fmp"])
            all_articles.extend(arts)

        time.sleep(0.1)  # 限流保护

    if not all_articles:
        return pd.DataFrame()

    df = pd.DataFrame(all_articles)

    # 去重：同一 ticker + 相似标题
    df["title_norm"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates(subset=["ticker", "title_norm"])
    df = df.drop(columns=["title_norm"])

    # 过滤发布源
    df = df[df["publisher"].isin(ALLOWED_PUBLISHERS)]

    print(f"  📰 共 {len(df)} 篇（去重+过滤后）")
    return df


# ── FinBERT 打标 ──

_finbert_model = None
_finbert_tokenizer = None


def load_finbert():
    """懒加载 FinBERT 模型（只加载一次），优先 GPU。"""
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        print("  🧠 加载 FinBERT 模型...")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        if torch.cuda.is_available():
            _finbert_model = _finbert_model.to("cuda")
            print("  🧠 GPU 加速: ON")
    return _finbert_tokenizer, _finbert_model


def score_batch(texts: list) -> list:
    """批量打分。返回 [(sentiment, confidence), ...]"""
    import torch

    tokenizer, model = load_finbert()
    device = next(model.parameters()).device
    results = []

    for i in range(0, len(texts), 64):
        batch = texts[i:i + 64]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for j in range(len(batch)):
            pos, neg, neu = probs[j].tolist()
            sentiment = pos - neg  # -1 ~ +1
            confidence = max(probs[j]).item()
            results.append((round(sentiment, 4), round(confidence, 4)))

    return results


def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """给新闻 DataFrame 打情绪分。"""
    if df.empty:
        return df

    print(f"  🧠 FinBERT 打标 {len(df)} 篇...")
    start = time.time()

    # 合并 title + text 作为输入
    texts = (df["title"].fillna("") + ". " + df["text"].fillna("")).tolist()
    scores = score_batch(texts)

    df["sentiment"] = [s[0] for s in scores]
    df["confidence"] = [s[1] for s in scores]

    elapsed = time.time() - start
    print(f"  ✅ 完成: {len(df)} 篇, {elapsed:.1f}s ({len(df)/elapsed:.0f} 篇/秒)")

    return df


# ── Parquet 存储 ──

def save_parquet(df: pd.DataFrame):
    """按 year/month/ticker 分区保存 Parquet。"""
    if df.empty:
        return

    # 解析时间
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df.copy()
    df["year"] = df["published_at"].dt.year.astype(int)
    df["month"] = df["published_at"].dt.month.astype(int)

    saved = 0
    for (year, month, ticker), group in df.groupby(["year", "month", "ticker"]):
        out_dir = DATA_DIR / f"year={year}" / f"month={month:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"ticker={ticker}.parquet"

        # 如果已有文件，合并（增量）
        if out_file.exists():
            existing = pd.read_parquet(out_file)
            group = pd.concat([existing, group], ignore_index=True)
            group = group.drop_duplicates(subset=["title", "published_at"])

        # 只保存需要的列
        cols = ["ticker", "published_at", "title", "text", "source", "publisher", "sentiment", "confidence"]
        group[cols].to_parquet(out_file, index=False)
        saved += 1

    print(f"  💾 保存: {saved} 个 Parquet 文件")


# ── 命令: score ──

def cmd_score(tickers: list, start: str, end: str):
    """对指定 ticker 日期范围打标。"""
    print(f"📊 打标: {len(tickers)} tickers, {start} → {end}")
    df = fetch_news(tickers, start, end)
    df = score_articles(df)
    save_parquet(df)
    return df


# ── 命令: backfill ──

def cmd_backfill(months: int = 6):
    """回填当前交易宇宙最近 N 个月的新闻情绪。"""
    tickers = load_universe()
    print(f"📊 回填: {len(tickers)} tickers, 最近 {months} 个月")

    end = datetime.now()
    for m in range(months):
        month_end = end - timedelta(days=30 * m)
        month_start = month_end - timedelta(days=30)
        start_str = month_start.strftime("%Y-%m-%d")
        end_str = month_end.strftime("%Y-%m-%d")
        print(f"\n── {start_str} → {end_str} ──")
        cmd_score(tickers, start_str, end_str)


# ── 命令: daily ──

def cmd_daily():
    """每日增量：拉取昨天的新闻并打标。"""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")
    tickers = load_universe()
    print(f"📊 每日增量: {len(tickers)} tickers, {yesterday}")
    cmd_score(tickers, yesterday, today)


# ── 命令: status ──

def cmd_status():
    """检查覆盖率。"""
    if not DATA_DIR.exists():
        print("❌ /data/finbert_sentiment 不存在")
        return

    files = list(DATA_DIR.rglob("*.parquet"))
    if not files:
        print("❌ 无 Parquet 文件")
        return

    total_rows = 0
    tickers = set()
    date_range = [None, None]

    for f in files:
        df = pd.read_parquet(f)
        total_rows += len(df)
        tickers.update(df["ticker"].unique())
        if "published_at" in df.columns:
            dates = pd.to_datetime(df["published_at"])
            mn, mx = dates.min(), dates.max()
            if date_range[0] is None or mn < date_range[0]:
                date_range[0] = mn
            if date_range[1] is None or mx > date_range[1]:
                date_range[1] = mx

    print(f"📊 FinBERT 情绪库状态")
    print(f"  文件数:   {len(files)}")
    print(f"  总记录:   {total_rows:,}")
    print(f"  Tickers:  {len(tickers)}")
    if date_range[0]:
        print(f"  时间范围: {date_range[0].strftime('%Y-%m-%d')} → {date_range[1].strftime('%Y-%m-%d')}")

    # 月度分布
    print(f"\n  月度分布:")
    for year_dir in sorted(DATA_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            month_files = list(month_dir.glob("*.parquet"))
            month_rows = sum(len(pd.read_parquet(f)) for f in month_files[:5])  # 采样
            est_total = month_rows / max(len(month_files[:5]), 1) * len(month_files)
            print(f"    {year_dir.name}-{month_dir.name}: {len(month_files)} tickers, ~{est_total:,.0f} records")


# ── 主入口 ──

def main():
    parser = argparse.ArgumentParser(description="FinBERT 离线情绪打标")
    sub = parser.add_subparsers(dest="command")

    # score
    p_score = sub.add_parser("score", help="对指定 ticker 打标")
    p_score.add_argument("--tickers", nargs="+", required=True)
    p_score.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_score.add_argument("--end", required=True, help="YYYY-MM-DD")

    # backfill
    p_back = sub.add_parser("backfill", help="回填交易宇宙")
    p_back.add_argument("--months", type=int, default=6, help="回填月数")

    # daily
    sub.add_parser("daily", help="每日增量")

    # status
    sub.add_parser("status", help="检查覆盖率")

    args = parser.parse_args()

    if args.command == "score":
        cmd_score(args.tickers, args.start, args.end)
    elif args.command == "backfill":
        cmd_backfill(args.months)
    elif args.command == "daily":
        cmd_daily()
    elif args.command == "status":
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
