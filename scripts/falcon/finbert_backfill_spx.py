#!/usr/bin/env python3
"""
FinBERT 定向回填 — SPX 476只, 最近 N 个月
支持断点续传: 记录已处理的 ticker-month, 跳过已完成的。
"""
import sys
import os
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "finbert_sentiment"
FALCON_DIR = PROJECT_ROOT / "data" / "falcon"
PROGRESS_FILE = DATA_DIR / "backfill_progress.json"
ENV_PATH = PROJECT_ROOT / ".env"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finbert_pipeline import (
    fetch_fmp_news, fetch_massive_news, get_api_keys,
    score_articles, save_parquet, ALLOWED_PUBLISHERS,
)

from dotenv import load_dotenv
load_dotenv(ENV_PATH)


def load_spx_tickers():
    """从 Falcon 特征矩阵读 SPX ticker 列表。"""
    df = pd.read_parquet(FALCON_DIR / "features_v02.parquet", columns=["ticker"])
    return sorted(df["ticker"].unique())


def load_progress():
    """加载已处理的 ticker-month 组合。"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f))
    return set()


def save_progress(done_set):
    """保存进度。"""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(sorted(done_set), f)


def backfill_spx(start_date="2024-01-01", end_date="2024-12-31", delay=0.05):
    """定向回填 SPX ticker 指定日期范围的新闻情绪。"""
    tickers = load_spx_tickers()
    keys = get_api_keys()
    done = load_progress()

    # 生成月份列表 (从旧到新，按月分段)
    from dateutil.relativedelta import relativedelta
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    month_ranges = []
    cursor = start
    while cursor < end:
        month_end = min(cursor + relativedelta(months=1) - timedelta(days=1), end)
        month_ranges.append((cursor.strftime("%Y-%m-%d"), month_end.strftime("%Y-%m-%d")))
        cursor = month_end + timedelta(days=1)

    total_tasks = len(tickers) * len(month_ranges)
    completed = len(done)
    skipped = 0
    t0 = time.time()

    print(f"📊 SPX FinBERT Backfill")
    print(f"  Tickers: {len(tickers)}")
    print(f"  Months: {len(month_ranges)} ({month_ranges[0][0]} → {month_ranges[-1][1]})")
    print(f"  Total: {total_tasks} ticker-months")
    print(f"  Already done: {completed}")
    print()

    for mi, (start, end) in enumerate(month_ranges):
        month_key_base = start[:7]  # YYYY-MM
        print(f"\n── {start} → {end} ──")
        month_articles = 0

        for ti, ticker in enumerate(tickers):
            key = f"{ticker}:{month_key_base}"

            if key in done:
                skipped += 1
                continue

            # 拉取新闻
            articles = []
            if keys["massive"]:
                articles.extend(fetch_massive_news(ticker, start, end, keys["massive"]))
            if keys["fmp"]:
                articles.extend(fetch_fmp_news(ticker, start, end, keys["fmp"]))

            if not articles:
                done.add(key)
                continue

            # 去重 + 过滤
            df = pd.DataFrame(articles)
            df["title_norm"] = df["title"].str.lower().str.strip()
            df = df.drop_duplicates(subset=["ticker", "title_norm"])
            df = df.drop(columns=["title_norm"])
            df = df[df["publisher"].isin(ALLOWED_PUBLISHERS)]

            if df.empty:
                done.add(key)
                continue

            # FinBERT 打标
            df = score_articles(df)

            # 保存
            save_parquet(df)
            month_articles += len(df)
            done.add(key)
            save_progress(done)

            # 进度
            completed += 1
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total_tasks - completed) / rate if rate > 0 else 0
            print(f"  [{completed}/{total_tasks}] {ticker}: {len(df)}篇 "
                  f"({elapsed/60:.0f}m, ETA {remaining/60:.0f}m)")

            time.sleep(delay)

        print(f"  本月合计: {month_articles} 篇")

    elapsed = time.time() - t0
    print(f"\n✅ 完成: {completed} ticker-months, {elapsed/60:.1f}分钟")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--delay", type=float, default=0.05, help="API间隔秒数")
    parser.add_argument("--reset", action="store_true", help="清除进度文件重新开始")
    args = parser.parse_args()

    if args.reset and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("🗑️ 进度文件已清除")

    backfill_spx(start_date=args.start, end_date=args.end, delay=args.delay)
