#!/usr/bin/env python3
"""Score remaining tickers with FinBERT."""
import sys, os, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd
import numpy as np

sent_dir = PROJECT_ROOT / "data" / "features" / "sentiment"
done = {f.stem for f in sent_dir.glob("*.parquet")}

# Pending tickers with news
pending = ["VRDN", "VREX", "VMAR", "XOMO"]
pending = [t for t in pending if t not in done]

if not pending:
    print("All sentiment files done.")
else:
    # Load FinBERT
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    
    print("Loading FinBERT...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    
    for ticker in pending:
        print(f"Scoring {ticker}...")
        
        # Load news
        news_files = []
        for d in [PROJECT_ROOT / "data" / "raw" / "fmp" / "news",
                  PROJECT_ROOT / "data" / "raw" / "massive" / "news"]:
            f = d / f"{ticker}.parquet"
            if f.exists():
                news_files.append(f)
        
        if not news_files:
            print(f"  No news, skipping")
            continue
        
        dfs = [pd.read_parquet(f) for f in news_files]
        news = pd.concat(dfs, ignore_index=True)
        news["title_norm"] = news["title"].str.lower().str.strip()
        news = news.drop_duplicates(subset=["title_norm"])
        news = news[news["title"].str.len() > 10]
        
        if news.empty:
            print(f"  Empty news, skipping")
            continue
        
        # Score
        texts = (news["title"].fillna("") + ". " + news["text"].fillna("")).tolist()
        sentiments = []
        confidences = []
        
        for i in range(0, len(texts), 64):
            batch = texts[i:i+64]
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
        
        daily = news.groupby("date").agg(
            daily_avg_sentiment=("sentiment", "mean"),
            sentiment_volatility=("sentiment", "std"),
            news_count=("sentiment", "count"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()
        daily["ticker"] = ticker
        daily["date"] = pd.to_datetime(daily["date"])
        daily["sentiment_volatility"] = daily["sentiment_volatility"].fillna(0)
        
        daily.to_parquet(sent_dir / f"{ticker}.parquet", index=False)
        print(f"  ✅ {len(daily)} days")

# Create empty for no-news tickers
universe = pd.read_csv(PROJECT_ROOT / "config" / "universe_scored20.csv")["ticker"].tolist()
done = {f.stem for f in sent_dir.glob("*.parquet")}
for t in universe:
    if t not in done:
        empty = pd.DataFrame(columns=["date","ticker","daily_avg_sentiment","sentiment_volatility","news_count","avg_confidence"])
        empty.to_parquet(sent_dir / f"{t}.parquet", index=False)
        print(f"  {t}: empty (no news)")

print(f"\nTotal: {len(list(sent_dir.glob('*.parquet')))} sentiment files")
