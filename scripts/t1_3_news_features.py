#!/usr/bin/env python3
"""
T1.3 FinBERT新闻特征工程
1. 读取所有已处理的分区parquet (2022-2026)
2. 合并 all_scored.parquet (2024-2026)
3. 去重 (ticker+published_at+title)
4. 按(ticker, month)聚合月度特征
5. 保存为 data/falcon/news_features_v04.parquet
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Configuration
BASE_DIR = Path("/home/hermes/.hermes/openclaw-archive")
FINBERT_DIR = BASE_DIR / "data" / "finbert_sentiment"
OUTPUT_DIR = BASE_DIR / "data" / "falcon"
OUTPUT_FILE = OUTPUT_DIR / "news_features_v04.parquet"

def load_partitioned_data():
    """Load all partitioned parquet files from year=XXXX/month=XX/ticker=XXX.parquet"""
    all_dfs = []
    
    for year_dir in sorted(FINBERT_DIR.glob("year=*")):
        year = year_dir.name.split("=")[1]
        print(f"Processing {year_dir.name}...")
        
        for month_dir in sorted(year_dir.glob("month=*")):
            month = month_dir.name.split("=")[1]
            
            for ticker_file in sorted(month_dir.glob("ticker=*.parquet")):
                try:
                    df = pd.read_parquet(ticker_file)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"  Error reading {ticker_file}: {e}")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def load_all_scored():
    """Load all_scored.parquet (2024-2026)"""
    scored_file = FINBERT_DIR / "all_scored.parquet"
    if scored_file.exists():
        return pd.read_parquet(scored_file)
    return pd.DataFrame()

def deduplicate(df):
    """Remove duplicates by (ticker, published_at, title)"""
    # Normalize published_at to remove timezone for consistent comparison
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    
    # Sort by published_at descending to keep the most recent
    df = df.sort_values('published_at', ascending=False)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['ticker', 'title'], keep='first')
    
    return df

def compute_monthly_features(df):
    """Compute monthly aggregated features by (ticker, month)"""
    
    # Extract month from published_at
    df['month'] = df['published_at'].dt.to_period('M')
    
    # Group by (ticker, month)
    grouped = df.groupby(['ticker', 'month'])
    
    features = grouped.agg(
        news_avg_sentiment=('sentiment', 'mean'),
        news_sentiment_vol=('sentiment', 'std'),
        news_neg_ratio=('sentiment', lambda x: (x < -0.1).mean()),
        news_pos_ratio=('sentiment', lambda x: (x > 0.1).mean()),
        news_article_count=('sentiment', 'count'),
        news_confidence_avg=('confidence', 'mean')
    ).reset_index()
    
    # Fill NaN in sentiment_vol (when only 1 article)
    features['news_sentiment_vol'] = features['news_sentiment_vol'].fillna(0)
    
    # Convert month to end-of-month date for alignment
    features['date'] = features['month'].dt.to_timestamp() + pd.offsets.MonthEnd(0)
    
    # Select output columns
    output = features[[
        'ticker', 'date', 'news_avg_sentiment', 'news_sentiment_vol',
        'news_neg_ratio', 'news_pos_ratio', 'news_article_count', 'news_confidence_avg'
    ]]
    
    return output

def main():
    print("=" * 60)
    print("T1.3 FinBERT News Feature Engineering")
    print("=" * 60)
    
    # Step 1: Load partitioned data (2022-2026)
    print("\n1. Loading partitioned data...")
    partitioned_df = load_partitioned_data()
    print(f"   Loaded {len(partitioned_df):,} rows from partitioned files")
    
    # Step 2: Load all_scored.parquet (2024-2026)
    print("\n2. Loading all_scored.parquet...")
    scored_df = load_all_scored()
    print(f"   Loaded {len(scored_df):,} rows from all_scored.parquet")
    
    # Step 3: Merge data sources
    print("\n3. Merging data sources...")
    # Ensure consistent columns
    common_cols = ['ticker', 'published_at', 'title', 'text', 'source', 'publisher', 'sentiment', 'confidence']
    
    # Filter both dataframes to common columns
    partitioned_df = partitioned_df[common_cols].copy()
    scored_df = scored_df[common_cols].copy()
    
    # Combine
    combined_df = pd.concat([partitioned_df, scored_df], ignore_index=True)
    print(f"   Combined: {len(combined_df):,} rows")
    
    # Step 4: Deduplicate
    print("\n4. Deduplicating...")
    combined_df = deduplicate(combined_df)
    print(f"   After dedup: {len(combined_df):,} rows")
    
    # Step 5: Compute monthly features
    print("\n5. Computing monthly features...")
    features_df = compute_monthly_features(combined_df)
    print(f"   Generated {len(features_df):,} monthly feature records")
    
    # Step 6: Summary statistics
    print("\n6. Feature Summary:")
    print(f"   Tickers: {features_df['ticker'].nunique()}")
    print(f"   Date range: {features_df['date'].min()} to {features_df['date'].max()}")
    print(f"\n   Feature Statistics:")
    for col in ['news_avg_sentiment', 'news_sentiment_vol', 'news_neg_ratio', 
                'news_pos_ratio', 'news_article_count', 'news_confidence_avg']:
        print(f"   {col:25s}: mean={features_df[col].mean():.4f}, std={features_df[col].std():.4f}")
    
    # Step 7: Save output
    print("\n7. Saving output...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"   Saved to: {OUTPUT_FILE}")
    print(f"   File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return features_df

if __name__ == "__main__":
    features_df = main()
