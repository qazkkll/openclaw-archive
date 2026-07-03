#!/usr/bin/env python3
"""
快速下载2026年价格数据 + 重建features（只跑2026年）
"""
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"

# S&P 500 tickers (top 100 by market cap for speed)
SPX_TOP100 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "GOOG", "AVGO", "LLY",
    "JPM", "TSLA", "V", "UNH", "XOM", "MA", "JNJ", "PG", "COST", "HD",
    "ABBV", "MRK", "CRM", "BAC", "NFLX", "AMD", "KO", "CVX", "PEP", "TMO",
    "WMT", "LIN", "CSCO", "ACN", "MCD", "ABT", "DHR", "ORCL", "TXN", "ADBE",
    "NEE", "PM", "INTC", "CMCSA", "VZ", "IBM", "CRM", "QCOM", "BA", "CAT",
    "GE", "AMGN", "LOW", "INTU", "SPGI", "ISRG", "RTX", "PLD", "NOW", "T",
    "DE", "BLK", "SYK", "GS", "ADI", "MDLZ", "BKNG", "MMC", "AXP", "SCHW",
    "CB", "LRCX", "GILD", "VRTX", "REGN", "CI", "MO", "CME", "PANW", "CL",
    "ZTS", "FI", "SNPS", "CDNS", "SHW", "ICE", "DUK", "SO", "BSX", "AON",
    "COP", "SLB", "EOG", "PSX", "MPC", "VLO", "OXY", "HES", "DVN", "FANG",
]

def download_price(ticker, start="2026-01-01"):
    try:
        df = yf.download(ticker, start=start, progress=False, timeout=15)
        if df.empty:
            return None
        df = df.reset_index()
        df["ticker"] = ticker
        df = df.rename(columns={"Date": "date", "Open": "open", "High": "high", 
                                "Low": "low", "Close": "close", "Volume": "volume"})
        return df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None

def main():
    print("🦅 快速下载2026年价格数据 (Top 100)...")
    
    # Check existing prices
    if PRICES_PATH.exists():
        existing = pd.read_parquet(PRICES_PATH)
        existing_dates = set(existing["date"].astype(str).unique())
        print(f"  已有价格数据: {len(existing)} 行, {len(existing_dates)} 天")
        if "2026-07-01" in existing_dates:
            print("  ✅ 数据已是最新的，跳过下载")
            return
    else:
        existing = None
        print("  无价格数据，开始下载...")
    
    # Download with thread pool
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_price, t): t for t in SPX_TOP100}
        done = 0
        for future in as_completed(futures):
            ticker = futures[future]
            done += 1
            try:
                df = future.result()
                if df is not None and len(df) > 0:
                    results[ticker] = df
                    if done % 20 == 0:
                        print(f"  进度: {done}/{len(SPX_TOP100)} ({len(results)} 成功)")
            except Exception:
                pass
    
    print(f"  ✅ 下载完成: {len(results)}/{len(SPX_TOP100)} 成功")
    
    # Merge with existing
    new_data = pd.concat(results.values(), ignore_index=True)
    new_data["date"] = new_data["date"].astype(str)
    
    if existing is not None:
        # Keep existing data, add new
        existing["date"] = existing["date"].astype(str)
        all_data = pd.concat([existing, new_data], ignore_index=True)
        all_data = all_data.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        all_data = new_data
    
    all_data.to_parquet(PRICES_PATH, index=False)
    print(f"  💾 保存: {len(all_data)} 行 → {PRICES_PATH}")
    
    # Quick feature build (simplified - just use prices for simulation)
    print("\n📊 构建简化特征...")
    prices = all_data.copy()
    prices["date"] = prices["date"].astype(str)
    
    # For each ticker, compute basic features
    feature_frames = []
    for ticker in prices["ticker"].unique():
        tdf = prices[prices["ticker"] == ticker].sort_values("date").copy()
        if len(tdf) < 30:
            continue
        
        tdf["ret_1d"] = tdf["close"].pct_change()
        tdf["ret_5d"] = tdf["close"].pct_change(5)
        tdf["ret_20d"] = tdf["close"].pct_change(20)
        tdf["vol_20d"] = tdf["ret_1d"].rolling(20).std()
        tdf["rsi_14"] = 100 - (100 / (1 + tdf["ret_1d"].rolling(14).apply(
            lambda x: x[x > 0].sum() / abs(x[x < 0].sum()) if abs(x[x < 0].sum()) > 0 else 1)))
        tdf["ma_20"] = tdf["close"].rolling(20).mean()
        tdf["ma_50"] = tdf["close"].rolling(50).mean()
        tdf["price_to_ma20"] = tdf["close"] / tdf["ma_20"]
        tdf["ticker"] = ticker
        feature_frames.append(tdf)
    
    features = pd.concat(feature_frames, ignore_index=True)
    features.to_parquet(DATA_DIR / "features_quick_2026.parquet", index=False)
    print(f"  💾 保存: {len(features)} 行 → features_quick_2026.parquet")
    
    # Also save as features_v04_1 for compatibility
    # (Need to merge with fundamental data if available)
    print("\n⚠️ 注意: 这是简化特征（只有价格/技术面），不含基本面因子")
    print("  完整V0.4.6评分需要 features_v04_1.parquet（含53个基本面因子）")
    print("  重建完整特征需要 FMP 数据 + build_features_v041.py")

if __name__ == "__main__":
    main()
