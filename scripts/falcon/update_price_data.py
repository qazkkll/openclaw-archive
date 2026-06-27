#!/usr/bin/env python3
"""
更新 features_v02.parquet 的价格数据到最新。
从 Massive.io 拉取 2025-01-01 到今天的价格，重新计算技术特征。
"""
import json, os, sys, time, urllib.request
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(Path('/home/hermes/.hermes/openclaw-archive/.env'))
KEY = os.environ.get('MASSIVE_API_KEY', '')
DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')
PARQUET = DATA_DIR / 'features_v02.parquet'

def fetch_prices(ticker, from_date='2025-01-01', to_date='2026-06-27'):
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?apiKey={KEY}&limit=5000"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        results = data.get('results', [])
        if not results:
            return None
        rows = []
        for bar in results:
            d = datetime.fromtimestamp(bar['t']/1000).strftime('%Y-%m-%d')
            rows.append({
                'date': d, 'open': bar.get('o'), 'high': bar.get('h'),
                'low': bar.get('l'), 'close': bar.get('c'),
                'volume': bar.get('v'), 'vwap': bar.get('vw'),
            })
        return rows
    except:
        return None

def compute_technical_features(df):
    """计算技术特征（与原features_v02一致）。"""
    df = df.sort_values('date').copy()
    c = df['close']
    v = df['volume']
    
    # MA
    df['ma5'] = c.rolling(5).mean()
    df['ma20'] = c.rolling(20).mean()
    df['ma60'] = c.rolling(60).mean()
    df['ma_bias20'] = (c - df['ma20']) / df['ma20']
    df['ma_align'] = ((df['ma5'] > df['ma20']) & (df['ma20'] > df['ma60'])).astype(float)
    df['ma_cross_5_20'] = (df['ma5'] > df['ma20']).astype(float)
    df['ma_cross_20_60'] = (df['ma20'] > df['ma60']).astype(float)
    
    # Price position
    h60 = df['high'].rolling(60).max()
    l60 = df['low'].rolling(60).min()
    df['price_position'] = (c - l60) / (h60 - l60 + 1e-10)
    
    # Returns
    for d in [1,5,10,20,30,60,90]:
        df[f'ret{d}'] = c.pct_change(d)
    
    # Momentum
    df['momentum_6m'] = c.pct_change(126)
    df['momentum_1m'] = c.pct_change(21)
    df['mom_divergence'] = df['momentum_6m'] - df['momentum_1m']
    df['trend_accel'] = df['ret20'] - df['ret20'].shift(20)
    
    # Volatility
    df['vol20'] = c.pct_change().rolling(20).std() * np.sqrt(252)
    df['vol5'] = c.pct_change().rolling(5).std() * np.sqrt(252)
    df['vol_ratio'] = df['vol5'] / (df['vol20'] + 1e-10)
    df['vol_change'] = df['vol20'] - df['vol20'].shift(20)
    df['vol_regime'] = (df['vol20'] > df['vol20'].rolling(60).mean()).astype(float)
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi14'] = 100 - (100 / (1 + rs))
    df['rsi_change'] = df['rsi14'] - df['rsi14'].shift(5)
    df['rsi_zone'] = ((df['rsi14'] > 30) & (df['rsi14'] < 70)).astype(float)
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_roc'] = df['macd_hist'].diff()
    
    # Bollinger Bands
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_std'] = bb_std
    df['bb_width'] = (2 * bb_std) / (bb_mid + 1e-10)
    df['bb_pos'] = (c - bb_mid) / (2 * bb_std + 1e-10)
    
    # Quality
    df['ret_quality'] = df['ret20'] / (df['vol20'] + 1e-10)
    df['range_ratio'] = (df['high'] - df['low']) / (c + 1e-10)
    df['avg_body'] = abs(c - df['open']).rolling(10).mean() / (c + 1e-10)
    df['vwap_drift'] = (c - df['vwap']) / (df['vwap'] + 1e-10) if 'vwap' in df.columns else 0
    
    # Drawdown
    peak = c.rolling(60).max()
    df['dd_60'] = (c - peak) / (peak + 1e-10)
    
    # Volume
    up_vol = v.where(c.diff() > 0, 0).rolling(20).sum()
    dn_vol = v.where(c.diff() < 0, 0).rolling(20).sum()
    df['ud_vol_ratio'] = up_vol / (dn_vol + 1e-10)
    
    # Beta (simplified)
    df['beta'] = 1.0  # placeholder, proper calc needs SPY returns
    
    return df

def main():
    t0 = time.time()
    
    # Load existing
    print("Loading existing parquet...")
    old = pd.read_parquet(PARQUET)
    tickers = sorted(old['ticker'].unique())
    old_max = old['date'].max()
    print(f"  {len(tickers)} tickers, data up to {old_max}")
    
    # Fetch new prices
    print(f"\nFetching prices from 2025-01-01 to today...")
    new_data = {}
    failed = []
    
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(fetch_prices, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            done += 1
            try:
                result = f.result()
                if result and len(result) > 0:
                    new_data[t] = result
                else:
                    failed.append(t)
            except:
                failed.append(t)
            if done % 100 == 0:
                print(f"  {done}/{len(tickers)}: {len(new_data)} OK, {len(failed)} failed")
    
    print(f"\n✅ Fetched: {len(new_data)}/{len(tickers)} ({len(failed)} failed)")
    
    if not new_data:
        print("❌ No new data fetched")
        return
    
    # Build new rows
    print("\nBuilding new rows + technical features...")
    new_rows = []
    for ticker, bars in new_data.items():
        df = pd.DataFrame(bars)
        df['ticker'] = ticker
        df = compute_technical_features(df)
        new_rows.append(df)
    
    new_df = pd.concat(new_rows, ignore_index=True)
    print(f"  New rows: {len(new_df)}")
    
    # Merge: keep old data + append new (deduplicate by ticker+date)
    print("\nMerging with existing data...")
    # Normalize date types — old has datetime.date objects, new has strings
    old['date'] = old['date'].astype(str)
    new_df['date'] = new_df['date'].astype(str)
    combined = pd.concat([old, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"  Combined: {len(combined)} rows, {combined['ticker'].nunique()} tickers")
    print(f"  Date range: {combined['date'].min()} ~ {combined['date'].max()}")
    
    # Save
    combined.to_parquet(PARQUET, index=False)
    print(f"\n💾 Saved to {PARQUET}")
    print(f"⏱️ {time.time()-t0:.0f}s")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Old: {len(old)} rows, up to {old_max}")
    print(f"  New: {len(combined)} rows, up to {combined['date'].max()}")
    print(f"  Added: {len(combined) - len(old)} rows")

if __name__ == '__main__':
    main()
