#!/usr/bin/env python3
"""
拉取2016-2021年价格数据，扩展features_v02.parquet到10年。
用yfinance批量下载（免费，无API key）。
"""
import sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')
PARQUET = DATA_DIR / 'features_v02.parquet'

# 加载现有数据获取ticker列表
print("📊 加载现有数据...")
existing = pd.read_parquet(PARQUET)
existing['date'] = existing['date'].astype(str)
tickers = sorted(existing['ticker'].unique())
print(f"   {len(tickers)} 只, 现有: {existing['date'].min()} ~ {existing['date'].max()}")

# 确定需要拉的日期范围
existing_min = existing['date'].min()
need_from = '2016-01-01'
need_to = existing_min  # 拉到现有数据的前一天

if need_from >= existing_min:
    print(f"   现有数据从{existing_min}开始，无需补充")
    sys.exit(0)

print(f"   需要拉: {need_from} ~ {need_to}")

# yfinance批量下载
print(f"\n📊 用yfinance批量下载 {len(tickers)} 只...")
t0 = time.time()

# yfinance download 批量接口
data = yf.download(
    tickers,
    start=need_from,
    end=need_to,
    group_by='ticker',
    auto_adjust=True,
    threads=True,
    progress=True,
)

print(f"   下载耗时: {time.time()-t0:.0f}秒")

# 转换为长格式
rows = []
failed = []
for ticker in tickers:
    try:
        if len(tickers) == 1:
            df = data
        else:
            df = data[ticker] if ticker in data.columns.get_level_values(0) else None
        
        if df is None or df.empty:
            failed.append(ticker)
            continue
        
        df = df.dropna(subset=['Close'])
        for idx, row in df.iterrows():
            d = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
            rows.append({
                'ticker': ticker,
                'date': d,
                'open': row.get('Open', None),
                'high': row.get('High', None),
                'low': row.get('Low', None),
                'close': row.get('Close', None),
                'volume': row.get('Volume', None),
                'vwap': None,  # yfinance没有VWAP
            })
    except Exception as e:
        failed.append(ticker)

new_df = pd.DataFrame(rows)
print(f"\n📊 结果:")
print(f"   成功: {len(tickers)-len(failed)}/{len(tickers)} 只")
print(f"   新数据: {len(new_df):,} 行")
print(f"   失败: {len(failed)} 只 ({','.join(failed[:10])}{'...' if len(failed)>10 else ''})")

if len(new_df) > 0:
    print(f"   日期范围: {new_df['date'].min()} ~ {new_df['date'].max()}")
    
    # 计算技术特征（与update_price_data.py一致）
    print("\n📊 计算技术特征...")
    
    all_features = []
    for i, ticker in enumerate(tickers):
        ticker_data = new_df[new_df['ticker'] == ticker].copy()
        if len(ticker_data) < 60:
            continue
        
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        c = ticker_data['close']
        v = ticker_data['volume']
        
        # MA
        ticker_data['ma5'] = c.rolling(5).mean()
        ticker_data['ma20'] = c.rolling(20).mean()
        ticker_data['ma60'] = c.rolling(60).mean()
        ticker_data['ma_bias20'] = (c - ticker_data['ma20']) / ticker_data['ma20']
        ticker_data['ma_align'] = ((ticker_data['ma5'] > ticker_data['ma20']) & (ticker_data['ma20'] > ticker_data['ma60'])).astype(float)
        ticker_data['ma_cross_5_20'] = (ticker_data['ma5'] > ticker_data['ma20']).astype(float)
        ticker_data['ma_cross_20_60'] = (ticker_data['ma20'] > ticker_data['ma60']).astype(float)
        
        # Price position
        h60 = ticker_data['high'].rolling(60).max()
        l60 = ticker_data['low'].rolling(60).min()
        ticker_data['price_position'] = (c - l60) / (h60 - l60 + 1e-10)
        
        # Returns
        for n in [1, 5, 10, 20, 30, 60, 90]:
            ticker_data[f'ret{n}'] = c.pct_change(n)
        
        # Momentum
        ticker_data['momentum_6m'] = c.pct_change(126)
        ticker_data['momentum_1m'] = c.pct_change(21)
        ticker_data['mom_divergence'] = ticker_data['momentum_6m'] - ticker_data['momentum_1m']
        ticker_data['trend_accel'] = ticker_data['momentum_1m'].diff(5)
        
        # Volatility
        ticker_data['vol20'] = c.pct_change().rolling(20).std() * np.sqrt(252)
        ticker_data['vol5'] = c.pct_change().rolling(5).std() * np.sqrt(252)
        ticker_data['vol_ratio'] = ticker_data['vol5'] / ticker_data['vol20'].clip(lower=1e-10)
        ticker_data['vol_change'] = ticker_data['vol20'].pct_change(5)
        ticker_data['vol_regime'] = (ticker_data['vol20'] > ticker_data['vol20'].rolling(60).mean()).astype(float)
        
        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.clip(lower=1e-10)
        ticker_data['rsi14'] = 100 - (100 / (1 + rs))
        ticker_data['rsi_change'] = ticker_data['rsi14'].diff(5)
        ticker_data['rsi_zone'] = pd.cut(ticker_data['rsi14'], bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(float)
        
        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        ticker_data['macd'] = ema12 - ema26
        ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9).mean()
        ticker_data['macd_hist'] = ticker_data['macd'] - ticker_data['macd_signal']
        ticker_data['macd_roc'] = ticker_data['macd_hist'].diff(3)
        
        # Bollinger Bands
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        ticker_data['bb_std'] = bb_std
        ticker_data['bb_width'] = (2 * bb_std) / bb_mid.clip(lower=1e-10)
        ticker_data['bb_pos'] = (c - bb_mid + bb_std) / (2 * bb_std + 1e-10)
        
        # Quality & Range
        ticker_data['ret_quality'] = c.pct_change().rolling(20).apply(lambda x: (x > 0).mean(), raw=True)
        ticker_data['range_ratio'] = (ticker_data['high'] - ticker_data['low']) / c.clip(lower=1e-10)
        ticker_data['avg_body'] = abs(c - ticker_data['open']) / c.clip(lower=1e-10)
        
        # VWAP drift (无VWAP数据时用MA20替代)
        ticker_data['vwap_drift'] = (c - ticker_data['ma20']) / ticker_data['ma20'].clip(lower=1e-10)
        
        # Drawdown & Volume
        ticker_data['dd_60'] = c / c.rolling(60).max() - 1
        ticker_data['ud_vol_ratio'] = (v * (c > c.shift(1)).astype(float)).rolling(20).sum() / (v * (c <= c.shift(1)).astype(float)).rolling(20).sum().clip(lower=1)
        
        # Beta (simplified: no SPY reference, use market proxy)
        mkt_ret = c.pct_change()  # placeholder
        ticker_data['beta'] = 1.0  # placeholder
        
        all_features.append(ticker_data)
        
        if (i+1) % 100 == 0:
            print(f"   {i+1}/{len(tickers)}")
    
    new_features = pd.concat(all_features, ignore_index=True)
    
    # 只保留价格+技术特征列（基本面列留空，由FMP PIT查询填充）
    price_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap',
                  'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'ma_cross_5_20', 'ma_cross_20_60',
                  'price_position', 'ret1', 'ret5', 'ret10', 'ret20', 'ret30', 'ret60', 'ret90',
                  'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
                  'vol20', 'vol5', 'vol_ratio', 'vol_change', 'vol_regime',
                  'rsi14', 'rsi_change', 'rsi_zone',
                  'macd', 'macd_signal', 'macd_hist', 'macd_roc',
                  'bb_std', 'bb_width', 'bb_pos',
                  'ret_quality', 'range_ratio', 'avg_body', 'vwap_drift',
                  'dd_60', 'ud_vol_ratio', 'beta']
    
    new_features = new_features[[c for c in price_cols if c in new_features.columns]]
    
    # 合并（新数据在前，旧数据在后，去重保留旧数据）
    print(f"\n📊 合并数据...")
    combined = pd.concat([new_features, existing], ignore_index=True)
    combined = combined.drop_duplicates(subset=['ticker', 'date'], keep='last')
    combined = combined.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    print(f"   合并后: {len(combined):,} 行")
    print(f"   日期范围: {combined['date'].min()} ~ {combined['date'].max()}")
    print(f"   Tickers: {combined['ticker'].nunique()}")
    
    # 保存
    combined.to_parquet(PARQUET, index=False)
    print(f"\n✅ 已保存: {PARQUET}")
    
    # 每年统计
    print(f"\n📊 每年数据量:")
    for y in range(2016, 2027):
        cnt = combined[combined['date'].str.startswith(str(y))]['date'].nunique()
        tick = combined[combined['date'].str.startswith(str(y))]['ticker'].nunique()
        print(f"   {y}: {cnt}天, {tick}只")

print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}秒")
