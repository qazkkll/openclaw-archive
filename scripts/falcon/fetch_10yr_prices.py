#!/home/hermes/.hermes/hermes-agent/venv/bin/python3
"""
拉2016-2021价格数据，扩展features_v02到10年。
yfinance批量下载 + 技术特征计算。
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = Path('/home/hermes/.hermes/openclaw-archive/data/falcon')
PARQUET = DATA_DIR / 'features_v02.parquet'

print("📊 加载现有数据...")
existing = pd.read_parquet(PARQUET)
existing['date'] = existing['date'].astype(str)
tickers = sorted(existing['ticker'].unique())
existing_min = existing['date'].min()
print(f"   {len(tickers)}只, 现有: {existing_min} ~ {existing['date'].max()}")

need_from = '2016-01-01'
need_to = existing_min
print(f"   需要: {need_from} ~ {need_to}")

t0 = time.time()

# 批量下载(分批50只避免超时)
BATCH = 50
all_rows = []
failed = []

for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i+BATCH]
    try:
        data = yf.download(batch, start=need_from, end=need_to,
                          group_by='ticker', auto_adjust=True,
                          threads=True, progress=False)
        
        for ticker in batch:
            try:
                if len(batch) == 1:
                    df = data
                else:
                    if ticker not in data.columns.get_level_values(0):
                        failed.append(ticker)
                        continue
                    df = data[ticker]
                
                df = df.dropna(subset=['Close'])
                if len(df) < 60:
                    failed.append(ticker)
                    continue
                
                for idx, row in df.iterrows():
                    d = idx.strftime('%Y-%m-%d')
                    all_rows.append({
                        'ticker': ticker, 'date': d,
                        'open': row['Open'], 'high': row['High'],
                        'low': row['Low'], 'close': row['Close'],
                        'volume': row['Volume'], 'vwap': None,
                    })
            except:
                failed.append(ticker)
    except Exception as e:
        failed.extend(batch)
    
    done = min(i+BATCH, len(tickers))
    if done % 200 == 0 or done == len(tickers):
        print(f"   {done}/{len(tickers)} ({time.time()-t0:.0f}s)")

print(f"\n📊 下载完成: {len(tickers)-len(failed)}/{len(tickers)}只, {len(all_rows):,}行")

if not all_rows:
    print("❌ 无数据"); sys.exit(1)

new_df = pd.DataFrame(all_rows)
print(f"   日期: {new_df['date'].min()} ~ {new_df['date'].max()}")

# 技术特征
print("\n📊 计算技术特征...")
features = []
for i, ticker in enumerate(tickers):
    if ticker in failed:
        continue
    td = new_df[new_df['ticker'] == ticker].sort_values('date').copy()
    if len(td) < 60:
        continue
    td = td.reset_index(drop=True)
    c = td['close']; v = td['volume']
    
    td['ma5'] = c.rolling(5).mean()
    td['ma20'] = c.rolling(20).mean()
    td['ma60'] = c.rolling(60).mean()
    td['ma_bias20'] = (c - td['ma20']) / td['ma20']
    td['ma_align'] = ((td['ma5'] > td['ma20']) & (td['ma20'] > td['ma60'])).astype(float)
    td['ma_cross_5_20'] = (td['ma5'] > td['ma20']).astype(float)
    td['ma_cross_20_60'] = (td['ma20'] > td['ma60']).astype(float)
    h60 = td['high'].rolling(60).max(); l60 = td['low'].rolling(60).min()
    td['price_position'] = (c - l60) / (h60 - l60 + 1e-10)
    for n in [1,5,10,20,30,60,90]: td[f'ret{n}'] = c.pct_change(n)
    td['momentum_6m'] = c.pct_change(126); td['momentum_1m'] = c.pct_change(21)
    td['mom_divergence'] = td['momentum_6m'] - td['momentum_1m']
    td['trend_accel'] = td['momentum_1m'].diff(5)
    td['vol20'] = c.pct_change().rolling(20).std() * np.sqrt(252)
    td['vol5'] = c.pct_change().rolling(5).std() * np.sqrt(252)
    td['vol_ratio'] = td['vol5'] / td['vol20'].clip(lower=1e-10)
    td['vol_change'] = td['vol20'].pct_change(5)
    td['vol_regime'] = (td['vol20'] > td['vol20'].rolling(60).mean()).astype(float)
    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = (-delta.clip(upper=0)).rolling(14).mean()
    td['rsi14'] = 100 - 100/(1+gain/loss.clip(lower=1e-10)); td['rsi_change'] = td['rsi14'].diff(5)
    td['rsi_zone'] = pd.cut(td['rsi14'], bins=[0,30,70,100], labels=[0,1,2]).astype(float)
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    td['macd'] = e12-e26; td['macd_signal'] = td['macd'].ewm(span=9).mean()
    td['macd_hist'] = td['macd']-td['macd_signal']; td['macd_roc'] = td['macd_hist'].diff(3)
    bb = c.rolling(20).mean(); bs = c.rolling(20).std()
    td['bb_std'] = bs; td['bb_width'] = 2*bs/bb.clip(lower=1e-10); td['bb_pos'] = (c-bb+bs)/(2*bs+1e-10)
    td['ret_quality'] = c.pct_change().rolling(20).apply(lambda x: (x>0).mean(), raw=True)
    td['range_ratio'] = (td['high']-td['low'])/c.clip(lower=1e-10)
    td['avg_body'] = abs(c-td['open'])/c.clip(lower=1e-10)
    td['vwap_drift'] = (c-td['ma20'])/td['ma20'].clip(lower=1e-10)
    td['dd_60'] = c/c.rolling(60).max()-1
    td['ud_vol_ratio'] = (v*(c>c.shift(1)).astype(float)).rolling(20).sum()/(v*(c<=c.shift(1)).astype(float)).rolling(20).sum().clip(lower=1)
    td['beta'] = 1.0
    features.append(td)
    if (i+1) % 100 == 0: print(f"   {i+1}/{len(tickers)-len(failed)}")

new_feat = pd.concat(features, ignore_index=True)
cols = ['ticker','date','open','high','low','close','volume','vwap',
        'ma5','ma20','ma60','ma_bias20','ma_align','ma_cross_5_20','ma_cross_20_60',
        'price_position','ret1','ret5','ret10','ret20','ret30','ret60','ret90',
        'momentum_6m','momentum_1m','mom_divergence','trend_accel',
        'vol20','vol5','vol_ratio','vol_change','vol_regime',
        'rsi14','rsi_change','rsi_zone','macd','macd_signal','macd_hist','macd_roc',
        'bb_std','bb_width','bb_pos','ret_quality','range_ratio','avg_body','vwap_drift',
        'dd_60','ud_vol_ratio','beta']
new_feat = new_feat[[c for c in cols if c in new_feat.columns]]

# 合并
print("\n📊 合并...")
combined = pd.concat([new_feat, existing], ignore_index=True)
combined = combined.drop_duplicates(subset=['ticker','date'], keep='last')
combined = combined.sort_values(['ticker','date']).reset_index(drop=True)

combined.to_parquet(PARQUET, index=False)
print(f"✅ 保存: {len(combined):,}行, {combined['ticker'].nunique()}只")
print(f"   日期: {combined['date'].min()} ~ {combined['date'].max()}")

for y in range(2016, 2027):
    cnt = combined[combined['date'].str.startswith(str(y))]['date'].nunique()
    print(f"   {y}: {cnt}天")

print(f"\n⏱️ {time.time()-t0:.0f}秒")
