#!/usr/bin/env python3
"""
us_dl_01_hist_aligned.py — 从本地us_hist_clean.parquet + yfinance日期索引
生成带日期列的多只股票K线数据集。
分批处理，断点续传，低内存。

输出: /home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_5y.parquet (2436只 x 5年 OHLCV)
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import yfinance as yf

# Config
HIST_PARQUET = '/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet'
OUTPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_5y.parquet'
CHECKPOINT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_dl_checkpoint.json'
BATCH_SIZE = 300          # 每批300只
N_YEARS = 5               # 取5年
N_BARS_TARGET = 252 * N_YEARS  # 1255根K线

print(f"us_dl_01: 从{HIST_PARQUET}生成{N_YEARS}年对齐日K线数据")

# 1. 拉日期索引（大盘ETF，快）
print("拉日期索引(SPY)...")
spy = yf.download('SPY', period=f'{N_YEARS}y', progress=False)
if hasattr(spy.columns, 'nlevels') and spy.columns.nlevels > 1:
    spy.columns = ['_'.join(c).strip('_') for c in spy.columns.to_flat_index()]
spy = spy.reset_index()
dates = spy['index'].values
ND = len(dates)
print(f"  日期范围: {dates[0]} ~ {dates[-1]}, {ND}天")

# 2. 读全部ticker
print("读ticker列表...")
src = pd.read_parquet(HIST_PARQUET)
all_tickers = sorted(src['ticker'].unique())
del src
print(f"  共 {len(all_tickers)} 只")

# 3. 断点检查
start_idx = 0
if os.path.exists(CHECKPOINT):
    cp = json.load(open(CHECKPOINT))
    start_idx = cp.get('completed_to', 0)
    print(f"  断点: 已处理 {start_idx}/{len(all_tickers)}")

# 4. 分批构建
T0 = time.time()
for batch_start in range(start_idx, len(all_tickers), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(all_tickers))
    t0 = time.time()
    
    # 读这批股票数据
    batch_src = pd.read_parquet(HIST_PARQUET, filters=[('ticker', 'in', all_tickers[batch_start:batch_end])])
    
    # 对齐日期并构建行
    rows = []
    for _, row in batch_src.iterrows():
        t = row['ticker']
        for i in range(ND):
            rows.append({
                'ticker': t,
                'date': dates[i],
                'close': float(row['c'][-(ND-i)]),
                'high': float(row['h'][-(ND-i)]),
                'low': float(row['l'][-(ND-i)]),
                'volume': int(row['v'][-(ND-i)]),
                'open': 0.0,
            })
    
    df_batch = pd.DataFrame(rows)
    del batch_src, rows
    
    # 写盘
    if batch_start == 0 or not os.path.exists(OUTPUT):
        df_batch.to_parquet(OUTPUT, index=False)
    else:
        old = pd.read_parquet(OUTPUT)
        combined = pd.concat([old, df_batch], ignore_index=True)
        combined.to_parquet(OUTPUT, index=False)
        del old, combined
    
    del df_batch
    json.dump({'completed_to': batch_end}, open(CHECKPOINT, 'w'))
    
    sec = time.time() - t0
    total = time.time() - T0
    print(f"  {batch_start}~{batch_end}: {sec:.0f}s, {total/60:.0f}min总", flush=True)

# 完成
df = pd.read_parquet(OUTPUT)
print(f"\nDONE! 总行数:{len(df):,}, 股票:{df['ticker'].nunique()}, 日期:{ND}, 耗时:{(time.time()-T0)/60:.0f}分钟")
if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
del df
