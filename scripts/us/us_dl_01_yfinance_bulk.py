#!/usr/bin/env python3
"""yfinance批量下载美股5年日K - 分批+断点续传"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(f"脚本启动: {__file__}")

import pandas as pd
import yfinance as yf

# 手动写路径（跳过_paths编码问题）
BASE = "/home/hermes/.hermes/openclaw-archive"
ML_DIR = f"{BASE}/ml"
HIST_FILE = f"{BASE}/data/us_hist_clean.parquet"
OUTPUT = f"{ML_DIR}/us_hist_yf_5y.parquet"
CHECKPOINT = f"{ML_DIR}/us_dl_checkpoint.json"
BATCH_SIZE = 100
MIN_BARS = 252 * 3
print(f"输出: {OUTPUT}")

# 读ticker
print("读ticker列表...")
df = pd.read_parquet(HIST_FILE)
all_tickers = sorted(df['ticker'].unique())
print(f"  共 {len(all_tickers)} 只")
del df

# 断点
start_idx = 0
if os.path.exists(CHECKPOINT):
    cp = json.load(open(CHECKPOINT))
    start_idx = cp.get('completed_to', 0)
    print(f"  发现断点: 已拉到第 {start_idx}/{len(all_tickers)} 只")

TOTAL_T0 = time.time()

for batch_start in range(start_idx, len(all_tickers), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(all_tickers))
    batch = all_tickers[batch_start:batch_end]
    t0 = time.time()
    print(f"\n批次 {batch_start}~{batch_end}/{len(all_tickers)}...", flush=True)
    
    rows = []
    for t in batch:
        try:
            hist = yf.download(t, period='5y', progress=False)
            if hist is None or len(hist) < MIN_BARS:
                continue
            # yfinance有时返回MultiIndex列名, flatten
            if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
                hist.columns = hist.columns.droplevel(1)
            hist = hist.reset_index()
            for _, r in hist.iterrows():
                rows.append({
                    'ticker': t,
                    'date': r['Date'],
                    'open': float(r['Open']),
                    'high': float(r['High']),
                    'low': float(r['Low']),
                    'close': float(r['Close']),
                    'volume': int(r['Volume']),
                })
        except Exception as e:
            continue

    if not rows:
        print(f"  批次 {batch_start}~{batch_end}: 0行")
        json.dump({'completed_to': batch_end}, open(CHECKPOINT, 'w'))
        continue

    df_batch = pd.DataFrame(rows)
    
    if batch_start == 0 or not os.path.exists(OUTPUT):
        df_batch.to_parquet(OUTPUT, index=False)
        print(f"  创建文件: {len(df_batch)}行")
    else:
        old = pd.read_parquet(OUTPUT)
        combined = pd.concat([old, df_batch], ignore_index=True)
        combined.to_parquet(OUTPUT, index=False)
        del old, combined
        print(f"  追加完成: {len(df_batch)}行")

    json.dump({'completed_to': batch_end}, open(CHECKPOINT, 'w'))
    elapsed = time.time() - TOTAL_T0
    print(f"  ✅ {time.time()-t0:.0f}s, 累计{(batch_end/len(all_tickers)*100):.0f}%, 总耗时{elapsed/60:.0f}分钟", flush=True)

# 最终统计
df = pd.read_parquet(OUTPUT)
print(f"\n{'='*50}")
print(f"✅ 全部完成!")
print(f"  总行数: {len(df):,}")
print(f"  股票数: {df['ticker'].nunique()}")
print(f"  日期: {df['date'].min()} ~ {df['date'].max()}")
print(f"  总耗时: {(time.time()-TOTAL_T0)/60:.0f}分钟")

if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)
