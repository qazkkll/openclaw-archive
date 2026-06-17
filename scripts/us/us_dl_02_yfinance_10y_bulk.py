#!/usr/bin/env python3
"""us_dl_02_yfinance_10y_bulk.py — yfinance批量下载美股10年日K
从us_dl_01复制改造，改period='max'（实际返回最多10-15年数据）
分批+断点续传

输出: /home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_10y.parquet
也同时补megacap的10年数据
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(f"脚本启动: {__file__}")

import pandas as pd
import yfinance as yf

BASE = "/home/hermes/.hermes/openclaw-archive"
ML_DIR = f"{BASE}/ml"
OUTPUT = f"{ML_DIR}/us_hist_yf_10y.parquet"
MEGA_OUTPUT = f"{BASE}/ml/us_hist_megacap_10y.parquet"
CHECKPOINT = f"{ML_DIR}/us_dl_10y_checkpoint.json"
BATCH_SIZE = 100
MIN_BARS = 252 * 3  # 至少3年（有些票IPO晚，但3年是最低线）

print(f"输出: {OUTPUT}")

# 合并ticker列表: yf5的2436只 + megacap的46只（去重）
print("读ticker列表...")
yf5 = pd.read_parquet(f"{ML_DIR}/us_hist_yf_5y.parquet")
yf_tickers = sorted(yf5['ticker'].unique())
print(f"  主池: {len(yf_tickers)} 只")

mega = pd.read_parquet(f"{BASE}/data/us_hist_megacap_dl.parquet")
mega_tickers = sorted(mega['sym'].unique())
# megacap里要去掉已经能在yf5中拉到的（yfinance本身能拉大盘股）
# 但为了完整性和更好的覆盖，megacap另起一支单独拉10年
print(f"  大盘: {len(mega_tickers)} 只（单独拉10年）")
del yf5, mega

# 断点
start_idx = 0
if os.path.exists(CHECKPOINT):
    cp = json.load(open(CHECKPOINT))
    start_idx = cp.get('completed_to', 0)
    print(f"  发现断点: 已拉到第 {start_idx}/{len(yf_tickers)} 只")

TOTAL_T0 = time.time()
failed = []

for batch_start in range(start_idx, len(yf_tickers), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(yf_tickers))
    batch = yf_tickers[batch_start:batch_end]
    t0 = time.time()
    print(f"\n批次 {batch_start}~{batch_end}/{len(yf_tickers)}...", flush=True)
    
    rows = []
    for t in batch:
        try:
            # period='max' — yfinance自动返回全部可用历史（一般10-15年）
            # 对美股来说max ≈ 1980年代至今，但这里不需要那么远
            # 用period='10y'更精确
            hist = yf.download(t, period='10y', progress=False)
            if hist is None or len(hist) < MIN_BARS:
                # 如果10年不够3年，试试max（可能是新股）
                hist = yf.download(t, period='max', progress=False)
            if hist is None or len(hist) < 504:  # 至少2年
                failed.append(t)
                continue
            if len(rows) == 0:
                print(f'    首只{t}: {len(hist)}行~...', flush=True)
            if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
                hist.columns = hist.columns.droplevel(1)
            hist = hist.reset_index()
            # yfinance新版reset_index后列名叫'index'（因为index没名字），rename成'date'
            if 'index' in hist.columns:
                hist.rename(columns={'index': 'date'}, inplace=True)
            # 某些旧版yfinance列名叫'Date'
            date_col = 'date' if 'date' in hist.columns else 'Date'
            for _, r in hist.iterrows():
                rows.append({
                    'ticker': t,
                    'date': r[date_col],
                    'open': float(r['Open']),
                    'high': float(r['High']),
                    'low': float(r['Low']),
                    'close': float(r['Close']),
                    'volume': int(r['Volume']),
                })
        except Exception as e:
            failed.append(t)
            continue

    if not rows:
        print(f"  批次 {batch_start}~{batch_end}: 0行（全部失败）")
        json.dump({'completed_to': batch_end, 'failed': failed}, open(CHECKPOINT, 'w'))
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

    json.dump({'completed_to': batch_end, 'failed': failed}, open(CHECKPOINT, 'w'))
    elapsed = time.time() - TOTAL_T0
    avg_min_per_100 = elapsed / (batch_end) * 100 / 60 if batch_end > 0 else 0
    remaining = (len(yf_tickers) - batch_end) / BATCH_SIZE * avg_min_per_100
    print(f"  ✅ {time.time()-t0:.0f}s, 累计{(batch_end/len(yf_tickers)*100):.0f}%, "
          f"ETA剩余:{remaining:.0f}分钟, 总耗时{elapsed/60:.0f}分", flush=True)

# 统计
df = pd.read_parquet(OUTPUT)
print(f"\n{'='*50}")
print(f"✅ 主池下载完成!")
print(f"  总行数: {len(df):,}")
print(f"  股票数: {df['ticker'].nunique()}")
print(f"  日期: {df['date'].min()} ~ {df['date'].max()}")
print(f"  失败: {len(failed)} 只")
if failed:
    print(f"  Failed: {failed}")
print(f"  总耗时: {(time.time()-TOTAL_T0)/60:.0f}分钟")

# === 第二步：拉megacap的10年数据（单独存盘）===
print(f"\n{'='*50}")
print(f"开始下载megacap 46只的10年数据...")
print(f"{'='*50}")
mega_rows = []
for t in mega_tickers:
    try:
        hist = yf.download(t, period='10y', progress=False)
        if hist is None or len(hist) < 504:
            hist = yf.download(t, period='max', progress=False)
        if hist is None or len(hist) < 504:
            print(f"  ❌ {t}: 数据不足")
            continue
        if hasattr(hist.columns, 'nlevels') and hist.columns.nlevels > 1:
            hist.columns = hist.columns.droplevel(1)
        hist = hist.reset_index()
        if 'index' in hist.columns:
            hist.rename(columns={'index': 'date'}, inplace=True)
        date_col = 'date' if 'date' in hist.columns else 'Date'
        for _, r in hist.iterrows():
            mega_rows.append({
                'sym': t,
                'date': r[date_col],
                'open': float(r['Open']),
                'high': float(r['High']),
                'low': float(r['Low']),
                'close': float(r['Close']),
                'volume': int(r['Volume']),
            })
        print(f"  ✅ {t}: {len(hist)}行", flush=True)
    except Exception as e:
        print(f"  ❌ {t}: {e}")
        continue

mega_df = pd.DataFrame(mega_rows)
mega_df.to_parquet(MEGA_OUTPUT, index=False)
print(f"\n✅ megacap完成: {len(mega_df):,}行, {mega_df['sym'].nunique()}只")

if os.path.exists(CHECKPOINT):
    os.remove(CHECKPOINT)

print(f"\n{'='*50}")
print(f"🎉 全部完成! 总耗时: {(time.time()-TOTAL_T0)/60:.0f}分钟")
print(f"{'='*50}")
