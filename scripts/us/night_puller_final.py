#!/usr/bin/env python3
"""
拉取剩余数据: daily_ohlcv + top_list (龙虎榜)
已完成的: daily_basic, moneyflow_hsgt
"""
import os, sys, time, datetime as dt
import tushare as ts
import pandas as pd

TS_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
pro = ts.pro_api(TS_TOKEN)

WORKDIR = r'C:\workspace\av2'

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

def save_with_retry(df, path):
    """Save parquet with retry"""
    for attempt in range(3):
        try:
            df.to_parquet(path, index=False)
            return True
        except:
            time.sleep(2)
    return False

# Get trading days
df_cal = pro.trade_cal(start_date='20210101', end_date='20260531')
trading_days = sorted(df_cal[df_cal['is_open'] == 1]['cal_date'].tolist())
log(f'Trading days: {len(trading_days)}')

existing = os.listdir(WORKDIR)

# === daily OHLCV ===
if 'daily_ohlcv.parquet' not in existing:
    log('Pulling daily OHLCV...')
    all_daily = []
    for i, date in enumerate(trading_days):
        try:
            df = pro.daily(trade_date=date, fields='ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount')
            if df is not None and len(df) > 0:
                all_daily.append(df)
            time.sleep(0.1)
        except:
            time.sleep(2)
        # Save checkpoint every 200 days
        if (i+1) % 200 == 0:
            log(f'  daily OHLCV: {i+1}/{len(trading_days)}')
            if all_daily:
                ckpt = pd.concat(all_daily, ignore_index=True)
                save_with_retry(ckpt, os.path.join(WORKDIR, f'daily_ohlcv_ckpt_{i+1}.parquet'))
    
    if all_daily:
        combined = pd.concat(all_daily, ignore_index=True)
        save_with_retry(combined, os.path.join(WORKDIR, 'daily_ohlcv.parquet'))
        # Clean checkpoints
        for f in os.listdir(WORKDIR):
            if f.startswith('daily_ohlcv_ckpt_'):
                os.remove(os.path.join(WORKDIR, f))
        log(f'Saved daily OHLCV: {len(combined)} rows')
else:
    log('daily_ohlcv already exists')

# === top_list (龙虎榜) ===
if 'top_list.parquet' not in existing:
    log('Pulling top_list (2018-2026)...')
    all_lhb = []
    lhb_days = [d for d in trading_days if int(d[:4]) >= 2018]
    for i, date in enumerate(lhb_days):
        try:
            df = pro.top_list(trade_date=date)
            if df is not None and len(df) > 0:
                all_lhb.append(df)
            time.sleep(0.1)
        except:
            time.sleep(2)
        if (i+1) % 200 == 0:
            log(f'  top_list: {i+1}/{len(lhb_days)}')
    
    if all_lhb:
        combined = pd.concat(all_lhb, ignore_index=True)
        combined.to_parquet(os.path.join(WORKDIR, 'top_list.parquet'), index=False)
        log(f'Saved top_list: {len(combined)} rows')
else:
    log('top_list already exists')

# Summary
log('\n✅ All data pull complete!')
for f in os.listdir(WORKDIR):
    if f.endswith('.parquet'):
        sz = os.path.getsize(os.path.join(WORKDIR, f))
        log(f'  {f}: {sz/1024/1024:.1f}MB ({sz} bytes)')
