#!/usr/bin/env python3
"""
Phase 1b: 继续拉取剩余因子数据
跳过已完成的 daily_basic，只拉moneyflow_hsgt + daily OHLCV + top_list
"""
import os, sys, json, time, datetime as dt
import tushare as ts
import pandas as pd

TS_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
pro = ts.pro_api(TS_TOKEN)

WORKDIR = r'C:\workspace\av2'
os.makedirs(WORKDIR, exist_ok=True)

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

# Check which files already exist
existing = [f for f in os.listdir(WORKDIR) if f.endswith('.parquet')]
log(f'Already done: {existing}')

# Generate trading days - same as before
df_cal = pro.trade_cal(start_date='20210101', end_date='20260531')
trading_days = sorted(df_cal[df_cal['is_open'] == 1]['cal_date'].tolist())
log(f'Trading days: {len(trading_days)}')

# === Pull moneyflow_hsgt (北向资金) ===
if 'moneyflow_hsgt.parquet' not in existing:
    log('Pulling moneyflow_hsgt by date...')
    all_hsgt = []
    for i, date in enumerate(trading_days):
        try:
            df = pro.moneyflow_hsgt(trade_date=date)
            if df is not None and len(df) > 0:
                all_hsgt.append(df)
            time.sleep(0.1)
        except:
            time.sleep(2)
        if (i+1) % 200 == 0:
            log(f'  moneyflow_hsgt: {i+1}/{len(trading_days)}')
            if len(all_hsgt) > 0:
                pd.concat(all_hsgt, ignore_index=True).to_parquet(
                    os.path.join(WORKDIR, f'moneyflow_hsgt_checkpoint.parquet'), index=False)
    
    if all_hsgt:
        combined = pd.concat(all_hsgt, ignore_index=True)
        combined.to_parquet(os.path.join(WORKDIR, 'moneyflow_hsgt.parquet'), index=False)
        log(f'Saved moneyflow_hsgt: {len(combined)} rows')
else:
    log('moneyflow_hsgt already exists, skipping')

# === Pull daily OHLCV ===
if 'daily_ohlcv.parquet' not in existing:
    log('Pulling daily OHLCV by date...')
    all_daily = []
    for i, date in enumerate(trading_days):
        try:
            df = pro.daily(trade_date=date,
                fields='ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount')
            if df is not None and len(df) > 0:
                all_daily.append(df)
            time.sleep(0.1)
        except:
            time.sleep(2)
        if (i+1) % 100 == 0:
            log(f'  daily: {i+1}/{len(trading_days)}')
    
    if all_daily:
        combined = pd.concat(all_daily, ignore_index=True)
        combined.to_parquet(os.path.join(WORKDIR, 'daily_ohlcv.parquet'), index=False)
        log(f'Saved daily OHLCV: {len(combined)} rows')
else:
    log('daily_ohlcv already exists, skipping')

# === Pull top_list (龙虎榜) ===
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
    log('top_list already exists, skipping')

log('\n✅ 剩余数据拉取完成!')
for f in os.listdir(WORKDIR):
    if f.endswith('.parquet'):
        sz = os.path.getsize(os.path.join(WORKDIR, f)) // (1024*1024)
        log(f'  {f}: {sz} MB')
