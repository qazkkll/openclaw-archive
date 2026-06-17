#!/usr/bin/env python3
"""Phase 1: 因子数据拉取器 v2 — 按日期拉比按股票拉快100倍"""
import os, sys, json, time, datetime as dt
import tushare as ts

TS_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
pro = ts.pro_api(TS_TOKEN)

WORKDIR = r'C:\workspace\av2'
os.makedirs(WORKDIR, exist_ok=True)

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

# === Generate trading day list ===
log('Getting trading calendar...')
df_cal = pro.trade_cal(start_date='20210101', end_date='20260531')
trading_days = sorted(df_cal[df_cal['is_open'] == 1]['cal_date'].tolist())
log(f'Trading days: {len(trading_days)}')

# === Pull daily_basic (PE/PB/turnover/vol_ratio) by date ===
# Much faster: one API call per trading day returns ALL stocks
log('Pulling daily_basic by date...')
all_db = []
for i, date in enumerate(trading_days):
    try:
        df = pro.daily_basic(trade_date=date, 
            fields='ts_code,trade_date,pe,pe_ttm,pb,turnover_rate,vol_ratio,ps,float_mv')
        if df is not None and len(df) > 0:
            all_db.append(df)
        time.sleep(0.12)
    except Exception as e:
        log(f'  Error {date}: {e}')
        time.sleep(2)
    if (i+1) % 100 == 0:
        log(f'  daily_basic: {i+1}/{len(trading_days)} days done')

log(f'Pulled {len(all_db)} days of daily_basic')
if all_db:
    import pandas as pd
    combined = pd.concat(all_db, ignore_index=True)
    combined.to_parquet(os.path.join(WORKDIR, 'daily_basic.parquet'), index=False)
    log(f'Saved daily_basic: {len(combined)} rows')
else:
    log('ERROR: No daily_basic data!')
    sys.exit(1)

# === Pull 北向资金 flow by date ===
log('Pulling northbound moneyflow (moneyflow_hsgt)...')
all_hsgt = []
for i, date in enumerate(trading_days):
    try:
        df = pro.moneyflow_hsgt(trade_date=date)
        if df is not None and len(df) > 0:
            all_hsgt.append(df)
        time.sleep(0.12)
    except:
        time.sleep(2)
    if (i+1) % 200 == 0:
        log(f'  moneyflow_hsgt: {i+1}/{len(trading_days)}')

if all_hsgt:
    dfs = [d for d in all_hsgt if d is not None and len(d) > 0]
    combined_hsgt = pd.concat(dfs, ignore_index=True)
    combined_hsgt.to_parquet(os.path.join(WORKDIR, 'moneyflow_hsgt.parquet'), index=False)
    log(f'Saved moneyflow_hsgt: {len(combined_hsgt)} rows')

# === Pull daily line data for factor computation ===
# Need OHLCV to compute additional factors and for backtesting
log('Pulling daily OHLCV data...')
all_daily = []
for i, date in enumerate(trading_days):
    try:
        df = pro.daily(trade_date=date,
            fields='ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount')
        if df is not None and len(df) > 0:
            all_daily.append(df)
        time.sleep(0.12)
    except:
        time.sleep(2)
    if (i+1) % 100 == 0:
        log(f'  daily: {i+1}/{len(trading_days)}')

if all_daily:
    combined_daily = pd.concat(all_daily, ignore_index=True)
    combined_daily.to_parquet(os.path.join(WORKDIR, 'daily_ohlcv.parquet'), index=False)
    log(f'Saved daily OHLCV: {len(combined_daily)} rows')

# === Pull 龙虎榜 (top_list per date) ===
# Only available from ~2018
log('Pulling 龙虎榜 (top_list)...')
all_lhb = []
for i, date in enumerate(trading_days):
    date_s = date
    if int(date_s[:4]) < 2018:  # Skip old data
        continue
    try:
        df = pro.top_list(trade_date=date_s)
        if df is not None and len(df) > 0:
            all_lhb.append(df)
        time.sleep(0.12)
    except:
        time.sleep(2)
    if (i+1) % 200 == 0:
        log(f'  top_list: {i+1}/{len(trading_days)}')

if all_lhb:
    combined_lhb = pd.concat(all_lhb, ignore_index=True)
    combined_lhb.to_parquet(os.path.join(WORKDIR, 'top_list.parquet'), index=False)
    log(f'Saved top_list: {len(combined_lhb)} rows')

print('\n✅ 因子数据拉取完成!')
print(f'  daily_basic: {os.path.getsize(os.path.join(WORKDIR,"daily_basic.parquet"))//1024//1024} MB')
print(f'  moneyflow_hsgt: {os.path.getsize(os.path.join(WORKDIR,"moneyflow_hsgt.parquet"))//1024//1024} MB')
print(f'  daily_ohlcv: {os.path.getsize(os.path.join(WORKDIR,"daily_ohlcv.parquet"))//1024//1024} MB')
