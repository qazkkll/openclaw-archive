#!/usr/bin/env python3
"""Phase 1: 夜间因子数据拉取器 - 跑在本地Windows"""
import os, sys, json, time, datetime as dt
import tushare as ts

TS_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
pro = ts.pro_api(TS_TOKEN)

WORKDIR = r'C:\workspace\av2'
os.makedirs(WORKDIR, exist_ok=True)

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

# === Step 1: Get quality pool stocks ===
log('Loading quality pool from cloud...')
# Use the existing 1500 pool
df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,industry,list_date,market')
df_all = df[df['list_date'].notna()].copy()
df_all = df_all[df_all['list_date'] <= '20260101']
# Filter: only A-share
df_all = df_all[df_all['ts_code'].str.endswith(('.SZ', '.SH'))]
# Remove BJ
df_all = df_all[~df_all['ts_code'].str.startswith('8')]
df_all = df_all[~df_all['ts_code'].str.startswith('4')]
# Save
df_all.to_csv(os.path.join(WORKDIR, 'stock_basic.csv'), index=False)
log(f'Total stocks: {len(df_all)}')

# === Step 2: Pull daily_basic (PE/PB/turnover/vol_ratio) ===
# Limit to top 1500 by last 3 years for now - use quality pool from fetch_universe_1500
# Actually let's just pull for ALL stocks in batches
log('Pulling daily_basic...')
codes_all = df_all['ts_code'].tolist()
# Split into batches of 200
batch_size = 200
batches = [codes_all[i:i+batch_size] for i in range(0, len(codes_all), batch_size)]

all_factors = []
for bi, batch in enumerate(batches):
    log(f'  daily_basic batch {bi+1}/{len(batches)} ({len(batch)} stocks)')
    for j, code in enumerate(batch):
        try:
            # Last 5 years of data
            df_daily = pro.daily_basic(ts_code=code, start_date='20210101', end_date='20260531',
                                       fields='ts_code,trade_date,pe,pe_ttm,pb,turnover_rate,vol_ratio,ps,float_mv')
            if df_daily is not None and len(df_daily) > 0:
                all_factors.append(df_daily)
            time.sleep(0.15)  # rate limit
        except Exception as e:
            log(f'    Error {code}: {e}')
            time.sleep(1)
    # Save intermediate after each batch
    if len(all_factors) > 0:
        combined = all_factors[0] if len(all_factors) == 1 else all_factors[0].copy()
        for dfp in all_factors[1:]:
            combined = combined._append(dfp, ignore_index=True) if hasattr(combined, '_append') else combined.append(dfp, ignore_index=True, ignore_index=True)
    log(f'  Pulled {len(all_factors)} batches so far')

# === Step 3: Pull moneyflow (个股资金流) ===
# Last 2 years
log('Pulling moneyflow...')
all_mf = []
for bi, batch in enumerate(batches):
    log(f'  moneyflow batch {bi+1}/{len(batches)}')
    for j, code in enumerate(batch):
        try:
            df_mf = pro.moneyflow(ts_code=code, start_date='20240101', end_date='20260531',
                                  fields='ts_code,trade_date,buy_sm_vol,sell_sm_vol,buy_md_vol,sell_md_vol,buy_lg_vol,sell_lg_vol,buy_elg_vol,sell_elg_vol,net_mf_vol')
            if df_mf is not None and len(df_mf) > 0:
                all_mf.append(df_mf)
            time.sleep(0.15)
        except Exception as e:
            log(f'    Error {code}: {e}')
            time.sleep(1)

log('Factor pulling complete!')

# Save (use json for simplicity)
log(f'Saving {len(all_factors)} factor records...')
print('Factor pull complete!')
