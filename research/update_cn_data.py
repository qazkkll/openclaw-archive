#!/usr/bin/env python3
"""
增量更新A股数据 (tushare)
更新 a_hist_10y.parquet 和 moneyflow_core.parquet
"""
import pandas as pd, numpy as np, json, time, os, sys
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("📊 A股数据增量更新")
print("="*60)

with open('data/config/tushare.json') as f:
    config = json.load(f)

import tushare as ts
ts.set_token(config['token'])
pro = ts.pro_api()

# 检查现有数据
print("\n[1] 检查现有数据...")
hist = pd.read_parquet('data/a_hist_10y.parquet')
hist['Date'] = hist['Date'].astype(str)
hist_last = hist['Date'].max()
print(f"  a_hist_10y: {hist_last} (最后日期), 类型: {hist['Date'].dtype}")

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['trade_date'] = mf['trade_date'].astype(str)
mf_last = mf['trade_date'].max()
print(f"  moneyflow_core: {mf_last} (最后日期)")

# 获取交易日历
print("\n[2] 获取交易日历...")
today_str = pd.Timestamp.now().strftime('%Y%m%d')
cal = pro.trade_cal(exchange='SSE', start_date=hist_last, end_date=today_str, is_open='1')
trade_dates = sorted(cal['cal_date'].tolist())
new_dates = [d for d in trade_dates if d > hist_last]
print(f"  需要更新: {len(new_dates)} 个交易日")

if not new_dates:
    print("\n✅ 数据已是最新")
    sys.exit(0)

# 拉取日线
print(f"\n[3] 拉取日线数据 ({len(new_dates)} 天)...")
all_daily = []
for i, d in enumerate(new_dates):
    try:
        df_d = pro.daily(trade_date=d)
        if df_d is not None and len(df_d) > 0:
            all_daily.append(df_d)
        time.sleep(0.35)
    except Exception as e:
        print(f"  ⚠️ {d}: {e}")
        time.sleep(1)

if all_daily:
    new_daily = pd.concat(all_daily, ignore_index=True)
    new_daily = new_daily.rename(columns={
        'ts_code': 'Code', 'trade_date': 'Date',
        'open': 'O', 'high': 'H', 'low': 'L', 'close': 'C', 'vol': 'V',
    })
    # tushare ts_code: "000001.SZ" -> existing: "000001"
    new_daily['Code'] = new_daily['Code'].str[:6]
    new_daily['Date'] = new_daily['Date'].astype(str)
    
    cols = ['Code', 'Date', 'O', 'H', 'L', 'C', 'V']
    new_daily = new_daily[[c for c in cols if c in new_daily.columns]]
    
    hist = pd.concat([hist, new_daily], ignore_index=True)
    hist = hist.drop_duplicates(subset=['Code', 'Date']).sort_values(['Code', 'Date']).reset_index(drop=True)
    hist.to_parquet('data/a_hist_10y.parquet', index=False)
    print(f"  ✅ 日线: +{len(new_daily)} 行, 总计 {len(hist):,}")
else:
    print("  ⚠️ 无新日线数据")

# 拉取资金流
print(f"\n[4] 拉取资金流数据...")
mf_new_dates = [d for d in new_dates if d > mf_last]
all_mf = []
for i, d in enumerate(mf_new_dates):
    try:
        df_m = pro.moneyflow(trade_date=d)
        if df_m is not None and len(df_m) > 0:
            all_mf.append(df_m)
        time.sleep(0.35)
    except Exception as e:
        print(f"  ⚠️ {d}: {e}")
        time.sleep(1)

if all_mf:
    new_mf = pd.concat(all_mf, ignore_index=True)
    new_mf['trade_date'] = new_mf['trade_date'].astype(str)
    
    mf = pd.concat([mf, new_mf], ignore_index=True)
    mf = mf.drop_duplicates(subset=['ts_code', 'trade_date']).sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    mf.to_parquet('data/cn/moneyflow_core.parquet', index=False)
    print(f"  ✅ 资金流: +{len(new_mf)} 行, 总计 {len(mf):,}")
else:
    print("  ⚠️ 无新资金流数据")

# 验证
print(f"\n[5] 验证...")
h2 = pd.read_parquet('data/a_hist_10y.parquet')
m2 = pd.read_parquet('data/cn/moneyflow_core.parquet')
print(f"  a_hist: {h2['Date'].astype(str).min()} ~ {h2['Date'].astype(str).max()}, {len(h2):,}行")
print(f"  moneyflow: {m2['trade_date'].astype(str).min()} ~ {m2['trade_date'].astype(str).max()}, {len(m2):,}行")
print(f"\n✅ 数据更新完成!")
