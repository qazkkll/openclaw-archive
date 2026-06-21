#!/usr/bin/env python3
"""
CEO: Test if fundamentals (PE/PB/PS/dividend) improve XGBoost
Quick IC test first, then full Walk-Forward if promising.
"""

import pandas as pd, numpy as np, json, time, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

t0 = time.time()
print(f"[CEO] Fundamental IC Test {time.strftime('%Y-%m-%d %H:%M')}")

# Load daily_basic
db = pd.read_parquet('data/cn/daily_basic.parquet')
db['sym'] = db['ts_code'].str[:6]
db['date'] = db['trade_date'].astype(int)

# Clean fundamentals
db['pe_clean'] = db['pe_ttm'].where(db['pe_ttm'] > 0, np.nan)
db['pb_clean'] = db['pb'].where(db['pb'] > 0, np.nan)
db['ps_clean'] = db['ps_ttm'].where(db['ps_ttm'] > 0, np.nan)
db['log_mv'] = np.log(db['circ_mv'].clip(lower=1))

# Rank features (cross-sectional)
for col in ['pe_clean', 'pb_clean', 'ps_clean', 'dv_ratio', 'log_mv', 'turnover_rate']:
    db[f'{col}_rk'] = db.groupby('date')[col].rank(pct=True)

# Load OHLCV for forward returns
df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)
df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()

# Forward returns
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)
df['fwd_20d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-20) / x - 1)

# Merge fundamentals
df = df.merge(db[['sym', 'date', 'pe_clean', 'pb_clean', 'ps_clean', 'dv_ratio', 'log_mv', 'turnover_rate',
                   'pe_clean_rk', 'pb_clean_rk', 'ps_clean_rk', 'dv_ratio_rk', 'log_mv_rk', 'turnover_rate_rk']],
              on=['sym', 'date'], how='left')

print(f"  Data: {len(df):,} rows ({time.time()-t0:.0f}s)")

# IC test: Pearson correlation between feature and forward return
fund_features = ['pe_clean', 'pb_clean', 'ps_clean', 'dv_ratio', 'log_mv', 'turnover_rate',
                 'pe_clean_rk', 'pb_clean_rk', 'ps_clean_rk', 'dv_ratio_rk', 'log_mv_rk', 'turnover_rate_rk']

print("\n📊 IC Test (Feature vs fwd_10d)")
print(f"{'Feature':>20} {'IC':>8} {'Rank IC':>8} {'Non-null':>10}")
print("-" * 50)

for feat in fund_features:
    valid = df[[feat, 'fwd_10d']].dropna()
    if len(valid) < 10000:
        continue
    ic = valid[feat].corr(valid['fwd_10d'])
    
    # Rank IC (Spearman)
    rank_ic = valid[feat].rank().corr(valid['fwd_10d'].rank())
    
    non_null = valid[feat].notna().mean()
    print(f"{feat:>20} {ic:>8.4f} {rank_ic:>8.4f} {non_null:>10.1%}")

# Also test with 20d forward
print("\n📊 IC Test (Feature vs fwd_20d)")
print(f"{'Feature':>20} {'IC':>8} {'Rank IC':>8}")
print("-" * 40)

for feat in fund_features:
    valid = df[[feat, 'fwd_20d']].dropna()
    if len(valid) < 10000:
        continue
    ic = valid[feat].corr(valid['fwd_20d'])
    rank_ic = valid[feat].rank().corr(valid['fwd_20d'].rank())
    print(f"{feat:>20} {ic:>8.4f} {rank_ic:>8.4f}")

# Test IC by year
print("\n📊 IC by Year (pe_clean_rk vs fwd_10d)")
df['year'] = df['date'] // 10000
for year in sorted(df['year'].unique()):
    if year < 2016 or year > 2026:
        continue
    yearly = df[df['year'] == year][['pe_clean_rk', 'fwd_10d']].dropna()
    if len(yearly) < 1000:
        continue
    ic = yearly['pe_clean_rk'].corr(yearly['fwd_10d'])
    print(f"  {year}: IC={ic:.4f}")

print(f"\n  Time: {time.time()-t0:.0f}s")
