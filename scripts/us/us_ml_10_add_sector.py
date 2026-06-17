#!/usr/bin/env python3
"""拉yfinance每只股票sector/industry，合并到特征"""
import sys, os, json, time, warnings; warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, yfinance as yf

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
OUTPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5_sector.parquet'
CKPT = '/home/hermes/.hermes/openclaw-project/data/models/us/sector_batch_ckpt.json'

df_tree = pd.read_parquet(INPUT)
tickers = sorted(df_tree['ticker'].unique())
print(f"{len(tickers)}只股票, 读取sector信息...")

# 读已有断点
sector_map = {}
start = 0
if os.path.exists(CKPT):
    cp = json.load(open(CKPT))
    sector_map = cp.get('sector_map', {})
    start = cp.get('completed_to', 0)
    print(f"断点: {len(sector_map)}只已有, 继续从{start}拉")

# 拉sector（必须每只单独请求）
BATCH_SIZE = 30
T0 = time.time()
for i in range(start, len(tickers)):
    t = tickers[i]
    try:
        info = yf.Ticker(t).info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        sector_map[t] = {'sector': sector, 'industry': industry}
    except Exception as e:
        sector_map[t] = {'sector': 'Unknown', 'industry': 'Unknown'}
    
    if (i+1) % BATCH_SIZE == 0 or i == len(tickers) - 1:
        json.dump({'sector_map': sector_map, 'completed_to': i+1}, open(CKPT, 'w'))
        eta = (time.time() - T0) / (i+1) * (len(tickers) - i - 1) / 60
        print(f"  {i+1}/{len(tickers)} ({sector_map[t]['sector']}) | {(time.time()-T0)/60:.0f}min | ETA: {eta:.0f}min", flush=True)

# 统计
sectors = set(v['sector'] for v in sector_map.values())
industries = set(v['industry'] for v in sector_map.values())
print(f"\nsector: {len(sectors)}类, industry: {len(industries)}类")
print(f"sectors: {sorted(sectors)}")

# 写到parquet
sector_df = pd.DataFrame.from_dict(sector_map, orient='index')
sector_df.index.name = 'ticker'
sector_df = sector_df.reset_index()

# 合并到特征
df_merged = df_tree.merge(sector_df, on='ticker', how='left').fillna('Unknown')

# OneHot编码sector
sector_dummies = pd.get_dummies(df_merged['sector'], prefix='sec')
industry_dummies = pd.get_dummies(df_merged['industry'], prefix='ind')

# 只保留覆盖率高的industry（避免维度爆炸）
ind_counts = df_merged['industry'].value_counts()
top_industries = ind_counts[ind_counts > 50000].index  # >5万行的industry
industry_dummies = industry_dummies[[c for c in industry_dummies.columns if c.startswith('ind_') and c[4:] in top_industries]]
print(f"缩略industry: {industry_dummies.shape[1]}列")

df_out = pd.concat([df_merged, sector_dummies, industry_dummies], axis=1)
df_out = df_out.drop(columns=['sector', 'industry'])
del df_merged, sector_dummies, industry_dummies

print(f"\n输出: {df_out.shape[1]}列, {len(df_out):,}行")
df_out.to_parquet(OUTPUT, index=False)
print(f"保存: {OUTPUT}")

if os.path.exists(CKPT):
    os.remove(CKPT)
