#!/usr/bin/env python3
"""
us_dl_03_mcap.py — 下载所有训练股的市场总值(marketCap)
一次性任务：2474只 x yfinance.info → 保存到JSON
多线程加速 + checkpoint断点续传
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

BASE='/home/hermes/.hermes/openclaw-archive'
ML=f'{BASE}/ml'
OUT=f'{ML}/us_sym_mcap.json'
CKPT=f'{ML}/us_sym_mcap_checkpoint.json'
MAX_WORKERS=8  # yfinance有rate limit，8线程够用

T0=time.time()
print('='*60,flush=True)
print('下载市值数据 (marketCap)',flush=True)
print('='*60,flush=True)

# 1. 获取所有ticker
print('\n[1/3] 获取股票列表...',flush=True)
m1=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet',columns=['ticker']).drop_duplicates()
m2=pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet',columns=['sym']).drop_duplicates()
m2.rename(columns={'sym':'ticker'},inplace=True)
all_syms=sorted(set(m1['ticker'].tolist())|set(m2['ticker'].tolist()))
print(f'  共{len(all_syms)}只股票',flush=True)
del m1,m2

# 2. 加载已有的checkpoint
results={}
if os.path.exists(OUT):
    results=json.load(open(OUT))
    print(f'  已有缓存: {len(results)}只',flush=True)

if os.path.exists(CKPT):
    ckpt=json.load(open(CKPT))
    if len(ckpt)>len(results):
        results=ckpt
        print(f'  从checkpoint恢复: {len(results)}只',flush=True)

pending=[s for s in all_syms if s not in results]
print(f'  待下载: {len(pending)}只',flush=True)

if len(pending)==0:
    print('  全部完成!',flush=True)
    sys.exit(0)

# 3. 多线程下载
print('\n[2/3] 下载市值...',flush=True)
downloaded=0
errors=0
last_save=time.time()

def fetch_mcap(sym):
    try:
        t=yf.Ticker(sym)
        info=t.info
        mcap=info.get('marketCap')
        if mcap is None:
            mcap=info.get('enterpriseValue',0)
        return (sym, mcap if mcap else 0)
    except:
        return (sym, None)

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    futures={exe.submit(fetch_mcap,s): s for s in pending}
    for fut in as_completed(futures):
        sym, mcap=fut.result()
        if mcap is not None:
            results[sym]=int(mcap)
            downloaded+=1
        else:
            results[sym]=0
            errors+=1
        # 每100只保存一次checkpoint
        if (downloaded+errors)%100==0:
            json.dump(results,open(CKPT,'w'))
            elapsed=time.time()-T0
            rate=(downloaded+errors)/max(elapsed,1)
            remaining=len(pending)-(downloaded+errors)
            eta=remaining/max(rate,1)
            print(f'  {downloaded+errors}/{len(pending)} ({rate:.1f}/s, ETA {eta:.0f}s, errors {errors})',flush=True)

# 4. 保存结果
print('\n[3/3] 保存结果...',flush=True)
json.dump(results,open(OUT,'w'))
# 清理checkpoint
if os.path.exists(CKPT):
    os.remove(CKPT)

# 统计
mcaps=[v for v in results.values() if v>0]
print(f'  成功: {downloaded}只, 失败: {errors}只',flush=True)
print(f'  有市值: {len(mcaps)}只',flush=True)
print(f'  市值范围: {min(mcaps):,} ~ {max(mcaps):,}',flush=True)
print(f'  市值中位数: {np.median(mcaps):,}',flush=True)
# 过滤统计
min_mcap=300_000_000
filtered_count=sum(1 for v in mcaps if v>=min_mcap)
print(f'  市值≥$3亿: {filtered_count}只',flush=True)
print(f'  耗时: {time.time()-T0:.0f}s',flush=True)
print('='*60,flush=True)
