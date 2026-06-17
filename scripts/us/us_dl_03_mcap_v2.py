#!/usr/bin/env python3
"""us_dl_03_mcap_v2.py — 补全mcap，串行+retry，不触发rate limit"""
import sys,os,json,time,warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf

ML='/home/hermes/.hermes/openclaw-archive/scripts/system'
OUT=f'{ML}/us_sym_mcap.json'
results=json.load(open(OUT)) if os.path.exists(OUT) else {}

# 原有成功数
existing=len(results)
print(f'已有: {existing}只, 其中{mcap_count}只有效' if 'mcap_count' not in dir() else f'已有: {existing}只')

# 找所有ticker
import pandas as pd
m1=pd.read_parquet(f'{ML}/us_hist_yf_10y.parquet',columns=['ticker']).drop_duplicates()
m2=pd.read_parquet(f'{ML}/us_hist_megacap_10y.parquet',columns=['sym']).drop_duplicates()
m2.rename(columns={'sym':'ticker'},inplace=True)
all_syms=sorted(set(m1['ticker'].tolist())|set(m2['ticker'].tolist()))

pending=[s for s in all_syms if s not in results or results[s]==0]
print(f'待下载: {len(pending)}只')
del m1,m2

done=0;fail=0;T0=time.time()
for i,sym in enumerate(pending):
    try:
        t=yf.Ticker(sym)
        info=t.info
        mcap=info.get('marketCap')
        if mcap is None:
            mcap=info.get('enterpriseValue',0) or 0
        results[sym]=int(mcap)
        done+=1
    except:
        results[sym]=0
        fail+=1
    if (i+1)%100==0:
        json.dump(results,open(OUT,'w'))
        elapsed=time.time()-T0
        print(f'  {i+1}/{len(pending)}, success {done}, fail {fail}, {elapsed:.0f}s',flush=True)
    time.sleep(1.0)  # 间隔1秒防ban

json.dump(results,open(OUT,'w'))
mcaps=[v for v in results.values() if v>0]
print(f'\n完成: success {done}, fail {fail}, total有市值 {len(mcaps)}只')
print(f'市值≥$3亿: {sum(1 for v in mcaps if v>=300_000_000)}只')
print(f'耗时: {time.time()-T0:.0f}s')
