#!/usr/bin/env python3
"""
us_7.1_s5_add_v19feats.py — 给V7.1/V3特征添加V19的ETF收益特征
用yfinance拉 SPY/QQQ/IWM + 行业ETF 的5日收益
合并到V3特征生成 us_ml_feats_v71_v19.parquet
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, yfinance as yf

BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'
T0=time.time()

print('='*60)
print('V7.1→V19: 添加ETF收益特征')
print('='*60)

# 行业→ETF映射
S2E={
    'Technology':'XLK','Financial Services':'XLF','Financial':'XLF',
    'Energy':'XLE','Healthcare':'XLV','Industrials':'XLI',
    'Consumer Defensive':'XLP','Consumer Cyclical':'XLY','Utilities':'XLU',
    'Basic Materials':'XLB','Materials':'XLB','Real Estate':'XLRE',
    'Communication Services':'XLC','Semiconductors':'SMH',
}

ETF_SYMS=['SPY','QQQ','IWM']+list(S2E.values())
ETF_SYMS=list(set(ETF_SYMS))
print(f'ETF数量: {len(ETF_SYMS)}: {ETF_SYMS}')

# === 1. 加载V3特征 ===
print('\n[1/4] 加载V3特征...')
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
print(f'  {len(df):,}行, {df.sym.nunique()}只, 日期{df.date.min()}~{df.date.max()}')

# === 2. 拉ETF数据 ===
print('\n[2/4] 下载ETF日K线...')
etf_data={}
for e in ETF_SYMS:
    try:
        t=yf.Ticker(e)
        h=t.history(period='6mo')
        if len(h)==0: continue
        h=h.reset_index()
        h['Date']=pd.to_datetime(h['Date'])
        # 提取date→close
        closes={d.strftime('%Y-%m-%d'):float(h.loc[i,'Close']) for i,d in enumerate(h['Date'])}
        etf_data[e]=closes
    except Exception as ex:
        print(f'  ⚠️ {e}: {ex}')
print(f'  成功: {len(etf_data)}只ETF')

# === 3. 计算5日收益 ===
print('\n[3/4] 计算5日收益...')
def get_ret5(closes):
    """返回 {date: ret5} 映射"""
    dates=sorted(closes.keys())
    ret={}
    for i,d in enumerate(dates):
        if i>=5:
            ret[d]=(closes[d]-closes[dates[i-5]])/closes[dates[i-5]]
        else:
            ret[d]=0.0
    return ret

all_etf_rets={}
for e,closes in etf_data.items():
    all_etf_rets[e]=get_ret5(closes)

# 添加大盘ETF收益列
for e,label in [('SPY','spy_ret5'),('QQQ','qqq_ret5'),('IWM','iwm_ret5')]:
    er=all_etf_rets.get(e,{})
    df[label]=df['date'].astype(str).map(er).fillna(0.0)
    print(f'  {label}: {df[label].notna().sum()}/{len(df)}')

# 行业ETF收益
def sector_etf_ret(row):
    s=row.get('sector')
    e=S2E.get(s)
    if not e or e not in all_etf_rets:
        return all_etf_rets.get('SPY',{}).get(str(row['date'])[:10],0.0)
    return all_etf_rets[e].get(str(row['date'])[:10],0.0)

df['sector_etf_ret5']=df.apply(sector_etf_ret,axis=1)
print(f'  sector_etf_ret5: {df.sector_etf_ret5.notna().sum()}/{len(df)}')

# sector编码
df['sc']=df['sector'].astype('category').cat.codes.astype(int)
print(f'  sc: {df.sc.nunique()}级')

# === 4. 保存 ===
print('\n[4/4] 保存合并特征...')
OUT=f'{ML_DIR}/us_ml_feats_v71_v19.parquet'
df.to_parquet(OUT,index=False)
print(f'  => {OUT}')
print(f'  列: {df.columns.tolist()}')
print(f'\n总耗时: {time.time()-T0:.0f}s')
print('='*60)
