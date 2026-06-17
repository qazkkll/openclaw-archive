#!/usr/bin/env python3
"""
us_7.1_s2b_fundamentals.py — 拉基本面数据+合并到V3特征
用yfinance Ticker.info拉:
  - pb, roe, rev_growth, profit_growth
  - pe_forward, pe_trailing, market_cap, beta
  
输入: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v3_dated.parquet
输出: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v3_dated.parquet (更新+新列)
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, yfinance as yf

BASE='/home/hermes/.hermes/openclaw-archive'
ML_DIR=f'{BASE}/ml'

# 已拉过的缓存
CACHE=f'{ML_DIR}/us_fundamentals_v71.json'

print('='*60)
print('us_7.1_s2b — 拉基本面到V3特征池')
print('='*60)

# 1. 加载股票列表
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
syms=df['sym'].unique()
print(f'特征池: {len(df):,}行, {len(syms)}只')
print(f'日期: {df.date.min()}~{df.date.max()}')

# 2. 若缓存存在则加载
cache={}
if os.path.exists(CACHE):
    cache=json.load(open(CACHE,'r'))
    print(f'缓存: {len(cache)}只')

# 3. 拉基本面
new_fetch=0
for i,sym in enumerate(syms):
    if sym in cache:
        continue
    
    try:
        t=yf.Ticker(sym)
        info=t.info
        
        # 拖取关键字段
        fin={}
        for k in ['marketCap','trailingPE','forwardPE','priceToBook',
                   'returnOnEquity','revenueGrowth','earningsGrowth',
                   'beta','dividendYield','debtToEquity','grossMargins',
                   'profitMargins','currentRatio','bookValue']:
            fin[k]=info.get(k,None)
        
        cache[sym]=fin
        new_fetch+=1
    except:
        cache[sym]={}
        new_fetch+=1
    
    if (i+1)%50==0:
        json.dump(cache,open(CACHE,'w'))
        print(f'  [{i+1}/{len(syms)}] {new_fetch}只新拉, {time.time():.0f}',flush=True)

# 存缓存
json.dump(cache,open(CACHE,'w'))
print(f'\n基本面缓存完成: {len(cache)}/{len(syms)}')

# 4. 合并到V3特征
print('\n合并基本面到特征池...')

fund_feats={
    'marketCap':'market_cap','trailingPE':'pe_trailing','forwardPE':'pe_forward',
    'priceToBook':'pb','returnOnEquity':'roe','revenueGrowth':'rev_growth',
    'earningsGrowth':'profit_growth','beta':'beta','dividendYield':'div_yield',
    'debtToEquity':'debt_equity','grossMargins':'gross_margin',
    'profitMargins':'profit_margin','bookValue':'book_value',
}

# 为每行填充基本面值（按sym映射）
for src,tgt in fund_feats.items():
    vals={}
    for sym,v in cache.items():
        if src in v and v[src] is not None:
            vals[sym]=v[src]
    
    df[tgt]=df['sym'].map(vals)
    
    # 填充NaN
    if df[tgt].isna().sum()>0:
        med=df[tgt].median()
        df[tgt]=df[tgt].fillna(med)
    
    # 无限值
    df[tgt]=df[tgt].replace([np.inf,-np.inf],0)

# 覆盖率
print('\n基本面特征覆盖率:')
for src,tgt in fund_feats.items():
    valid=df[tgt].notna().sum()
    print(f'  {tgt:20s}: {valid}/{len(df)} ({valid/len(df)*100:.1f}%)')

# 知名股票基本面
print('\n知名股票基本面:')
for s in ['NVDA','AAPL','MSFT','GOOGL','AMZN','ABT','NOK']:
    if s in cache:
        info=cache[s]
        print(f'  {s:>6}: PE={info.get("trailingPE","?"):>6} Forward={info.get("forwardPE","?"):>6} ',
              f'PB={info.get("priceToBook","?"):>5} ROE={info.get("returnOnEquity","?"):>5}')

# 保存
df.to_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet',index=False)
print(f'\n✅ 已保存 ({len(df):,}行, {df.sym.nunique()}只)')
print('='*60)
