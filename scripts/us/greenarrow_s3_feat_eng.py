#!/usr/bin/env python3
"""
绿箭 S3：特征工程 — 干净版
"""
import json, warnings, os, sys, time
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

DATA_DIR = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'
OUT_DIR = '/home/hermes/.hermes/openclaw-archive/output'

# 读取所有chunk
all_rows = []
for f in sorted(os.listdir(DATA_DIR)):
    if not f.startswith('sp500_chunk_') or not f.endswith('.json'):
        continue
    data = json.load(open(os.path.join(DATA_DIR, f), 'r'))
    for sym, bars in data.items():
        for b in bars:
            b['Code'] = sym
        all_rows.extend(bars)

df = pd.DataFrame(all_rows)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
print(f'总记录: {len(df)}, 股票: {df["Code"].nunique()}')

# ========= 单只股票特征计算 =========
def calc(sym, g):
    g = g.sort_values('Date').reset_index(drop=True)
    n = len(g)
    c = g['C'].values
    v = g['V'].values.astype(float)
    h = g['H'].values
    lo = g['L'].values
    
    feat = {'Code': [sym]*n, 'Date': g['Date'].values}
    
    # 动量 (ret_{d}d)
    for d in [1, 3, 5, 10, 20]:
        arr = np.full(n, np.nan)
        if n > d:
            arr[d:] = (c[d:] - c[:-d]) / c[:-d]
        feat[f'ret_{d}d'] = arr
    
    # 均线 + ratio
    for w in [5, 10, 20, 50]:
        ma = pd.Series(c).rolling(w).mean().values
        feat[f'ma_{w}'] = ma
        feat[f'ma_{w}_ratio'] = np.where(ma > 0, (c - ma) / ma, np.nan)
    
    # 日收益率
    dr = np.full(n, np.nan)
    dr[1:] = (c[1:] - c[:-1]) / c[:-1]
    
    # 波动率
    for w in [5, 10, 20]:
        feat[f'vol_{w}d'] = pd.Series(dr).rolling(w).std().values
    
    # RSI(14)
    gain = np.where(dr > 0, dr, 0)
    loss = np.where(dr < 0, -dr, 0)
    ag = pd.Series(gain).rolling(14).mean().values
    al = pd.Series(loss).rolling(14).mean().values
    rs = np.where(al > 1e-10, ag / al, 0)
    feat['rsi_14'] = 100 - 100 / (1 + rs)
    
    # 成交量
    for w in [5, 20]:
        vma = pd.Series(v).rolling(w).mean().values
        feat[f'vol_ma_{w}'] = vma
        feat[f'vol_ratio_{w}'] = np.where(vma > 0, v / vma, 1.0)
    
    # 价格位置 (HHV-LLV)
    for w in [20, 50, 100]:
        hh = pd.Series(h).rolling(w).max().values
        ll = pd.Series(lo).rolling(w).min().values
        feat[f'price_pos_{w}'] = np.where((hh - ll) > 1e-10, (c - ll) / (hh - ll), np.nan)
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    macd = ema12 - ema26
    sig = pd.Series(macd).ewm(span=9).mean().values
    feat['macd'] = macd
    feat['macd_sig'] = sig
    feat['macd_hist'] = macd - sig
    
    # ATR 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-lo[i], abs(h[i]-c[i-1]), abs(lo[i]-c[i-1]))
    atr = pd.Series(tr).rolling(14).mean().values
    feat['atr_14'] = atr
    feat['atr_pct'] = np.where(c > 0, atr / c, 0)
    
    # 标签：未来5日收益
    ret_f5 = np.full(n, np.nan)
    if n > 5:
        ret_f5[:-5] = (c[5:] - c[:-5]) / c[:-5]
    feat['ret_f5'] = ret_f5
    
    # 标签：未来5日涨>3% (分类标签)
    feat['label_buy'] = (ret_f5 > 0.03).astype(float)
    feat['label_sell'] = (ret_f5 < -0.02).astype(float)
    
    # 标签：无持仓（既非买也非卖）
    feat['label_hold'] = ((ret_f5 >= -0.02) & (ret_f5 <= 0.03)).astype(float)
    
    return pd.DataFrame(feat)

# 按股票分组计算
syms = sorted(df['Code'].unique())
print(f'计算特征: {len(syms)}只...')
results = []
t0 = time.time()
for i, sym in enumerate(syms):
    if (i+1) % 50 == 0:
        print(f'  [{i+1}/{len(syms)}] {time.time()-t0:.0f}s')
    g = df[df['Code'] == sym].copy()
    try:
        res = calc(sym, g)
        results.append(res)
    except Exception as e:
        print(f'  ERR {sym}: {str(e)[:60]}')

feat = pd.concat(results, ignore_index=True)
print(f'\n特征完成: {feat.shape}')
print(f'列: {list(feat.columns)}')

# 检查标签分布
print(f'\n标签分布:')
print(f'  买入(涨>3%): {feat["label_buy"].sum():.0f} ({feat["label_buy"].mean()*100:.1f}%)')
print(f'  卖出(跌>2%): {feat["label_sell"].sum():.0f} ({feat["label_sell"].mean()*100:.1f}%)')
print(f'  持有: {feat["label_hold"].sum():.0f} ({feat["label_hold"].mean()*100:.1f}%)')

# 保存
feat.to_parquet(f'{OUT_DIR}/sp500_feats.parquet', index=False)
print(f'\n保存: {OUT_DIR}/sp500_feats.parquet')
print(f'完成: {time.strftime("%Y-%m-%d %H:%M")}')
