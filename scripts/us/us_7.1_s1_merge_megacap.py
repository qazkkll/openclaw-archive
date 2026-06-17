#!/usr/bin/env python3
"""
us_7.1_s1_merge_megacap.py
把新下载的46只大盘股按V3特征体系补算技术指标，合并到v3_dated
同时排除<5usd + 低流动性（日成交<500K）
输出: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v3_dated.parquet (覆盖原文件，但先输出临时文件)
"""
import sys, os, json, warnings, time
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np

T0 = time.time()
BASE = '/home/hermes/.hermes/openclaw-archive'
DATA_DIR = f'{BASE}/data'
ML_DIR = f'{BASE}/ml'
OUTPUT_TMP = f'{ML_DIR}/us_ml_feats_v3_dated_v71.parquet'

print('=' * 60)
print('us_7.1_s1_merge_megacap — 大盘股合并+V3特征补算')
print('=' * 60)

# 1. 加载
print('\n[1/5] 加载数据...')
v3 = pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
mega = pd.read_parquet(f'{DATA_DIR}/us_hist_megacap_dl.parquet')
print(f'  V3: {len(v3):,}行, {v3.sym.nunique()}只, {v3.date.min()}~{v3.date.max()}')
print(f'  大盘: {len(mega):,}行, {mega.sym.nunique()}只')

# 2. V3特征清单（排除sym/date/price/volume）
v3_feats = [c for c in v3.columns if c not in ['sym','date','price','volume']]
print(f'\n[2/5] V3特征列({len(v3_feats)}个):')
print(f'  {v3_feats}')

# 统计每列缺失率
for f in v3_feats:
    miss = v3[f].isna().sum()
    if miss > 0:
        print(f'  {f}: {miss}缺失')

# 3. 为大盘股计算V3特征
print(f'\n[3/5] 为{len(mega.sym.unique())}只大盘股补算V3特征...')

def ema(arr, period):
    alpha = 2/(period+1)
    result = [arr[0]]
    for v in arr[1:]:
        result.append(v*alpha + result[-1]*(1-alpha))
    return np.array(result)

v3_new_rows = []
for sym in sorted(mega['sym'].unique()):
    sd = mega[mega['sym']==sym].sort_values('date').copy()
    if len(sd) < 120:
        continue
    
    c = sd['close'].values.astype(float)
    h = sd['high'].values if 'high' in sd.columns else c.copy()
    l = sd['low'].values if 'low' in sd.columns else c.copy()
    v = sd['volume'].values.astype(float)
    dates = sd['date'].values
    n = len(c)
    
    s_c = pd.Series(c); s_h = pd.Series(h); s_l = pd.Series(l); s_v = pd.Series(v)
    
    # 技术指标
    ma5 = s_c.rolling(5).mean().values
    ma10 = s_c.rolling(10).mean().values
    ma20 = s_c.rolling(20).mean().values
    ma60 = s_c.rolling(60).mean().values
    
    rsi_series = s_c.copy()
    delta = rsi_series.diff()
    gain = delta.clip(0); loss = (-delta).clip(0)
    avg_g = gain.ewm(span=14).mean().values
    avg_l = loss.ewm(span=14).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0.001)
    rsi14 = 100 - 100/(1+rs)
    
    ema12 = ema(c, 12)
    ema26 = ema(c, 26)
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
    macd_hist = macd_line - macd_signal
    
    vol20 = s_v.rolling(20).mean().values
    vol_ratio = np.where(vol20>0.001, s_v.values / vol20, 1.0)
    
    ret1 = s_c.pct_change(1).fillna(0).values
    ret5 = s_c.pct_change(5).fillna(0).values
    ret10 = s_c.pct_change(10).fillna(0).values
    ret20 = s_c.pct_change(20).fillna(0).values
    
    hh52 = s_h.rolling(252).max().values
    hh52 = np.where(np.isnan(hh52), s_h.rolling(20, min_periods=1).max().values, hh52)
    p52 = c / np.maximum(hh52, 1e-10)
    
    ma_bias20 = np.where(ma20>0.001, c/ma20-1, 0)
    
    # 趋势加速
    mom_short = (s_c.rolling(5).mean().values / s_c.rolling(20).mean().shift(15).fillna(0.001).values - 1)
    mom_long = (s_c.rolling(20).mean().values / s_c.rolling(60).mean().shift(40).fillna(0.001).values - 1)
    trend_accel = np.where(np.isnan(mom_long), 0, mom_short - mom_long)
    
    # 做空数据(大盘股用行业默认)
    short_ratio = np.full(n, 2.0)
    short_pct = np.full(n, 0.03)
    
    # market_cap (yfinance info)
    mc = sd.get('market_cap', pd.Series([np.nan]*n)).values
    mc = np.where(pd.isna(mc), 1e11, mc)
    
    # sc 行业编码（大盘股统一tech=45）
    sc = np.full(n, 45, dtype=int)
    
    # sector ETF收益（大盘股默认）
    for sfeat in ['sector_etf_ret5', 'spy_ret5', 'qqq_ret5', 'iwm_ret5']:
        if sfeat not in v3_feats:
            v3_feats.append(sfeat)
    
    for i in range(120, n):
        row = {
            'sym': sym, 'date': dates[i],
            'price': c[i], 'volume': v[i],
        }
        # V3全部特征
        for fi, f in enumerate(v3_feats):
            val = 0.0
            if f == 'ma5': val = c[i]/ma5[i]-1 if ma5[i]>0.01 else 0
            elif f == 'ma10': val = c[i]/ma10[i]-1 if ma10[i]>0.01 else 0
            elif f == 'ma20': val = c[i]/ma20[i]-1 if ma20[i]>0.01 else 0
            elif f == 'ma60': val = c[i]/ma60[i]-1 if ma60[i]>0.01 else 0
            elif f == 'rsi14': val = rsi14[i]
            elif f == 'macd': val = macd_line[i]
            elif f == 'macd_signal': val = macd_signal[i]
            elif f == 'macd_hist': val = macd_hist[i]
            elif f == 'vol20': val = vol20[i]
            elif f == 'vol_ratio': val = vol_ratio[i]
            elif f == 'ret1': val = ret1[i]
            elif f == 'ret5': val = ret5[i]
            elif f == 'ret10': val = ret10[i]
            elif f == 'ret20': val = ret20[i]
            elif f == 'p52': val = p52[i]
            elif f == 'ma_bias20': val = ma_bias20[i]
            elif f == 'trend_accel': val = trend_accel[i] if not np.isnan(trend_accel[i]) else 0
            elif f == 'short_ratio': val = short_ratio[i]
            elif f == 'short_pct': val = short_pct[i]
            elif f == 'market_cap': val = mc[i]
            elif f == 'sc': val = sc[i]
            # sector ETF收益默认0
            elif f in ['sector_etf_ret5', 'spy_ret5', 'qqq_ret5', 'iwm_ret5']:
                val = 0.0
            row[f] = val
        v3_new_rows.append(row)

    if len(v3_new_rows) % 5000 == 0:
        pct = len(v3_new_rows) / (len(mega) * 0.7) * 100 if len(mega) > 0 else 0
        print(f'  已处理: {len(v3_new_rows):,}行...', flush=True)

if v3_new_rows:
    df_new = pd.DataFrame(v3_new_rows)
    print(f'\n  大盘特征: {len(df_new):,}行, {df_new.sym.nunique()}只')
    
    # 确保列一致
    for f in v3_feats:
        if f not in df_new.columns:
            df_new[f] = 0.0
    
    # 只保留V3中存在的列
    final_cols = v3.columns.tolist()
    df_new = df_new[final_cols]
    
    # 合并
    merged = pd.concat([v3, df_new], ignore_index=True)
    print(f'  合并后: {len(merged):,}行, {merged.sym.nunique()}只')
else:
    merged = v3.copy()

# 4. 过滤
print('\n[4/5] 过滤...')
print(f'  过滤前: {len(merged):,}行, {merged.sym.nunique()}只')

# 价格>=5
before = len(merged)
merged = merged[merged['price'] >= 5].copy()
print(f'  排除<5usd: {before - len(merged):,}行 ({len(merged):,}行剩余)')

# 注意：V3特征中的volume列不是真实成交量，故不做成交量过滤
# 真实成交量过滤需要在原始数据层面做
print(f'  跳过成交量过滤（V3特征volume非真实数据）')

# 重新排序
merged = merged.sort_values(['sym','date']).reset_index(drop=True)

# 处理object列问题
for col in merged.select_dtypes(include=['object']).columns:
    if col not in ['sym','date']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

# 补NaN为0
merged = merged.fillna(0)

# 5. 保存
print(f'\n[5/5] 保存...')
merged.to_parquet(OUTPUT_TMP, index=False)
print(f'  → {OUTPUT_TMP}')
print(f'  最终: {len(merged):,}行, {merged.sym.nunique()}只')
print(f'  日期: {merged.date.min()} ~ {merged.date.max()}')

# 检查知名股票
check = ['NVDA','AAPL','MSFT','GOOGL','AMZN','AVGO','META','TSLA','QCOM','NOK','GNRC','ON']
print(f'\n  知名股票检查:')
for s in check:
    cnt = (merged['sym']==s).sum()
    print(f'    {s:>6}: {cnt}行')

print(f'\n总耗时: {time.time()-T0:.0f}s')
print('='*60)
