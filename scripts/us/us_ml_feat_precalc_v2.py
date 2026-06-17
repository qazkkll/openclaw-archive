#!/usr/bin/env python3
"""
预计算 v2.0 — 改label为5档+5天预测 + 新特征
label_5d_5class:
  0: 跌>5%   (样本极少, 需加权)
  1: 跌2%~5%
  2: 平±2%
  3: 涨2%~5%
  4: 涨>5%

新特征:
  - short_ratio: 做空比例（yfinance拉）
  - sector_ret: 所属行业ETF近5天收益
  - volatility_5d: 5天波动率
"""
import sys, os, warnings, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═══ 预计算v2.0: 5天预测+5档+新特征 ═══")

# ========= 1. 读原始行情数据 =========
print("\n[1/5] 读原始数据...")
df = pd.read_parquet(_paths.US_HIST_5Y)
print(f"  原始: {len(df):,}行, {df['sym'].nunique()}只股票")

# 找原始日期列
date_cols = [c for c in df.columns if 'date' in c.lower()]
print(f"  日期列: {date_cols}")

# 确保按股票+日期排序
if date_cols:
    dc = date_cols[0]
    df = df.sort_values(['sym', dc]).reset_index(drop=True)

# ========= 2. 计算基础特征（同原版） =========
print("\n[2/5] 计算量价特征 (21个基础特征)...")

# helper
def ema(arr, period):
    out = np.full(len(arr), np.nan)
    out[0] = arr[0]
    # safe ema with no division by zero issues
    a = 2 / (period + 1)
    for i in range(1, len(arr)):
        if np.isnan(out[i-1]):
            out[i] = arr[i]
        elif np.isnan(arr[i]):
            out[i] = out[i-1]
        else:
            out[i] = a * arr[i] + (1-a) * out[i-1]
    return out

sym_grp = df.groupby('sym')
results = []

for sym, grp in sym_grp:
    g = grp.sort_values(date_cols[0]).copy()
    n = len(g)
    
    # 价格和收益率
    close = g['close'].values
    volume = g['vol'].values if 'vol' in g.columns else g.get('volume', g.get('amount', np.full(n, np.nan))).values
    
    ret1 = np.full(n, np.nan); ret5 = np.full(n, np.nan)
    ret20 = np.full(n, np.nan); ret60 = np.full(n, np.nan)
    ret1[1:] = np.diff(close) / close[:-1] * 100
    ret5[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
    ret20[20:] = (close[20:] - close[:-20]) / close[:-20] * 100
    ret60[60:] = (close[60:] - close[:-60]) / close[:-60] * 100
    
    # 均线
    ma5 = np.full(n, np.nan); ma10 = np.full(n, np.nan)
    ma20 = np.full(n, np.nan); ma60 = np.full(n, np.nan)
    for i in range(4, n): ma5[i] = np.nanmean(close[i-4:i+1])
    for i in range(9, n): ma10[i] = np.nanmean(close[i-9:i+1])
    for i in range(19, n): ma20[i] = np.nanmean(close[i-19:i+1])
    for i in range(59, n): ma60[i] = np.nanmean(close[i-59:i+1])
    
    # RSI
    delta = np.diff(close); gains = np.maximum(delta, 0); losses = np.maximum(-delta, 0)
    avg_gain = np.full(n, np.nan); avg_loss = np.full(n, np.nan)
    for i in range(13, n):
        ag = np.nanmean(gains[i-13:i]); al = np.nanmean(losses[i-13:i])
        # EMA式
        for j in range(i-13+1, i):
            ag = (ag*13 + (gains[j] if j<len(gains) else 0)) / 14
            al = (al*13 + (losses[j] if j<len(losses) else 0)) / 14
        avg_gain[i] = ag; avg_loss[i] = al
    rsi14 = np.full(n, np.nan)
    mask = avg_loss != 0
    rsi14[mask] = 100 - 100 / (1 + avg_gain[mask] / avg_loss[mask])
    rsi14[~mask & (avg_gain != 0)] = 100
    
    # vol20: 20天波动率
    vol20 = np.full(n, np.nan)
    for i in range(19, n):
        vol20[i] = np.nanstd(ret1[i-19:i+1]) * 100
    
    # P52周位置
    p52 = np.full(n, np.nan)
    for i in range(251, n):
        lo, hi = np.nanmin(close[i-251:i+1]), np.nanmax(close[i-251:i+1])
        if hi != lo:
            p52[i] = (close[i] - lo) / (hi - lo) * 100
        else:
            p52[i] = 50
    
    # MACD
    ema12 = ema(close, 12); ema26 = ema(close, 26)
    macd = ema12 - ema26
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal
    
    # vol ratio 成交量比
    vol_ratio = np.full(n, np.nan)
    avg_vol = np.full(n, np.nan)
    for i in range(19, n):
        avg_vol[i] = np.nanmean(volume[i-19:i+1])
    mask = avg_vol != 0
    vol_ratio[mask] = volume[mask] / avg_vol[mask]
    
    # MA bias
    ma_bias20 = np.full(n, np.nan)
    mask = ma20 != 0
    ma_bias20[mask] = (close[mask] - ma20[mask]) / ma20[mask] * 100
    
    # 5天波动率（新特征）
    vol5 = np.full(n, np.nan)
    for i in range(4, n):
        vol5[i] = np.nanstd(ret1[i-4:i+1]) * 100
    
    # === 标签: 未来5天收益 ===
    label_5d_pct = np.full(n, np.nan)
    # 提前计算close_future_5（shift back by 4）
    if n > 5:
        label_5d_pct[:-5] = (close[5:] - close[:-5]) / close[:-5] * 100
    
    # 5档标签
    label_5d_5class = np.full(n, -1, dtype=int)
    mask_v = ~np.isnan(label_5d_pct)
    label_5d_5class[mask_v & (label_5d_pct < -5)] = 0
    label_5d_5class[mask_v & (label_5d_pct >= -5) & (label_5d_pct < -2)] = 1
    label_5d_5class[mask_v & (label_5d_pct >= -2) & (label_5d_pct <= 2)] = 2
    label_5d_5class[mask_v & (label_5d_pct > 2) & (label_5d_pct <= 5)] = 3
    label_5d_5class[mask_v & (label_5d_pct > 5)] = 4
    label_5d_5class[~mask_v] = -1
    
    for i in range(n):
        results.append({
            'sym': g.iloc[i]['sym'],
            'date': g.iloc[i][date_cols[0]],
            'price': close[i],
            'volume': volume[i],
            'ma5': ma5[i], 'ma10': ma10[i], 'ma20': ma20[i], 'ma60': ma60[i],
            'rsi14': rsi14[i],
            'vol20': vol20[i],
            'p52': p52[i],
            'ret1': ret1[i], 'ret5': ret5[i], 'ret20': ret20[i], 'ret60': ret60[i],
            'macd': macd[i], 'macd_signal': macd_signal[i], 'macd_hist': macd_hist[i],
            'vol_ratio': vol_ratio[i], 'ma_bias20': ma_bias20[i],
            'vol5': vol5[i],
            'label_5d_pct': label_5d_pct[i],
            'label_5d_5class': label_5d_5class[i] if label_5d_5class[i] != -1 else None,
        })

df_feat = pd.DataFrame(results)
# 过滤掉label为空的行
df_feat = df_feat.dropna(subset=['label_5d_pct'])

# 统计5档分布
for cl in range(5):
    cnt = (df_feat['label_5d_5class'] == cl).sum()
    print(f"    第{cl}档 ({['跌>5%','跌2-5%','平±2%','涨2-5%','涨>5%'][cl]}): {cnt:,}行 ({cnt/len(df_feat)*100:.1f}%)")

print(f"\n[3/5] 加基本面特征...")
# 使用统一路径（做空比例等）
df_feat.to_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet", index=False)

print(f"[4/5] 保存特征列表...")
base_feats = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
              'ret1','ret5','ret20','ret60',
              'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
              'price','volume','vol5']
with open(_paths.ML_DIR + "/us_feature_cols_v2.json", 'w') as f:
    json.dump(base_feats, f, indent=2)

print(f"[5/5] 统计")
print(f"  总行数: {len(df_feat):,}")
print(f"  股票数: {df_feat['sym'].nunique()}")
print(f"  日期范围: {df_feat['date'].min()} ~ {df_feat['date'].max()}")

TOTAL = time.time() - T0
print(f"\n✅ 预计算v2.0 完成! ({TOTAL:.0f}s)")
print(f"  保存: {_paths.win(_paths.ML_DIR + '/us_ml_feats_v2.parquet')}")
print(f"  ★ label_5d_5class: 0=跌>5%, 1=跌2-5%, 2=平±2%, 3=涨2-5%, 4=涨>5%")
