#!/usr/bin/env python3
"""
美股ML特征预计算
================
1. 从 us_hist_clean.parquet 批量计算技术指标
2. 保存为 parquet + feature_cols.json
3. 后续训练直接读parquet，跳过重复计算

用法: python3 scripts/us_ml_feat_precalc.py
输出: /home/hermes/.hermes/openclaw-archive/data/us_ml_feats.parquet
      /home/hermes/.hermes/openclaw-archive/data/us_feature_cols.json
"""

import sys, json, os, time, math, math
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np

WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
BASE = os.path.join(WORKSPACE, "data")
ML_DIR = "/home/hermes/.hermes/openclaw-archive_ml"
os.makedirs(ML_DIR, exist_ok=True)
T0 = time.time()

# ─── 参数 ───
MIN_DAYS = 252
N_LATEST = 750  # 每只只取最后750天（约3年，特征充分且数据够用）

print("[1/3] 加载美股K线...")
t = time.time()
with open(f"{BASE}/us_hist_clean.parquet", 'r') as f:
    all_data = json.load(f)
print(f"  总池: {len(all_data)}只 ({time.time()-t:.0f}s)")

# ─── 特征计算函数（单只股票） ───
def compute_feats(c, v, sym):
    """输入：close数组, volume数组；输出：list[dict]"""
    n = len(c)
    if n < MIN_DAYS:
        return []
    
    start = max(0, n - N_LATEST)
    c = c[start:]
    v = v[start:]
    n = len(c)
    
    # ── 均线 ──
    def sma(data, p):
        res = [float('nan')] * (p - 1)
        for i in range(p - 1, len(data)):
            res.append(sum(data[i-p+1:i+1]) / p)
        return res
    
    ma5 = sma(c, 5)
    ma10 = sma(c, 10)
    ma20 = sma(c, 20)
    ma60 = sma(c, 60)
    
    # ── RSI-14 ──
    rsi14 = [float('nan')] * n
    gains = [max(c[i] - c[i-1], 0) for i in range(1, n)]
    losses = [max(c[i-1] - c[i], 0) for i in range(1, n)]
    for i in range(14, n):
        if i == 14:
            avg_g = sum(gains[:14]) / 14
            avg_l = sum(losses[:14]) / 14
        else:
            avg_g = (avg_g * 13 + gains[i-1]) / 14
            avg_l = (avg_l * 13 + losses[i-1]) / 14
        rsi14[i] = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100
    
    # ── 收益率(日/周/月) ──
    ret1 = [float('nan')] + [(c[i] / c[i-1] - 1) * 100 for i in range(1, n)]
    ret5 = [float('nan')] * 5 + [(c[i] / c[i-5] - 1) * 100 for i in range(5, n)]
    ret20 = [float('nan')] * 20 + [(c[i] / c[i-20] - 1) * 100 for i in range(20, n)]
    ret60 = [float('nan')] * 60 + [(c[i] / c[i-60] - 1) * 100 for i in range(60, n)]
    
    # ── 波动率 ──
    vol20 = [float('nan')] * 20
    returns = [c[i] / c[i-1] - 1 for i in range(1, n)]
    for i in range(20, n):
        vol20.append(np.std(returns[i-20:i]) * math.sqrt(252) * 100)
    
    # ── 52周高低位 ──
    p52 = [float('nan')] * 252
    for i in range(252, n):
        lo = min(c[i-252:i+1])
        hi = max(c[i-252:i+1])
        p52.append((c[i] - lo) / (hi - lo) * 100 if hi > lo else 50)
    
    # ── MACD ──
    def ema(data, p):
        k = 2 / (p + 1)
        res = [data[0]]
        for v in data[1:]:
            res.append(v * k + res[-1] * (1 - k))
        return res
    e12 = ema(c, 12)
    e26 = ema(c, 26)
    macd = [e12[i] - e26[i] for i in range(n)]
    macd_signal = sma(macd, 9)
    macd_hist = [macd[i] - (macd_signal[i] if not np.isnan(macd_signal[i]) else 0) for i in range(n)]
    
    # ── 量相关 ──
    vol_ma5 = sma(v, 5)
    vol_ratio = [v[i] / vol_ma5[i] if vol_ma5[i] > 0 else float('nan') for i in range(n)]
    
    # ── 价格与MA偏离 ──
    ma_bias20 = [(c[i] / ma20[i] - 1) * 100 if ma20[i] > 0 else float('nan') for i in range(n)]
    
    # ── 标签（明日涨跌幅） ──
    label_pct = [float('nan')] * (n - 1) + [float('nan')]
    for i in range(n - 1):
        label_pct[i] = (c[i+1] / c[i] - 1) * 100
    
    # ── 组装行 ──
    rows = []
    for i in range(60, n):  # 需要60根预热数据
        vals = {
            'sym': sym, 'price': c[i], 'volume': v[i] / 1e6,
            'ma5': ma5[i], 'ma10': ma10[i], 'ma20': ma20[i], 'ma60': ma60[i],
            'rsi14': rsi14[i], 'vol20': vol20[i], 'p52': p52[i],
            'ret1': ret1[i], 'ret5': ret5[i], 'ret20': ret20[i], 'ret60': ret60[i],
            'macd': macd[i], 'macd_signal': macd_signal[i], 'macd_hist': macd_hist[i],
            'vol_ratio': vol_ratio[i], 'ma_bias20': ma_bias20[i],
            'label_pct': label_pct[i],
        }
        # 过滤掉任何nan的特征（但不检查label_pct）
        skip = False
        for k, vv in vals.items():
            if k == 'label_pct':
                continue
            if isinstance(vv, float) and (np.isnan(vv) or np.isinf(vv)):
                skip = True
                break
        if not skip:
            rows.append(vals)
    return rows


print("[2/3] 计算特征（批量）...")
t = time.time()
all_rows = []
syms = list(all_data.keys())
skip = 0

batch_size = 200
for batch_start in range(0, len(syms), batch_size):
    batch = syms[batch_start:batch_start + batch_size]
    for sym in batch:
        d = all_data[sym]
        c = d.get('c', [])
        v = d.get('v', [])
        if len(c) < MIN_DAYS or len(v) < MIN_DAYS:
            skip += 1
            continue
        try:
            rows = compute_feats(c, v, sym)
            all_rows.extend(rows)
        except:
            skip += 1
            continue
    
    pct = min(100, (batch_start + len(batch)) / len(syms) * 100)
    print(f"  {batch_start + len(batch)}/{len(syms)} ({pct:.0f}%) 有效{len(all_rows)}行 ({time.time()-t:.0f}s)", flush=True)

print(f"  skip={skip}, 总行数={len(all_rows)}, 耗时={time.time()-t:.0f}s")

if len(all_rows) == 0:
    print("❌ 没有有效数据")
    sys.exit(1)

# ── 转DataFrame ──
df = pd.DataFrame(all_rows)

# 分档标签
def bucket(pct):
    if pd.isna(pct):
        return -1
    if pct > 5: return 3
    if pct > 0: return 2
    if pct > -5: return 1
    return 0
df['label_bucket'] = df['label_pct'].apply(bucket)

print(f"\n  Label分布:")
for i, name in [(0,'大跌<-5%'),(1,'小跌-5~0%'),(2,'小涨0~5%'),(3,'大涨>5%')]:
    n = (df['label_bucket'] == i).sum()
    print(f"    {name}: {n} ({n/len(df)*100:.1f}%)")

# ── 特征列定义 ──
feature_cols = ['ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                'ret1','ret5','ret20','ret60',
                'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                'price','volume']
print(f"\n  特征数: {len(feature_cols)}")
print(f"  特征: {feature_cols}")

# 保存
print("\n[3/3] 保存parquet...")
t = time.time()
df.to_parquet(f"{ML_DIR}/us_ml_feats.parquet", index=False)
with open(f"{ML_DIR}/us_feature_cols.json", 'w') as f:
    json.dump(feature_cols, f)

TOTAL = time.time() - T0
print(f"  parquet: {os.path.getsize(f'{ML_DIR}/us_ml_feats.parquet')/1024/1024:.0f}MB")
print(f"  总耗时: {TOTAL:.0f}s ({TOTAL/60:.1f}min)")
print("✅ 美股特征预计算完成！")
