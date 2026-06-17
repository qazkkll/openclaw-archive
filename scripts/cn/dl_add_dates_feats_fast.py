"""特征文件加日期 — 从us_hist_10y.json的连续交易日反推
使用yfinance批量拉取少量样本找日期跨度
"""
import sys, os, json, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═══ 智能补日期 ═══")

# 只拉一只大盘股找到日期序列
print("拉取SPY获取交易日历...")
spy = yf.Ticker("SPY")
spy_hist = spy.history(period='3y')
spy_dates = spy_hist.index.strftime('%Y-%m-%d').tolist()
print(f"SPY日期数: {len(spy_dates)}, 范围: {spy_dates[0]} ~ {spy_dates[-1]}")

# 读取特征文件
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
syms = sorted(df['sym'].unique())
sym_counts = df['sym'].value_counts()
max_rows = sym_counts.max()
print(f"最大行数: {max_rows}")

# SPY的交易日历只有约750天, 但特征每只股票有493行, 取最近的493天
# 日期对齐: 每只股票的日期序列 = SPY日期序列的近端
date_seq = spy_dates[-min(max_rows, len(spy_dates)):]

# 每只股票的特征行数不同, 取对应的日期尾巴
print("应用日期...")
df['date'] = '1900-01-01'
for sym in syms:
    n_rows = sym_counts[sym]
    if n_rows <= len(date_seq):
        dates = date_seq[-n_rows:]
    else:
        dates = ['1900-01-01'] * (n_rows - len(date_seq)) + date_seq
    df.loc[df['sym'] == sym, 'date'] = dates

# 过滤无效
before = len(df)
df = df[df['date'] != '1900-01-01'].copy()
print(f"过滤后: {len(df):,}/{before:,}")

df.to_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet", index=False)
print(f"保存: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v3_dated.parquet ({time.time()-T0:.0f}s)")
