"""给特征文件补日期 — 从yfinance拉最近3年日K"""
import sys, json, os, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import yfinance as yf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═══ 特征文件补日期 ═══")

# 读取所有sym
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
syms = sorted(df['sym'].unique())
print(f"共 {len(syms)} 只股票需要补日期")

# 每只股票对应在特征文件的行数
sym_counts = df['sym'].value_counts()
print(f"每只股票平均 {sym_counts.mean():.0f} 行")

# 分批次从yfinance拉数据
BATCH_SIZE = 50
date_map = {}  # sym -> list of dates (与特征行对应)
failed = []

for batch_start in range(0, len(syms), BATCH_SIZE):
    batch = syms[batch_start:batch_start+BATCH_SIZE]
    print(f"  拉取 {batch_start+1}-{batch_start+len(batch)}/{len(syms)} ...", flush=True)
    
    for sym in batch:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period='3y')
            if len(hist) == 0:
                failed.append(sym)
                continue
            
            # 保存日期列表
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            # 特征文件按时间顺序有 N 行, 取最近的 N 个日期
            n_rows = sym_counts[sym]
            if len(dates) >= n_rows:
                date_map[sym] = dates[-n_rows:]
            else:
                # 填充前面的NaN日期
                padded = ['1900-01-01'] * (n_rows - len(dates)) + dates
                date_map[sym] = padded
        except Exception as e:
            failed.append(sym)
    
print(f"\n拉取完成！成功率: {len(date_map)}/{len(syms)}")
print(f"失败: {len(failed)} 只")

# 应用到特征文件
print("\n应用日期到特征文件...")
df['date'] = '1900-01-01'
for sym, dates in date_map.items():
    mask = df['sym'] == sym
    n_avail = min(len(dates), mask.sum())
    df.loc[mask, 'date'] = dates[:n_avail] + ['1900-01-01'] * (mask.sum() - n_avail)

# 过滤掉无日期的行
before = len(df)
df = df[df['date'] != '1900-01-01'].copy()
print(f"过滤后: {len(df):,} / {before:,} 行保留")

# 保存
df.to_parquet(_paths.ML_DIR + "/us_ml_feats_v3_dated.parquet", index=False)
print(f"\n保存到 /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v3_dated.parquet")
print(f"用时: {time.time()-T0:.0f}s")
