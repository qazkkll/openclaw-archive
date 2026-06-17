"""修复v2.1 -- 从旧parquet读，保留sym列"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

print("═══ 修复v2.1: 保留sym列 ═══")

old = pd.read_parquet(_paths.US_ML_FEATS)
sym = old['sym'].values

# 读刚生成的v2文件，但没有sym
new = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")

# 因为groupby后顺序变了，不能简单拼接
# 重新算，一次性保留sym
df = old.copy()
feats = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
         'ret1','ret5','ret20','ret60',
         'macd','macd_signal','macd_hist','vol_ratio','ma_bias20']

results = []
print(f"处理 {df['sym'].nunique()} 只股票...")
for sym, g in df.groupby('sym'):
    g = g.sort_index().reset_index(drop=True)
    n = len(g)
    close = g['price'].values
    
    # 5天label
    fwd5 = np.full(n, np.nan)
    fwd5[:n-5] = (close[5:] - close[:n-5]) / close[:n-5] * 100
    g['label_5d_pct'] = fwd5
    
    cls = np.full(n, -1, dtype=int)
    cls[fwd5 < -5] = 0
    cls[(fwd5 >= -5) & (fwd5 < -2)] = 1
    cls[(fwd5 >= -2) & (fwd5 <= 2)] = 2
    cls[(fwd5 > 2) & (fwd5 <= 5)] = 3
    cls[fwd5 > 5] = 4
    g['label_5d_5class'] = cls
    
    # vol5
    ret1 = g['ret1'].values
    vol5 = np.full(n, np.nan)
    for i in range(4, n):
        vol5[i] = np.nanstd(ret1[i-4:i+1]) * 100
    g['vol5'] = vol5
    
    # trend_accel
    g['trend_accel'] = g['ret20'].fillna(0) - g['ret5'].fillna(0)
    
    results.append(g)

df2 = pd.concat(results, ignore_index=True)
df2 = df2.dropna(subset=['label_5d_pct'])
print(f"  总行: {len(df2):,}")
print(f"  列: {df2.columns.tolist()}")

for cl in range(5):
    cnt = (df2['label_5d_5class'] == cl).sum()
    print(f"    {cl}: {cnt:,}行 ({cnt/len(df2)*100:.1f}%)")

df2.to_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet", index=False)
print(f"✅ 保存完成")
