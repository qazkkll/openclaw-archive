"""
绿箭v16 — 加做空/基本面特征的预计算
在现有us_ml_feats_v2.parquet基础上，按sym对齐基本面数据
"""
import sys, json, os, time, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═══ 绿箭v16: 加基本面特征 ═══")

# 读现有特征
df = pd.read_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet")
print(f"现有特征: {len(df):,}行, {df['sym'].nunique()}只")

# 基本面数据
with open(_paths.ML_DIR + "/us_fundamentals.json", 'r') as f:
    fundamentals = json.load(f)
print(f"基本面: {len(fundamentals)}只")

# 创建sym→基本面映射的DataFrame
rows = []
for sym, d in fundamentals.items():
    rows.append({
        'sym': sym,
        'short_ratio': d.get('shortRatio'),
        'short_pct': d.get('shortPct'),
        'sector': d.get('sector'),
        'industry': d.get('industry'),
        'market_cap': d.get('marketCap'),
        'pe_trailing': d.get('trailingPE'),
        'pe_forward': d.get('forwardPE'),
        'div_yield': d.get('dividendYield'),
        'beta': d.get('beta'),
        'fund_price': d.get('price'),
    })
fd = pd.DataFrame(rows)
print(f"基本面df: {len(fd)}行")

# 合并
df2 = df.merge(fd, on='sym', how='left')
print(f"合并后: {len(df2):,}行")
print(f"列: {df2.columns.tolist()}")

# 统计基本面字段覆盖率
for c in ['short_ratio','sector','market_cap','pe_trailing','beta']:
    cov = df2[c].notna().mean() * 100
    print(f"  {c}: {cov:.0f}%")

# 清理异常值
for col in ['pe_trailing','pe_forward','div_yield']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
# 替换无穷大
import numpy as np
df2 = df2.replace([np.inf, -np.inf], np.nan)

# 保存
df2.to_parquet(_paths.ML_DIR + "/us_ml_feats_v2.parquet", index=False)
TOTAL = time.time() - T0
print(f"\n✅ 绿箭v16预计算完成 ({TOTAL:.0f}s)")
