"""
更精细地调查日期对应关系
"""
import pandas as pd
import json
import numpy as np

# 特征数据
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
print(f'特征数据: {len(df)}行, {df["sym"].nunique()}只股票')

# 找一个存在于特征数据的股票
syms_in_feat = df['sym'].unique()[:5]
print(f'前5只特征股票: {syms_in_feat.tolist()}')

for sym in syms_in_feat[:3]:
    sdf = df[df['sym']==sym]
    print(f'\n{sym}: {len(sdf)}行, 日期={sdf["date"].min()} ~ {sdf["date"].max()}')
    # 看看MA指标是否正确（MA正常说明价格数据存在）
    print(f'  ma5前5: {sdf["ma5"].values[:5]}')

# 加载hist
print('\n加载历史K线...')
with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet','r',encoding='utf-8',errors='replace') as f:
    hist = json.load(f)

# 看同一些股票
for sym in syms_in_feat[:3]:
    if sym in hist:
        h = hist[sym]
        print(f'\n{sym} in hist: {len(h["c"])}根K线')
        print(f'  close前5: {h["c"][:5]}')
        print(f'  high前5: {h["h"][:5]}')
        print(f'  low前5: {h["l"][:5]}')
    else:
        print(f'\n{sym}: 不在hist中！')

# 检查特征数据是否有直接的close价格
# 用ma5判断: 5日均价 = ma5, 说明特征数据是从价格算的
# 但没存close本身
print('\n\n检查特征数据的价格构造...')
print(f'列: {df.columns.tolist()[:20]}')
print()

# 特征数据里股票的占比
print(f'总股票数: {df["sym"].nunique()}')
print(f'在hist存在的股票数: {len(set(df["sym"].unique()) & set(hist.keys()))}')
print(f'在hist不存在的股票数: {len(set(df["sym"].unique()) - set(hist.keys()))}')

# 选一只两者都存在的
common = list(set(df['sym'].unique()) & set(hist.keys()))
print(f'交集: {len(common)}只')
if common:
    sym = common[0]
    sdf = df[df['sym']==sym]
    h = hist[sym]
    print(f'\n=== 第一只交集股票: {sym} ===')
    print(f'特征行数: {len(sdf)}, hist K线数: {len(h["c"])}')
    print(f'特征日期: {sdf["date"].min()} ~ {sdf["date"].max()}')
    
    # 检查特征数据有几年的记录
    feat_years = (sdf['date'].max() - sdf['date'].min()).days / 365
    print(f'特征年份跨度: {feat_years:.1f}年')
    
    # 看ma5能否反推close
    # 如果特征行数 < hist K线数: 特征只包含部分历史
    if len(sdf) <= len(h['c']):
        print(f'特征缩略版: 用hist最后{len(sdf)}根对齐')
        print(f'hist尾部close: {h["c"][-5:]}')
        print(f'最后日期对应的hist close: {h["c"][-1]}')
    else:
        print(f'特征比hist还长...奇怪')
