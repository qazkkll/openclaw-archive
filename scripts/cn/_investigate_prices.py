"""
调查特征数据和历史K线的日期对应关系
"""
import pandas as pd
import json
import numpy as np

# 加载特征数据
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
print(f'特征数据: {len(df)}行, {df["sym"].nunique()}只股票')

# 看几只小盘股的日期范围
print('\n股票日期范围:')
for sym in ['AACG', 'AAL', 'AAOI', 'AAON', 'AAME']:
    sdf = df[df['sym']==sym]
    print(f'  {sym}: {len(sdf)}行, {sdf["date"].min()} ~ {sdf["date"].max()}')

# 加载hist
print('\n加载历史K线...')
with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet','r',encoding='utf-8',errors='replace') as f:
    hist = json.load(f)

# 看这几只股票在hist中的长度
print('\n股票在hist中的长度:')
for sym in ['AACG', 'AAL', 'AAOI', 'AAON', 'AAME']:
    if sym in hist:
        h = hist[sym]
        print(f'  {sym}: K线数={len(h["c"])}, 首close={h["c"][0]:.2f}, 尾close={h["c"][-1]:.2f}')
    else:
        print(f'  {sym}: 不在hist中')

# 关键——用特证数据的日期与hist对齐
# 假设：每个股票在特征数据中的日期间隔为1个交易日
# hist的长度应该和特征数据行数匹配（或接近）
print('\n日期对齐检查（股票间一致性）:')
# 选一只常见小盘股
common_syms = [s for s in ['AAOI','AAON','AAL','AAME'] if s in hist]
for sym in common_syms:
    sdf = df[df['sym']==sym]
    n_feat = len(sdf)
    n_hist = len(hist[sym]['c'])
    diff = n_feat - n_hist
    print(f'  {sym}: feat={n_feat}行, hist={n_hist}K线, 差异={diff}')
    
    # 特征数据第一行日期对应hist第一根K线
    feat_start = str(sdf['date'].min())[:10]
    feat_end = str(sdf['date'].max())[:10]
    
    # 如果hist短于feat：取feat中最后n_hist天的日期
    if n_hist <= n_feat:
        aligned_dates = sdf['date'].iloc[-n_hist:].values
        print(f'      feat范围: {feat_start} ~ {feat_end}')
        print(f'      hist对齐到feat最后{n_hist}天')
        print(f'      对齐后日期: {str(aligned_dates[0])[:10]} ~ {str(aligned_dates[-1])[:10]}')
    else:
        print(f'      hist比feat长, 无法简单对齐')

# 最好的办法：从特征数据提取所有日期的排序序列
all_dates = sorted(df['date'].unique())
print(f'\n全局交易日序列: {len(all_dates)}天, {str(all_dates[0])[:10]} ~ {str(all_dates[-1])[:10]}')

# 日期索引映射: 用全局交易日序列做统一对齐
date_to_idx = {str(d)[:10]: i for i, d in enumerate(all_dates)}
print(f'日期索引映射已创建, 覆盖{len(date_to_idx)}个交易日')
