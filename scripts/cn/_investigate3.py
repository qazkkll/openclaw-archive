"""
彻底搞清楚价格的来源方案
"""
import pandas as pd
import json
import numpy as np

# 特征数据
df = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v75.parquet')
print(f'特征数据: {len(df)}行, {df["sym"].nunique()}只')
print(f'列: {df.columns.tolist()}')
print()

# 关键列检查
print('=== 关键列 ===')
print(f'fwd_5d_ret: 非空{df["fwd_5d_ret"].notna().sum()}/{len(df)}')
print(f'sc: 非空{df["sc"].notna().sum()}/{len(df)}')

# AAP的fwd_5d_ret
aapl = df[df['sym']=='AAP']
print(f'\nAAP的日期和fwd_5d_ret:')
print(aapl[['date','fwd_5d_ret','ma5','ma10','price_position']].tail(10).to_string())

# 检查是否有其他价格源
# volume有但close没有——需要从hist取
print('\n\n=== 从hist获取价格的对齐方案 ===')

with open('/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet','r',encoding='utf-8',errors='replace') as f:
    hist = json.load(f)

# AAP在hist中的长度
aapl_hist = hist['AAP']
print(f'AAP hist K线数: {len(aapl_hist["c"])}')

# 方案A: 用特征数据的date列和hist尾部对齐
# 特征数据每只股票有date, 行数固定=2423（所有股票相同）
# hist长度不同，但特征数据是从股票上市日开始算的
# 
# 检查SANA: feat 1253行, hist 1337行
# 那么特征对应hist最后1253根K线
# 我们需要知道特征第1行的日期 = hist第(1337-1253)=84根K线的日期

sana_feat = df[df['sym']=='SANA']
sana_hist = hist['SANA']
print(f'\n=== SANA验证 ===')
print(f'特征行数: {len(sana_feat)}')
print(f'hist K线数: {len(sana_hist["c"])}')
print(f'尾部对齐偏移: {len(sana_hist["c"]) - len(sana_feat)}')
offset = len(sana_hist['c']) - len(sana_feat)
print(f'特征第1行 => hist第{offset}根')
print(f'特征最后1行 => hist第{len(sana_hist["c"])-1}根（尾部）')
print(f'hist尾部close: {sana_hist["c"][-5:]}')
print(f'特征ma5尾部: {sana_feat["ma5"].values[-5:]}')

# 检查特征第1行的ma5是否能对应hist位置
# 验证：特征第1行日期 = ？ hist第offset根的日期
# 全局交易日序列all_dates
all_dates = sorted(df['date'].unique())
print(f'\n全局交易日: {len(all_dates)}天')
# SANA特征第一行日期
sana_first_date = sana_feat['date'].min()
sana_last_date = sana_feat['date'].max()
print(f'SANA特征日期: {sana_first_date} ~ {sana_last_date}')

# 检查全局日期序列中SANA第一天的位置
sana_first_idx = list(all_dates).index(sana_first_date)
print(f'SANA第一天在全局中的索引: {sana_first_idx}')

# 那么SANA hist第offset根对应的全局日期 = all_dates[sana_first_idx] 才对
print(f'如果SANA特征第1行对应hist第{offset}根')
print(f'验证：特征ma5[0]={sana_feat["ma5"].values[0]:.2f}')
print(f'验证：hist前5根close={sana_hist["c"][:5]}')
# hist前5是5日均价
print(f'验证：hist第{offset}根附近的close = {sana_hist["c"][offset:offset+5]}')
# 如果ma5 = 5日均价，需要5根close的均值
if offset >= 4:
    ma5_calc = np.mean(sana_hist['c'][offset-4:offset+1])
    print(f'验证：hist[{offset-4}:{offset+1}] close均值 = {ma5_calc:.4f}')
    print(f'验证：特征ma5[0] = {sana_feat["ma5"].values[0]:.4f}')
    print(f'两者相差: {abs(ma5_calc - sana_feat["ma5"].values[0]):.6f}')
    print(f'✅ 对齐验证{"成" if abs(ma5_calc - sana_feat["ma5"].values[0]) < 0.01 else "败"}!')
