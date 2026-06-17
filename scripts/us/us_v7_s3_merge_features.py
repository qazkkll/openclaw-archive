"""
V7 Step 3: 合并基本面特征到v3_dated
输入: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v3_dated.parquet
       /home/hermes/.hermes/openclaw-project/data/us_fundamentals_v7_raw.json
输出: /home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v7_full.parquet (37列)

注意: 基本面数据是静态的（当前值），v3_dated是日频时间序列
合并策略: 每只sym的每日行都补上该sym的当前基本面值
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import json
from pathlib import Path

v3_path = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v3_dated.parquet'
fund_path = r'/home/hermes/.hermes/openclaw-archive/data\us_fundamentals_v7_raw.json'
out_path = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v7_full.parquet'

# 1. 读基本面数据
print('加载基本面数据...')
with open(fund_path, 'r', encoding='utf-8') as f:
    fund_raw = json.load(f)

# 转成DataFrame
fund_df = pd.DataFrame.from_dict(fund_raw, orient='index')
fund_df.index.name = 'sym'
fund_df = fund_df.reset_index()
print(f'  基本面: {len(fund_df)} 只, 列: {list(fund_df.columns)}')

# 统计覆盖率
for col in ['pb','roe','rev_growth','profit_growth','debt_equity','gross_margin','profit_margin']:
    if col in fund_df.columns:
        valid = fund_df[col].notna().sum()
        print(f'  {col:15s}: {valid}/{len(fund_df)} ({valid/len(fund_df)*100:.1f}%)')

# 2. 读v3_dated
print('加载v3_dated特征集...')
v3 = pq.ParquetFile(v3_path)
print(f'  行数: {v3.metadata.num_rows}')
print(f'  列数: {len(v3.schema_arrow.names)}')
print(f'  列名: {v3.schema_arrow.names[:5]} ...')

# 分批读+合并（避免内存爆）
batch_size = 50000
writer = None
total_merged = 0

print('开始合并特征...')
for batch_idx, batch in enumerate(v3.iter_batches(batch_size=batch_size)):
    df_batch = batch.to_pandas()
    
    # merge基本面（左连接，sym列必须存在）
    if 'sym' in df_batch.columns:
        df_merged = df_batch.merge(fund_df, on='sym', how='left')
    else:
        # 默认列名可能是ts_code
        df_batch.rename(columns={'ts_code': 'sym'}, inplace=True)
        df_merged = df_batch.merge(fund_df, on='sym', how='left')
    
    # 写parquet（分批append）
    table = pa.Table.from_pandas(df_merged, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    total_merged += len(df_merged)
    
    if (batch_idx + 1) % 5 == 0:
        print(f'  合并批次 {batch_idx+1}: {total_merged} 行')

if writer:
    writer.close()

# 检查输出
out_pf = pq.ParquetFile(out_path)
print(f'\n完成！')
print(f'输出: {out_path}')
print(f'  行数: {out_pf.metadata.num_rows}')
print(f'  列数: {len(out_pf.schema_arrow.names)}')
cols_set = set(out_pf.schema_arrow.names)
new_cols = [c for c in fund_df.columns if c in cols_set and c != 'sym']
print(f'  新增基本面列: {new_cols}')
print(f'  总列名: {out_pf.schema_arrow.names[:10]} ...')

# 列存储到索引
print(f'\n✅ V7 特征集生成完成')
print(f'路径: {out_path}')
