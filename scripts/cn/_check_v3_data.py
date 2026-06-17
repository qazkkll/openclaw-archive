import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

path = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v3_dated.parquet'
pf = pq.ParquetFile(path)

# 用read().to_pandas()只读前几行
cols = ['sym','price','market_cap','pe_trailing','pe_forward','sector','industry','beta','div_yield']
table = pf.read_row_group(0, columns=cols)
df = table.to_pandas()
print('=== 第一行样例 ===')
print(df.head(3).to_string())

print('\n=== 全量空值统计 ===')
# 也只看第一行组
valid_cnt = df.count()
total = len(df)
for c in cols:
    print(f'  {c:20s} -> 非空 {valid_cnt[c]}/{total} = {valid_cnt[c]/total*100:.1f}%')

print(f'\n总行数(第一组): {total}')
