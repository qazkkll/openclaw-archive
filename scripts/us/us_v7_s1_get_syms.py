import sys
sys.stdout.reconfigure(encoding='utf-8')
import pyarrow.parquet as pq
import pandas as pd

path = r'/home/hermes/.hermes/openclaw-archive/scripts/system\us_ml_feats_v3_dated.parquet'
pf = pq.ParquetFile(path)

# 读sym列看有多少唯一值、时间范围
t = pf.read(columns=['sym','date','price'])
df = t.to_pandas()
syms = df['sym'].unique()
print(f'唯一种类: {len(syms)}')
print(f'总行数: {len(df)}')
print(f'日期范围: {df["date"].min()} ~ {df["date"].max()}')
print(f'sample syms: {", ".join(sorted(syms)[:10])} ... {", ".join(sorted(syms)[-10:])}')

# 先输出sym列表，等会儿用
with open(r'/home/hermes/.hermes/openclaw-archive/scripts/system\v3_syms.txt', 'w') as f:
    for s in sorted(syms):
        f.write(s + '\n')
print(f'\nsym列表已保存到 v3_syms.txt ({len(syms)} 只)')
