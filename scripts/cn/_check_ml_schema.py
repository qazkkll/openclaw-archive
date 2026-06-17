import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

fund_kw = ['mkt','market','pe','pb','roe','sector','industry','profit','revenue',
           'beta','asset','debt','equity','margin','income','growth','cap','size',
           'div','yield','vol','volume']

for fname in ['us_ml_feats_v3_dated.parquet', 'us_ml_feats_v5.parquet', 'us_ml_feats_v4.parquet']:
    path = rf'/home/hermes/.hermes/openclaw-archive/scripts/system\{fname}'
    if not Path(path).exists():
        print(f'{fname}: 不存在')
        continue
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        fund_cols = [c for c in cols if any(k in c.lower() for k in fund_kw)]
        print(f'{fname}:')
        print(f'  总列: {len(cols)}, 行: {pf.metadata.num_rows}')
        print(f'  基本面列: {fund_cols if fund_cols else "无"}')
        print(f'  前10列: {cols[:10]}')
        print()
    except Exception as e:
        print(f'{fname}: 报错 {e}')
