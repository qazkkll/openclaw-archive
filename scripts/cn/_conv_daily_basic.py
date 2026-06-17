# 转换 daily_basic_data.parquet -> parquet
# 结构: {"YYYYMMDD": [{记录...}], ...}
# 用批量读JSON + pandas 分批转换
import os, json, shutil, time, gc

try:
    import pyarrow as pa, pyarrow.parquet as pq, pandas as pd
except:
    print('Need pyarrow'); raise

dp = r'/home/hermes/.hermes/openclaw-archive/data'
raw = os.path.join(dp, 'raw_json')
os.makedirs(raw, exist_ok=True)

fn = 'daily_basic_data.parquet'
fp_in = os.path.join(dp, fn)
fp_out = os.path.join(dp, 'daily_basic_data.parquet')

bak = os.path.join(raw, fn)
if not os.path.exists(bak):
    print('Backing up...')
    shutil.copy2(fp_in, bak)

t0 = time.time()
sz_mb = os.path.getsize(fp_in) / (1024*1024)
print('{}: {:.0f} MB'.format(fn, sz_mb))

# 因为 structure 是 {date: [records]} 
# 用流式：读全部文本，按日期切分
with open(fp_in, 'r', encoding='utf-8') as f:
    content = f.read()

total_len = len(content)
total = 0
writer = None
batch = []
last_pct = -1
i = 0

print('Parsing...')

while i < total_len:
    # 找 "YYYYMMDD": [
    m = __import__('re').search(r'"(\d{8})":\s*\[', content[i:])
    if not m:
        break
    
    date_str = m.group(1)
    bracket_pos = i + m.start(0) + len(m.group(0)) - 1  # position of [
    
    # 找到匹配的 ]
    depth = 0
    in_str = False
    escape = False
    j = bracket_pos
    
    while j < total_len:
        ch = content[j]
        if escape:
            escape = False
        elif ch == '\\':
            escape = True
        elif ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch == '[': depth += 1
            elif ch == ']': depth -= 1
        if depth == 0:
            break
        j += 1
    
    if depth != 0:
        print('  Unbalanced at date {} pos {}'.format(date_str, i))
        break
    
    block = content[bracket_pos:j+1]
    
    try:
        records = json.loads(block)
        for rec in records:
            if isinstance(rec, dict):
                batch.append(rec)
                total += 1
    except Exception as e:
        print('  Parse error at {}: {}'.format(date_str, e))
    
    if len(batch) >= 50000:
        df = pd.DataFrame(batch)
        tbl = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(fp_out, tbl.schema)
        writer.write_table(tbl)
        batch = []
        print('  {} rows done'.format(total), flush=True)
    
    pct = int(100 * j / total_len)
    if pct != last_pct and pct % 5 == 0:
        print('  {}% ({} rows)'.format(pct, total), flush=True)
        last_pct = pct
    
    i = j + 1

if batch:
    df = pd.DataFrame(batch)
    tbl = pa.Table.from_pandas(df)
    if writer is None:
        writer = pq.ParquetWriter(fp_out, tbl.schema)
    writer.write_table(tbl)
if writer:
    writer.close()

elapsed = time.time() - t0
out_sz = os.path.getsize(fp_out) / (1024*1024) if os.path.exists(fp_out) else 0
print('\nDONE: {} rows, {:.1f}s'.format(total, elapsed))
print('Output: {:.0f} MB'.format(out_sz))
