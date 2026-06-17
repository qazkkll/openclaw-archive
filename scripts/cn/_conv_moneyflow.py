# 转换 moneyflow_data.parquet -> parquet
# 结构: {"CODE": [{"ts_code":"...", "trade_date":"...", ...},...], "CODE2": [...]}
import os, json, shutil, time, gc, re

try:
    import pyarrow as pa, pyarrow.parquet as pq, pandas as pd
except:
    print('Need pyarrow'); raise

dp = r'/home/hermes/.hermes/openclaw-archive/data'
raw = os.path.join(dp, 'raw_json')
os.makedirs(raw, exist_ok=True)

fn = 'moneyflow_data.parquet'
fp_in = os.path.join(dp, fn)
fp_out = os.path.join(dp, 'moneyflow_data.parquet')

# backup
bak = os.path.join(raw, fn)
if not os.path.exists(bak):
    print('Backing up...')
    shutil.copy2(fp_in, bak)

t0 = time.time()
sz_mb = os.path.getsize(fp_in) / (1024*1024)
print('{}: {:.0f} MB'.format(fn, sz_mb))

with open(fp_in, 'r', encoding='utf-8') as f:
    content = f.read()

total_len = len(content)
total = 0
writer = None
batch = []
last_pct = -1
i = 0

# 模式: "CODE": [...]
# 扫描找 "CODE": [
# 然后提取到匹配的 ] 结束
# 注意value是数组，不是对象

print('Parsing...')

while i < total_len:
    # 找 "CODE":
    quote_pos = content.find('"', i)
    if quote_pos == -1 or quote_pos + 50 > total_len:
        break
    
    # 检查后面是不是 ": ["
    colon_pos = content.find('":', quote_pos)
    if colon_pos == -1 or colon_pos > quote_pos + 20:
        i = quote_pos + 1
        continue
    
    # 提取code
    code = content[quote_pos:colon_pos].strip('"')
    
    # 找 [ 
    bracket_pos = content.find('[', colon_pos)
    if bracket_pos == -1:
        i = colon_pos + 1
        continue
    
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
        print('  Unbalanced JSON at code {} around pos {}'.format(code, i))
        break
    
    block = content[bracket_pos:j+1]
    
    try:
        records = json.loads(block)
        for rec in records:
            if isinstance(rec, dict):
                batch.append(rec)
                total += 1
    except Exception as e:
        pass
    
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
