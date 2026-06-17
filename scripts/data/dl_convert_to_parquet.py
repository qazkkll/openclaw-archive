# JSON->Parquet转换 v2 - 纯流式，不用ijson
# 对a1_daily/moneyflow/daily_basic 这类 {date:{code:{fields}}} 结构
# 手动按日期切分解析
import os, json, shutil, time, gc, re

try:
    import pyarrow as pa, pyarrow.parquet as pq, pandas as pd
except ImportError:
    print('Need pyarrow/pandas'); raise

DP = r'/home/hermes/.hermes/openclaw-archive/data'
RAW_DIR = os.path.join(DP, 'raw_json')
os.makedirs(RAW_DIR, exist_ok=True)

def batch_write(rows, fp_out, writer=None):
    df = pd.DataFrame(rows)
    tbl = pa.Table.from_pandas(df)
    if writer is None:
        writer = pq.ParquetWriter(fp_out, tbl.schema)
    writer.write_table(tbl)
    return writer

def conv_ticker_dict(fp_in, fp_out):
    """{ticker: {c:[],h:[],...}}"""
    t0 = time.time()
    with open(fp_in, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('  Loaded {} tickers'.format(len(data)))
    batch = []
    for tk, v in data.items():
        row = {'ticker': tk}
        for fld in ['c','h','l','o','v','dates']:
            if fld in v:
                row[fld] = v[fld]
        batch.append(row)
    w = batch_write(batch, fp_out)
    w.close()
    print('  Done: {} rows, {:.1f}s'.format(len(batch), time.time()-t0))

def conv_date_list(fp_in, fp_out):
    """{date: [{...}]}"""
    t0 = time.time()
    with open(fp_in, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('  Loaded {} dates'.format(len(data)))
    batch = []
    for d, recs in data.items():
        for r in recs:
            if isinstance(r, dict):
                r['date'] = str(d)
                batch.append(r)
    print('  Flattened {} rows'.format(len(batch)))
    w = batch_write(batch, fp_out)
    w.close()
    print('  Done: {:.1f}s'.format(time.time()-t0))

def conv_big_date_code(fp_in, fp_out, desc=''):
    """{date: {code: {fields}}} - 流式处理大文件"""
    t0 = time.time()
    sz_mb = os.path.getsize(fp_in) / (1024*1024)
    print('  Size: {:.0f} MB, streaming...'.format(sz_mb))
    
    batch = []
    writer = None
    total = 0
    
    with open(fp_in, 'r', encoding='utf-8') as f:
        # 逐字符扫描，找日期key后截取JSON块
        content = f.read()
    
    total_len = len(content)
    print('  Read into memory, parsing...')
    
    # 找模式: "YYYYMMDD": {
    i = 0
    last_pct = -1
    
    while i < total_len:
        # 找 "8位数字":
        m = re.search(r'"(\d{8})":\s*\{', content[i:])
        if not m:
            break
        
        date_str = m.group(1)
        start = i + m.start(0)
        # 找到匹配的 } 
        depth = 0
        j = start
        in_str = False
        escape = False
        
        while j < total_len:
            ch = content[j]
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == '{': depth += 1
                elif ch == '}': depth -= 1
            if depth == 0:
                break
            j += 1
        
        if depth != 0:
            print('  WARN: unbalanced JSON at pos {}'.format(i))
            break
        
        block = content[start:j+1]
        
        # 解析
        try:
            obj = json.loads(block)
            # obj = {"YYYYMMDD": {code: {fields}}}
            for code, fields in list(obj.values())[0].items():
                if isinstance(fields, dict):
                    fields['date'] = date_str
                    fields['code'] = str(code)
                    batch.append(fields)
                    total += 1
        except Exception as e:
            print('  Parse error at date {}: {}'.format(date_str, e))
        
        if len(batch) >= 50000:
            writer = batch_write(batch, fp_out, writer)
            batch = []
        
        # 进度
        pct = int(100 * i / total_len)
        if pct != last_pct and pct % 5 == 0:
            print('  {}% ({} rows)'.format(pct, total), flush=True)
            last_pct = pct
        
        i = j + 1
    
    if batch:
        writer = batch_write(batch, fp_out, writer)
    if writer:
        writer.close()
    
    elapsed = time.time() - t0
    speed = sz_mb / elapsed if elapsed > 0 else 0
    out_sz = os.path.getsize(fp_out) / (1024*1024) if os.path.exists(fp_out) else 0
    print('  DONE: {} rows, {:.1f}s ({:.0f} MB/s)'.format(total, elapsed, speed))
    print('  Output: {:.0f} MB'.format(out_sz))

# 只跑还未转换的文件
def main():
    # 先检查哪些已有parquet
    targets = [
        ('us_hist_clean.parquet', conv_ticker_dict, 'US 5Y'),
        ('us_hist_10y.json', conv_ticker_dict, 'US 10Y'),
        ('a_hist_10y.parquet', conv_ticker_dict, 'A 10Y'),
        ('top_list_data.json', conv_date_list, 'Dragon/Tiger'),
        # 大文件流式处理
        ('a1_daily.json', conv_big_date_code, 'A1 daily'),
        ('moneyflow_data.parquet', conv_big_date_code, 'Moneyflow'),
        ('daily_basic_data.parquet', conv_big_date_code, 'Daily basic'),
    ]
    
    all_t0 = time.time()
    
    for fname, conv, desc in targets:
        fp_in = os.path.join(DP, fname)
        if not os.path.exists(fp_in):
            print('\n[{}] NOT FOUND'.format(fname))
            continue
        
        base = os.path.splitext(fname)[0]
        fp_out = os.path.join(DP, base + '.parquet')
        
        if os.path.exists(fp_out):
            print('\n[{}] parquet exists, skip'.format(fname))
            continue
        
        # 备份原始JSON
        bak = os.path.join(RAW_DIR, fname)
        if not os.path.exists(bak):
            print('\n[{}] backing up to raw_json/'.format(fname))
            shutil.copy2(fp_in, bak)
        else:
            print('\n[{}] backup exists'.format(fname))
        
        print('=== {} ({}) ==='.format(fname, desc))
        try:
            conv(fp_in, fp_out)
        except Exception as e:
            print('  FAIL: {}'.format(e))
            import traceback
            traceback.print_exc()
        
        gc.collect()
        gc.collect()
    
    print('\nTOTAL: {:.0f}s'.format(time.time() - all_t0))

if __name__ == '__main__':
    main()
