# 检查所有JSON的结构、行数、是否可以全量加载
import json, time, os

dp = r'/home/hermes/.hermes/openclaw-archive/data'
targets = [
    ('a1_daily.json', 'nested'),
    ('us_hist_clean.parquet', 'nested'),
    ('us_hist_10y.json', 'nested'),
    ('moneyflow_data.parquet', 'jsonl'),
    ('daily_basic_data.parquet', 'jsonl'),
    ('a_hist_10y.parquet', 'nested'),
]

for fn, fmt_hint in targets:
    fp = dp + '/' + fn
    if not os.path.exists(fp):
        print('{}: NOT FOUND'.format(fn))
        continue
    
    sz_mb = os.path.getsize(fp) / (1024*1024)
    print('\n=== {} ({:.0f} MB) ==='.format(fn, sz_mb))
    
    # detect format from first bytes
    with open(fp, 'rb') as f:
        head = f.read(200)
    txt = head.decode('utf-8', errors='replace').strip()
    
    if txt.startswith('['):
        print('  Format: JSON array')
    elif txt.startswith('{'):
        # check if second char is newline (pretty-printed nested dict)
        if txt[1] == '\n':
            print('  Format: pretty-printed nested dict')
        else:
            # could be compressed nested dict or JSONL
            print('  Format: compressed nested dict (need streaming)')
    else:
        print('  Format: unknown')
        continue
    
    # test loading if < 500MB
    if sz_mb < 500:
        print('  Loading full file...')
        t0 = time.time()
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
            t1 = time.time()
            print('  {} top keys, {:.1f}s'.format(len(data), t1-t0))
            k0 = list(data.keys())[0]
            v0 = data[k0]
            print('  key[0] = "{}", type={}'.format(k0[:20], type(v0).__name__))
            if isinstance(v0, dict):
                subkeys = list(v0.keys())[:5]
                print('  subkeys: {}'.format(subkeys))
                if v0:
                    subv = v0[subkeys[0]]
                    if isinstance(subv, dict):
                        print('  fields: {}'.format(list(subv.keys())))
                    elif isinstance(subv, list):
                        print('  list len: {}'.format(len(subv)))
            elif isinstance(v0, list):
                print('  list len: {}'.format(len(v0)))
                if v0:
                    print('  first item: {}'.format(type(v0[0]).__name__))
        except Exception as e:
            print('  FAIL: {}'.format(e))
    else:
        print('  >500MB, skip full load')
        
        # peek at structure by reading first 10KB
        with open(fp, 'r', encoding='utf-8') as f:
            chunk = f.read(10000)
        
        # find first date key
        import re
        dates = re.findall(r'"(\d{8})"', chunk)
        if dates:
            print('  Sample dates: {}'.format(dates[:3]))
        
        # guess value structure
        if '"code"' in chunk or '"ts_code"' in chunk:
            print('  Contains ticker/code fields')
        if '"net_mf"' in chunk:
            print('  Contains moneyflow fields (moneyflow_data)')
