import os, json

dp = r'/home/hermes/.hermes/openclaw-archive/data'

def check(fp, fn):
    sz = os.path.getsize(fp)/(1024*1024)
    print(f'{fn} ({sz:.0f} MB)')
    with open(fp, 'r', encoding='utf-8') as f:
        chunk = f.read(200000)
    chunk = chunk.strip()
    
    if chunk.startswith('['):
        print('  format: JSON array []')
        return
    
    if not chunk.startswith('{'):
        print(f'  format: unknown, starts with {repr(chunk[:20])}')
        return
    
    try:
        # 找到第一个完整的 } 来解析第一个对象
        end_brace = chunk.find('}') + 1
        partial = chunk[:end_brace]
        # 可能有多层嵌套，再找第二个 } 
        while True:
            try:
                obj = json.loads(partial)
                break
            except json.JSONDecodeError:
                next_brace = chunk.find('}', end_brace)
                if next_brace == -1:
                    print('  could not parse even first key')
                    return
                partial = chunk[:next_brace + 1]
                end_brace = next_brace + 1
        
        keys = list(obj.keys())
        print(f'  format: top-level dict, {len(keys)} top keys (in sample)')
        print(f'  first key: "{keys[0]}"')
        first_val = obj[keys[0]]
        print(f'  value type: {type(first_val).__name__}')
        
        if isinstance(first_val, dict):
            subkeys = list(first_val.keys())[:10]
            print(f'  inner keys: {subkeys}')
            for sk in subkeys[:3]:
                sv = first_val[sk]
                print(f'    {sk}: {type(sv).__name__} = {str(sv)[:60]}')
        elif isinstance(first_val, list):
            print(f'  list len: {len(first_val)}')
            if len(first_val) > 0:
                print(f'  first item type: {type(first_val[0]).__name__}')
                if isinstance(first_val[0], dict):
                    print(f'  first item keys: {list(first_val[0].keys())[:10]}')
                else:
                    print(f'  first item: {str(first_val[0])[:60]}')
    except Exception as e:
        print(f'  error: {e}')

for fn in ['a1_daily.json','us_hist_clean.parquet','us_hist_10y.json','moneyflow_data.parquet','daily_basic_data.parquet','top_list_data.json','a_hist_10y.parquet']:
    fp = os.path.join(dp, fn)
    if os.path.exists(fp):
        check(fp, fn)
    else:
        print(f'{fn}: NOT FOUND')
    print()
