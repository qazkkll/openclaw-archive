# 检查每个JSON的深层结构，写转换脚本用
import os, json, sys

dp = r'/home/hermes/.hermes/openclaw-archive/data'

def peek_json(fp, max_depth=2):
    """读一个小的样本来判断结构"""
    sz = os.path.getsize(fp)
    with open(fp, 'r', encoding='utf-8') as f:
        # 读前50KB
        chunk = f.read(50000)
    # 找到JSON开头
    chunk = chunk.strip()
    
    if chunk.startswith('['):
        # JSON array
        try:
            small = json.loads(chunk + ']')  # 补全
        except:
            small = json.loads(chunk[:chunk.rindex('}')+1] + ']')
        if isinstance(small, list) and len(small) > 0:
            item = small[0]
            print(f'  type: JSON array [{len(small)} items in sample]')
            print(f'  sample item keys: {list(item.keys())[:10]}')
            for k, v in list(item.items())[:3]:
                print(f'    {k}: {type(v).__name__} = {str(v)[:60]}')
            return 'array', list(item.keys())
    elif chunk.startswith('{'):
        # 可能是 dict 或 JSONL
        try:
            small = json.loads(chunk[:chunk.rindex('}')+1] + '}')
        except:
            print(f'  cannot parse as single dict, trying JSONL...')
            return 'jsonl', None
        
        if isinstance(small, dict):
            keys = list(small.keys())[:5]
            print(f'  type: top-level dict [{len(small)} keys in sample]')
            print(f'  sample keys: {keys}')
            first_key = keys[0]
            first_val = small[first_key]
            print(f'  key="{first_key}" value type: {type(first_val).__name__}')
            if isinstance(first_val, dict):
                subkeys = list(first_val.keys())[:5]
                print(f'    value keys: {subkeys}')
                sk = subkeys[0]
                sv = first_val[sk]
                print(f'    {sk}: {type(sv).__name__} = {str(sv)[:60]}')
                if isinstance(sv, dict):
                    print(f'    inner keys: {list(sv.keys())[:10]}')
            elif isinstance(first_val, list):
                if len(first_val) > 0:
                    print(f'    first item: {type(first_val[0]).__name__} = {str(first_val[0])[:60]}')
            return 'nested_dict', None
        else:
            print(f'  unexpected: {type(small).__name__}')
            return 'unknown', None
    else:
        # 可能是JSONL
        lines = chunk.split('\n')
        for line in lines:
            line = line.strip()
            if line and line.startswith('{'):
                try:
                    obj = json.loads(line)
                    print(f'  type: JSONL - sample has {len(obj)} keys')
                    print(f'  keys: {list(obj.keys())[:10]}')
                    for k, v in list(obj.items())[:3]:
                        print(f'    {k}: {type(v).__name__} = {str(v)[:60]}')
                    return 'jsonl', list(obj.keys())
                except:
                    continue
        print(f'  type: unknown first-char={chunk[0:10]}')
        return 'unknown', None

targets = [
    'a1_daily.json',
    'us_hist_clean.parquet',
    'us_hist_10y.json',
    'moneyflow_data.parquet',
    'daily_basic_data.parquet',
    'top_list_data.json',
    'a_hist_10y.parquet',
]

for fn in targets:
    fp = os.path.join(dp, fn)
    if not os.path.exists(fp):
        print(f'{fn}: NOT FOUND')
        continue
    sz = os.path.getsize(fp)
    print(f'\n{'='*50}')
    print(f'{fn} ({sz/(1024*1024):.0f} MB)')
    peek_json(fp)
