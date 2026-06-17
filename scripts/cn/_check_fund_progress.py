import json
d = json.load(open(r'/home/hermes/.hermes/openclaw-archive/data\us_fundamentals_v7_raw.json','r',encoding='utf-8'))
print(f'缓存大小: {len(d)}')
has_pb = sum(1 for v in d.values() if v and v.get('pb') is not None)
print(f'pb有值: {has_pb}')
print(f'AAPL in cache: {"AAPL" in d}')
if 'AAPL' in d:
    print(f'AAPL: {d["AAPL"]}')
# 看最后修改时间
import os
mtime = os.path.getmtime(r'/home/hermes/.hermes/openclaw-archive/data\us_fundamentals_v7_raw.json')
import datetime
print(f'最后修改: {datetime.datetime.fromtimestamp(mtime)}')
