import json, time
dp = r'/home/hermes/.hermes/openclaw-archive/data'
t0 = time.time()
with open(dp + '/top_list_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
t1 = time.time()
print('top_list: {} top keys, {:.1f}s'.format(len(data), t1-t0))
for k, v in list(data.items())[:2]:
    print('  key={}: list of {} items'.format(k, len(v)))
    if v:
        print('  keys: {}'.format(list(v[0].keys())))
