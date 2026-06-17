import json
d = json.load(open('/home/hermes/.hermes/openclaw-project/data/a_ml_feats_cache.json'))
print(f'X: {len(d["X"])} rows, y: {len(d["y"])} rows')
print(f'pos rate: {sum(d["y"])/len(d["y"]):.2%}')
print(f'dim: {len(d["X"][0])} features')
