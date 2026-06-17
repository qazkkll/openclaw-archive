"""Debug data format for A-share training"""
import json
with open('/home/hermes/.hermes/openclaw-project/data/a_hist_10y.json', 'r', encoding='utf-8') as f:
    hist = json.load(f)
codes = [c for c in hist if c.startswith(('60','00')) and len(hist[c].get('dates',[])) >= 750]
print(f'Loaded {len(hist)} stocks, filtered {len(codes)}')
for c in codes[:5]:
    h = hist[c]
    c_vals = h.get('c', [])
    v_vals = h.get('v', [])
    h_vals = h.get('h', [])
    l_vals = h.get('l', [])
    print(f'{c}: keys={list(h.keys())} len_c={len(c_vals)} len_v={len(v_vals)} len_h={len(h_vals)} len_l={len(l_vals)} type_c={type(c_vals)}')
    # Check if lists contain strings
    if c_vals:
        print(f'  c[0] type={type(c_vals[0])}, val={c_vals[0]}')
        
# Check for stocks with missing/incomplete fields
missing = []
for c in codes:
    h = hist[c]
    for k in ['c','v','h','l']:
        if k not in h or len(h[k]) < 200:
            missing.append((c, k, len(h.get(k, []))))
if missing:
    print(f'\nStocks with issues: {len(missing)}')
    for c, k, l in missing[:10]:
        print(f'  {c}: {k} len={l}')
else:
    print('\nAll stocks have complete fields')
