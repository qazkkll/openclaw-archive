#!/usr/bin/env python3
"""下载A股长期历史数据 (akshare)"""
import akshare as ak, json, time, sys

with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)
EXCLUDED = {'地产基建','农业','交通物流'}
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}

codes = sorted([c for c,sec in smap.items() if sec not in EXCLUDED and c not in ETFS])
print(f"📥 下载 {len(codes)} 只 (2s间隔)")

hist = {}; errors = 0; start = time.time()

for i, code in enumerate(codes):
    prefix = 'sz' if code.startswith(('0','3')) else 'sh'
    try:
        df = ak.stock_zh_a_daily(symbol=f'{prefix}{code}', adjust='qfq')
        if df is not None and len(df) >= 200:
            dt = df['date'].dt.strftime('%Y-%m-%d').tolist()
            cl = [round(float(x),2) for x in df['close']]
            hi = [round(float(x),2) for x in df['high']]
            lo = [round(float(x),2) for x in df['low']]
            hist[code] = {'dates':dt,'close':cl,'high':hi,'low':lo}
    except:
        errors += 1
    
    if (i+1) % 20 == 0:
        elapsed = time.time()-start
        rate = (i+1)/elapsed
        eta = (len(codes)-(i+1))/rate/60
        print(f"  {i+1}/{len(codes)} ✅{len(hist)} ❌{errors} {eta:.0f}分剩余")
        sys.stdout.flush()
        # Auto-save every 100
        if (i+1) % 100 == 0:
            with open('/home/admin/.openclaw/workspace/data/backtest_hist_long.json','w') as f:
                json.dump(hist, f)
    
    time.sleep(2)

# Final save
elapsed = (time.time()-start)/60
print(f"\n✅ {len(hist)} ❌{errors} ⏱{elapsed:.0f}分")
earliest = min(d['dates'][0] for d in hist.values())
latest = max(d['dates'][-1] for d in hist.values())
print(f"📅 {earliest} ~ {latest}")
with open('/home/admin/.openclaw/workspace/data/backtest_hist_long.json','w') as f:
    json.dump(hist, f)
print("💾 OK")
