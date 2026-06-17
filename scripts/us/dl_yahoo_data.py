#!/usr/bin/env python3
"""下载A股2015-2026数据 via Yahoo Finance"""
import yfinance as yf, json, time, sys, warnings
warnings.filterwarnings('ignore')

with open('/home/admin/.openclaw/workspace/data/sector_map.json') as f: smap = json.load(f)
EXCLUDED = {'地产基建','农业','交通物流'}
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}

codes = sorted([c for c,sec in smap.items() if sec not in EXCLUDED and c not in ETFS])
print(f"📥 下载 {len(codes)} 只 (Yahoo Finance)")

hist = {}; errors = 0; start = time.time()

for i, code in enumerate(codes):
    suffix = '.SZ' if code.startswith(('0','3')) else '.SS'
    try:
        df = yf.download(code+suffix, start='2015-01-01', end='2026-05-15', progress=False, auto_adjust=True)
        if df is not None and len(df) > 500:
            df = df.dropna()
            dates = df.index.strftime('%Y-%m-%d').tolist()
            closes = [round(float(x),2) for x in df['Close'].values]
            highs = [round(float(x),2) for x in df['High'].values]
            lows = [round(float(x),2) for x in df['Low'].values]
            hist[code] = {'dates':dates,'close':closes,'high':highs,'low':lows}
    except:
        errors += 1
    
    if (i+1) % 50 == 0:
        elapsed = time.time()-start
        rate = (i+1)/elapsed
        eta = (len(codes)-(i+1))/rate/60
        pct = (i+1)/len(codes)*100
        print(f"  {pct:.0f}% | ✅{len(hist)} ❌{errors} | ⏱{elapsed/60:.1f}分 | ETA:{eta:.0f}分", flush=True)
        # Auto-save
        with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json','w') as f:
            json.dump(hist, f)

total = (time.time()-start)/60
print(f"\n✅ {len(hist)}只成功 ❌{errors}失败 ⏱{total:.0f}分")
if hist:
    dates_all = sorted(set(d for h in hist.values() for d in h['dates']))
    print(f"📅 {dates_all[0]} ~ {dates_all[-1]} ({len(dates_all)}天)")

with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json','w') as f:
    json.dump(hist, f)
print("💾 已保存 backtest_hist_yahoo.json")
