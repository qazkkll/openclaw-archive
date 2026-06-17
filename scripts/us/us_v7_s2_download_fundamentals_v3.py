"""
V7: 下载基本面数据 - 续传版
读取已有缓存，只补全缺失的sym
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import yfinance as yf
import json
import time
import requests

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

with open(r'/home/hermes/.hermes/openclaw-archive/scripts/system\v3_syms.txt') as f:
    syms = [l.strip() for l in f if l.strip()]

cache_path = r'/home/hermes/.hermes/openclaw-archive/data\us_fundamentals_v7_raw.json'

# 读已有缓存
try:
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)
except:
    cache = {}

existing = len(cache)
missing = [s for s in syms if s not in cache]
print(f'已有: {existing}  缺少: {len(missing)}  总共: {len(syms)}')

FIELDS = {
    'pb': 'priceToBook',
    'roe': 'returnOnEquity',
    'rev_growth': 'revenueGrowth',
    'profit_growth': 'earningsGrowth',
    'debt_equity': 'debtToEquity',
    'gross_margin': 'grossMargins',
    'profit_margin': 'profitMargins',
}

errors = []
t0 = time.time()

for idx, sym in enumerate(missing):
    try:
        time.sleep(0.8)
        tk = yf.Ticker(sym, session=session)
        info = tk.info or {}
        result = {}
        for field, yf_key in FIELDS.items():
            v = info.get(yf_key)
            if v is not None and v != '':
                result[field] = round(v, 6) if isinstance(v, (float, int)) else v
            else:
                result[field] = None
        cache[sym] = result
    except Exception as e:
        err = str(e)
        if 'Rate limited' in err or '401' in err or 'Too Many' in err:
            print(f'  限速 @{sym}，等30秒...')
            time.sleep(30)
            try:
                tk = yf.Ticker(sym, session=session)
                info = tk.info or {}
                result = {}
                for field, yf_key in FIELDS.items():
                    v = info.get(yf_key)
                    if v is not None and v != '':
                        result[field] = round(v, 6) if isinstance(v, (float, int)) else v
                    else:
                        result[field] = None
                cache[sym] = result
                continue
            except:
                pass
        cache[sym] = {k: None for k in FIELDS}
        errors.append((sym, str(e)[:60]))
    
    if (idx + 1) % 25 == 0:
        total_has = sum(1 for v in cache.values() if v and any(v.get(f) is not None for f in FIELDS))
        total_full = sum(1 for v in cache.values() if v and all(v.get(f) is not None for f in FIELDS))
        elapsed = time.time() - t0
        eta = (elapsed / (idx+1) * (len(missing) - idx - 1)) / 60
        print(f'  [{idx+1}/{len(missing)}] 缓存{len(cache)} 有值{total_has} 全字段{total_full} 用时{elapsed:.0f}s ETA{eta:.1f}min')
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)

with open(cache_path, 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False)

t = time.time() - t0
total_has = sum(1 for v in cache.values() if v and any(v.get(f) is not None for f in FIELDS))
total_full = sum(1 for v in cache.values() if v and all(v.get(f) is not None for f in FIELDS))

print(f'\n完成！总耗时 {t:.0f}s')
print(f'总缓存: {len(cache)}  有值: {total_has}  全字段: {total_full}  失败: {len(errors)}')
for f in FIELDS:
    cnt = sum(1 for v in cache.values() if v and v.get(f) is not None)
    print(f'  {f:15s}: {cnt:5d} ({cnt/len(cache)*100:.1f}%)')
