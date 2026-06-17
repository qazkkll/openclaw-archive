#!/usr/bin/env python3
"""
yfinance 批量下载美股历史数据 — 一站到底版
一次运行，补全NASDQ+NYSE所有正常股票
"""
import yfinance as yf, json, os, time, re, pandas as pd
import warnings
warnings.filterwarnings('ignore')

CACHE = r'/home/hermes/.hermes/openclaw-archive/data\cache'
LOG_FILE = r'/home/hermes/.hermes/openclaw-archive\logs\yf_download.log'
os.makedirs(CACHE, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(m):
    t = time.strftime('%H:%M:%S')
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write('[%s] %s\n' % (t, m))
    print('[%s] %s' % (t, m), flush=True)

log('='*60)
log('yfinance批量下载 v2 — 一站到底')
log('='*60)

# 1. 清单
log('[1] 加载清单...')
pool = json.load(open(r'/home/hermes/.hermes/openclaw-archive/data\us_active_pool.json'))
cached = set(f.replace('.json','') for f in os.listdir(CACHE) if f.endswith('.json'))
all_tickers = [t for t in pool['tickers'] if t not in cached]
normal = [t for t in all_tickers if re.match(r'^[A-Z]{1,5}$', t) or re.match(r'^[A-Z]{1,5}[\.\-][A-Z]{1,2}$', t)]
log('  已有缓存: %d' % len(cached))
log('  待下载: %d' % len(normal))

# 2. 下载
log('[2] 批量下载（每批50只, 单次拉满12年）...')
t_start = time.time()
downloaded = 0
failed = 0
cheap = 0
batch_size = 50

for i in range(0, len(normal), batch_size):
    batch = normal[i:i+batch_size]
    
    try:
        # 一次下载12年数据
        hist = yf.download(batch, start='2014-01-01', end='2026-06-01', auto_adjust=True, threads=True, progress=False)
        
        if hist.empty or 'Close' not in hist.columns:
            failed += len(batch)
        else:
            closes = hist['Close']
            opens = hist['Open']
            highs = hist['High']
            lows = hist['Low']
            volumes = hist['Volume']
            
            for t in batch:
                try:
                    if t not in closes.columns:
                        failed += 1
                        continue
                    vals = closes[t].dropna()
                    if len(vals) < 252:
                        failed += 1
                        continue
                    if float(vals.iloc[-1]) < 5:
                        cheap += 1
                        continue
                    
                    rows = []
                    for idx in hist.index:
                        c = float(closes[t].get(idx, 0))
                        if c <= 0: continue
                        rows.append({
                            'date': idx.strftime('%Y-%m-%d'),
                            'open': float(opens[t].get(idx, c)),
                            'high': float(highs[t].get(idx, c)),
                            'low': float(lows[t].get(idx, c)),
                            'close': c,
                            'volume': int(float(volumes[t].get(idx, 0) or 0)),
                        })
                    
                    avg_p = sum(r['close'] for r in rows) / len(rows)
                    if avg_p < 5:
                        cheap += 1
                        continue
                    
                    json.dump({'ticker': t, 'source': 'yfinance',
                               'updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                               'data': rows},
                              open(os.path.join(CACHE, t+'.json'), 'w', encoding='utf-8'),
                              ensure_ascii=False)
                    downloaded += 1
                    
                except:
                    failed += 1
    
    except Exception as e:
        failed += len(batch)
    
    if (i // batch_size) % 10 == 0:
        elapsed = time.time() - t_start
        total_cache = len([f for f in os.listdir(CACHE) if f.endswith('.json')])
        log('  %d/%d: +%d 缓存=%d 失败=%d 仙股=%d [%.0fs]' % (
            i+batch_size, len(normal), downloaded, total_cache,
            failed, cheap, elapsed))

# 完成
elapsed = time.time() - t_start
total_cache = len([f for f in os.listdir(CACHE) if f.endswith('.json')])
log('')
log('='*60)
log('完成! %.0fs' % elapsed)
log('  新下载: %d' % downloaded)
log('  缓存总计: %d' % total_cache)
log('  失败/无效: %d' % failed)
log('  仙股过滤: %d' % cheap)
log('='*60)
