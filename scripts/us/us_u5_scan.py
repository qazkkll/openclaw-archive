import futu as ft, json, sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
OUT = r'/home/hermes/.hermes/openclaw-archive/data'
LOG = r'/home/hermes/.hermes/openclaw-archive\logs\bt_opend.log'
os.makedirs(OUT, exist_ok=True)

def log(m):
    t = time.strftime('%H:%M:%S')
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('[%s] %s\n' % (t, m))
    print(m, flush=True)

# Load pool
with open(r'/home/hermes/.hermes/openclaw-archive/data\us_active_pool.json') as f:
    tickers = json.load(f)['tickers']
log('美股活跃池: %d只' % len(tickers))

# ========== STEP 1: U5 Current Scan (V5 + R5) ==========
log('===== U5当前扫描 =====')
results = []
ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)

batch_sz = 100
for i in range(0, len(tickers), batch_sz):
    batch = tickers[i:i+batch_sz]
    for t in batch:
        ret, data, _ = ctx.request_history_kline('US.'+t, start='2026-04-01', end='2026-06-01', ktype=ft.KLType.K_DAY)
        if ret != 0 or len(data) < 35: continue
        c = list(data['close'].astype(float))
        pr = c[-1]; d1 = (c[-1]/c[-2]-1)*100
        m5 = (c[-1]/c[-5]-1)*100 if len(c)>=5 else 0
        m30 = (c[-1]/c[-30]-1)*100 if len(c)>=30 else 0
        gains = sum(max(c[i]-c[i-1],0) for i in range(-14,0))
        losses = sum(max(c[i-1]-c[i],0) for i in range(-14,0))
        rsi = 100-100/(1+gains/losses) if losses>0 else 100
        hp52 = max(c[-252:]) if len(c)>=252 else max(c)
        p52 = pr/hp52*100
        
        v5_ok = m30>0 and m5>0 and rsi<59
        r5_ok = d1<-3 and rsi<35 and m30>3
        score = m30 * (1 - min(max(0,(p52-50)/50)*0.3, 1)) if v5_ok or r5_ok else 0
        
        results.append({
            'ticker': t, 'price': round(pr,2), 'u5_score': round(score,1),
            'v5_pass': v5_ok, 'r5_pass': r5_ok,
            'm30': round(m30,1), 'm5': round(m5,1), 'd1': round(d1,1),
            'rsi': round(rsi,0), 'p52': round(p52,0)
        })
    time.sleep(0.2)  # small delay between batches to avoid throttle
    if (i+batch_sz) % 500 == 0:
        log('  进度: %d/%d' % (i+batch_sz, len(tickers)))

ctx.close()
log('扫描完成: %d只收录' % len(results))

# Sort by score
results.sort(key=lambda x: -x['u5_score'])
v5_only = [r for r in results if r['v5_pass'] and not r['r5_pass']]
r5_only = [r for r in results if r['r5_pass']]

log('U5过关: %d只 (V5:%d | R5:%d)' % (len(v5_only)+len(r5_only), len(v5_only), len(r5_only)))
for r in results[:15]:
    typ = 'V5' if r['v5_pass'] else 'R5' if r['r5_pass'] else ''
    log('  %s %-6s $%.0f score=%.1f 30d%+.1f%% d1%+.1f%% RSI%.0f' % (
        typ, r['ticker'], r['price'], r['u5_score'], r['m30'], r['d1'], r['rsi']))

# Save U5 current results
u5_current = {
    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'pool': len(tickers), 'passed': len(v5_only)+len(r5_only),
    'top_v5': [r for r in results if r['v5_pass']][:10],
    'top_r5': [r for r in results if r['r5_pass']][:5],
    'all_scored': len(results)
}
with open(os.path.join(OUT, 'u5_current.json'), 'w', encoding='utf-8') as f:
    json.dump(u5_current, f, indent=2)
log('U5当前结果已保存')

log('')
log('===== 完成 =====')
log('结果: u5_current.json')
