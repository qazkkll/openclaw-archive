#!/usr/bin/env python3
"""R5独立扫描器 - 全市场实时抄底信号"""
import yfinance as yf, json, warnings, numpy as np, time, os, pandas as pd
warnings.filterwarnings('ignore')

CACHE = '/home/admin/.openclaw/workspace/data/r5_cache'
os.makedirs(CACHE, exist_ok=True)

# R5参数
CONFIG = {
    'd1_min': -3,     # 单日跌>3%
    'rsi_max': 35,    # RSI<35超卖
    'vr_min': 1.3,    # 量>1.3倍均值
    'm30_min': 3,     # 30天趋势>3%
    'price_min': 5,   # 股价>$5
    'scan_batch': 100 # 每批扫描100只
}

def calc_rsi(c):
    n = len(c)
    if n < 15: return 50
    gains = sum(max(c[j]-c[j-1],0) for j in range(n-14,n))
    losses = sum(max(c[j-1]-c[j],0) for j in range(n-14,n))
    return 100-100/(1+gains/losses) if losses > 0 else 100 if gains > 0 else 50

def scan_stock(t):
    """单只R5扫描，返回信号或None"""
    try:
        h = yf.Ticker(t).history(period='3mo', interval='1d')
        if len(h) < 30: return None
        c = list(h['Close'])
        v = list(h['Volume'])
        pr = c[-1]
        if pr < CONFIG['price_min']: return None
        
        m30 = (pr/c[-30]-1)*100
        d1 = (c[-1]/c[-2]-1)*100
        if d1 >= CONFIG['d1_min']: return None
        if m30 < CONFIG['m30_min']: return None
        
        rsi = calc_rsi(c)
        if rsi >= CONFIG['rsi_max']: return None
        
        vol5 = np.mean(v[-5:])
        vol20 = np.mean(v[-20:]) if len(v) >= 20 else vol5
        vr = vol5/vol20 if vol20 > 0 else 0
        if vr < CONFIG['vr_min']: return None
        
        return {
            'ticker': t, 'price': round(pr,2),
            'drop_pct': round(d1,1), 'rsi': round(rsi),
            'vol_ratio': round(vr,1), 'trend_30d': round(m30,1)
        }
    except: return None

# 获取全市场股票
print('获取股票列表...', flush=True)
import urllib.request
tickers = set()
for url in ['https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt',
            'https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt']:
    try:
        data = urllib.request.urlopen(url, timeout=10).read().decode().split('\n')
        for line in data[:5000]:
            if '|' in line:
                t = line.split('|')[0].strip()
                if t and not t.startswith('$') and len(t) <= 5:
                    tickers.add(t)
    except: pass
tickers = sorted(tickers)
print('总股票: %d' % len(tickers), flush=True)

# 分批扫描
signals = []
t0 = time.time()
for i in range(0, min(len(tickers), 1000), CONFIG['scan_batch']):
    batch = tickers[i:i+CONFIG['scan_batch']]
    # 批量下载价格
    try:
        df = yf.download(' '.join(batch), period='3mo', interval='1d', progress=False, auto_adjust=True)
        for t in batch:
            try:
                if not isinstance(df.columns, pd.MultiIndex): continue
                c = df['Close'][t].dropna().values
                if len(c) < 30: continue
                v = df['Volume'][t].dropna().values if 'Volume' in df.columns.levels[0] else np.array([])
                pr = c[-1]
                if pr < CONFIG['price_min']: continue
                m30 = (pr/c[-30]-1)*100 if c[-30] > 0 else 0
                d1 = (c[-1]/c[-2]-1)*100
                if d1 >= CONFIG['d1_min']: continue
                if m30 < CONFIG['m30_min']: continue
                rsi = calc_rsi(c)
                if rsi >= CONFIG['rsi_max']: continue
                vol5 = np.mean(v[-5:]) if len(v) >= 5 else 0
                vol20 = np.mean(v[-20:]) if len(v) >= 20 else vol5
                vr = vol5/vol20 if vol20 > 0 else 0
                if vr < CONFIG['vr_min']: continue
                signals.append({
                    'ticker': t, 'price': round(pr,2),
                    'drop_pct': round(d1,1), 'rsi': round(rsi),
                    'vol_ratio': round(vr,1), 'trend_30d': round(m30,1)
                })
            except: pass
    except: pass
    
    if (i//CONFIG['scan_batch']) % 3 == 0:
        print('  %d/%d (%.0fs, %d signals)' % (i, min(len(tickers),3000), time.time()-t0, len(signals)), flush=True)

signals.sort(key=lambda x: x['drop_pct'])  # 最暴跌的排前面

# 保存
result = {
    'time': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_scanned': min(len(tickers),3000),
    'signals_found': len(signals),
    'signals': signals[:20]
}
json.dump(result, open('/home/admin/.openclaw/workspace/data/r5_signals.json','w'))

print('\n=== R5信号 %d只 ===' % len(signals))
for s in signals[:15]:
    print('  %-6s $%7.2f 今日%+.1f%% RSI=%d 量%.1f倍 30天%+.1f%%'%(
        s['ticker'],s['price'],s['drop_pct'],s['rsi'],s['vol_ratio'],s['trend_30d']))

if not signals:
    print('  今日无R5信号')

print('\n完成 (%.0fs)' % (time.time()-t0))
