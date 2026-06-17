#!/usr/bin/env python3
"""V4 美股 · 优化版回测（一次下载，分区间切片）"""

import yfinance as yf, warnings, numpy as np
warnings.filterwarnings('ignore')

# 候选股
ALL_TICKERS = ['NVDA','AMD','MU','INTC','AVGO','QCOM','AMAT','MSFT','CRM','ADBE',
    'ORCL','NOW','GOOGL','AMZN','META','NFLX','HD','COST','WMT','JPM',
    'V','MA','UNH','LLY','PFE','MRK','CAT','GE','TSLA','AAPL']

SECTOR = {}
for t in ['NVDA','AMD','MU','INTC','AVGO','QCOM','AMAT']: SECTOR[t] = '半导体'
for t in ['MSFT','CRM','ADBE','ORCL','NOW']: SECTOR[t] = '软件'
for t in ['GOOGL','AMZN','META','NFLX']: SECTOR[t] = '互联网'
for t in ['HD','COST','WMT']: SECTOR[t] = '消费'
for t in ['JPM','V','MA']: SECTOR[t] = '金融'
for t in ['UNH','LLY','PFE','MRK']: SECTOR[t] = '医疗'
for t in ['CAT','GE']: SECTOR[t] = '工业'
SECTOR['TSLA'] = '汽车'
SECTOR['AAPL'] = '消费电子'

def gv(series, d):
    try:
        v = series.asof(d)
        if hasattr(v, 'iloc'): v = v.iloc[0]
        if hasattr(v, 'item'): v = v.item()
        return float(v) if v == v else None
    except: return None

print("=" * 70)
print("V4 美股方案对比")
print("A: 纯动量(前5) | B: 过滤版(剔除52周>95%+行业分散) | C: SPY")
print("=" * 70)

# 一次性下载全周期数据
print("\n下载数据...")
spy = yf.download('SPY', start="2009-01-01", end="2026-05-18", progress=False)
spy_s = spy['Close'].squeeze()

prices = {}
for i, t in enumerate(ALL_TICKERS):
    try:
        h = yf.download(t, start="2009-01-01", end="2026-05-18", progress=False)
        if h is not None and len(h) > 200:
            prices[t] = h['Close'].squeeze()
    except: pass
    if (i+1) % 10 == 0: print(f"  {i+1}/{len(ALL_TICKERS)}")

print(f"  加载 {len(prices)} 只股票\n")

BULLS = [
    ("2009-2020", "2009-03-09", "2020-02-19"),
    ("2016-2018", "2016-07-01", "2018-09-30"),
    ("2020-2022", "2020-04-01", "2022-01-03"),
    ("2023-2026", "2023-01-01", "2026-05-18"),
]

for pname, start, end in BULLS:
    sd = spy_s.loc[start:end]
    if len(sd) < 40: continue
    dates = sd.index.tolist()
    
    sa_rets = []; sb_rets = []
    
    for i in range(0, len(dates)-40, 20):
        d_p = dates[i]; d_b = dates[i+20]; d_s = dates[min(i+40, len(dates)-1)]
        
        mom = []
        for t, p in prices.items():
            # 确保这个ticker在这个区间有数据
            vp = gv(p, d_p); vb = gv(p, d_b)
            if vp and vb and vp > 0.01 and vb > 0.01:
                pct = (vb / vp - 1) * 100
                hp52 = float(p.loc[d_p:d_b].max()) if len(p.loc[d_p:d_b]) > 0 else vb
                p52 = (vb / hp52) * 100 if hp52 > 0 else 100
                mom.append((t, pct, p52))
        
        if len(mom) < 10: continue
        mom.sort(key=lambda x: x[1], reverse=True)
        
        # A: 纯动量前5
        ta = mom[:5]
        ra = [gv(prices[t], d_s)/gv(prices[t], d_b)-1 for t,_,_ in ta if gv(prices[t], d_b) and gv(prices[t], d_b) > 0]
        ra = [r*100 for r in ra if r is not None]
        if ra: sa_rets.append(np.mean(ra))
        
        # B: 过滤动量
        fb = [x for x in mom if x[2] < 95]
        sc = {}; tb = []
        for t, m, p52 in fb:
            sec = SECTOR.get(t, '其他')
            if sc.get(sec, 0) < 2:
                tb.append((t, m)); sc[sec] = sc.get(sec, 0) + 1
            if len(tb) >= 5: break
        if len(tb) < 5:
            for t, m, _ in mom:
                if len(tb) >= 5: break
                if (t, m) not in tb:
                    sec = SECTOR.get(t, '其他')
                    if sc.get(sec, 0) < 2:
                        tb.append((t, m)); sc[sec] = sc.get(sec, 0) + 1
        
        rb = [gv(prices[t], d_s)/gv(prices[t], d_b)-1 for t,_ in tb[:5] if gv(prices[t], d_b) and gv(prices[t], d_b) > 0]
        rb = [r*100 for r in rb if r is not None]
        if rb: sb_rets.append(np.mean(rb))
    
    spy_r = float((sd.iloc[-1] / sd.iloc[0] - 1) * 100)
    
    print(f"\n📈 {pname}")
    if sa_rets: print(f"  A纯动量: +{sum(sa_rets):.1f}%  (均{np.mean(sa_rets):+.2f}%/期)")
    if sb_rets: print(f"  B过滤版: +{sum(sb_rets):.1f}%  (均{np.mean(sb_rets):+.2f}%/期)")
    print(f"  C-SPY:   +{spy_r:.1f}%")

print("\n✅ 完成")
