#!/usr/bin/env python3
"""每日持仓信号 + 换仓建议 推送"""
import json, sys, urllib.request, os, urllib.parse, math
from datetime import datetime
from collections import defaultdict

BOT_TOKEN = "7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI"
CHAT_ID = "7908145929"
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def send_tg(msg):
    data = json.dumps({"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data=data, headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("ok")
    except: return False

send_tg("⏳ 持仓信号计算中...")

# ===== 工具 =====
def ema(arr, p):
    k = 2/(p+1); r = [arr[0]]
    for v in arr[1:]: r.append(v*k + r[-1]*(1-k))
    return r

def sma(arr, p):
    if len(arr) < p: return [None]*len(arr)
    return [None]*(p-1) + [sum(arr[i-p+1:i+1])/p for i in range(p-1, len(arr))]

def safe(arr, i):
    return arr[i] if arr and 0 <= i < len(arr) and arr[i] is not None else None

# ===== A股评分 =====
def score_cn(code, c, h, l):
    n = len(c)
    if n < 60: return 0
    
    m5 = sma(c, 5); m20 = sma(c, 20); m60 = sma(c, 60)
    e12 = ema(c, 12); e26 = ema(c, 26)
    ml = [e12[i]-e26[i] for i in range(n)]
    sg = ema(ml, 9); mh = [ml[i]-sg[i] for i in range(n)]
    
    gl, ll = [], []
    for i in range(1, n):
        diff = c[i]-c[i-1]; gl.append(max(diff,0)); ll.append(max(-diff,0))
    rsi = [None]*14
    ag = sum(gl[:14])/14 if len(gl)>=14 else 0; al = sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14, n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i < len(gl): ag = (ag*13+gl[i])/14; al = (al*13+ll[i])/14
    
    # ADX
    adx = [None]*27; tr_h, dp_h, dm_h = [], [], []
    for i in range(1, n):
        tr = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        dp = max(0, h[i]-h[i-1]); dm = max(0, l[i-1]-l[i])
        tr_h.append(tr); dp_h.append(dp); dm_h.append(dm)
        if i < 14: continue
        tr14 = sum(tr_h[-14:]); dp14 = sum(dp_h[-14:]); dm14 = sum(dm_h[-14:]); atr = tr14/14
        if atr == 0: adx.append(0); continue
        dip = dp14/14/atr*100; dim = dm14/14/atr*100
        if dip+dim == 0: adx.append(0); continue
        dx = abs(dip-dim)/(dip+dim)*100
        if i < 27: adx.append(dx); continue
        adx.append((sum(a for a in adx[-13:] if a is not None)+dx)/14)
    while len(adx) < n: adx.append(None)
    
    p52 = [None]*251
    for i in range(251, n):
        lo = min(c[i-250:i+1]); hi = max(c[i-250:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    
    di = n-1
    mhv = safe(mh, di); mhpv = safe(mh, di-1)
    ms = 0
    if mhv and mhpv:
        if mhv > 0 and mhpv <= 0: ms = 20
        elif mhv > 0 and mhv > mhpv: ms = 12
        elif mhv > 0: ms = 6
    if ms <= 0: return 0
    
    p52v = safe(p52, di)
    ws = 0
    if p52v is not None:
        if p52v < 20: ws = 20
        elif p52v < 35: ws = 15
        elif p52v < 50: ws = 10
        elif p52v < 65: ws = 6
        elif p52v < 80: ws = 3
    
    pr = safe(c, di); m5v = safe(m5, di); m20v = safe(m20, di); m60v = safe(m60, di)
    mas = 0
    if pr and m20v and pr > m20v: mas += 7
    if m5v and m20v and m5v > m20v: mas += 7
    if m20v and m60v and m20v > m60v: mas += 6
    
    av = safe(adx, di)
    ads = -5
    if av is not None:
        if av >= 35: ads = 20
        elif av >= 28: ads = 15
        elif av >= 22: ads = 10
        elif av >= 18: ads = 5
    
    rv = safe(rsi, di)
    rs = 0
    if rv is not None:
        if rv < 25: rs = 20
        elif rv < 35: rs = 14
        elif rv < 50: rs = 10
        elif rv < 65: rs = 6
        elif rv < 75: rs = 2
        elif rv >= 75: rs = -5
    
    tr = av is not None and av >= 22
    wl = [25,15,15,25,20] if tr else [10,30,15,10,35]
    ttl = ms*(wl[0]/20) + ws*(wl[1]/20) + mas*(wl[2]/20) + ads*(wl[3]/20) + rs*(wl[4]/20)
    return min(ttl/sum(wl)*100, 100), c[di], m20v

# ===== 获取A股K线 =====
def get_sina_cn(code, days=250):
    prefix = 'sh' if code.startswith(('6','5')) else 'sz'
    url = f'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData?symbol={prefix}{code}&scale=240&datalen={days}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            d = json.loads(res.read())
            if isinstance(d, list) and len(d) >= 20:
                c = [float(x['close']) for x in d]
                h = [float(x['high']) for x in d]
                l = [float(x['low']) for x in d]
                return c, h, l
    except: pass
    return None, None, None

# ===== 美股评分 (V2) =====
def score_us(code, c, h, l):
    n = len(c)
    if n < 60: return 0
    
    m20 = sma(c, 20); m50 = sma(c, 50)
    e12 = ema(c, 12); e26 = ema(c, 26)
    ml = [e12[i]-e26[i] for i in range(n)]
    sg = ema(ml, 9); mh = [ml[i]-sg[i] for i in range(n)]
    
    gl, ll = [], []
    for i in range(1, n):
        diff = c[i]-c[i-1]; gl.append(max(diff,0)); ll.append(max(-diff,0))
    rsi = [None]*14
    ag = sum(gl[:14])/14 if len(gl)>=14 else 0; al = sum(ll[:14])/14 if len(ll)>=14 else 0
    for i in range(14, n):
        rsi.append(100-100/(1+ag/al) if al>0 else 100)
        if i < len(gl): ag = (ag*13+gl[i])/14; al = (al*13+ll[i])/14
    
    di = n-1
    mhv = safe(mh, di); mhpv = safe(mh, di-1)
    ms = 0
    if mhv and mhpv:
        if mhv > 0 and mhpv <= 0: ms = 25
        elif mhv > 0 and mhv > mhpv: ms = 15
        elif mhv > 0: ms = 8
    if mhv is None or mhv <= 0: return 0
    
    av = None  # simplified
    ads = 20 if (c[di] > safe(m20, di) if safe(m20, di) else False) else -5
    
    pr = c[di]; m20v = safe(m20, di)
    mas = 0
    if pr and m20v and pr > m20v: mas += 7
    if safe(c, di) and safe(m50, di) and safe(c, di) > safe(m50, di): mas += 7
    if m20v and safe(m50, di) and m20v > safe(m50, di): mas += 6
    
    rv = safe(rsi, di)
    rs = 0
    if rv is not None:
        if rv < 25: rs = 18
        elif rv < 35: rs = 14
        elif rv < 50: rs = 10
        elif rv < 65: rs = 6
        elif rv < 75: rs = 2
        else: rs = -5
    
    p52v = None
    if n >= 251:
        lo = min(c[-250:]); hi = max(c[-250:])
        p52v = (pr-lo)/(hi-lo)*100 if hi > lo else 50
    ws = 0
    if p52v is not None:
        if p52v < 20: ws = 15
        elif p52v < 35: ws = 12
        elif p52v < 50: ws = 8
        elif p52v < 65: ws = 5
        elif p52v < 80: ws = 2
    
    total = (ms*(15/25) + ads*(20/22) + mas*(15/20) + rs*(20/18) + ws*(30/15)) / 100 * 100
    return min(total, 95), pr, m20v

def get_yahoo_us(sym):
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=1y&interval=1d'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            d = json.loads(res.read())
            q = d['chart']['result'][0]
            cls = q['indicators']['quote'][0]['close']
            hi = q['indicators']['quote'][0]['high']
            lo = q['indicators']['quote'][0]['low']
            li = len(cls)-1
            while li >= 0 and cls[li] is None: li -= 1
            if li < 60: return None, None, None
            c = [x for x in cls[:li+1] if x is not None]
            h = [x for x in hi[:li+1] if x is not None]
            loo = [x for x in lo[:li+1] if x is not None]
            return c, h, loo
    except: pass
    return None, None, None

# ===== 主逻辑 =====
try:
    pf = json.load(open(f"{WORKSPACE}/data/portfolio.json"))
except:
    send_tg("❌ 无法读取持仓数据"); sys.exit(1)

report = f"📊 持仓信号日报 | {datetime.now().strftime('%m/%d %H:%M')}\n"
report += "━━━━━━━━━━━━━━━━━━━━\n\n── A股 ──\n"

cn_items = []
for s in pf.get("a_stock", []):
    c, h, l = get_sina_cn(s["code"])
    if c is None:
        report += f"❌ {s['name']} 数据获取失败\n"
        continue
    
    result = score_cn(s["code"], c, h, l)
    if result == 0:
        report += f"❌ {s['name']} MACD阻断 | 评分0\n"
        continue
    
    sc, price, m20 = result
    pnl = (price - s["cost"]) / s["cost"] * 100
    val = price * s["shares"]
    
    above_m20 = price > m20 if m20 else False
    if sc >= 62 and above_m20: lt, act = "🟢", "可加仓"
    elif sc >= 62 and not above_m20: lt, act = "🔵", "可关注(等站回MA20)"
    elif sc >= 48 and above_m20: lt, act = "🟡", "持有观望"
    elif sc >= 48 and not above_m20: lt, act = "🟠", "警惕(跌破MA20)"
    else: lt, act = "🔴", "建议卖出"
    
    cn_items.append((lt, s["name"], s["code"], sc, price, pnl, s["shares"], val, act))
    report += f"{lt} {s['name']}({s['code']}) {sc}分 | ¥{price:.2f} | 盈亏{pnl:+.1f}%\n   {act} | {s['shares']}股\n"

report += "\n── 美股 ──\n"
for s in pf.get("us_stock", []):
    c, h, l = get_yahoo_us(s["code"])
    if c is None:
        report += f"❌ {s['name']} 数据获取失败\n"
        continue
    
    result = score_us(s["code"], c, h, l)
    if result == 0:
        report += f"❌ {s['name']} MACD阻断\n"
        continue
    
    sc, price, m20 = result
    pnl = (price - s["cost"]) / s["cost"] * 100
    
    above_m20 = price > m20 if m20 else False
    if sc >= 60 and above_m20: lt, act = "🟢", "可加仓"
    elif sc >= 50 and above_m20: lt, act = "🔵", "可关注"
    elif sc >= 30 and above_m20: lt, act = "🟡", "持有观望"
    elif sc >= 30 and not above_m20: lt, act = "🟠", "警惕(跌破MA20)"
    else: lt, act = "🔴", "建议卖出"
    
    report += f"{lt} {s['name']}({s['code']}) {sc}分 | ${price:.2f} | 盈亏{pnl:+.1f}%\n   {act} | {s['shares']}股\n"

# 警报
warnings = []
for s in pf.get("a_stock", []):
    c, _, _ = get_sina_cn(s["code"])
    if c:
        pr = c[-1]
        pnl = (pr - s["cost"]) / s["cost"] * 100
        if pnl <= -6:
            warnings.append(f"⚠️ {s['name']} 浮亏{pnl:.1f}% (止损{s['cost']*0.92:.2f})")
for s in pf.get("us_stock", []):
    c, _, _ = get_yahoo_us(s["code"])
    if c:
        pr = c[-1]
        pnl = (pr - s["cost"]) / s["cost"] * 100
        if pnl <= -6:
            warnings.append(f"⚠️ {s['name']} 浮亏{pnl:.1f}% (止损{s['cost']*0.92:.2f})")

if warnings:
    report += "\n━━━━━━━━━━━━━━━━━━━━\n"
    for w in warnings:
        report += w + "\n"

report += "\n━━━━━━━━━━━━━━━━━━━━\n🍤 小钳轮动 v2.5"

sent = send_tg(report)
print(f"{'✅ 已推送' if sent else '❌ 推送失败'}")
