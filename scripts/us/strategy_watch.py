#!/usr/bin/env python3
"""策略状态监控 - 每天收盘后跑
检测长期趋势变化，主动提醒修改策略"""

import json, urllib.request, os
from collections import defaultdict

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOT_TOKEN = "7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI"
CHAT_ID = "7908145929"

def send_tg(msg):
    data = json.dumps({"chat_id": CHAT_ID, "text": msg}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data=data, headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        return resp.get("ok")
    except: return False

# ===== 检测1: 排除行业动量 =====
def check_excluded_sectors():
    """检测地产基建等行业动量是否在上升"""
    try:
        with open(f"{WORKSPACE}/data/backtest_hist_v3_extended.json") as f:
            hist = json.load(f)
    except: return None
    
    with open(f"{WORKSPACE}/data/sector_map.json") as f: smap = json.load(f)
    
    # 最近交易日
    dates = sorted(set(d for c in hist for d in hist[c].get('dates',[]) if '2026' in d))
    if not dates: return None
    latest = dates[-1]
    
    # 计算各行业20日动量
    sectors = defaultdict(list)
    for c in hist:
        sec = smap.get(c, '其他')
        sectors[sec].append(c)
    
    mom = {}
    for sec, codes in sectors.items():
        rets = []
        for c in codes[:20]:
            d = hist[c]
            try:
                di = d['dates'].index(latest)
            except:
                for x in range(len(d['dates'])-1, -1, -1):
                    if d['dates'][x] <= latest:
                        di = x
                        break
                else: continue
            if di < 20: continue
            p1 = d['close'][di]; p2 = d['close'][di-20]
            if p2 and p2 > 0: rets.append((p1-p2)/p2*100)
        if len(rets) >= 2:
            mom[sec] = sum(rets)/len(rets)
    
    if not mom: return None
    
    ranked = sorted(mom.items(), key=lambda x: -x[1])
    
    # 找排除行业的排名
    alerts = []
    excluded = ['地产基建', '农业', '交通物流']
    for i, (sec, val) in enumerate(ranked):
        if sec in excluded:
            rank = i + 1
            if rank <= 4:
                alerts.append(f"⚠️ 策略提醒: {sec} 进入行业动量的前{rank}名! ({val:+.1f}%)")
                alerts.append(f"   当前策略排除了该行业，建议审视是否继续排除")
    
    return alerts

# ===== 检测2: 美股趋势切换 =====
def check_us_regime():
    """检测SPY相对MA200的趋势变化"""
    import yfinance as yf
    try:
        spy = yf.download('SPY', start='2025-01-01', end='2026-05-16', progress=False, auto_adjust=True)
        if spy is None or len(spy) < 200: return None
        
        prices = spy['Close'].values
        current = prices[-1]
        ma200 = sum(prices[-200:]) / 200
        ma50 = sum(prices[-50:]) / 50
        
        alerts = []
        if current < ma200:
            alerts.append(f"⚠️ 美股趋势: SPY跌破MA200 (${current:.0f}<${ma200:.0f})")
            alerts.append(f"   建议关注: 是否切换风险模式")
        elif current > ma200 * 1.05:
            pass  # 正常牛市，不提
        elif ma50 < ma200:
            alerts.append(f"🟡 美股趋势: MA50在MA200下方 (短期均线走弱)")
        
        return alerts
    except:
        return None

# ===== 检测3: 策略缓存 =====
STATE_FILE = f"{WORKSPACE}/data/strategy_watch_state.json"
def load_state():
    try: return json.load(open(STATE_FILE))
    except: return {'consecutive_weeks': {}}
def save_state(s):
    json.dump(s, open(STATE_FILE, 'w'))

# ===== 主逻辑 =====
alerts = []

# 检测排除行业
sector_alerts = check_excluded_sectors()
if sector_alerts:
    alerts.extend(sector_alerts)

# 检测美股趋势
us_alerts = check_us_regime()
if us_alerts:
    alerts.extend(us_alerts)

# 推送
if alerts:
    msg = "📋 策略状态提醒\n" + "━" * 20 + "\n" + "\n".join(alerts)
    send_tg(msg)
    print(f"✅ 已推送: {len(alerts)}条提醒")
else:
    print("✅ 无异常，不推送")
