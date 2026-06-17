#!/usr/bin/env python3
"""
🔥 晨扫+开盘判断（A股/美股通用）

用法:
  python3 scripts/macro_collector.py --scan A --pre      # A股开盘前
  python3 scripts/macro_collector.py --scan A --open     # A股开盘后10分钟
  python3 scripts/macro_collector.py --scan US --pre     # 美股开盘前
  python3 scripts/macro_collector.py --scan US --open    # 美股开盘后10分钟
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TUSHARE_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
TUSHARE_URL = 'http://api.tushare.pro'

def fetch_north_money():
    """拉北向资金"""
    import requests
    try:
        p = {'api_name': 'moneyflow_hsgt', 'token': TUSHARE_TOKEN,
             'params': {'end_date': '20260529', 'limit': 5},
             'fields': 'trade_date,north_money'}
        r = requests.post(TUSHARE_URL, json=p, timeout=10)
        return r.json()['data']['items']
    except:
        return []

def fetch_top_list():
    """拉龙虎榜"""
    import requests
    try:
        p = {'api_name': 'top_list', 'token': TUSHARE_TOKEN,
             'params': {'trade_date': '20260529'},
             'fields': 'ts_code,name,pct_change,net_amount,amount'}
        r = requests.post(TUSHARE_URL, json=p, timeout=10)
        items = r.json()['data']['items']
        buys = [i for i in items if i[3] and float(i[3]) > 0]
        buys.sort(key=lambda x: float(x[3]), reverse=True)
        return buys[:5]
    except:
        return []

def scan_a_pre():
    """A股开盘前：输出数据供基金经理判断"""
    from data_source import AShareKline
    kl = AShareKline()
    out = []
    
    # 1. 大盘
    sh = kl.get_kline('000001', 120)
    if sh and len(sh) >= 60:
        close = [d['close'] for d in sh]
        last, prev = close[-1], close[-2]
        ma20 = sum(close[-20:]) / 20
        out.append(f"上证: {last:.0f} ({(last/prev-1)*100:+.2f}%)")
        out.append(f"MA20: {ma20:.0f} | {'多头' if last>ma20 else '空头'}")
    
    # 2. 北向
    north = fetch_north_money()
    if north:
        for item in north[-3:]:
            d, n = item[0], float(item[1])/10000
            out.append(f"北向 {d[-5:]}: +{n:.0f}亿")
    
    # 3. 龙虎榜
    top = fetch_top_list()
    if top:
        out.append("龙虎榜净买Top:")
        for item in top[:3]:
            out.append(f"  {item[1]} +{float(item[3])/10000:.0f}万")
    
    return "\n".join(out)

def scan_a_open():
    """A股开盘后10分钟"""
    from data_source import AShareRealtime
    rt = AShareRealtime()
    try:
        etf = rt.get_realtime('510050')
        if etf:
            return f"上证50ETF: {etf.get('change_pct', 0):+.2f}%"
    except:
        pass
    return "实时数据: (开盘数据待接入)"

def scan_us_pre():
    """美股开盘前"""
    try:
        import yfinance as yf
        spy = yf.download('SPY', period='5d')['Close']
        if len(spy) >= 2:
            chg = (spy[-1]/spy[-2]-1)*100
            return f"SPY: {spy[-1]:.0f} ({chg:+.2f}%)"
    except:
        pass
    return "美股数据: yfinance待确认"

def scan_us_open():
    return "美股开盘判断: (建设中)"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan', choices=['A','US'], default='A')
    parser.add_argument('--pre', action='store_true')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    
    if args.scan == 'A':
        out = scan_a_pre() if args.pre else (scan_a_open() if args.open else 'use --pre or --open')
    else:
        out = scan_us_pre() if args.pre else (scan_us_open() if args.open else 'use --pre or --open')
    print(out)
