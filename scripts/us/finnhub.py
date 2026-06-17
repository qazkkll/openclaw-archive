#!/usr/bin/env python3
"""
Finnhub 数据整合 🍤
用法:
  python3 scripts/finnhub.py news NVDA            → 拉新闻
  python3 scripts/finnhub.py profile NVDA         → 公司概况
  python3 scripts/finnhub.py metrics NVDA         → 财务指标
  python3 scripts/finnhub.py recommendations QCOM → 分析师评级
  python3 scripts/finnhub.py earnings             → 未来3天财报日历
  python3 scripts/finnhub.py portfolio             → 全量拉持仓数据
"""
import json
import sys
import os
import urllib.request
from datetime import datetime, timedelta

API_KEY = "d87hklhr01qmhakfrh0gd87hklhr01qmhakfrh10"
WORKSPACE = '/home/admin/.openclaw/workspace'
BASE = "https://finnhub.io/api/v1"
CACHE = os.path.join(WORKSPACE, 'data/finnhub_cache.json')

def _get(path, params=None):
    url = f"{BASE}{path}?token={API_KEY}"
    if params:
        url += '&' + '&'.join(f'{k}={urllib.request.quote(str(v))}' for k,v in params.items())
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {'error': str(e)}

def get_news(ticker, days=2):
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    data = _get(f'/company-news', {'symbol': ticker, 'from': from_date, 'to': to_date})
    if isinstance(data, list):
        return [{
            'headline': a.get('headline', '')[:100],
            'source': a.get('source', ''),
            'date': a.get('datetime', ''),
            'summary': (a.get('summary', '') or '')[:150],
            'url': a.get('url', ''),
            'sentiment': a.get('sentiment', {}).get('score', 0),
        } for a in data[:4]]
    return []

def get_profile(ticker):
    return _get(f'/stock/profile2', {'symbol': ticker})

def get_metrics(ticker):
    data = _get(f'/stock/metric', {'symbol': ticker, 'metric': 'all'})
    return data.get('metric', {})

def get_recommendations(ticker):
    data = _get(f'/stock/recommendation', {'symbol': ticker})
    return data if isinstance(data, list) else []

def get_earnings_calendar(days_ahead=7):
    today = datetime.now().strftime('%Y-%m-%d')
    end = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
    data = _get(f'/calendar/earnings', {'from': today, 'to': end})
    return data.get('earningsCalendar', [])

def format_news_report(tickers):
    parts = ["📰 新闻速览"]
    for t in tickers:
        news = get_news(t)
        if not news:
            continue
        parts.append(f"\n**{t}**")
        for a in news:
            sent = '🟢' if a['sentiment'] > 0.2 else ('🔴' if a['sentiment'] < -0.2 else '🟡')
            parts.append(f"  {sent} {a['source']} | {a['headline'][:65]}")
    return '\n'.join(parts) if len(parts) > 1 else "📭 无新闻"

def format_earnings_report():
    cal = get_earnings_calendar(5)
    portfolio = ['NVDA','QCOM','ADBE','INTC','MU','AMD','GOOGL','ZS','BE','AAPL']
    hits = [r for r in cal if r.get('symbol') in portfolio]
    
    if not hits:
        return "📭 持仓股未来5天无财报"
    
    lines = ["📅 近期财报"]
    for r in hits:
        est = r.get('epsEstimate') or '?'
        lines.append(f"  {r['date']} {r['symbol']} Q{r.get('quarter')} 预估EPS:{est}")
    return '\n'.join(lines)

def update_cache(ticker, data):
    try:
        with open(CACHE) as f: cache = json.load(f)
    except: cache = {}
    cache[ticker] = data
    cache['_updated'] = datetime.now().isoformat()
    with open(CACHE, 'w') as f: json.dump(cache, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: finnhub.py news|profile|metrics|recommendations|earnings|portfolio [ticker]')
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'news':
        tickers = sys.argv[2:] or ['NVDA','QCOM','ADBE','INTC']
        for t in tickers:
            news = get_news(t)
            update_cache(t, {'news': news, 'ts': datetime.now().isoformat()})
            print(f"\n**{t}** ({len(news)}条)")
            for a in news:
                s = '🟢' if a['sentiment'] > 0.2 else ('🔴' if a['sentiment'] < -0.2 else '🟡')
                print(f"  {s} {a['source']} | {a['headline']}")
    
    elif cmd == 'profile':
        for t in sys.argv[2:]:
            p = get_profile(t)
            print(f"\n**{t}**")
            for k in ['name','exchange','finnhubIndustry','marketCapitalization','ipo']:
                print(f"  {k}: {p.get(k, 'N/A')}")
    
    elif cmd == 'metrics':
        for t in sys.argv[2:]:
            m = get_metrics(t)
            print(f"\n**{t}**")
            for k in ['peBasicExclExtraTTM','priceToBookTTM','epsBasicExclExtraItemsTTM',
                       'revenueGrowth','grossMarginTTM','operatingMarginTTM',
                       'currentRatio','beta','52WeekHigh','52WeekLow']:
                if k in m and m[k]:
                    print(f"  {k}: {m[k]}")
    
    elif cmd == 'recommendations':
        for t in sys.argv[2:]:
            recs = get_recommendations(t)
            if recs:
                r = recs[0]
                print(f"{t}: 买入{r.get('buy',0)} 持有{r.get('hold',0)} 卖出{r.get('sell',0)}")
    
    elif cmd == 'earnings':
        print(format_earnings_report())
    
    elif cmd == 'portfolio':
        portfolio = ['NVDA','QCOM','ADBE','INTC','MU','AMD','GOOGL','ZS','BE','AAPL']
        for t in portfolio:
            p = get_profile(t)
            m = get_metrics(t)
            recs = get_recommendations(t)
            news = get_news(t, days=1)
            data = {
                'name': p.get('name',''),
                'industry': p.get('finnhubIndustry',''),
                'marketCap': p.get('marketCapitalization',0),
                'pe': m.get('peBasicExclExtraTTM'),
                'beta': m.get('beta'),
                '52wHigh': m.get('52WeekHigh'),
                '52wLow': m.get('52WeekLow'),
                'recommendations': recs[0] if recs else {},
                'news': news
            }
            update_cache(t, data)
            print(f"✅ {t}")

    else:
        print(f'未知命令: {cmd}')