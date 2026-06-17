#!/usr/bin/env python3
"""
新闻抓取器 🍤
用 NewsAPI 拉取股票相关新闻（限制金融源减少噪音）。
同时提供缓存供 advisor/其他脚本使用。

用法:
  python3 scripts/news_fetcher.py NVDA
  python3 scripts/news_fetcher.py NVDA QCOM ADBE
  python3 scripts/news_fetcher.py --portfolio   → 持仓新闻
"""
import json
import sys
import os
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

API_KEY = "7d8e0ca352664b6d9ccd96405949b5ea"
WORKSPACE = '/home/admin/.openclaw/workspace'
CACHE_FILE = os.path.join(WORKSPACE, 'data/news_cache.json')

# 免费版NewsAPI可用的金融相关sources（limited）
FIN_SOURCES = 'business-insider,fortune,reuters,the-washington-post,associated-press'

# 股票→搜索词
TICKER_QUERIES = {
    'NVDA': '(NVIDIA OR NVDA) AND (stock OR earnings OR AI)',
    'QCOM': '(Qualcomm OR QCOM) AND (stock OR chip)',
    'ADBE': '(Adobe OR ADBE) AND (stock OR software)',
    'INTC': '(Intel OR INTC) AND stock',
    'MU': '(Micron OR MU) AND stock',
    'AMD': '(AMD) AND (stock OR chip)',
    'ZS': '(Zscaler OR ZS) AND stock',
    'BE': '(Bloom Energy OR BE) AND stock',
}

def is_relevant(ticker, title, desc):
    """严格过滤：确保新闻确实跟这只股票相关"""
    text = f"{title} {desc}".lower()
    ticker_lower = ticker.lower()
    
    # 排除明显错误的匹配
    noise = ['tuna', 'screen reader', 'tinned fish', 'pizza hut', 
             'intuit', 'lmw reports', 'grab bets', 'robot']
    for n in noise:
        if n in text:
            return False
    
    # 必须出现公司名或代码
    names = {
        'NVDA': ['nvidia', 'nvda', 'nvidia corporation'],
        'QCOM': ['qualcomm', 'qcom', 'snapdragon'],
        'ADBE': ['adobe', 'adbe'],
        'INTC': ['intel', 'intc'],
        'MU': ['micron', 'mu'],
        'AMD': ['amd', 'advanced micro'],
        'ZS': ['zscaler', 'zs'],
        'BE': ['bloom energy', 'be'],
    }
    
    ticker_names = names.get(ticker.upper(), [ticker_lower])
    return any(n in text for n in ticker_names)

def fetch_news(ticker, days_back=2):
    """抓取指定股票的新闻"""
    query = TICKER_QUERIES.get(ticker.upper(), f'{ticker} stock')
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # 先用不限source的广泛搜索
    params = urllib.parse.urlencode({
        'q': query,
        'from': from_date,
        'sortBy': 'relevancy',
        'pageSize': 5,
        'apiKey': API_KEY,
        'language': 'en'
    })
    
    url = f"https://newsapi.org/v2/everything?{params}"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        
        if data.get('status') != 'ok':
            return {'ticker': ticker, 'articles': [], 'error': data.get('message', 'unknown')}
        
        articles = []
        for a in data.get('articles', []):
            title = a.get('title', '')
            desc = a.get('description', '') or ''
            
            if not is_relevant(ticker, title, desc):
                continue
            
            articles.append({
                'title': title[:100],
                'source': a['source']['name'],
                'date': a['publishedAt'][:10],
                'url': a['url'],
            })
            if len(articles) >= 3:
                break
        
        return {'ticker': ticker, 'articles': articles}
        
    except Exception as e:
        return {'ticker': ticker, 'articles': [], 'error': str(e)}

def sentiment(title, desc=''):
    text = f"{title} {desc}".lower()
    pos = ['beat', 'surge', 'raise', 'upgrade', 'buy', 'positive', 'growth', 'record', 
           'bullish', 'outperform', 'strong', 'profit', 'rally', 'innovation']
    neg = ['crash', 'drop', 'cut', 'downgrade', 'sell', 'negative', 'loss', 'decline',
           'bearish', 'underperform', 'weak', 'fall', 'lawsuit', 'risk', 'investigation']
    
    pos_c = sum(1 for w in pos if w in text)
    neg_c = sum(1 for w in neg if w in text)
    
    if pos_c > neg_c: return '🟢'
    if neg_c > pos_c: return '🔴'
    return '🟡'

def batch_fetch(tickers):
    results = []
    for t in tickers:
        results.append(fetch_news(t))
        import time; time.sleep(0.3)
    return results

def format_report(results):
    parts = ["📰 持仓新闻摘要"]
    has_any = False
    for r in results:
        arts = r.get('articles', [])
        if not arts:
            continue
        has_any = True
        parts.append(f"\n**{r['ticker']}**")
        for a in arts:
            s = sentiment(a['title'])
            parts.append(f"  {s} {a['date']} {a['source']} | {a['title'][:65]}")
    if not has_any:
        return "📭 近2天无相关新闻"
    return '\n'.join(parts)

def get_portfolio_tickers():
    try:
        with open(os.path.join(WORKSPACE, 'data/portfolio.json')) as f:
            p = json.load(f)
    except:
        p = {}
    tickers = []
    for item in p.get('us_stock', []):
        code = item.get('code', '')
        if code: tickers.append(code)
    return tickers

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python3 news_fetcher.py <ticker> [ticker2...] 或 --portfolio')
        sys.exit(1)
    
    if sys.argv[1] == '--portfolio':
        tickers = get_portfolio_tickers()
    else:
        tickers = [t.upper() for t in sys.argv[1:]]
    
    if not tickers:
        print('❌ 无股票')
        sys.exit(1)
    
    results = batch_fetch(tickers)
    
    # 写缓存
    cache = {}
    try:
        with open(CACHE_FILE) as f: cache = json.load(f)
    except: pass
    for r in results:
        cache[r['ticker']] = {'articles': r['articles'], 'updated': datetime.now().isoformat()}
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    
    print(format_report(results))