#!/usr/bin/env python3
"""
🍤 质量池刷新 — 每日收盘后自动更新

数据源: Tushare daily_basic（拉全市场基本面+换手率）
流程: 拉数据 → 过滤垃圾 → 活跃度排名 → 取Top N → 保存
"""
import sys, json, urllib.request, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
CONFIG = os.path.join(ROOT, 'config', 'strategy.json')
FULL_MARKET = os.path.join(ROOT, 'data', 'full_market_stocks.json')
OUTPUT = os.path.join(ROOT, 'data', 'quality_pool.json')

def load_stock_map():
    with open(FULL_MARKET) as f:
        stocks = json.load(f)
    sm = {}
    for s in stocks:
        code = s['code']
        ts = code + '.SH' if code[:2] in ('60','68') else code + '.SZ'
        sm[ts] = s
    return sm

def fetch_daily_basic():
    """从Tushare拉全市场基本面"""
    token = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
    url = 'http://api.tushare.pro'
    import datetime
    today = datetime.date.today().strftime('%Y%m%d')
    
    payload = json.dumps({
        'api_name': 'daily_basic',
        'token': token,
        'params': {'trade_date': today}
    }).encode()
    
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    items = data.get('data', {}).get('items', [])
    fields = data.get('data', {}).get('fields', [])
    
    if not items:
        # 今天数据还没出，用昨天的
        yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y%m%d')
        payload2 = json.dumps({
            'api_name': 'daily_basic',
            'token': token,
            'params': {'trade_date': yesterday}
        }).encode()
        req2 = urllib.request.Request(url, data=payload2, headers={'Content-Type': 'application/json'})
        resp2 = urllib.request.urlopen(req2, timeout=60)
        data2 = json.loads(resp2.read())
        items = data2.get('data', {}).get('items', [])
        fields = data2.get('data', {}).get('fields', [])
        print(f'  今日数据未出，使用{yesterday}')
    
    return fields, items

def build_pool(fields, items, stock_map):
    idx = {f: i for i, f in enumerate(fields)}
    
    with open(CONFIG) as f:
        strategy = json.load(f)
    qp = strategy.get('quality_pool', {})
    TOP_N = qp.get('daily_scan_top', 1500)
    MIN_MV = qp.get('min_mv_亿', 15)
    MIN_TURNOVER = qp.get('min_turnover_pct', 0.3)
    MIN_PRICE = qp.get('min_price', 2)
    
    candidates = []
    
    for item in items:
        ts_code = item[idx['ts_code']]
        code = ts_code.split('.')[0]
        
        total_mv = item[idx['total_mv']] or 0
        turnover = item[idx['turnover_rate']] or 0
        vol_ratio = item[idx['volume_ratio']] or 0
        close = item[idx['close']] or 0
        
        info = stock_map.get(ts_code, {})
        name = info.get('name', code)
        board = info.get('board', '其他')
        mv_亿 = total_mv / 10000 if total_mv else 0
        
        try:
            close_f = float(close)
            turnover_f = float(turnover)
            vol_ratio_f = float(vol_ratio)
        except:
            continue
        
        if 'ST' in name or '退' in name: continue
        if mv_亿 < MIN_MV: continue
        if turnover_f < MIN_TURNOVER: continue
        if close_f < MIN_PRICE: continue
        if not board: continue
        
        candidates.append({
            'code': code, 'name': name, 'board': board,
            'tradeable': board in ('上证主板', '深证主板'),
            'total_mv_亿': round(mv_亿, 1),
            'turnover': turnover_f, 'vol_ratio': vol_ratio_f, 'close': close_f,
            'activity_score': round(turnover_f * 0.7 + min(vol_ratio_f, 5) * 0.3, 1)
        })
    
    candidates.sort(key=lambda x: x['activity_score'], reverse=True)
    daily_pool = candidates[:TOP_N]
    scan_codes = [s['code'] for s in daily_pool]
    
    pool = {
        'total_quality': len(candidates),
        'daily_scan_top': TOP_N,
        'tradeable_count': len([s for s in daily_pool if s['tradeable']]),
        'stocks': daily_pool,
        'scan_codes': scan_codes,
        'updated_at': None  # 你可以在外部set
    }
    
    return pool

if __name__ == '__main__':
    print('🔄 刷新质量池...')
    stock_map = load_stock_map()
    print(f'  loaded {len(stock_map)} stocks')
    
    fields, items = fetch_daily_basic()
    print(f'  daily_basic: {len(items)}只')
    
    pool = build_pool(fields, items, stock_map)
    
    import datetime
    pool['updated_at'] = datetime.datetime.now().isoformat()
    
    with open(OUTPUT, 'w') as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    
    print(f'  ✅ 质量池: {pool["total_quality"]}合格 → Top {pool["daily_scan_top"]} (可买{pool["tradeable_count"]})')
    print(f'  保存: {OUTPUT}')

# 审计记录
try:
    from audit_engine import audit
    audit('refresh_pool', 'success', '质量池刷新完成')
except:
    pass
