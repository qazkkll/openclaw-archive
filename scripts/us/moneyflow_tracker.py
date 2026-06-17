#!/usr/bin/env python3
"""
Money Flow Tracker - Tushare daily money flow + northbound monitoring

Pipeline:
  1. Get northbound money flow (沪深港通)
  2. Get individual stock money flow for top candidates
  3. Save to data/moneyflow_today.json
"""

import json, os, urllib.request, datetime, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
TUSHARE_URL = 'http://api.tushare.pro'

def ts_api(api_name, params):
    payload = json.dumps({'api_name': api_name, 'token': TOKEN, 'params': params}).encode()
    req = urllib.request.Request(TUSHARE_URL, data=payload, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    return data.get('data', {})

def get_trade_date():
    today = datetime.date.today()
    for offset in range(3):
        d = (today - datetime.timedelta(days=offset)).strftime('%Y%m%d')
        result = ts_api('daily_basic', {'trade_date': d})
        if result.get('items'):
            return d
    return today.strftime('%Y%m%d')

def track_moneyflow():
    print('Money Flow Daily Tracker')
    print('=' * 50)

    trade_date = get_trade_date()
    print('Trade date:', trade_date)

    # 1. Northbound money
    print('\nNorthbound (沪深港通):')
    hsgt = ts_api('moneyflow_hsgt', {'start_date': trade_date, 'end_date': trade_date})
    if hsgt.get('items'):
        fields = hsgt['fields']
        items = hsgt['items'][0]
        idx = {f: i for i, f in enumerate(fields)}
        north = float(items[idx.get('north_money', idx.get('ggt_ss', 0))]) / 10000
        print(f'  Net northbound: {north:.0f} 万元')
    else:
        print('  No data (non-trading day)')
        north = 0

    # 2. Top stock money flow
    print('\nTop stock money flow:')
    pool_path = os.path.join(ROOT, 'data', 'quality_pool.json')
    candidates = []
    try:
        with open(pool_path) as f:
            pool = json.load(f)
        candidates = pool.get('stocks', [])[:30]
    except Exception as e:
        print('  Cannot read quality pool:', e)

    stock_flows = []
    for s in candidates:
        code = s['code']
        ts_code = f"{code}.SH" if code[:2] in ('60','68') else f"{code}.SZ"
        mf = ts_api('moneyflow', {'ts_code': ts_code, 'start_date': trade_date, 'end_date': trade_date})
        if mf.get('items'):
            fields = mf['fields']
            item = mf['items'][0]
            idx = {f: i for i, f in enumerate(fields)}
            net_amount = float(item[idx.get('net_amount', idx.get('buy_lg_amount', 0))] or 0)
            if abs(net_amount) > 100:
                stock_flows.append({
                    'code': code, 'name': s['name'],
                    'net_amount_wan': round(net_amount / 10000, 1),
                    'score': s.get('activity_score', 0)
                })

    stock_flows.sort(key=lambda x: abs(x['net_amount_wan']), reverse=True)
    for sf in stock_flows[:10]:
        direction = '+' if sf['net_amount_wan'] > 0 else '-'
        print(f'  {direction} {sf["name"]}({sf["code"]}) 净{sf["net_amount_wan"]:+.1f}万')

    # 3. Save results
    result = {
        'date': trade_date,
        'north_flow_wan': north,
        'top_stock_flows': stock_flows[:10],
        'updated_at': datetime.datetime.now().isoformat()
    }
    output_path = os.path.join(ROOT, 'data', 'moneyflow_today.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {output_path}')

    print('\n' + '=' * 50)

    try:
        from scripts.audit_engine import audit
        audit('moneyflow_tracker', 'success', f'Money flow done: {trade_date}')
    except:
        pass

if __name__ == '__main__':
    track_moneyflow()
