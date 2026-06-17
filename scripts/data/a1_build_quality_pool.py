#!/usr/bin/env python3
"""质量池生成器：美股TOP50+持仓 + A股TOP500+ETF"""
import json, os, subprocess
from datetime import datetime

BASE = r'/home/hermes/.hermes/openclaw-archive'
DATA = os.path.join(BASE, 'data')
CLOUD = 'admin@8.217.51.136'
SSH_KEY = r'C:\Users\admin\.ssh\id_ed25519'

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def count_items(data):
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return len(v)
    return 0

def build_us():
    pool = {'updated': datetime.now().isoformat(), 'us_stocks': [], 'holdings': []}
    
    scores = load_json(os.path.join(DATA, 'daily_score.json')) or load_json(os.path.join(DATA, 'us_scored.json')) or []
    if isinstance(scores, dict):
        scores = scores.get('scores', scores.get('results', scores.get('stocks', scores.get('data', []))))
    
    pdata = load_json(os.path.join(BASE, 'portfolio_root.json')) or {}
    if isinstance(pdata, dict):
        held = set(pdata.keys())
        pf = [{'symbol':k, **v} for k,v in pdata.items()]
    else:
        pf = pdata if isinstance(pdata, list) else []
        held = {h.get('symbol','') for h in pf}
    
    # Top 50
    scored = []
    for s in scores:
        symbol = s.get('symbol', s.get('code', s.get('ticker', '')))
        scored.append({
            'symbol': symbol,
            'name': s.get('name',''),
            'sector': s.get('sector',''),
            'price': s.get('price', s.get('current_price', 0)),
            'v5': float(s.get('v5_score', s.get('v5', s.get('score', 0)))),
            'r5d': float(s.get('r5d_score', s.get('defensive', s.get('r5d', 0)))),
            'r5c': float(s.get('r5c_score', s.get('lottery', s.get('r5c', 0)))),
            'rsi': s.get('rsi', 50),
            'is_holding': symbol in held,
            'comment': s.get('comment','')
        })
    scored.sort(key=lambda x: x['v5'], reverse=True)
    pool['us_stocks'] = scored[:300]
    
    # 持仓（强制加入，不在前50的就追加）
    for h in pf:
        sym = h.get('symbol','')
        if sym and sym not in [s['symbol'] for s in pool['us_stocks']]:
            found = [s for s in scored if s['symbol'] == sym]
            if found:
                pool['us_stocks'].append(found[0])
        pool['holdings'].append({
            'symbol': sym,
            'name': h.get('name',''),
            'shares': h.get('shares', h.get('quantity', 0)),
            'cost': h.get('cost', h.get('avg_cost', 0))
        })
    
    return pool

def build_a():
    pool = {'updated': datetime.now().isoformat(), 'a_stocks': [], 'etf': []}
    
    # A股全量数据（有full更好，没有就拿top100）
    data = load_json(os.path.join(DATA, 'a_share_top100.json')) or load_json(os.path.join(DATA, 'a_share_full.json'))
    if isinstance(data, dict):
        # a_share_top100.json 格式: {timestamp, total_scored, top100: [...]}
        data = data.get('top500', data.get('top100', data.get('stocks', data.get('results', []))))
    data = data or []
    
    scored = []
    for s in data:
        symbol = s.get('symbol', s.get('code', s.get('ts_code','')))
        v1 = float(s.get('v1_score', s.get('v1', s.get('score', 0))))
        scored.append({
            'symbol': symbol,
            'name': s.get('name',''),
            'sector': s.get('sector', s.get('industry','')),
            'price': s.get('price', s.get('current_price', s.get('close', 0))),
            'v1': v1,
            'rsi': s.get('rsi', 50),
            'ma20': s.get('ma20', s.get('ma_20', None)),
            'chg_pct': s.get('chg_pct', s.get('pct_chg', None)),
            'comment': s.get('comment','')
        })
    scored.sort(key=lambda x: x['v1'], reverse=True)
    pool['a_stocks'] = scored[:400]
    
    # ETF
    for s in data:
        symbol = s.get('symbol', s.get('code', s.get('ts_code','')))
        if symbol.startswith('51') or symbol.startswith('159'):
            v1 = float(s.get('v1_score', s.get('v1', s.get('score', 0))))
            pool['etf'].append({
                'symbol': symbol,
                'name': s.get('name',''),
                'price': s.get('price', s.get('current_price', s.get('close', 0))),
                'v1': v1,
                'comment': s.get('comment','')
            })
    
    return pool

def save_and_scp(name, data):
    path = os.path.join(DATA, name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'{name}: {os.path.getsize(path)/1024:.0f} KB ({count_items(data)}条)')
    
    r = subprocess.run([
        'scp', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=10', path, f'{CLOUD}:/home/admin/.openclaw/workspace/data/{name}'
    ], capture_output=True, text=True, timeout=15)
    ok = r.returncode == 0
    print(f'  SCP: {"ok" if ok else "fail"}')
    return ok

if __name__ == '__main__':
    t = datetime.now().strftime('%H:%M')
    print(f'[{t}] Build quality pools...')
    
    us = build_us()
    a = build_a()
    
    ok_us = save_and_scp('us_quality_pool.json', us) if len(us.get('us_stocks',[])) > 0 else False
    ok_a = save_and_scp('a_quality_pool.json', a) if len(a.get('a_stocks',[])) > 0 else False
    
    print(f'  US: {len(us["us_stocks"])}只 + 持仓{len(us["holdings"])}只')
    print(f'  A股: {len(a["a_stocks"])}只 + ETF{len(a["etf"])}只')
    print(f'  US data: {"exist" if ok_us else "empty"}, A data: {"exist" if ok_a else "empty"}')
    print('  Done')
