#!/usr/bin/env python3
"""
🍤 推荐追踪系统 — 记录每次建议，一个月后复盘

用法:
    python3 scripts/track.py recommend --code NVDA --action buy --price 217 --reason "MA20线上，分析师强共识"
    python3 scripts/track.py snapshot     # 记录当前持仓快照
    python3 scripts/track.py review       # 生成复盘报告
"""
import sys, json, os, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_PATH = os.path.join(ROOT, 'data', 'recommendations.json')
SNAP_PATH = os.path.join(ROOT, 'data', 'portfolio_snapshots.json')

def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return [] if 'snapshot' not in path else {}

def _save(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ===== 记录推荐 =====
def record_recommendation(code, name='', action='buy/sell/hold', price=0, reason='', market='us'):
    recs = _load(REC_PATH)
    recs.append({
        'date': datetime.date.today().isoformat(),
        'time': datetime.datetime.now().strftime('%H:%M'),
        'code': code.upper(),
        'name': name,
        'action': action,
        'price': price,
        'reason': reason,
        'market': market,
        'status': 'open'  # open/closed
    })
    _save(REC_PATH, recs)
    print(f'✅ 已记录: {action.upper()} {code} @${price}')

# ===== 记录持仓快照 =====
def snapshot_portfolio(positions, total_value, cash):
    """记录某一天的持仓快照"""
    snaps = _load(SNAP_PATH)
    today = datetime.date.today().isoformat()
    
    snaps[today] = {
        'date': today,
        'time': datetime.datetime.now().strftime('%H:%M'),
        'positions': positions,  # [{code, shares, price, cost}]
        'total_value': total_value,
        'cash': cash,
        'recommendations': _load(REC_PATH)
    }
    _save(SNAP_PATH, snaps)
    print(f'✅ 快照已保存: {today}')
    print(f'   {len(positions)}只持仓, 总值${total_value:.0f}')

# ===== 复盘报告 =====
def generate_review():
    recs = _load(REC_PATH)
    snaps = _load(SNAP_PATH)
    
    print('📊 一个月复盘报告')
    print(f'='*50)
    print(f'统计周期: 2026-05-22 至 2026-06-22')
    print(f'总推荐数: {len(recs)}')
    print()
    
    # 按股票统计
    by_stock = {}
    for r in recs:
        code = r['code']
        if code not in by_stock:
            by_stock[code] = []
        by_stock[code].append(r)
    
    print('各股票推荐记录:')
    for code, items in sorted(by_stock.items()):
        print(f'  {code}: {len(items)}次')
        for item in items:
            print(f'    {item["date"]} {item["action"]} @${item["price"]} — {item["reason"][:50]}')
    
    print()
    print('持仓变化(快照):')
    for date in sorted(snaps.keys()):
        snap = snaps[date]
        positions = snap.get('positions', [])
        print(f'  {date}: {len(positions)}只持仓, 总值${snap.get("total_value",0):.0f}')
        for p in positions:
            print(f'    {p.get("code","?")}: {p.get("shares",0)}股 @${p.get("cost","?")} 现${p.get("price","?")}')
    
    return {'recommendations': len(recs), 'snapshots': len(snaps)}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: track.py recommend|snapshot|review')
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == 'recommend':
        record_recommendation(
            code=sys.argv[sys.argv.index('--code')+1],
            action=sys.argv[sys.argv.index('--action')+1],
            price=float(sys.argv[sys.argv.index('--price')+1]),
            reason=sys.argv[sys.argv.index('--reason')+1] if '--reason' in sys.argv else ''
        )
    elif cmd == 'snapshot':
        # 从stdin或参数读持仓
        print('请输入持仓JSON:')
        import ast
        try:
            data = json.loads(sys.stdin.read())
            snapshot_portfolio(data.get('positions',[]), data.get('total',0), data.get('cash',0))
        except:
            print('需传入持仓数据')
    elif cmd == 'review':
        generate_review()
