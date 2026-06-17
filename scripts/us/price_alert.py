#!/usr/bin/env python3
"""
价格预警检查器 🍤
监控指定股票是否接近/达到心理预期价位，触发通知。
用法: python3 scripts/price_alert.py [ticker]
"""
import json
import sys
import os
from datetime import datetime

WORKSPACE = '/home/admin/.openclaw/workspace'
ALERTS_FILE = os.path.join(WORKSPACE, 'data/price_alerts.json')
LOG_FILE = os.path.join(WORKSPACE, 'logs/price_alerts.log')

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

def get_price(ticker):
    """从advisor获取当前价格"""
    import subprocess
    r = subprocess.run(
        ['python3', 'scripts/advisor.py', ticker],
        capture_output=True, text=True, cwd=WORKSPACE, timeout=30
    )
    for line in r.stdout.split('\n'):
        if '· $' in line:
            # "🇺🇸 **NVDA** · $218.93"
            parts = line.split('· $')
            if len(parts) > 1:
                price_str = parts[1].strip().rstrip('*')
                try:
                    return float(price_str)
                except:
                    pass
    return None

def load_alerts():
    try:
        with open(ALERTS_FILE) as f:
            return json.load(f)
    except:
        return {}

def save_alerts(alerts):
    os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)

def check_alerts(alerts):
    triggered = []
    for key, info in list(alerts.items()):
        if isinstance(info, dict) and 'target' in info:
            ticker = info.get('ticker', key.split('@')[0])
        else:
            continue
        try:
            price = get_price(ticker)
        except:
            log(f'⚠️ {ticker} 获取价格超时')
            continue
        if price is None:
            log(f'⚠️ {ticker} 无法获取价格')
            continue
        
        log(f'{ticker} 现价${price:.2f} | 目标${info["target"]} | 方向{info["direction"]}')
        
        # 检查价格触碰/超过目标
        hit = False
        if info['direction'] == 'up' and price >= info['target']:
            hit = True
        elif info['direction'] == 'down' and price <= info['target']:
            hit = True
        
        # 接近目标 (98%以上=距目标2%以内) 但还没到
        approaching = None
        threshold = 0.98  # 距目标2%以内才算接近
        if info['direction'] == 'up' and price >= info['target'] * threshold and price < info['target']:
            approaching = 'up'
        elif info['direction'] == 'down' and price <= info['target'] / threshold and price > info['target']:
            approaching = 'down'
        
        if approaching:
            pct = abs(price / info['target'] - 1) * 100
            triggered.append({
                'ticker': ticker,
                'type': 'approaching',
                'direction': info['direction'],
                'current_price': round(price, 2),
                'target': info['target'],
                'gap_pct': round(pct, 1),
                'note': info.get('note', '')
            })
        
        if hit and not info.get('notified_hit'):
            triggered.append({
                'ticker': ticker,
                'type': 'hit',
                'direction': info['direction'],
                'current_price': round(price, 2),
                'target': info['target'],
                'note': info.get('note', '')
            })
            info['notified_hit'] = True
            info['hit_at'] = datetime.now().isoformat()
    
    save_alerts(alerts)
    return triggered

def setup(ticker, target, direction, note=''):
    alerts = load_alerts()
    key = f'{ticker.upper()}@{target}{"U" if direction == "up" else "D"}'
    alerts[key] = {
        'ticker': ticker.upper(),
        'target': target,
        'direction': direction,
        'note': note,
        'created_at': datetime.now().isoformat(),
        'notified_hit': False
    }
    save_alerts(alerts)
    price = get_price(ticker.upper())
    price_str = f'现价${price:.2f}' if price else '未知'
    log(f'✅ 已设置 {ticker} 预警: 目标${target} ({direction}) | {price_str} | {note}')
    print(f'✅ {ticker} 预警已设置: {"突破" if direction == "up" else "跌破"}${target}时通知')
    print(f'   当前价格: {price_str}')
    if note:
        print(f'   备注: {note}')

def list_alerts():
    alerts = load_alerts()
    if not alerts:
        print('📭 当前没有价格预警')
        return
    print('📋 当前预警列表:')
    for key, info in sorted(alerts.items()):
        if not isinstance(info, dict) or 'target' not in info:
            continue
        ticker = info.get('ticker', key.split('@')[0])
        price = get_price(ticker)
        price_str = f'现价${price:.2f}' if price else '未知'
        dist = ''
        if price:
            if info['direction'] == 'up':
                dist = f'(还需+{((info["target"]/price)-1)*100:.1f}%)'
            else:
                dist = f'(还需-{((1-info["target"]/price))*100:.1f}%)'
        notified = ' 🔔已触发' if info.get('notified_hit') else ''
        print(f'  {ticker}: {"突破" if info["direction"]=="up" else "跌破"}' 
              f'${info["target"]} {dist} {price_str}{notified}')
        if info.get('note'):
            print(f'     备注: {info["note"]}')

def delete(ticker):
    alerts = load_alerts()
    to_del = [k for k in alerts if isinstance(alerts[k], dict) and alerts[k].get('ticker', k.split('@')[0]).upper() == ticker.upper()]
    for k in to_del:
        del alerts[k]
    save_alerts(alerts)
    log(f'🗑️ 删除 {ticker} 预警({len(to_del)}个)')
    print(f'🗑️ {ticker} 预警已删除({len(to_del)}个)')

if __name__ == '__main__':
    # python3 price_alert.py check    → 检查所有预警
    # python3 price_alert.py list     → 列出所有预警
    # python3 price_alert.py set NVDA 238 up "目标说明"
    # python3 price_alert.py del NVDA → 删除预警
    
    if len(sys.argv) < 2:
        print('用法:')
        print('  python3 price_alert.py set <ticker> <target> <up/down> [备注]')
        print('  python3 price_alert.py check')
        print('  python3 price_alert.py list')
        print('  python3 price_alert.py del <ticker>')
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'set' and len(sys.argv) >= 4:
        ticker = sys.argv[2].upper()
        target = float(sys.argv[3])
        direction = sys.argv[4] if len(sys.argv) > 4 else 'up'
        note = ' '.join(sys.argv[5:]) if len(sys.argv) > 5 else ''
        setup(ticker, target, direction, note)
    
    elif cmd == 'check':
        alerts = load_alerts()
        if not alerts:
            sys.exit(0)
        triggered = check_alerts(alerts)
        if triggered:
            # 输出结构化JSON — cron捕获后会推送到会话，AI做分析
            report = json.dumps({'alerts': triggered, 'ts': datetime.now().isoformat()}, ensure_ascii=False)
            print(report)
        # 无触发则完全静默，不浪费token
    
    elif cmd == 'list':
        list_alerts()
    
    elif cmd == 'del' and len(sys.argv) >= 3:
        delete(sys.argv[2].upper())
    
    else:
        print('参数错误')
        sys.exit(1)
