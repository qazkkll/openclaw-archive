#!/usr/bin/env python3
"""
持仓监控器 v1.0 — 全周期持仓跟踪+止损提醒+到期评估
=====================================================
用法:
  python3 position_monitor.py open '{"ticker":"301002","name":"崧盛股份","entry_price":41.82,...}'
  python3 position_monitor.py check                    # 每日检查所有活跃持仓
  python3 position_monitor.py status                   # 显示当前持仓状态
  python3 position_monitor.py close P_abc123 42.50     # 手动平仓(含最终价格)
  python3 position_monitor.py history [days]           # 历史平仓记录

设计原则:
  - 每次推荐buy → 自动open position
  - 每天15:30 cron → 自动check所有活跃持仓
  - 触发止损/到期 → 自动提醒，不自动平仓（提醒Andy决策）
  - 所有session共享同一个positions.json
"""
import json, os, sys, hashlib
from datetime import datetime, timedelta

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
POS_FILE = os.path.join(ROOT, 'data/positions.json')
REC_FILE = os.path.join(ROOT, 'data/recommendations.json')

# ============================================================
# Data I/O
# ============================================================

def load_positions():
    if os.path.exists(POS_FILE):
        with open(POS_FILE) as f:
            return json.load(f)
    return {'positions': [], 'meta': {'created': datetime.now().isoformat(), 'version': 1}}

def save_positions(data):
    os.makedirs(os.path.dirname(POS_FILE), exist_ok=True)
    with open(POS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def make_pos_id(ticker, date):
    raw = f"{ticker}{date}position"
    return 'P_' + hashlib.md5(raw.encode()).hexdigest()[:8]

# ============================================================
# Price Fetching
# ============================================================

def get_realtime_price(ticker, market='cn'):
    """获取实时价格。cn用tushare, us用yfinance"""
    if market == 'cn':
        import tushare as ts
        pro = ts.pro_api()
        # 确保ticker格式正确
        ts_code = ticker
        if '.' not in ticker:
            if ticker.startswith('6') or ticker.startswith('9'):
                ts_code = ticker + '.SH'
            else:
                ts_code = ticker + '.SZ'
        df = ts.realtime_quote(ts_code=ts_code)
        if len(df) > 0:
            return float(df.iloc[0]['PRICE'])
    elif market == 'us':
        import yfinance as yf
        data = yf.download(ticker, period='1d', progress=False, auto_adjust=True)
        if hasattr(data.columns, 'levels'):
            data.columns = [c[0] for c in data.columns]
        if len(data) > 0:
            return float(data['Close'].iloc[-1])
    return None

def get_close_price(ticker, date_str, market='cn'):
    """获取历史收盘价"""
    if market == 'cn':
        import tushare as ts
        pro = ts.pro_api()
        ts_code = ticker
        if '.' not in ticker:
            if ticker.startswith('6') or ticker.startswith('9'):
                ts_code = ticker + '.SH'
            else:
                ts_code = ticker + '.SZ'
        df = pro.daily(ts_code=ts_code, start_date=date_str, end_date=date_str)
        if len(df) > 0:
            return float(df.iloc[0]['close'])
    return None

# ============================================================
# Core Operations
# ============================================================

def open_position(entry):
    """开仓
    entry = {
        "ticker": "301002",
        "name": "崧盛股份",
        "market": "cn",
        "entry_price": 41.82,
        "hold_days": 10,
        "stop_loss_pct": -2.0,
        "model_score": 0.0516,
        "signal": "Y",
        "rec_id": "R_abc123"  # optional, link to recommendation
    }
    """
    ticker = entry['ticker']
    date_str = datetime.now().strftime('%Y%m%d')
    pos_id = make_pos_id(ticker, date_str)

    data = load_positions()

    # 去重：同一天同一ticker不重复开仓
    for p in data['positions']:
        if p['ticker'] == ticker and p['entry_date'] == date_str and p['status'] == 'active':
            print(f"⚠️ 已存在活跃持仓: {ticker} ({pos_id})")
            return pos_id

    pos = {
        'id': pos_id,
        'rec_id': entry.get('rec_id', ''),
        'market': entry.get('market', 'cn'),
        'ticker': ticker,
        'name': entry.get('name', ''),
        'entry_price': float(entry['entry_price']),
        'entry_date': date_str,
        'hold_days': int(entry.get('hold_days', 10)),
        'stop_loss_pct': float(entry.get('stop_loss_pct', -2.0)),
        'model_score': float(entry.get('model_score', 0)),
        'signal': entry.get('signal', ''),
        'status': 'active',
        'current_price': None,
        'pnl_pct': None,
        'days_held': 0,
        'checks': [],
        'close_price': None,
        'close_date': None,
        'close_reason': None,
    }
    data['positions'].append(pos)
    save_positions(data)
    print(f"✅ 开仓: {pos_id} {ticker} {entry.get('name','')} @ {entry['entry_price']}")
    print(f"   持仓{entry.get('hold_days',10)}天, 止损{entry.get('stop_loss_pct',-2)}%")
    return pos_id

def check_positions():
    """每日检查所有活跃持仓，返回alerts列表"""
    data = load_positions()
    active = [p for p in data['positions'] if p['status'] == 'active']

    if not active:
        print("📭 没有活跃持仓")
        return []

    today = datetime.now().strftime('%Y%m%d')
    alerts = []

    print(f"📊 持仓检查 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    print(f"{'#':>2} {'代码':>8} {'名称':>6} {'入场':>7} {'现价':>7} {'盈亏':>7} {'天数':>5} {'状态':>8}")
    print(f"{'-'*70}")

    for i, pos in enumerate(active):
        ticker = pos['ticker']
        market = pos.get('market', 'cn')
        entry_price = pos['entry_price']

        # 计算持仓天数(交易日)
        entry_date = datetime.strptime(pos['entry_date'], '%Y%m%d')
        days_held = 0
        d = entry_date
        while d < datetime.now():
            d += timedelta(days=1)
            if d.weekday() < 5:  # 跳过周末
                days_held += 1
        pos['days_held'] = days_held

        # 获取当前价格
        current = get_realtime_price(ticker, market)
        if current is None:
            print(f"{i+1:>2} {ticker:>8} {pos['name'][:4]:>6} {entry_price:>7.2f} {'N/A':>7} {'N/A':>7} {days_held:>5} {'数据缺失':>8}")
            continue

        pos['current_price'] = current
        pnl = (current - entry_price) / entry_price * 100
        pos['pnl_pct'] = round(pnl, 2)

        # 状态判断
        status = '✅正常'
        alert_type = None

        # 止损检查
        if pnl <= pos['stop_loss_pct']:
            status = '🔴止损!'
            alert_type = 'stop_loss'
            alerts.append({
                'type': 'stop_loss',
                'pos_id': pos['id'],
                'ticker': ticker,
                'name': pos['name'],
                'pnl': pnl,
                'threshold': pos['stop_loss_pct'],
                'message': f"🔴 止损触发! {ticker} {pos['name']} 盈亏{pnl:+.2f}% (止损线{pos['stop_loss_pct']}%)"
            })

        # 到期检查
        elif days_held >= pos['hold_days']:
            status = '⏰到期'
            alert_type = 'mature'
            alerts.append({
                'type': 'mature',
                'pos_id': pos['id'],
                'ticker': ticker,
                'name': pos['name'],
                'pnl': pnl,
                'message': f"⏰ 到期! {ticker} {pos['name']} 持仓{days_held}天, 盈亏{pnl:+.2f}%"
            })

        # 接近止损警告(止损线的50%)
        elif pnl <= pos['stop_loss_pct'] * 0.5:
            status = '⚠️警告'
            alerts.append({
                'type': 'warning',
                'pos_id': pos['id'],
                'ticker': ticker,
                'name': pos['name'],
                'pnl': pnl,
                'message': f"⚠️ 接近止损! {ticker} {pos['name']} 盈亏{pnl:+.2f}% (止损线{pos['stop_loss_pct']}%)"
            })

        # 记录每日检查
        pos['checks'].append({
            'date': today,
            'price': current,
            'pnl_pct': pos['pnl_pct'],
            'days_held': days_held,
            'status': status
        })

        print(f"{i+1:>2} {ticker:>8} {pos['name'][:4]:>6} {entry_price:>7.2f} {current:>7.2f} {pnl:>+6.2f}% {days_held:>3}/{pos['hold_days']} {status:>8}")

    save_positions(data)

    # 打印alerts
    if alerts:
        print(f"\n{'='*70}")
        print("🚨 需要处理:")
        for a in alerts:
            print(f"  {a['message']}")

    return alerts

def show_status():
    """显示所有活跃持仓汇总"""
    data = load_positions()
    active = [p for p in data['positions'] if p['status'] == 'active']
    closed = [p for p in data['positions'] if p['status'] in ('closed', 'stopped')]

    print(f"📊 持仓总览")
    print(f"  活跃: {len(active)}")
    print(f"  已平: {len(closed)}")

    if active:
        total_pnl = 0
        print(f"\n活跃持仓:")
        for p in active:
            pnl = p.get('pnl_pct', 0) or 0
            total_pnl += pnl
            days = p.get('days_held', 0)
            print(f"  {p['id']} {p['ticker']} {p['name'][:4]} 入{p['entry_price']:.2f} 盈亏{pnl:+.2f}% {days}/{p['hold_days']}天")
        print(f"  平均盈亏: {total_pnl/len(active):+.2f}%")

def close_position(pos_id, close_price=None, reason='manual'):
    """平仓"""
    data = load_positions()
    for p in data['positions']:
        if p['id'] == pos_id and p['status'] == 'active':
            if close_price is None:
                close_price = get_realtime_price(p['ticker'], p.get('market', 'cn'))
            p['status'] = 'closed' if reason == 'manual' else 'stopped'
            p['close_price'] = close_price
            p['close_date'] = datetime.now().strftime('%Y%m%d')
            p['close_reason'] = reason
            if close_price:
                p['pnl_pct'] = round((close_price - p['entry_price']) / p['entry_price'] * 100, 2)
            save_positions(data)
            print(f"✅ 平仓: {pos_id} {p['ticker']} {p['name']} @ {close_price}")
            print(f"   入场{p['entry_price']:.2f} → 出场{close_price:.2f} = {p['pnl_pct']:+.2f}%")
            print(f"   持仓{p['days_held']}天, 原因: {reason}")
            return
    print(f"❌ 未找到活跃持仓: {pos_id}")

def show_history(days=30):
    """显示历史平仓记录"""
    data = load_positions()
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    closed = [p for p in data['positions']
              if p['status'] in ('closed', 'stopped') and p.get('close_date', '') >= cutoff]

    if not closed:
        print(f"最近{days}天没有平仓记录")
        return

    total_pnl = 0
    wins = 0
    print(f"📋 最近{days}天平仓记录 ({len(closed)}笔)")
    print(f"{'代码':>8} {'名称':>6} {'入场':>7} {'出场':>7} {'盈亏':>7} {'天数':>5} {'原因':>6}")
    print('-' * 55)
    for p in closed:
        pnl = p.get('pnl_pct', 0) or 0
        total_pnl += pnl
        if pnl > 0:
            wins += 1
        print(f"{p['ticker']:>8} {p['name'][:4]:>6} {p['entry_price']:>7.2f} {p.get('close_price',0):>7.2f} {pnl:>+6.2f}% {p.get('days_held',0):>5} {p.get('close_reason',''):>6}")

    print(f"\n命中率: {wins}/{len(closed)} = {wins/len(closed)*100:.0f}%")
    print(f"平均盈亏: {total_pnl/len(closed):+.2f}%")

# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: position_monitor.py [open|check|status|close|history]")
        print()
        print("  open '{...}'     开仓(JSON)")
        print("  check            每日检查所有活跃持仓")
        print("  status           显示持仓总览")
        print("  close ID [PRICE] 手动平仓")
        print("  history [days]   历史平仓记录")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'open':
        entry = json.loads(sys.argv[2])
        open_position(entry)
    elif cmd == 'check':
        alerts = check_positions()
        # 返回非零退出码如果有止损/到期
        if any(a['type'] in ('stop_loss', 'mature') for a in alerts):
            sys.exit(2)
    elif cmd == 'status':
        show_status()
    elif cmd == 'close':
        pos_id = sys.argv[2]
        price = float(sys.argv[3]) if len(sys.argv) > 3 else None
        reason = sys.argv[4] if len(sys.argv) > 4 else 'manual'
        close_position(pos_id, price, reason)
    elif cmd == 'history':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        show_history(days)
    else:
        print(f"未知命令: {cmd}")
        sys.exit(1)
