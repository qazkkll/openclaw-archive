#!/usr/bin/env python3
"""
实盘日志系统 — 自动记录推荐+追踪表现
=====================================
用法:
  python log_tracker.py add <代码> <名称> <买入价> <推理理由>
    例: python log_tracker.py add 000063 中兴通讯 38.51 "资金流Top1，大单净+5.2%"

  python log_tracker.py list              查看所有活跃持仓
  python log_tracker.py update <ID> <当前价>  更新某票追踪
  python log_tracker.py report             输出表现汇总

日志位置: data/position_log.jsonl
"""
import json, os, sys, traceback
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TZ = timezone(timedelta(hours=8))
LOGPATH = os.path.join(os.path.dirname(__file__), "..", "data", "position_log.jsonl")
LOGPATH = os.path.abspath(LOGPATH)

now = datetime.now(TZ)
today = now.strftime("%Y-%m-%d")
now_ts = now.isoformat()

# ─── 工具 ──────────────────────────────────────────────────

def load_positions():
    records = []
    if os.path.isfile(LOGPATH):
        with open(LOGPATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def save_records(records):
    with open(LOGPATH, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def get_next_id(records):
    ids = [r.get('id', 0) for r in records]
    return (max(ids) + 1) if ids else 1

# ─── 功能 ──────────────────────────────────────────────────

def cmd_add(code, name, buy_price, reason):
    records = load_positions()
    rid = get_next_id(records)
    
    entry = {
        'id': rid,
        'type': 'position',
        'code': code,
        'name': name,
        'buy_price': buy_price,
        'current_price': buy_price,
        'reason': reason,
        'entry_date': today,
        'last_update': now_ts,
        'status': 'active',  # active | closed
        'exit_price': None,
        'exit_date': None,
        'exit_reason': None,
        'pnl_pct': 0,
        'check_ins': [],
    }
    
    records.append(entry)
    save_records(records)
    print(f"✅ 已记录 #{rid} {name}({code}) 买入价{buy_price}")

def cmd_list():
    records = load_positions()
    active = [r for r in records if r.get('status') == 'active']
    closed = [r for r in records if r.get('status') == 'closed']
    
    if active:
        print(f"\n=== 活跃持仓 ({len(active)}) ===")
        for r in active:
            chg = (r['current_price'] - r['buy_price']) / r['buy_price'] * 100 if r['buy_price'] > 0 else 0
            warn = " ⚠️" if chg < -12 else ""
            print(f"  #{r['id']} {r['name']}({r['code']}) 买{r['buy_price']} 现{r['current_price']} {chg:+.2f}%{warn}")
            print(f"    买入日: {r['entry_date']} | 理由: {r['reason']}")
    
    if closed:
        print(f"\n=== 已平仓 ({len(closed)}) ===")
        for r in closed[:10]:
            chg = (r['exit_price'] - r['buy_price']) / r['buy_price'] * 100 if r['buy_price'] > 0 else 0
            print(f"  #{r['id']} {r['name']}({r['code']}) {r['entry_date']}→{r['exit_date']} {chg:+.2f}% | {r['exit_reason']}")
    
    print(f"\n总计: {len(active)}活跃 + {len(closed)}已平仓")

def cmd_update(rid, current_price):
    records = load_positions()
    found = False
    for r in records:
        if r['id'] == rid and r.get('status') == 'active':
            r['current_price'] = current_price
            r['pnl_pct'] = round((current_price - r['buy_price']) / r['buy_price'] * 100, 2)
            r['last_update'] = now_ts
            r.setdefault('check_ins', []).append({
                'time': now_ts,
                'price': current_price,
                'pnl_pct': r['pnl_pct']
            })
            found = True
            break
    if found:
        save_records(records)
        print(f"✅ #{rid} 已更新为{current_price}")
    else:
        print(f"❌ 未找到活跃持仓 #{rid}")

def cmd_close(rid, exit_price, reason="手动平仓"):
    records = load_positions()
    found = False
    for r in records:
        if r['id'] == rid and r.get('status') == 'active':
            r['status'] = 'closed'
            r['exit_price'] = exit_price
            r['exit_date'] = today
            r['exit_reason'] = reason
            r['pnl_pct'] = round((exit_price - r['buy_price']) / r['buy_price'] * 100, 2)
            found = True
            break
    if found:
        save_records(records)
        print(f"✅ #{rid} 已平仓，盈亏{r['pnl_pct']:+.2f}%")
    else:
        print(f"❌ 未找到活跃持仓 #{rid}")

def cmd_report():
    records = load_positions()
    active = [r for r in records if r.get('status') == 'active']
    closed = [r for r in records if r.get('status') == 'closed']
    
    total_trades = len(records)
    win_trades = len([r for r in closed if r.get('pnl_pct', 0) > 0])
    loss_trades = len([r for r in closed if r.get('pnl_pct', 0) <= 0])
    
    avg_win = 0
    avg_loss = 0
    if win_trades > 0:
        avg_win = sum([r['pnl_pct'] for r in closed if r['pnl_pct'] > 0]) / win_trades
    if loss_trades > 0:
        avg_loss = sum([r['pnl_pct'] for r in closed if r['pnl_pct'] <= 0]) / loss_trades
    
    print(f"\n=== 实盘日志汇总 ===")
    print(f"总交易数: {total_trades}")
    print(f"活跃: {len(active)} | 已平仓: {len(closed)}")
    print(f"胜率: {win_trades}/{closed if closed else 1} ({win_trades * 100 / closed if closed else 0:.1f}%)")
    print(f"平均盈利: {avg_win:+.2f}%")
    print(f"平均亏损: {avg_loss:+.2f}%")
    
    if closed:
        print(f"\n盈亏曲线:")
        cumulative = 0
        for r in sorted(closed, key=lambda x: x.get('entry_date', '')):
            cumulative += r['pnl_pct']
            print(f"  {r['entry_date']} {r['name']}({r['code']}): {r['pnl_pct']:+.2f}% → 累计{cumulative:+.2f}%")

# ─── CLI ──────────────────────────────────────────────────

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        cmd_list()
    elif args[0] == 'add' and len(args) >= 4:
        cmd_add(args[1], args[2], float(args[3]), ' '.join(args[4:]) if len(args) > 4 else "无理由")
    elif args[0] == 'update' and len(args) >= 3:
        cmd_update(int(args[1]), float(args[2]))
    elif args[0] == 'close' and len(args) >= 3:
        cmd_close(int(args[1]), float(args[2]), ' '.join(args[3:]) if len(args) > 3 else '手动平仓')
    elif args[0] == 'report':
        cmd_report()
    elif args[0] == 'list':
        cmd_list()
    else:
        print("用法:")
        print("  python log_tracker.py add <代码> <名称> <买入价> [理由]")
        print("  python log_tracker.py update <ID> <当前价>")
        print("  python log_tracker.py close <ID> <平仓价> [理由]")
        print("  python log_tracker.py list")
        print("  python log_tracker.py report")
