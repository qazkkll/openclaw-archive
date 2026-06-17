#!/usr/bin/env python3
"""
🍤 审计引擎 — 每条链路执行后调用，记录结果，有错弹你

用法:
    from audit_engine import audit
    audit('morning_scan', 'success', '晨扫完成, 100只评分正常')
    audit('data_source', 'error', '新浪API超时, 降级到Tushare')
"""
import json, os, datetime, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_LOG = os.path.join(ROOT, 'data', 'audit_events.jsonl')
SUMMARY_FILE = os.path.join(ROOT, 'data', 'audit_summary.json')

def audit(module, level, message, detail=''):
    """
    记录审计事件
    
    参数:
        module: 模块名 (如 morning_scan, scoring, data_source)
        level: 级别 (success, warning, error, critical)
        message: 简要描述
        detail: 详细错误信息（可选）
    """
    event = {
        'time': datetime.datetime.now().isoformat(),
        'module': module,
        'level': level,
        'message': message,
        'detail': str(detail)[:500]
    }
    
    # 写审计日志(JSONL)
    os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)
    with open(AUDIT_LOG, 'a') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    
    # 如果是错误/严重级别，发送给Andy
    if level in ('error', 'critical'):
        try:
            from notify import send
            send(f'🚨 审计告警 [{module}] {message}')
        except:
            pass
    
    return event

def get_daily_summary():
    """获取当日审计汇总"""
    today = datetime.date.today().isoformat()
    events = []
    try:
        with open(AUDIT_LOG) as f:
            for line in f:
                e = json.loads(line)
                if e['time'].startswith(today):
                    events.append(e)
    except:
        pass
    
    summary = {
        'date': today,
        'total': len(events),
        'by_level': {},
        'by_module': {},
        'errors': [e for e in events if e['level'] in ('error', 'critical')]
    }
    
    for e in events:
        summary['by_level'][e['level']] = summary['by_level'].get(e['level'], 0) + 1
        summary['by_module'][e['module']] = summary['by_module'].get(e['module'], 0) + 1
    
    return summary

def send_daily_report():
    """每天汇总发送给Andy"""
    summary = get_daily_summary()
    error_count = len(summary['errors'])
    
    lines = [f'📋 审计日报 · {summary["date"]}']
    lines.append(f'总事件: {summary["total"]} | 错误: {error_count}')
    
    if summary['by_level']:
        lines.append('')
        lines.append('级别分布:')
        for level in ['critical', 'error', 'warning', 'success']:
            c = summary['by_level'].get(level, 0)
            if c > 0:
                lines.append(f'  {level}: {c}次')
    
    if summary['by_module']:
        lines.append('')
        lines.append('模块分布:')
        for mod, c in sorted(summary['by_module'].items()):
            lines.append(f'  {mod}: {c}次')
    
    if error_count > 0:
        lines.append('')
        lines.append(f'🚨 {error_count}个错误待处理:')
        for e in summary['errors'][:10]:
            lines.append(f'  [{e["module"]}] {e["message"]}')
    
    report = '\n'.join(lines)
    
    try:
        from notify import send
        send(report)
    except:
        pass
    
    return report

if __name__ == '__main__':
    # 直接运行=发送今日日报
    from notify import send
    report = send_daily_report()
    print(report)
# ===== Gateway 断联监控 =====
RESTART_FLAG = '/tmp/.openclaw_restart_pending'

def mark_planned_restart(reason=''):
    """标记一次计划内的重启"""
    with open(RESTART_FLAG, 'w') as f:
        json.dump({
            'planned': True,
            'time': datetime.datetime.now().isoformat(),
            'reason': reason
        }, f)
    audit('gateway', 'warning', f'Gateway计划内重启: {reason}')
    try:
        from notify import send
        send(f'⚠️ Gateway即将重启: {reason}')
    except:
        pass

def check_unplanned_restart():
    """启动时检查上次重启是否是计划内的"""
    try:
        with open(RESTART_FLAG) as f:
            data = json.load(f)
        planned = data.get('planned', False)
        os.remove(RESTART_FLAG)
        
        if not planned:
            audit('gateway', 'critical', '⚠️ Gateway非计划断联！请检查原因')
    except FileNotFoundError:
        pass  # 正常情况，没有重启标记文件=正常运行
    except Exception as e:
        audit('gateway', 'error', f'网关自检异常: {e}')

# 启动时自动检查 — 仅限直接运行，不污染import
if __name__ == '__main__':
    check_unplanned_restart()
