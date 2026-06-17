#!/usr/bin/env python3
"""
🍤 报告新鲜度检测 — 确保每天的早报/复盘按时生成

检查:
- morning_report.txt 是不是今天的？
- 如果没更新 → 弹 Andy

用法: python3 scripts/check_stale_reports.py [--silent]
"""
import sys, os, json, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKS = {
    'A股晨扫': {
        'file': os.path.join(ROOT, 'data', 'morning_report.txt'),
        'max_age_minutes': 120,  # 08:15跑, 最晚09:00前应有
        'severity': 'error',
    },
    '质量池刷新': {
        'file': os.path.join(ROOT, 'data', 'quality_pool.json'),
        'max_age_hours': 72,     # 覆盖周末(周五17:00→周一09:00约64h)
        'severity': 'warning',
    },
}

def check():
    now = datetime.datetime.now()
    failures = []
    
    for name, cfg in CHECKS.items():
        fpath = cfg['file']
        if not os.path.exists(fpath):
            failures.append((name, f'文件不存在: {fpath}', cfg['severity']))
            continue
        
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
        age = now - mtime
        
        if 'max_age_minutes' in cfg:
            max_age = cfg['max_age_minutes']
            unit = '分钟'
            actual = age.total_seconds() / 60
        else:
            max_age = cfg.get('max_age_hours', 24) * 60
            unit = '分钟'
            actual = age.total_seconds() / 60
        
        if actual > max_age:
            failures.append((name, f'上次更新: {mtime.strftime("%m-%d %H:%M")} (已过{int(actual)}分钟)', cfg['severity']))
    
    return failures

if __name__ == '__main__':
    silent = '--silent' in sys.argv
    
    failures = check()
    
    if failures:
        lines = ['🚨 报告新鲜度检查 — 以下文件过期:']
        for name, detail, severity in failures:
            icon = '🚨' if severity == 'error' else '⚠️'
            lines.append(f'  {icon} {name}: {detail}')
        
        msg = '\n'.join(lines)
        
        if not silent:
            try:
                from notify import send
                send(msg)
            except:
                pass
        
        # 记录审计
        try:
            sys.path.insert(0, ROOT)
            from scripts.audit_engine import audit
            audit('stale_check', 'error' if any(s == 'error' for _, _, s in failures) else 'warning', msg)
        except:
            pass
        
        sys.exit(1)
    else:
        if not silent:
            print('✅ 所有报告文件正常')
        sys.exit(0)
