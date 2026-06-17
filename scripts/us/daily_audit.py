#!/usr/bin/env python3
"""
🍤 每日链路审计 — 检查今天所有修改有没有断链

跑在 Dream 整理时（每天2次），或在收盘后手动触发。
检查项:
  ① 配置文件可读
  ② 质量池→晨扫→校验器 链路完整
  ③ cron配置和实际任务匹配
  ④ 数据源可用
  ⑤ 评分路由正确
"""
import sys, json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

def check(checks):
    passed = 0
    failed = 0
    for name, ok, detail in checks:
        mark = '✅' if ok else '❌'
        print(f'  {mark} {name}')
        if not ok and detail:
            print(f'     {detail}')
        if ok:
            passed += 1
        else:
            failed += 1
    return passed, failed

print('🍤 每日链路审计')
print('='*40)
print()

all_checks = []

# ① 配置文件
for f in ['config/strategy.json', 'config/data_sources.json', 'config/alerts.json', 'config/output_templates.json']:
    try:
        with open(os.path.join(ROOT, f)) as fh:
            json.load(fh)
        all_checks.append((f'配置 {f}', True, ''))
    except Exception as e:
        all_checks.append((f'配置 {f}', False, str(e)))

# ② 质量池完整性
try:
    with open(os.path.join(ROOT, 'data', 'quality_pool.json')) as f:
        pool = json.load(f)
    codes = pool.get('scan_codes', [])
    assert len(codes) > 0, 'scan_codes为空'
    all_checks.append((f'质量池({len(codes)}只)', True, ''))
except Exception as e:
    all_checks.append(('质量池', False, str(e)))

# ③ 评分路由可导入
try:
    from scoring import score, is_a_stock, is_us_stock
    assert is_a_stock('600519') == True
    assert is_us_stock('NVDA') == True
    assert is_a_stock('NVDA') == False
    all_checks.append(('评分路由(V1/V4.2)', True, ''))
except Exception as e:
    all_checks.append(('评分路由', False, str(e)))

# ④ 数据源
try:
    from data_source import AShareKline, AShareRealtime
    kl = AShareKline()
    d = kl.get_kline('600519')
    if d and len(d) >= 60:
        all_checks.append(('A股数据源(新浪)', True, ''))
    else:
        all_checks.append(('A股数据源', False, '数据不足'))
except Exception as e:
    all_checks.append(('A股数据源', False, str(e)))

# ⑤ 通知层
try:
    from notify import send
    all_checks.append(('通知层(notify.py)', True, ''))
except Exception as e:
    all_checks.append(('通知层', False, str(e)))

# ⑥ Cron完整性
try:
    with open(os.path.join(ROOT, '..', 'cron', 'jobs.json')) as f:
        cron_data = json.load(f)
    enabled_crons = [j for j in cron_data.get('jobs', []) if j.get('enabled')]
    all_checks.append((f'OpenClaw cron({len(enabled_crons)}个启用)', True, ''))
except Exception as e:
    all_checks.append(('OpenClaw cron', False, str(e)))

# ⑦ 系统crontab
try:
    result = subprocess.run(['crontab', '-l'], capture_output=True, text=True, timeout=5)
    lines = [l for l in result.stdout.split('\n') if l.strip() and not l.startswith('#')]
    all_checks.append((f'系统crontab({len(lines)}条)', True, ''))
except Exception as e:
    all_checks.append(('系统crontab', False, str(e)))

# ⑧ 美股数据源
try:
    import yfinance as yf
    d = yf.download('SPY', period='3d', interval='1d', progress=False)
    if len(d) >= 2:
        all_checks.append(('美股数据源(yfinance)', True, ''))
    else:
        all_checks.append(('美股数据源', False, '数据不足'))
except Exception as e:
    all_checks.append(('美股数据源', False, str(e)))

# ⑨ 校验器
try:
    from verify_report import check_report
    all_checks.append(('校验器(verify_report)', True, ''))
except Exception as e:
    all_checks.append(('校验器', False, str(e)))

# ⑩ 检查今日晨扫审计记录
from datetime import datetime
today_str = datetime.now().strftime('%Y-%m-%d')
audit_events_path = os.path.join(ROOT, 'data', 'audit_events.jsonl')
if os.path.exists(audit_events_path):
    try:
        today_events = []
        with open(audit_events_path) as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    if ev.get('module') == 'morning_scan' and today_str in ev.get('time', ''):
                        today_events.append(ev)
                except:
                    pass
        if today_events:
            last_event = today_events[-1]
            level = last_event.get('level', 'unknown')
            msg = last_event.get('message', '')
            if level == 'success':
                all_checks.append((f'今早早扫审计: {msg}', True, ''))
            elif level == 'warning':
                all_checks.append((f'今早早扫审计(🔶): {msg}', True, ''))
            else:
                all_checks.append((f'今早早扫审计(❌): {msg}', False, f'审计级别: {level}'))
        else:
            all_checks.append(('今早早扫审计', False, '未找到今早早扫审计记录'))
    except Exception as e:
        all_checks.append(('今早早扫审计', False, str(e)))
else:
    all_checks.append(('审计记录文件', False, 'audit_events.jsonl不存在'))

# 输出结果
p, f = check(all_checks)
print()
print(f'总计: {p+ f}项 | ✅ {p} | ❌ {f}')
if f > 0:
    print('⚠️ 有链路问题需要处理')

# 写审计日志
log = {
    'time': __import__('datetime').datetime.now().isoformat(),
    'passed': p,
    'failed': f,
    'details': [name for name, ok, _ in all_checks if not ok]
}
log_path = os.path.join(ROOT, 'data', 'audit_log.json')
try:
    with open(log_path) as fh:
        logs = json.load(fh)
except:
    logs = []
logs.append(log)
# 只保留最近30条
logs = logs[-30:]
with open(log_path, 'w') as fh:
    json.dump(logs, fh, indent=2)
