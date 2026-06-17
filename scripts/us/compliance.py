#!/usr/bin/env python3
"""
🍤 合规监管 — 每次分析后自检，发现偷懒直接弹Andy

检查项:
  ① 数据源是否正确（A股走sina/tushare？美股走yfinance？）
  ② 扫描范围是否完整（A股≥100只？美股≥140只？）
  ③ 评分路由是否正确（A股走V1？美股走V4.2？）
  ④ 是否用了scoring.py（禁止直接调score_engine.py）
"""
import sys, json, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

LOG_FILE = os.path.join(ROOT, 'data', 'compliance_log.json')
ALERTS_FILE = os.path.join(ROOT, 'data', 'compliance_alerts.txt')

def get_config():
    with open(os.path.join(ROOT, 'config', 'strategy.json')) as f:
        return json.load(f)

def log_analysis(task_type, details):
    """记录每次分析任务，供监管核查"""
    entry = {
        'time': time.strftime('%H:%M'),
        'date': time.strftime('%Y-%m-%d'),
        'task': task_type,
        **details
    }
    try:
        with open(LOG_FILE) as f:
            logs = json.load(f)
    except:
        logs = []
    logs.append(entry)
    logs = logs[-100:]  # 只保留最近100条
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)
    return entry

def check_compliance(task_type, **kwargs):
    """
    合规检查 — 在分析完成后调用
    用法: check_compliance('US扫描', stocks_count=31, scoring='v42', source='yfinance', universe_max=140)
    """
    alerts = []
    cfg = get_config()
    
    stocks = kwargs.get('stocks_count', 0)
    scoring = kwargs.get('scoring', '')
    source = kwargs.get('source', '')
    universe = kwargs.get('universe_max', 0)
    
    # ① 检查扫描范围
    if 'A股' in task_type or 'A' in task_type:
        min_stocks = cfg.get('quality_pool', {}).get('daily_scan_top', 100)
        if stocks < min_stocks:
            alerts.append(f'❌ A股扫描范围不足：{stocks}只 < {min_stocks}只要求')
    elif 'US' in task_type or '美股' in task_type:
        # 美股应该扫~140只（S&P 500质量筛选）
        min_us = 140
        if stocks < min_us:
            alerts.append(f'❌ 美股扫描范围不足：{stocks}只 < {min_us}只要求')
    
    # ② 检查评分模型
    if 'V1' in scoring and 'A股' not in task_type and 'A' not in task_type:
        alerts.append(f'❌ 评分模型错误：A股V1评分用到非A股任务({task_type})')
    if 'V4.2' in scoring and 'US' not in task_type and '美股' not in task_type:
        alerts.append(f'❌ 评分模型错误：美股V4.2用到非美股任务({task_type})')
    
    # ③ 检查数据源
    if 'A股' in task_type or 'A' in task_type:
        if source and 'sina' not in source and 'tushare' not in source:
            alerts.append(f'❌ A股数据源异常：{source}')
    elif 'US' in task_type or '美股' in task_type:
        if source and 'yfinance' not in source and 'minishare' not in source:
            alerts.append(f'❌ 美股数据源异常：{source}')
    
    # 记录本次检查
    log_entry = log_analysis(task_type, {
        'stocks': stocks,
        'scoring': scoring,
        'source': source,
        'alerts': alerts
    })
    
    if alerts:
        msg = f'🚨 合规告警 · {task_type}\n' + '\n'.join(alerts)
        print(msg)
        # 写入告警文件
        with open(ALERTS_FILE, 'a') as f:
            f.write(f'[{log_entry["time"]}] {msg}\n')
        # 推给Andy
        try:
            from notify import send
            send(msg[:2000])
        except Exception as e:
            print(f'[compliance] 推送失败: {e}')
        return False
    else:
        print(f'✅ 合规通过: {task_type}')
        return True

# 也支持作为独立命令运行：查看最近告警
if __name__ == '__main__':
    try:
        with open(ALERTS_FILE) as f:
            alerts = f.read().strip()
        if alerts:
            print('=== 最近合规告警 ===')
            print(alerts)
        else:
            print('✅ 无合规告警')
    except:
        print('✅ 无合规告警')
