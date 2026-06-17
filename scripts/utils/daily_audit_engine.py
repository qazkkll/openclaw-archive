#!/usr/bin/env python3
"""
审计引擎 - 检查每个cron节点是否按计划执行
每天08:00晨流时读这个文件，告诉我哪些任务没做
"""
import json, os
from datetime import datetime, timezone, timedelta

AUDIT_DIR = r'/home/hermes/.hermes/openclaw-archive\audit'
os.makedirs(AUDIT_DIR, exist_ok=True)

# 定义扫描任务列表（香港时间）
SCAN_SCHEDULE = [
    {"task": "scan_1900_full", "label": "19:00全扫2000只", "hk_time": "19:00"},
    {"task": "scan_2100_full", "label": "21:00全扫2000只", "hk_time": "21:00"},
    {"task": "scan_2130_full", "label": "21:30全扫2000只", "hk_time": "21:30"},
    {"task": "scan_2230_top500", "label": "22:30前500名+异动", "hk_time": "22:30"},
    {"task": "scan_2330_top500", "label": "23:30前500名+异动", "hk_time": "23:30"},
    {"task": "scan_0030_top500", "label": "00:30前500名+异动", "hk_time": "00:30"},
    {"task": "scan_0130_full", "label": "01:30全扫2000只", "hk_time": "01:30"},
]

AUDIT_FILE = os.path.join(AUDIT_DIR, 'audit_log.json')

def mark_done(task_name, status="ok", extra=None):
    """标记任务已完成"""
    data = {}
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, encoding='utf-8') as f:
            data = json.load(f)
    
    today = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d')
    if today not in data:
        data[today] = {}
    
    data[today][task_name] = {
        "status": status,
        "time": datetime.now().strftime('%H:%M'),
        "extra": extra or {}
    }
    
    with open(AUDIT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def check_today():
    """检查今天的任务完成情况，返回文本报告"""
    data = {}
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, encoding='utf-8') as f:
            data = json.load(f)
    
    today = datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d')
    today_data = data.get(today, {})
    
    lines = []
    lines.append(f'📋 {today} 扫描审计报告')
    lines.append('')
    
    completed = 0
    total = len(SCAN_SCHEDULE)
    
    for s in SCAN_SCHEDULE:
        task = s["task"]
        if task in today_data:
            t = today_data[task]
            lines.append(f'  ✅ {s["label"]} → {t["time"]} ({t["status"]})')
            completed += 1
        else:
            lines.append(f'  ⏳ {s["label"]} → 未执行')
    
    lines.append('')
    lines.append(f'完成: {completed}/{total}')
    if completed == total:
        lines.append('✅ 全部完成')
    else:
        lines.append(f'⚠️ 还有{total-completed}个任务未执行')
    
    return '\n'.join(lines)

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == 'mark':
        extra = json.loads(sys.argv[3]) if len(sys.argv) >= 4 else {}
        mark_done(sys.argv[2], extra=extra)
        print(f'✅ marked {sys.argv[2]} done')
    else:
        print(check_today())
