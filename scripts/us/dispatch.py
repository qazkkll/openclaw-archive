#!/usr/bin/env python3
"""
🍤 任务分派系统 — ☁️基金经理 → 🏠研究员

用法: python3 dispatch.py <任务ID> <优先级> <任务内容>
例: python3 dispatch.py TASK001 urgent "查ZS实时价+成交量"
"""
import sys, json, os, datetime, urllib.request

# 🏠的gateway
TOKEN = "0543b9558438a07e7ae3caf55555d4aafa004daa00025397"
URL = "http://127.0.0.1:18790/tools/invoke"

def send_task(task_id, priority, content):
    msg = f"[{task_id}] [{priority}] ☁️→🏠：{content}\n\n收到请回复 ✅"
    data = json.dumps({
        "tool": "sessions_send",
        "args": {"sessionKey": "agent:main:main", "message": msg}
    }).encode()
    req = urllib.request.Request(URL, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        print(f"✅ {task_id} 已发送")
        return True
    except Exception as e:
        print(f"❌ {task_id} 发送失败: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: dispatch.py <ID> <urgent/normal> <内容>")
        sys.exit(1)
    send_task(sys.argv[1], sys.argv[2], sys.argv[3])
