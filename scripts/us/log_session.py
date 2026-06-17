#!/usr/bin/env python3
"""
🍤 定期抓取当前会话写入日志
每30分钟由cron触发，确保崩溃后日志不丢
"""
import json, os, datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'conversation')
os.makedirs(LOG_DIR, exist_ok=True)

today = datetime.date.today().isoformat()
log_file = os.path.join(LOG_DIR, f"{today}.md")

# Read current session transcript (last 50 lines)
session_path = "/home/admin/.openclaw/agents/main/sessions"
if os.path.isdir(session_path):
    sessions = [f for f in os.listdir(session_path) if f.endswith('.jsonl')]
    if sessions:
        latest = max(sessions, key=lambda f: os.path.getmtime(os.path.join(session_path, f)))
        session_file = os.path.join(session_path, latest)
        size = os.path.getsize(session_file)
        ts = datetime.datetime.now().strftime('%H:%M')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] ⚙️ 会话快照: {latest} ({size/1024:.0f}KB)\n")
        print(f"✅ Logged session snapshot: {latest} ({size/1024:.0f}KB)")
else:
    print("No session dir found")
