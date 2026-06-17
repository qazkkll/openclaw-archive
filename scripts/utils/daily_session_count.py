#!/usr/bin/env python3
"""
检查当前session消息数，超过阈值打印提醒
用法: python check_session_count.py [--warn 100] [--danger 150]
"""
import json, os, sys, glob
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
SESSIONS_DIR = "C:\\Users\\admin\\.openclaw\\agents\\main\\sessions"

WARN = 100
DANGER = 150

# 取当前session文件（按修改时间最新的.jsonl，不是trajectory）
files = [f for f in glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")) 
         if not f.endswith(".trajectory.jsonl")]
files.sort(key=os.path.getmtime, reverse=True)

if not files:
    print("❌ 未找到session文件")
    sys.exit(1)

latest = files[0]
name = os.path.basename(latest)
size_mb = round(os.path.getsize(latest) / (1024*1024), 1)

# 读消息数
count = 0
try:
    with open(latest, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                count += 1
except:
    pass

mtime = datetime.fromtimestamp(os.path.getmtime(latest), tz=TZ)

# 纯ASCII输出，防GBK报错
def out(s, fd=sys.stdout):
    try:
        fd.write(s + '\n')
    except:
        fd.write(s.encode('ascii', errors='replace').decode() + '\n')

out(f"== Session ==")
out(f"File: {name}")
out(f"Size: {size_mb}MB")
out(f"Messages: {count}")
out(f"Updated: {mtime.strftime('%H:%M')}")

if count >= DANGER:
    out("")
    out("*** DANGER ***")
    out(f"!!! Current session has {count} messages, exceeded danger threshold {DANGER}!")
    out("!!! Action: let me archive and restart session")
    out("*** DANGER ***")
elif count >= WARN:
    out("")
    out("** WARNING **")
    out(f"!! Current session has {count} messages, approaching limit (threshold {WARN})")
    out("** WARNING **")
else:
    out(f"OK ({count}/{WARN})")
