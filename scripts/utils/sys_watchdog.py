#!/usr/bin/env python3
"""
Session预警 + 自动归档触发器
每2小时运行一次（08:00-22:00），检查当前活跃session大小
- 超过阈值（默认35MB）→ 发tg提醒
- 超过危险值（45MB）→ 自动触发归档
"""
import json, os, sys, glob, shutil, zipfile
from datetime import datetime, timedelta, timezone
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TZ = timezone(timedelta(hours=8))
WORKSPACE = "/home/hermes/.hermes/openclaw-archive"
SESSIONS_DIR = os.path.join(os.path.dirname(WORKSPACE), "agents", "main", "sessions")
ARCHIVE_DIR = os.path.join(WORKSPACE, "conversation_archive")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")

# 阈值
WARN_MB = 35
DANGER_MB = 45

def get_session_sizes():
    """获取所有活跃session文件大小"""
    sessions = []
    target_dir = SESSIONS_DIR
    if not os.path.isdir(target_dir):
        # fallback到workspace/logs
        alt = os.path.join(WORKSPACE, "logs", "conversation", "raw")
        if os.path.isdir(alt):
            target_dir = alt
    
    if os.path.isdir(target_dir):
        for f in glob.glob(os.path.join(SESSIONS_DIR, "*.jsonl")):
            size = os.path.getsize(f) / (1024*1024)
            sessions.append((os.path.basename(f), round(size, 1)))
    
    # 也检查 trajectory 文件
    traj_dir = os.path.join(WORKSPACE, "logs", "conversation", "raw")
    if os.path.isdir(traj_dir):
        # 检查今天
        today = datetime.now(TZ).strftime("%Y-%m-%d")
        today_dir = os.path.join(traj_dir, today)
        if os.path.isdir(today_dir):
            for f in glob.glob(os.path.join(today_dir, "*.jsonl")):
                size = os.path.getsize(f) / (1024*1024)
                sessions.append((f"raw/{os.path.basename(f)}", round(size, 1)))
    
    return sorted(sessions, key=lambda x: -x[1])

def check_session_health():
    sessions = get_session_sizes()
    if not sessions:
        print(f"[session_watchdog] 未找到session文件")
        print("STATUS:no_sessions")
        return
    
    max_name, max_mb = sessions[0]
    total_mb = sum(s[1] for s in sessions)
    
    print(f"[session_watchdog] {datetime.now(TZ).strftime('%H:%M')}")
    print(f"  ├ 最大session: {max_name} ({max_mb}MB)")
    print(f"  ├ 总大小: {round(total_mb, 1)}MB")
    print(f"  ├ 文件数: {len(sessions)}")
    
    if total_mb >= DANGER_MB:
        print(f"  └ ⚠️ 危险！总大小超越危险阈值 ({DANGER_MB}MB)")
        print("STATUS:danger")
        print(f"ACTION:auto_archive_needed")
    elif total_mb >= WARN_MB:
        print(f"  └ ⚡ 预警！总大小超越警告阈值 ({WARN_MB}MB)")
        print("STATUS:warn")
    else:
        print(f"  └ ✅ 正常")
        print("STATUS:ok")
    
    # 输出完整列表
    if len(sessions) > 0:
        print("\n所有session:")
        for name, mb in sessions:
            flag = "⚠️" if mb >= DANGER_MB else ("⚡" if mb >= WARN_MB else "  ")
            print(f"  {flag} {name}: {mb}MB")

if __name__ == '__main__':
    check_session_health()
