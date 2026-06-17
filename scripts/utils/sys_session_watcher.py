#!/usr/bin/env python3
"""
Session Watchdog - 自动清理过大的 session transcript

Bug: https://github.com/openclaw/openclaw/issues/65501
forceFlushTranscriptBytes 在新 session 上不生效，导致 transcript 无限增长

解决方案：定期检查 session 文件大小，超过阈值时自动归档
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Fix Windows GBK encoding for emoji output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 配置
SESSIONS_DIR = Path.home() / ".openclaw" / "agents" / "main" / "sessions"
MAX_SIZE_MB = 2.0  # 超过 2MB 就归档
ARCHIVE_DIR = SESSIONS_DIR / "archived"

def get_session_files():
    """获取所有 session 文件及其大小"""
    files = []
    for f in SESSIONS_DIR.glob("*.jsonl"):
        if f.name.endswith(".bak") or "archived" in str(f):
            continue
        size = f.stat().st_size
        files.append({
            "path": f,
            "name": f.name,
            "size_mb": size / (1024 * 1024),
            "mtime": datetime.fromtimestamp(f.stat().st_mtime)
        })
    return sorted(files, key=lambda x: -x["size_mb"])

def archive_session(session_file, reason="oversized"):
    """归档一个 session 文件"""
    ARCHIVE_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{session_file.stem}.{reason}.{timestamp}.jsonl"
    archive_path = ARCHIVE_DIR / archive_name
    
    # 移动文件
    shutil.move(str(session_file), str(archive_path))
    
    # 同时清理相关的 sessions.json 条目（可选）
    # 这需要解析 sessions.json 并删除对应条目
    
    return archive_path

def main():
    print(f"🔍 Session Watchdog - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Sessions dir: {SESSIONS_DIR}")
    print(f"📏 Threshold: {MAX_SIZE_MB} MB")
    print()
    
    if not SESSIONS_DIR.exists():
        print(f"❌ Sessions directory not found: {SESSIONS_DIR}")
        return
    
    files = get_session_files()
    
    if not files:
        print("✅ No session files found")
        return
    
    print(f"Found {len(files)} session files:")
    for f in files[:10]:  # 显示前10个
        status = "⚠️" if f["size_mb"] > MAX_SIZE_MB else "✅"
        print(f"  {status} {f['name']}: {f['size_mb']:.2f} MB")
    
    # 归档过大的 session
    oversized = [f for f in files if f["size_mb"] > MAX_SIZE_MB]
    
    if not oversized:
        print("\n✅ All sessions within threshold")
        return
    
    print(f"\n⚠️ Found {len(oversized)} oversized sessions")
    
    for f in oversized:
        print(f"\n📦 Archiving: {f['name']} ({f['size_mb']:.2f} MB)")
        try:
            archive_path = archive_session(f["path"])
            print(f"   → Archived to: {archive_path}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    print(f"\n✅ Done. Archived {len(oversized)} sessions.")

if __name__ == "__main__":
    main()
