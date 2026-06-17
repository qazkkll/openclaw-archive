#!/usr/bin/env python3
"""
每日对话归档 — 替代旧的 archive_raw_sessions.py
每天03:00运行，归档前一天的对话记录到 conversation_archive/

流程：
1. 读取 session 原始 JSONL 文件（从 sessions/ 目录）
2. 转换 md 可读格式
3. 打包为 zip
4. 更新索引
5. （可选）推云端
"""
import json, os, sys, shutil, zipfile, glob
from datetime import datetime, timedelta, timezone
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TZ = timezone(timedelta(hours=8))
WORKSPACE = "D:\\openclaw-workspace"
ARCHIVE_DIR = os.path.join(WORKSPACE, "conversation_archive")
RAW_SESSIONS_DIR = os.path.join(WORKSPACE, "logs", "conversation", "raw")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
INDEX_FILE = os.path.join(ARCHIVE_DIR, "index.json")
CLOUD = "admin@8.217.51.136"

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"totalArchives": 0, "lastArchive": None, "archives": []}

def save_index(idx):
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False, indent=2)

def archive_previous_day():
    """归档前一天的记录"""
    today = datetime.now(TZ)
    target_date = today - timedelta(days=1)
    date_str = target_date.strftime("%Y-%m-%d")
    year_month = target_date.strftime("%Y-%m")
    
    index = load_index()
    
    # 如果已归档过今天，跳过
    for a in index.get("archives", []):
        if a.get("date") == date_str:
            print(f"[archive_daily] {date_str} 已归档，跳过")
            return
    
    out_dir = os.path.join(ARCHIVE_DIR, "sessions", year_month)
    ensure_dir(out_dir)
    
    # 找 session 文件 — 从 logs/conversation/raw/ 读取
    session_files = []
    raw_day_dir = os.path.join(RAW_SESSIONS_DIR, date_str)
    if os.path.isdir(raw_day_dir):
        for f in glob.glob(os.path.join(raw_day_dir, "*.jsonl"), recursive=False):
            session_files.append(f)
    # 也读 readable 目录 — 复制已有的 md 摘要
    readable_day_dir = os.path.join(WORKSPACE, "logs", "conversation", "readable", date_str)
    if os.path.isdir(readable_day_dir):
        arch_readable = os.path.join(ARCHIVE_DIR, "readable", date_str)
        ensure_dir(arch_readable)
        for f in glob.glob(os.path.join(readable_day_dir, "*.md")):
            shutil.copy2(f, arch_readable)
            readable_files.append(f)
    
    archived_sessions = []
    readable_files = []
    
    for sf_path in session_files:
        sid = os.path.splitext(os.path.basename(sf_path))[0]
        try:
            # 复制原始文件到归档
            dst = os.path.join(out_dir, f"{sid}.jsonl")
            shutil.copy2(sf_path, dst)
            archived_sessions.append(sid)
            
            # 生成可读 md 摘要
            readable_dir = os.path.join(ARCHIVE_DIR, "readable", date_str)
            ensure_dir(readable_dir)
            md_path = os.path.join(readable_dir, f"{sid}.md")
            
            with open(sf_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            with open(md_path, 'w', encoding='utf-8') as mf:
                mf.write(f"# Session {sid}\n")
                mf.write(f"> 日期: {date_str}\n\n")
                for line in lines[-50:]:  # 最后50条对话
                    try:
                        msg = json.loads(line.strip())
                        role = msg.get("role", "?")
                        content = msg.get("content", "")
                        if isinstance(content, str) and content.strip():
                            mf.write(f"**{role}**: {content[:200]}\n\n")
                    except:
                        pass
            readable_files.append(md_path)
        except Exception as e:
            print(f"  ⚠ {sid}: {e}")
    
    # 关联的 memory 文件
    memory_files = []
    if os.path.isdir(MEMORY_DIR):
        for fname in [f"{date_str}.md"]:
            fp = os.path.join(MEMORY_DIR, fname)
            if os.path.exists(fp):
                memory_files.append(fname)
    
    # 打包
    zip_name = f"archive_{date_str}.zip"
    zip_path = os.path.join(ARCHIVE_DIR, zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in session_files:
            zf.write(f, os.path.basename(f))
        for mf_name in memory_files:
            mf_path = os.path.join(MEMORY_DIR, mf_name)
            if os.path.exists(mf_path):
                zf.write(mf_path, f"memory/{mf_name}")
    
    # 更新索引
    entry = {
        "date": date_str,
        "sessions": archived_sessions[:10] if len(archived_sessions) > 10 else archived_sessions,
        "totalSessions": len(archived_sessions),
        "memoryFiles": memory_files,
        "archiveZip": zip_name,
        "sizeMB": round(os.path.getsize(zip_path) / (1024*1024), 1),
        "timestamp": datetime.now(TZ).isoformat()
    }
    index.setdefault("archives", []).append(entry)
    index["totalArchives"] = len(index["archives"])
    index["lastArchive"] = date_str
    save_index(index)
    
    # 推云端
    push_result = "skipped"
    try:
        import subprocess
        r = subprocess.run(
            f'scp "{zip_path}" {CLOUD}:/home/admin/archive/ 2>&1',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            push_result = "ok"
        else:
            push_result = f"scp_fail: {r.stderr.strip()[:50]}"
    except Exception as e:
        push_result = f"scp_err: {e}"
    
    print(f"[archive_daily] ✅ {date_str}")
    print(f"  ├ sessions: {len(archived_sessions)}")
    print(f"  ├ memory: {memory_files}")
    print(f"  ├ zip: {entry['sizeMB']}MB")
    print(f"  └ cloud: {push_result}")
    
    # 也写一个 daily 总览
    daily_path = os.path.join(ARCHIVE_DIR, f"{date_str}.md")
    with open(daily_path, 'w', encoding='utf-8') as f:
        f.write(f"# {date_str} 存档总结\n\n")
        f.write(f"- **sessions**: {len(archived_sessions)}\n")
        f.write(f"- **memory**: {memory_files}\n")
        f.write(f"- **大小**: {entry['sizeMB']}MB\n\n")
        f.write(f"## 会话列表\n\n")
        for s in archived_sessions:
            f.write(f"- {s}\n")
    
    return entry

def main():
    print(f"[archive_daily] 开始归档...")
    print(f"  时间: {datetime.now(TZ).isoformat()}")
    print(f"  工作区: {WORKSPACE}")
    print()
    
    result = archive_previous_day()
    if result:
        print(f"\n[archive_daily] 完成 ✅")
    else:
        print(f"\n[archive_daily] 无新数据")

if __name__ == '__main__':
    main()
