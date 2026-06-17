#!/usr/bin/env python3
"""
任务队列 — ☁️（我）写任务，📡（本地小钳）执行
用法: python3 scripts/task_queue.py <subcmd> [args]

子命令:
  write <type> <params_json>  — 写一条新任务
  list [status]               — 列出incoming任务
  clean [--all]               — 清理已完成任务
"""

import json, sys, os, time
from datetime import datetime

INCOMING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "incoming")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def next_task_id():
    today = datetime.now().strftime("%Y%m%d")
    existing = [f for f in os.listdir(INCOMING) if f.startswith(f"task_{today}")]
    seq = len(existing) + 1
    return f"{today}_{seq:03d}"

def write_task(task_type, params, priority="normal"):
    task = {
        "task_id": f"{datetime.now().strftime('%Y%m%d')}_{next_task_id().split('_')[-1]}",
        "type": task_type,
        "params": json.loads(params) if isinstance(params, str) else params,
        "status": "pending",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "priority": priority
    }
    fname = f"task_{task['task_id']}.json"
    fpath = os.path.join(INCOMING, fname)
    with open(fpath, "w") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    print(f"✅ 任务已写入: {fpath}")
    print(json.dumps(task, indent=2, ensure_ascii=False))
    return task["task_id"]

def list_tasks(status_filter=None):
    tasks = []
    for f in sorted(os.listdir(INCOMING)):
        if not f.endswith(".json") or f == "README.md":
            continue
        try:
            with open(os.path.join(INCOMING, f)) as fh:
                task = json.load(fh)
            if status_filter and task.get("status") != status_filter:
                continue
            tasks.append(task)
        except:
            pass
    return tasks

def show_tasks(status_filter=None):
    tasks = list_tasks(status_filter)
    if not tasks:
        print("📭 无待处理任务" if status_filter else "📭 incoming/ 为空")
        return
    print(f"📋 共 {len(tasks)} 个任务:")
    for t in tasks:
        sid = t.get("task_id", "?")
        stype = t.get("type", "?")
        sstatus = t.get("status", "?")
        stime = t.get("created_at", "?")
        print(f"  [{sstatus}] {sid}  {stype}  ({stime})")

def clean_tasks(all_flag=False):
    count = 0
    for f in os.listdir(INCOMING):
        if f == "README.md":
            continue
        fpath = os.path.join(INCOMING, f)
        if all_flag:
            os.remove(fpath)
            count += 1
        elif f.endswith(".done"):
            os.remove(fpath)
            count += 1
    print(f"🧹 清理了 {count} 个文件")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    sub = sys.argv[1]
    if sub == "write":
        if len(sys.argv) < 4:
            print("用法: task_queue.py write <type> <params_json> [priority]")
            sys.exit(1)
        p = sys.argv[4] if len(sys.argv) > 4 else "normal"
        write_task(sys.argv[2], sys.argv[3], p)
    elif sub == "list":
        sf = sys.argv[2] if len(sys.argv) > 2 else None
        show_tasks(sf)
    elif sub == "clean":
        clean_tasks("--all" in sys.argv)
    else:
        print(f"未知子命令: {sub}")
        print(__doc__)
