#!/usr/bin/env python3
"""
任务队列监控 v2 — 专用于系统cron，绝对静默
写入通知队列供主session读取分析
"""

import json, os, sys
from datetime import datetime

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCOMING = os.path.join(WORKSPACE, "incoming")
NOTIF_FILE = os.path.join(WORKSPACE, "data", "task_notifications.json")
STATE_FILE = os.path.join(WORKSPACE, "data", "task_monitor_state.json")

def load_json(fp):
    try:
        with open(fp) as f:
            return json.load(f)
    except:
        return {}

def save_json(fp, data):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def check_tasks():
    state = load_json(STATE_FILE)
    notified = set(state.get("notified", []))
    new_notifs = []

    seen_ids = set()
    done_extensions = (".json.done", ".done")  # 长后缀优先

    for fname in sorted(os.listdir(INCOMING)):
        if fname == "README.md" or fname.startswith("."):
            continue
        fpath = os.path.join(INCOMING, fname)

        done_id = None
        for ext in done_extensions:
            if fname.endswith(ext):
                done_id = fname[: -len(ext)].replace("task_", "", 1)
                break

        # .done / .json.done 文件（本地小钳的标记方式）
        if done_id:
            if done_id in notified or done_id in seen_ids:
                continue
            seen_ids.add(done_id)
            # 查结果文件（多种可能路径）
            result = {}
            candidates = [
                os.path.join(WORKSPACE, "data", "task_results", f"task_{done_id}.json"),
                os.path.join(WORKSPACE, "data", "task_results", f"{done_id}.json"),
            ]
            for rp in candidates:
                if os.path.exists(rp):
                    try:
                        with open(rp, encoding='utf-8') as f:
                            raw = f.read().strip()
                        try:
                            result = json.loads(raw)
                        except:
                            result = {"_raw": raw[:500]}
                    except:
                        result = {}
                    break
            new_notifs.append({"task_id": done_id, "source": "done_file", "result": result})
            notified.add(done_id)
            continue

        # 跳过.done后缀已经被处理，只处理标准.json文件
        if not fname.endswith(".json"):
            continue
        try:
            with open(fpath) as f:
                task = json.load(f)
        except:
            continue
        task_id = task.get("task_id", fname)
        if task_id in seen_ids:
            continue
        seen_ids.add(task_id)
        status = task.get("status")
        if status in ("done", "error") and task_id not in notified:
            # 查结果
            result_file = task.get("result_file")
            result = {}
            if result_file:
                rp = os.path.join(WORKSPACE, result_file)
                if os.path.exists(rp):
                    try:
                        with open(rp, encoding='utf-8') as f:
                            raw = f.read().strip()
                        try:
                            result = json.loads(raw)
                        except:
                            result = {"_raw": raw[:500]}
                    except:
                        result = {}
            new_notifs.append({"task_id": task_id, "source": "json", "status": status, "task": task, "result": result})
            notified.add(task_id)

    # 保存状态
    state["notified"] = list(notified)
    state["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_json(STATE_FILE, state)

    # 写入通知队列
    if new_notifs:
        existing = load_json(NOTIF_FILE)
        if isinstance(existing, list):
            existing.extend(new_notifs)
        else:
            existing = new_notifs
        save_json(NOTIF_FILE, existing)

    return new_notifs

if __name__ == "__main__":
    notifs = check_tasks()
    # 绝对不输出任何内容到stdout（系统cron用）
    # 所有结果写入 data/task_notifications.json
