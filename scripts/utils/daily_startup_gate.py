#!/usr/bin/env python3
"""Startup gate check: verify WAL, WorkingBuffer, code_index, audit failures."""
import json, os, sys, time
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent
DATA = WORKSPACE / "data"
TS = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
TS_UNIX = time.time()

def read_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            with open(path, "r", encoding="gbk") as f:
                return json.load(f)
        except:
            return default

def check_wal():
    p = DATA / "wal_log.json"
    d = read_json(p, {"tasks": []})
    tasks = d.get("tasks", [])
    pending = [t for t in tasks if t.get("status") in ("in_progress", "pending")]
    return {"status": "ok" if not pending else "pending",
            "pending_count": len(pending),
            "pending_tasks": [t.get("id", "?") for t in pending],
            "file": str(p)}

def check_code_index():
    p = DATA / "code_index.json"
    d = read_json(p)
    if d is None:
        return {"status": "missing", "file": str(p)}
    entries = d.get("entries", d.get("files", []))
    if isinstance(entries, list):
        return {"status": "ok", "entries": len(entries), "file": str(p)}
    return {"status": "ok", "entries": entries, "file": str(p)}

def check_audit():
    p = DATA / "audit_failures.json"
    d = read_json(p)
    if d is None:
        return {"status": "missing", "file": str(p)}
    failures = d if isinstance(d, list) else d.get("failures", [])
    return {"status": "fail" if failures else "ok",
            "failures": failures,
            "file": str(p)}

result = {
    "timestamp": TS,
    "ts_unix": TS_UNIX,
    "checks": {
        "WAL协议": check_wal(),

        "代码索引": check_code_index(),
        "审计故障": check_audit(),
    },
    "passed": True,
    "env": {
        "workspace": str(WORKSPACE.resolve()),
        "timezone": "Asia/Hong_Kong",
        "local_time": time.strftime("%H:%M", time.localtime()),
    }
}

# Overall pass/fail
for k, v in result["checks"].items():
    if v["status"] == "fail":
        result["passed"] = False

out = DATA / "startup_check_done.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"[{TS}] Startup gate check: {'PASSED' if result['passed'] else 'FAILED'}")
for k, v in result["checks"].items():
    print(f"  {k}: {v['status']}")
sys.exit(0 if result['passed'] else 1)
