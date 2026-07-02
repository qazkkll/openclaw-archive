#!/usr/bin/env python3
"""
数据新鲜度检查日志 — 记录每个数据源最后被检查的时间
由us_data_update_all.py和cn_data_update_all.py调用
格式: {source_name: "2026-07-02T05:00:00"}
"""
import json
from pathlib import Path
from datetime import datetime

LOG_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "falcon" / "freshness_check_log.json"


def mark_checked(source_name: str):
    """标记某个数据源刚被检查过"""
    log = {}
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE) as f:
                log = json.load(f)
        except:
            log = {}
    log[source_name] = datetime.now().isoformat()
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def mark_batch(source_names: list):
    """批量标记多个数据源"""
    log = {}
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE) as f:
                log = json.load(f)
        except:
            log = {}
    now = datetime.now().isoformat()
    for name in source_names:
        log[name] = now
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def get_last_check(source_name: str) -> str:
    """获取某个数据源最后检查时间，返回ISO字符串或None"""
    if not LOG_FILE.exists():
        return None
    try:
        with open(LOG_FILE) as f:
            log = json.load(f)
        return log.get(source_name)
    except:
        return None


if __name__ == "__main__":
    # 显示所有检查记录
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)
        for k, v in sorted(log.items()):
            print(f"  {k}: {v}")
    else:
        print("No check log found")
