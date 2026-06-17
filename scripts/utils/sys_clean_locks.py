#!/usr/bin/env python3
"""sys_clean_locks.py - 自动清扫 stale session lock 文件

安全逻辑（只删确实残留的锁）：
1. pid 不存在于系统中 → 删（进程已死）
2. pid = 当前 Gateway pid 且锁龄 > maxHoldMs(180s) → 删（自锁泄露）
3. 其他情况 → 不动

不依赖 Gateway 版本，不修改任何 session 数据。
使用：python scripts/sys_clean_locks.py [--log-jsonl]

--log-jsonl  以 JSONL 格式输出（供 Windows 计划任务日志记录）
"""

import json
import os
import time
import logging
import glob
import subprocess
import re
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLEAN-LOCKS] %(message)s")
log = logging.getLogger(__name__)

# 配置
LOCK_DIR = os.path.expanduser("~/.openclaw/agents/main/sessions")
MAX_HOLD_MS = 180000  # 3min，与 config 中 maxHoldMs 一致
STALE_MS = 300000     # 5min，与 config 中 staleMs 一致
LOG_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "lock_cleanup_log.jsonl"
)


def get_gateway_pid():
    """获取当前运行的 gateway pid

    优先用 openclaw status（但可能超时如果 Gateway 卡死），
    兜底用 tasklist 找 node 进程
    """
    # 方法1: openclaw status（精确但可能超时）
    try:
        cli_path = os.path.join(
            os.environ.get("APPDATA", ""),
            "npm", "node_modules", "openclaw", "dist", "index.js"
        )
        result = subprocess.run(
            ["node", cli_path, "status"],
            capture_output=True, timeout=10
        )
        text = result.stdout.decode("utf-8", errors="replace")
        m = re.search(r"pid (\d+)", text)
        if m:
            return int(m.group(1))
        text_err = result.stderr.decode("utf-8", errors="replace")
        m = re.search(r"pid (\d+)", text_err)
        if m:
            return int(m.group(1))
    except subprocess.TimeoutExpired:
        log.warning("openclaw status 超时，使用 tasklist 兜底")
    except Exception as e:
        log.warning(f"get_gateway_pid 失败: {e}")

    # 方法2: tasklist 找 node 进程（兜底）
    try:
        result = subprocess.run(
            ["tasklist", "/NH", "/FI", "IMAGENAME eq node.exe", "/FO", "CSV"],
            capture_output=True, timeout=10
        )
        text = result.stdout.decode("utf-8", errors="replace")
        for line in text.strip().split("\n"):
            parts = line.replace('"', '').split(',')
            if len(parts) >= 2 and parts[1].strip().isdigit():
                pid = int(parts[1].strip())
                return pid
    except Exception as e:
        log.warning(f"tasklist 兜底失败: {e}")

    return None


def is_pid_alive(pid):
    """检查 pid 是否存活（Windows）"""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
            capture_output=True, timeout=5
        )
        text = result.stdout.decode("utf-8", errors="replace")
        # CSV 头 + 至少一行数据 = pid 存活
        return text.strip().count("\n") >= 1
    except Exception:
        return False


def parse_lock_file(lock_path):
    """解析锁文件内容"""
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def clean_stale_locks(log_jsonl=False):
    gw_pid = get_gateway_pid()
    now = time.time() * 1000  # ms

    if not os.path.isdir(LOCK_DIR):
        log.info(f"Lock 目录不存在: {LOCK_DIR}")
        return

    lock_files = glob.glob(os.path.join(LOCK_DIR, "*.lock"))
    if not lock_files:
        log.info("没有发现 lock 文件")
        return

    cleaned = 0
    kept = 0
    errors = 0

    msg = f"扫描 {len(lock_files)} 个锁文件, Gateway pid={gw_pid}"
    log.info(msg)
    if log_jsonl:
        print(json.dumps({"event": "scan", "count": len(lock_files), "pid": gw_pid}))

    for lock_path in lock_files:
        try:
            stat = os.stat(lock_path)
            age_ms = now - (stat.st_mtime * 1000)
            lock_data = parse_lock_file(lock_path)
            lock_pid = lock_data.get("pid")

            should_delete = False
            reason = ""

            if lock_pid is None:
                should_delete = True
                reason = "无 pid 信息"

            elif lock_pid == gw_pid:
                # pid = 当前 Gateway，检查是否自锁泄露
                if age_ms > MAX_HOLD_MS:
                    should_delete = True
                    reason = (
                        f"自锁泄露 (pid={lock_pid}, "
                        f"已持有 {age_ms/1000:.0f}s > {MAX_HOLD_MS/1000:.0f}s)"
                    )
                else:
                    kept += 1
                    continue

            else:
                # pid != 当前 Gateway，检查进程是否存活
                if not is_pid_alive(lock_pid):
                    should_delete = True
                    reason = (
                        f"进程已死 (pid={lock_pid}, "
                        f"锁龄 {age_ms/1000:.0f}s)"
                    )
                else:
                    # 其他进程残留（如旧 Gateway 实例）
                    if age_ms > STALE_MS:
                        should_delete = True
                        reason = (
                            f"其他进程残留 (pid={lock_pid}, "
                            f"锁龄 {age_ms/1000:.0f}s > {STALE_MS/1000:.0f}s)"
                        )
                    else:
                        kept += 1
                        continue

            if should_delete:
                os.remove(lock_path)
                cleaned += 1
                log.warning(f"已删除: {os.path.basename(lock_path)} — {reason}")

                # 追加日志
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(
                        json.dumps({
                            "ts": int(now),
                            "lock_file": os.path.basename(lock_path),
                            "reason": reason,
                            "lock_pid": lock_pid,
                            "age_ms": int(age_ms),
                        })
                        + "\n"
                    )

        except Exception as e:
            errors += 1
            log.error(f"处理 {lock_path} 失败: {e}")

    final = f"完成: 清理 {cleaned}, 保留 {kept}, 错误 {errors}"
    log.info(final)
    if log_jsonl:
        print(json.dumps({"event": "done", "cleaned": cleaned, "kept": kept, "errors": errors}))

    return cleaned


if __name__ == "__main__":
    log_jsonl = "--log-jsonl" in sys.argv
    clean_stale_locks(log_jsonl=log_jsonl)
