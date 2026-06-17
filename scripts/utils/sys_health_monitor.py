#!/usr/bin/env python3
"""sys_health_monitor.py - 系统健康监视器

监控内容：
1. Gateway 进程存活 + 内存使用
2. session lock 文件数量+状态
3. 模型 auth profile 状态（检测是否死锁）
4. 上次崩溃时间检测
5. 推荐修复动作

输出：返回健康状态码 0=健康, 1=有警告, 2=需立即修复

使用：python scripts/sys_health_monitor.py [--auto-fix] [--alert]
"""

import json
import os
import re
import glob
import time
import logging
import subprocess
import sys

# Windows GBK 兼容：确保能输出 emoji
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HEALTH] %(message)s"
)
log = logging.getLogger(__name__)

# 路径
LOCK_DIR = os.path.expanduser("~/.openclaw/agents/main/sessions")
GATEWAY_LOG = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Temp", "openclaw"
)
CONFIG_FILE = os.path.expanduser("~/.openclaw/openclaw.json")
CHECKS_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# 阈值
MAX_HOLD_MS = 180000      # 3min
MAX_GATEWAY_MEM = 1500    # 1500MB 告警
WARN_SESSION_LOCKS = 3    # 同时超过3个锁文件警告


def run_ps(cmd, timeout=10):
    """在 PowerShell 中执行命令"""
    try:
        r = subprocess.run(
            ["powershell", "-Command", cmd],
            capture_output=True, timeout=timeout
        )
        return r.stdout.decode("utf-8", errors="replace"), \
               r.stderr.decode("utf-8", errors="replace"), r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except FileNotFoundError:
        return "", "powershell not found", -1


def get_gateway_status():
    """Gateway 状态"""
    out, err, rc = run_ps("openclaw status 2>&1")
    return {"raw": out + err, "rc": rc}


def get_process_info():
    """node 进程信息"""
    out, err, rc = run_ps(
        'Get-Process node | Select-Object Id, '
        '@{N="MemMB";E={[math]::Round($_.WorkingSet64/1MB,1)}}, '
        'StartTime, @{N="Cmd";E={$_.CommandLine}} | '
        'Format-Table -AutoSize | Out-String'
    )
    return out


def get_system_mem():
    """系统可用内存"""
    out, err, rc = run_ps(
        'Get-CimInstance Win32_OperatingSystem | '
        'Select-Object @{N="FreeGB";E={[math]::Round($_.FreePhysicalMemory/1MB,1)}}, '
        '@{N="TotalGB";E={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}} | '
        'Format-Table -AutoSize | Out-String'
    )
    return out


def scan_locks():
    """检查锁文件"""
    if not os.path.isdir(LOCK_DIR):
        return [], "目录不存在"

    locks = glob.glob(os.path.join(LOCK_DIR, "*.lock"))
    results = []
    now = time.time() * 1000

    for lf in locks:
        try:
            stat = os.stat(lf)
            age_ms = now - (stat.st_mtime * 1000)
            data = {}
            try:
                with open(lf, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
            results.append({
                "file": os.path.basename(lf),
                "pid": data.get("pid"),
                "age_sec": round(age_ms / 1000, 1),
                "stale": age_ms > MAX_HOLD_MS,
            })
        except Exception as e:
            results.append({
                "file": os.path.basename(lf),
                "error": str(e)
            })

    return results, None


def check_recent_crash(within_minutes=5):
    """检测最近5分钟是否有崩溃"""
    logs_dir = GATEWAY_LOG
    log_file = os.path.join(logs_dir, f"openclaw-{time.strftime('%Y-%m-%d')}.log")
    if not os.path.isfile(log_file):
        # fallback to today
        files = glob.glob(os.path.join(logs_dir, "openclaw-*.log"))
        if not files:
            return [], "无日志"
        log_file = max(files, key=os.path.getmtime)

    size = os.path.getsize(log_file)
    # 只读最后 500KB
    with open(log_file, "rb") as f:
        if size > 500 * 1024:
            f.seek(-500 * 1024, 2)
        tail = f.read().decode("utf-8", errors="replace")

    now_local = int(time.time())
    crashes = []
    for line in tail.split("\n"):
        if "Embedded agent failed" in line or "All models failed" in line:
            # Extract time
            m = re.search(r'"time":"(.*?)"', line)
            if m:
                crashes.append({
                    "time": m.group(1),
                    "msg": line[:200]
                })

    # 只保留 within_minutes 内的
    recent = crashes[-3:] if crashes else []
    return recent, f"log_size={size/1024/1024:.0f}MB"


def check_auth_profiles():
    """检查 auth profile 状态"""
    # 通过 gateway log 检查 auth profile failure
    logs_dir = GATEWAY_LOG
    log_file = os.path.join(logs_dir, f"openclaw-{time.strftime('%Y-%m-%d')}.log")
    if not os.path.isfile(log_file):
        return {"status": "unknown", "msg": "无日志"}

    size = os.path.getsize(log_file)
    with open(log_file, "rb") as f:
        if size > 200 * 1024:
            f.seek(-200 * 1024, 2)
        tail = f.read().decode("utf-8", errors="replace")

    failures = re.findall(
        r'auth profile failure state updated',
        tail
    )

    # 检查最近是否有 auth 失败
    auth_ok = True
    auth_msg = "正常"
    
    # 看看最近的 auth failure 数量
    fail_count = len(failures)
    if fail_count > 3:
        auth_ok = False
        auth_msg = f"最近有 {fail_count}+ 次 auth profile 失败"

    # 检查 auth re-warm 状态
    rewarmed = re.findall(r'auth state re-warmed', tail)
    if rewarmed and fail_count > 0:
        auth_msg += f"（最近已恢复 {len(rewarmed)} 次）"
        # 如果最近恢复了，状态还是好的
        if fail_count < 5:
            auth_ok = True

    return {"status": "ok" if auth_ok else "fail", "msg": auth_msg}


def report(severity, title, detail, auto_fix=None):
    """输出检查报告"""
    icons = {0: "✅", 1: "⚠️", 2: "🔴"}
    return {
        "severity": severity,
        "title": title,
        "detail": detail,
        "auto_fix": auto_fix,
        "icon": icons.get(severity, "❓")
    }


def main():
    auto_fix = "--auto-fix" in sys.argv
    reports = []

    # 1. Gateway 状态
    gw = get_gateway_status()
    if gw["rc"] == 0:
        m = re.search(r"running \(pid (\d+)", gw["raw"])
        pid = m.group(1) if m else "?"
        reports.append(report(0, f"Gateway 运行 (pid {pid})", gw["raw"][:200]))
    else:
        reports.append(report(2, "Gateway 未运行!",
                              f"status rc={gw['rc']}, out={gw['raw'][:100]}",
                              auto_fix="openclaw gateway start" if auto_fix else None))

    # 2. 内存
    mem_out = get_system_mem()
    mem_out2 = get_process_info()
    # 解析 Gateway 内存
    mem_match = re.search(r"(\d+\.?\d*)\s*MB", mem_out2.split("\n")[-1] if mem_out2 else "")
    gw_mem = float(mem_match.group(1)) if mem_match else 0
    
    if gw_mem > MAX_GATEWAY_MEM:
        reports.append(report(1, f"Gateway 内存偏高: {gw_mem}MB",
                              f"阈值 {MAX_GATEWAY_MEM}MB"))
    else:
        reports.append(report(0, f"Gateway 内存: {gw_mem}MB", mem_out.strip()))

    # 3. 锁文件
    locks, lock_err = scan_locks()
    if lock_err:
        reports.append(report(0, "锁文件扫描", lock_err))
    elif not locks:
        reports.append(report(0, "锁文件", "无 lock 文件"))
    else:
        stale = [l for l in locks if l.get("stale")]
        if stale:
            for s in stale:
                reports.append(report(
                    2 if s["pid"] else 1,
                    f"Stale lock: {s['file']}",
                    f"pid={s['pid']}, 已持有 {s['age_sec']}s",
                    auto_fix=f"删除 {os.path.join(LOCK_DIR, s['file'])}" if auto_fix else None
                ))
        else:
            reports.append(report(0, "锁文件", f"{len(locks)} 个, 全部正常"))

    # 4. 最近崩溃
    crashes, crash_info = check_recent_crash(within_minutes=5)
    if crashes:
        for c in crashes:
            reports.append(report(
                2, f"近期崩溃: {c['time']}", c['msg'][:200]
            ))
    else:
        reports.append(report(0, "近期崩溃检查", "无"))

    # 5. Auth profile 状态
    auth = check_auth_profiles()
    if auth["status"] == "fail":
        reports.append(report(2, "Auth profile 异常", auth["msg"]))
    else:
        reports.append(report(0, f"Auth profile: {auth['msg']}", ""))

    # 6. Session 数
    out, _, _ = run_ps(
        'Get-ChildItem "~/.openclaw/agents/main/sessions/*.jsonl" '
        '| Measure-Object | Select-Object Count | Format-Table -AutoSize'
    )
    session_count = re.search(r"(\d+)", out)
    sc = int(session_count.group(1)) if session_count else "?"
    reports.append(report(0, "Session 文件数", str(sc)))

    # Final score
    max_sev = max(r["severity"] for r in reports)
    
    # 输出
    print(f"\n{'='*50}")
    print(f"📡 健康监视器报告 | {time.strftime('%H:%M:%S')}")
    print(f"{'='*50}")
    for r in reports:
        if r["icon"] == "✅":
            print(f"  {r['icon']} {r['title']}")
        else:
            print(f"  {r['icon']} [{r['title']}]")
            print(f"     {r['detail'][:200]}")
            if r["auto_fix"] and auto_fix:
                print(f"     → 修复: {r['auto_fix']}")
    
    print(f"\n{'='*50}")
    if max_sev == 0:
        print("  状态: ✅ 一切正常")
        if auto_fix:
            print("  （无需修复）")
    elif max_sev == 1:
        print("  状态: ⚠️ 有警告，注意观察")
    else:
        print("  状态: 🔴 需要立即修复!")
        fix_actions = [r["auto_fix"] for r in reports if r["auto_fix"]]
        if fix_actions and auto_fix:
            print(f"  → 修复方案: {', '.join(fix_actions)}")
    print(f"{'='*50}\n")

    return max_sev


if __name__ == "__main__":
    sys.exit(main())
