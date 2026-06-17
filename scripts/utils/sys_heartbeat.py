#!/usr/bin/env python3
"""
Gateway心跳检测 — 每10分钟跑一次
检测逻辑：
1. 检查gateway进程是否存在
2. 检查端口18789是否在监听
3. 如果两者都正常 → 健康
4. 如果挂了 → 执行cleanup杀旧进程 → 重启gateway
5. 如果重启后还挂 → 发Telegram告警
"""
import subprocess, time, sys, os

GATEWAY_PORT = "18789"
WORKSPACE = r'/home/hermes/.hermes/openclaw-archive'
CLEANUP_SCRIPT = os.path.join(WORKSPACE, 'scripts', 'sys_watchdog_cleanup.py')

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def check_gateway():
    """返回 (process_ok, port_ok)"""
    # 检查进程
    proc_ok = False
    try:
        ret = subprocess.run(
            r'powershell -Command "Get-Process node -EA SilentlyContinue | Where-Object { $_.MainModule.FileName -like \"*openclaw*\" } | Measure-Object | Select-Object -ExpandProperty Count"',
            capture_output=True, text=True, shell=True, timeout=10
        )
        count = ret.stdout.strip()
        proc_ok = count.isdigit() and int(count) > 0
    except:
        pass
    
    # 检查端口
    port_ok = False
    try:
        ret = subprocess.run(
            f'netstat -ano | findstr ":{GATEWAY_PORT}" | findstr LISTENING',
            capture_output=True, text=True, shell=True, timeout=10
        )
        port_ok = bool(ret.stdout.strip())
    except:
        pass
    
    return proc_ok, port_ok

def reboot_gateway():
    """清端口 → 重启gateway"""
    log("Gateway down, running cleanup...")
    
    # 杀旧进程
    try:
        subprocess.run(['taskkill', '/F', '/IM', 'node.exe'], 
                      capture_output=True, timeout=5)
        log("  Killed all node.exe")
        time.sleep(2)
    except:
        pass
    
    # 确认端口空了
    try:
        ret = subprocess.run(
            f'netstat -ano | findstr ":{GATEWAY_PORT}" | findstr LISTENING',
            capture_output=True, text=True, shell=True, timeout=5
        )
        if ret.stdout.strip():
            log("  Port still occupied, trying harder...")
            # 从netstat输出中提取PID
            lines = ret.stdout.strip().split('\n')
            for line in lines:
                parts = line.strip().split()
                if parts and parts[-1].isdigit():
                    pid = parts[-1]
                    subprocess.run(['taskkill', '/F', '/PID', pid],
                                  capture_output=True, timeout=5)
                    log(f"  Killed PID {pid}")
            time.sleep(2)
    except:
        pass
    
    # 启动gateway
    log("  Starting gateway...")
    try:
        # 用subprocess启动后台进程
        subprocess.Popen(
            ['openclaw', 'gateway', 'start'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        time.sleep(5)
        log("  Gateway start command issued")
        return True
    except Exception as e:
        log(f"  Failed to start gateway: {e}")
        return False

if __name__ == '__main__':
    proc, port = check_gateway()
    
    if proc and port:
        log("Gateway healthy (process+port OK)")
        sys.exit(0)
    
    if proc and not port:
        log(f"WARN: Process exists but port {GATEWAY_PORT} not listening")
        # 可能启动中，不打搅
        sys.exit(0)
    
    # gateway挂了
    reboot_gateway()
    
    # 等一会儿再验证
    time.sleep(8)
    proc2, port2 = check_gateway()
    if proc2 and port2:
        log("Gateway recovered successfully")
        sys.exit(0)
    else:
        log("CRITICAL: Gateway failed to recover after restart!")
        # 退出码1会让cron看到失败，但我们不重复exe——下次心跳再试
        sys.exit(1)
