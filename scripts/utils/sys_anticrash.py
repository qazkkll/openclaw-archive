#!/usr/bin/env python3
"""
Gateway 启动锁 + 防撞检测 — 安全重启屏障
==========================================
放到 gateway 重启前的预检查步骤：

用法:
  python gateway_anticrash.py check    # 检测+修复重复进程
  python gateway_anticrash.py lock     # 获取启动锁
  python gateway_anticrash.py unlock   # 释放启动锁
  python gateway_anticrash.py restart  # 完整安全重启流程

Windows 处理流程:
  lock → kill旧进程(等3s) → 启动新gateway → 等5s检查 → unlock
"""
import subprocess, os, json, re, sys, time, signal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LOCK_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gateway.lock')
LOCK_PATH = os.path.abspath(LOCK_PATH)

def get_gateway_pids():
    """找出所有 gateway 相关的 node.exe 进程"""
    try:
        r = subprocess.run(['tasklist', '/FO', 'CSV', '/NH'], capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().split('\n')
        all_nodes = []
        for line in lines:
            parts = [p.strip('" ') for p in line.split('","')]
            if len(parts) >= 2 and parts[0].lower() == 'node.exe':
                all_nodes.append(int(parts[1]))
        
        gateway_pids = []
        for pid in all_nodes:
            try:
                r2 = subprocess.run(['wmic', 'process', 'where', f'ProcessId={pid}', 'get', 'CommandLine', '/format:list'],
                                    capture_output=True, text=True, timeout=5)
                cl = r2.stdout.lower()
                if 'gateway' in cl or 'openclaw' in cl:
                    gateway_pids.append(pid)
            except:
                pass
        return gateway_pids
    except Exception as e:
        print(f"警告: 无法获取进程列表: {e}")
        return []

def kill_graceful(pid, timeout=5):
    """先优雅终止，再强制杀死"""
    try:
        subprocess.run(['taskkill', '/PID', str(pid)], capture_output=True, timeout=2)
        time.sleep(1)
    except:
        pass
    try:
        proc = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                            capture_output=True, text=True, timeout=5)
        if str(pid) in proc.stdout:
            subprocess.run(['taskkill', '/PID', str(pid), '/F'], capture_output=True, timeout=2)
            time.sleep(2)
    except:
        pass

def cmd_check():
    pids = get_gateway_pids()
    print(f"Gateway 进程: {pids} ({len(pids)}个)")
    
    if len(pids) > 1:
        print(f"\n⚠️ 发现 {len(pids)} 个 gateway 进程！")
        pids.sort()
        keep = pids[-1]
        kill_list = [p for p in pids if p != keep]
        print(f"保留: PID {keep}")
        print(f"终止: {kill_list}")
        for kpid in kill_list:
            kill_graceful(kpid)
            print(f"  ✅ 已终止 PID {kpid}")
        
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
            print("  🧹 锁文件已清除")
        
        return 1
    elif len(pids) == 1:
        print(f"✅ 正常 (PID {pids[0]})")
        return 0
    else:
        print("⚠️ 无 gateway 进程")
        return 2

def cmd_lock():
    """获取启动锁 — 如果锁已存在且进程活着，拒绝"""
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, 'r') as f:
                data = json.load(f)
            old_pid = data.get('pid', 0)
            # 检查老进程是否还活着
            r = subprocess.run(['tasklist', '/FI', f'PID eq {old_pid}', '/FO', 'CSV', '/NH'],
                             capture_output=True, text=True, timeout=5)
            if str(old_pid) in r.stdout:
                print(f"❌ 锁已存在 (PID {old_pid})，拒绝重复启动")
                return 1
            else:
                print(f"🧹 锁残留 (PID {old_pid}已死)，清理后继续")
                os.remove(LOCK_PATH)
        except:
            os.remove(LOCK_PATH)
    
    lock = {
        'pid': os.getpid(),
        'time': time.time(),
        'host': os.environ.get('COMPUTERNAME', 'unknown')
    }
    with open(LOCK_PATH, 'w') as f:
        json.dump(lock, f)
    print(f"🔒 启动锁已获取 (PID {lock['pid']})")
    return 0

def cmd_unlock():
    """释放锁"""
    if os.path.exists(LOCK_PATH):
        try:
            with open(LOCK_PATH, 'r') as f:
                data = json.load(f)
            if data.get('pid') == os.getpid() or True:  # True 表示强制释放
                os.remove(LOCK_PATH)
                print("🔓 启动锁已释放")
                return 0
        except:
            os.remove(LOCK_PATH)
            print("🔓 启动锁(异常)已清除")
            return 0
    print("🔓 无锁文件")
    return 0

def cmd_restart():
    """全自动安全重启 — 锁+杀+启动+验证"""
    print("🛡️ Gateway 安全重启流程")
    print("=" * 40)
    
    # 1. 锁
    if cmd_lock() != 0:
        print("❌ 无法获取锁，放弃重启")
        return 1
    
    # 2. 杀旧进程
    pids = get_gateway_pids()
    for pid in pids:
        print(f"  终止旧进程: PID {pid}")
        kill_graceful(pid)
    
    # 3. 启动
    print(f"  启动新 gateway...")
    os.chdir(os.path.expanduser("~"))
    
    node_path = "C:\\Program Files\\nodejs\\node.exe"
    openclaw_path = "C:\\Users\\admin\\AppData\\Roaming\\npm\\node_modules\\openclaw\\bin\\openclaw.js"
    log_path = os.path.expanduser("~/.openclaw/gateway.log")
    
    cmd = f'start /B "" "{node_path}" "{openclaw_path}" gateway start > "{log_path}" 2>&1'
    subprocess.run(cmd, shell=True, timeout=5)
    
    # 4. 等待+验证
    print(f"  等待启动 (5s)...")
    time.sleep(5)
    
    new_pids = get_gateway_pids()
    if len(new_pids) >= 1:
        print(f"✅ Gateway 已启动 (PID {new_pids})")
    else:
        print(f"⚠️ 未检测到新进程，可能正在启动")
    
    # 5. 释放锁
    cmd_unlock()
    return 0

if __name__ == '__main__':
    args = sys.argv[1:]
    cmd = args[0] if args else 'check'
    
    if cmd == 'check':
        sys.exit(cmd_check())
    elif cmd == 'lock':
        sys.exit(cmd_lock())
    elif cmd == 'unlock':
        sys.exit(cmd_unlock())
    elif cmd == 'restart':
        sys.exit(cmd_restart())
    else:
        print(f"未知命令: {cmd}")
        print("用法: check | lock | unlock | restart")
        sys.exit(1)
