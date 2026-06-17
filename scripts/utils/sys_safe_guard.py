"""
Windows版金刚罩 — 安全重启+健康巡检+配置备份
用法: python safe_guard.py [restart|health|backup]
"""
import sys, os, json, time, shutil
from datetime import datetime

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
BACKUP_DIR = os.path.join(WORKSPACE, 'backup')
CONFIG = os.path.expanduser(r'~\.openclaw\openclaw.json')
GUARD_LOG = os.path.join(WORKSPACE, 'logs', 'guard.log')

def log(msg):
    os.makedirs(os.path.dirname(GUARD_LOG), exist_ok=True)
    with open(GUARD_LOG, 'a') as f:
        f.write(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}\n')
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)

def backup():
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_config = os.path.join(BACKUP_DIR, f'openclaw_{ts}.json')
    backup_agents = os.path.join(BACKUP_DIR, f'AGENTS_{ts}.md')
    backup_memory = os.path.join(BACKUP_DIR, f'MEMORY_{ts}.md')
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    files = [(CONFIG, backup_config), (os.path.join(WORKSPACE, 'AGENTS.md'), backup_agents),
             (os.path.join(WORKSPACE, 'MEMORY.md'), backup_memory)]
    for src, dst in files:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            log(f'Backup: {os.path.basename(dst)}')
    return ts

def safe_restart():
    log('=== SAFE RESTART ===')
    # 1. 验证配置
    r = os.system('openclaw config validate >nul 2>&1')
    if r != 0:
        log('Config validation FAILED, aborting restart')
        return False
    log('Config valid')
    # 2. 备份
    ts = backup()
    log(f'Backup snapshot {ts}')
    # 3. 重启
    log('Restarting gateway...')
    r = os.system('start /min cmd /c "openclaw gateway restart & pause"')
    # 4. 验证
    time.sleep(5)
    status = os.popen('openclaw gateway status 2>&1').read()
    if 'running' in status.lower() or 'ready' in status.lower():
        log('Restart OK')
        return True
    else:
        log('Restart FAILED, manual rollback may be needed')
        return False

def health():
    log('=== HEALTH CHECK ===')
    checks = []
    # Gateway
    import subprocess
    try:
        s = subprocess.check_output('openclaw gateway status 2>&1', shell=True, encoding='utf-8', errors='replace')
        gw = 'running' in s.lower()
    except Exception:
        gw = False
    checks.append(('Gateway', 'OK' if gw else 'FAIL'))
    # Cron — check at least one enabled job is scheduled
    try:
        c = subprocess.check_output('openclaw cron list 2>&1', shell=True, encoding='utf-8', errors='replace')
        lines = [l for l in c.split('\n') if l.strip()]
        cron_ok = len(lines) > 4  # header + separator + at least 1 job
    except Exception:
        cron_ok = False
    checks.append(('Cron', 'OK' if cron_ok else 'FAIL'))
    # OpenD
    import socket
    op = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    op_r = op.connect_ex(('127.0.0.1', 11111))
    op.close()
    checks.append(('FutuOpenD', 'OK' if op_r == 0 else 'DOWN'))
    
    for name, status in checks:
        log(f'  {name}: {status}')
    return all(s == 'OK' for _, s in checks)

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'health'
    if cmd == 'restart':
        safe_restart()
    elif cmd == 'health':
        health()
    elif cmd == 'backup':
        backup()
    else:
        print(f'Usage: python safe_guard.py [restart|health|backup]')
