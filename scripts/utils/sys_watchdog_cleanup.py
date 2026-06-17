# watchdog预清理脚本 — 重启gateway前先杀干净旧进程和端口
# 用法: python sys_watchdog_cleanup.py
# 由watchdog在每次重启前调用

import os, subprocess, time, sys

def kill_nodes():
    """杀掉所有openclaw相关的node进程"""
    nodes_before = 0
    try:
        ret = subprocess.run(
            'tasklist /FI "IMAGENAME eq node.exe" /FO CSV /NH',
            capture_output=True, text=True, shell=True, timeout=5
        )
        nodes_before = len([l for l in ret.stdout.strip().split('\n') if l.strip()])
    except:
        pass
    
    if nodes_before == 0:
        print(f'[{time.strftime("%H:%M:%S")}] No node.exe running, clean')
        return True
    
    print(f'[{time.strftime("%H:%M:%S")}] Found {nodes_before} node.exe processes, killing...')
    
    # 先找openclaw相关的node进程
    ps_script = r'''
    Get-Process node -EA SilentlyContinue | Where-Object {
        $_.MainModule.FileName -like '*openclaw*'
    } | ForEach-Object { $_.Id }
    '''
    try:
        ret = subprocess.run(['powershell', '-Command', ps_script],
                           capture_output=True, text=True, timeout=10)
        pids = [p.strip() for p in ret.stdout.strip().split('\n') if p.strip()]
        if pids:
            for pid in pids:
                subprocess.run(['taskkill', '/F', '/PID', pid],
                             capture_output=True, timeout=5)
                print(f'  Killed PID {pid}')
            time.sleep(2)
    except:
        # fallback: kill all node
        subprocess.run('taskkill /F /IM node.exe', shell=True, capture_output=True, timeout=5)
        print('  Killed all node.exe (fallback)')
        time.sleep(2)
    
    # 确认端口18789已释放
    try:
        ret = subprocess.run(
            'netstat -ano | findstr ":18789" | findstr LISTENING',
            capture_output=True, text=True, shell=True, timeout=5
        )
        if ret.stdout.strip():
            print('  WARN: Port 18789 still occupied after kill')
            return False
        else:
            print(f'  Port 18789 free, ready')
            return True
    except:
        return True

if __name__ == '__main__':
    ok = kill_nodes()
    sys.exit(0 if ok else 1)
