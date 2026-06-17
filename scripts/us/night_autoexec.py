#!/usr/bin/env python3
"""
Windows自动执行器 - 数据就绪后自动启动暴力破解
每2分钟检查一次数据状态，就绪后开始
"""
import os, sys, json, time, subprocess, datetime as dt

WORKDIR = r'C:\workspace\av2'

def log(msg):
    print(f'[{dt.datetime.now():%H:%M:%S}] {msg}', flush=True)

def check_ready():
    required = ['daily_basic.parquet', 'daily_ohlcv.parquet', 'moneyflow_hsgt.parquet']
    existing = os.listdir(WORKDIR)
    
    missing = [f for f in required if f not in existing]
    if missing:
        log(f'Waiting for: {missing}')
        return False
    
    # Check non-zero sizes
    for f in required:
        path = os.path.join(WORKDIR, f)
        if os.path.getsize(path) < 1000:
            log(f'{f} too small, still writing')
            return False
    
    log('All data files ready!')
    return True

def check_free_memory():
    """Check if we have enough free memory for parallel processing"""
    import ctypes
    kernel32 = ctypes.windll.kernel32
    state = kernel32.GlobalMemoryStatusEx(ctypes.create_string_buffer(128))
    # Simplified - just return true
    return True

def run_bruteforce():
    log('Launching brute-force...')
    cmd = [
        r'C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe',
        os.path.join(WORKDIR, 'night_bruteforce.py')
    ]
    subprocess.Popen(cmd, cwd=WORKDIR, creationflags=subprocess.CREATE_NO_WINDOW)

def main():
    log('Auto-executor started')
    log(f'Checking {WORKDIR}')
    
    max_checks = 60  # 2 hours max
    for i in range(max_checks):
        if check_ready():
            log('Starting brute-force...')
            run_bruteforce()
            log('Brute-force launched successfully')
            return
        
        time.sleep(120)  # Check every 2 minutes
    
    log('ERROR: Data not ready after 2 hours')

if __name__ == '__main__':
    main()
