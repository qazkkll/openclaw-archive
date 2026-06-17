#!/usr/bin/env python3
"""夜间任务监控 - 每15分钟检查进度"""
import os, json, datetime as dt

def check_progress():
    now = dt.datetime.now()
    print(f'\n{"="*50}')
    print(f'夜间任务监控 @ {now:%H:%M:%S}')
    print(f'{"="*50}')
    
    # 1. Windows puller
    win_log = r'C:\workspace\av2\puller_log.txt'
    win_file = os.path.expanduser('~/.openclaw/workspace/av2_data') if not os.name == 'nt' else None
    
    # 2. Cloud factor scan
    scan_log = '/tmp/factor_scan.log'
    if os.path.exists(scan_log):
        with open(scan_log) as f:
            lines = f.readlines()
        last = lines[-5:] if len(lines) > 5 else lines
        print(f'\nCloud Factor Scan:')
        print(f'  Last update: {lines[-1].strip() if lines else "N/A"}')
        print(f'  Total lines: {len(lines)}')
    
    # 3. Check for output files  
    results_dir = '/tmp'
    files = [f for f in os.listdir(results_dir) if 'factor' in f.lower() and 'scan' in f.lower()]
    if files:
        print(f'  Output files: {files}')
    
    # 4. Time elapsed
    print(f'\nRunning since ~01:45, elapsed: {now - dt.datetime(now.year, now.month, now.day, 1, 45)}')

if __name__ == '__main__':
    check_progress()
