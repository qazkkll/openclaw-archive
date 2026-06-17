#!/usr/bin/env python3
"""开机后拉云端反馈+新文件 — 失败优雅处理，永远exit 0"""
import subprocess, os, json, sys
from datetime import datetime

WORKSPACE = r'/home/hermes/.hermes/openclaw-archive'
SSH_KEY = r'C:\Users\admin\.ssh\id_ed25519'
CLOUD_HOST = 'admin@8.217.51.136'
DEST = os.path.join(WORKSPACE, 'data', 'cloud_feedback_backup.json')
MOM_DIR = os.path.join(WORKSPACE, 'data')

def scp_pull(remote_path, local_path):
    try:
        r = subprocess.run([
            'scp', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=10',
            f'{CLOUD_HOST}:{remote_path}', local_path
        ], capture_output=True, text=True, timeout=15)
        return r.returncode == 0
    except Exception as e:
        print(f'scp exception: {e}')
        return False

results = []

# 1. 拉cloud_feedback.json
ok1 = scp_pull('cloud_feedback.json', DEST)
if not ok1:
    # 写空备份，确保文件存在
    with open(DEST, 'w', encoding='utf-8') as f:
        json.dump({"note": "no cloud data", "pulled_at": datetime.now().isoformat()}, f)
    print(f'cloud_feedback: not found or fail, wrote empty backup')
else:
    print(f'cloud_feedback: ok')
results.append(f'cloud_feedback: {"ok" if ok1 else "empty fallback"}')

# 2. 拉mom_portfolio.md
mom_path = os.path.join(MOM_DIR, 'mom_portfolio.md')
ok2 = scp_pull('mom_portfolio.md', mom_path)
if not ok2 and not os.path.exists(mom_path):
    with open(mom_path, 'w', encoding='utf-8') as f:
        f.write(f'# mom_portfolio\n\n*no sync at {datetime.now().strftime("%Y-%m-%d %H:%M")}*\n')
    print(f'mom_portfolio: not found or fail, wrote placeholder')
results.append(f'mom_portfolio: {"ok" if ok2 else "placeholder"}')

# 3. 拉白天新写的备份包
try:
    r = subprocess.run([
        'ssh', '-i', SSH_KEY, '-o', 'StrictHostKeyChecking=no',
        '-o', 'ConnectTimeout=10', CLOUD_HOST,
        'ls -t backup/*.zip 2>/dev/null | head -2'
    ], capture_output=True, text=True, timeout=10)

    if r.returncode == 0 and r.stdout.strip():
        for f in r.stdout.strip().split('\n'):
            local = os.path.join(WORKSPACE, 'backup', os.path.basename(f))
            ok = scp_pull(f, local)
            results.append(f'backup {os.path.basename(f)}: {"ok" if ok else "fail"}')
    else:
        print('no new backup packages')
        results.append('backup: none')
except Exception as e:
    print(f'backup check exception: {e}')
    results.append(f'backup: error ({e})')

summary = ' | '.join(results)
print(f'Pull complete at {datetime.now().strftime("%H:%M")}: {summary}')

# 写pull标记
flag = os.path.join(WORKSPACE, 'data', 'cloud_pull_flag.json')
with open(flag, 'w', encoding='utf-8') as f:
    json.dump({
        "pulled_at": datetime.now().isoformat(),
        "results": results,
        "any_ok": ok1 or ok2
    }, f)

sys.exit(0)  # 永远退出0，不阻断流程
