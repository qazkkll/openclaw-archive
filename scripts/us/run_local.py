#!/usr/bin/env python3
"""在本地Windows上运行Python脚本"""
import subprocess, sys, json

KEY = '/home/admin/.ssh/id_ed25519'
PORT = 18792
HOST = 'admin@localhost'
WIN_WS = 'C:\\Users\\admin\\Desktop\\openclaw'

def run(script, *args):
    """同步脚本到Windows并执行"""
    # 1. Sync script
    from pathlib import Path
    local_path = Path(__file__).parent.parent / 'scripts' / script
    if not local_path.exists():
        print(f"❌ 脚本不存在: scripts/{script}")
        return None
    
    win_path = f"{WIN_WS}\\scripts\\{script}"
    scp_cmd = f"scp -P {PORT} -i {KEY} -o StrictHostKeyChecking=no {local_path} {HOST}:\"{win_path}\""
    subprocess.run(scp_cmd, shell=True, capture_output=True)
    
    # 2. Run on Windows
    ssh_cmd = f"ssh -i {KEY} -p {PORT} {HOST} \"cd {WIN_WS} && python scripts\\{script} {' '.join(args)}\""
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=300)
    
    print(result.stdout)
    if result.stderr:
        print(f"⚠️ 错误: {result.stderr[-500:]}")
    return result

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python3 run_local.py <脚本名> [参数...]")
        sys.exit(1)
    run(sys.argv[1], *sys.argv[2:])
