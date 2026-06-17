#!/usr/bin/env python3
"""
🦐 小钳 × 富途OpenAPI 桥接脚本
FutuOpenD 管理 + 下单/查询接口
"""
import sys, os, json, subprocess, time, threading
from datetime import datetime
from pathlib import Path

FUTU_DIR = Path('/home/admin/.openclaw/futu')
OPEND_DIR = FUTU_DIR / 'Futu_OpenD_10.6.6608_Ubuntu18.04/Futu_OpenD_10.6.6608_Ubuntu18.04'
OPEND_BIN = OPEND_DIR / 'FutuOpenD'
OPEND_XML = OPEND_DIR / 'FutuOpenD.xml'
FUTU_LOG = Path('/home/admin/.openclaw/workspace/logs/futu_opend.log')

def status():
    """检查FutuOpenD运行状态"""
    result = subprocess.run(
        ['ps', 'aux'], capture_output=True, text=True
    )
    running = 'FutuOpenD' in result.stdout and 'grep' not in result.stdout.split('\n')[-2] if result.stdout else False
    
    # 检查端口
    port_check = subprocess.run(
        ['ss', '-tlnp'], capture_output=True, text=True
    )
    port_open = '11111' in port_check.stdout
    
    return {'running': running, 'port_open': port_open}

def start():
    """启动FutuOpenD"""
    s = status()
    if s['running']:
        return "FutuOpenD 已在运行中"
    
    log_file = open(FUTU_LOG, 'a')
    proc = subprocess.Popen(
        [str(OPEND_BIN), '-cfg_file', str(OPEND_XML)],
        cwd=str(OPEND_DIR),
        stdout=log_file, stderr=log_file,
        env={**os.environ, 'LD_LIBRARY_PATH': str(OPEND_DIR)}
    )
    
    time.sleep(3)
    s2 = status()
    if s2['running']:
        return f"✅ FutuOpenD 已启动 (PID {proc.pid})"
    else:
        return "❌ FutuOpenD 启动失败，查看日志"

def stop():
    """停止FutuOpenD"""
    subprocess.run(['pkill', '-f', 'FutuOpenD'], capture_output=True)
    time.sleep(1)
    s = status()
    return "✅ FutuOpenD 已停止" if not s['running'] else "❌ 停止失败"

def tail_log(lines=20):
    """查看日志"""
    if not FUTU_LOG.exists():
        return "日志文件不存在"
    result = subprocess.run(['tail', '-n', str(lines), str(FUTU_LOG)], capture_output=True, text=True)
    return result.stdout or "日志为空"

if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'start':
        print(start())
    elif cmd == 'stop':
        print(stop())
    elif cmd == 'restart':
        print(stop())
        time.sleep(2)
        print(start())
    elif cmd == 'log':
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(tail_log(lines))
    else:
        s = status()
        print(f"FutuOpenD: {'🟢 运行中' if s['running'] else '🔴 已停止'}")
        print(f"端口11111: {'🟢 开放' if s['port_open'] else '🔴 未监听'}")
