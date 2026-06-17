#!/usr/bin/env python3
"""双模式晨扫
模式A: 本地Windows在线 → 同步脚本过去执行，取回结果
模式B: 本地离线 → 云端执行
"""
import subprocess, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "/home/admin/.ssh/id_ed25519"
PORT = "18792"
HOST = "admin@localhost"
WIN_WS = r"C:\Users\admin\Desktop\openclaw"

def _ssh(cmd, timeout=10):
    full = f"ssh -i {KEY} -p {PORT} -o ConnectTimeout=3 -o StrictHostKeyChecking=no {HOST} \"{cmd}\""
    r = subprocess.run(full, shell=True, capture_output=False, timeout=timeout)
    # 不回传stdout，直接在终端显示
    return r

def _scp(src, dst, timeout=15):
    full = f"scp -P {PORT} -i {KEY} -o StrictHostKeyChecking=no {src} {HOST}:\"{dst}\""
    return subprocess.run(full, shell=True, capture_output=True, text=True, timeout=timeout)

def local_online():
    try:
        r = _ssh("echo 1", timeout=5)
        return r.returncode == 0
    except:
        return False

def run_local():
    print("🖥️ 本地Windows在线 → 推过去跑")
    
    # 1. 创建无中文的数据/配置
    os.makedirs(f"{ROOT}/tmp_ascii", exist_ok=True)
    for src, dst in [
        (f"{ROOT}/config/data_sources.json", "config/data_sources.json"),
        (f"{ROOT}/config/strategy.json", "config/strategy.json"),
    ]:
        data = json.load(open(src, encoding='utf-8'))
        json.dump(data, open(f"{ROOT}/tmp_ascii/{os.path.basename(dst)}", 'w'), ensure_ascii=True)
    
    # 2. 同步
    _ssh(f"if not exist {WIN_WS}\\scripts mkdir {WIN_WS}\\scripts", timeout=5)
    _ssh(f"if not exist {WIN_WS}\\data mkdir {WIN_WS}\\data", timeout=5)
    _ssh(f"if not exist {WIN_WS}\\config mkdir {WIN_WS}\\config", timeout=5)
    
    for f in ["data_source.py", "score_engine.py", "remote_refresh.py"]:
        _scp(f"{ROOT}/scripts/{f}", f"{WIN_WS}\\scripts\\{f}")
    
    for f in ["data_sources.json", "tushare.json"]:
        _scp(f"{ROOT}/tmp_ascii/{f}" if f == "data_sources.json" else f"{ROOT}/config/{f}", f"{WIN_WS}\\config\\{f}")
    
    _scp(f"{ROOT}/data/quality_pool.json", f"{WIN_WS}\\data\\quality_pool.json")
    
    # 3. 执行（纯独立脚本，无notify/audit）
    t0 = time.time()
    r = _ssh(f'cd {WIN_WS} && python scripts/remote_refresh.py', timeout=300)
    elapsed = time.time() - t0
    
    print(r.stdout[-500:] if r.stdout else "")
    if r.returncode != 0:
        print(f"⚠️ 本地执行返回码={r.returncode}")
        if r.stderr:
            print(f"错误: {r.stderr[-300:]}")
    
    # 4. 取回结果
    _scp(f"{HOST}:{WIN_WS}\\data\\morning_top100.json", f"{ROOT}/data/morning_top100.json", timeout=10)
    
    result_file = f"{ROOT}/data/morning_top100.json"
    if os.path.exists(result_file):
        size = os.path.getsize(result_file)
        print(f"✅ 本地完成 ({elapsed:.0f}s), 结果文件 {size} bytes")
    else:
        print(f"⚠️ 结果文件未取回, 用时{elapsed:.0f}s")

def run_cloud():
    print("☁️ 本地离线 → 云端执行")
    os.chdir(ROOT)
    t0 = time.time()
    r = subprocess.run(["python3", "scripts/A_refresh_top100.py"], capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    print(r.stdout[-500:] if r.stdout else "")
    print(f"✅ 云端完成 ({elapsed:.0f}s)")

if __name__ == '__main__':
    if local_online():
        run_local()
    else:
        run_cloud()
