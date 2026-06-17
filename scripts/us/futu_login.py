#!/usr/bin/env python3
"""富途 OpenD 一键登录 + 验证码提交"""
import subprocess, time, sys, os, signal

OPEND_DIR = "/home/admin/Futu_OpenD_10.5.6508_Ubuntu18.04/Futu_OpenD_10.5.6508_Ubuntu18.04"
OPEND_BIN = f"{OPEND_DIR}/FutuOpenD"

def start_opend():
    """后台启动 OpenD"""
    proc = subprocess.Popen(
        [OPEND_BIN, "--console=1", "--no_monitor=1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=OPEND_DIR
    )
    return proc

def wait_for_prompt(proc, timeout=30):
    """等待 OpenD 输出 'req_phone_verify_code'"""
    start = time.time()
    output = b""
    while time.time() - start < timeout:
        try:
            chunk = proc.stdout.read1(1024)
            if chunk:
                output += chunk
                # 只要不崩溃就继续
        except:
            pass
        if b"req_phone_verify_code" in output:
            return True
        time.sleep(1)
    return False

def submit_code(proc, code):
    """提交验证码"""
    proc.stdin.write(f"{code}\n".encode())
    proc.stdin.flush()
    time.sleep(5)
    return proc.poll() is None  # 仍在运行 = 登录成功

if __name__ == "__main__":
    # 先杀旧进程
    subprocess.run(["pkill", "-9", "-f", "FutuOpenD"], capture_output=True)
    time.sleep(1)
    
    print("启动 OpenD...")
    proc = start_opend()
    time.sleep(5)
    
    if wait_for_prompt(proc, timeout=25):
        print("需要验证码，请输入：", end="", flush=True)
        code = sys.stdin.readline().strip()
        if submit_code(proc, code):
            print("✅ 登录成功！OpenD 运行中")
            # 切换回 systemd 服务
            subprocess.run(["sudo", "systemctl", "restart", "futu-opend"])
            print("已切换至 systemd 守护")
        else:
            print("❌ 登录失败")
    else:
        print("⏰ 超时，未检测到验证码请求")
        proc.kill()
