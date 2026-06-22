#!/usr/bin/env python3
"""
Dashboard 自动刷新脚本
每5分钟更新：持仓 + 评分 + 看板数据
纯 Python，不需要 API token
"""
import json, os, sys, subprocess, time
from datetime import datetime

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
OUTPUT_DIR = os.path.join(ROOT, 'output')
STATE_DIR = os.path.join(OUTPUT_DIR, 'state')
PYTHON = sys.executable  # 用当前venv的python，不用系统python

os.makedirs(STATE_DIR, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def run_cmd(cmd, timeout=120):
    """运行命令，返回成功/失败"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)
        if r.returncode != 0:
            log(f"  ⚠️ cmd failed: {r.stderr[:200]}")
        return r.returncode == 0
    except Exception as e:
        log(f"  ⚠️ cmd exception: {e}")
        return False

def update_portfolio():
    """更新持仓数据 — 从OpenD一步到位"""
    log("📊 更新持仓...")
    if run_cmd([PYTHON, os.path.join(ROOT, 'scripts/sync_portfolio_from_opend.py')], timeout=30):
        log("  ✅ 持仓已从OpenD同步")
        return True
    else:
        log("  ⚠️ OpenD 持仓同步失败（可能未启动）")
        return False

def update_scores():
    """更新评分数据"""
    log("📈 更新评分...")
    
    # 蓝盾 V6
    if run_cmd([PYTHON, os.path.join(ROOT, 'scripts/us/blueshield_v6_score.py'), '--top', '15'], timeout=300):
        log("  ✅ 蓝盾V6 完成")
    
    # 绿箭 V11
    if run_cmd([PYTHON, os.path.join(ROOT, 'scripts/us/arrow_v11_score.py'), '--top', '10'], timeout=300):
        log("  ✅ 绿箭V11 完成")
    
    # 转换格式
    convert_script = '''
import json, os
ROOT = os.path.expanduser("~/.hermes/openclaw-archive/")

for src, dst in [("v6_latest.json", "shield_scores.json"), ("v11_latest.json", "arrow_scores.json")]:
    src_path = os.path.join(ROOT, "output", src)
    if os.path.exists(src_path):
        with open(src_path) as f:
            data = json.load(f)
        stocks = []
        for p in data.get("picks", []):
            stocks.append({"ticker": p["ticker"], "score": p.get("pred_rank", 0), "price": p["price"], "signal": p["signal"]})
        with open(os.path.join(ROOT, "output", dst), "w") as f:
            json.dump({"timestamp": data["timestamp"], "model": data["model"], "total": data["total"], "stocks": stocks}, f, indent=2, ensure_ascii=False)
'''
    run_cmd([PYTHON, '-c', convert_script])
    return True

def update_dashboard():
    """重新生成看板数据"""
    log("🔄 生成看板数据...")
    if run_cmd([PYTHON, os.path.join(ROOT, 'scripts/dashboard_engine.py')], timeout=60):
        log("  ✅ 看板数据已更新")
        return True
    return False

def main():
    log("=" * 50)
    log("🚀 Dashboard 自动刷新启动")
    log(f"   Python: {PYTHON}")
    log("=" * 50)
    
    while True:
        try:
            log(f"\n⏰ 开始刷新 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            
            update_portfolio()
            update_scores()
            update_dashboard()
            
            log("✅ 刷新完成，等待5分钟...")
        except Exception as e:
            log(f"❌ 错误: {e}")
        
        time.sleep(300)  # 5分钟

if __name__ == '__main__':
    main()
