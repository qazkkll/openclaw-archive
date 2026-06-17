#!/usr/bin/env python3
"""
统一评分路由 — V5-S单模型

2026-06-08 更新：
- 不再调用v5m/v5l/v5_combined
- 所有评分改为V5-S单模型 + 硬过滤（删除仙股/低流动性/超低波动/短数据）
- 统一走 us_s1_scan.py
"""

import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE = "/home/hermes/.hermes/openclaw-archive"
BASE = os.path.join(WORKSPACE, "data")

def us_v5_single():
    """美股V5-S单模型扫描（调用us_s1_scan）"""
    from scripts.us_v5s_s1_scan import main as scan_main
    # 直接执行us_s1_scan
    exec(open(os.path.join(WORKSPACE, "scripts", "us_v5s_s1_scan.py")).read())

def us_load_scored():
    """加载us_scored.json（兼容旧数据）"""
    p = os.path.join(BASE, "us_scored.json")
    if os.path.exists(p):
        return json.load(open(p, "r"))
    return {}

def us_load_training_pool():
    """加载训练池"""
    p = os.path.join(BASE, "us_training_pool.json")
    if os.path.exists(p):
        return json.load(open(p, "r"))
    return None

VERSION = "V5-S (2026-06-08)"

if __name__ == "__main__":
    print(f"统一评分路由 v2.0 — {VERSION}")
    print("直接运行 us_s1_scan.py 进行扫描")
    us_v5_single()
