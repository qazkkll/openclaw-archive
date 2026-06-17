#!/usr/bin/env python3
"""
OpenClaw 路径适配器 — Windows → Linux
将所有脚本中的 D:/openclaw/ 路径替换为 ~/.hermes/openclaw-project/
"""

import os
import re
import sys

PROJECT_DIR = os.path.expanduser("~/.hermes/openclaw-project")

# 路径映射规则
PATH_MAPPINGS = [
    # 数据目录
    (r"D:/openclaw/data/", f"{PROJECT_DIR}/data/"),
    (r"D:\\openclaw\\data\\", f"{PROJECT_DIR}/data/"),
    (r"D:/openclaw/ml/", f"{PROJECT_DIR}/scripts/system/"),
    (r"D:\\openclaw\\ml\\", f"{PROJECT_DIR}/scripts/system/"),
    
    # 模型目录
    (r"D:/openclaw/data/models/", f"{PROJECT_DIR}/models/us/"),
    (r"D:\\openclaw\\data\\models\\", f"{PROJECT_DIR}/models/us/"),
    
    # 具体文件路径
    (r"D:/openclaw/data/a_hist_10y.parquet", f"{PROJECT_DIR}/data/cn/a_hist_10y.parquet"),
    (r"D:/openclaw/data/moneyflow_core.parquet", f"{PROJECT_DIR}/data/cn/moneyflow_core.parquet"),
    (r"D:/openclaw/data/a3_moneyflow_factors.parquet", f"{PROJECT_DIR}/data/cn/a3_moneyflow_factors.parquet"),
    (r"D:/openclaw/data/us_hist_clean.parquet", f"{PROJECT_DIR}/data/us/us_hist_clean.parquet"),
    (r"D:/openclaw/data/sp500_symbols.json", f"{PROJECT_DIR}/data/config/sp500_symbols.json"),
    (r"D:/openclaw/data/quality_pool.json", f"{PROJECT_DIR}/data/config/quality_pool.json"),
    (r"D:/openclaw/data/stock_info.json", f"{PROJECT_DIR}/data/config/stock_info.json"),
    
    # 输出目录
    (r"D:/openclaw/data/ld3_scored_", f"{PROJECT_DIR}/output/ld3_scored_"),
    (r"D:/openclaw/data/scored_v75_", f"{PROJECT_DIR}/output/scored_v75_"),
    (r"D:/openclaw/data/a2_scored_", f"{PROJECT_DIR}/output/a2_scored_"),
    (r"D:/openclaw/data/us_scored_", f"{PROJECT_DIR}/output/us_scored_"),
    
    # 特征文件
    (r"D:/openclaw/sp500_feats.parquet", f"{PROJECT_DIR}/data/us/sp500_feats.parquet"),
    
    # Workspace路径
    (r"C:/Users/admin/.openclaw/workspace/", f"{PROJECT_DIR}/"),
    (r"C:\\Users\\admin\\.openclaw\\workspace\\", f"{PROJECT_DIR}/"),
    
    # tushare配置
    (r"C:\\Users\\admin\\.tushare\\tushare.cfg", os.path.expanduser("~/.tushare/tushare.cfg")),
]


def adapt_file(filepath):
    """适配单个文件的路径"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return False, str(e)
    
    original = content
    for old_path, new_path in PATH_MAPPINGS:
        content = content.replace(old_path, new_path)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "modified"
    return True, "no changes"


def main():
    scripts_dir = os.path.expanduser("~/.hermes/openclaw-project/scripts")
    modified = 0
    errors = []
    
    for root, dirs, files in os.walk(scripts_dir):
        for f in files:
            if not f.endswith('.py'):
                continue
            filepath = os.path.join(root, f)
            ok, msg = adapt_file(filepath)
            if ok and msg == "modified":
                modified += 1
                print(f"  ✓ {f}")
            elif not ok:
                errors.append((f, msg))
    
    print(f"\n适配完成: {modified} 个文件修改")
    if errors:
        print(f"错误: {len(errors)} 个")
        for f, e in errors:
            print(f"  ✗ {f}: {e}")


if __name__ == "__main__":
    main()
