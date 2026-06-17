#!/usr/bin/env python3
"""提取历史关键记忆 — 纯文件输出"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA_DIR
WORKSPACE = DATA_DIR.replace("/data", "/workspace") if "/data" in DATA_DIR else r"/home/hermes/.hermes/openclaw-archive"
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
OUTPUT = os.path.join(WORKSPACE, "memory", "_seeded.md")

KEYWORDS = [
    "NVDA", "AAPL", "AMD", "A股", "美股",  
    "买入", "卖出", "推荐", "评分", "建议",
    "token", "错误", "bug", "改了", "注意",
    "skill", "cron", "备份", "装", "删",
    "path", "路径", "工作区", "D盘", "迁移",
]

lines_out = ["# 历史记忆种子", "> 从memory/*.md中提取的关键行", "", ""]

count = 0
for fname in sorted(os.listdir(MEMORY_DIR)):
    if not fname.endswith(".md") or fname.startswith("rolling") or fname.startswith("_"):
        continue
    fp = os.path.join(MEMORY_DIR, fname)
    try:
        content = open(fp, "r", encoding="utf-8", errors="replace").read()
    except:
        continue
    if len(content.strip()) < 30:
        continue
    matched = []
    for line in content.split("\n"):
        stripped = line.strip()
        if any(kw.lower() in stripped.lower() for kw in KEYWORDS):
            matched.append(stripped)
    if matched:
        tag = fname.replace(".md", "")
        lines_out.append(f"## {tag}")
        for m in matched[:15]:
            lines_out.append(f"- {m[:150]}")
        lines_out.append("")
        count += 1

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines_out))

print(f"✅ 已提取 {count} 天的关键记忆")
print(f"   写入 {OUTPUT}")
print(f"   共 {len(lines_out)} 行")
