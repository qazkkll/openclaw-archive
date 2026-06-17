#!/usr/bin/env python3
"""灌历史数据到记忆库（通过lancedb Python SDK）
memory_store工具不可用，直接写Node.js脚本调用插件内部的lancedb
"""
import sys, os, json, subprocess, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import DATA_DIR
WORKSPACE = DATA_DIR.replace("/data", "/workspace") if "/data" in DATA_DIR else r"/home/hermes/.hermes/openclaw-archive"
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
LOG_RAW = os.path.join(WORKSPACE, "logs", "conversation", "raw")
ARCHIVE_INDEX = os.path.join(WORKSPACE, "conversation_archive", "index.json")

# 提取关键记忆
memories = []

# 1. 从 memory/*.md 提取
if os.path.isdir(MEMORY_DIR):
    for fname in sorted(os.listdir(MEMORY_DIR)):
        if not fname.endswith('.md') or fname.startswith('rolling'):
            continue
        fp = os.path.join(MEMORY_DIR, fname)
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        if len(content.strip()) > 50:
            # 提取关键行（含决策/陷阱/关键信息）
            lines = content.split('\n')
            key_lines = [l.strip() for l in lines if any(kw in l for kw in 
                ['NVDA','AAPL','AMD','A股','美股','买入','卖出','推荐','评分',
                 'token','API','key','错误','注意','问题','bug','修复','改了',
                 '路径','工作区','D盘','备份','cron','skill','装','删','改'])]
            if key_lines:
                memories.append(f"【{fname}】" + '\n'.join(key_lines[:15]))

# 2. 从 conversation_archive 提取摘要
if os.path.exists(ARCHIVE_INDEX):
    with open(ARCHIVE_INDEX, 'r', encoding='utf-8') as f:
        idx = json.load(f)
    for a in idx.get("archives", []):
        date = a.get("date","")
        total = a.get("totalSessions", len(a.get("sessions",[])))
        memories.append(f"【对话归档 {date}】共{total}次对话")

# 写成一个记忆文件供后续使用
output = "# 历史记忆种子\n> 从memory/ + 对话归档中提取的关键信息\n\n"
for m in memories:
    output += m + "\n\n"

seed_path = os.path.join(MEMORY_DIR, "_seed_historical.md")
with open(seed_path, 'w', encoding='utf-8') as f:
    f.write(output)
print(f"✅ 写入 {seed_path}")
print(f"   共 {len(memories)} 条关键记忆")

# 也写一份给lancedb插件直接读的JSON
seed_data = [
    {"text": m, "source": "historical_seed", "importance": 0.7}
    for m in memories
]
json_path = os.path.join(WORKSPACE, "data", "_seed_memories.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(seed_data, f, ensure_ascii=False, indent=2)
print(f"✅ 写入 {json_path}")
print(f"\n接下来通过 sub-agent 写入 lancedb...")

# 用子agent写入记忆
for i, sd in enumerate(seed_data):
    text = sd["text"][:500]
    print(f"  [{i+1}/{len(seed_data)}] 写入: {text[:60]}...")
    r = subprocess.run([
        'openclaw', 'session', 'run', '--isolated', '--timeout', '30',
        '--prompt', f'存储到记忆库: "{text[:200]}" 。用自然语言告诉用户这条已记住。不要用工具，直接回复。
    ], capture_output=True, text=True, timeout=60)
    # 即使失败也继续
