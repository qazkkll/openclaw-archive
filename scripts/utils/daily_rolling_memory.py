#!/usr/bin/env python3
"""
编译 memory/YYYY-MM-DD.md → memory/rolling_7day.md
读取最近7天的日存档（最多7个文件），按日期倒序写入 rolling_7day.md
"""
import os, glob, re, sys
from datetime import datetime, timedelta, timezone

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
ROLLING_FILE = os.path.join(MEMORY_DIR, "rolling_7day.md")
TZ = timezone(timedelta(hours=8))

def get_daily_files(days=7):
    """获取最近 N 天的 memory/YYYY-MM-DD.md 文件（不要求每天都有）"""
    today = datetime.now(TZ)
    files = []
    for i in range(days):
        d = today - timedelta(days=i)
        fname = f"{d.strftime('%Y-%m-%d')}.md"
        fpath = os.path.join(MEMORY_DIR, fname)
        if os.path.exists(fpath):
            files.append((d.strftime('%Y-%m-%d'), fpath))
    return files

def strip_memory_links(text):
    """去掉记忆skill的链接符号，保持纯文本"""
    return text

def build_rolling(files):
    """生成 rolling_7day.md 内容"""
    today = datetime.now(TZ)
    lines = []
    lines.append(f"# 7天滚动记忆")
    lines.append(f"> 追加模式 | 上次更新: {today.strftime('%Y-%m-%dT%H:%M:%S')} | 已有 {len(files)} 天")
    lines.append(f"> 存档机制（供下个session阅读）：")
    lines.append(f"> - 每次会话结束后写入 `memory/YYYY-MM-DD.md`")
    lines.append(f"> - 晨流 cron 定时编译进 `rolling_7day.md`")
    lines.append(f"> - 超过7天的内容保留在 memory/*.md 中，不自动删除")
    lines.append(f"> - AGENTS.md + SOUL.md + MEMORY.md 是长期灵魂文件")
    lines.append("")
    
    # 按日期倒序（最新的在最上面）
    files.sort(key=lambda x: x[0], reverse=True)
    
    for date_str, fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            content = f"*读取失败: {e}*"
        
        # 去掉开头的日期标题（如果有）
        content = re.sub(r'^# .*\n', '', content, count=1).strip()
        
        lines.append(f"## {date_str}")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)

def main():
    files = get_daily_files(7)
    if not files:
        print(f"[gen_rolling_memory] ⚠️ 最近7天没有 memory/*.md 文件")
        return
    
    content = build_rolling(files)
    with open(ROLLING_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[gen_rolling_memory] ✅ rolling_7day.md 已更新")
    print(f"  ├ 日期范围: {files[-1][0]} ~ {files[0][0]}")
    print(f"  ├ 覆盖天数: {len(files)}")
    print(f"  └ 文件大小: {os.path.getsize(ROLLING_FILE)/1024:.1f} KB")

if __name__ == '__main__':
    main()
