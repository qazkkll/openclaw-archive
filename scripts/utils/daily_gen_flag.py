#!/usr/bin/env python3
"""检查最近1小时新文件/变更，写入new_files_flag.json"""
import os, json, sys, subprocess
from datetime import datetime, timezone, timedelta

WORKSPACE = r'/home/hermes/.hermes/openclaw-archive'
TZ = timezone(timedelta(hours=8))
SINCE = 3600  # 最近1小时

now = datetime.now(TZ)
cutoff = now.timestamp() - SINCE

new_files = []

# 扫描关键目录
for root, dirs, files in os.walk(WORKSPACE):
    # 跳过隐藏目录和node_modules
    rel = os.path.relpath(root, WORKSPACE)
    if rel.startswith('.') or 'node_modules' in rel:
        continue
    for f in files:
        fp = os.path.join(root, f)
        try:
            mtime = os.path.getmtime(fp)
            if mtime > cutoff:
                relpath = os.path.relpath(fp, WORKSPACE)
                size = os.path.getsize(fp)
                new_files.append({
                    'path': relpath,
                    'mtime': datetime.fromtimestamp(mtime, TZ).isoformat(),
                    'size_kb': round(size/1024, 1)
                })
        except:
            pass

new_files.sort(key=lambda x: x['mtime'], reverse=True)

flag = {
    'generated_at': now.isoformat(),
    'since': f'{(now.timestamp()-cutoff)/60:.0f} minutes ago',
    'new_count': len(new_files),
    'new_files': new_files[:30],  # 最多30条
    'has_new': len(new_files) > 0,
}

out_path = os.path.join(WORKSPACE, 'data', 'new_files_flag.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(flag, f, ensure_ascii=False, indent=2)

print(f'new_files_flag: {len(new_files)} new/changed files in last 1h')
if new_files:
    for nf in new_files[:5]:
        print(f'  {nf["path"]} ({nf["size_kb"]}KB)')
sys.exit(0)
