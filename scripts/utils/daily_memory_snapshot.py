#!/usr/bin/env python3
"""
daily_memory_snapshot.py — 每日记忆快照 + 自动导入

每天05:00 HKT跑一次。输出文件为 data/daily_memory_snapshot.json，
包含最近3天的关键信息（决策、经验、评分概览），
大小控制在10KB以内，确保新session启动时自动加载不失真。

三不原则：
- 不替换 active_mission.json（避免冲突，只读）
- 不写经验日志（避免循环）
- 不做判断（只录入事实，不做建议）
"""

import json, os, sys, glob
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = os.environ.get('OPENCLAW_WORKSPACE', r'/home/hermes/.hermes/openclaw-archive')
DATA_DIR   = r'/home/hermes/.hermes/openclaw-archive/data'
OUT_PATH   = os.path.join(WORKSPACE, 'data', 'daily_memory_snapshot.json')

now = datetime.now()
today = now.strftime('%Y-%m-%d')
yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
two_days_ago = (now - timedelta(days=2)).strftime('%Y-%m-%d')
relevant_dates = [today, yesterday, two_days_ago]

snapshot = {
    'date': today,
    'version': 2,
    'summary': {},
    'decisions': [],
    'lessons': [],
    'scores_overview': [],
    'pending_sync': [],
    'open_issues': []
}

# ── 1. 最近决策 ────────────────────────────────────
dec_path = os.path.join(WORKSPACE, 'data', 'decision_history.jsonl')
if os.path.exists(dec_path):
    with open(dec_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                d = rec.get('date', '')
                if any(d.startswith(rd) for rd in relevant_dates):
                    snapshot['decisions'].append({
                        'date': d,
                        'code': rec.get('code', rec.get('stock', '')),
                        'action': rec.get('action', rec.get('recommendation', '')),
                        'reason': rec.get('reason', rec.get('rationale', ''))
                    })
            except:
                pass

# ── 2. 最近经验教训 ────────────────────────────────
exp_path = os.path.join(WORKSPACE, 'data', 'experience_log.jsonl')
if os.path.exists(exp_path):
    with open(exp_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                d = rec.get('date', '')
                if any(d.startswith(rd) for rd in relevant_dates):
                    snapshot['lessons'].append({
                        'date': d,
                        'event': rec.get('event', ''),
                        'lesson': rec.get('lesson', ''),
                        'fix': rec.get('action', rec.get('fix', ''))
                    })
            except:
                pass

# ── 3. 评分文件检查 ────────────────────────────────
for fname in os.listdir(DATA_DIR):
    for rd in relevant_dates:
        rd_compact = rd.replace('-', '')
        if (fname.startswith('fusion_rec_') and fname[11:-5] == rd):
            snapshot['scores_overview'].append({'type': '美股融合', 'date': rd, 'file': fname})
        elif (fname.startswith('a2_scored_') and fname[10:-5] == rd_compact):
            snapshot['scores_overview'].append({'type': 'A股A2', 'date': rd, 'file': fname})
        elif (fname.startswith('ld3_scored_') and fname[12:-5] == rd):
            snapshot['scores_overview'].append({'type': '蓝盾', 'date': rd, 'file': fname})

# ── 4. active_mission 中的待办 ─────────────────────
mission_path = os.path.join(WORKSPACE, 'data', 'active_mission.json')
if os.path.exists(mission_path):
    try:
        with open(mission_path, 'r', encoding='utf-8') as f:
            mission = json.load(f)
        ps = mission.get('pending_sync', [])
        if ps:
            snapshot['pending_sync'] = ps
        ki = mission.get('known_issues_20260613', {})
        for k, v in ki.items():
            if v.get('status') in ('unresolved', 'needs_discussion'):
                snapshot['open_issues'].append(k)
    except:
        pass

# ── 5. 摘要 ────────────────────────────────────────
snapshot['summary'] = {
    'decisions_today': len(snapshot['decisions']),
    'lessons_today': len(snapshot['lessons']),
    'scores_files': len(snapshot['scores_overview']),
    'pending_tasks': len(snapshot['pending_sync']),
    'open_issues': len(snapshot['open_issues'])
}

# ── 写出 ────────────────────────────────────────────
with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(snapshot, f, indent=2, ensure_ascii=False)

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f'✅ 记忆快照：{OUT_PATH} ({size_kb:.1f} KB)')
print(f'   {len(snapshot["decisions"])} 条决策, {len(snapshot["lessons"])} 条经验, {len(snapshot["scores_overview"])} 个评分文件')
if snapshot['pending_sync']:
    print(f'   ⚠️ 待同步：{len(snapshot["pending_sync"])} 条')
if snapshot['open_issues']:
    print(f'   ⚠️ 未决问题：{", ".join(snapshot["open_issues"])}')
