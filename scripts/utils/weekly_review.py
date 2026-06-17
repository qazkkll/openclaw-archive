#!/usr/bin/env python3
"""
weekly_review.py — 每周模型表现与经验自动总结

目标：每周五跑一次，把这一周的模型评分、推荐记录、经验教训
     汇总成一篇"新session醒来第一读"的快照。
     让Andy（和我下次醒来）不用翻6天对话记录。
"""

import json, os, sys
from datetime import datetime, timedelta
sys.stdout.reconfigure(encoding='utf-8')

# ── 路径 ──────────────────────────────────────────
WORKSPACE = os.environ.get('OPENCLAW_WORKSPACE', r'/home/hermes/.hermes/openclaw-archive')
DATA_DIR   = r'/home/hermes/.hermes/openclaw-archive/data'
MEMORY_DIR = os.path.join(WORKSPACE, 'memory')
DEST_DIR   = os.path.join(MEMORY_DIR, 'weekly-review')
os.makedirs(DEST_DIR, exist_ok=True)

# 当前周
now = datetime.now()
week_start = now - timedelta(days=now.weekday())
iso_year, iso_week, _ = now.isocalendar()
dest_file = os.path.join(DEST_DIR, f'{now.year}-W{iso_week:02d}.md')

# ── 1. 读经验日志 ──────────────────────────────────
exp_path = os.path.join(WORKSPACE, 'data', 'experience_log.jsonl')
lessons = []
if os.path.exists(exp_path):
    with open(exp_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                rec_date = rec.get('date', '')
                if rec_date and week_start.strftime('%Y-%m-%d') <= rec_date:
                    lessons.append(rec)
            except:
                pass

# ── 2. 读决策历史 ──────────────────────────────────
dec_path = os.path.join(WORKSPACE, 'data', 'decision_history.jsonl')
decisions = []
if os.path.exists(dec_path):
    with open(dec_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
                rec_date = rec.get('date', '')
                if rec_date and week_start.strftime('%Y-%m-%d') <= rec_date:
                    decisions.append(rec)
            except:
                pass

# ── 3. 扫本周评分输出 ──────────────────────────────
scores_found = []
for fname in os.listdir(DATA_DIR):
    # A2: a2_scored_YYYYMMDD.json
    # 绿箭: scored_v75_lottery_YYYY-MM-DD.json
    # 蓝盾: ld3_scored_YYYY-MM-DD.json
    # 融合: fusion_rec_YYYY-MM-DD.json
    date_str = ''
    if fname.startswith('a2_scored_') and fname.endswith('.json'):
        date_str = fname[10:-5]
        try:
            date_str_iso = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}'
            if date_str_iso >= week_start.strftime('%Y-%m-%d'):
                scores_found.append(('A2', date_str_iso, fname))
        except:
            pass
    elif fname.startswith('scored_v75_lottery_') and fname.endswith('.json'):
        date_str = fname[20:-5]
        if date_str >= week_start.strftime('%Y-%m-%d'):
            scores_found.append(('绿箭V8', date_str, fname))
    elif fname.startswith('ld3_scored_') and fname.endswith('.json'):
        date_str = fname[12:-5]
        if date_str >= week_start.strftime('%Y-%m-%d'):
            scores_found.append(('蓝盾', date_str, fname))
    elif fname.startswith('fusion_rec_') and fname.endswith('.json'):
        date_str = fname[11:-5]
        if date_str >= week_start.strftime('%Y-%m-%d'):
            scores_found.append(('融合推荐', date_str, fname))

# 尝试读融合推荐找命中记录
hits = []
for s_type, s_date, s_fname in scores_found:
    if s_type == '融合推荐':
        try:
            with open(os.path.join(DATA_DIR, s_fname), 'r', encoding='utf-8') as f:
                rec = json.load(f)
            top5 = rec.get('v8_top5', rec.get('green_arrow_top5', []))
            for item in top5[:5]:
                code = item.get('code', item.get('ticker', '?'))
                score = item.get('score', '?')
                verdict = item.get('judgment', item.get('verdict', ''))
                hits.append(f'  - {code} (评分{score}) | {verdict}')
        except:
            pass

# ── 4. 写周报 ──────────────────────────────────────
lines = []
lines.append(f'# 周报：{now.year}-W{iso_week:02d}（{week_start.strftime("%m/%d")} - {now.strftime("%m/%d")}）')
lines.append(f'生成时间：{now.strftime("%Y-%m-%d %H:%M")}')
lines.append('')
lines.append('---')
lines.append('')

# 4a. 评分覆盖
lines.append('## 评分运行情况')
lines.append('')
if scores_found:
    lines.append(f'本周共发现 {len(scores_found)} 个评分输出文件：')
    lines.append('')
    latest_by_type = {}
    for s_type, s_date, s_fname in scores_found:
        lines.append(f'- {s_type} | {s_date} → {s_fname}')
        if s_type not in latest_by_type or s_date > latest_by_type[s_type]['date']:
            latest_by_type[s_type] = {'date': s_date, 'fname': s_fname}
    lines.append('')
    lines.append('评分覆盖检查：')
    for s_type, info in sorted(latest_by_type.items()):
        days_old = (now - datetime.strptime(info['date'], '%Y-%m-%d')).days if info['date'] else 999
        status = '✅' if days_old <= 2 else '⚠️' if days_old <= 5 else '❌'
        lines.append(f'- {status} {s_type}: 最新 {info["date"]}（{days_old}天前）')
else:
    lines.append('⚠️ 本周未发现评分输出文件（可能周一还没开盘，或Cron未执行）')
lines.append('')

# 4b. 融合推荐摘要
lines.append('## 融合推荐记录')
lines.append('')
if hits:
    lines.append(f'本周共 {len(hits)} 条主动推荐（只列前5）：')
    lines.append('')
    lines.extend(hits)
else:
    lines.append('本周未产生融合推荐记录。')
lines.append('')

# 4c. 经验教训
lines.append('## 经验教训汇总')
lines.append('')
if lessons:
    for l in lessons:
        lines.append(f'- [{l.get("date","?")}] {l.get("event","")}')
        lines.append(f'  教训：{l.get("lesson","")}')
        lines.append(f'  措施：{l.get("action",l.get("fix",""))}')
        lines.append('')
else:
    lines.append('本周无新增经验教训。')
lines.append('')

# 4d. 决策记录
lines.append('## 投资决策记录')
lines.append('')
if decisions:
    for d in decisions:
        action = d.get('action', d.get('recommendation', ''))
        code = d.get('code', d.get('stock', ''))
        lines.append(f'- [{d.get("date","?")}] {code} → {action}')
        lines.append(f'  理由：{d.get("reason", d.get("rationale", ""))}')
        lines.append('')
else:
    lines.append('本周无新增决策记录。')
lines.append('')

# 4e. 未完成的待办
lines.append('## 未完成待办')
lines.append('')
mission_path = os.path.join(WORKSPACE, 'data', 'active_mission.json')
if os.path.exists(mission_path):
    with open(mission_path, 'r', encoding='utf-8') as f:
        mission = json.load(f)
    lines.append(f'当前mission状态：{mission.get("status","?")}')
    lines.append(f'mission标题：{mission.get("mission","")}')
    ps = mission.get('pending_sync', [])
    if ps:
        lines.append(f'待同步事项（{len(ps)}条）：')
        for p in ps:
            lines.append(f'  - {p}')
    ki = mission.get('known_issues_20260613', {})
    if ki:
        lines.append('已知未决问题：')
        for k, v in ki.items():
            lines.append(f'  - {k}: {v.get("status","?")}')
    lines.append('')

lines.append('---')
lines.append(f'自动生成：weekly_review.py | 下次cron：next Friday')

report = '\n'.join(lines)
with open(dest_file, 'w', encoding='utf-8') as f:
    f.write(report)

print(f'✅ 周报已生成：{dest_file}')
print(f'   {len(lines)} 行，{len(lessons)} 条经验，{len(decisions)} 条决策，{len(scores_found)} 个评分文件')
