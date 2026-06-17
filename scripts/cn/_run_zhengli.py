"""
_run_zhengli.py — 存档验证脚本（轻量版）
用法：python scripts/_run_zhengli.py
输出：核心4步检查 + 顺带完整性报告
"""
import json, os, sys, datetime
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

w = Path.cwd()
today = datetime.date.today().isoformat()
ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

print(f"{'='*50}")
print(f"📋 存档验证 — {today} {ts}")
print()

# ── ⚡ Step 0: Session 预清理（防 compaction 死锁） ──
cleanup = w / 'scripts' / 'sys_session_cleanup.py'
if cleanup.exists():
    import subprocess
    sr = subprocess.run([sys.executable, str(cleanup)], capture_output=True, encoding='utf-8', timeout=120)
    if sr.stdout.strip():
        for line in sr.stdout.strip().split('\n')[-3:]:
            print(f"  [CLEANUP] {line.strip()}")
    if sr.returncode != 0:
        print(f"  [WARN] Session清理返回非0 (exit={sr.returncode})")

# ── 核心4步 ──
checks = []

# Step 1: 日记
diary = w / 'memory' / f'{today}.md'
if diary.exists():
    checks.append(("①日记", True, f"{today}.md ({diary.stat().st_size} bytes)"))
else:
    checks.append(("①日记", False, "未创建"))

# Step 2a: decision_history
dh = w / 'data' / 'decision_history.jsonl'
if dh.exists():
    lines = [l for l in dh.read_text(encoding='utf-8', errors='replace').split('\n') if l.strip() and l.strip().startswith('{')]
    today_entries = [l for l in lines if today in l]
    checks.append(("②a decision_history", True, f"{len(today_entries)}条今日"))
else:
    checks.append(("②a decision_history", False, "文件缺失"))

# Step 2b: experience_log
exp = w / 'data' / 'experience_log.jsonl'
if exp.exists():
    exp_lines = [l for l in exp.read_text(encoding='utf-8', errors='replace').split('\n') if l.strip()]
    checks.append(("②b experience_log", True, f"{len(exp_lines)}条"))
else:
    checks.append(("②b experience_log", False, "文件缺失"))

# Step 2c: bug_log
bl = w / 'bug_log.md'
if bl.exists():
    checks.append(("②c bug_log", True, f"({bl.stat().st_size} bytes)"))
else:
    checks.append(("②c bug_log", False, "文件缺失"))

# Step 3: active_mission
am = w / 'data' / 'active_mission.json'
if am.exists():
    amd = json.loads(am.read_text(encoding='utf-8'))
    checks.append(("③ active_mission", True, f"status={amd.get('status','?')}"))
else:
    checks.append(("③ active_mission", False, "文件缺失"))

# Step 4: 验证脚本（自动更新索引+生成报告）
print("  ⏳ ④ 更新索引...")
gen_index = w / 'scripts' / '_gen_index.py'
import subprocess
r = subprocess.run([sys.executable, str(gen_index)], capture_output=True, encoding='utf-8', timeout=60)
if r.returncode == 0:
    checks.append(("④验证/索引", True, "索引已更新"))
    for line in r.stdout.strip().split('\n'):
        print(f"     → {line}")
else:
    checks.append(("④验证/索引", False, f"报错: {r.stderr[:200]}"))

# ── 输出结果 ──
print()
for label, ok, detail in checks:
    icon = "✅" if ok else "❌"
    print(f"  {icon} {label}: {detail}")

print()
oks = sum(1 for _, ok, _ in checks if ok)
fails = sum(1 for _, ok, _ in checks if not ok)
print(f"{'='*50}")
print(f"📊 核心: ✅ {oks} / ❌ {fails}")
if fails == 0:
    print("✅ 存档完成")
else:
    print(f"⚠️ {fails}项未完成，下次session启动前补也行")

# ── 顺带生成存档报告 ──
report_lines = [
    f"# 存档验证报告 — {today}",
    f"生成时间: {ts}",
    "",
    "## 核心4步",
    "",
    "| # | 步骤 | 状态 | 详情 |",
    "|---|:---|---:|---|",
]
for label, ok, detail in checks:
    icon = "✅" if ok else "❌"
    report_lines.append(f"| {label} | | {icon} | {detail} |")

# 顺带检查追踪文件状态
report_lines.append("")
report_lines.append("## 顺带检查（不影响存档完成）")
tracks = []
op = w / 'data' / 'operation_plan.json'
if op.exists():
    d = json.loads(op.read_text(encoding='utf-8'))
    pending = [p for p in d.get('plans',[]) if p.get('status') in ('pending','watch')]
    tracks.append(f"operation_plan: {len(pending)}项待处理")
tr = w / 'data' / 'recommendation_tracker.json'
if tr.exists():
    d = json.loads(tr.read_text(encoding='utf-8'))
    active = [e for e in d.get('entries',[]) if e.get('status') in ('pending','active')]
    tracks.append(f"recommendation_tracker: {len(active)}项活跃")
tp = w / 'data' / 'tomorrow_plan.json'
if tp.exists():
    d = json.loads(tp.read_text(encoding='utf-8'))
    tracks.append(f"tomorrow_plan: {len(d.get('items',[]))}项")

for t in tracks:
    report_lines.append(f"- {t}")

report_lines.append("")
report_lines.append(f"---")
report_lines.append(f"*存档完成: {ts}*")

archive_path = w / 'data' / f'archive_{today}.md'
archive_path.write_text('\n'.join(report_lines), encoding='utf-8')
print(f"📦 报告: data/archive_{today}.md")
