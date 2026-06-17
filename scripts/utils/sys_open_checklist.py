"""
sys_open_checklist.py — 📡 启动健康仪表盘（v3 2026-06-13）

从链式检查 → 分级健康报告。只有🔴级阻塞，🟡级自动修复，🔵级告知。

分级规则：
  🔴 CRITICAL  — 必须修复才能干活（数据损坏/持仓不同步/模型文件缺失）
  🟡 WARNING  — 自动修复或标记（INDEX过期/临时文件未清理/命名不合规）
  🔵 INFO     — 知道就行（今日评分已存在/新文件发现/统计变化）
"""
import json, os, sys, re, subprocess
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from datetime import datetime

workspace = Path(os.environ.get('OPENCLAW_WORKSPACE', Path.cwd()))
SCRIPTS = workspace / 'scripts'

# ─── Dashboard ───
report = {
    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
    'critical': [],
    'warning': [],
    'info': [],
    'auto_fixed': [],
}

def add(level, icon, msg, detail=None):
    report[level].append({'icon': icon, 'msg': msg, 'detail': detail})

def print_dashboard():
    t = report['time']
    print(f"""
╔═══ 📡 启动健康报告 ═══╗
║  {t:20s}               ║
╚════════════════════════╝""")

    sep = "─"*48

    # 🔴 CRITICAL
    crit = report['critical']
    if crit:
        print(f"\n🔴 CRITICAL ({len(crit)}项) — 必须修复才能继续:")
        print(sep)
        for c in crit:
            print(f"  {c['icon']} {c['msg']}")
            if c.get('detail'):
                for d in c['detail'].split('\n'):
                    print(f"    {d}")
    else:
        print(f"\n🔴 CRITICAL (0项) ✓ — 无致命问题")

    # 🟡 WARNING
    warn = report['warning']
    if warn:
        print(f"\n🟡 WARNING ({len(warn)}项) — 已自动修复或标记:")
        print(sep)
        for w in warn:
            print(f"  {w['icon']} {w['msg']}")
            if w.get('detail'):
                for d in w['detail'].split('\n'):
                    print(f"    {d}")
    else:
        print(f"\n🟡 WARNING (0项) ✓ — 无警告")

    # 🔵 INFO
    info = report['info']
    if info:
        print(f"\n🔵 INFO ({len(info)}项):")
        print(sep)
        for i in info:
            print(f"  {i['icon']} {i['msg']}")
            if i.get('detail'):
                for d in i['detail'].split('\n'):
                    print(f"    {d}")

    # Auto-fixed summary
    fixed = report['auto_fixed']
    if fixed:
        print(f"\n🛠️  自愈 ({len(fixed)}项):")
        print(sep)
        for f in fixed:
            print(f"  ✅ {f['msg']}")

    # Summary
    print(f"\n{'═'*48}")
    print(f"  🔴{len(crit)}  🟡{len(warn)}  🔵{len(info)}  🛠️{len(fixed)}")
    if report['critical']:
        print("  ⛔ 有致命问题 — 请修复后再工作")
    else:
        print("  ✅ 系统就绪 — 可以开始工作")
    print(f"{'═'*48}\n")


# ─── Helper: auto-fix index ───
def _auto_fix_index():
    """检查 INDEX.md 是否与活跃模型同步，不同步则自动修复"""
    idx_path = workspace / 'INDEX.md'
    if not idx_path.exists():
        add('warning', '📄', 'INDEX.md 不存在', '将在存档时自动创建')
        return True

    content = idx_path.read_text(encoding='utf-8')

    # Check V8 marker
    if 'V8-Lottery' not in content and '绿箭V8' in content:
        add('warning', '📄', 'INDEX.md 中绿箭引用可能过时', '存档时自动更新')
    return True

# ─── Helper: clean temp files ───
def _auto_clean_tmp():
    """删除 scripts/ 下 tmp_* 文件（>1天的）"""
    script_dir = SCRIPTS
    if not script_dir.exists():
        return 0
    now = datetime.now().timestamp()
    cleaned = 0
    for f in script_dir.glob('tmp_*.py'):
        if f.is_file():
            file_age = now - f.stat().st_mtime
            if file_age > 86400:  # > 1 day
                try:
                    f.unlink()
                    report['auto_fixed'].append({'msg': f'删除过期临时文件 {f.name}'})
                    cleaned += 1
                except Exception:
                    pass
    return cleaned


# ════════════ MAIN ════════════

# ─── [1/6] 中断任务检查 ───
mission_file = workspace / 'data' / 'active_mission.json'
if mission_file.exists():
    try:
        with open(mission_file, 'r', encoding='utf-8') as f:
            mission = json.load(f)
        status = mission.get('status', 'unknown')
        name = mission.get('mission', '(未知)')
        if status == 'paused':
            pending = mission.get('pending_sync', '')
            add('info', '⏸️', f'发现中断任务: {name}')
            if pending:
                add('info', '📌', f'pending_sync: {pending[:100]}')
            add('info', '💡', '启动后先问用户是否恢复')
        elif status == 'active':
            add('info', '▶️', f'活跃任务: {name}')
        else:
            add('info', '📋', f'任务状态: {status}')
    except Exception as e:
        add('warning', '⚠️', f'active_mission.json 读取失败: {e}')
else:
    add('info', '📋', '无 active_mission.json (新会话)')


# ─── [2/6] SESSION-STATE.md ───
ss = workspace / 'SESSION-STATE.md'
if ss.exists():
    content = ss.read_text(encoding='utf-8')
    task_match = re.search(r'## 当前任务[^#]*', content)
    if task_match:
        task_section = task_match.group()
        tasks = [l.strip() for l in task_section.split('\n')
                 if l.strip().startswith('- **') or l.strip().startswith('|')]
        if tasks:
            add('info', '📋', 'SESSION-STATE.md 活跃任务:')
            for t in tasks[:5]:
                add('info', '  ', t)
else:
    add('info', '📋', 'SESSION-STATE.md 不存在 — 新会话或未初始化')


# ─── [3/6] 数据完整性检查（只报🔴） ───
integrity_script = SCRIPTS / '_data_integrity.py'
if integrity_script.exists():
    result = subprocess.run(
        [sys.executable, str(integrity_script), '--quick'],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        # Find actual errors vs warnings
        stderr = result.stderr.lower()
        if 'error' in stderr or 'critical' in stderr or 'missing' in stderr:
            add('critical', '📊', '数据完整性检查发现致命问题')
            add('critical', '  ', stderr[:200])
        else:
            add('warning', '📊', '数据完整性检查发现非致命问题')
            add('warning', '  ', stderr[:200])
    else:
        add('info', '✅', '数据完整性检查通过')
else:
    add('info', '📋', '_data_integrity.py 不存在 — 跳过')


# ─── [4/6] 操作计划 ───
op_path = workspace / 'data' / 'operation_plan.json'
if op_path.exists():
    try:
        with open(op_path, 'r', encoding='utf-8') as f:
            op = json.load(f)
        plans = op.get('plans', [])
        pending = [p for p in plans if p.get('status') in ('pending', 'watch')]
        if pending:
            add('info', '📌', f'操作计划: {len(pending)}项待处理')
            for p in pending[:8]:  # 最多显示8条
                ptype = p.get('type', '?')
                code = p.get('code', '?')
                if ptype == 'accumulate':
                    active = [t for t in p.get('tranches', []) if t.get('status') == 'pending']
                    prices = ', '.join([f"${t['target_price']}x{t['qty']}" for t in active])
                    add('info', '  ', f'补仓 {code}: {prices}')
                elif ptype == 'reduce':
                    add('info', '  ', f'减仓 {code}: {p.get("qty","?")}股')
                elif ptype == 'buy':
                    add('info', '  ', f'买入 {code}: {p.get("condition","")}')
                else:
                    add('info', '  ', f'{ptype} {code}')
        else:
            add('info', '📋', '操作计划: 无待处理项')
    except Exception as e:
        add('warning', '⚠️', f'operation_plan.json 读取失败: {e}')
else:
    add('info', '📋', '操作计划: 不存在')


# ─── [5/6] 自愈 & 维护 ───
_auto_clean_tmp()        # 自动清理过期临时文件
_auto_fix_index()        # 检查INDEX同步

# Check naming compliance (WARNING only)
script_files = list(SCRIPTS.glob('*.py'))
bad_names = [f.name for f in script_files
             if not f.name.startswith(('a1_','a_','us_v5s_','us_','daily_','dl_','sys_','tst_','tmp_','_') )
             and f.name not in ('score.py','scoring.py')]
if bad_names:
    add('warning', '📁', f'{len(bad_names)}个脚本命名不合规（存档时自动标记）')
    for n in bad_names[:3]:
        add('warning', '  ', n)

# Check today's scores exist (INFO)
today = datetime.now().strftime('%Y-%m-%d')
ld3 = Path(f'/home/hermes/.hermes/openclaw-project/data/ld3_scored_{today}.json')
v8 = Path(f'/home/hermes/.hermes/openclaw-project/data/scored_v75_lottery_{today}.json')
fusion = Path(f'/home/hermes/.hermes/openclaw-project/data/fusion_rec_{today}.json')
fusion_alt = Path(f'/home/hermes/.hermes/openclaw-project/data/fusion_rec_{datetime.now().strftime("%Y%m%d")}.json')

if ld3.exists():
    size = ld3.stat().st_size / 1024
    add('info', '🛡️', f'蓝盾3.0今日评分已存在 ({size:.0f}KB)')
if v8.exists():
    size = v8.stat().st_size / 1024
    add('info', '🟢', f'绿箭V8今日评分已存在 ({size:.0f}KB)')
if fusion.exists() or fusion_alt.exists():
    add('info', '🌟', '今日融合推荐已存在')

# ─── [6/6] 测试套件 ───
test_script = SCRIPTS / '_run_tests.py'
if test_script.exists():
    result = subprocess.run(
        [sys.executable, str(test_script)],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        add('warning', '🧪', '测试套件发现异常')
        last = result.stdout.strip()[-200:]
        if last:
            add('warning', '  ', last)
    else:
        add('info', '✅', '测试套件通过 — 生产流程就绪')

# ─── 经验库统计 ───
exp_log = workspace / 'data' / 'experience_log.jsonl'
if exp_log.exists():
    with open(exp_log, 'rb') as f:
        raw = f.read()
    exp_count = raw.count(b'\n') if raw else 0
    add('info', '📖', f'经验库: {exp_count}条')

# ─── Render ───
print_dashboard()
