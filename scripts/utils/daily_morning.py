#!/usr/bin/env python3
"""
📡 晨流 — 08:00 自动运行

当前架构 (2026-06-13):
  🟢 绿箭 V8-Lottery — $1-10彩票爆发预测
  🛡️ 蓝盾 3.0 — 大盘技术评分
  📈 A1资金流模型 — A股评分

流程:
  1. 清晨摘要 (morning_summary.json)
  2. 新文件检查 (new_files_flag.json)
  3. 美股持仓同步
  4. 美股评分检查 + 昨日结果摘要
  5. A1每日报告
  6. Layer1宏观信号
  7. 滚动记忆生成
"""
import os, sys, subprocess, json
from datetime import datetime, timezone, timedelta

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = r'/home/hermes/.hermes/openclaw-archive/data'
TZ = timezone(timedelta(hours=8))

def run_script(name, args=None):
    """执行 scripts/ 下的 Python 脚本"""
    script = os.path.join(WORKSPACE, "scripts", name)
    if not os.path.exists(script):
        print(f"  ⚠️  跳过: {name} (不存在)")
        return False
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=300)
        out = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        err = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        if result.returncode == 0:
            print(f"  ✅ {name} 完成")
            if out.strip():
                for line in out.strip().split('\n')[-3:]:
                    print(f"     {line}")
            return True
        else:
            print(f"  ❌ {name} 失败 (rc={result.returncode})")
            if err.strip():
                for line in err.strip().split('\n')[-3:]:
                    print(f"     {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ⏰ {name} 超时(300s)")
        return False
    except Exception as e:
        print(f"  ❌ {name} 异常: {e}")
        return False

def check_file(path_or_rel, desc):
    """检查文件状态（支持绝对路径和相对路径）"""
    if os.path.isabs(path_or_rel):
        fpath = path_or_rel
    else:
        fpath = os.path.join(WORKSPACE, path_or_rel)
        if not os.path.exists(fpath):
            fpath = os.path.join(DATA_DIR, path_or_rel)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath), tz=TZ)
        age_h = (datetime.now(TZ) - mtime).total_seconds() / 3600
        print(f"  📄 {desc}: {size/1024:.1f}KB (更新于{age_h:.1f}h前)")
        return True
    else:
        print(f"  ⚠️  {desc}: 文件不存在")
        return False

def check_us_scores():
    """检查美股昨日评分结果（蓝盾3.0 + 绿箭V8）"""
    today = datetime.now(TZ).strftime('%Y-%m-%d')
    yesterday = (datetime.now(TZ) - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print()
    print("【美股评分状态】")
    
    for label, fname in [('🛡️ 蓝盾3.0', f'ld3_scored_{today}.json'),
                         ('🛡️ 蓝盾3.0(昨日)', f'ld3_scored_{yesterday}.json'),
                         ('🟢 绿箭V8', f'v75_scored_{today}.json'),
                         ('🟢 绿箭V8(昨日)', f'v75_scored_{yesterday}.json')]:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and 'scores' in raw:
                    scores = raw['scores']
                else:
                    scores = raw if isinstance(raw, list) else []
                total = len(scores) if scores else 0
                top = scores[:3] if scores else []
                top_str = ', '.join([f"{s.get('code','?')}={s.get('score','?')}" for s in top])
                print(f"  {label}: {fname} | {total}只 | Top3: {top_str}")
            except Exception as e:
                print(f"  {label}: {fname} ⚠️ 读取出错: {e}")
        else:
            print(f"  {label}: {fname} ❌ 文件不存在")

def check_pending_operations():
    """检查待执行操作"""
    print()
    print("【待处理任务】")
    for fname in ['operation_plan.json', 'tomorrow_plan.json', 'active_mission.json']:
        fpath = os.path.join(WORKSPACE, 'data', fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                if isinstance(content, dict):
                    status = content.get('status', content.get('mode', '?'))

                    tasks = content.get('tasks', content.get('steps', []))
                    pending = sum(1 for t in tasks if t.get('status') in ('pending', 'in_progress'))
                    print(f"  📋 {fname}: status={status}, pending={pending}/{len(tasks)}")
                else:
                    print(f"  📋 {fname}: {len(content)}条")
            except Exception as e:
                print(f"  📋 {fname}: ⚠️ {e}")
        else:
            print(f"  📋 {fname}: ❌ 不存在")

def main():
    t0 = datetime.now(TZ)
    today = t0.strftime('%Y-%m-%d')
    print(f"{'='*55}")
    print(f"📡 晨流报告 — {today} {t0.strftime('%H:%M')}")
    print(f"   架构: 🟢绿箭V8-Lottery + 🛡️蓝盾3.0 + 📈A1")
    print(f"{'='*55}")
    print()
    
    results = {}
    
    # 第1步: 清晨摘要
    print("【步骤1/7】清晨摘要")
    results['morning_summary'] = run_script("daily_gen_morning_summary.py")
    print()
    
    # 第2步: 新文件检查
    print("【步骤2/7】新文件检查")
    results['new_files'] = run_script("daily_gen_flag.py")
    print()
    
    # 第3步: 美股持仓同步
    print("【步骤3/7】美股持仓同步")
    results['sync_portfolio'] = run_script("us_sync_portfolio.py")
    print()
    
    # 第4步: 美股评分检查
    check_us_scores()
    
    # 第4.5步: 待处理任务
    check_pending_operations()
    print()
    
    # 第5步: A1每日报告
    print("【步骤5/7】A1每日报告")
    results['a1_report'] = run_script("a1_daily_report.py")
    print()
    
    # 第6步: Layer1宏观信号
    print("【步骤6/7】Layer1宏观信号")
    results['a1_macro'] = run_script("a1_layer1_daily.py")
    print()
    
    # 第7步: 滚动记忆
    print("【步骤7/7】生成滚动记忆")
    results['rolling_memory'] = run_script("daily_rolling_memory.py")
    print()
    
    # 总结
    elapsed = (datetime.now(TZ) - t0).total_seconds()
    ok_count = sum(1 for v in results.values() if v)
    failed = [k for k, v in results.items() if not v]
    
    print(f"{'='*55}")
    print(f"✅ 晨流完成 | 耗时: {elapsed:.1f}s | 成功: {ok_count}/{len(results)}")
    if failed:
        for f in failed:
            print(f"  ❌ {f} 失败")
    print()
    
    # 数据文件状态
    print("【关键数据文件状态】")
    checks = [
        (f"/home/hermes/.hermes/openclaw-project/data/ld3_scored_{today}.json", "蓝盾3.0今日评分"),
        (f"/home/hermes/.hermes/openclaw-project/data/v75_scored_{today}.json", "绿箭V8今日评分"),
        ("data/morning_summary.json", "今日摘要"),
        ("data/a1_daily.json", "A1资金流数据"),
        ("data/operation_plan.json", "操作计划"),
        ("data/active_mission.json", "活跃任务"),

    ]
    for fpath, desc in checks:
        check_file(fpath, desc)
    
    print(f"\n⏱ 总耗时: {elapsed:.1f}s")

if __name__ == '__main__':
    main()
