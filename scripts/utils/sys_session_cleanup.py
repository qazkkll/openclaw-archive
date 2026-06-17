#!/usr/bin/env python3
"""
sys_session_cleanup.py — 大任务前 Session 预清理
===================================================
用途: 在跑评分/回测/数据下载之前调用，清理过期 Session 防止 compaction 死锁。
集成位置:
  - us_cron_wrapper.py（每日评分前）
  - _run_zhengli.py（存档前）
  - 回测脚本入口（手动）

依赖: openclaw CLI (npm .cmd)
"""
import subprocess, sys, datetime
sys.stdout.reconfigure(encoding='utf-8')

CLI = r'C:\Users\admin\AppData\Roaming\npm\openclaw.cmd'

def clean_sessions(dry_run: bool = False) -> bool:
    """复用入口：from sys_session_cleanup import clean_sessions"""
    return main(dry_run=dry_run)

def main(dry_run: bool = False):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [SESSION CLEANUP] 开始 ...")

    # ── 第1步：常规清理（过期+超量） ──
    cmd = [CLI, 'sessions', 'cleanup', '--enforce']
    if dry_run:
        cmd.append('--dry-run')
        print(f"  dry-run: {' '.join(cmd)}")
    else:
        print(f"  执行: sessions cleanup --enforce")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        encoding='utf-8', errors='replace'
    )

    output = (result.stdout or '').strip()
    if output:
        lines = [l for l in output.split('\n')
                 if l.strip() and 'duration' not in l.lower() and 'ms' not in l.lower()[:4]]
        for l in lines[-10:]:
            print(f"  {l}")

    if result.returncode != 0:
        err = (result.stderr or '').strip()
        print(f"  [WARN] cleanup 退出码 {result.returncode}" + (f": {err[:200]}" if err else ""))
    else:
        print(f"  [OK] basic cleanup 完成")

    # ── 第2步：orphan 清理（transcript 文件丢失的脏记录） ──
    if not dry_run:
        fix_cmd = [CLI, 'sessions', 'cleanup', '--enforce', '--fix-missing']
        fix_result = subprocess.run(
            fix_cmd, capture_output=True, text=True, timeout=30,
            encoding='utf-8', errors='replace'
        )
        if fix_result.returncode == 0:
            fix_out = (fix_result.stdout or '').strip()
            if fix_out and 'nothing' not in fix_out.lower():
                last = [l for l in fix_out.split('\n') if l.strip()][-3:]
                for l in last:
                    print(f"  {l}")
            print(f"  [OK] orphan cleanup 完成")
        else:
            print(f"  [WARN] orphan cleanup 跳过 (exit={fix_result.returncode})")

    print(f"  [DONE] Session 清理完毕")
    return result.returncode == 0

if __name__ == '__main__':
    dry = '--dry-run' in sys.argv
    ok = main(dry_run=dry)
    sys.exit(0 if ok else 1)
