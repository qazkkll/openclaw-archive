"""
dual_backup.py — 包装器，被Windows定时任务openclaw_backup_0100调用
实际执行 backup_daily.py
"""
import sys
import subprocess

if __name__ == "__main__":
    result = subprocess.run(
        ["python3", "D:\\openclaw-workspace\\scripts\\backup_daily.py"],
        capture_output=True, text=True, timeout=120
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
