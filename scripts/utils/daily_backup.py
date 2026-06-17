"""
backup_daily.py — 每日备份流水线
纯Python执行，不依赖模型调用，可在无agentTurn的cron中运行。

功能：
1. 打包 sessions + memory + config_keys + key data
2. 本地存到 backup/ 目录
3. SCP到云端ECS
4. 清理30天前的旧备份
5. 写入 archive_state.json

用法：
  python scripts/backup_daily.py              # 正常执行
  python scripts/backup_daily.py --dry-run    # 试跑不真正写入
  python scripts/backup_daily.py --cleanup    # 只清理旧备份
"""

import json
import os
import sys
import zipfile
import shutil
import subprocess
import re
from datetime import datetime, timedelta

# === 配置 ===
WORKSPACE = r"/home/hermes/.hermes/openclaw-archive"
BACKUP_DIR = os.path.join(WORKSPACE, "backup")
STATE_FILE = os.path.join(WORKSPACE, "data", "archive_state.json")
CLOUD_USER = "admin"
CLOUD_HOST = "8.217.51.136"
CLOUD_ARCHIVE_DIR = "archive"
LOG_FILE = os.path.join(WORKSPACE, "backup", "backup_log.txt")
RETENTION_DAYS = 30  # 本地保留天数

# === 需要备份的路径 ===
BACKUP_ITEMS = [
    # Memory
    (os.path.join(WORKSPACE, "memory"), "memory", True),
    # Key scripts (核心工具)
    (os.path.join(WORKSPACE, "scripts", "config_keys.py"), "scripts", False),
    (os.path.join(WORKSPACE, "scripts", "us_score_engine.py"), "scripts", False),
    (os.path.join(WORKSPACE, "scripts", "us_scoring.py"), "scripts", False),
    (os.path.join(WORKSPACE, "scripts", "sys_safe_guard.py"), "scripts", False),
    (os.path.join(WORKSPACE, "scripts", "daily_startup_gate.py"), "scripts", False),
    (os.path.join(WORKSPACE, "scripts", "sys_provider_health.py"), "scripts", False),
    (os.path.join(WORKSPACE, "scripts", "daily_backup.py"), "scripts", False),
    # Config files
    (os.path.join(os.path.expanduser("~"), ".openclaw", "openclaw.json"), "config", False),
    (os.path.join(os.path.expanduser("~"), ".openclaw", "gateway.cmd"), "config", False),
    # Data (not the huge files)
    (os.path.join(WORKSPACE, "data", "archive_state.json"), "data", False),
    (os.path.join(WORKSPACE, "data", "code_index.json"), "data", False),
    (os.path.join(WORKSPACE, "data", "recommendations.md"), "data", False),
    # Models architecture
    (os.path.join(WORKSPACE, "models", "ARCHITECTURE.md"), "models", False),
    (os.path.join(WORKSPACE, "models", "A1-B"), "models", True),
    # Learnings
    (os.path.join(WORKSPACE, ".learnings"), "learnings", True),
]


def _strip_emoji(s):
    emoji = re.compile("["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji.sub('?', s)

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    safe = _strip_emoji(line)
    print(safe)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_zip(zip_path, items):
    """创建zip包，items = [(path, arc_subdir, is_dir), ...]"""
    count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc_subdir, is_dir in items:
            if not os.path.exists(src):
                log(f"  SKIP (not found): {src}")
                continue
            if is_dir:
                for root, dirs, files in os.walk(src):
                    for fn in files:
                        fp = os.path.join(root, fn)
                        rel = os.path.relpath(fp, os.path.dirname(src))
                        arcname = f"{arc_subdir}/{rel}"
                        zf.write(fp, arcname)
                        count += 1
            else:
                fn = os.path.basename(src)
                arcname = f"{arc_subdir}/{fn}"
                zf.write(src, arcname)
                count += 1
    return count


def scp_to_cloud(local_path, remote_dir):
    """SCP到云端，返回成功/失败"""
    try:
        # 先确保云端目录存在
        subprocess.run(
            ["ssh", f"{CLOUD_USER}@{CLOUD_HOST}", f"mkdir -p {remote_dir}"],
            capture_output=True, text=True, timeout=10
        )
        result = subprocess.run(
            ["scp", local_path, f"{CLOUD_USER}@{CLOUD_HOST}:{remote_dir}/"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:300]
    except Exception as e:
        return False, str(e)


def cleanup_old_backups():
    """清理超过RETENTION_DAYS的本地旧备份"""
    now = datetime.now()
    cleaned = 0
    if not os.path.exists(BACKUP_DIR):
        return 0
    for fn in os.listdir(BACKUP_DIR):
        if fn.endswith(".zip"):
            fp = os.path.join(BACKUP_DIR, fn)
            mtime = datetime.fromtimestamp(os.path.getmtime(fp))
            if (now - mtime).days > RETENTION_DAYS:
                os.remove(fp)
                cleaned += 1
                log(f"  CLEANED: {fn} ({mtime.date()}, age {(now-mtime).days}d)")
    return cleaned


def main():
    dry_run = "--dry-run" in sys.argv
    cleanup_only = "--cleanup" in sys.argv

    if cleanup_only:
        log("=== 只清理旧备份 ===")
        cleaned = cleanup_old_backups()
        log(f"Cleaned: {cleaned} old backups")
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    date_dir = today  # YYYY-MM-DD
    
    log("=== 每日备份开始 ===")
    log(f"Date: {today}")
    if dry_run:
        log("DRY RUN MODE - 不会实际写入")

    # 1. 创建备份目录
    local_archive_dir = os.path.join(BACKUP_DIR, date_dir)
    os.makedirs(local_archive_dir, exist_ok=True)
    zip_filename = f"backup_{today}.zip"
    zip_path = os.path.join(BACKUP_DIR, zip_filename)

    # 2. 打包
    log("Packing files...")
    file_count = make_zip(zip_path, BACKUP_ITEMS)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    log(f"  Packed: {file_count} files → {zip_filename} ({size_mb:.1f} MB)")

    # 3. 写入本地副本（按日期存放方便查看）
    if not dry_run:
        copy_path = os.path.join(local_archive_dir, zip_filename)
        shutil.copy2(zip_path, copy_path)
        log(f"  Local copy: {copy_path}")

    # 4. SCP到云端
    if not dry_run:
        log("Uploading to cloud...")
        ok, err = scp_to_cloud(zip_path, f"{CLOUD_ARCHIVE_DIR}/{date_dir}")
        if ok:
            log("  ✅ Cloud upload OK")
        else:
            log(f"  ❌ Cloud upload failed: {err}")

        # 也上传memory文件单独一份方便云端读取
        memory_zip = os.path.join(BACKUP_DIR, f"memory_{today}.zip")
        memory_items = [(os.path.join(WORKSPACE, "memory"), "memory", True)]
        make_zip(memory_zip, memory_items)
        scp_to_cloud(memory_zip, f"{CLOUD_ARCHIVE_DIR}/{date_dir}")
        os.remove(memory_zip)

    # 5. 清理旧备份
    if not dry_run:
        cleaned = cleanup_old_backups()
        log(f"Cleaned {cleaned} old backups (>{RETENTION_DAYS}d)")

    # 6. 写入状态文件
    state = {
        "status": "completed",
        "date": today,
        "archiveTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "local": {
            "package": f"backup/{zip_filename} ({size_mb:.1f} MB)",
            "files": file_count
        },
        "cloud": {
            "path": f"{CLOUD_ARCHIVE_DIR}/{date_dir}/{zip_filename}"
        }
    }
    if not dry_run:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log(f"  State written: {STATE_FILE}")

    # 7. Git推送（纯代码，不含数据文件）
    if not dry_run:
        try:
            log("Git push...")
            git_r = subprocess.run(
                ['git', '-C', WORKSPACE, 'add', '-A'],
                capture_output=True, text=True, timeout=30
            )
            git_r = subprocess.run(
                ['git', '-C', WORKSPACE, 'commit', '--allow-empty', '-m', f'backup {today}'],
                capture_output=True, text=True, timeout=30
            )
            git_r = subprocess.run(
                ['git', '-C', WORKSPACE, 'push', 'origin', 'HEAD'],
                capture_output=True, text=True, timeout=60
            )
            if git_r.returncode == 0:
                log("  ✅ Git push OK")
            else:
                log(f"  ⚠ Git push: {git_r.stderr.strip()[-80:]}")
        except Exception as e:
            log(f"  ⚠ Git push skipped: {e}")

    log("=== 每日备份完成 ===")
    return 0 if not dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
