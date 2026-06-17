#!/usr/bin/env python3
"""
美股每日双模型评分 — Cron包装脚本
====================================
21:00 美盘前运行。负责：
  1. 运行 us_v7_5_daily_score.py（绿箭V8-Lottery）
  2. 运行 us_ld3_daily_score.py（蓝盾3.0 S&P 500）
  3. 运行 us_daily_recommend.py（双模型融合）
  4. 检查每一步退出码 + 检查输出文件
  5. 任何步骤失败则报错到stdout（cron会推到Telegram）

退出码：0=成功, 1=有错误
"""

import subprocess, sys, os, datetime, json, glob
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

SCRIPTS_DIR = os.path.dirname(__file__)
D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'

today = datetime.date.today().strftime('%Y-%m-%d')
today_c = today.replace('-', '')
errors = []
warnings = []

print("=" * 50)
print(f"[US] 美股每日双模型评分 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 50)

# ── ⚡ Step 0: Session 预清理（防卡死） ──
cleanup_script = os.path.join(SCRIPTS_DIR, 'sys_session_cleanup.py')
print(f"\n[Step 0/4] [CLEANUP] Session 预清理: {cleanup_script}")
sys.stdout.flush()
result = subprocess.run(
    ['python', cleanup_script],
    capture_output=True, text=True, timeout=120,
    encoding='utf-8', errors='replace'
)
if result.stdout.strip():
    print(result.stdout.strip())
if result.stderr.strip():
    print(f"[stderr-cleanup] {result.stderr.strip()[-200:]}")
print(f"   退出码: {result.returncode}")
if result.returncode != 0:
    warnings.append(f"[WARN] Session 清理返回非0 (exit={result.returncode})")

# ── 绿箭V8-Lottery ──
green_script = os.path.join(SCRIPTS_DIR, 'us_v7_5_daily_score.py')
print(f"\n[Step 1/3] [GREEN] 绿箭V8-Lottery: {green_script}")
sys.stdout.flush()

result = subprocess.run(
    ['python', green_script],
    capture_output=True, text=True, timeout=180,
    encoding='utf-8', errors='replace'
)
if result.stdout.strip():
    print(result.stdout.strip()[-800:])
if result.stderr.strip():
    print(f"[stderr] {result.stderr.strip()[-300:]}")
print(f"   退出码: {result.returncode}")
if result.returncode != 0:
    errors.append(f"[ERROR] 绿箭V8评分失败 (exit={result.returncode})")

# ── 蓝盾3.0 ──
bs_script = os.path.join(SCRIPTS_DIR, 'us_ld3_daily_score.py')
print(f"\n[Step 2/3] [BLUE] 蓝盾3.0: {bs_script}")
sys.stdout.flush()

result = subprocess.run(
    ['python', bs_script],
    capture_output=True, text=True, timeout=180,
    encoding='utf-8', errors='replace'
)
if result.stdout.strip():
    print(result.stdout.strip()[-800:])
if result.stderr.strip():
    print(f"[stderr] {result.stderr.strip()[-300:]}")
print(f"   退出码: {result.returncode}")
if result.returncode != 0:
    errors.append(f"[ERROR] 蓝盾3.0评分失败 (exit={result.returncode})")

# ── 双模型融合 ──
fusion_script = os.path.join(SCRIPTS_DIR, 'us_daily_recommend.py')
print(f"\n[Step 3/3] [FUSION] 双模型融合: {fusion_script}")
sys.stdout.flush()

result = subprocess.run(
    ['python', fusion_script],
    capture_output=True, text=True, timeout=180,
    encoding='utf-8', errors='replace'
)
if result.stdout.strip():
    print(result.stdout.strip()[-1500:])
if result.stderr.strip():
    print(f"[stderr] {result.stderr.strip()[-500:]}")
print(f"   退出码: {result.returncode}")
if result.returncode != 0:
    errors.append(f"[ERROR] 双模型融合失败 (exit={result.returncode})")

# ── 输出文件检查 ──
print("\n" + "-" * 50)
print("检查输出文件 ...")

green_glob = os.path.join(D_DATA, f'scored_v75_*{today}*')
green_glob2 = os.path.join(D_DATA, f'scored_v75_*{today_c}*')
green_files = glob.glob(green_glob) + glob.glob(green_glob2)
if green_files:
    print(f"[OK] 绿箭输出: {os.path.basename(green_files[0])}")
else:
    errors.append(f"[ERROR] 绿箭输出文件未找到 (searched: {green_glob})")

bs_glob = os.path.join(D_DATA, f'ld3_scored_*{today}*')
bs_glob2 = os.path.join(D_DATA, f'ld3_scored_*{today_c}*')
bs_files = glob.glob(bs_glob) + glob.glob(bs_glob2)
if bs_files:
    print(f"[OK] 蓝盾输出: {os.path.basename(bs_files[0])}")
else:
    errors.append(f"[ERROR] 蓝盾输出文件未找到 (searched: {bs_glob})")

fusion_glob = os.path.join(D_DATA, f'fusion_rec_*{today}*')
fusion_glob2 = os.path.join(D_DATA, f'fusion_rec_*{today_c}*')
fusion_files = glob.glob(fusion_glob) + glob.glob(fusion_glob2)
if fusion_files:
    print(f"[OK] 融合输出: {os.path.basename(fusion_files[0])}")
else:
    errors.append(f"[ERROR] 融合输出文件未找到 (searched: {fusion_glob})")

# ── 总结 ──
print("\n" + "=" * 50)
if errors:
    print(f"[FAIL] {len(errors)} 个错误")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
elif warnings:
    print(f"[WARN] 成功但有警告: {len(warnings)} 个")
    for w in warnings:
        print(f"  {w}")
else:
    print("[PASS] 全部通过")
print("=" * 50)
