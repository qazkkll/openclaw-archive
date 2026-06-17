#!/usr/bin/env python3
"""
A2每日盘前评分 — Cron包装脚本
=================================
9:00 A股开盘前运行。负责：
  1. 运行 a1_layer3_5_scoring.py（数据更新+评分+推荐）
  2. 检查每一步是否成功
  3. 检查输出文件是否存在
  4. 任何步骤失败则报错到stdout（cron会推到Telegram）

退出码：0=成功, 1=有错误（cron会报告）
"""

import subprocess, sys, os, datetime, json
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

D_DATA = r'/home/hermes/.hermes/openclaw-archive/data'
SCRIPT = os.path.join(os.path.dirname(__file__), 'a1_layer3_5_scoring.py')

errors = []
warnings = []
today = datetime.date.today().strftime('%Y%m%d')
expected_output = f'a2_scored_{today}.json'

print("=" * 50)
print(f"[A2] 每日盘前评分 | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 50)

# ── Step 1: 运行评分脚本 ──
print("\n[Step 1/2] 运行 a1_layer3_5_scoring.py ...")
sys.stdout.flush()

start = datetime.datetime.now()
result = subprocess.run(
    ['python', SCRIPT],
    capture_output=True, text=True, timeout=300,
    encoding='utf-8', errors='replace'
)
elapsed = (datetime.datetime.now() - start).total_seconds()

if result.returncode != 0:
    errors.append(f"[ERROR] a1_layer3_5_scoring.py 退出码 {result.returncode}")
    if result.stderr:
        errors.append(f"   stderr: {result.stderr[-500:]}")

stdout = result.stdout.strip()
if stdout:
    print(stdout)
stderr = result.stderr.strip()
if stderr:
    print(f"[stderr]\n{stderr[-300:]}", file=sys.stderr)

print(f"\n[耗时] 脚本: {elapsed:.0f}s")

# ── Step 2: 检查输出文件 ──
print("\n[Step 2/2] 检查输出文件 ...")

output_path = os.path.join(D_DATA, expected_output)
if os.path.exists(output_path):
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"[OK] 输出文件存在: {expected_output}")
        print(f"   评分股票数: {data.get('total_scored', '?')}")
        print(f"   数据日期: {data.get('date', '?')}")
        print(f"   资金流日期: {data.get('mf_date', '?')}")
        print(f"   大盘情绪: {data.get('mood', '?')}")

        kline_date = data.get('date', '')
        mf_date = data.get('mf_date', '')
        today_dt = datetime.date.today()
        try:
            kline_dt = datetime.datetime.strptime(kline_date, '%Y%m%d').date()
            delta = (today_dt - kline_dt).days
            max_lag = 3 if today_dt.weekday() >= 5 else 1
            if delta > max_lag:
                warnings.append(f"[WARN] K线数据滞后{delta}天: {kline_date}")
        except:
            pass
        try:
            mf_dt = datetime.datetime.strptime(mf_date, '%Y%m%d').date()
            delta = (today_dt - mf_dt).days
            max_lag = 3 if today_dt.weekday() >= 5 else 1
            if delta > max_lag:
                warnings.append(f"[WARN] 资金流数据滞后{delta}天: {mf_date}")
        except:
            pass

        top10 = data.get('top10', [])
        if top10:
            print(f"   Top1: {top10[0].get('name','?')} 评分{top10[0].get('score',0):+.1f}")
        else:
            warnings.append("[WARN] Top10为空")
    except Exception as e:
        errors.append(f"[ERROR] 输出文件解析失败: {e}")
else:
    errors.append(f"[ERROR] 输出文件不存在: {output_path}")

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

print(f"[耗时] 总: {elapsed:.0f}s")
print("=" * 50)
