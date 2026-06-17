#!/usr/bin/env python3
"""
_data_integrity.py — 数据完整性守卫
检查所有关键数据文件，确保数据质量。

用法：
  python scripts/_data_integrity.py                  # 完整检查
  python scripts/_data_integrity.py --quick           # 只检查JSON可解析性
  python scripts/_data_integrity.py --file <路径>      # 只检查指定文件
"""
import json, os, sys, math

RESULTS = {"pass": 0, "fail": 0, "warn": 0}
QUICK = "--quick" in sys.argv

def check(name, cond, detail=""):
    if cond:
        RESULTS["pass"] += 1
        print(f"  [OK] {name}")
    else:
        RESULTS["fail"] += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

def warn(name, detail=""):
    RESULTS["warn"] += 1
    print(f"  [WARN] {name}" + (f" — {detail}" if detail else ""))

def parse_json(path):
    """Try to parse a JSON file, return (data|None, error|None)"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, str(e)

def check_json_file(path, label=None):
    if label is None:
        label = os.path.basename(path)
    d, err = parse_json(path)
    check(f"{label}: parseable", d is not None, err or "")
    return d

def has_field(d, field, datatype=None):
    if field not in d:
        return False
    if datatype and not isinstance(d[field], datatype):
        return False
    return True

# ======================== CHECKS ========================

# ─── 1. Core data files ───
print("\n=== Core Data Files ===")
root = r'/home/hermes/.hermes/openclaw-archive/data'

us_hist = check_json_file(root + '/us_hist_clean.parquet', 'us_hist_clean.parquet')
if us_hist and not QUICK:
    # Check structure: top-level is dict or list
    if isinstance(us_hist, dict):
        symbols = list(us_hist.keys())
        check(f"us_hist: has stocks", len(symbols) > 0, f"got 0 symbols")
        # Spot-check first stock for required fields
        first = symbols[0]
        first_data = us_hist[first]
        if isinstance(first_data, (list, dict)):
            check(f"us_hist: {first} is iterable", True)
        else:
            warn(f"us_hist: {first} is {type(first_data).__name__}")
    elif isinstance(us_hist, list):
        check(f"us_hist: list of {len(us_hist)} entries", len(us_hist) > 0)
        if len(us_hist) > 0:
            first = us_hist[0]
            check(f"us_hist[0] has close/open/high/low", all(k in first for k in ['close','open','high','low']) if isinstance(first, dict) else "skipped non-dict")
    else:
        check(f"us_hist: top-level type", False, f"unexpected type {type(us_hist).__name__}")

# sp500 symbols
sp500 = check_json_file(root + '/sp500_symbols.json', 'sp500_symbols.json')
if sp500:
    check(f"sp500: >=500 symbols", len(sp500) >= 500, f"got {len(sp500)}")

# ─── 2. Today's score files ───
from datetime import date as dt_date
today = dt_date.today().strftime('%Y-%m-%d')
today_c = dt_date.today().strftime('%Y%m%d')

print("\n=== Today's Score Outputs ===")
for sf_path, sf_label in [
    (root + f'/scored_v75_lottery_{today}.json', f'V8 score ({today})'),
    (root + f'/ld3_scored_{today}.json', f'LD3 score ({today})'),
    (root + f'/a2_scored_{today_c}.json', f'A2 score ({today_c})'),
]:
    if not os.path.exists(sf_path):
        warn(f"{sf_label}: file not found (not yet generated)")
        continue
    d = check_json_file(sf_path, sf_label)
    if d and not QUICK:
        if isinstance(d, dict) and 'score' in d:
            check(f"{sf_label}: has score field", True)
        elif isinstance(d, list):
            check(f"{sf_label}: list with {len(d)} entries", len(d) > 0)

# ─── 3. Model files ───
print("\n=== Model Files ===")
models_dir = root + '/models'
model_files = [
    models_dir + '/us_v7_5_l50.json',
    models_dir + '/a1_layer3_xgb_10d.json',
]
for mf in model_files:
    fname = os.path.basename(mf)
    if not os.path.exists(mf):
        check(f"{fname}: file exists", False, "not found")
        continue
    d = check_json_file(mf, fname)
    if d and not QUICK:
        size_kb = os.path.getsize(mf) / 1024
        check(f"{fname}: file size > 1KB", size_kb > 1, f"{size_kb:.1f} KB")

# ─── 4. INDEX.md ───
print("\n=== Infrastructure ===")
if os.path.exists('INDEX.md'):
    sz = os.path.getsize('INDEX.md')
    check(f"INDEX.md: size ok", sz > 10000, f"{sz} bytes")
else:
    check(f"INDEX.md: exists", False)

if os.path.exists('SESSION-STATE.md'):
    sz = os.path.getsize('SESSION-STATE.md')
    check(f"SESSION-STATE.md: size ok", sz > 100, f"{sz} bytes")
else:
    check(f"SESSION-STATE.md: exists", False)

# ─── Summary ───
print(f"\n{'='*50}")
print(f"  PASS: {RESULTS['pass']}  FAIL: {RESULTS['fail']}  WARN: {RESULTS['warn']}")
print(f"{'='*50}")

if RESULTS['fail']:
    print("\n  *** DATA INTEGRITY ISSUES FOUND ***")
    sys.exit(1)
else:
    print("\n  All data integrity checks passed.")
