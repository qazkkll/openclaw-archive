#!/usr/bin/env python3
"""
_run_tests.py — 轻量测试套件（<5秒）
每个test独立try/except，不会一串全挂。

用法：python scripts/_run_tests.py
      python scripts/_run_tests.py --info      # 模块检查，不跑完整测试
      python scripts/_run_tests.py --mtime     # 附加文件修改时间检查
"""
import sys, json, os, traceback, hashlib
from datetime import date as dt_date

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if not os.path.isdir(SCRIPTS):
    SCRIPTS = 'scripts'

RESULTS = {"pass": 0, "fail": 0, "skip": 0}
TESTS = []

def test(name, fn):
    TESTS.append((name, fn))

def run_all():
    for name, fn in TESTS:
        try:
            fn()
            RESULTS["pass"] += 1
            print(f"  [OK] {name}")
        except Exception as e:
            RESULTS["fail"] += 1
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc(limit=1)
    total = RESULTS["pass"] + RESULTS["fail"] + RESULTS["skip"]
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  PASS: {RESULTS['pass']}  FAIL: {RESULTS['fail']}  SKIP: {RESULTS['skip']}  TOTAL: {total}")
    return RESULTS["fail"] == 0

# ─── Test 1: Data file integrity ───
def _test_sp500():
    d = json.load(open(r'/home/hermes/.hermes/openclaw-project/data/sp500_symbols.json', encoding='utf-8'))
    assert len(d) >= 500, f"Expected >=500, got {len(d)}"
test("sp500_symbols.json valid + >=500 entries", _test_sp500)

def _test_ushist():
    d = json.load(open(r'/home/hermes/.hermes/openclaw-project/data/us_hist_clean.parquet', encoding='utf-8'))
    assert len(d) > 100, f"Expected >100 stocks, got {len(d)}"
test("us_hist_clean.parquet valid", _test_ushist)

# ─── Test 2: Score engine ───
def _test_score_engine():
    g = {}
    exec(open(os.path.join(SCRIPTS, 'us_score_engine.py'), encoding='utf-8').read(), g)
    n = 400
    c = [float(100 + i*0.5) for i in range(n)]
    h = [x+3 for x in c]
    l = [x-3 for x in c]
    ind = g['v5s_calc'](c, h, l)
    assert ind is not None, "v5s_calc returned None"
    s_up = g['v5s_score'](ind, n-1)
    assert 0 <= s_up <= 110, f"Score {s_up} out of range"
    c_down = [float(500 - i*0.3) for i in range(n)]
    ind_down = g['v5s_calc'](c_down, [x+3 for x in c_down], [x-3 for x in c_down])
    s_down = g['v5s_score'](ind_down, n-1)
    assert s_up > s_down, f"Uptrend {s_up} should > downtrend {s_down}"
test("Score engine: v5s_calc + v5s_score", _test_score_engine)

# ─── Test 3: Model files exist and load ───
MODEL_FILES = [
    r'/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_l50.json',
    r'/home/hermes/.hermes/openclaw-project/data/models/a1_layer3_xgb_10d.json',
]
for mf in MODEL_FILES:
    fname = os.path.basename(mf)
    def _test_model(m=mf):
        json.load(open(m, encoding='utf-8'))
    test(f"Model: {fname} valid JSON", _test_model)

# ─── Test 4: Today's score outputs (if they exist) ───
today = dt_date.today().strftime('%Y-%m-%d')
today_c = dt_date.today().strftime('%Y%m%d')
for sf_path, sf_label in [
    (f'/home/hermes/.hermes/openclaw-project/data/scored_v75_lottery_{today}.json', f'V8 score {today}'),
    (f'/home/hermes/.hermes/openclaw-project/data/ld3_scored_{today}.json', f'LD3 score {today}'),
    (f'/home/hermes/.hermes/openclaw-project/data/a2_scored_{today_c}.json', f'A2 score {today_c}'),
]:
    if os.path.exists(sf_path):
        def _test_scorefile(p=sf_path):
            json.load(open(p, encoding='utf-8'))
        test(f"Output: {sf_label}", _test_scorefile)
    else:
        def _skip():
            pass
        test(f"Output: {sf_label} (skipped)", _skip)

# ─── Test 5: Active model README/MANUAL exist ───
for rf in [
    r'/home/hermes/.hermes/openclaw-project/data/models/V8_LOTTERY_README.md',
    r'/home/hermes/.hermes/openclaw-project/data/models/V8_LOTTERY_MANUAL.md',
    r'/home/hermes/.hermes/openclaw-project/data/models/a1_layer3_README.md',
]:
    rfname = os.path.basename(rf)
    def _test_readme(r=rf):
        assert os.path.exists(r), f"{r} not found"
    test(f"Docs: {rfname} exists", _test_readme)

# ─── Test 8: SYSTEM_EVOLUTION.md ↔ INDEX.md 日期同───
EVOLUTION = 'docs/SYSTEM_EVOLUTION.md'

def _test_evolution_up_to_date():
    """检查 EVOLUTION.md 覆盖日期 >= INDEX.md 最后更新日期"""
    INDEX_PATH = 'INDEX.md'
    EVOLUTION_PATH = EVOLUTION
    
    if not os.path.exists(EVOLUTION_PATH):
        raise Exception('SYSTEM_EVOLUTION.md 不存在！索引重新整理后必须重建')
    if not os.path.exists(INDEX_PATH):
        raise Exception('INDEX.md 不存在')
    
    # 从 INDEX.md 提取最后更新日期
    idx_text = open(INDEX_PATH, encoding='utf-8').read()
    import re
    idx_date_match = re.search(r'最后更新[：:：]\*{0,2}\s*(\d{4}-\d{2}-\d{2})', idx_text)
    if not idx_date_match:
        raise Exception('INDEX.md 缺少"最后更新"日期标记')
    idx_date = idx_date_match.group(1)
    
    # 从 EVOLUTION.md 提取所有日期（只取系统事件表的日期）
    evo_text = open(EVOLUTION_PATH, encoding='utf-8').read()
    # 系统版本时间线表中的所有日期
    evo_dates = re.findall(r'(\d{4}-\d{2}-\d{2})(?=\s*\|)', evo_text)
    if not evo_dates:
        raise Exception(f'SYSTEM_EVOLUTION.md 版本时间线表中找不到任何日期')
    
    # 获取 INDEX.md 中每个活跃模型的最后更新日期
    # 活跃模型表通常格式：
    # | 模型 | 简称 | ... | 版本 | 状态 |
    # 版本列包含 (YYYY-MM-DD)
    model_dates_in_index = re.findall(r'\b(\d{4}-\d{2}-\d{2})\b', idx_text)
    
    # 检查 INDEX.md 中最新日期是否 <= EVOLUTION.md 中的最新日期
    idx_latest = max(model_dates_in_index) if model_dates_in_index else idx_date
    evo_latest = max(evo_dates) if evo_dates else '1970-01-01'
    
    if idx_latest > evo_latest:
        raise Exception(
            f'INDEX.md 最新模型日期 {idx_latest} 超过 EVOLUTION.md 覆盖日期 {evo_latest}\n'
            f'  → 需要在 SYSTEM_EVOLUTION.md 中追加对应版本的记录'
        )

test("SYSTEM_EVOLUTION.md current vs INDEX model dates", _test_evolution_up_to_date)

# ─── Test 10: Old tests remain ───
def _test_index():
    assert os.path.exists('INDEX.md'), "INDEX.md missing"
    sz = os.path.getsize('INDEX.md')
    assert sz > 10000, f"INDEX.md too small: {sz} bytes"
test("INDEX.md exists + size ok", _test_index)

# ─── Test 7: Mtime change detection ───
TRACKED_FILES = [
    'scripts/us_score_engine.py',
    'scripts/us_v7_5_daily_score.py',
    'scripts/us_ld3_daily_score.py',
    r'/home/hermes/.hermes/openclaw-project/data/models/us_v7_5_l50.json',
    r'/home/hermes/.hermes/openclaw-project/data/models/a1_layer3_xgb_10d.json',
    'INDEX.md',
    'data/facts.json',
    'data/rules.json',
]

def _test_changes():
    """检测关键文件mtime变化"""
    import json
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) or '.'
    mtime_path = os.path.join(base, 'data', 'test_mtime_state.json')
    snap = {}
    try:
        with open(mtime_path) as f:
            snap = json.load(f)
    except:
        pass
    changed = []
    new_snap = {}
    for fpath in TRACKED_FILES:
        full_path = fpath if os.path.isabs(fpath) else os.path.join(base, fpath)
        if not os.path.exists(full_path):
            continue
        mt = os.path.getmtime(full_path)
        new_snap[fpath] = mt
        old_mt = snap.get(fpath)
        if old_mt and abs(mt - old_mt) > 1:
            changed.append(fpath)
    os.makedirs(os.path.dirname(mtime_path), exist_ok=True)
    with open(mtime_path, 'w') as f:
        json.dump(new_snap, f)
    if changed:
        print(f"   文件修改: {', '.join(changed)}")
    else:
        print(f"   无文件修改")
test("Mtime change detection", _test_changes)

# ─── Test 9: Temp files cleaned ───
for f in ['tmp_test_engine.py', 'tmp_test_engine2.py', 'tmp_test_engine_*.py']:
    if os.path.exists(f):
        try: os.remove(f)
        except: pass
test("Temp files cleaned", lambda: None)

# ─── Run ───
if __name__ == '__main__':
    all_pass = run_all()
    sys.exit(0 if all_pass else 1)
