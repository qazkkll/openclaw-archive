# audit v2: scripts rename check - strict import/from only
import os, re

sp = r'/home/hermes/.hermes/openclaw-archive\scripts'
errors = []
warnings = []

all_files = sorted([f for f in os.listdir(sp) if f.endswith('.py')])

# --- 1. naming convention ---
viol = [f for f in all_files if not any(f.startswith(p) for p in ['a1_','a_','us_','daily_','dl_','sys_','tst_','tmp_','_'])]
if viol:
    errors.append(f'unformatted: {viol}')
else:
    print('[PASS] all named by convention')

# --- 2. stale imports ---
stale_imports = [
    ('score_engine', 'us_score_engine'),  # without us_ prefix
    ('s1_scan', 'us_v5s_s1_scan'),
    ('s2_candidates', 'us_v5s_s2_candidates'),
    ('layer1_daily', 'a1_layer1_daily'),
    ('layer1_market_state', 'a1_layer1_market_state'),
    ('layer3_4_scoring', 'a1_layer3_4_scoring'),
    ('u5_scan', 'us_u5_scan'),
    ('backtest', 'us_v5s_backtest'),  # bare backtest
]

stale_found = []
for pf in all_files:
    fp = os.path.join(sp, pf)
    content = open(fp, 'r', encoding='utf-8', errors='replace').read()
    lines = content.split('\n')
    for li, line in enumerate(lines, 1):
        stripped = line.strip()
        # only care about: import X, from X import, open("X"), open('X'), load("X")
        if not (stripped.startswith('import ') or stripped.startswith('from ') or 
                'open(' in stripped or 'load(' in stripped.lower() or 'read(' in stripped.lower()):
            continue
        for old_name, new_name in stale_imports:
            if old_name in stripped:
                # verify it doesn't already use new_name
                if new_name not in stripped:
                    stale_found.append((pf, li, old_name, stripped[:90]))

if stale_found:
    errors.append(f'stale imports ({len(stale_found)}):')
    for f, li, old, ctx in stale_found:
        errors.append(f'  {f}:{li} -> {old}: {ctx}')
else:
    print('[PASS] no stale imports')

# --- 3. duplicate prefix (us_us_) ---
dup = []
for pf in all_files:
    fp = os.path.join(sp, pf)
    content = open(fp, 'r', encoding='utf-8', errors='replace').read()
    lines = content.split('\n')
    for li, line in enumerate(lines, 1):
        stripped = line.strip()
        if 'us_us_' in stripped and (stripped.startswith('import ') or stripped.startswith('from ')
                                      or 'open(' in stripped):
            dup.append((pf, li, stripped[:90]))
if dup:
    errors.append(f'duplicate us_us_ prefix ({len(dup)}):')
    for f, li, ctx in dup:
        errors.append(f'  {f}:{li}: {ctx}')
else:
    print('[PASS] no duplicate us_us_ prefix')

# --- 4. INDEX.md ---
idx_path = os.path.join(sp, '..', 'INDEX.md')
if os.path.exists(idx_path):
    txt = open(idx_path, 'r', encoding='utf-8').read()
    if '命名规范' in txt and '变更日志' in txt:
        print('[PASS] INDEX.md exists')
    else:
        warnings.append('INDEX.md missing sections')
else:
    errors.append('INDEX.md missing')

# --- 5. AGENTS.md naming rules ---
ag_path = os.path.join(sp, '..', 'AGENTS.md')
if os.path.exists(ag_path):
    txt = open(ag_path, 'r', encoding='utf-8').read()
    if '脚本文件命名规范' in txt:
        print('[PASS] AGENTS.md has naming rules')
    else:
        warnings.append('AGENTS.md missing naming section')
else:
    errors.append('AGENTS.md missing')

# summary
formal_ct = sum(1 for f in all_files if f.startswith(('a1_','a_','us_','daily_','dl_','sys_')))
tst_ct = sum(1 for f in all_files if f.startswith('tst_'))
int_ct = sum(1 for f in all_files if f.startswith('_'))
tmp_ct = sum(1 for f in all_files if f.startswith('tmp_'))

print(f'\n==== AUDIT SUMMARY ====')
print(f'Files: total={len(all_files)} formal={formal_ct} test={tst_ct} internal={int_ct} temp={tmp_ct}')
if errors:
    print(f'\n[FAIL] {len(errors)} issues:')
    for e in errors:
        print(f'  {e}')
    print('\nAUDIT NOT PASSED - FIX BEFORE PROCEED')
else:
    print('\n[PASS] audit passed')
if warnings:
    print(f'\n[WARN] {len(warnings)}:')
    for w in warnings:
        print(f'  {w}')
