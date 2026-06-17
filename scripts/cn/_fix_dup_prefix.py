# 修复重复前缀问题 us_us_ -> us_
import os

sp = r'/home/hermes/.hermes/openclaw-archive\scripts'
fixes = {
    'us_us_score_engine': 'us_score_engine',
    'us_us_v5s_s1_scan': 'us_v5s_s1_scan',
    'us_v5s_us_v5s_s1_scan': 'us_v5s_s1_scan',
}

py_files = sorted([f for f in os.listdir(sp) if f.endswith('.py') and not f.startswith('_')])
fixed = []

for pf in py_files:
    fp = os.path.join(sp, pf)
    txt = open(fp, 'r', encoding='utf-8', errors='replace').read()
    orig = txt
    for old, new in fixes.items():
        txt = txt.replace(old, new)
    if txt != orig:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(txt)
        fixed.append(pf)
        print(f'[FIXED DUPLICATE] {pf}')

print(f'\nFixed: {len(fixed)} files')

# 最终验证
print('\n--- 最终验证 ---')
remaining_still = 0
for pf in py_files:
    txt = open(os.path.join(sp, pf), 'r', encoding='utf-8', errors='replace').read()
    for line_i, line in enumerate(txt.split('\n'), 1):
        if 'score_engine' in line and ('import' in line or 'from' in line):
            if 'us_score_engine' not in line and 'score_engine' in line:
                print(f'  [BAD] {pf}:{line_i}: {line.strip()[:80]}')
                remaining_still += 1
            elif 'us_us_' in line:
                print(f'  [BAD DUP] {pf}:{line_i}: {line.strip()[:80]}')
                remaining_still += 1
        if ('s1_scan' in line or 's2_candidates' in line) and 'import' in line or 'from' in line:
            if 'us_v5s_' not in line and (old in line for old in ['s1_scan','s2_candidates']):
                print(f'  [BAD SCAN] {pf}:{line_i}: {line.strip()[:80]}')
                remaining_still += 1

if remaining_still:
    print(f'\n[FAIL] {remaining_still} remaining')
else:
    print('[OK] all imports clean')
