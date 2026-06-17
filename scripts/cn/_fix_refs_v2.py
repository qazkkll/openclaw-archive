# 二次修复残留引用
import os

sp = r'/home/hermes/.hermes/openclaw-archive\scripts'
fixes = {
    'score_engine': 'us_score_engine',
    's1_scan': 'us_v5s_s1_scan',
    's2_candidates': 'us_v5s_s2_candidates',
}
not_fixed = []
fix_log = []

py_files = sorted([f for f in os.listdir(sp) if f.endswith('.py') and not f.startswith('_')])

for pf in py_files:
    fp = os.path.join(sp, pf)
    txt = open(fp, 'r', encoding='utf-8', errors='replace').read()
    orig = txt

    for old_short, new_name in fixes.items():
        # 替换所有import/from/open引用中的旧名
        if old_short in txt:
            # 在import/from/open行中替换
            lines = txt.split('\n')
            changed = False
            for i, line in enumerate(lines):
                if (old_short in line and ('import' in line or 'from' in line or 'open(' in line.lower() or 'read()' in line.lower())):
                    old_full = old_short
                    new_full = new_name
                    if old_full in line:
                        lines[i] = line.replace(old_full, new_full)
                        changed = True
            if changed:
                txt = '\n'.join(lines)

    if txt != orig:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(txt)
        fix_log.append(pf)
        print(f'[FIXED] {pf}')

# 验证
print('\n--- 验证 ---')
for pf in py_files:
    txt = open(os.path.join(sp, pf), 'r', encoding='utf-8', errors='replace').read()
    for old_short in fixes:
        if old_short in txt:
            for li, line in enumerate(txt.split('\n'), 1):
                if old_short in line and ('import' in line or 'from' in line or 'open(' in line.lower()):
                    if f'{old_short}_' not in line:  # 跳过已修复
                        print(f'  [REMAINS] {pf}:{li}: {line.strip()[:80]}')

print(f'\nFixed: {len(fix_log)} files')
if not fix_log:
    print('Nothing to fix - all clean')
