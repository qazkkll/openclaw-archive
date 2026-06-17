# 检查正式脚本中引用 data/*.json 的地方
import os, re

dp = r'/home/hermes/.hermes/openclaw-archive\scripts'
formal = [f for f in sorted(os.listdir(dp)) if f.endswith('.py') and not f.startswith('_') and not f.startswith('tmp_') and not f.startswith('tst_')]

issues = []
for pf in formal:
    fp = os.path.join(dp, pf)
    content = open(fp, 'r', encoding='utf-8', errors='replace').read()
    lines = content.split('\n')
    for li, line in enumerate(lines, 1):
        if '.json' not in line:
            continue
        # 检查是不是读数据的引用
        low = line.strip()
        if not ('import' in low or 'from' in low or 'open(' in low or 'load(' in low or '.read()' in low):
            continue
        if 'raw_json/' in low or 'example' in low.lower():
            continue
        
        # 提取json文件名
        found = re.findall(r'[\x27\x22]([^\x27\x22]+\.json)[\x27\x22]', line)
        for j in found:
            # 只报 .json 结尾的（不是 .jsonl 等）
            if j.endswith('.json') and 'raw' not in j:
                parquet = j.rsplit('.', 1)[0] + '.parquet'
                issues.append((pf, li, j, parquet, line.strip()[:100]))

if not issues:
    print('NO JSON REFERENCES FOUND in formal scripts')
else:
    print(f'FOUND {len(issues)} JSON references in formal scripts:')
    for f, li, j, p, ctx in issues:
        base = j.rsplit('/')[-1].rsplit('\\')[-1]
        par_base = base.replace('.json', '.parquet')
        print(f'  {f}:{li}')
        print(f'    ref: {j}')
        print(f'    change to: {par_base}')
        print(f'    code: {ctx}')
        print()

print()
print('--- ALSO check tests ---')
tst_scripts = [f for f in sorted(os.listdir(dp)) if f.endswith('.py') and f.startswith('tst_')]
tst_issues = []
for pf in tst_scripts:
    fp = os.path.join(dp, pf)
    content = open(fp, 'r', encoding='utf-8', errors='replace').read()
    lines = content.split('\n')
    for li, line in enumerate(lines, 1):
        if '.json' not in line: continue
        low = line.strip()
        if not ('import' in low or 'open(' in low or 'load(' in low): continue
        if 'raw_json/' in low: continue
        found = re.findall(r'[\x27\x22]([^\x27\x22]+\.json)[\x27\x22]', line)
        for j in found:
            if j.endswith('.json') and 'raw' not in j:
                p = j.rsplit('.',1)[0] + '.parquet'
                tst_issues.append((pf, li, j, p, line.strip()[:100]))

if tst_issues:
    print(f'TST scripts: {len(tst_issues)} references')
    for f, li, j, p, ctx in tst_issues:
        print(f'  {f}:{li} -> {j} (change to {p})')
else:
    print('TST scripts: none')
