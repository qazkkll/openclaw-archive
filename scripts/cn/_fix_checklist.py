import sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'/home/hermes/.hermes/openclaw-archive\scripts\sys_open_checklist.py'

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: jsonl instead of json for experience_log
old1 = '        exp = json.load(f)'
new1 = '        exp = [json.loads(line) for line in f if line.strip()]'
assert old1 in content, 'old1 not found'
content = content.replace(old1, new1, 1)
print('fix 1 done')

# Fix 2: len(exp['experiences']) -> len(exp)
old2 = "    print(f\"\\n[经验库] {len(exp['experiences'])}条\")"
new2 = "    print(f\"\\n[经验库] {len(exp)}条\")"
if old2 not in content:
    # try without escaping
    old2 = '    print(f"\\n[经验库] {len(exp[\'experiences\'])}条")'
    new2 = '    print(f"\\n[经验库] {len(exp)}条")'
assert old2 in content, 'old2 not found'
content = content.replace(old2, new2, 1)
print('fix 2 done')

# Fix 3: exp['experiences'][-3:] -> exp[-3:]
old3 = "    for e in exp['experiences'][-3:]:"
new3 = '    for e in exp[-3:]:'
assert old3 in content, 'old3 not found'
content = content.replace(old3, new3, 1)
print('fix 3 done')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('ALL DONE')
