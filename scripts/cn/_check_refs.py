import os, re

sp = r'/home/hermes/.hermes/openclaw-archive\scripts'

old_names = [
    'layer1_daily', 'layer1_market_state', 'layer3_4_scoring',
    's1_scan', 's2_candidates',
    'score_engine', 'u5_scan',
]

py_files = sorted([f for f in os.listdir(sp) if f.endswith('.py') and not f.startswith('_')])
issues = []

for pf in py_files:
    content = open(os.path.join(sp, pf), 'r', encoding='utf-8', errors='replace').read()
    for old in old_names:
        if old in content:
            lines = content.split('\n')
            for li, line in enumerate(lines, 1):
                if old in line and ('import' in line or 'from' in line or 'open' in line.lower() or 'load' in line.lower()):
                    issues.append((pf, li, old, line.strip()[:120]))

print(f'检查{len(py_files)}个文件')
if issues:
    print(f'\n发现{len(issues)}处嵌套引用：')
    for f, li, old_name, context in issues:
        print(f'  {f}:{li} — 引用了旧名"{old_name}"')
        print(f'    代码: {context}')
else:
    print('✅ 无嵌套引用问题')
