import os, json

root = r'/home/hermes/.hermes/openclaw-archive'
report = {}

# 根目录文件
root_items = []
for f in sorted(os.listdir(root)):
    fp = os.path.join(root, f)
    if os.path.isfile(fp) and not f.startswith('.'):
        sz = round(os.path.getsize(fp) / 1024, 1)
        root_items.append({'name': f, 'size_kb': sz})
report['root'] = root_items

# scripts/
script_items = []
sp = os.path.join(root, 'scripts')
for f in sorted(os.listdir(sp)):
    if f.endswith('.py'):
        sz = round(os.path.getsize(os.path.join(sp, f)) / 1024, 1)
        script_items.append({'name': f, 'size_kb': sz})
report['scripts'] = script_items

# data/
data_items = []
dp = os.path.join(root, 'data')
for f in sorted(os.listdir(dp)):
    fp = os.path.join(dp, f)
    if os.path.isfile(fp):
        sz = round(os.path.getsize(fp) / (1024*1024), 2)
        data_items.append({'name': f, 'size_mb': sz})
    elif os.path.isdir(fp):
        total = 0
        children = []
        for sf in sorted(os.listdir(fp)):
            sfp = os.path.join(fp, sf)
            if os.path.isfile(sfp):
                total += os.path.getsize(sfp)
                children.append(sf)
        data_items.append({'name': f + '/', 'size_kb_total': round(total/1024, 1), 'file_count': len(children)})
report['data'] = data_items

# docs/
doc_items = []
dp2 = os.path.join(root, 'docs')
for f in sorted(os.listdir(dp2)):
    fp = os.path.join(dp2, f)
    if os.path.isfile(fp):
        sz = round(os.path.getsize(fp) / 1024, 1)
        doc_items.append({'name': f, 'size_kb': sz})
report['docs'] = doc_items

# skills/
skills_dirs = sorted(os.listdir(os.path.join(root, 'skills')))
report['skills'] = skills_dirs

# memory/
mem_items = []
mp = os.path.join(root, 'memory')
for f in sorted(os.listdir(mp)):
    fp = os.path.join(mp, f)
    if os.path.isfile(fp) and f.endswith('.md'):
        sz = round(os.path.getsize(fp) / 1024, 1)
        mem_items.append({'name': f, 'size_kb': sz})
report['memory'] = mem_items

with open(os.path.join(root, 'data', '_inventory_report.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 打印
print('文件清单完成，保存到 data/_inventory_report.json')
print()
print('=== 根目录文件 ===')
for f in report['root']: print(f'  {f["name"]:30s} {f["size_kb"]:>8.1f} KB')
print(f'\n=== scripts/ ({len(report["scripts"])} 个) ===')
for f in report['scripts']: print(f'  {f["name"]:35s} {f["size_kb"]:>8.1f} KB')
print(f'\n=== data/ ({len(report["data"])} 项) ===')
for f in report['data']:
    if 'size_mb' in f: print(f'  {f["name"]:35s} {f["size_mb"]:>8.1f} MB')
    else: print(f'  {f["name"]:35s} {f["size_kb_total"]:>8.1f} KB ({f["file_count"]} files)')
print(f'\n=== docs/ ({len(report["docs"])} 个) ===')
for f in report['docs']: print(f'  {f["name"]:35s} {f["size_kb"]:>8.1f} KB')
print(f'\n=== skills/ ({len(report["skills"])} 个) ===')
for f in report['skills']: print(f'  {f}')
print(f'\n=== memory/ ({len(report["memory"])} 个) ===')
for f in report['memory']: print(f'  {f["name"]:35s} {f["size_kb"]:>8.1f} KB')
