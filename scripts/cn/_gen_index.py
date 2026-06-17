"""
_gen_index.py — 生成 INDEX.md + data/code_index.json
同时覆盖:
  - workspace/scripts/*.py
  - /home/hermes/.hermes/openclaw-project/scripts/system/*.py
  - /home/hermes/.hermes/openclaw-project/data/models/*.json
"""
import os, datetime, json

ROOT = r'/home/hermes/.hermes/openclaw-archive'
SCRIPTS_DIR = os.path.join(ROOT, 'scripts')
ML_DIR = r'/home/hermes/.hermes/openclaw-archive/scripts/system'
MODELS_DIR = r'/home/hermes/.hermes/openclaw-archive/data\models'
ROOT_DATA = os.path.join(ROOT, 'data')

# ── 1. 收集C盘脚本 ──
files = sorted([f for f in os.listdir(SCRIPTS_DIR) if f.endswith('.py')])

groups = {}
for f in files:
    if f.startswith('a1_'): k='a1'
    elif f.startswith('a_'): k='a_'
    elif f.startswith('daily_'): k='daily'
    elif f.startswith('dl_'): k='dl'
    elif f.startswith('sys_'): k='sys'
    elif f.startswith('us_'): k='us'
    elif f.startswith('tst_'): k='tst'
    elif f.startswith('tmp_'): k='tmp'
    elif f.startswith('_'): k='_internal'
    else: k='其他'
    groups.setdefault(k, []).append(f)

# ── 2. 收集D盘ML脚本 ──
ml_files = sorted([f for f in os.listdir(ML_DIR) if f.endswith('.py')])

# ── 3. 收集模型文件 ──
model_files = []
for root, dirs, files2 in os.walk(MODELS_DIR):
    for f in files2:
        if f.endswith('.json') and not f.startswith('tmp_'):
            rel = os.path.relpath(os.path.join(root, f), MODELS_DIR)
            sz = os.path.getsize(os.path.join(root, f))
            model_files.append((rel, sz))

# ── 4. 生成 code_index.json ──
code_index = {}
for f in files:
    fpath = os.path.join(SCRIPTS_DIR, f)
    try:
        content = open(fpath, 'r', encoding='utf-8', errors='replace').read()
        desc = ''
        for line in content.split('\n')[:8]:
            ls = line.strip()
            if ls.startswith('#'):
                desc = ls.lstrip('#').strip()[:100]
                break
            if '"""' in line:
                parts = content.split('"""', 2)
                if len(parts) >= 2:
                    ds = parts[1].strip().split('\n')[0]
                    desc = ds[:100]
                    break
    except:
        desc = ''
    code_index[f] = {
        'path': f'scripts/{f}',
        'desc': desc,
        'location': 'C'
    }

for f in ml_files:
    fpath = os.path.join(ML_DIR, f)
    try:
        content = open(fpath, 'r', encoding='utf-8', errors='replace').read()
        desc = ''
        for line in content.split('\n')[:8]:
            ls = line.strip()
            if ls.startswith('#'):
                desc = ls.lstrip('#').strip()[:100]
                break
            if '"""' in line:
                parts = content.split('"""', 2)
                if len(parts) >= 2:
                    ds = parts[1].strip().split('\n')[0]
                    desc = ds[:100]
                    break
    except:
        desc = ''
    code_index[f] = {
        'path': f'/home/hermes/.hermes/openclaw-project/scripts/system/{f}',
        'desc': desc,
        'location': 'D'
    }

for rel, sz in model_files:
    code_index[f'model:{rel}'] = {
        'path': f'/home/hermes/.hermes/openclaw-project/data/models/{rel}',
        'desc': f'ML模型文件 ({sz//1024}KB)',
        'location': 'D'
    }

idx_path = os.path.join(ROOT, 'INDEX.md')

# ── 5. 生成 INDEX.md ──
with open(idx_path, 'w', encoding='utf-8') as idx:
    idx.write('# INDEX.md — 脚本索引\n\n')
    idx.write(f'最后更新: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    idx.write('## 命名规范\n\n')
    idx.write('每个文件必须遵守: 项目前缀_序列号_功能名.py\n\n')
    idx.write('| 前缀 | 项目 | 序列规则 |\n')
    idx.write('|---|:---|---:|\n')
    idx.write('| a1_ | A1资金流模型 | a1_layerN_功能名.py |\n')
    idx.write('| a_ | A股个股分析 | a_功能名.py |\n')
    idx.write('| us_v5s_ | 美股V5三模型 | us_v5s_sN_功能名.py |\n')
    idx.write('| us_ | 美股其他工具 | us_功能名.py |\n')
    idx.write('| daily_ | 日常事务 | daily_功能名.py |\n')
    idx.write('| dl_ | 数据下载 | dl_功能名.py |\n')
    idx.write('| sys_ | 系统维护 | sys_功能名.py |\n')
    idx.write('| tst_ | 测试/实验 | tst_项目_功能名.py |\n')
    idx.write('| tmp_ | 临时(不过夜) | tmp_功能名.py |\n')
    idx.write('| _ | 内部工具 | _功能名.py(不可直接调用) |\n\n')
    
    idx.write('---\n\n')
    
    order = ['a1','a_','us','daily','dl','sys','tst','tmp','_internal','其他']
    for gname in order:
        if gname not in groups: continue
        ncomply = sum(1 for f in groups[gname])
        idx.write(f'## {gname} ({ncomply}个)\n\n')
        for f in sorted(groups[gname]):
            e = code_index.get(f, {})
            desc = e.get('desc', '')
            idx.write(f'- **{f}**')
            if desc:
                idx.write(f' — {desc}')
            idx.write('\n')
        idx.write('\n')
    
    idx.write('---\n\n')
    idx.write(f'## ML训练脚本 (/home/hermes/.hermes/openclaw-project/scripts/system/, {len(ml_files)}个)\n\n')
    for f in ml_files:
        e = code_index.get(f, {})
        desc = e.get('desc', '')
        idx.write(f'- **{f}**')
        if desc:
            idx.write(f' — {desc}')
        idx.write('\n')
    idx.write('\n')
    
    idx.write(f'## 模型文件 (/home/hermes/.hermes/openclaw-project/data/models/, {len(model_files)}个)\n\n')
    for rel, sz in sorted(model_files):
        idx.write(f'- **{rel}** ({sz//1024}KB)\n')
    idx.write('\n')
    
    idx.write('---\n\n')
    idx.write('## 变更日志\n\n')
    idx.write('| 时间 | 操作 | 详情 |\n')
    idx.write('|:---|---:|---|\n')

# ── 6. 写回 code_index.json（C盘+D盘都存） ──
ci_path_c = os.path.join(ROOT_DATA, 'code_index.json')
ci_path_d = os.path.join(r'/home/hermes/.hermes/openclaw-archive/data', 'code_index.json')
with open(ci_path_c, 'w', encoding='utf-8') as f:
    json.dump(code_index, f, ensure_ascii=False, indent=2)
with open(ci_path_d, 'w', encoding='utf-8') as f:
    json.dump(code_index, f, ensure_ascii=False, indent=2)

print(f'INDEX.md 已生成 ({len(files)}个C盘脚本)')
print(f'ML脚本: {len(ml_files)}个 | 模型: {len(model_files)}个')
print(f'code_index.json 已写入C盘+盘中（共{len(code_index)}条）')
