"""
修复所有嵌套引用 + 写入INDEX.md变更日志
"""
import os, datetime

sp = r'/home/hermes/.hermes/openclaw-archive\scripts'

# 修复映射: 旧文件名 → 新文件名
fix_map = {
    'from score_engine import': 'from us_score_engine import',
    'import score_engine': 'import us_score_engine',
    'from layer3_4_scoring import': 'from a1_layer3_4_scoring import',
    'import layer3_4_scoring': 'import a1_layer3_4_scoring',
    'os.path.join(WORKSPACE, "scripts", "us_s1_scan.py")': 'os.path.join(WORKSPACE, "scripts", "us_v5s_s1_scan.py")',
    "WORKSPACE + '/data/s2_candidates_latest.json'": "WORKSPACE + '/data/us_v5s_s2_candidates_latest.json'",
}

# 实际文件修改记录
changes = []

py_files = sorted([f for f in os.listdir(sp) if f.endswith('.py') and not f.startswith('_')])

for pf in py_files:
    fp = os.path.join(sp, pf)
    content = open(fp, 'r', encoding='utf-8', errors='replace').read()
    original = content
    
    for old_str, new_str in fix_map.items():
        if old_str in content:
            content = content.replace(old_str, new_str)
    
    if content != original:
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(content)
        changes.append(pf)
        print(f'  OK {pf}: fixed')

# 更新INDEX.md的变更日志
idx_path = os.path.join(sp, '..', 'INDEX.md')
idx_content = open(idx_path, 'r', encoding='utf-8').read()

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
log_entries = []

# 加日志行
log_entries.append(f'| {now} | 批量更名 | 86个py文件按`项目前缀_序列_功能名`规范统一命名 |')
log_entries.append(f'| {now} | 修复引用 | 修复{len(changes)}个文件的嵌套旧名引用 |')

# 在变更日志表前面加
idx_content = idx_content.replace('## 变更日志\n\n| 时间 | 操作 | 详情 |\n|:---|---:|---|\n',
                                   '## 变更日志\n\n| 时间 | 操作 | 详情 |\n|:---|---:|---|\n' + '\n'.join(log_entries) + '\n')

with open(idx_path, 'w', encoding='utf-8') as f:
    f.write(idx_content)

print(f'\n引用修复完成: {len(changes)}个文件')
print(f'变更日志已追加到INDEX.md')
