# 扫描D盘所有可能存基本面数据的文件
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

scanned = []
base_dirs = [
    r'/home/hermes/.hermes/openclaw-archive/data',
    r'/home/hermes/.hermes/openclaw-archive/scripts/system',
    r'/home/hermes/.hermes/openclaw-archive',  # 可能有
    r'/home/hermes/.hermes/openclaw-archive/data',
]

keywords = ['fund', 'fin', 'info', 'basic', 'company', 'stock', 'sp500', 's&p', 
            'profile', 'stat', 'financial', 'income', 'balance', 'cashflow',
            'key', 'metrics', 'overview', 'valuation', 'pe', 'pb', 'roe']

for d in base_dirs:
    if not os.path.isdir(d):
        continue
    for root, dirs, files in os.walk(d):
        for f in files:
            if f.endswith('.json') or f.endswith('.parquet'):
                fpath = os.path.join(root, f)
                fsize = os.path.getsize(fpath)
                # 检查文件名是否含关键词
                matched = [kw for kw in keywords if kw in f.lower()]
                scanned.append({
                    'path': fpath,
                    'size': fsize,
                    'size_mb': round(fsize/1024/1024, 1),
                    'matched_kw': matched
                })

# 按大小排序
scanned.sort(key=lambda x: x['size'], reverse=True)

print(f"共扫描 {len(scanned)} 个文件\n")
print(f"{'文件名':60s} {'大小':>8s} {'匹配关键词':20s}")
print("-"*90)
for s in scanned[:40]:
    fn = s['path'][:60]
    kw = ','.join(s['matched_kw']) if s['matched_kw'] else '-'
    print(f"{fn:60s} {str(s['size_mb'])+'MB':>8s} {kw:20s}")

print("\n--- 未匹配关键词但>100MB的文件 ---")
for s in scanned:
    if not s['matched_kw'] and s['size_mb'] > 100:
        print(f"  {s['path'][:70]}  {s['size_mb']}MB")
