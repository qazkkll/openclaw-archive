import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'C:\Users\admin\.openclaw\cron\jobs.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for j in d['jobs']:
    sched = j['schedule']['kind']
    if j['schedule']['kind'] == 'cron':
        sched = f"cron({j['schedule'].get('expr','?')})"
    elif j['schedule']['kind'] == 'every':
        sched = f"every({j['schedule']['everyMs']//60000}min)"
    elif j['schedule']['kind'] == 'at':
        sched = f"at({j['schedule'].get('at','?')})"
    status = 'ENABLED' if j.get('enabled') else 'DISABLED'
    runtime = j.get('lastDurationMs', 0)
    if runtime:
        runtime_s = f"last={runtime}ms"
    else:
        runtime_s = "no runs"
    print(f"  [{status}] {j['name']:40s} | {sched:25s} | {runtime_s}")
