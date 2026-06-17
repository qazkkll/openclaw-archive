import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\admin\.openclaw\cron\jobs.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

targets = ['backup_midnight', 'quality_pool_midnight', 'archive_daily_0300', 'session_cleanup_0300', 'us_ml_progress']
for job in d['jobs']:
    name = job.get('name', '?')
    if name in targets:
        payload = job.get('payload', {})
        print(f'=== {name} ===')
        print(f'  会话目标: {job.get("sessionTarget")}')
        print(f'  payload.kind: {payload.get("kind")}')
        print(f'  payload.message: {str(payload.get("message",""))[:200]}')
        print(f'  timeout: {payload.get("timeoutSeconds")}')
        tools = payload.get('toolsAllow', [])
        print(f'  toolsAllow: {tools}')
        sched = job.get('schedule', {})
        print(f'  schedule: {json.dumps(sched, ensure_ascii=False)[:100]}')
        print()
