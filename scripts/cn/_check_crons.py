import json, sys
sys.stdout.reconfigure(encoding='utf-8')
cronfile = r'C:\Users\admin\.openclaw\cron\jobs.json'
with open(cronfile, 'r', encoding='utf-8') as f:
    jobs = json.load(f)
print(json.dumps(jobs, indent=2, ensure_ascii=False))
