import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'C:\Users\admin\.openclaw\cron\jobs.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
for j in d['jobs']:
    print(f"{j.get('id','?')[:8]} | {j.get('name')}")
