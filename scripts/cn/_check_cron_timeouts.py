import re, sys
sys.stdout.reconfigure(encoding='utf-8')
logfile = r'C:\Users\admin\AppData\Local\Temp\openclaw\openclaw-2026-06-11.log'
timeouts = []
with open(logfile, 'r', encoding='utf-8') as f:
    for line in f:
        if 'timeout' in line.lower() or 'stalled' in line.lower() or 'stall' in line.lower():
            # extract time
            m = re.search(r'"time":"([^"]+)"', line)
            ts = m.group(1) if m else '??'
            # extract message
            m2 = re.search(r'"message":"([^"]+)"', line)
            msg = m2.group(1)[:120] if m2 else ''
            # extract job name
            m3 = re.search(r'jobName":"([^"]+)"', line)
            job = m3.group(1) if m3 else 'N/A'
            timeouts.append((ts, job, msg))

print(f"Found {len(timeouts)} timeout/stall entries:")
for ts, job, msg in timeouts:
    print(f"  {ts} | {job:40s} | {msg[:100]}")
