#!/usr/bin/env python3
"""
🍤 对话日志记录器
使用: python3 scripts/log_conversation.py "<sender>" "<message>"
示例: python3 scripts/log_conversation.py "Andy" "你好"
"""
import sys, os, datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'conversation')
os.makedirs(LOG_DIR, exist_ok=True)

today = datetime.date.today().isoformat()
log_file = os.path.join(LOG_DIR, f"{today}.md")
archive_dir = os.path.join(LOG_DIR, 'archive')

# Archive files older than 30 days
for f in os.listdir(LOG_DIR):
    if f.endswith('.md') and f != f"{today}.md" and f != "README.md":
        try:
            fdate = datetime.date.fromisoformat(f.replace('.md',''))
            if (datetime.date.today() - fdate).days > 30:
                os.makedirs(archive_dir, exist_ok=True)
                os.rename(os.path.join(LOG_DIR, f), os.path.join(archive_dir, f))
        except:
            pass

def log(sender, message):
    ts = datetime.datetime.now().strftime('%H:%M')
    line = f"[{ts}] {sender}: {message}\n"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(line)
    print(f"✅ Logged: {line.strip()}")

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        log(sys.argv[1], ' '.join(sys.argv[2:]))
    else:
        # Read from stdin for piping
        sender = sys.argv[1] if len(sys.argv) > 1 else "🍤"
        for line in sys.stdin:
            if line.strip():
                log(sender, line.strip())
