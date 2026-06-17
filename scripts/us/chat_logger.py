#!/usr/bin/env python3
"""🍤 聊天记录器 v5 — 保存30天可读对话日志，满月复盘"""
import os, json, json
from datetime import datetime, timedelta

SESSIONS_DIR = '/home/admin/.openclaw/agents/main/sessions'
CHAT_LOG_DIR = '/home/admin/.openclaw/workspace/logs/chat'
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

def extract_messages(session_file):
    msgs = {}
    if not os.path.exists(session_file): return msgs
    try:
        with open(session_file) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: msg = json.loads(line)
                except: continue
                if msg.get('type') != 'message': continue
                
                m = msg.get('message', {})
                role = m.get('role', '')
                content = m.get('content', '')
                ts = msg.get('timestamp', '')[:19]
                mid = msg.get('id', ts)
                
                text = ''
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get('type') == 'text':
                            text = c.get('text', '')[:500]
                            if text: break
                elif isinstance(content, str):
                    text = content[:500]
                
                if text and role:
                    sender = '👤 Andy' if role == 'user' else '🍤 小钳'
                    msgs[mid] = f'{ts} {sender}: {text}'
    except: pass
    return msgs

def save_chat_log():
    now = datetime.now()
    cutoff = now - timedelta(hours=720)  # 30天，满月复盘用
    all_msgs = {}
    
    for f in os.listdir(SESSIONS_DIR):
        if not f.endswith('.jsonl') or f.endswith('.lock') or 'trajectory' in f:
            continue
        fpath = os.path.join(SESSIONS_DIR, f)
        if datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
            continue
        all_msgs.update(extract_messages(fpath))
    
    if not all_msgs: return 0
    
    today = now.strftime('%Y-%m-%d')
    with open(os.path.join(CHAT_LOG_DIR, f'{today}.md'), 'w', encoding='utf-8') as f:
        f.write(f'# 🍤 聊天记录 · {today}\n> 保留最近3天 | ⚠️ 仅供Andy查阅\n\n')
        f.write('\n'.join(sorted(all_msgs.values())))
    
    for f in os.listdir(CHAT_LOG_DIR):
        if f.endswith('.md'):
            fpath = os.path.join(CHAT_LOG_DIR, f)
            if datetime.fromtimestamp(os.path.getmtime(fpath)) < cutoff:
                os.remove(fpath)
    return len(all_msgs)

if __name__ == '__main__':
    c = save_chat_log()
    print(f'✅ {c}条消息')
