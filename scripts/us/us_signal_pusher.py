#!/usr/bin/env python3
"""us_signal_pusher.py — 纯 Python 信号推送器，零 AI token

流程:
  1. 读 realtime_signals.json（watch_dashboard.py 产出）
  2. 读 signal_tracking.json（已发送记录）
  3. 找出未发送的新信号
  4. 推送 Telegram 消息（直接 Bot API）
  5. 更新 tracking

用法:
  python scripts/us_signal_pusher.py
  建议 cron 频率: 每 10-30 分钟
"""

import json, os, sys, time, datetime

# ─── 配置 ─────────────────────────────────────────────
BASE_DIR = '/home/hermes/.hermes/openclaw-archive/data'
SIGNALS_FILE = os.path.join(BASE_DIR, 'realtime_signals.json')
TRACKING_FILE = os.path.join(BASE_DIR, 'signal_tracking.json')
# 从 config.json 读 bot token（不硬编码）
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
BOT_TOKEN = None
CHAT_ID = '7908145929'
try:
    with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
        _cfg = json.load(f)
    BOT_TOKEN = _cfg.get('channels', {}).get('telegram', {}).get('botToken', '')
    aid = _cfg.get('channels', {}).get('telegram', {}).get('account', '')
    if aid and aid.startswith('telegram:'):
        CHAT_ID = aid.split(':', 1)[1]
except Exception:
    pass
if not BOT_TOKEN:
    BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')

# ─── 加载信号 ─────────────────────────────────────────
def load_signals():
    if not os.path.exists(SIGNALS_FILE):
        return []
    try:
        with open(SIGNALS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('signals', [])
    except (json.JSONDecodeError, IOError) as e:
        print(f'[pusher] load signals error: {e}')
        return []

# ─── 加载/更新 tracking ───────────────────────────────
def load_tracking():
    if not os.path.exists(TRACKING_FILE):
        return {'sent_fingerprints': []}
    try:
        with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {'sent_fingerprints': []}

def save_tracking(tracking):
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, ensure_ascii=False, indent=2)

# ─── 指纹去重 ──────────────────────────────────────────
def make_fingerprint(sig):
    """用 type+code 做指纹，同类型同代码不重复发"""
    return f"{sig.get('type')}:{sig.get('code')}"

# ─── Telegram 推送 ────────────────────────────────────
def push_telegram(message):
    """通过 Bot API 发送消息"""
    if not BOT_TOKEN:
        print('[pusher] No BOT_TOKEN, cannot send')
        return False
    if not message.strip():
        return True
    
    import urllib.request
    import urllib.parse
    
    url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendMessage'
    data = urllib.parse.urlencode({
        'chat_id': CHAT_ID,
        'text': message,
        'parse_mode': 'HTML',
    }).encode()
    
    retries = 2
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'})
            resp = urllib.request.urlopen(req, timeout=15)
            result = json.loads(resp.read().decode())
            if result.get('ok'):
                print(f'[pusher] sent {len(message)} chars to Telegram')
                return True
            else:
                print(f'[pusher] Telegram API error: {result}')
        except Exception as e:
            print(f'[pusher] attempt {attempt+1}/{retries} failed: {e}')
            if attempt < retries - 1:
                time.sleep(3)
    return False

# ─── 格式化信号消息 ────────────────────────────────────
def format_signals(new_signals):
    """多条信号合并成一条消息"""
    parts = []
    for s in new_signals:
        parts.append(s.get('msg', ''))
    
    if not parts:
        return ''
    
    # 单条直接发
    if len(parts) == 1:
        return parts[0]
    
    # 多条合并（最多5条防超长）
    MAX_MSGS = 5
    lines = parts[:MAX_MSGS]
    if len(parts) > MAX_MSGS:
        lines.append(f'\n... 还有 {len(parts) - MAX_MSGS} 条信号')
    
    return '\n\n'.join(lines)

# ─── 主流程 ────────────────────────────────────────────
def main():
    print(f'[pusher] {datetime.datetime.now().isoformat()} start')
    
    signals = load_signals()
    if not signals:
        print('[pusher] no signals')
        return
    
    tracking = load_tracking()
    sent_fps = set(tracking.get('sent_fingerprints', []))
    
    # 去重
    new_signals = []
    for s in signals:
        fp = make_fingerprint(s)
        if fp not in sent_fps:
            new_signals.append(s)
            sent_fps.add(fp)
    
    if not new_signals:
        print('[pusher] no new signals')
        return
    
    print(f'[pusher] {len(new_signals)} new signals to push')
    
    # 推送
    msg = format_signals(new_signals)
    if msg:
        ok = push_telegram(msg)
    else:
        ok = False
    
    # 更新 tracking（即使推送失败也标记已发，防止重复推送）
    if new_signals:
        tracking['sent_fingerprints'] = list(sent_fps)
        tracking['updated_at'] = datetime.datetime.now().isoformat()
        save_tracking(tracking)
        print(f'[pusher] tracking updated: {len(sent_fps)} total fingerprints')
    
    if not ok:
        print('[pusher] WARNING: push may have failed')
        sys.exit(1)
    else:
        print('[pusher] done')

if __name__ == '__main__':
    main()
