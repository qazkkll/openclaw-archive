"""
🍤 统一通知层 — 所有脚本从这里发消息

用法:
    from notify import send
    send("消息内容")
    
切换渠道只需改这个文件（目前TG，后续可加飞书/邮件）
"""
import urllib.request, json, os

# ===== 配置（以后可放 config/）=====
BOT_TOKEN = '7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI'
ADMIN_CHAT_ID = '7908145929'

def send(text, chat_id=None):
    """发送通知到用户"""
    _send_telegram(text, chat_id or ADMIN_CHAT_ID)

def _send_telegram(text, chat_id):
    """走Telegram Bot API"""
    url = 'https://api.telegram.org/bot' + BOT_TOKEN + '/sendMessage'
    data = json.dumps({
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML'
    }).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:
        print(f'[notify] TG发送失败: {e}')
        return False

def send_to_group(text, group_id):
    """发送到群聊"""
    return _send_telegram(text, group_id)
