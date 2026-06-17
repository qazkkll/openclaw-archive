#!/usr/bin/env python3
"""往群里发消息（直连Telegram API，绕过sessions_send bug）"""
import urllib.request, json, sys

BOT_TOKEN = "7792764974:AAFrFrZ3JAjdhkCsphy2N-gd99U5puRywUI"
CHAT_ID = "-1003900769838"

def send(msg):
    data = json.dumps({"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = json.loads(urllib.request.urlopen(req).read())
    if resp.get("ok"):
        print(f"✅ 已发送 (msg_id={resp['result']['message_id']})")
    else:
        print(f"❌ 失败: {resp.get('description')}")

if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()
    if msg: send(msg)