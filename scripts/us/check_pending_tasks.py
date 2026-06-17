#!/usr/bin/env python3
"""检查是否有待处理的任务通知。用于主session启动/心跳时调用"""
import json, os

NOTIF_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "data", "task_notifications.json")

def get_pending():
    try:
        with open(NOTIF_FILE) as f:
            notifs = json.load(f)
    except:
        return []
    return notifs if isinstance(notifs, list) else []

def clear_pending():
    with open(NOTIF_FILE, "w") as f:
        json.dump([], f)

def analyze_notifications(notifs):
    """整理通知摘要"""
    if not notifs:
        return None
    
    lines = []
    lines.append(f"📋 有 {len(notifs)} 个任务完成待分析\n")
    
    for n in notifs:
        tid = n.get("task_id", "?")
        r = n.get("result", {})
        
        # 提取结果消息
        msg = ""
        result_obj = r.get("result", {})
        if isinstance(result_obj, dict):
            msg = result_obj.get("message", "")
        if not msg:
            msg = r.get("message", "")
        if not msg:
            msg = r.get("status", "")
        
        # 提取其他关键字段
        extra = ""
        for k in ["note", "conclusion", "summary"]:
            v = r.get(k, "") or (result_obj.get(k, "") if isinstance(result_obj, dict) else "")
            if v:
                extra = f" | {v}"
                break
        
        lines.append(f"  ✅ {tid}: {msg}{extra}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    notifs = get_pending()
    if notifs:
        print(analyze_notifications(notifs))
    else:
        print("📭 无待处理任务通知")
