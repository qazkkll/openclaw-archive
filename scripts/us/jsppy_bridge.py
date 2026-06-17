#!/usr/bin/env python3
"""
jsppy Bridge — 持久进程，通过 stdin/stdout JSON 与 Node.js 通信
用法: python3 scripts/jsppy_bridge.py <token>
输出: 一行JSON结束（Node.js读取最后一行即可）
"""

import os
import sys
import json
import traceback

os.environ['LANG'] = 'zh_CN.UTF-8'
os.environ['LC_ALL'] = 'zh_CN.UTF-8'

# 先初始化 jsppy（会打印一些杂音到stdout，但我们马上就会切掉）
from jsppy import *

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "需要token参数"}), flush=True)
        return

    token = sys.argv[1]

    # 登录
    login(token)

    # 正常输出一行的 OK 标记
    print(json.dumps({"ok": True, "type": "login_ok"}), flush=True)

    # 持久命令循环
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "JSON解析失败"}), flush=True)
            continue

        try:
            result = handle_action(cmd.get("action", ""), cmd.get("data", {}))
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()}), flush=True)

    login_out()


def handle_action(action, data):
    if action == "hq":
        codes = data.get("codes", [])
        df = get_real_hq(codes)
        return {"ok": True, "type": "hq", "data": df_to_json(df)}

    elif action == "kzz":
        codes = data.get("codes", [])
        df = get_real_kzz(codes)
        return {"ok": True, "type": "kzz", "data": df_to_json(df)}

    elif action == "history":
        df = get_history_data(
            data.get("code", ""),
            data.get("start", ""),
            data.get("end", ""),
            data.get("freq", "D")
        )
        return {"ok": True, "type": "history", "data": df_to_json(df)}

    elif action == "index":
        df = get_index_data(
            data.get("code", ""),
            data.get("start", ""),
            data.get("end", ""),
            data.get("freq", "D")
        )
        return {"ok": True, "type": "index", "data": df_to_json(df)}

    elif action == "tick":
        df = get_tick(data.get("code", ""), data.get("date", ""))
        return {"ok": True, "type": "tick", "data": df_to_json(df)}

    elif action == "ping":
        return {"ok": True, "type": "pong"}

    elif action == "logout":
        login_out()
        return {"ok": True, "type": "logged_out"}

    else:
        return {"ok": False, "error": f"未知指令: {action}"}


def df_to_json(df):
    if df is None or df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(str)
    return json.loads(df.to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    main()
