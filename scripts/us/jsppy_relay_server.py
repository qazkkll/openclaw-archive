#!/usr/bin/env python3
"""
jsppy HTTP 中继服务器
部署在大陆 VPS 上，香港服务器通过 HTTP 请求获取 A 股数据

用法:
  export JSPPY_TOKEN="your_token"
  python3 jsppy_relay_server.py [port]

支持接口:
  POST /hq       — 实时五档行情  body: {"codes": ["SH.600000"]}
  POST /history  — 历史K线       body: {"code": "SZ.000001", "start": "2026-01-01", "end": "2026-05-12", "freq": "D"}
  POST /index    — 指数K线       body: 同上
  GET  /ping     — 心跳检测
"""

import os
import sys
import json
import atexit
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# 设置中文本地化
os.environ['LANG'] = 'en_US.utf8'
os.environ['LC_ALL'] = 'en_US.utf8'

from jsppy import *

TOKEN = os.environ.get('JSPPY_TOKEN', '')
if not TOKEN:
    print("❌ 请设置 JSPPY_TOKEN 环境变量")
    sys.exit(1)

# 初始化连接
print(f"🔄 登录中...", flush=True)
login(TOKEN)
print(f"✅ 登录成功", flush=True)
atexit.register(login_out)


class JsppyHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器"""

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/ping':
            self._json_response({"ok": True, "type": "pong"})
        else:
            self._json_response({"ok": False, "error": "路径不存在"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path

        # 读取请求体
        length = int(self.headers.get('Content-Length', 0))
        if length == 0:
            self._json_response({"ok": False, "error": "请求体为空"}, 400)
            return

        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({"ok": False, "error": "JSON解析失败"}, 400)
            return

        try:
            if path == '/hq':
                result = self._handle_hq(data)
            elif path == '/history':
                result = self._handle_history(data)
            elif path == '/index':
                result = self._handle_index(data)
            elif path == '/tick':
                result = self._handle_tick(data)
            else:
                result = {"ok": False, "error": f"路径不存在: {path}"}
                self._json_response(result, 404)
                return
        except Exception as e:
            result = {"ok": False, "error": str(e)}

        self._json_response(result)

    def _handle_hq(self, data):
        codes = data.get('codes', [])
        if not codes:
            return {"ok": False, "error": "需要 codes 参数"}
        df = get_real_hq(codes)
        return {"ok": True, "type": "hq", "data": df_to_json(df), "count": len(codes)}

    def _handle_history(self, data):
        code = data.get('code', '')
        start = data.get('start', '')
        end = data.get('end', '')
        freq = data.get('freq', 'D')
        if not code:
            return {"ok": False, "error": "需要 code 参数"}
        df = get_history_data(code, start, end, freq)
        rows = df_to_json(df)
        return {"ok": True, "type": "history", "data": rows, "count": len(rows)}

    def _handle_index(self, data):
        code = data.get('code', '')
        start = data.get('start', '')
        end = data.get('end', '')
        freq = data.get('freq', 'D')
        if not code:
            return {"ok": False, "error": "需要 code 参数"}
        df = get_index_data(code, start, end, freq)
        rows = df_to_json(df)
        return {"ok": True, "type": "index", "data": rows, "count": len(rows)}

    def _handle_tick(self, data):
        code = data.get('code', '')
        date = data.get('date', '')
        if not code or not date:
            return {"ok": False, "error": "需要 code 和 date 参数"}
        df = get_tick(code, date)
        rows = df_to_json(df)
        return {"ok": True, "type": "tick", "data": rows, "count": len(rows)}

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode('utf-8'))

    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {format % args}", flush=True)


def df_to_json(df):
    if df is None or df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(str)
    return json.loads(df.to_json(orient="records", force_ascii=False))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

    server = HTTPServer(('0.0.0.0', port), JsppyHandler)
    print(f"🚀 jsppy 中继服务器运行在 http://0.0.0.0:{port}")
    print(f"    GET  /ping     — 心跳检测")
    print(f"    POST /hq       — 实时五档行情")
    print(f"    POST /history  — 历史K线")
    print(f"    POST /index    — 指数K线")
    print(f"    POST /tick     — 分笔成交")
    print(f"    Token: {TOKEN[:8]}...{TOKEN[-4:]}")
    print(f"    到期: {csd.stock.str_bg}")
    print(f"-----------------------------------------")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务器关闭")
        server.server_close()


if __name__ == '__main__':
    main()
