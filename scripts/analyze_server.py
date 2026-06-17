#!/usr/bin/env python3
"""
Hermes 分析服务器 — 一键触发分析

用法：
    python3 analyze_server.py              # 启动服务器 (端口 8080)
    python3 analyze_server.py --port 9000  # 自定义端口
    python3 analyze_server.py --open       # 启动后自动打开浏览器

浏览器打开 http://localhost:8080，点击"一键分析"按钮触发实时评分。
"""

import json, os, sys, time, argparse, threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")

sys.path.insert(0, SCRIPTS_DIR)

# 全局锁，防止并发分析
ANALYZE_LOCK = threading.Lock()
ANALYZE_STATUS = {"running": False, "last_run": None, "last_error": None}


class AnalyzeHandler(SimpleHTTPRequestHandler):
    """处理静态文件和分析API"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=OUTPUT_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/analyze":
            self.handle_analyze()
        elif path == "/api/status":
            self.handle_status()
        elif path == "/" or path == "/dashboard.html":
            self.path = "/dashboard.html"
            super().do_GET()
        else:
            super().do_GET()

    def handle_analyze(self):
        """执行分析"""
        if ANALYZE_STATUS["running"]:
            self.send_json({"status": "busy", "message": "分析正在进行中..."}, 429)
            return

        def run_analysis():
            ANALYZE_STATUS["running"] = True
            try:
                # 导入并执行分析
                from live_monitor import (
                    get_market_context, get_prev_market_context, save_market_context,
                    load_v9_data, load_v9_model, score_v9_lottery, score_shield_v3,
                    generate_actions, US_UNIVERSE
                )
                from dashboard_engine import build_dashboard_data
                from dashboard_renderer import generate_new_dashboard

                # 1. 市场数据
                context = get_market_context()
                prev_context = get_prev_market_context()

                # 2. 加载数据
                v9_data = load_v9_data()
                if v9_data is None:
                    ANALYZE_STATUS["last_error"] = "V9数据未就绪"
                    return

                v9_model = load_v9_model()
                if v9_model is None:
                    ANALYZE_STATUS["last_error"] = "V9模型未就绪"
                    return

                # 3. 评分
                arrow_top10, _ = score_v9_lottery(v9_data, v9_model, US_UNIVERSE)
                shield_tickers = list(set([a["ticker"] for a in arrow_top10] + US_UNIVERSE[:50]))
                shield_results = score_shield_v3(shield_tickers)

                # 4. 构建仪表盘数据
                data = build_dashboard_data(shield_results, arrow_top10, context, prev_context)

                # 5. 保存数据
                with open(os.path.join(OUTPUT_DIR, "live_data.json"), "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

                # 6. 生成 HTML
                html = generate_new_dashboard(data)
                with open(os.path.join(OUTPUT_DIR, "dashboard.html"), "w") as f:
                    f.write(html)

                # 7. 保存市场数据
                save_market_context(context)

                ANALYZE_STATUS["last_run"] = datetime.now().isoformat()
                ANALYZE_STATUS["last_error"] = None

            except Exception as e:
                ANALYZE_STATUS["last_error"] = str(e)
            finally:
                ANALYZE_STATUS["running"] = False

        thread = threading.Thread(target=run_analysis, daemon=True)
        thread.start()

        self.send_json({
            "status": "started",
            "message": "分析已启动，请等待...",
            "timestamp": datetime.now().isoformat(),
        })

    def handle_status(self):
        """查询分析状态"""
        self.send_json(ANALYZE_STATUS)

    def send_json(self, data, code=200):
        """发送 JSON 响应"""
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        """简化日志"""
        if "/api/" in str(args[0]) if args else False:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Hermes 分析服务器")
    parser.add_argument("--port", type=int, default=8080, help="端口号")
    parser.add_argument("--open", action="store_true", help="自动打开浏览器")
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), AnalyzeHandler)

    print(f"{'━' * 50}")
    print(f"🚀 Hermes 分析服务器")
    print(f"{'━' * 50}")
    print(f"📡 地址: http://localhost:{args.port}")
    print(f"📊 看板: http://localhost:{args.port}/dashboard.html")
    print(f"🔄 分析: http://localhost:{args.port}/api/analyze")
    print(f"{'━' * 50}")
    print(f"💡 点击看板上的「一键分析」按钮触发实时评分")
    print(f"   按 Ctrl+C 停止服务器")

    if args.open:
        import webbrowser
        webbrowser.open(f"http://localhost:{args.port}/dashboard.html")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务器已停止")
        server.server_close()


if __name__ == "__main__":
    main()
