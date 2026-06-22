#!/usr/bin/env python3
"""
Dashboard API Server
提供 HTTP API 用于刷新持仓、评分等操作
"""
import json, os, subprocess, sys, logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# 日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser('~/.hermes/openclaw-archive/output/api.log'), mode='a'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('dashboard_api')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8080
PYTHON = sys.executable  # 使用当前运行的python，确保subprocess用同一个环境

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/api/refresh-portfolio':
            self.handle_refresh_portfolio()
        elif parsed.path == '/api/refresh-scores':
            self.handle_refresh_scores()
        elif parsed.path == '/api/refresh-all':
            self.handle_refresh_all()
        elif parsed.path == '/api/status':
            self.handle_status()
        elif parsed.path == '/' or parsed.path == '/dashboard.html':
            self.serve_dashboard()
        elif parsed.path.startswith('/output/'):
            self.serve_static(parsed.path)
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        """提供 dashboard.html"""
        dashboard_path = os.path.join(ROOT, 'dashboard.html')
        if os.path.exists(dashboard_path):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            with open(dashboard_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404, 'Dashboard not found')
    
    def serve_static(self, path):
        """提供静态文件（output目录）"""
        # 安全检查：防止路径遍历
        if '..' in path:
            self.send_error(403)
            return
        
        file_path = os.path.join(ROOT, path.lstrip('/'))
        if os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            if path.endswith('.json'):
                self.send_header('Content-Type', 'application/json')
            elif path.endswith('.html'):
                self.send_header('Content-Type', 'text/html')
            elif path.endswith('.css'):
                self.send_header('Content-Type', 'text/css')
            elif path.endswith('.js'):
                self.send_header('Content-Type', 'application/javascript')
            else:
                self.send_header('Content-Type', 'application/octet-stream')
            self.end_headers()
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)
    
    def handle_refresh_portfolio(self):
        """刷新持仓：调用 sync_portfolio_from_opend.py 一步到位"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        def run():
            try:
                log.info("Refresh: sync_portfolio_from_opend.py")
                r1 = subprocess.run(
                    [PYTHON, os.path.join(ROOT, 'scripts/sync_portfolio_from_opend.py')],
                    capture_output=True, text=True, timeout=30
                )
                log.info(f"Refresh: sync done, rc={r1.returncode}")
                if r1.returncode != 0:
                    log.error(f"sync failed: {r1.stderr[:300]}")
                
                # 重新生成 dashboard 数据
                log.info("Refresh: dashboard_engine.py")
                r2 = subprocess.run(
                    [PYTHON, os.path.join(ROOT, 'scripts/dashboard_engine.py')],
                    capture_output=True, text=True, timeout=60
                )
                log.info(f"Refresh: engine done, rc={r2.returncode}")
                
                result = {
                    "success": r1.returncode == 0,
                    "message": "持仓已刷新" if r1.returncode == 0 else "持仓同步失败",
                    "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                log.info("Refresh: ALL DONE")
            except Exception as e:
                log.error(f"Refresh FAILED: {e}")
                result = {"success": False, "message": str(e)}
            
            # 异步发送结果（这里简化处理）
        
        # 启动后台线程
        thread = threading.Thread(target=run)
        thread.start()
        
        # 立即返回
        self.wfile.write(json.dumps({
            "success": True,
            "message": "正在刷新持仓...",
            "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }).encode())
    
    def handle_refresh_scores(self):
        """刷新评分"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        def run():
            try:
                subprocess.run(
                     [PYTHON, os.path.join(ROOT, 'scripts/us/blueshield_v6_score.py'), '--top', '15'],
                    capture_output=True, text=True, timeout=300
                )
                subprocess.run(
                     [PYTHON, os.path.join(ROOT, 'scripts/us/arrow_v11_score.py'), '--top', '10'],
                    capture_output=True, text=True, timeout=300
                )
            except:
                pass
        
        thread = threading.Thread(target=run)
        thread.start()
        
        self.wfile.write(json.dumps({
            "success": True,
            "message": "正在刷新评分..."
        }).encode())
    
    def handle_refresh_all(self):
        """刷新全部"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        def run():
            try:
                # 持仓
                subprocess.run( [PYTHON, os.path.join(ROOT, 'scripts/us/futu_positions.py')], 
                             capture_output=True, timeout=60)
                # 评分
                subprocess.run( [PYTHON, os.path.join(ROOT, 'scripts/us/blueshield_v6_score.py'), '--top', '15'],
                             capture_output=True, timeout=300)
                subprocess.run( [PYTHON, os.path.join(ROOT, 'scripts/us/arrow_v11_score.py'), '--top', '10'],
                             capture_output=True, timeout=300)
                # 重新生成
                subprocess.run( [PYTHON, os.path.join(ROOT, 'scripts/dashboard_engine.py')],
                             capture_output=True, timeout=60)
            except:
                pass
        
        thread = threading.Thread(target=run)
        thread.start()
        
        self.wfile.write(json.dumps({
            "success": True,
            "message": "正在刷新全部数据..."
        }).encode())
    
    def handle_status(self):
        """返回状态"""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        self.wfile.write(json.dumps({
            "status": "running",
            "timestamp": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }).encode())
    
    def log_message(self, format, *args):
        pass  # 禁用日志

if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), DashboardHandler)
    print(f"🚀 Dashboard API Server running on http://0.0.0.0:{PORT}")
    print(f"   刷新持仓: http://localhost:{PORT}/api/refresh-portfolio")
    print(f"   刷新评分: http://localhost:{PORT}/api/refresh-scores")
    print(f"   刷新全部: http://localhost:{PORT}/api/refresh-all")
    server.serve_forever()
