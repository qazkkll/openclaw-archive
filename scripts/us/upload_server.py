#!/usr/bin/env python3
"""简易文件上传接收服务器 — 接收本地计算节点传回的结果"""
import http.server
import cgi
import os
import json
from datetime import datetime

UPLOAD_DIR = '/home/admin/.openclaw/workspace/incoming'
PORT = 9999

class UploadHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )
        
        file_item = form['file']
        if file_item and file_item.filename:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_name = f"{ts}_{file_item.filename}"
            path = os.path.join(UPLOAD_DIR, safe_name)
            
            with open(path, 'wb') as f:
                f.write(file_item.file.read())
            
            size_kb = os.path.getsize(path) / 1024
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'ok', 'file': safe_name, 'size_kb': round(size_kb, 1)}
            self.wfile.write(json.dumps(response).encode())
            print(f'📥 收到: {safe_name} ({size_kb:.1f}KB)')
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"status":"error","message":"no file"}')

if __name__ == '__main__':
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    server = http.server.HTTPServer(('0.0.0.0', PORT), UploadHandler)
    print(f'🚀 上传服务启动: http://0.0.0.0:{PORT}')
    print(f'📁 文件保存到: {UPLOAD_DIR}')
    server.serve_forever()
