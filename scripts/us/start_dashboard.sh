#!/bin/bash
# Hermes量化看板 — 启动脚本
# 用法: ./start_dashboard.sh
# 手机访问: 启动后查看输出的URL

ROOT="$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"
cd "$ROOT"

PORT=8899
CLOUDFLARED="$HOME/.local/bin/cloudflared"

echo "🛡️ Hermes量化看板启动中..."

# 生成最新看板
echo "📊 生成看板..."
python3 scripts/us/generate_pro_dashboard_v7.py

# 杀掉旧进程
pkill -f "python3 -m http.server $PORT" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

# 启动HTTP服务器
echo "🌐 启动HTTP服务器 (port $PORT)..."
python3 -m http.server $PORT --bind 0.0.0.0 &
HTTP_PID=$!
sleep 1

# 启动Cloudflare隧道
echo "🔗 启动Cloudflare隧道..."
$CLOUDFLARED tunnel --url http://localhost:$PORT 2>&1 &
TUNNEL_PID=$!
sleep 5

# 获取URL
URL=$(grep -o 'https://[a-zA-Z0-9-]*.trycloudflare.com' /proc/$TUNNEL_PID/fd/2 2>/dev/null || echo "")

if [ -z "$URL" ]; then
    # 尝试从日志获取
    sleep 3
    URL=$(ps aux | grep cloudflared | grep -o 'https://[a-zA-Z0-9-]*.trycloudflare.com' | head -1)
fi

echo ""
echo "✅ 看板已启动！"
echo "📱 手机访问: ${URL:-请查看cloudflared输出}"
echo "💻 本地访问: http://localhost:$PORT/dashboard.html"
echo ""
echo "按 Ctrl+C 停止"

# 等待
wait
