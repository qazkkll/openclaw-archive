#!/usr/bin/env bash
# Falcon 开盘执行：检查数据 → 更新 → 评分 → 交易 → Telegram报告
# 纽约9:30开盘时运行 (HKT 21:30 Mon-Fri)
set -euo pipefail

cd /home/hermes/.hermes/openclaw-archive
PYTHON=/home/hermes/.hermes/hermes-agent/venv/bin/python3
LOG_DIR=data/falcon/logs
mkdir -p "$LOG_DIR"

echo "🦅 Falcon 开盘执行 $(date '+%Y-%m-%d %H:%M')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 0: 数据新鲜度检查
echo "📅 检查数据新鲜度..."
FRESH=$($PYTHON scripts/falcon/check_data_fresh.py 2>&1 | head -1)
echo "  $FRESH"

if echo "$FRESH" | grep -q "🔴\|🚨"; then
    echo "  ⚠️ 数据过期，自动更新..."
    $PYTHON scripts/falcon/update_price_data.py 2>&1 | tail -3
fi

# Step 1: 评分
echo ""
echo "📊 评分中..."
$PYTHON scripts/falcon/falcon_score.py --universe spx 2>&1 | tee "$LOG_DIR/score_$(date +%Y%m%d).log"
SCORE_EXIT=${PIPESTATUS[0]}

if [ $SCORE_EXIT -ne 0 ]; then
    echo "❌ 评分失败 (exit $SCORE_EXIT)"
    exit 1
fi

# Step 2: 交易执行
echo ""
echo "💹 交易执行中..."
$PYTHON scripts/falcon/falcon_trade_exec.py 2>&1 | tee "$LOG_DIR/trade_$(date +%Y%m%d).log"

echo ""
echo "✅ 完成 $(date '+%H:%M')"
