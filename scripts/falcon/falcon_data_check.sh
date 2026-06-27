#!/usr/bin/env bash
# Falcon 每日数据更新 — HKT 16:00 (纽约收盘后12小时)
# 检查数据新鲜度 → 过期则自动更新 → 输出报告
set -euo pipefail

cd /home/hermes/.hermes/openclaw-archive
PYTHON=/home/hermes/.hermes/hermes-agent/venv/bin/python3

echo "📅 Falcon 每日数据检查 $(date '+%Y-%m-%d %H:%M')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. 新鲜度检查
REPORT=$($PYTHON scripts/falcon/check_data_fresh.py 2>&1)
echo "$REPORT"
echo ""

# 2. 如果过期，自动更新
if echo "$REPORT" | grep -q "🔴\|🚨"; then
    echo "⚠️ 数据过期，自动更新中..."
    $PYTHON scripts/falcon/update_price_data.py 2>&1 | tail -5
    echo ""
    
    # 3. 重新检查
    echo "更新后复查:"
    $PYTHON scripts/falcon/check_data_fresh.py 2>&1
elif echo "$REPORT" | grep -q "⚠️.*滞后"; then
    echo "⚠️ 数据轻微滞后，尝试更新..."
    $PYTHON scripts/falcon/update_price_data.py 2>&1 | tail -5
else
    echo "✅ 数据正常，无需更新"
fi
