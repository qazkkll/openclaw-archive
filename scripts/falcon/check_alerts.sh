#!/usr/bin/env bash
# ═══════════════════════════════════════════════════
# Falcon Alert Checker — 零token cron脚本
# ═══════════════════════════════════════════════════
# 每5分钟检查是否有新的异动告警。
# 有告警 → stdout输出(触发agent推送)
# 无告警 → 静默退出(不花token)
#
# cron配置:
#   */5 21-8 * * 1-5 /path/to/check_alerts.sh
#   (覆盖盘前7-9:30 + 盘中9:30-16 + 盘后16-20, ET时间转换为HKT)

set -euo pipefail

ALERTS_FILE="/home/hermes/.hermes/openclaw-archive/data/falcon/alerts/pending.json"

# 检查文件是否存在且非空
if [ ! -f "$ALERTS_FILE" ] || [ ! -s "$ALERTS_FILE" ]; then
    exit 0  # 无告警, 静默退出
fi

# 读取告警
ALERTS=$(cat "$ALERTS_FILE")

# 检查是否为空数组
if [ "$ALERTS" = "[]" ] || [ -z "$ALERTS" ]; then
    rm -f "$ALERTS_FILE"
    exit 0
fi

# 输出告警 (stdout → cron deliver到Telegram)
echo "🦅 Falcon 异动告警"
echo "━━━━━━━━━━━━━━━━━━"
echo "$ALERTS" | /home/hermes/.hermes/hermes-agent/venv/bin/python3 -c "
import json, sys
alerts = json.load(sys.stdin)
for a in alerts:
    ts = a.get('timestamp', '?')[:16]
    print(f'  [{ts}] {a[\"message\"]}')
print(f'\n共 {len(alerts)} 条告警')
"

# 清空已处理的告警
echo "[]" > "$ALERTS_FILE"
