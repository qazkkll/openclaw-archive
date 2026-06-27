#!/usr/bin/env bash
# Falcon 收盘报告：持仓快照 + 今日交易回顾
# 纽约16:30运行 (HKT 04:30 Tue-Sat)
set -euo pipefail

cd /home/hermes/.hermes/openclaw-archive
PYTHON=/home/hermes/.hermes/hermes-agent/venv/bin/python3

echo "🦅 Falcon 收盘报告 $(date '+%Y-%m-%d %H:%M')"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 持仓报告
$PYTHON scripts/falcon/falcon_trade_exec.py --report 2>&1

echo ""

# 今日交易记录
TRADE_LOG="data/falcon/trades/trade_journal.jsonl"
if [ -f "$TRADE_LOG" ]; then
    TODAY=$(date +%Y-%m-%d)
    TODAY_TRADES=$(grep "$TODAY" "$TRADE_LOG" 2>/dev/null | wc -l)
    echo "📝 今日交易: ${TODAY_TRADES}笔"
    if [ "$TODAY_TRADES" -gt 0 ]; then
        grep "$TODAY" "$TRADE_LOG" | $PYTHON -c "
import json, sys
for line in sys.stdin:
    t = json.loads(line.strip())
    sym = t.get('symbol','?')
    side = t.get('side','?')
    qty = t.get('qty',0)
    price = t.get('price',0)
    reason = t.get('reason','')
    print(f'  {side} {sym} {qty}股 @ \${price:.2f} — {reason}')
"
    fi
else
    echo "📝 暂无交易记录"
fi
