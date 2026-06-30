#!/bin/bash
# Falcon系统启动脚本
cd /home/hermes/.hermes/openclaw-archive
export PYTHONPATH=/home/hermes/.hermes/openclaw-archive/scripts

case "$1" in
    dashboard)
        python3 scripts/falcon_system/dashboard/app.py "${@:2}"
        ;;
    premarket)
        python3 scripts/falcon_system/daily_pipeline.py --premarket "${@:2}"
        ;;
    intraday)
        python3 scripts/falcon_system/daily_pipeline.py --intraday "${@:2}"
        ;;
    postmarket)
        python3 scripts/falcon_system/daily_pipeline.py --postmarket "${@:2}"
        ;;
    score)
        python3 -c "
import sys
sys.path.insert(0, 'scripts')
from falcon_system.engine.scorer import run_scoring
result = run_scoring()
print(f'日期: {result.date}')
print(f'信号数: {len(result.signals)}')
for i, s in enumerate(result.signals[:10], 1):
    print(f'{i}. {s.signal_type} {s.ticker} | 分数{s.score:.4f} | \${s.close:.2f}')
"
        ;;
    monitor)
        python3 -c "
import sys
sys.path.insert(0, 'scripts')
from falcon_system.trading.monitor import run_monitor_check
alerts, report = run_monitor_check()
print(report)
"
        ;;
    freshness)
        python3 -c "
import sys
sys.path.insert(0, 'scripts')
from falcon_system.core.data_manager import data_manager
is_fresh, issues = data_manager.is_all_fresh()
print('✅ 所有数据新鲜' if is_fresh else '❌ 有数据过期')
for issue in issues:
    print(f'  • {issue}')
"
        ;;
    *)
        echo "用法: $0 {dashboard|premarket|intraday|postmarket|score|monitor|freshness} [args]"
        exit 1
        ;;
esac
