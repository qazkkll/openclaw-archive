# DEPRECATED - falcon_system/

**⚠️ This directory is DEPRECATED. Do NOT use these files for production.**

This directory contains the **V0.4.4-era modular Falcon architecture** (16-file modular layout). It has been replaced by the flat `scripts/falcon/` directory.

## What replaced it

| Component | Old Location | New Location |
|-----------|-------------|-------------|
| Scoring | `falcon_system/engine/scorer.py` | `scripts/falcon/falcon_score.py` (V0.4.6) |
| Trading | `falcon_system/trading/broker.py` | `scripts/falcon/falcon_trade_exec.py` |
| Gatekeeper | `falcon_system/trading/monitor.py` | `scripts/falcon/falcon_gatekeeper.py` |
| Observer | `falcon_system/trading/monitor.py` | `scripts/falcon/falcon_observer.py` |
| Daily Pipeline | `falcon_system/daily_pipeline.py` | `scripts/falcon/falcon_daily_update_all.py` |
| Dashboard | `falcon_system/dashboard/app.py` | `scripts/falcon/dashboard/` (unified) |
| Backtest | `falcon_system/engine/backtest.py` | `scripts/falcon/backtest/` (unified framework) |
| Config | `falcon_system/core/config.py` | Inline in each script / `falcon_config.py` |
| Data Manager | `falcon_system/core/data_manager.py` | Inline data loading in `falcon_score.py` |

## Why it was replaced

The modular architecture (`core/`, `engine/`, `trading/`, `dashboard/`) introduced unnecessary abstraction layers that made debugging, testing, and modification harder without providing meaningful flexibility gains. The flat `scripts/falcon/` directory is simpler, faster to navigate, and easier to maintain.

## Keep for reference only

These files are retained for historical reference. They document the V0.4.4 modular design decisions and may be useful for understanding the system's evolution.
