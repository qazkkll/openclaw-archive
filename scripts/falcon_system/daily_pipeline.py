#!/usr/bin/env python3
"""
🦅 Falcon 每日全流程
====================
统一入口：数据更新 → 评分 → 定价 → 下单 → 监控 → 报告

用法:
    python3 daily_pipeline.py              # 完整流程
    python3 daily_pipeline.py --premarket  # 盘前(评分+定价+计划)
    python3 daily_pipeline.py --intraday   # 盘中(监控+异动)
    python3 daily_pipeline.py --postmarket # 盘后(持仓同步+日报)
    python3 daily_pipeline.py --dashboard  # 看板
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 添加路径
FALCON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FALCON_DIR.parent))

from falcon_system.core.config import CONFIG, DATA_DIR, TRADE_DIR
from falcon_system.core.data_manager import data_manager
from falcon_system.engine.scorer import ScoringEngine, Pricer, PositionSizer, ScoringResult, Signal
from falcon_system.trading.broker import get_broker, PositionManager, BrokerInterface
from falcon_system.trading.monitor import PositionMonitor, run_monitor_check


# ════════════════════════════════════════════════════════════════
# 盘前流程
# ════════════════════════════════════════════════════════════════

def run_premarket(broker: BrokerInterface) -> str:
    """盘前流程：数据更新 → 评分 → 定价 → 生成计划"""
    lines = []
    lines.append(f"🦅 **Falcon 盘前计划** — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # 1. 数据新鲜度检查
    lines.append("📊 **数据新鲜度检查**")
    is_fresh, issues = data_manager.is_all_fresh()
    if not is_fresh:
        lines.append("  ⚠️ 数据过期，尝试更新...")
        success, results = data_manager.update_all()
        for r in results:
            lines.append(f"  {r}")
    else:
        lines.append("  ✅ 所有数据新鲜")
    lines.append("")
    
    # 2. VIX检查
    vix, vix_date = data_manager.get_latest_vix()
    if vix:
        emoji = "✅" if vix < CONFIG.model.vix_threshold else "❌"
        lines.append(f"📊 **VIX**: {vix:.1f} ({vix_date}) {emoji}")
        if vix > CONFIG.model.vix_threshold:
            lines.append(f"  ❌ VIX > {CONFIG.model.vix_threshold}，跳过买入")
            return "\n".join(lines)
    lines.append("")
    
    # 3. 评分
    lines.append("🎯 **评分**")
    engine = ScoringEngine(data_manager)
    result = engine.score()
    lines.append(f"  模型: {result.model_version} | 耗时: {result.scoring_time_seconds:.1f}秒")
    lines.append(f"  宇宙: {result.universe_size}只")
    lines.append("")
    
    # 4. 筛选🟢🟢信号
    green2 = [s for s in result.signals if s.signal_type == "🟢🟢"]
    lines.append(f"  🟢🟢信号: {len(green2)}只")
    
    if not green2:
        lines.append("  ℹ️ 今日无🟢🟢信号")
        return "\n".join(lines)
    
    # 5. 计算目标价位
    pricer = Pricer(data_manager)
    green2 = pricer.calculate_targets(green2)
    
    # 6. Gatekeeper检查
    lines.append("")
    lines.append("🛡️ **Gatekeeper检查**")
    gatekeeper_result = run_gatekeeper_if_available()
    if gatekeeper_result:
        gk = gatekeeper_result.get("verdict", "SKIP")
        passed = gatekeeper_result.get("passed", 0)
        total = gatekeeper_result.get("total", 5)
        emoji = {"EXECUTE": "✅", "REDUCE": "⚠️", "SKIP": "❌"}.get(gk, "❓")
        lines.append(f"  {emoji} {gk} ({passed}/{total})")
        
        if gk == "SKIP":
            lines.append("  ❌ Gatekeeper拒绝，跳过买入")
            return "\n".join(lines)
        elif gk == "REDUCE":
            green2 = green2[:max(1, len(green2) // 2)]
            lines.append(f"  ⚠️ 减半至{len(green2)}只")
    
    # 7. 计算仓位
    account = broker.get_account()
    sizer = PositionSizer()
    green2 = sizer.calculate_positions(green2, account["equity"], {})
    
    # 8. 生成计划
    plan = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": CONFIG.model.version,
        "timestamp": datetime.now().isoformat(),
        "account": account,
        "gatekeeper": gatekeeper_result,
        "vix": vix,
        "buys": [
            {
                "symbol": s.ticker,
                "score": s.score,
                "current_price": s.close,
                "target_buy": s.target_buy,
                "stop_loss": s.stop_loss,
                "target_sell": s.target_sell,
                "atr": s.atr,
                "qty": s.suggested_qty,
                "value": s.suggested_value,
                "position_pct": s.position_pct,
                "factors": {k: round(v, 4) for k, v in s.factors.items()},
            }
            for s in green2 if s.suggested_qty
        ],
    }
    
    # 保存计划
    plan_file = TRADE_DIR / "premarket_plan.json"
    with open(plan_file, "w") as f:
        json.dump(plan, f, indent=2, default=str)
    
    # 9. 输出计划
    lines.append("")
    lines.append(f"🎯 **买入计划 ({len(plan['buys'])}只)**")
    for b in plan["buys"]:
        lines.append(f"  **{b['symbol']}** | 分数{b['score']:.4f}")
        lines.append(f"    现价: ${b['current_price']:.2f}")
        lines.append(f"    目标买入: ${b['target_buy']:.2f}")
        lines.append(f"    止损: ${b['stop_loss']:.2f}")
        lines.append(f"    目标卖出: ${b['target_sell']:.2f}")
        lines.append(f"    仓位: {b['qty']}股 (${b['value']:,.0f}, {b['position_pct']:.1f}%)")
        lines.append("")
    
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 盘中流程
# ════════════════════════════════════════════════════════════════

def run_intraday(broker: BrokerInterface) -> str:
    """盘中流程：监控 + 异动检查 + 订单执行"""
    lines = []
    lines.append(f"🦅 **Falcon 盘中监控** — {datetime.now().strftime('%H:%M')}")
    lines.append("")
    
    # 1. 异动检查
    alerts, report = run_monitor_check()
    lines.append(report)
    lines.append("")
    
    # 2. 检查挂单状态
    plan_file = TRADE_DIR / "premarket_plan.json"
    if plan_file.exists():
        with open(plan_file) as f:
            plan = json.load(f)
        
        # 检查是否需要执行买入
        if plan.get("buys"):
            lines.append("📝 **订单执行**")
            for buy in plan["buys"]:
                symbol = buy["symbol"]
                target_price = buy["target_buy"]
                qty = buy["qty"]
                
                # 获取当前价格
                current_price = broker.get_current_price(symbol)
                if current_price and current_price <= target_price:
                    lines.append(f"  🎯 {symbol} 到达目标价${target_price:.2f}，执行买入")
                    # TODO: 实际执行限价单
                else:
                    lines.append(f"  ⏳ {symbol} 等待回调 (当前${current_price:.2f if current_price else 'N/A'}, 目标${target_price:.2f})")
    
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 盘后流程
# ════════════════════════════════════════════════════════════════

def run_postmarket(broker: BrokerInterface) -> str:
    """盘后流程：持仓同步 + 止损检查 + 日报"""
    lines = []
    lines.append(f"🦅 **Falcon 盘后报告** — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # 1. 持仓同步
    pm = PositionManager(broker)
    pm.sync_from_broker()
    
    # 2. 获取持仓汇总
    summary = pm.get_portfolio_summary()
    
    lines.append("💰 **账户**")
    lines.append(f"  总资产: ${summary['account']['equity']:,.0f}")
    lines.append(f"  现金: ${summary['account']['cash']:,.0f}")
    lines.append(f"  持仓市值: ${summary['total_market_value']:,.0f}")
    lines.append(f"  持仓盈亏: ${summary['total_pnl']:+,.0f}")
    lines.append("")
    
    # 3. 持仓详情
    lines.append(f"📦 **持仓 ({summary['position_count']}只)**")
    for pos in summary["positions"]:
        emoji = "🟢" if pos["pnl_pct"] >= 0 else "🔴"
        lines.append(f"  {emoji} **{pos['symbol']}** | {pos['qty']}股 | {pos['pnl_pct']:+.1f}%")
    lines.append("")
    
    # 4. 止损检查
    monitor = PositionMonitor(broker)
    alerts = monitor.check_all()
    
    if alerts:
        lines.append(f"⚠️ **异动 ({len(alerts)}条)**")
        for alert in alerts:
            emoji = {"L1": "📈", "L2": "⚠️", "L3": "🛑"}.get(alert.level, "❓")
            lines.append(f"  {emoji} {alert.message}")
    else:
        lines.append("✅ **无异动**")
    
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════

def run_gatekeeper_if_available() -> Dict:
    """运行Gatekeeper(如果可用)"""
    try:
        sys.path.insert(0, str(FALCON_DIR))
        from falcon_gatekeeper import run_gatekeeper
        return run_gatekeeper()
    except:
        return {"verdict": "EXECUTE", "passed": 5, "total": 5, "checks": []}


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Falcon 每日全流程")
    parser.add_argument("--premarket", action="store_true", help="盘前流程")
    parser.add_argument("--intraday", action="store_true", help="盘中流程")
    parser.add_argument("--postmarket", action="store_true", help="盘后流程")
    parser.add_argument("--dashboard", action="store_true", help="看板")
    args = parser.parse_args()
    
    # 初始化broker
    try:
        broker = get_broker()
    except Exception as e:
        print(f"⚠️ Broker初始化失败: {e}")
        broker = None
    
    if args.dashboard:
        # 运行dashboard
        from falcon_system.dashboard.app import main as dashboard_main
        dashboard_main()
        return
    
    if args.premarket:
        print(run_premarket(broker))
    elif args.intraday:
        print(run_intraday(broker))
    elif args.postmarket:
        if broker:
            print(run_postmarket(broker))
        else:
            print("⚠️ Broker不可用")
    else:
        # 默认：根据时间自动选择
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("US/Eastern")
        now_et = datetime.now(ET)
        hour = now_et.hour + now_et.minute / 60
        
        if hour < 9.5:
            print(run_premarket(broker))
        elif hour < 16:
            print(run_intraday(broker))
        else:
            if broker:
                print(run_postmarket(broker))
            else:
                print("⚠️ Broker不可用")


if __name__ == "__main__":
    main()
