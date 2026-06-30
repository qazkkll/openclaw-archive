#!/usr/bin/env python3
"""
🦅 Falcon Dashboard — 中控台
=============================
统一展示：模型状态、持仓、信号、数据新鲜度、交易记录。
输出格式：Telegram Markdown。

用法:
    python3 falcon_dashboard.py              # 完整看板
    python3 falcon_dashboard.py --brief      # 简要模式
    python3 falcon_dashboard.py --positions  # 只看持仓
    python3 falcon_dashboard.py --signals    # 只看信号
    python3 falcon_dashboard.py --freshness  # 只看数据新鲜度
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 添加路径
FALCON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FALCON_DIR.parent))

from falcon_system.core.config import CONFIG, DATA_DIR, TRADE_DIR
from falcon_system.core.data_manager import data_manager
from falcon_system.engine.scorer import ScoringEngine, Pricer, run_scoring
from falcon_system.trading.broker import get_broker, PositionManager
from falcon_system.trading.monitor import run_monitor_check


# ════════════════════════════════════════════════════════════════
# Dashboard组件
# ════════════════════════════════════════════════════════════════

def section_freshness() -> str:
    """数据新鲜度板块"""
    lines = []
    lines.append("📊 **数据新鲜度**")
    lines.append("")
    
    freshness = data_manager.get_all_freshness()
    all_fresh = True
    
    for name, status in freshness.items():
        if status.is_fresh:
            emoji = "✅"
        else:
            emoji = "❌"
            all_fresh = False
        
        age_str = f"{status.age_hours:.1f}h" if status.age_hours < 48 else f"{status.age_hours/24:.1f}d"
        lines.append(f"  {emoji} **{name}**: {status.source}")
        lines.append(f"     更新: {status.last_update.strftime('%Y-%m-%d %H:%M') if status.last_update else 'N/A'} | 年龄: {age_str} | 记录: {status.record_count:,}")
        
        if status.error:
            lines.append(f"     ⚠️ {status.error}")
    
    lines.append("")
    if all_fresh:
        lines.append("✅ 所有数据新鲜")
    else:
        lines.append("❌ 有数据过期，需要更新")
    
    return "\n".join(lines)


def section_vix() -> str:
    """VIX板块"""
    vix, vix_date = data_manager.get_latest_vix()
    if vix is None:
        return "⚠️ VIX数据不可用"
    
    emoji = "✅" if vix < CONFIG.model.vix_threshold else "❌"
    skip = "跳过买入" if vix > CONFIG.model.vix_threshold else "正常买入"
    
    return f"📊 **VIX**: {vix:.1f} ({vix_date}) {emoji}\n   阈值: {CONFIG.model.vix_threshold} → {skip}"


def section_account(broker) -> str:
    """账户板块"""
    try:
        account = broker.get_account()
        positions = broker.get_positions()
        
        total_value = sum(p.market_value for p in positions)
        total_pnl = sum(p.unrealized_plpc * p.market_value for p in positions)
        total_pnl_pct = total_pnl / total_value if total_value > 0 else 0
        
        lines = []
        lines.append("💰 **账户**")
        lines.append(f"  总资产: ${account['equity']:,.0f}")
        lines.append(f"  现金: ${account['cash']:,.0f}")
        lines.append(f"  持仓市值: ${total_value:,.0f}")
        lines.append(f"  持仓盈亏: ${total_pnl:+,.0f} ({total_pnl_pct:+.1%})")
        lines.append(f"  持仓数量: {len(positions)}只")
        
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 账户信息获取失败: {e}"


def section_positions(broker) -> str:
    """持仓板块"""
    try:
        positions = broker.get_positions()
        
        if not positions:
            return "📦 **持仓**: 无"
        
        lines = []
        lines.append(f"📦 **持仓 ({len(positions)}只)**")
        lines.append("")
        
        # 按盈亏排序
        positions.sort(key=lambda p: p.unrealized_plpc, reverse=True)
        
        for pos in positions:
            pnl = pos.unrealized_plpc * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            
            lines.append(f"  {emoji} **{pos.symbol}** | {pos.qty}股")
            lines.append(f"     成本: ${pos.avg_entry_price:.2f} → 现价: ${pos.current_price:.2f}")
            lines.append(f"     盈亏: {pnl:+.1f}% (${pos.unrealized_plpc * pos.market_value:+,.0f})")
            lines.append(f"     市值: ${pos.market_value:,.0f}")
            
            # 加载本地元数据
            local_data = _load_local_position(pos.symbol)
            if local_data:
                if local_data.get("score"):
                    lines.append(f"     评分: {local_data['score']:.4f}")
                if local_data.get("target_sell"):
                    lines.append(f"     目标: ${local_data['target_sell']:.2f}")
                if local_data.get("stop_loss"):
                    lines.append(f"     止损: ${local_data['stop_loss']:.2f}")
            
            lines.append("")
        
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 持仓信息获取失败: {e}"


def section_signals() -> str:
    """信号板块"""
    try:
        result = run_scoring()
        
        lines = []
        lines.append(f"🎯 **今日信号** ({result.date})")
        lines.append(f"  模型: {result.model_version} | 耗时: {result.scoring_time_seconds:.1f}秒")
        vix_str = f'{result.vix_value:.1f}' if result.vix_value else 'N/A'
        lines.append(f'  VIX: {vix_str} | {"❌跳过" if result.vix_skip else "✅正常"}')
        lines.append(f"  宇宙: {result.universe_size}只")
        lines.append("")
        
        # Top 10
        top10 = result.signals[:10]
        for i, sig in enumerate(top10, 1):
            emoji = {"🟢🟢": "🟢🟢", "🟢": "🟢", "🟡": "🟡", "🔴": "🔴"}.get(sig.signal_type, "❓")
            lines.append(f"  {i:>2}. {emoji} **{sig.ticker}** | 分数{sig.score:.4f} | 排名{sig.rank_pct*100:.1f}%")
            lines.append(f"      价格: ${sig.close:.2f}")
            
            # 因子详情
            factors = sig.factors
            top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
            factor_str = " | ".join(f"{k}:{v:.0%}" for k, v in top_factors)
            lines.append(f"      因子: {factor_str}")
            
            if sig.target_buy:
                lines.append(f"      目标买入: ${sig.target_buy:.2f} | 止损: ${sig.stop_loss:.2f}")
            
            lines.append("")
        
        # 统计
        green2 = sum(1 for s in result.signals if s.signal_type == "🟢🟢")
        green1 = sum(1 for s in result.signals if s.signal_type == "🟢")
        yellow = sum(1 for s in result.signals if s.signal_type == "🟡")
        
        lines.append(f"  统计: 🟢🟢{green2} | 🟢{green1} | 🟡{yellow}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"⚠️ 信号获取失败: {e}"


def section_alerts() -> str:
    """异动板块"""
    alerts_file = DATA_DIR / "alerts" / "pending.json"
    if not alerts_file.exists():
        return "✅ **异动**: 无"
    
    try:
        with open(alerts_file) as f:
            alerts = json.load(f)
        
        if not alerts:
            return "✅ **异动**: 无"
        
        lines = []
        lines.append(f"⚠️ **异动 ({len(alerts)}条)**")
        
        for alert in alerts[-5:]:  # 最近5条
            emoji = {"L1": "📈", "L2": "⚠️", "L3": "🛑"}.get(alert.get("level", ""), "❓")
            lines.append(f"  {emoji} {alert.get('message', '')}")
            if alert.get("action_required"):
                lines.append(f"     建议: {alert['action_required']}")
        
        return "\n".join(lines)
    except:
        return "✅ **异动**: 无"


def section_model_performance() -> str:
    """模型表现板块"""
    journal_file = TRADE_DIR / "trade_journal.jsonl"
    if not journal_file.exists():
        return "📈 **模型表现**: 无交易记录"
    
    try:
        trades = []
        with open(journal_file) as f:
            for line in f:
                if line.strip():
                    trades.append(json.loads(line))
        
        if not trades:
            return "📈 **模型表现**: 无交易记录"
        
        # 统计
        buys = [t for t in trades if t.get("type") == "BUY"]
        sells = [t for t in trades if t.get("type") == "SELL"]
        
        wins = [s for s in sells if s.get("pnl_pct", 0) > 0]
        losses = [s for s in sells if s.get("pnl_pct", 0) <= 0]
        
        total_pnl = sum(s.get("pnl_pct", 0) for s in sells)
        avg_pnl = total_pnl / len(sells) if sells else 0
        win_rate = len(wins) / len(sells) if sells else 0
        
        lines = []
        lines.append("📈 **模型表现**")
        lines.append(f"  总交易: {len(buys)}买 {len(sells)}卖")
        lines.append(f"  胜率: {win_rate:.0%}")
        lines.append(f"  平均盈亏: {avg_pnl:+.1%}")
        
        if wins:
            avg_win = sum(s.get("pnl_pct", 0) for s in wins) / len(wins)
            lines.append(f"  平均盈利: {avg_win:+.1%}")
        if losses:
            avg_loss = sum(s.get("pnl_pct", 0) for s in losses) / len(losses)
            lines.append(f"  平均亏损: {avg_loss:+.1%}")
        
        return "\n".join(lines)
    except:
        return "📈 **模型表现**: 数据读取失败"


def _load_local_position(symbol: str) -> dict:
    """加载本地持仓元数据"""
    pos_file = TRADE_DIR / "positions.json"
    if pos_file.exists():
        try:
            with open(pos_file) as f:
                data = json.load(f)
            return data.get("positions", {}).get(symbol, {})
        except:
            pass
    return {}


# ════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Falcon Dashboard")
    parser.add_argument("--brief", action="store_true", help="简要模式")
    parser.add_argument("--positions", action="store_true", help="只看持仓")
    parser.add_argument("--signals", action="store_true", help="只看信号")
    parser.add_argument("--freshness", action="store_true", help="只看数据新鲜度")
    args = parser.parse_args()
    
    # 初始化broker
    try:
        broker = get_broker()
    except Exception as e:
        print(f"⚠️ Broker初始化失败: {e}")
        broker = None
    
    lines = []
    lines.append(f"🦅 **Falcon Dashboard** — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"⚙️ 模型: {CONFIG.model.version} | 调仓: {CONFIG.model.hold_days}天 | 止损: {CONFIG.model.stop_loss*100:.0f}%")
    lines.append("")
    
    if args.freshness:
        lines.append(section_freshness())
    elif args.positions and broker:
        lines.append(section_positions(broker))
    elif args.signals:
        lines.append(section_signals())
    else:
        # 完整看板
        lines.append(section_freshness())
        lines.append("")
        lines.append(section_vix())
        lines.append("")
        
        if broker:
            lines.append(section_account(broker))
            lines.append("")
            lines.append(section_positions(broker))
            lines.append("")
        
        if not args.brief:
            lines.append(section_signals())
            lines.append("")
        
        lines.append(section_alerts())
        lines.append("")
        lines.append(section_model_performance())
    
    print("\n".join(lines))


if __name__ == "__main__":
    main()
