#!/usr/bin/env python3
"""
🦅 Falcon Weekly Review — 每周+每月投资复盘
============================================
每周六自动运行，复盘：
1. 本周异动统计（L1/L2/L3）
2. 系统推荐 vs 实际操作对比
3. 推荐准确率
4. 本周/本月交易结果
5. 持仓表现排名
6. 下周关注点

输出: Telegram报告 + 存档

用法:
    python3 falcon_weekly_review.py            # 本周复盘
    python3 falcon_weekly_review.py --month    # 本月复盘
"""

import json
import sys
import glob
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import Counter

# ── Paths ──
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
ANALYSIS_DIR = DATA_DIR / "analysis"
TRADE_DIR = DATA_DIR / "trades"
JOURNAL_FILE = TRADE_DIR / "trade_journal.jsonl"
POSITIONS_FILE = TRADE_DIR / "positions.json"
REVIEW_DIR = DATA_DIR / "reviews"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def load_analysis_archive(days: int = 7) -> list:
    """加载最近N天的分析记录。"""
    records = []
    today = datetime.now()
    for i in range(days):
        date = (today - timedelta(days=i)).strftime("%Y%m%d")
        f = ANALYSIS_DIR / f"analysis_{date}.json"
        if f.exists():
            try:
                with open(f) as fh:
                    data = json.load(fh)
                records.extend(data)
            except Exception:
                pass
    return records


def load_trade_journal(days: int = 30) -> list:
    """加载交易日志。"""
    if not JOURNAL_FILE.exists():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    records = []
    with open(JOURNAL_FILE) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                if r.get("timestamp", "") >= cutoff:
                    records.append(r)
            except Exception:
                pass
    return records


def load_current_positions() -> dict:
    """加载当前持仓。"""
    if not POSITIONS_FILE.exists():
        return {}
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f).get("positions", {})
    except Exception:
        return {}


def get_alpaca_positions() -> list:
    """从Alpaca获取实际持仓。"""
    try:
        sys.path.insert(0, str(FALCON_DIR))
        from broker_adapter import get_broker
        broker = get_broker()
        return broker.get_positions()
    except Exception:
        return []


def analyze_recommendations(analyses: list) -> dict:
    """分析推荐统计。"""
    if not analyses:
        return {"total": 0, "by_level": {}, "by_rec": {}, "by_ticker": {}}

    by_level = Counter()
    by_rec = Counter()
    by_ticker = Counter()
    high_conf_recs = []

    for a in analyses:
        by_level[a.get("alert_level", "L1")] += 1
        by_rec[a.get("model_recommendation", "hold")] += 1
        by_ticker[a.get("ticker", "?")] += 1

        if a.get("model_recommendation") != "hold":
            high_conf_recs.append({
                "ticker": a["ticker"],
                "rec": a.get("model_recommendation", "hold"),
                "reasoning": a.get("model_reasoning", "")[:100],
                "news_context": a.get("news_context", "")[:80],
                "timestamp": a.get("timestamp", "")[:16],
            })

    return {
        "total": len(analyses),
        "by_level": dict(by_level),
        "by_rec": dict(by_rec),
        "by_ticker": dict(by_ticker.most_common(10)),
        "high_conf_recs": high_conf_recs,
    }


def analyze_trades(trades: list) -> dict:
    """分析交易结果。"""
    if not trades:
        return {"buys": 0, "sells": 0, "total_pnl": 0, "trades": []}

    buys = [t for t in trades if t.get("side") == "BUY"]
    sells = [t for t in trades if t.get("side") == "SELL"]

    total_pnl = sum(t.get("pnl_pct", 0) for t in sells)
    stop_losses = [t for t in sells if "止损" in t.get("reason", "")]
    expiries = [t for t in sells if "到期" in t.get("reason", "")]

    return {
        "buys": len(buys),
        "sells": len(sells),
        "stop_losses": len(stop_losses),
        "expiries": len(expiries),
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(sells) if sells else 0,
        "buys_list": buys,
        "sells_list": sells,
    }


def generate_report(days: int = 7) -> str:
    """生成复盘报告。"""
    analyses = load_analysis_archive(days)
    trades = load_trade_journal(days)
    positions = load_current_positions()
    alpaca_positions = get_alpaca_positions()

    rec_stats = analyze_recommendations(analyses)
    trade_stats = analyze_trades(trades)

    lines = []
    period = "本周" if days <= 7 else "本月"
    lines.append(f"🦅 **Falcon {period}复盘**")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d')} | 回顾{days}天")
    lines.append("")

    # ── 1. 异动统计 ──
    lines.append(f"📊 **异动统计**")
    if rec_stats["total"] > 0:
        level_map = {"L1": "🟡观察", "L2": "🟠关注", "L3": "🔴警报"}
        for level, count in sorted(rec_stats["by_level"].items()):
            lines.append(f"  {level_map.get(level, level)}: {count}次")

        rec_map = {"hold": "⏳持有", "reduce": "⚠️减仓", "stop_loss": "🛑止损", "take_profit": "💰止盈"}
        for rec, count in sorted(rec_stats["by_rec"].items(), key=lambda x: -x[1]):
            lines.append(f"  {rec_map.get(rec, rec)}: {count}次")

        # 最活跃ticker
        top_tickers = list(rec_stats["by_ticker"].items())[:5]
        if top_tickers:
            lines.append(f"  最活跃: {', '.join(f'{t}({c}次)' for t, c in top_tickers)}")
    else:
        lines.append(f"  无异动记录（分析存档为空）")
    lines.append("")

    # ── 2. 高置信度推荐 ──
    if rec_stats.get("high_conf_recs"):
        lines.append(f"🎯 **模型触发的操作信号**")
        for r in rec_stats["high_conf_recs"][:5]:
            rec_map = {"reduce": "减仓", "stop_loss": "止损", "expire": "到期"}
            lines.append(f"  {r['timestamp']} {r['ticker']}: {rec_map.get(r['rec'], r['rec'])}")
            if r.get("reasoning"):
                lines.append(f"    模型: {r['reasoning']}")
            if r.get("news_context"):
                lines.append(f"    新闻: {r['news_context'][:60]}")
        lines.append("")

    # ── 3. 交易结果 ──
    lines.append(f"📈 **交易结果**")
    if trade_stats["buys"] + trade_stats["sells"] > 0:
        lines.append(f"  买入: {trade_stats['buys']}笔 | 卖出: {trade_stats['sells']}笔")
        lines.append(f"  止损: {trade_stats['stop_losses']}次 | 到期: {trade_stats['expiries']}次")
        if trade_stats["sells"] > 0:
            lines.append(f"  平均盈亏: {trade_stats['avg_pnl']:+.1f}%")
        lines.append("")

        if trade_stats.get("sells_list"):
            lines.append(f"  **卖出明细:**")
            for s in trade_stats["sells_list"][-5:]:
                emoji = "🟢" if s.get("pnl_pct", 0) >= 0 else "🔴"
                lines.append(f"    {emoji} {s['symbol']} {s.get('pnl_pct', 0):+.1f}% | {s.get('reason', '')[:50]}")
    else:
        lines.append(f"  无交易记录")
    lines.append("")

    # ── 4. 当前持仓 ──
    lines.append(f"📦 **当前持仓** ({len(alpaca_positions)}只)")
    if alpaca_positions:
        total_value = sum(float(p.market_value) for p in alpaca_positions)
        for pos in sorted(alpaca_positions, key=lambda p: float(p.unrealized_plpc)):
            pnl = float(pos.unrealized_plpc) * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"  {emoji} {pos.symbol} {pos.qty}股 | {pnl:+.1f}% | ${float(pos.market_value):,.0f}")
        lines.append(f"  总持仓: ${total_value:,.0f}")
    else:
        lines.append(f"  无持仓")
    lines.append("")

    # ── 5. 系统推荐 vs 实际 ──
    if rec_stats.get("high_conf_recs") and trades:
        lines.append(f"🔍 **推荐 vs 实际**")
        rec_tickers = {r["ticker"] for r in rec_stats["high_conf_recs"]}
        actual_traded = {t.get("symbol") for t in trades}
        acted = rec_tickers & actual_traded
        missed = rec_tickers - actual_traded
        if acted:
            lines.append(f"  ✅ 已执行: {', '.join(acted)}")
        if missed:
            lines.append(f"  ❌ 未执行: {', '.join(missed)}")
        lines.append("")

    # ── 6. 下周关注 ──
    if alpaca_positions:
        lines.append(f"👀 **下周关注**")
        for pos in alpaca_positions:
            pnl = float(pos.unrealized_plpc) * 100
            if pnl <= -8:
                lines.append(f"  ⚠️ {pos.symbol} 亏损{pnl:+.1f}%，接近止损线")
            elif pnl >= 15:
                lines.append(f"  💰 {pos.symbol} 盈利{pnl:+.1f}%，考虑止盈")

    report = "\n".join(lines)

    # 存档
    review_file = REVIEW_DIR / f"review_{datetime.now().strftime('%Y%m%d')}.json"
    with open(review_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "period_days": days,
            "report": report,
            "stats": {
                "analysis_count": rec_stats["total"],
                "trade_count": trade_stats["buys"] + trade_stats["sells"],
                "position_count": len(alpaca_positions),
            },
        }, f, indent=2, ensure_ascii=False, default=str)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Falcon Weekly Review")
    parser.add_argument("--month", action="store_true", help="本月复盘(30天)")
    parser.add_argument("--days", type=int, default=None, help="自定义天数")
    args = parser.parse_args()

    if args.days:
        days = args.days
    elif args.month:
        days = 30
    else:
        days = 7

    report = generate_report(days)
    print(report)
