#!/usr/bin/env python3
"""
推荐模板引擎 — 从config.json读取配置，不硬编码
"""

import json, os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.json")

# ── 加载配置 ──
def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)

CFG = load_config()


def traffic_light(score, model_type="shield"):
    """红绿灯 — 从config读取阈值"""
    if model_type == "shield":
        thresholds = CFG["display"]["traffic_light"]
        for emoji, rule in thresholds.items():
            if score >= rule["min_score"]:
                return emoji, rule["label"]
    else:
        thresholds = CFG["display"]["arrow_light"]
        for emoji, rule in thresholds.items():
            if score >= rule["min_prob"]:
                return emoji, rule["label"]
    return "⚪", "未知"


def sector_emoji(sector):
    sector_map = {
        "Technology": "💻", "Healthcare": "🏥", "Financial": "🏦",
        "Industrial": "🏭", "Consumer Cyclical": "🛍️",
        "Consumer Defensive": "🛡️", "Energy": "⚡",
        "Utilities": "🔌", "Real Estate": "🏠",
        "Basic Materials": "🧱", "Communication Services": "📡",
    }
    return sector_map.get(sector, "📊")


def format_pct(val, show_sign=True):
    if val is None: return "—"
    sign = "+" if val > 0 and show_sign else ""
    return f"{sign}{val:.1f}%"


def generate_actions(shield_list, arrow_list):
    """调仓建议 — 从config读取规则"""
    actions = []
    seen = set()
    rules = CFG["portfolio_actions"]["rules"]
    
    # 蓝盾规则
    for s in shield_list[:5]:
        for rule in rules:
            if rule["condition"].startswith("shield_score"):
                threshold = int(rule["condition"].split(">=")[1])
                if s["score"] >= threshold and s["ticker"] not in seen:
                    reason = rule["reason"].format(score=s["score"], rsi=s.get("rsi", "—"), prob="—")
                    actions.append({"ticker": s["ticker"], "action": rule["action"], "reason": reason})
                    seen.add(s["ticker"])
                    break
    
    # 绿箭规则
    for a in arrow_list[:3]:
        for rule in rules:
            if rule["condition"].startswith("arrow_prob"):
                threshold = float(rule["condition"].split(">=")[1])
                if a["prob"] >= threshold and a["ticker"] not in seen:
                    reason = rule["reason"].format(score="—", rsi="—", prob=f"{a['prob']*100:.0f}%")
                    actions.append({"ticker": a["ticker"], "action": rule["action"], "reason": reason})
                    seen.add(a["ticker"])
                    break
    
    # RSI过热
    for s in shield_list[:10]:
        if s.get("rsi", 50) > 70 and s["ticker"] not in seen:
            actions.append({"ticker": s["ticker"], "action": "HOLD", "reason": f"RSI {s['rsi']:.0f}超买"})
            seen.add(s["ticker"])
    
    return actions


def generate_risk_warnings(shield_list, arrow_list):
    """风险提示 — 从config读取规则"""
    warnings = []
    rules = CFG["risk_rules"]
    
    for rule in rules:
        if rule["type"] == "sector_concentration" and shield_list:
            from collections import Counter
            sectors = [s.get("sector", "Unknown") for s in shield_list[:10]]
            top_sector, top_count = Counter(sectors).most_common(1)[0]
            if top_count >= rule["threshold"]:
                warnings.append(rule["message"].format(sector=top_sector, count=top_count, tickers=""))
        
        elif rule["type"] == "rsi_overbought" and shield_list:
            high_rsi = [s["ticker"] for s in shield_list[:5] if s.get("rsi", 50) > rule["threshold"]]
            if high_rsi:
                warnings.append(rule["message"].format(tickers=", ".join(high_rsi)))
        
        elif rule["type"] == "arrow_weak" and arrow_list:
            if arrow_list[0].get("prob", 0) < rule["threshold"]:
                warnings.append(rule["message"])
    
    return warnings


def generate_report(scores, market_context=None):
    """生成标准化报告"""
    now = scores.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))
    market = scores.get("market", "US")
    shield_list = scores.get("shield", [])
    arrow_list = scores.get("arrow", [])
    context = market_context or {}
    
    shield_name = CFG["models"]["shield"]["name"]
    arrow_name = CFG["models"]["arrow"]["name"]
    market_name = "🇺🇸 美股" if market == "US" else "🇨🇳 A股"
    
    lines = []
    lines.append(f"{'━'*40}")
    lines.append(f"📊 {market_name} 盘中评分  {now}")
    lines.append(f"{'━'*40}")
    
    # 市场概览
    if context:
        lines.append("")
        lines.append("📈 市场概览")
        for k, v in context.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    
    # 蓝盾
    if shield_list:
        lines.append("━" * 40)
        lines.append(f"🔵 {shield_name} · 趋势跟踪")
        lines.append("━" * 40)
        lines.append(f"{'排名':<4} {'标的':<6} {'分数':<5} {'信号':<8} {'现价':<10} {'日涨跌':<8} {'RSI':<5}")
        lines.append("─" * 60)
        
        for i, s in enumerate(shield_list[:CFG["scoring"]["shield"]["top_n"]], 1):
            light, signal = traffic_light(s["score"], "shield")
            lines.append(
                f" {light} {i:<2} {s['ticker']:<6} {s['score']:<5} "
                f"{signal:<8} ${s.get('price', 0):<9} "
                f"{format_pct(s.get('daily_return')):<8} {s.get('rsi', '—')}"
            )
        lines.append("")
    
    # 绿箭
    if arrow_list:
        lines.append("━" * 40)
        lines.append(f"🟢 {arrow_name} · 量化彩票")
        lines.append("━" * 40)
        lines.append(f"{'排名':<4} {'标的':<6} {'概率':<7} {'信号':<8} {'现价':<10} {'日涨跌':<8}")
        lines.append("─" * 55)
        
        for i, s in enumerate(arrow_list[:CFG["scoring"]["arrow"]["top_n"]], 1):
            light, signal = traffic_light(s["prob"], "arrow")
            lines.append(
                f" {light} {i:<2} {s['ticker']:<6} {s['prob']*100:.1f}%  "
                f"{signal:<8} ${s.get('price', 0):<9} "
                f"{format_pct(s.get('daily_return')):<8}"
            )
        lines.append("")
    
    # 调仓建议
    actions = generate_actions(shield_list, arrow_list)
    if actions:
        lines.append("━" * 40)
        lines.append("💡 调仓建议")
        lines.append("━" * 40)
        
        buy = [a for a in actions if a["action"] in ("BUY", "ADD")]
        sell = [a for a in actions if a["action"] in ("SELL", "REDUCE")]
        hold = [a for a in actions if a["action"] == "HOLD"]
        
        if buy:
            lines.append("")
            lines.append("🟢 买入/加仓")
            for a in buy:
                icon = "🟢🟢" if a["action"] == "ADD" else "🟢"
                lines.append(f"   {icon} {a['ticker']} — {a['reason']}")
        
        if sell:
            lines.append("")
            lines.append("🔴 卖出/减仓")
            for a in sell:
                lines.append(f"   🔴 {a['ticker']} — {a['reason']}")
        
        if hold:
            lines.append("")
            lines.append("🟡 持有观望")
            for a in hold:
                lines.append(f"   🟡 {a['ticker']} — {a['reason']}")
        lines.append("")
    
    # 风险提示
    risks = generate_risk_warnings(shield_list, arrow_list)
    if risks:
        lines.append("━" * 40)
        lines.append("⚠️ 风险提示")
        lines.append("━" * 40)
        for r in risks:
            lines.append(f"  ❗ {r}")
        lines.append("")
    
    lines.append(f"{'━'*40}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="评分JSON文件")
    args = parser.parse_args()
    
    if args.data:
        with open(args.data) as f:
            scores = json.load(f)
        print(generate_report(scores))
    else:
        print("用法: python3 recommendation_template.py --data scores.json")
