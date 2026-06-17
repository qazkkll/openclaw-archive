#!/usr/bin/env python3
"""
OpenClaw 看门狗 — 从config.json读取配置

用法：
    python3 watchdog.py              # 单次检查
    python3 watchdog.py --init       # 初始化状态（不触发）
    python3 watchdog.py --status     # 显示当前状态
"""

import json, os, sys, time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
STATE_FILE = os.path.join(OUTPUT_DIR, "watchdog_state.json")
ALERT_FILE = os.path.join(OUTPUT_DIR, "watchdog_alert.json")
CONFIG_PATH = os.path.join(ROOT, "config.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 加载配置 ──
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

TRIGGERS = CFG["watchdog"]["triggers"]
COOLDOWN_SEC = CFG["watchdog"]["cooldown_minutes"] * 60


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def save_alert(alerts):
    with open(ALERT_FILE, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "alerts": alerts,
            "count": len(alerts),
        }, f, indent=2, ensure_ascii=False)


def run_scoring():
    """运行评分"""
    sys.path.insert(0, SCRIPTS_DIR)
    from live_monitor import (
        load_v9_data, load_v9_model, score_v9_lottery,
        score_shield_v3, get_market_context, US_UNIVERSE
    )
    
    v9_data = load_v9_data()
    if v9_data is None:
        return None
    
    v9_model = load_v9_model()
    if v9_model is None:
        return None
    
    arrow_top10, _ = score_v9_lottery(v9_data, v9_model)
    
    shield_tickers = list(set(
        [a["ticker"] for a in arrow_top10] + US_UNIVERSE[:30]
    ))
    shield_top10 = score_shield_v3(shield_tickers)[:10]
    
    context = get_market_context()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "shield_top5": [s["ticker"] for s in shield_top10[:5]],
        "shield_scores": {s["ticker"]: s["score"] for s in shield_top10[:5]},
        "arrow_top5": [a["ticker"] for a in arrow_top10[:5]],
        "arrow_probs": {a["ticker"]: a["prob"] for a in arrow_top10[:5]},
        "context": context,
    }


def detect_changes(current, previous):
    """检测重大变化 — 从config读取阈值"""
    alerts = []
    if previous is None:
        return alerts
    
    cooldowns = previous.get("cooldowns", {})
    now = time.time()
    
    # Top5变动
    if TRIGGERS["top5_change"]:
        prev_shield = set(previous.get("shield_top5", []))
        curr_shield = set(current.get("shield_top5", []))
        
        for t in curr_shield - prev_shield:
            if now - cooldowns.get(f"shield_{t}", 0) > COOLDOWN_SEC:
                score = current["shield_scores"].get(t, 0)
                alerts.append({
                    "type": "shield_new_entry",
                    "ticker": t,
                    "message": f"🔵 {t} 新进Top5（{score}分）",
                    "priority": "high" if score >= CFG["scoring"]["shield"]["thresholds"]["strong_buy"] else "medium",
                })
                cooldowns[f"shield_{t}"] = now
        
        for t in prev_shield - curr_shield:
            alerts.append({
                "type": "shield_exit",
                "ticker": t,
                "message": f"🔵 {t} 跌出Top5",
                "priority": "medium",
            })
    
    # 蓝盾分数剧变
    for ticker in set(current.get("shield_top5", [])) & set(previous.get("shield_top5", [])):
        prev_score = previous.get("shield_scores", {}).get(ticker, 0)
        curr_score = current.get("shield_scores", {}).get(ticker, 0)
        delta = abs(curr_score - prev_score)
        
        if delta >= TRIGGERS["shield_score_delta"]:
            direction = "↑" if curr_score > prev_score else "↓"
            alerts.append({
                "type": "shield_score_change",
                "ticker": ticker,
                "message": f"🔵 {ticker} {direction}{delta}分（{prev_score}→{curr_score}）",
                "priority": "high",
            })
    
    # 绿箭高确定性
    for ticker, prob in current.get("arrow_probs", {}).items():
        if prob >= TRIGGERS["arrow_prob_threshold"]:
            prev_prob = previous.get("arrow_probs", {}).get(ticker, 0)
            if prev_prob < TRIGGERS["arrow_prob_threshold"]:
                if now - cooldowns.get(f"arrow_{ticker}", 0) > COOLDOWN_SEC:
                    alerts.append({
                        "type": "arrow_breakout",
                        "ticker": ticker,
                        "message": f"🟢 {ticker} 概率突破{prob*100:.0f}%",
                        "priority": "high",
                    })
                    cooldowns[f"arrow_{ticker}"] = now
    
    # VIX异动
    prev_vix = float(previous.get("context", {}).get("VIX", 0))
    curr_vix = float(current.get("context", {}).get("VIX", 0))
    if prev_vix > 0 and curr_vix > 0:
        vix_delta = abs(curr_vix - prev_vix)
        if vix_delta >= TRIGGERS["vix_delta"]:
            direction = "↑" if curr_vix > prev_vix else "↓"
            alerts.append({
                "type": "vix_move",
                "message": f"📊 VIX {direction}{vix_delta:.1f}（{prev_vix:.1f}→{curr_vix:.1f}）",
                "priority": "high",
            })
    
    # S&P异动
    prev_sp = previous.get("context", {}).get("S&P 500", "")
    curr_sp = current.get("context", {}).get("S&P 500", "")
    if prev_sp and curr_sp:
        try:
            prev_val = float(prev_sp.replace(",", ""))
            curr_val = float(curr_sp.replace(",", ""))
            if prev_val > 0:
                pct = abs(curr_val / prev_val - 1) * 100
                if pct >= TRIGGERS["sp500_pct_delta"]:
                    direction = "↑" if curr_val > prev_val else "↓"
                    alerts.append({
                        "type": "sp500_move",
                        "message": f"📈 S&P 500 {direction}{pct:.1f}%",
                        "priority": "high",
                    })
        except ValueError:
            pass
    
    current["cooldowns"] = cooldowns
    return alerts


def format_alert(alerts, current):
    if not alerts:
        return None
    
    now = datetime.now().strftime("%H:%M")
    lines = [f"🚨 OpenClaw警报 {now}", "━" * 30]
    
    high = [a for a in alerts if a.get("priority") == "high"]
    medium = [a for a in alerts if a.get("priority") != "high"]
    
    if high:
        lines.append("")
        lines.append("🔴 重大变化")
        for a in high:
            lines.append(f"  {a['message']}")
    
    if medium:
        lines.append("")
        lines.append("🟡 一般变化")
        for a in medium:
            lines.append(f"  {a['message']}")
    
    lines.append("")
    lines.append("📊 快照")
    shield = current.get("shield_top5", [])
    arrow = current.get("arrow_top5", [])
    if shield:
        lines.append(f"  蓝盾: {' / '.join(shield[:3])}")
    if arrow:
        lines.append(f"  绿箭: {' / '.join(arrow[:3])}")
    
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    
    if args.status:
        state = load_state()
        if state:
            print(f"上次: {state.get('timestamp', 'N/A')}")
            print(f"蓝盾: {state.get('shield_top5', [])}")
            print(f"绿箭: {state.get('arrow_top5', [])}")
        else:
            print("无状态")
        return
    
    current = run_scoring()
    if current is None:
        print("❌ 评分失败")
        return
    
    previous = load_state()
    alerts = detect_changes(current, previous)
    
    if alerts:
        msg = format_alert(alerts, current)
        save_alert(alerts)
        print(msg)
    elif not args.init:
        print("✅ 无重大变化")
    
    save_state(current)


if __name__ == "__main__":
    main()
