#!/usr/bin/env python3
"""
🦅 Falcon Smart Alert Checker — 动态告警推送（去重v2）
====================================================
规则：
- 无持仓 → 只推L3（止损级），L1/L2静默
- 有持仓 → L2/L3推，L1静默（除非盘中首次触发）
- 同ticker+type 4小时内不重复推
- 盘后/盘前只推L3

用法 (Hermes cron no_agent=True):
  stdout有内容 → 推Telegram
  stdout为空 → 静默（零token）
"""

import json, os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Paths ──
PROJECT_ROOT = Path.home() / ".hermes" / "openclaw-archive"
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
PENDING_ALERTS = DATA_DIR / "alerts" / "pending.json"
STATE_FILE = DATA_DIR / "alerts" / "checker_state.json"
POSITIONS_FILE = DATA_DIR / "trades" / "positions.json"

# ── Timezone ──
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
except Exception:
    try:
        import pytz
        ET = pytz.timezone("US/Eastern")
    except Exception:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tzdata", "-q"])
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("US/Eastern")

# ── Market hours (ET) ──
PREMARKET_START = 7
MARKET_OPEN = 9.5
MARKET_CLOSE = 16
POSTMARKET_END = 20

# ── 去重窗口：同ticker+type 4小时内不重复推 ──
DEDUP_WINDOW_SEC = 14400  # 4 hours

def get_session(now_et: datetime) -> str:
    hour = now_et.hour + now_et.minute / 60
    if hour < PREMARKET_START:
        return "closed"
    elif hour < MARKET_OPEN:
        return "premarket"
    elif hour < MARKET_CLOSE:
        return "market"
    elif hour < POSTMARKET_END:
        return "postmarket"
    else:
        return "closed"


def get_portfolio_tickers() -> set:
    """读取当前持仓ticker"""
    if not POSITIONS_FILE.exists():
        return set()
    try:
        with open(POSITIONS_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {p.get("ticker", "") for p in data if p.get("ticker")}
        elif isinstance(data, dict):
            return {k for k in data.keys()}
    except Exception:
        pass
    return set()


def classify_alert(alert: dict) -> str:
    atype = alert.get("type", "")
    severity = alert.get("severity", "info")
    msg = alert.get("message", "")
    data = alert.get("data", {})
    if atype == "stop_loss" or severity == "critical" or "止损" in msg:
        return "L3"
    pnl = data.get("pnl")
    if pnl is not None and pnl <= -0.10:
        return "L2"
    change = data.get("change")
    if change is not None and abs(change) >= 0.05:
        return "L2"
    if "L2" in msg or "减仓" in msg:
        return "L2"
    return "L1"


def alert_fingerprint(alert: dict) -> str:
    atype = alert.get("type", "unknown")
    ticker = alert.get("ticker", "?")
    return f"{atype}:{ticker}"


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"sent_alerts": {}, "boost_until": 0, "last_push_ts": 0}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    now_et = datetime.now(ET)
    session = get_session(now_et)
    now_ts = time.time()

    # ── 市场关闭 → 静默 ──
    if session == "closed":
        return

    # ── 读取持仓 ──
    portfolio = get_portfolio_tickers()
    has_positions = len(portfolio) > 0

    state = load_state()

    # ── 读取告警 ──
    if not PENDING_ALERTS.exists():
        return
    try:
        with open(PENDING_ALERTS) as f:
            alerts = json.load(f)
    except Exception:
        return

    if not alerts or not isinstance(alerts, list):
        return

    # ── 核心去重 ──
    sent = state.get("sent_alerts", {})
    sent = {k: v for k, v in sent.items() if now_ts - v < DEDUP_WINDOW_SEC * 2}

    new_alerts = []
    for a in alerts:
        fp = alert_fingerprint(a)
        last_sent = sent.get(fp, 0)
        if now_ts - last_sent >= DEDUP_WINDOW_SEC:
            new_alerts.append(a)

    # 清空pending
    try:
        with open(PENDING_ALERTS, "w") as f:
            json.dump([], f)
    except Exception:
        pass

    if not new_alerts:
        state["sent_alerts"] = sent
        save_state(state)
        return

    # ── 分级 ──
    classified = []
    for a in new_alerts:
        lv = classify_alert(a)
        ticker = a.get("ticker", "?")
        is_holding = ticker in portfolio
        classified.append((a, lv, is_holding))

    # ── 过滤规则 ──
    # 无持仓：只推L3
    # 有持仓：推L2+持仓的L1，非持仓的L1静默
    # 盘前/盘后：只推L3
    to_push = []
    to_mark_sent = []

    for a, lv, is_holding in classified:
        if session in ("premarket", "postmarket"):
            if lv == "L3":
                to_push.append((a, lv))
            else:
                to_mark_sent.append(a)
        elif not has_positions:
            # 无持仓：只推L3
            if lv == "L3":
                to_push.append((a, lv))
            else:
                to_mark_sent.append(a)
        else:
            # 有持仓：持仓股推L2+，非持仓只推L3
            if lv == "L3":
                to_push.append((a, lv))
            elif lv == "L2" and is_holding:
                to_push.append((a, lv))
            elif lv == "L2" and not is_holding:
                # 非持仓的L2降级为L1静默
                to_mark_sent.append(a)
            else:
                # L1静默
                to_mark_sent.append(a)

    # 标记静默的为已发送
    for a in to_mark_sent:
        sent[alert_fingerprint(a)] = now_ts

    if not to_push:
        state["sent_alerts"] = sent
        save_state(state)
        return

    # ── 格式化输出 ──
    lines = ["🦅 **Falcon 异动告警**"]
    lines.append(f"⏰ {now_et.strftime('%H:%M ET')} | {session}")
    if not has_positions:
        lines.append("📋 无持仓 | 仅推止损级告警")
    lines.append("")

    l3 = [(a, lv) for a, lv in to_push if lv == "L3"]
    l2 = [(a, lv) for a, lv in to_push if lv == "L2"]

    if l3:
        lines.append("🔴 **L3 警报**")
        for a, _ in l3:
            lines.append(f"  {a['message']}")
            analysis = a.get("analysis", {})
            if analysis:
                rec = analysis.get("recommendation", "")
                reasoning = analysis.get("model_reasoning", "")[:80]
                if rec:
                    rec_map = {"reduce": "⚠️减仓", "stop_loss": "🛑止损", "hold": "⏳持有"}
                    lines.append(f"  → {rec_map.get(rec, rec)}: {reasoning}")
        lines.append("")

    if l2:
        lines.append("🟠 **L2 关注**")
        for a, _ in l2:
            lines.append(f"  {a['message']}")
            analysis = a.get("analysis", {})
            if analysis:
                rec = analysis.get("recommendation", "")
                reasoning = analysis.get("model_reasoning", "")[:60]
                if rec:
                    rec_map = {"reduce": "⚠️减仓", "stop_loss": "🛑止损", "hold": "⏳持有"}
                    lines.append(f"  → {rec_map.get(rec, rec)}: {reasoning}")
        lines.append("")

    lines.append(f"共 {len(to_push)} 条新增")

    print("\n".join(lines))

    # 记录已发送
    for a, _ in to_push:
        sent[alert_fingerprint(a)] = now_ts
    state["sent_alerts"] = sent
    state["last_push_ts"] = now_ts
    max_lv = "L3" if l3 else ("L2" if l2 else "L1")
    state["last_level"] = max_lv
    save_state(state)


if __name__ == "__main__":
    main()
