#!/usr/bin/env python3
"""
🦅 Falcon Smart Alert Checker — 动态告警推送（去重版）
====================================================
核心规则：同一个ticker+类型，30分钟内只推一次。
Observer每5分钟写pending.json，本脚本负责去重。

用法 (Hermes cron no_agent=True):
  stdout有内容 → 推Telegram
  stdout为空 → 静默（零token）
"""

import json, os, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
PENDING_ALERTS = DATA_DIR / "alerts" / "pending.json"
STATE_FILE = DATA_DIR / "alerts" / "checker_state.json"

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

# ── 去重窗口：同ticker+type 30分钟内不重复推 ──
DEDUP_WINDOW_SEC = 1800

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
    """生成告警指纹：type+ticker（价格变化>1%视为新告警）"""
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

    # ── 核心去重：过滤掉30分钟内已推过的 ──
    sent = state.get("sent_alerts", {})
    # 清理过期的sent记录
    sent = {k: v for k, v in sent.items() if now_ts - v < DEDUP_WINDOW_SEC * 2}

    new_alerts = []
    for a in alerts:
        fp = alert_fingerprint(a)
        last_sent = sent.get(fp, 0)
        if now_ts - last_sent >= DEDUP_WINDOW_SEC:
            new_alerts.append(a)

    # 清空pending（不管是否new都清，避免Observer重复写）
    try:
        with open(PENDING_ALERTS, "w") as f:
            json.dump([], f)
    except Exception:
        pass

    if not new_alerts:
        # 全部是重复的，不推
        state["sent_alerts"] = sent
        save_state(state)
        return

    # ── 分级 ──
    max_level = "L1"
    for a in new_alerts:
        lv = classify_alert(a)
        if lv == "L3":
            max_level = "L3"
            break
        elif lv == "L2":
            max_level = "L2"

    # ── 盘前/盘后: 只推L2+ ──
    if session in ("premarket", "postmarket") and max_level == "L1":
        # 记录为已发送（避免下次重复），但不推
        for a in new_alerts:
            sent[alert_fingerprint(a)] = now_ts
        state["sent_alerts"] = sent
        save_state(state)
        return

    # ── 格式化输出 ──
    lines = ["🦅 **Falcon 异动告警**"]
    lines.append(f"⏰ {now_et.strftime('%H:%M ET')} | {session}")
    lines.append("")

    l3 = [a for a in new_alerts if classify_alert(a) == "L3"]
    l2 = [a for a in new_alerts if classify_alert(a) == "L2"]
    l1 = [a for a in new_alerts if classify_alert(a) == "L1"]

    if l3:
        lines.append("🔴 **L3 警报**")
        for a in l3:
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
        for a in l2:
            lines.append(f"  {a['message']}")
            analysis = a.get("analysis", {})
            if analysis:
                rec = analysis.get("recommendation", "")
                reasoning = analysis.get("model_reasoning", "")[:60]
                if rec:
                    rec_map = {"reduce": "⚠️减仓", "stop_loss": "🛑止损", "hold": "⏳持有"}
                    lines.append(f"  → {rec_map.get(rec, rec)}: {reasoning}")
        lines.append("")

    if l1:
        lines.append("🟡 **L1 观察**")
        for a in l1:
            lines.append(f"  {a['message']}")
        lines.append("")

    lines.append(f"共 {len(new_alerts)} 条新增")

    # 输出
    print("\n".join(lines))

    # 记录已发送
    for a in new_alerts:
        sent[alert_fingerprint(a)] = now_ts
    state["sent_alerts"] = sent
    state["last_push_ts"] = now_ts
    state["last_level"] = max_level
    save_state(state)


if __name__ == "__main__":
    main()
