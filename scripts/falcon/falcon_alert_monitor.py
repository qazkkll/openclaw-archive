#!/usr/bin/env python3
"""
🦅 Falcon Dynamic Alert Monitor — 智能告警推送
==============================================
替代固定 */5 cron 的动态告警系统。

核心逻辑：
  - 市场关闭 → 静默（不检查）
  - 盘前(ET 7-9:30) → 每20分钟检查（可能有gap/news）
  - 盘中正常 → 每10分钟检查
  - 有L1告警 → 每5分钟检查（持续30分钟）
  - 有L2告警 → 每2分钟检查（持续30分钟）
  - 有L3告警 → 每1分钟检查（持续60分钟）
  - 30分钟无任何变化 → 降级回正常频率

架构：
  Observer daemon (写 pending.json) → 本脚本 (读+推Telegram)
  不再经过 Hermes agent cron → 零token, 零延迟

用法：
  python3 falcon_alert_monitor.py              # 正常运行
  python3 falcon_alert_monitor.py --test       # 测试模式（单次检查）
  python3 falcon_alert_monitor.py --once       # 单次检查后退出
"""

import json, os, sys, time, signal as sig
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
PENDING_ALERTS = DATA_DIR / "alerts" / "pending.json"
STATE_FILE = DATA_DIR / "observer_state.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "falcon.yaml"
LOCK_FILE = DATA_DIR / "alerts" / "monitor.lock"

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
PREMARKET_START = 7      # 7:00 AM
MARKET_OPEN = 9.5        # 9:30 AM
MARKET_CLOSE = 16        # 4:00 PM
POSTMARKET_END = 20      # 8:00 PM

# ── Dynamic intervals (seconds) ──
INTERVALS = {
    "closed":      None,    # 不检查
    "premarket":   1200,    # 20分钟 (盘前，可能有gap/news)
    "market":      600,     # 10分钟 (盘中正常)
    "postmarket":  1200,    # 20分钟 (盘后，降频)
    "l1_active":   300,     # 5分钟  (有L1告警，加频30分钟)
    "l2_active":   120,     # 2分钟  (有L2告警，加频30分钟)
    "l3_active":   60,      # 1分钟  (有L3告警，加频60分钟)
}

# 加频持续时间 (秒)
BOOST_DURATION = {
    "L1": 1800,   # 30分钟
    "L2": 1800,   # 30分钟
    "L3": 3600,   # 60分钟
}

# ── Alert level thresholds (与 Observer 一致) ──
LEVEL_THRESHOLDS = {
    "L3_keywords": ["stop_loss", "止损触发"],
    "L2_keywords": ["L2", "亏损", "↓5", "↓6", "↓7", "↓8", "↓9", "↓10",
                    "减仓", "gap", "低开 5", "低开 6", "低开 7", "低开 8"],
}


def classify_alert(alert: dict) -> str:
    """从告警消息判断级别 (与 Observer classify_alert_level 保持一致)"""
    msg = alert.get("message", "")
    atype = alert.get("type", "")
    severity = alert.get("severity", "info")

    # L3: 止损
    if atype == "stop_loss" or severity == "critical":
        return "L3"
    if "止损" in msg:
        return "L3"

    # L2: 大幅波动/亏损
    if "L2" in msg or "减仓" in msg:
        return "L2"
    # 检查亏损幅度 (持仓亏≥10%)
    data = alert.get("data", {})
    pnl = data.get("pnl")
    if pnl is not None and pnl <= -0.10:
        return "L2"
    # 检查价格波动幅度 (≥5%)
    change = data.get("change")
    if change is not None and abs(change) >= 0.05:
        return "L2"

    return "L1"


def get_session(now_et: datetime) -> str:
    """判断当前市场阶段"""
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


def load_pending_alerts() -> List[dict]:
    """读取待推送告警"""
    if not PENDING_ALERTS.exists():
        return []
    try:
        with open(PENDING_ALERTS) as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


def clear_pending_alerts():
    """清空已处理的告警"""
    try:
        PENDING_ALERTS.parent.mkdir(parents=True, exist_ok=True)
        with open(PENDING_ALERTS, "w") as f:
            json.dump([], f)
    except Exception:
        pass


def format_alert_message(alerts: List[dict]) -> str:
    """格式化告警为Telegram消息"""
    lines = ["🦅 **Falcon 异动告警**", "━━━━━━━━━━━━━━━━━━"]

    # 按级别分组
    l3 = [a for a in alerts if classify_alert(a) == "L3"]
    l2 = [a for a in alerts if classify_alert(a) == "L2"]
    l1 = [a for a in alerts if classify_alert(a) == "L1"]

    if l3:
        lines.append("")
        lines.append("🔴 **L3 警报** (需立即处理)")
        for a in l3:
            ts = a.get("timestamp", "")[:16]
            lines.append(f"  [{ts}] {a['message']}")
            # 附加分析推荐
            analysis = a.get("analysis", {})
            if analysis:
                rec = analysis.get("recommendation", "")
                reasoning = analysis.get("model_reasoning", "")[:80]
                if rec:
                    rec_map = {"reduce": "⚠️减仓", "stop_loss": "🛑止损",
                               "expire": "⏰到期", "hold": "⏳持有"}
                    lines.append(f"    → {rec_map.get(rec, rec)}: {reasoning}")

    if l2:
        lines.append("")
        lines.append("🟠 **L2 关注**")
        for a in l2:
            ts = a.get("timestamp", "")[:16]
            lines.append(f"  [{ts}] {a['message']}")
            analysis = a.get("analysis", {})
            if analysis:
                rec = analysis.get("recommendation", "")
                reasoning = analysis.get("model_reasoning", "")[:60]
                if rec:
                    rec_map = {"reduce": "⚠️减仓", "stop_loss": "🛑止损",
                               "expire": "⏰到期", "hold": "⏳持有"}
                    lines.append(f"    → {rec_map.get(rec, rec)}: {reasoning}")

    if l1:
        lines.append("")
        lines.append("🟡 **L1 观察**")
        for a in l1:
            ts = a.get("timestamp", "")[:16]
            lines.append(f"  [{ts}] {a['message']}")

    lines.append("")
    lines.append(f"共 {len(alerts)} 条 | {datetime.now(ET).strftime('%H:%M ET')}")
    return "\n".join(lines)


def send_telegram(text: str, token: str, chat_id: str) -> bool:
    """直接发送Telegram消息"""
    import urllib.request
    import urllib.parse

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode()

    try:
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as e:
        print(f"  ❌ Telegram send failed: {e}")
        return False


def load_telegram_config() -> Optional[tuple]:
    """从 .env 或环境变量加载 Telegram 配置"""
    env_path = PROJECT_ROOT / ".env"
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id:
        # 从 .env 读取
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key == "TELEGRAM_BOT_TOKEN":
                        token = val
                    elif key == "TELEGRAM_CHAT_ID":
                        chat_id = val

    if token and chat_id:
        return token, chat_id
    return None


def acquire_lock() -> bool:
    """简单文件锁，防止多实例运行"""
    if LOCK_FILE.exists():
        try:
            with open(LOCK_FILE) as f:
                lock_data = json.load(f)
            pid = lock_data.get("pid", 0)
            # 检查进程是否还活着
            try:
                os.kill(pid, 0)
                # 进程存在，检查启动时间
                started = lock_data.get("started", "")
                if started:
                    from datetime import datetime as dt
                    start_time = dt.fromisoformat(started)
                    if (datetime.now(timezone.utc) - start_time).total_seconds() > 86400:
                        # 锁超过24小时，可能stale
                        pass
                    else:
                        return False
            except ProcessLookupError:
                pass  # 进程已死，可以接管
        except Exception:
            pass

    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_FILE, "w") as f:
        json.dump({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}, f)
    return True


def release_lock():
    """释放文件锁"""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def run_monitor(test_mode=False, once=False):
    """主监控循环"""

    # 加载Telegram配置
    tg_config = load_telegram_config()
    if not tg_config:
        print("❌ Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")
        print("  Falling back to stdout mode (for cron delivery)")
    tg_token, tg_chat_id = tg_config if tg_config else (None, None)

    # 获取锁
    if not once and not test_mode:
        if not acquire_lock():
            print("❌ Another monitor instance is running. Exiting.")
            return

    print(f"🦅 Falcon Dynamic Alert Monitor")
    print(f"{'=' * 50}")
    if tg_config:
        print(f"  📱 Telegram: connected")
    else:
        print(f"  📱 Telegram: stdout mode")

    # Graceful shutdown
    running = [True]
    def handle_signal(signum, frame):
        print(f"\n🛑 Signal {signum}, shutting down...")
        running[0] = False
    sig.signal(sig.SIGINT, handle_signal)
    sig.signal(sig.SIGTERM, handle_signal)

    # 状态追踪
    current_interval = INTERVALS["market"]
    boost_until = {}  # {level: expire_timestamp}
    last_alert_time = 0
    consecutive_empty = 0  # 连续无告警次数

    try:
        while running[0]:
            now_et = datetime.now(ET)
            session = get_session(now_et)

            # ── Step 1: 确定当前检查间隔 ──
            if session == "closed":
                if once or test_mode:
                    print(f"  😴 Market closed. Would sleep until pre-market.")
                    if once:
                        break
                else:
                    # 计算到下次开盘的睡眠时间
                    next_start = now_et.replace(hour=PREMARKET_START, minute=0, second=0, microsecond=0)
                    if now_et.hour >= POSTMARKET_END:
                        next_start += timedelta(days=1)
                    sleep_sec = max(60, (next_start - now_et).total_seconds())
                    print(f"  😴 Market closed. Sleeping {sleep_sec/60:.0f} min until pre-market...")
                    # 分段sleep以便响应signal
                    for _ in range(int(sleep_sec / 60)):
                        if not running[0]:
                            break
                        time.sleep(60)
                    continue

            # 检查是否有加频到期
            now_ts = time.time()
            active_boosts = {lv: exp for lv, exp in boost_until.items() if exp > now_ts}

            if "L3" in active_boosts:
                current_interval = INTERVALS["l3_active"]
                mode = "🔴L3"
            elif "L2" in active_boosts:
                current_interval = INTERVALS["l2_active"]
                mode = "🟠L2"
            elif "L1" in active_boosts:
                current_interval = INTERVALS["l1_active"]
                mode = "🟡L1"
            else:
                current_interval = INTERVALS.get(session, INTERVALS["market"])
                mode = session

            # ── Step 2: 读取告警 ──
            alerts = load_pending_alerts()

            if test_mode:
                print(f"  [{now_et.strftime('%H:%M:%S')}] {mode} | interval={current_interval}s | alerts={len(alerts)}")
                if alerts:
                    for a in alerts:
                        level = classify_alert(a)
                        print(f"    {level}: {a.get('message', '?')[:60]}")
                if once:
                    break
                time.sleep(current_interval)
                continue

            if alerts:
                # ── Step 3: 有告警 → 推送 ──
                msg = format_alert_message(alerts)

                # 判断最高告警级别
                max_level = "L1"
                for a in alerts:
                    lv = classify_alert(a)
                    if lv == "L3":
                        max_level = "L3"
                        break
                    elif lv == "L2":
                        max_level = "L2"

                # 推送
                if tg_config and tg_token and tg_chat_id:
                    ok = send_telegram(msg, tg_token, tg_chat_id)
                    if ok:
                        print(f"  ✅ Pushed {len(alerts)} alerts ({max_level}) to Telegram")
                    else:
                        print(f"  ❌ Push failed, alerts preserved in pending.json")
                        continue  # 不清空，下次重试
                else:
                    # stdout模式（给cron用）
                    print(msg)

                # 更新加频状态
                boost_until[max_level] = now_ts + BOOST_DURATION.get(max_level, 1800)

                # 清空已处理
                clear_pending_alerts()
                last_alert_time = now_ts
                consecutive_empty = 0

                print(f"  ⚡ Boosted to {max_level} for {BOOST_DURATION.get(max_level, 1800)/60:.0f} min")

            else:
                # ── 无告警 ──
                consecutive_empty += 1

                # 如果加频已过期，自动降级
                expired = [lv for lv, exp in boost_until.items() if exp <= now_ts]
                for lv in expired:
                    del boost_until[lv]
                    print(f"  📉 {lv} boost expired, reverting to normal")

                # 如果连续30分钟无告警 + 无加频 → 恢复正常频率
                if consecutive_empty >= 3 and not active_boosts:
                    consecutive_empty = 0  # 避免重复log

                now_str = now_et.strftime("%H:%M:%S")
                if consecutive_empty <= 1 or consecutive_empty % 6 == 0:
                    # 只在前几次和每隔1小时打印一次状态
                    print(f"  [{now_str}] {mode} ✅ | next in {current_interval}s")

            # ── Step 4: 等待 ──
            if once:
                break

            # 分段sleep，每30秒检查一次running状态
            for _ in range(int(current_interval / 30) + 1):
                if not running[0]:
                    break
                time.sleep(min(30, current_interval))

    finally:
        release_lock()
        print(f"\n🦅 Monitor stopped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Falcon Dynamic Alert Monitor")
    parser.add_argument("--test", action="store_true", help="Test mode (verbose)")
    parser.add_argument("--once", action="store_true", help="Single check then exit")
    args = parser.parse_args()

    run_monitor(test_mode=args.test, once=args.once)
