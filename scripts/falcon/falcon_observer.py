#!/usr/bin/env python3
"""
🦅 Falcon Observer — 盘前/盘中/盘后全天候监控
================================================
零token daemon: 纯脚本监控，只在异动时写alert文件。
Dashboard读 state.json，cron读 alerts → 触发agent推送。

架构:
  observer (daemon) → state.json (Dashboard)
                    → alerts/pending.json (cron → agent → Telegram)

用法:
  python3 falcon_observer.py                # 正常运行
  python3 falcon_observer.py --test         # 连通测试
  python3 falcon_observer.py --once         # 单次执行(调试)
"""

import json, os, sys, time, signal as sig, glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any
from trading_calendar import is_trading_day, next_trading_day

# ── Broker Adapter (统一持仓接口) ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from broker_adapter import get_broker, Position

# ── Paths ──
OBSERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = OBSERVER_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
STATE_FILE = DATA_DIR / "observer_state.json"
ALERTS_DIR = DATA_DIR / "alerts"
PENDING_ALERTS = ALERTS_DIR / "pending.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "falcon.yaml"
LOG_DIR = DATA_DIR / "logs"
REF_CACHE = DATA_DIR / "ref_cache.json"  # cached reference data (prev_close, avg_vol)

# ── Timezone ──
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
except Exception:
    try:
        import pytz
        ET = pytz.timezone("US/Eastern")
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tzdata", "-q"])
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("US/Eastern")

# ── Constants ──
POLL_INTERVAL = 300  # 5 minutes
PREMARKET_START = 7   # 7:00 AM ET
MARKET_OPEN = 9.5     # 9:30 AM ET
MARKET_CLOSE = 16     # 4:00 PM ET
POSTMARKET_END = 20   # 8:00 PM ET

# ── Alert thresholds ──
THRESHOLDS = {
    "stop_loss": -0.15,          # -15%: 止损
    "price_move_pct": 3.0,       # ±3%: 大幅波动
    "volume_spike_ratio": 3.0,   # 3x均量: 放量
    "gap_pct": 5.0,              # ±5%: 盘前大gap
    "entry_vwap_score": 60,      # 入场评分≥60 + price≤VWAP
    # L2/L3 增强阈值
    "price_move_l2": 5.0,        # ±5%: L2 关注级
    "pnl_warn": -0.10,           # -10%: 持仓预警线
}

# ── 异动分级器 + 深度分析 ──
# L1: 推消息，继续正常监控
# L2: 推消息 + 拉新闻 + FinBERT + 分析建议 + 加频1分钟
# L3: 推消息 + 新闻 + 分析 + 建议立即止损/止盈
_enhanced_tickers = {}  # {ticker: expire_ts} — 加频监控到期时间
_analyzer_loaded = False

def classify_alert_level(alert: dict, cfg: dict) -> str:
    """将原始alert分级为L1/L2/L3。"""
    atype = alert.get("type", "")
    data = alert.get("data", {})
    severity = alert.get("severity", "info")

    # L3: 止损触发
    if atype == "stop_loss" or severity == "critical":
        return "L3"

    # L2: 持仓亏损≥10% / 价格波动≥5% / 盘前gap≥7% / 信号退化
    if atype == "signal_degradation":
        return "L2"
    
    pnl = data.get("pnl")
    if pnl is not None and pnl <= cfg.get("pnl_warn", -0.10):
        return "L2"

    change = abs(data.get("change", 0))
    if atype == "price_move" and change >= THRESHOLDS["price_move_l2"] / 100:
        return "L2"

    if atype == "gap" and abs(data.get("gap", 0)) >= 0.07:
        return "L2"

    return "L1"

def run_deep_analysis(alert: dict, cfg: dict) -> Optional[dict]:
    """对L2/L3告警执行深度分析(模型规则+新闻上下文)。

    设计原则: 决策归模型，新闻只是上下文。
    """
    global _analyzer_loaded
    try:
        if not _analyzer_loaded:
            sys.path.insert(0, str(OBSERVER_DIR))
            _analyzer_loaded = True

        from falcon_alert_analyzer import analyze_ticker, format_analysis_telegram

        ticker = alert.get("ticker", "")
        data = alert.get("data", {})
        price = data.get("price", 0)
        entry = data.get("entry")
        pnl = data.get("pnl")
        level = alert.get("alert_level", "L2")

        # 计算持有天数(如果持仓信息里有)
        hold_days = None
        entry_date = data.get("entry_date")
        if entry_date:
            try:
                from datetime import datetime as dt
                ed = dt.fromisoformat(str(entry_date).replace("Z", "+00:00"))
                hold_days = (dt.now(ed.tzinfo) - ed).days
            except Exception:
                pass

        result = analyze_ticker(
            ticker=ticker,
            alert_type=alert.get("type", ""),
            current_price=price,
            entry_price=entry,
            pnl_pct=pnl,
            hold_days=hold_days,
            alert_level=level,
        )

        # 加频: L2→1分钟持续30分钟, L3→1分钟持续60分钟
        boost_min = 60 if level == "L3" else 30
        _enhanced_tickers[ticker] = time.time() + boost_min * 60

        return result
    except Exception as e:
        print(f"  ⚠️ Deep analysis failed for {alert.get('ticker', '?')}: {e}")
        return None


# ═══════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════
def load_config() -> dict:
    """Load falcon.yaml."""
    cfg = {
        "top_n": 5,
        "hold_days": 30,
        "stop_loss": -0.15,
        "vwap_trigger": 0.0,
        "fallback_minutes": 60,
        "min_entry_score": 60,
    }
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        t = raw.get("trading", {})
        cfg["hold_days"] = t.get("hold_days", 30)
        cfg["stop_loss"] = t.get("stop_loss", -0.15)
        e = t.get("entry", {})
        cfg["vwap_trigger"] = e.get("vwap_trigger", 0.0)
        cfg["fallback_minutes"] = e.get("fallback_minutes", 60)
        cfg["min_entry_score"] = e.get("min_entry_score", 60)
        m = raw.get("model", {})
        cfg["top_n"] = m.get("top_n", 5)
    except Exception:
        pass
    return cfg


# ═══════════════════════════════════════════════
# Alpaca
# ═══════════════════════════════════════════════
def get_alpaca_client():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.historical import StockHistoricalDataClient
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials in .env")
    return StockHistoricalDataClient(key, secret)


def get_snapshots(client, tickers: List[str]) -> Dict[str, Any]:
    """Get real-time snapshots for tickers."""
    from alpaca.data.requests import StockSnapshotRequest
    try:
        from alpaca.data.enums import DataFeed
        req = StockSnapshotRequest(symbol_or_symbols=tickers, feed=DataFeed.IEX)
        snaps = client.get_stock_snapshot(req)
        result = {}
        for sym, snap in snaps.items():
            trade = snap.latest_trade
            quote = snap.latest_quote
            result[sym] = {
                "price": float(trade.price) if trade else None,
                "trade_time": trade.timestamp.isoformat() if trade else None,
                "bid": float(quote.bid_price) if quote else None,
                "ask": float(quote.ask_price) if quote else None,
                "bid_size": int(quote.bid_size) if quote else None,
                "ask_size": int(quote.ask_size) if quote else None,
            }
        return result
    except Exception as e:
        return {"error": str(e)}


def get_recent_bars(client, tickers: List[str], days: int = 1) -> Dict[str, Any]:
    """Get recent daily bars for reference prices."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 15)  # wider window for weekends/holidays
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,  # free tier only allows IEX
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        result = {}
        for sym in tickers:
            if sym in df.index.get_level_values(0):
                sub = df.loc[sym]
                if len(sub) > 0:
                    last = sub.iloc[-1]
                    prev = sub.iloc[-2] if len(sub) > 1 else last
                    result[sym] = {
                        "prev_close": float(prev["close"]),
                        "last_close": float(last["close"]),
                        "last_volume": float(last["volume"]),
                        "avg_volume_20d": float(sub["volume"].tail(20).mean()) if len(sub) >= 5 else float(last["volume"]),
                        "last_high": float(last["high"]),
                        "last_low": float(last["low"]),
                    }
        return result
    except Exception as e:
        return {"error": str(e)}


def get_minute_bars_today(client, tickers: List[str]) -> Dict[str, Any]:
    """Get today's minute bars (for VWAP calculation)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    try:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=8, minute=0, second=0, microsecond=0)  # pre-market
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Minute,
            start=start,
            end=now,
            feed=DataFeed.IEX,  # free tier only allows IEX
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        result = {}
        for sym in tickers:
            if sym in df.index.get_level_values(0):
                sub = df.loc[sym]
                if len(sub) > 0:
                    # Calculate VWAP
                    cum_vol = sub["volume"].cumsum()
                    cum_pv = (sub["close"] * sub["volume"]).cumsum()
                    vwap = cum_pv / cum_vol.replace(0, 1)
                    result[sym] = {
                        "bars_count": len(sub),
                        "day_open": float(sub.iloc[0]["open"]),
                        "day_high": float(sub["high"].max()),
                        "day_low": float(sub["low"].min()),
                        "day_volume": float(sub["volume"].sum()),
                        "vwap": float(vwap.iloc[-1]) if len(vwap) > 0 else None,
                        "trade_count": int(sub["trade_count"].sum()) if "trade_count" in sub.columns else None,
                    }
        return result
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════
# Reference Data Cache (fixes null prev_close when market closed)
# ═══════════════════════════════════════════════
def save_ref_cache(data: dict):
    """Cache reference data so observer works when market is closed."""
    REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    cache = {"timestamp": datetime.now(timezone.utc).isoformat(), "data": data}
    with open(REF_CACHE, "w") as f:
        json.dump(cache, f, indent=2)

def load_ref_cache() -> dict:
    """Load cached reference data. Returns {} if stale (>24h)."""
    if not REF_CACHE.exists():
        return {}
    try:
        with open(REF_CACHE) as f:
            cache = json.load(f)
        from datetime import datetime as dt
        ts = dt.fromisoformat(cache["timestamp"].replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        if age_hours > 48:  # stale after 48h (covers weekends)
            return {}
        return cache.get("data", {})
    except Exception:
        return {}


# ═══════════════════════════════════════════════
# Portfolio State (from broker, live)
# ═══════════════════════════════════════════════
def load_portfolio() -> Dict[str, dict]:
    """Load current portfolio from Alpaca (活接口)。

    返回格式: {symbol: {entry_price, entry_date, shares, unrealized_plpc}}
    后续切换Futu OpenD只需改broker_adapter.py, 这里不用动。
    """
    portfolio = {}
    try:
        broker = get_broker()
        positions = broker.get_positions()
        for pos in positions:
            portfolio[pos.symbol] = {
                "entry_price": pos.avg_entry_price,
                "entry_date": pos.entry_date or "",
                "shares": pos.qty,
                "unrealized_plpc": pos.unrealized_plpc,
                "current_price": pos.current_price,
            }
    except Exception as e:
        print(f"  ⚠️ broker持仓读取失败: {e}, 回退到本地缓存")
        # 回退: 从positions.json读(备份)
        pos_file = DATA_DIR / "trades" / "positions.json"
        if pos_file.exists():
            try:
                with open(pos_file) as f:
                    data = json.load(f)
                for sym, info in data.get("positions", {}).items():
                    portfolio[sym] = {
                        "entry_price": info.get("entry_price", 0),
                        "entry_date": info.get("entry_date", ""),
                        "shares": info.get("qty", 0),
                    }
            except Exception:
                pass
    return portfolio


def load_signal(top_n: int = 5) -> List[dict]:
    """Load latest Falcon signal. 每次调用都读最新文件。"""
    pattern = str(DATA_DIR / "falcon_v046_scored_*.json")
    if not glob.glob(pattern):
        pattern = str(DATA_DIR / "falcon_v046_scored_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    latest_file = files[-1]

    # I3: 信号文件新鲜度检查 — ✅ 验证通过: 30天阈值
    # 验证结果: IC>0.05持续到30天, IC半衰期3天但21天回升到0.206
    I3_STALE_DAYS = 30  # 数据验证: 20个评分日, Top5信号IC在30天内保持>0.05
    try:
        fname = Path(latest_file).stem
        date_str = fname.split("_")[-1]
        file_date = datetime.strptime(date_str, "%Y%m%d").date()
        age_days = (datetime.now().date() - file_date).days
        if age_days > I3_STALE_DAYS:
            print(f"  ⚠️ 信号文件过期{age_days}天(阈值{I3_STALE_DAYS}天): {Path(latest_file).name}")
    except Exception:
        pass

    with open(latest_file) as f:
        data = json.load(f)
    picks = data.get("top_n", data.get("picks", []))
    targets = []
    for p in picks[:top_n]:
        ticker = p.get("sym") or p.get("ticker", "")
        if ticker:
            targets.append({
                "ticker": ticker,
                "score": float(p.get("score", 0)),
                "close": float(p.get("close", 0)),
                "signal": p.get("signal", ""),
            })
    return targets


# ═══════════════════════════════════════════════
# Anomaly Detection
# ═══════════════════════════════════════════════
def detect_anomalies(
    ticker: str,
    snapshot: dict,
    daily_ref: dict,
    minute_data: dict,
    portfolio: dict,
    cfg: dict,
    session: str,
) -> List[dict]:
    """Check all anomaly conditions for a ticker. Returns list of alerts."""
    alerts = []
    price = snapshot.get("price")
    if not price:
        return alerts

    prev_close = daily_ref.get("prev_close") or daily_ref.get("last_close")
    if not prev_close or prev_close <= 0:
        return alerts

    # ── Daily change ──
    daily_change = (price - prev_close) / prev_close

    # ── Portfolio P&L ──
    pos = portfolio.get(ticker)
    unrealized_pnl = None
    if pos and pos.get("entry_price", 0) > 0:
        unrealized_pnl = (price - pos["entry_price"]) / pos["entry_price"]

    # ── VWAP ──
    vwap = minute_data.get("vwap")
    vwap_dev = None
    if vwap and vwap > 0:
        vwap_dev = (price - vwap) / vwap

    # ── Volume ──
    day_volume = minute_data.get("day_volume", 0)
    avg_volume = daily_ref.get("avg_volume_20d", 0)
    vol_ratio = day_volume / avg_volume if avg_volume > 0 else 0

    # ── Gap (pre-market) ──
    gap = None
    if session == "premarket":
        gap = daily_change  # price vs prev_close IS the gap

    # ── Entry score (simplified) ──
    entry_score = None
    if vwap_dev is not None:
        day_high = minute_data.get("day_high", price)
        day_low = minute_data.get("day_low", price)
        price_pos = (price - day_low) / (day_high - day_low) if day_high > day_low else 0.5
        # Simplified entry score
        entry_score = (
            0.30 * max(0, min(100, (1 - price_pos) * 100)) +
            0.30 * max(0, min(100, (0.01 - vwap_dev) * 5000)) +
            0.20 * max(0, min(100, vol_ratio * 50)) +
            0.20 * 50  # momentum placeholder
        )

    # ═══ Alert conditions ═══

    # 1. Stop loss (highest priority)
    if unrealized_pnl is not None and unrealized_pnl <= cfg["stop_loss"]:
        alerts.append({
            "type": "stop_loss",
            "severity": "critical",
            "ticker": ticker,
            "message": f"🔴 止损触发: {ticker} 亏损 {unrealized_pnl*100:.1f}% (阈值 {cfg['stop_loss']*100:.0f}%)",
            "data": {"price": price, "entry": pos.get("entry_price"), "pnl": unrealized_pnl},
        })

    # 2. Large price move
    if abs(daily_change) >= THRESHOLDS["price_move_pct"] / 100:
        direction = "↑" if daily_change > 0 else "↓"
        alerts.append({
            "type": "price_move",
            "severity": "warning",
            "ticker": ticker,
            "message": f"🟡 {ticker} {direction}{abs(daily_change)*100:.1f}% (阈值±{THRESHOLDS['price_move_pct']}%)",
            "data": {"price": price, "change": daily_change},
        })

    # 3. Volume spike (only during market hours)
    if session == "market" and vol_ratio >= THRESHOLDS["volume_spike_ratio"]:
        alerts.append({
            "type": "volume_spike",
            "severity": "info",
            "ticker": ticker,
            "message": f"🟡 {ticker} 放量 {vol_ratio:.1f}x (阈值{THRESHOLDS['volume_spike_ratio']}x)",
            "data": {"volume": day_volume, "avg_volume": avg_volume, "ratio": vol_ratio},
        })

    # 4. Pre-market gap
    if gap is not None and abs(gap) >= THRESHOLDS["gap_pct"] / 100:
        direction = "高开" if gap > 0 else "低开"
        alerts.append({
            "type": "gap",
            "severity": "warning",
            "ticker": ticker,
            "message": f"🟡 {ticker} 盘前{direction} {abs(gap)*100:.1f}%",
            "data": {"price": price, "prev_close": prev_close, "gap": gap},
        })

    # 5. Entry signal (price ≤ VWAP + good score)
    if (session == "market" and vwap_dev is not None and
            vwap_dev <= cfg["vwap_trigger"] and
            entry_score is not None and entry_score >= cfg["min_entry_score"]):
        alerts.append({
            "type": "entry_signal",
            "severity": "info",
            "ticker": ticker,
            "message": f"🟢 {ticker} 入场信号: price≤VWAP, 评分{entry_score:.0f}",
            "data": {"price": price, "vwap": vwap, "vwap_dev": vwap_dev, "score": entry_score},
        })

    # W6: 信号退化检测 — 持仓股评分是否降级
    # 在detect_anomalies中无法直接访问最新评分(top_n),
    # 这个检查在run_observer的poll_once后由专门逻辑处理

    return alerts


# ═══════════════════════════════════════════════
# State Management
# ═══════════════════════════════════════════════
def write_state(state: dict):
    """Write current state to JSON for dashboard."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


def write_alerts(alerts: List[dict]):
    """Append alerts to pending file, with proper dedup and expiry."""
    ALERTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    if PENDING_ALERTS.exists():
        try:
            with open(PENDING_ALERTS) as f:
                existing = json.load(f)
        except Exception:
            pass

    now = datetime.now(timezone.utc)

    # 1. 清理过期告警 (>2小时的直接丢弃)
    expiry = (now - timedelta(hours=2)).isoformat()
    existing = [e for e in existing if e.get("timestamp", "") > expiry]

    # 2. 去重: 同type+ticker, 且price没变 → 跳过
    dedup_window = now - timedelta(minutes=30)
    filtered = []
    for a in alerts:
        is_dup = False
        a_price = a.get("data", {}).get("price")
        for e in existing:
            if (e.get("type") == a["type"] and
                    e.get("ticker") == a["ticker"] and
                    e.get("timestamp", "") > dedup_window.isoformat()):
                # 价格没变才算重复
                e_price = e.get("data", {}).get("price")
                if a_price is not None and e_price is not None and abs(a_price - e_price) < 0.01:
                    is_dup = True
                    break
        if not is_dup:
            a["timestamp"] = now.isoformat()
            filtered.append(a)

    if filtered:
        existing.extend(filtered)
        # Keep only last 50 alerts
        existing = existing[-50:]
        with open(PENDING_ALERTS, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════
def get_session(now_et: datetime) -> str:
    """Determine current market session."""
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


def poll_once(client, tickers: List[str], cfg: dict) -> dict:
    """Single poll cycle: fetch data, detect anomalies, return state."""
    now_et = datetime.now(ET)
    session = get_session(now_et)

    # Fetch data
    snapshots = get_snapshots(client, tickers)
    daily_ref = get_recent_bars(client, tickers, days=25)

    # Cache reference data when available; fall back to cache when API returns empty
    if daily_ref and not daily_ref.get("error"):
        save_ref_cache(daily_ref)
    else:
        cached = load_ref_cache()
        if cached:
            daily_ref = cached
            print(f"  📦 Using cached reference data ({len(cached)} tickers)")

    minute_data = get_minute_bars_today(client, tickers)
    portfolio = load_portfolio()

    # Build state
    tickers_state = {}
    all_alerts = []

    for t in tickers:
        snap = snapshots.get(t, {})
        ref = daily_ref.get(t, {})
        mdata = minute_data.get(t, {})

        price = snap.get("price")
        prev_close = ref.get("prev_close") or ref.get("last_close")

        state = {
            "price": price,
            "prev_close": prev_close,
            "bid": snap.get("bid"),
            "ask": snap.get("ask"),
            "vwap": mdata.get("vwap"),
            "day_volume": mdata.get("day_volume", 0),
            "avg_volume_20d": ref.get("avg_volume_20d", 0),
        }

        # Compute derived metrics
        if price and prev_close and prev_close > 0:
            state["daily_change_pct"] = round((price - prev_close) / prev_close * 100, 2)
        if state["vwap"] and price:
            state["vwap_dev_pct"] = round((price - state["vwap"]) / state["vwap"] * 100, 3)
        if state["avg_volume_20d"] > 0:
            state["vol_ratio"] = round(state["day_volume"] / state["avg_volume_20d"], 2)

        # Portfolio info
        pos = portfolio.get(t)
        if pos:
            state["entry_price"] = pos.get("entry_price")
            state["entry_date"] = pos.get("entry_date")
            if pos.get("entry_price", 0) > 0 and price:
                state["unrealized_pnl_pct"] = round((price - pos["entry_price"]) / pos["entry_price"] * 100, 2)
                state["hold_days"] = (now_et.date() - datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()).days

        tickers_state[t] = state

        # Detect anomalies
        alerts = detect_anomalies(t, snap, ref, mdata, portfolio, cfg, session)
        all_alerts.extend(alerts)

    # W6: 信号退化检测 — 持仓股是否还在最新评分Top-N中
    # 只在market/postmarket session检查(评分在收盘后更新)
    if session in ("market", "postmarket") and portfolio:
        try:
            latest_signals = load_signal(top_n=cfg.get("top_n", 5))
            signal_tickers = {s["ticker"] for s in latest_signals}
            signal_scores = {s["ticker"]: s.get("score", 0) for s in latest_signals}
            
            for t in portfolio:
                if t not in signal_tickers:
                    # 持仓股不在最新Top-N → 信号退化
                    entry_score = portfolio[t].get("score", 0)
                    current_score = signal_scores.get(t, 0)
                    
                    all_alerts.append({
                        "type": "signal_degradation",
                        "severity": "warning",
                        "ticker": t,
                        "message": f"🟡 {t} 信号退化: 不在最新Top-{cfg.get('top_n',5)}中",
                        "data": {
                            "entry_score": entry_score,
                            "current_rank": "不在Top-N",
                            "in_portfolio": True,
                        },
                    })
        except Exception:
            pass  # 信号文件不存在等, 静默跳过

    # Build full state
    state = {
        "timestamp": now_et.isoformat(),
        "session": session,
        "poll_interval": POLL_INTERVAL,
        "tickers": tickers_state,
        "signal_tickers": tickers,
        "has_alerts": len(all_alerts) > 0,
        "alert_count": len(all_alerts),
    }

    # Write state (for dashboard)
    write_state(state)

    # ── L1/L2/L3 分级 + L2/L3深度分析 ──
    if all_alerts:
        enhanced_alerts = []
        for alert in all_alerts:
            level = classify_alert_level(alert, cfg)
            alert["alert_level"] = level

            if level in ("L2", "L3"):
                # 深度分析: 新闻+FinBERT+推荐
                analysis = run_deep_analysis(alert, cfg)
                if analysis:
                    alert["analysis"] = {
                        "recommendation": analysis.get("recommendation", "hold"),
                        "model_reasoning": analysis.get("model_reasoning", ""),
                        "news_context": analysis.get("news_context", ""),
                        "sentiment": analysis.get("sentiment", {}),
                    }
                    # 模型推荐(主要) + 新闻摘要(次要)
                    rec = analysis.get("recommendation", "hold")
                    rec_map = {"reduce": "⚠️减仓", "stop_loss": "🛑止损",
                               "expire": "⏰到期", "hold": "⏳持有"}
                    alert["message"] += f"\n  模型: {rec_map.get(rec, rec)}"
                    alert["message"] += f" | {analysis.get('model_reasoning', '')[:60]}"

                print(f"  🔍 {level} {alert['ticker']}: deep analysis done")

            enhanced_alerts.append(alert)

        write_alerts(enhanced_alerts)

    return state


def run_observer(test_mode: bool = False, once: bool = False):
    """Main observer loop.

    改动: 每次poll重新加载信号+持仓, 确保监控列表始终最新。
    """
    cfg = load_config()

    print(f"🦅 Falcon Observer — {'Test' if test_mode else 'Live'} Mode")
    print(f"=" * 60)

    # Connect to Alpaca (data client for snapshots)
    print(f"🔌 Connecting to Alpaca...")
    client = get_alpaca_client()
    print(f"  ✅ Connected")

    # Signal refresh helper
    def get_current_tickers() -> List[str]:
        """每次调用都读最新信号+持仓, 合并去重。"""
        targets = load_signal(cfg["top_n"])
        signal_tickers = {t["ticker"] for t in targets}
        # 加上实际持仓ticker(可能不在Top5中)
        try:
            portfolio = load_portfolio()
            portfolio_tickers = set(portfolio.keys())
        except Exception:
            portfolio_tickers = set()
        combined = signal_tickers | portfolio_tickers
        # 加上加频监控的ticker
        now_ts = time.time()
        boosted = {t for t, exp in _enhanced_tickers.items() if exp > now_ts}
        combined = combined | boosted
        return list(combined) if combined else list(signal_tickers)

    if test_mode:
        tickers = get_current_tickers()
        print(f"  Targets: {', '.join(tickers)}")
        print(f"  Thresholds: SL={cfg['stop_loss']*100:.0f}%, "
              f"move=±{THRESHOLDS['price_move_pct']}%, "
              f"vol={THRESHOLDS['volume_spike_ratio']}x, "
              f"gap=±{THRESHOLDS['gap_pct']}%")

        print(f"\n📊 Test poll...")
        state = poll_once(client, tickers, cfg)
        print(f"  Session: {state['session']}")
        for t, s in state["tickers"].items():
            price = s.get("price", "?")
            change = s.get("daily_change_pct", "?")
            vwap = s.get("vwap_dev_pct", "?")
            vol = s.get("vol_ratio", "?")
            print(f"  {t}: ${price} ({change}%) VWAP={vwap}% vol={vol}x")
        if state["has_alerts"]:
            print(f"\n  ⚠️ {state['alert_count']} alert(s) detected!")
        else:
            print(f"\n  ✅ No anomalies")
        return

    # Graceful shutdown
    running = [True]

    def handle_signal(signum, frame):
        print(f"\n🛑 Signal {signum}, shutting down...")
        running[0] = False

    sig.signal(sig.SIGINT, handle_signal)
    sig.signal(sig.SIGTERM, handle_signal)

    # Main loop — 每次刷新信号+持仓
    print(f"\n🔄 Starting observer loop (every {POLL_INTERVAL}s)...")
    last_refresh = 0  # 信号刷新时间戳
    tickers = []

    while running[0]:
        now_et = datetime.now(ET)
        session = get_session(now_et)

        # ── 交易日历: 非交易日直接休眠 ──
        today = now_et.date()
        if not is_trading_day(today) and not once:
            nxt = next_trading_day(today)
            days_ahead = (nxt - today).days
            print(f"  📅 非交易日 ({today}), 休眠{days_ahead}天, 下一交易日: {nxt}")
            # 睡到下一个交易日的盘前 (最多睡1小时，循环检查)
            time.sleep(min(3600, days_ahead * 86400))
            continue

        if session == "closed" and not once:
            next_start = now_et.replace(hour=PREMARKET_START, minute=0, second=0, microsecond=0)
            if now_et.hour >= POSTMARKET_END:
                next_start += timedelta(days=1)
            sleep_sec = max(60, (next_start - now_et).total_seconds())
            print(f"  😴 Market closed. Sleeping {sleep_sec/60:.0f} min until pre-market...")
            time.sleep(min(sleep_sec, 3600))
            continue

        # 每5分钟刷新信号+持仓(或首次)
        now_ts = time.time()
        if now_ts - last_refresh > 300 or not tickers:
            tickers = get_current_tickers()
            last_refresh = now_ts

        try:
            state = poll_once(client, tickers, cfg)

            now_str = now_et.strftime("%H:%M:%S")
            alerts_str = f" ⚠️{state['alert_count']}" if state["has_alerts"] else ""
            prices = " | ".join(
                f"{t}:{s.get('price', '?')}" for t, s in state["tickers"].items()
            )
            print(f"  [{now_str}] {session}{alerts_str} | {prices}")

        except Exception as e:
            print(f"  ❌ Poll error: {e}")

        if once:
            break

        # ── 加频监控: L2/L3 ticker → 1分钟轮询 ──
        now_ts2 = time.time()
        active_boosts = {t: exp for t, exp in _enhanced_tickers.items() if exp > now_ts2}
        if active_boosts:
            remaining = min(active_boosts.values()) - now_ts2
            print(f"  ⚡ 加频模式: {list(active_boosts.keys())} | 剩余{remaining/60:.0f}分钟")
            time.sleep(60)  # 1分钟轮询
        else:
            time.sleep(POLL_INTERVAL)

    print(f"\n🦅 Observer stopped.")


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Falcon Observer")
    parser.add_argument("--test", action="store_true", help="Single test poll")
    parser.add_argument("--once", action="store_true", help="Single poll then exit")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))

    if args.test:
        run_observer(test_mode=True)
    elif args.once:
        run_observer(once=True)
    else:
        run_observer()
