#!/usr/bin/env python3
"""
🦅 Falcon Intraday Monitor — 最终版
=====================================
L2 = 盘中异动（每15分钟，需要知道但不急）
  - 个股异常下跌（相对大盘）
  - 大幅单日波动
  - 放量异常
  - 浮亏接近止损线
  - 组合回撤

L3 = 趋势丢失/关键信号（需要动作）
  - 止损触发
  - 趋势逆转（MA20+7天+放量）
  - 信号退化（Falcon评分下降）
  - 分析师目标价集体下调
  - 接近目标价（止盈信号）

大盘阀门：VIX+SPY趋势动态调整阈值
"""

import json, os, sys, time, glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

# ── Paths ──
PROJECT_ROOT = Path.home() / ".hermes" / "openclaw-archive"
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
STATE_FILE = DATA_DIR / "alerts" / "monitor_state.json"
TRIGGER_FILE = DATA_DIR / "alerts" / "trigger.json"
ARCHIVE_DIR = DATA_DIR / "alerts" / "archive"
POSITIONS_FILE = DATA_DIR / "trades" / "positions.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "falcon.yaml"
REF_CACHE = DATA_DIR / "ref_cache.json"
MARKET_REGIME_CACHE = DATA_DIR / "alerts" / "market_regime.json"

# ── Timezone ──
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
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

# ── 冷却 ──
COOLDOWN = {
    "L2": 28800,       # 8小时
    "L2_portfolio": 28800,
    "L3": 86400,       # 24小时（趋势信号每天最多一次）
}


# ═══════════════════════════════════════════════
# 大盘阀门：VIX + SPY趋势
# ═══════════════════════════════════════════════
def get_market_regime() -> dict:
    """获取大盘状态，返回regime配置。

    regime:
      calm   (VIX<20, SPY>MA50) → 正常阈值
      watch  (VIX20-25 或 SPY<MA50) → 略放宽
      stress (VIX25-30 或 SPY连续下跌) → 明显放宽
      panic  (VIX>30) → 大幅放宽，止损暂停
    """
    regime = {
        "level": "calm",
        "vix": 16.0,
        "spy_above_ma50": True,
        "l2_mult": 1.0,    # L2阈值倍数（越大越难触发）
        "l3_mult": 1.0,    # L3阈值倍数
        "stop_loss_active": True,  # panic时暂停止损（大盘崩盘不割肉）
    }

    try:
        import yfinance as yf

        # VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if len(vix_hist) > 0:
            regime["vix"] = float(vix_hist["Close"].iloc[-1])

        # SPY趋势
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="60d")
        if len(spy_hist) >= 50:
            closes = spy_hist["Close"].values
            ma50 = closes[-50:].mean()
            current = float(closes[-1])
            regime["spy_above_ma50"] = current > ma50

            # 连续下跌天数
            consecutive_down = 0
            for i in range(len(closes)-1, 0, -1):
                if closes[i] < closes[i-1]:
                    consecutive_down += 1
                else:
                    break
            regime["spy_consecutive_down"] = consecutive_down

        # 判定regime
        vix = regime["vix"]
        if vix >= 30:
            regime["level"] = "panic"
            regime["l2_mult"] = 2.0   # L2阈值翻倍（减少噪音）
            regime["l3_mult"] = 1.5   # L3止损放宽50%
            regime["stop_loss_active"] = False  # 恐慌期不自动止损
        elif vix >= 25 or not regime["spy_above_ma50"]:
            regime["level"] = "stress"
            regime["l2_mult"] = 1.5
            regime["l3_mult"] = 1.3
        elif vix >= 20:
            regime["level"] = "watch"
            regime["l2_mult"] = 1.2
            regime["l3_mult"] = 1.1
        else:
            regime["level"] = "calm"
            regime["l2_mult"] = 1.0
            regime["l3_mult"] = 1.0

    except Exception as e:
        print(f"  ⚠️ 市场数据获取失败: {e}", file=sys.stderr)

    # 缓存（避免频繁API调用）
    save_json(MARKET_REGIME_CACHE, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
    })

    return regime


# ═══════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════
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
    return "closed"


def safe_parse_date(date_str: str) -> Optional[datetime]:
    if not date_str or not date_str.strip():
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def append_archive(alert: dict, now_et: datetime):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive_file = ARCHIVE_DIR / f"{now_et.strftime('%Y-%m-%d')}.jsonl"
    with open(archive_file, "a") as f:
        f.write(json.dumps({"timestamp": now_et.isoformat(), **alert}, ensure_ascii=False, default=str) + "\n")


def load_config() -> dict:
    cfg = {"top_n": 5, "hold_days": 30, "stop_loss": -0.15}
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
        t = raw.get("trading", {})
        cfg["stop_loss"] = t.get("stop_loss", -0.15)
        cfg["hold_days"] = t.get("hold_days", 30)
        cfg["top_n"] = raw.get("model", {}).get("top_n", 5)
    except Exception:
        pass
    return cfg


# ═══════════════════════════════════════════════
# Data Fetching
# ═══════════════════════════════════════════════
def get_alpaca_client():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    from alpaca.data.historical import StockHistoricalDataClient
    key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca credentials")
    return StockHistoricalDataClient(key, secret)


def get_snapshots(client, tickers: List[str]) -> Dict[str, dict]:
    from alpaca.data.requests import StockSnapshotRequest
    try:
        from alpaca.data.enums import DataFeed
        req = StockSnapshotRequest(symbol_or_symbols=tickers, feed=DataFeed.IEX)
        snaps = client.get_stock_snapshot(req)
        return {sym: {"price": float(snap.latest_trade.price) if snap.latest_trade else None}
                for sym, snap in snaps.items()}
    except Exception as e:
        return {"error": str(e)}


def get_daily_bars(client, tickers: List[str], days: int = 60) -> Dict[str, dict]:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days + 15)
        req = StockBarsRequest(symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
                               start=start, end=end, feed=DataFeed.IEX)
        bars = client.get_stock_bars(req)
        df = bars.df
        result = {}
        for sym in tickers:
            if sym not in df.index.get_level_values(0):
                continue
            sub = df.loc[sym]
            if len(sub) < 20:
                continue
            closes = sub["close"].values
            highs = sub["high"].values
            lows = sub["low"].values

            # ATR
            trs = []
            for i in range(1, len(sub)):
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                trs.append(tr)
            atr = sum(trs[-14:]) / min(14, len(trs)) if trs else 0

            # MA20
            ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else float(closes[-1])

            # 连续下跌
            consecutive_down = 0
            for i in range(len(closes)-1, 0, -1):
                if closes[i] < closes[i-1]:
                    consecutive_down += 1
                else:
                    break

            # 分析师目标价
            analyst_target = None
            try:
                fund_file = PROJECT_ROOT / "data" / "us" / "fundamentals_latest.parquet"
                if fund_file.exists():
                    import pandas as pd
                    fund_df = pd.read_parquet(fund_file)
                    if sym in fund_df.index:
                        row = fund_df.loc[sym]
                        analyst_target = getattr(row, 'targetMeanPrice', None) or getattr(row, 'analystTargetPrice', None)
            except Exception:
                pass

            result[sym] = {
                "prev_close": float(sub.iloc[-2]["close"]),
                "last_close": float(sub.iloc[-1]["close"]),
                "last_volume": float(sub.iloc[-1]["volume"]),
                "avg_volume_20d": float(sub["volume"].tail(20).mean()),
                "atr": atr,
                "atr_pct": atr / float(closes[-1]) if float(closes[-1]) > 0 else 0,
                "ma20": ma20,
                "above_ma20": float(closes[-1]) > ma20,
                "consecutive_down": consecutive_down,
                "analyst_target": analyst_target,
            }
        return result
    except Exception as e:
        return {"error": str(e)}


def save_ref_cache(data: dict):
    save_json(REF_CACHE, {"timestamp": datetime.now(timezone.utc).isoformat(), "data": data})


def load_ref_cache() -> dict:
    cache = load_json(REF_CACHE)
    if not cache:
        return {}
    try:
        ts = datetime.fromisoformat(cache["timestamp"].replace("Z", "+00:00"))
        if (datetime.now(timezone.utc) - ts).total_seconds() > 172800:
            return {}
        return cache.get("data", {})
    except Exception:
        return {}


# ═══════════════════════════════════════════════
# Portfolio
# ═══════════════════════════════════════════════
def load_portfolio() -> Dict[str, dict]:
    portfolio = {}
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))
        from broker_adapter import get_broker
        broker = get_broker()
        now = datetime.now(ET)
        for pos in broker.get_positions():
            entry_dt = safe_parse_date(pos.entry_date or "")
            portfolio[pos.symbol] = {
                "entry_price": pos.avg_entry_price,
                "entry_date": pos.entry_date or "",
                "shares": pos.qty,
                "hold_days": (now - entry_dt).days if entry_dt else None,
            }
    except Exception:
        data = load_json(POSITIONS_FILE)
        if data:
            for sym, info in data.get("positions", {}).items():
                entry_dt = safe_parse_date(info.get("entry_date", ""))
                portfolio[sym] = {
                    "entry_price": info.get("entry_price", 0),
                    "entry_date": info.get("entry_date", ""),
                    "shares": info.get("qty", 0),
                    "hold_days": (datetime.now(ET) - entry_dt).days if entry_dt else None,
                }
    return portfolio


def load_signal_tickers(top_n: int = 5) -> List[str]:
    pattern = str(DATA_DIR / "falcon*_scored_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return []
    data = load_json(Path(files[-1]))
    if not data:
        return []
    return [p.get("sym") or p.get("ticker", "") for p in data.get("picks", [])[:top_n]
            if p.get("sym") or p.get("ticker")]


def load_signal_scores() -> Dict[str, float]:
    pattern = str(DATA_DIR / "falcon*_scored_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    data = load_json(Path(files[-1]))
    if not data:
        return {}
    return {p.get("sym") or p.get("ticker", ""): float(p.get("score", 0))
            for p in data.get("picks", []) if p.get("sym") or p.get("ticker")}


# ═══════════════════════════════════════════════
# 获取SPY当日涨跌（大盘基准）
# ═══════════════════════════════════════════════
def get_spy_daily_change(client) -> float:
    """返回SPY当日涨跌幅。"""
    try:
        from alpaca.data.requests import StockSnapshotRequest
        from alpaca.data.enums import DataFeed
        req = StockSnapshotRequest(symbol_or_symbols=["SPY"], feed=DataFeed.IEX)
        snaps = client.get_stock_snapshot(req)
        spy = snaps.get("SPY")
        if spy and spy.latest_trade:
            price = float(spy.latest_trade.price)
            # 用ref_cache的prev_close
            ref = load_ref_cache()
            spy_ref = ref.get("SPY", {})
            prev = spy_ref.get("prev_close")
            if prev and prev > 0:
                return (price - prev) / prev
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════
def detect_alerts(
    ticker: str,
    price: float,
    daily_ref: dict,
    portfolio_info: Optional[dict],
    signal_score: Optional[float],
    spy_change: float,
    regime: dict,
    cfg: dict,
) -> List[dict]:
    alerts = []
    prev_close = daily_ref.get("prev_close") or daily_ref.get("last_close")
    if not prev_close or prev_close <= 0:
        return alerts

    daily_change = (price - prev_close) / prev_close
    excess_change = daily_change - spy_change  # 个股相对大盘的超额跌幅
    atr_pct = daily_ref.get("atr_pct", 0)
    ma20 = daily_ref.get("ma20")
    above_ma20 = daily_ref.get("above_ma20", True)
    consecutive_down = daily_ref.get("consecutive_down", 0)
    analyst_target = daily_ref.get("analyst_target")

    # 持仓上下文
    unrealized_pnl = None
    entry_price = None
    hold_days = None
    if portfolio_info and portfolio_info.get("entry_price", 0) > 0:
        entry_price = portfolio_info["entry_price"]
        unrealized_pnl = (price - entry_price) / entry_price
        hold_days = portfolio_info.get("hold_days")

    stop_loss = cfg.get("stop_loss", -0.15)
    l2m = regime["l2_mult"]
    l3m = regime["l3_mult"]

    # ═══ L3: 趋势丢失/关键信号 ═══

    # 3a. 止损（恐慌期暂停）
    if regime["stop_loss_active"] and unrealized_pnl is not None and unrealized_pnl <= stop_loss * l3m:
        alerts.append({
            "level": "L3", "type": "stop_loss", "ticker": ticker,
            "message": f"🔴 止损触发: {ticker} 浮亏{unrealized_pnl*100:.1f}% (阈值{stop_loss*100:.0f}%, regime={regime['level']})",
            "data": {"price": price, "entry": entry_price, "pnl": unrealized_pnl, "hold_days": hold_days, "regime": regime["level"]},
        })

    # 3b. 趋势逆转：跌破MA20 + 连续7天下跌 + 放量确认
    if (not above_ma20 and consecutive_down >= 7):
        # 放量确认：最近一天成交量≥1.5x均量
        vol = daily_ref.get("last_volume", 0)
        avg_vol = daily_ref.get("avg_volume_20d", 0)
        vol_confirm = (vol / avg_vol >= 1.5) if avg_vol > 0 else False
        if vol_confirm:
            alerts.append({
                "level": "L3", "type": "trend_break", "ticker": ticker,
                "message": f"🔴 {ticker} 趋势逆转: 跌破MA20(${ma20:.2f}), 连续{consecutive_down}天下跌, 放量确认",
                "data": {"price": price, "ma20": ma20, "consecutive_down": consecutive_down, "hold_days": hold_days},
            })

    # 3c. 接近分析师目标价（止盈信号）
    if analyst_target and analyst_target > 0:
        distance = (price - analyst_target) / analyst_target
        if -0.03 <= distance <= 0.02:  # 距目标价-3%~+2%
            alerts.append({
                "level": "L3", "type": "near_target", "ticker": ticker,
                "message": f"💰 {ticker} 接近目标价${analyst_target:.2f} (当前${price:.2f}, 差距{distance*100:.1f}%)",
                "data": {"price": price, "target": analyst_target, "distance": distance},
            })

    # ═══ L2: 盘中异动 ═══

    # 2a. 个股异常下跌（相对大盘，阈值随regime调整）
    base_threshold = 0.05 * l2m  # calm=5%, stress=7.5%, panic=10%
    if excess_change <= -base_threshold and spy_change > -0.02:
        # 大盘没怎么跌，个股跌了5%+（调整后）
        alerts.append({
            "level": "L2", "type": "excess_drop", "ticker": ticker,
            "message": f"🟠 {ticker} 异常下跌{daily_change*100:.1f}% (大盘{spy_change*100:+.1f}%, 超额{excess_change*100:.1f}%)",
            "data": {"price": price, "change": daily_change, "spy_change": spy_change, "excess": excess_change},
        })

    # 2b. 大幅单日波动（阈值随regime调整）
    big_move = max(0.08, 2.5 * atr_pct) * l2m if atr_pct > 0 else 0.08 * l2m
    if abs(daily_change) >= big_move and abs(excess_change) < base_threshold:
        # 已经被2a捕获的不重复
        direction = "↑" if daily_change > 0 else "↓"
        alerts.append({
            "level": "L2", "type": "large_move", "ticker": ticker,
            "message": f"🟠 {ticker} {direction}{abs(daily_change)*100:.1f}% (阈值{big_move*100:.1f}%, regime={regime['level']})",
            "data": {"price": price, "change": daily_change, "regime": regime["level"]},
        })

    # 2c. 浮亏接近止损线
    warn_line = -0.12 * l2m  # calm=12%, stress=18%, panic=24%
    if unrealized_pnl is not None and stop_loss < unrealized_pnl <= warn_line:
        alerts.append({
            "level": "L2", "type": "pnl_warn", "ticker": ticker,
            "message": f"🟠 {ticker} 浮亏{unrealized_pnl*100:.1f}% (止损线{stop_loss*100:.0f}%, entry=${entry_price:.2f})",
            "data": {"price": price, "entry": entry_price, "pnl": unrealized_pnl, "hold_days": hold_days},
        })

    # 2d. 放量≥5x
    avg_vol = daily_ref.get("avg_volume_20d", 0)
    last_vol = daily_ref.get("last_volume", 0)
    if avg_vol > 0 and last_vol > 0:
        now_et = datetime.now(ET)
        market_min = (now_et.hour * 60 + now_et.minute) - 570
        if market_min > 30:
            est_vol = last_vol * (390 / market_min)
            if est_vol / avg_vol >= 5.0:
                alerts.append({
                    "level": "L2", "type": "volume_spike", "ticker": ticker,
                    "message": f"🟠 {ticker} 放量{est_vol/avg_vol:.1f}x均量",
                    "data": {"volume": last_vol, "avg_volume": avg_vol},
                })

    return alerts


def detect_portfolio_alerts(portfolio: Dict[str, dict], prices: Dict[str, float], regime: dict) -> List[dict]:
    alerts = []
    if not portfolio:
        return alerts
    total_cost = sum(info.get("entry_price", 0) * info.get("shares", 0) for info in portfolio.values())
    total_value = sum(prices.get(sym, info.get("current_price", 0)) * info.get("shares", 0)
                      for sym, info in portfolio.items())
    if total_cost > 0:
        pnl = (total_value - total_cost) / total_cost
        threshold = -0.10 * regime["l2_mult"]
        if pnl <= threshold:
            alerts.append({
                "level": "L2", "type": "portfolio_drawdown", "ticker": "PORTFOLIO",
                "message": f"🟠 组合回撤{pnl*100:.1f}% (阈值{threshold*100:.0f}%, regime={regime['level']})",
                "data": {"pnl": pnl, "value": total_value, "cost": total_cost, "regime": regime["level"]},
            })
    return alerts


# ═══════════════════════════════════════════════
# State / Cooldown
# ═══════════════════════════════════════════════
def load_state() -> dict:
    return load_json(STATE_FILE) or {"sent": {}, "last_run": 0}


def save_state(state: dict):
    save_json(STATE_FILE, state)


def is_cooled_down(state: dict, alert: dict, now_ts: float) -> bool:
    sent = state.get("sent", {})
    ticker = alert["ticker"]
    level = alert["level"]
    atype = alert["type"]

    if ticker == "PORTFOLIO":
        key = f"portfolio_{level}"
        cd = COOLDOWN.get("L2_portfolio", 28800)
    else:
        key = f"{ticker}:{level}:{atype}"
        cd = COOLDOWN.get(level, 28800)

    return now_ts - sent.get(key, 0) >= cd


def mark_sent(state: dict, alert: dict, now_ts: float):
    sent = state.setdefault("sent", {})
    key = f"portfolio_{alert['level']}" if alert["ticker"] == "PORTFOLIO" else f"{alert['ticker']}:{alert['level']}:{alert['type']}"
    sent[key] = now_ts
    state["sent"] = {k: v for k, v in sent.items() if now_ts - v < 172800}


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════
def main():
    now_et = datetime.now(ET)
    session = get_session(now_et)
    now_ts = time.time()

    if session == "closed":
        return

    cfg = load_config()
    state = load_state()

    # 大盘阀门
    regime = get_market_regime()

    try:
        client = get_alpaca_client()
    except Exception as e:
        print(f"❌ Alpaca连接失败: {e}", file=sys.stderr)
        sys.exit(1)

    # SPY当日涨跌
    spy_change = get_spy_daily_change(client)

    signal_tickers = load_signal_tickers(cfg["top_n"])
    signal_scores = load_signal_scores()
    portfolio = load_portfolio()
    all_tickers = list(set(signal_tickers) | set(portfolio.keys()))
    if not all_tickers:
        return

    snapshots = get_snapshots(client, all_tickers)
    if snapshots.get("error"):
        if not load_ref_cache():
            return

    daily_ref = get_daily_bars(client, all_tickers, days=60)
    if daily_ref.get("error"):
        daily_ref = load_ref_cache()
        if not daily_ref:
            return
    else:
        save_ref_cache(daily_ref)

    # 检测
    all_alerts = []
    prices = {}
    for ticker in all_tickers:
        snap = snapshots.get(ticker, {})
        price = snap.get("price")
        if not price:
            continue
        prices[ticker] = price
        ref = daily_ref.get(ticker, {})
        pos_info = portfolio.get(ticker)
        score = signal_scores.get(ticker)
        all_alerts.extend(detect_alerts(ticker, price, ref, pos_info, score, spy_change, regime, cfg))

    all_alerts.extend(detect_portfolio_alerts(portfolio, prices, regime))

    # 存档
    for alert in all_alerts:
        append_archive(alert, now_et)

    # 冷却过滤
    to_send = []
    for alert in all_alerts:
        if is_cooled_down(state, alert, now_ts):
            to_send.append(alert)
            mark_sent(state, alert, now_ts)

    # L2/L3写trigger.json
    if to_send:
        save_json(TRIGGER_FILE, {
            "timestamp": now_et.isoformat(),
            "session": session,
            "regime": regime,
            "spy_change": spy_change,
            "alerts": to_send,
            "portfolio": {sym: {"entry_price": info.get("entry_price"), "shares": info.get("shares"), "hold_days": info.get("hold_days")}
                          for sym, info in portfolio.items()},
            "prices": prices,
        })

    # 输出
    if to_send:
        regime_label = {"calm": "🟢正常", "watch": "🟡关注", "stress": "🟠压力", "panic": "🔴恐慌"}
        lines = [f"🦅 **Falcon 异动告警**"]
        lines.append(f"⏰ {now_et.strftime('%H:%M ET')} | {regime_label.get(regime['level'], regime['level'])} VIX={regime['vix']:.0f} SPY{spy_change*100:+.1f}%")
        lines.append("")

        l3 = [a for a in to_send if a["level"] == "L3"]
        l2 = [a for a in to_send if a["level"] == "L2"]

        if l3:
            lines.append("🔴 **L3 趋势信号**")
            for a in l3:
                lines.append(f"  {a['message']}")
            lines.append("")

        if l2:
            lines.append("🟠 **L2 盘中异动**")
            for a in l2:
                lines.append(f"  {a['message']}")
            lines.append("")

        lines.append(f"🧠 LLM分析已触发（{len(to_send)}条）")
        print("\n".join(lines))

    state["last_run"] = now_ts
    save_state(state)


if __name__ == "__main__":
    main()
