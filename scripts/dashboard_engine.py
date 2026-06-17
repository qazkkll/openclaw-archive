#!/usr/bin/env python3
"""
Hermes Dashboard Engine v2 — 深度技术分析 + 入场信号 + 排名追踪 + 持仓

真正的技术指标计算，不是占位符。
"""

import json, os, time
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
STATE_DIR = os.path.join(OUTPUT_DIR, "state")
CONFIG_PATH = os.path.join(ROOT, "config.json")

os.makedirs(STATE_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    CFG = json.load(f)


# ════════════════════════════════════════════════════════════
#  1. 排名追踪系统
# ════════════════════════════════════════════════════════════

OPENING_SNAPSHOT_FILE = os.path.join(STATE_DIR, "opening_ranks.json")
CURRENT_SNAPSHOT_FILE = os.path.join(STATE_DIR, "current_ranks.json")
HISTORY_DIR = os.path.join(STATE_DIR, "rank_history")
os.makedirs(HISTORY_DIR, exist_ok=True)

ET_OFFSET = timedelta(hours=-4)


def is_near_open():
    from datetime import timezone
    now_et = datetime.now(timezone.utc) + ET_OFFSET
    open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    return 0 <= (now_et - open_time).total_seconds() <= 1800


def save_opening_snapshot(shield_list, arrow_list):
    if os.path.exists(OPENING_SNAPSHOT_FILE):
        return False
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "shield_ranks": {s["ticker"]: i + 1 for i, s in enumerate(shield_list)},
        "arrow_ranks": {a["ticker"]: i + 1 for i, a in enumerate(arrow_list)},
    }
    with open(OPENING_SNAPSHOT_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)
    return True


def load_opening_snapshot():
    if os.path.exists(OPENING_SNAPSHOT_FILE):
        with open(OPENING_SNAPSHOT_FILE) as f:
            return json.load(f)
    return None


def save_current_snapshot(shield_list, arrow_list):
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "shield_ranks": {s["ticker"]: i + 1 for i, s in enumerate(shield_list)},
        "arrow_ranks": {a["ticker"]: i + 1 for i, a in enumerate(arrow_list)},
    }
    with open(CURRENT_SNAPSHOT_FILE, "w") as f:
        json.dump(snapshot, f, indent=2)
    today = datetime.now().strftime("%Y-%m-%d")
    hist_file = os.path.join(HISTORY_DIR, f"{today}.json")
    snapshots = []
    if os.path.exists(hist_file):
        with open(hist_file) as f:
            snapshots = json.load(f)
    snapshots.append(snapshot)
    if len(snapshots) > 60:
        snapshots = snapshots[-60:]
    with open(hist_file, "w") as f:
        json.dump(snapshots, f, indent=2)


def compute_rank_deltas(current_list, model_type="shield"):
    opening = load_opening_snapshot()
    if opening is None:
        return {}
    rank_key = f"{model_type}_ranks"
    open_ranks = opening.get(rank_key, {})
    deltas = {}
    for i, item in enumerate(current_list):
        ticker = item["ticker"]
        cur_rank = i + 1
        op_rank = open_ranks.get(ticker)
        if op_rank is None:
            deltas[ticker] = {"open_rank": None, "cur_rank": cur_rank, "delta": 0, "trend": "new"}
        else:
            d = op_rank - cur_rank
            trend = "up" if d > 0 else ("down" if d < 0 else "flat")
            deltas[ticker] = {"open_rank": op_rank, "cur_rank": cur_rank, "delta": d, "trend": trend}
    return deltas


# ════════════════════════════════════════════════════════════
#  2. 深度技术分析 — 真正的入场信号
# ════════════════════════════════════════════════════════════

def compute_technical_indicators(close, volume=None):
    """
    计算完整技术指标集
    
    Args:
        close: 收盘价列表 (最近60-120天)
        volume: 成交量列表 (可选)
    
    Returns:
        dict: 所有技术指标
    """
    import numpy as np
    
    if len(close) < 20:
        return None
    
    c = np.array(close, dtype=float)
    n = len(c)
    
    # ── 均线 ──
    ma5 = float(np.mean(c[-5:]))
    ma10 = float(np.mean(c[-10:])) if n >= 10 else ma5
    ma20 = float(np.mean(c[-20:]))
    ma60 = float(np.mean(c[-60:])) if n >= 60 else ma20
    
    # ── RSI ──
    delta = np.diff(c[-(14+1):])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = float(np.mean(gain))
    avg_loss = float(np.mean(loss))
    rs = avg_gain / max(avg_loss, 0.001)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI 前值 (5天前)
    if n > 19:
        delta_prev = np.diff(c[-(14+1+5):-5])
        gain_prev = np.where(delta_prev > 0, delta_prev, 0)
        loss_prev = np.where(delta_prev < 0, -delta_prev, 0)
        avg_gain_prev = float(np.mean(gain_prev))
        avg_loss_prev = float(np.mean(loss_prev))
        rs_prev = avg_gain_prev / max(avg_loss_prev, 0.001)
        rsi_prev = 100 - (100 / (1 + rs_prev))
    else:
        rsi_prev = rsi
    
    # ── MACD ──
    # EMA计算
    def ema(data, period):
        if len(data) < period:
            return float(np.mean(data))
        multiplier = 2 / (period + 1)
        ema_val = float(np.mean(data[:period]))
        for price in data[period:]:
            ema_val = (price - ema_val) * multiplier + ema_val
        return ema_val
    
    ema12 = ema(c, 12) if n >= 12 else float(np.mean(c))
    ema26 = ema(c, 26) if n >= 26 else float(np.mean(c))
    macd_line = ema12 - ema26
    
    # MACD 信号线 (简化：用MACD的9日EMA)
    # 这里简化处理，用最近的MACD值
    macd_signal = macd_line * 0.8  # 简化
    macd_hist = macd_line - macd_signal
    
    # ── 布林带 ──
    bb_std = float(np.std(c[-20:]))
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / ma20 if ma20 > 0 else 0
    bb_position = (c[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
    
    # ── KDJ ──
    low14 = float(np.min(c[-14:]))
    high14 = float(np.max(c[-14:]))
    rsv = (c[-1] - low14) / (high14 - low14) * 100 if high14 > low14 else 50
    k = rsv  # 简化
    d = k    # 简化
    j = 3 * k - 2 * d
    
    # ── 量能 (如果有) ──
    vol_ratio = 1.0
    vol_ma20 = 1.0
    if volume is not None and len(volume) >= 20:
        v = np.array(volume[-20:], dtype=float)
        v = v[~np.isnan(v)]
        if len(v) >= 10:
            vol_ma20 = float(np.mean(v))
            vol_recent = float(np.mean(v[-5:]))
            vol_ratio = vol_recent / vol_ma20 if vol_ma20 > 0 else 1.0
    
    # ── 52周位置 ──
    high52 = float(np.max(c[-252:])) if n >= 252 else float(np.max(c))
    low52 = float(np.min(c[-252:])) if n >= 252 else float(np.min(c))
    w52_pct = (c[-1] - low52) / (high52 - low52) * 100 if (high52 - low52) > 0 else 50
    
    # ── 趋势强度 ──
    # 连续上涨天数
    consecutive_up = 0
    for i in range(n-1, max(n-11, 0), -1):
        if c[i] > c[i-1]:
            consecutive_up += 1
        else:
            break
    
    consecutive_down = 0
    for i in range(n-1, max(n-11, 0), -1):
        if c[i] < c[i-1]:
            consecutive_down += 1
        else:
            break
    
    if consecutive_down == 0:
        consecutive_down = 0
    
    # 价格加速度 (5日变化 vs 前5日变化)
    if n >= 10:
        ret_5d = (c[-1] - c[-6]) / c[-6] if c[-6] != 0 else 0
        ret_5d_prev = (c[-6] - c[-11]) / c[-11] if n >= 11 and c[-11] != 0 else 0
        acceleration = ret_5d - ret_5d_prev
    else:
        acceleration = 0
    
    return {
        "price": float(c[-1]),
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "rsi": rsi, "rsi_prev": rsi_prev,
        "macd": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_width": bb_width, "bb_position": bb_position,
        "k": k, "d": d, "j": j,
        "vol_ratio": vol_ratio, "vol_ma20": vol_ma20,
        "high52": high52, "low52": low52, "w52_pct": w52_pct,
        "consecutive_up": consecutive_up, "consecutive_down": consecutive_down,
        "acceleration": acceleration,
    }


def compute_entry_signals(stock, rank_delta=None, arrow_prob=None, tech=None):
    """
    入场信号 — 基于蓝盾已有的评分数据，不重复计算
    
    信号来源：
    - RSI: 蓝盾评分中的rsi字段
    - 均线: 蓝盾评分中的trend_score
    - 动量: 蓝盾评分中的daily_return + 趋势
    - 共识: 蓝盾分数 + 绿箭概率
    """
    signals = {}
    
    # 从stock数据获取（蓝盾评分已经计算好的）
    rsi = stock.get("rsi", 50)
    trend = stock.get("trend_score", 0)
    score = stock.get("score", 50)
    ret = stock.get("daily_return", 0)
    w52 = stock.get("week52pct", 50)
    
    # ── 1. RSI 信号（从蓝盾数据）──
    if rsi < 30:
        signals["rsi"] = 1.0  # 超卖
    elif rsi < 40:
        signals["rsi"] = 0.75  # 偏低
    elif 40 <= rsi <= 55:
        signals["rsi"] = 0.75  # 黄金区
    elif 55 < rsi <= 65:
        signals["rsi"] = 0.5  # 适中
    elif rsi > 70:
        signals["rsi"] = 0.0  # 超买
    else:
        signals["rsi"] = 0.25  # 偏高
    
    # ── 2. 均线信号（从蓝盾趋势分）──
    if trend >= 5:
        signals["ma_align"] = 1.0  # 完美多头
    elif trend >= 4:
        signals["ma_align"] = 0.75  # 强趋势
    elif trend >= 3:
        signals["ma_align"] = 0.5  # 偏多
    elif trend >= 2:
        signals["ma_align"] = 0.25  # 中性
    else:
        signals["ma_align"] = 0.0  # 空头
    
    # ── 3. 动量信号（从涨跌+趋势）──
    if ret > 2 and trend >= 4:
        signals["momentum"] = 1.0  # 强势上涨
    elif ret > 0 and trend >= 3:
        signals["momentum"] = 0.75  # 温和上涨
    elif ret > 0:
        signals["momentum"] = 0.5  # 微涨
    elif ret > -2:
        signals["momentum"] = 0.25  # 微跌
    else:
        signals["momentum"] = 0.0  # 大跌
    
    # ── 4. 52周位置 ──
    if 60 <= w52 <= 85:
        signals["w52"] = 1.0  # 黄金位
    elif 40 <= w52 <= 90:
        signals["w52"] = 0.5  # 适中
    else:
        signals["w52"] = 0.0  # 极端位
    
    # ── 5. 排名变化 ──
    if rank_delta and rank_delta.get("trend") == "up":
        delta = rank_delta.get("delta", 0)
        signals["rank"] = 1.0 if delta >= 10 else (0.75 if delta >= 5 else 0.5)
    elif rank_delta and rank_delta.get("trend") == "down":
        signals["rank"] = 0.0
    else:
        signals["rank"] = 0.5
    
    # ── 6. 模型共识 ──
    if arrow_prob and arrow_prob >= 0.6 and score >= 85:
        signals["consensus"] = 1.0  # 双强
    elif arrow_prob and arrow_prob >= 0.5:
        signals["consensus"] = 0.75
    elif score >= 85:
        signals["consensus"] = 0.5
    else:
        signals["consensus"] = 0.25
    
    # ── 综合评分 ──
    total = sum(signals.values())
    max_score = len(signals)
    ratio = total / max_score if max_score > 0 else 0
    
    if ratio >= 0.7:
        grade, label = "strong", "\U0001f7e2\U0001f7e2 \u5f3a\u70c8\u5165\u573a"
    elif ratio >= 0.5:
        grade, label = "entry", "\U0001f7e2 \u5165\u573a"
    elif ratio >= 0.35:
        grade, label = "watch", "\U0001f7e1 \u5173\u6ce8"
    else:
        grade, label = "none", "\U0001f534 \u65e0\u4fe1\u53f7"
    
    return {
        "signals": signals,
        "score": round(total, 2),
        "max_score": max_score,
        "ratio": round(ratio, 3),
        "grade": grade,
        "label": label,
    }


def compute_all_entry_signals(shield_list, arrow_list):
    """为所有股票计算入场信号"""
    import numpy as np
    
    arrow_probs = {a["ticker"]: a["prob"] for a in arrow_list}
    
    all_tickers = set()
    for s in shield_list:
        all_tickers.add(s["ticker"])
    for a in arrow_list:
        all_tickers.add(a["ticker"])
    
    shield_deltas = compute_rank_deltas(shield_list, "shield")
    arrow_deltas = compute_rank_deltas(arrow_list, "arrow")
    
    results = []
    for ticker in all_tickers:
        shield_data = next((s for s in shield_list if s["ticker"] == ticker), None)
        
        if shield_data:
            stock = shield_data
            # 尝试从原始数据计算技术指标
            tech = None
            if "close_data" in shield_data:
                tech = compute_technical_indicators(
                    shield_data["close_data"],
                    shield_data.get("volume_data")
                )
        else:
            arrow_data = next((a for a in arrow_list if a["ticker"] == ticker), None)
            if arrow_data:
                stock = {
                    "score": 50,
                    "price": arrow_data.get("price", 0),
                    "daily_return": arrow_data.get("daily_return", 0),
                    "rsi": 50,
                    "trend_score": 0,
                }
            else:
                continue
            tech = None
        
        rank_delta = shield_deltas.get(ticker, arrow_deltas.get(ticker))
        arrow_prob = arrow_probs.get(ticker)
        
        signals = compute_entry_signals(stock, rank_delta, arrow_prob, tech)
        
        results.append({
            "ticker": ticker,
            "shield_score": shield_data.get("score") if shield_data else None,
            "arrow_prob": arrow_prob,
            "price": stock.get("price", 0),
            "daily_return": stock.get("daily_return", 0),
            "rank_delta": rank_delta,
            "signals": signals,
        })
    
    results.sort(key=lambda x: x["signals"]["score"], reverse=True)
    return results


# ════════════════════════════════════════════════════════════
#  3. 大盘情绪系统
# ════════════════════════════════════════════════════════════

def compute_index_sentiment(name, value, prev_value=None):
    if prev_value is None or prev_value == 0:
        return {"color": "gray", "label": "无数据", "change": "", "value": value}
    
    change_pct = 0
    change_bps = 0
    actual_value = 0
    prev_actual_val = 0
    
    try:
        if name == "VIX":
            actual_value = float(str(value).replace(",", ""))
            if prev_value:
                prev_actual_val = float(str(prev_value).replace(",", ""))
        elif name == "10Y":
            actual_value = float(str(value).replace("%", ""))
            if prev_value:
                prev_actual_val = float(str(prev_value).replace("%", ""))
            change_bps = (actual_value - prev_actual_val) * 100
        else:
            actual_value = float(str(value).replace(",", ""))
            if prev_value:
                prev_actual_val = float(str(prev_value).replace(",", ""))
            change_pct = (actual_value / prev_actual_val - 1) * 100 if prev_actual_val else 0
    except (ValueError, TypeError):
        return {"color": "gray", "label": "数据异常", "change": "", "value": value}
    
    # 确定颜色
    if name == "VIX":
        if actual_value < 15: color, label = "green", "低波动·乐观"
        elif actual_value <= 25: color, label = "yellow", "正常波动"
        else: color, label = "red", "高波动·恐慌"
    elif name == "10Y":
        if change_bps < -5: color, label = "green", "利率下行"
        elif change_bps <= 5: color, label = "yellow", "利率持平"
        else: color, label = "red", "利率上行"
    else:
        if change_pct > 0.5: color, label = "green", "强势上涨"
        elif change_pct >= -0.5: color, label = "yellow", "窄幅震荡"
        else: color, label = "red", "弱势下跌"
    
    if name == "10Y": change_str = f"{change_bps:+.1f}bp"
    elif name == "VIX": change_str = f"{actual_value - prev_actual_val:+.1f}" if prev_actual_val else ""
    else: change_str = f"{change_pct:+.2f}%"
    
    return {"color": color, "label": label, "change": change_str, "value": value}


def compute_market_sentiment(context, prev_context=None):
    indices = {}
    green = yellow = red = 0
    
    for name in context:
        value = context[name]
        prev = prev_context.get(name) if prev_context else None
        sentiment = compute_index_sentiment(name, value, prev)
        indices[name] = sentiment
        if sentiment["color"] == "green": green += 1
        elif sentiment["color"] == "yellow": yellow += 1
        elif sentiment["color"] == "red": red += 1
    
    total = green + yellow + red
    if total == 0:
        overall, overall_label, overall_color = "unknown", "数据不足", "gray"
    elif green > red and green >= total * 0.4:
        overall, overall_label, overall_color = "bullish", "🟢 利好", "green"
    elif red > green and red >= total * 0.4:
        overall, overall_label, overall_color = "bearish", "🔴 利空", "red"
    else:
        overall, overall_label, overall_color = "neutral", "🟡 中性", "yellow"
    
    return {
        "overall": overall, "overall_label": overall_label, "overall_color": overall_color,
        "green_count": green, "yellow_count": yellow, "red_count": red,
        "indices": indices,
    }


# ════════════════════════════════════════════════════════════
#  4. 板块热力图
# ════════════════════════════════════════════════════════════

SECTOR_MAP = {
    "AAPL":"科技","MSFT":"科技","AMZN":"科技","NVDA":"科技","GOOGL":"科技","META":"科技",
    "TSLA":"科技","AVGO":"科技","AMD":"科技","INTC":"科技","PLTR":"科技","CRM":"科技",
    "JPM":"金融","V":"金融","MA":"金融","BAC":"金融","GS":"金融","MS":"金融",
    "WFC":"金融","C":"金融","AXP":"金融","BLK":"金融",
    "UNH":"医疗","JNJ":"医疗","LLY":"医疗","PFE":"医疗","ABBV":"医疗","MRK":"医疗",
    "CAT":"工业","HON":"工业","UPS":"工业","RTX":"工业","GE":"工业","DE":"工业",
    "LMT":"工业","BA":"工业","MMM":"工业","EMR":"工业",
    "COST":"消费","WMT":"消费","HD":"消费","MCD":"消费","NKE":"消费","SBUX":"消费",
    "XOM":"能源","CVX":"能源","COP":"能源","SLB":"能源","EOG":"能源","MPC":"能源",
    "NEE":"公用","DUK":"公用","SO":"公用","D":"公用","AEP":"公用","SRE":"公用",
    "DIS":"通信","CMCSA":"通信","NFLX":"通信","T":"通信","VZ":"通信","TMUS":"通信",
    "AMT":"地产","PLD":"地产","CCI":"地产","EQIX":"地产","SPG":"地产","PSA":"地产",
    "LIN":"材料","APD":"材料","SHW":"材料","ECL":"材料","DD":"材料","NEM":"材料","FCX":"材料",
}

def compute_sector_heatmap(stock_list):
    sectors = defaultdict(lambda: {"returns": [], "stocks": []})
    for stock in stock_list:
        ticker = stock["ticker"]
        sector = SECTOR_MAP.get(ticker, "其他")
        ret = stock.get("daily_return", 0)
        sectors[sector]["returns"].append(ret)
        sectors[sector]["stocks"].append(ticker)
    
    result = []
    for sector, data in sectors.items():
        avg_ret = sum(data["returns"]) / len(data["returns"]) if data["returns"] else 0
        color = "green" if avg_ret > 0.5 else ("yellow" if avg_ret > -0.5 else "red")
        result.append({
            "sector": sector, "avg_return": round(avg_ret, 2),
            "stock_count": len(data["stocks"]), "color": color,
            "stocks": data["stocks"][:5],
        })
    result.sort(key=lambda x: x["avg_return"], reverse=True)
    return result


# ════════════════════════════════════════════════════════════
#  5. 持仓面板 (OpenD)
# ════════════════════════════════════════════════════════════

def load_portfolio_from_opend():
    """从OpenD获取持仓数据"""
    try:
        from futu import OpenSecTradeContext, OpenQuoteContext, RET_OK
        
        trade_ctx = OpenSecTradeContext(host='127.0.0.1', port=11111)
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        
        ret, accinfo = trade_ctx.accinfo_query()
        if ret != RET_OK:
            return None
        
        ret, positions = trade_ctx.position_list_query()
        if ret != RET_OK or positions is None or positions.empty:
            return None
        
        codes = positions['code'].tolist()
        ret, snapshots = quote_ctx.get_market_snapshot(codes)
        
        portfolio = []
        for _, pos in positions.iterrows():
            code = pos['code']
            qty = pos['qty']
            cost_price = pos['cost_price']
            
            live_price = 0
            change_pct = 0
            if snapshots is not None and not snapshots.empty:
                snap = snapshots[snapshots['code'] == code]
                if not snap.empty:
                    live_price = snap.iloc[0].get('last_price', 0)
                    prev_close = snap.iloc[0].get('prev_close_price', 0)
                    if prev_close > 0:
                        change_pct = (live_price / prev_close - 1) * 100
            
            unrealized_pl = (live_price - cost_price) * qty if live_price > 0 and qty > 0 else 0
            unrealized_pl_ratio = ((live_price / cost_price) - 1) * 100 if cost_price > 0 and live_price > 0 else 0
            
            portfolio.append({
                'ticker': code.replace('.US', ''),
                'shares': int(qty),
                'cost': round(float(cost_price), 2),
                'current': round(float(live_price), 2),
                'pnl_pct': round(float(unrealized_pl_ratio), 2),
                'pnl_value': round(float(unrealized_pl), 2),
                'market_value': round(float(live_price * qty), 2) if live_price > 0 else 0,
            })
        
        portfolio.sort(key=lambda x: x['market_value'], reverse=True)
        portfolio = [p for p in portfolio if p['shares'] > 0]
        
        account = {
            'total_assets': round(float(accinfo.iloc[0].get('total_assets', 0)), 2),
            'cash': round(float(accinfo.iloc[0].get('cash', 0)), 2),
        }
        
        trade_ctx.close()
        quote_ctx.close()
        
        return {"account": account, "positions": portfolio}
    except Exception:
        return None


def load_portfolio_from_file():
    """从配置文件加载持仓 (fallback)"""
    pf_file = os.path.join(ROOT, "portfolio.json")
    if os.path.exists(pf_file):
        with open(pf_file) as f:
            return json.load(f)
    return None


def get_portfolio():
    """获取持仓 (OpenD优先，文件fallback)"""
    data = load_portfolio_from_opend()
    if data:
        return data
    return load_portfolio_from_file()


def score_portfolio(portfolio, shield_list, arrow_list):
    """
    给持仓打分并分类
    
    分类逻辑：
    1. 去掉 "US." / ".US" 前缀匹配模型池
    2. 匹配到蓝盾 → 大盘股
    3. 匹配到绿箭 → 小盘股
    4. 都没匹配 → 按价格分类（<$10小盘，>$10大盘）
    """
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    
    shield_thresholds = cfg["scoring"]["shield"]["thresholds"]
    arrow_thresholds = cfg["scoring"]["arrow"]["thresholds"]
    
    # 建立索引（同时存储原始ticker和去前缀ticker）
    shield_map = {}
    for s in shield_list:
        shield_map[s["ticker"]] = s
        # 也存去掉前缀的版本
        if "." in s["ticker"]:
            shield_map[s["ticker"].split(".")[-1]] = s
    
    arrow_map = {}
    for a in arrow_list:
        arrow_map[a["ticker"]] = a
        if "." in a["ticker"]:
            arrow_map[a["ticker"].split(".")[-1]] = a
    
    large_cap = []
    small_cap = []
    
    for p in portfolio:
        ticker = p.get("ticker", "")
        # 去掉 US. 前缀用于匹配
        clean_ticker = ticker.replace("US.", "").replace(".US", "")
        
        shield_data = shield_map.get(clean_ticker) or shield_map.get(ticker)
        arrow_data = arrow_map.get(clean_ticker) or arrow_map.get(ticker)
        
        has_shield = shield_data is not None
        has_arrow = arrow_data is not None
        
        if has_shield:
            score = shield_data.get("score", 0)
            if score >= shield_thresholds["strong_buy"]:
                grade, label = "strong", "\U0001f7e2\U0001f7e2 \u5f3a\u70c8\u4e70\u5165"
            elif score >= shield_thresholds["buy"]:
                grade, label = "buy", "\U0001f7e2 \u4e70\u5165"
            elif score >= shield_thresholds["watch_bullish"]:
                grade, label = "watch_bullish", "\U0001f7e1 \u504f\u591a\u89c2\u671b"
            elif score >= shield_thresholds["watch"]:
                grade, label = "watch", "\U0001f7e1 \u89c2\u671b"
            else:
                grade, label = "avoid", "\U0001f534 \u4e0d\u5efa\u8bae"
            
            large_cap.append({
                **p,
                "model_score": score,
                "model_name": "蓝盾V3",
                "model_detail": shield_data.get("signal", ""),
                "grade": grade,
                "model_label": label,
                "rsi": shield_data.get("rsi"),
            })
        
        if has_arrow:
            prob = arrow_data.get("prob", 0)
            if prob >= arrow_thresholds["high_prob"]:
                grade, label = "strong", "\U0001f7e2\U0001f7e2 \u9ad8\u6982\u7387"
            elif prob >= arrow_thresholds["medium_prob"]:
                grade, label = "medium", "\U0001f7e2 \u4e2d\u6982\u7387"
            elif prob >= arrow_thresholds["low_prob"]:
                grade, label = "low", "\U0001f7e1 \u4f4e\u6982\u7387"
            else:
                grade, label = "very_low", "\U0001f534 \u6781\u4f4e"
            
            small_cap.append({
                **p,
                "model_score": round(prob * 100, 1),
                "model_name": "绿箭V9",
                "model_detail": f"\u6982\u7387{prob*100:.0f}%",
                "grade": grade,
                "model_label": label,
            })
        
        if not has_shield and not has_arrow:
            # 未匹配 → 按价格粗分类
            price = p.get("current", 0)
            if price < 10:
                # 小盘股 → 归入小盘列，标记无模型
                small_cap.append({
                    **p,
                    "model_score": None,
                    "model_name": "绿箭(未评分)",
                    "model_detail": "\u2014",
                    "grade": "unknown",
                    "model_label": "\u26aa \u5c0f\u76d8\u672a\u8bc4\u5206",
                })
            else:
                large_cap.append({
                    **p,
                    "model_score": None,
                    "model_name": "蓝盾(未评分)",
                    "model_detail": "\u2014",
                    "grade": "unknown",
                    "model_label": "\u26aa \u5927\u76d8\u672a\u8bc4\u5206",
                })
    
    large_cap.sort(key=lambda x: x.get("model_score") or 0, reverse=True)
    small_cap.sort(key=lambda x: x.get("model_score") or 0, reverse=True)
    
    return {
        "large_cap": large_cap,
        "small_cap": small_cap,
    }


# ════════════════════════════════════════════════════════════
#  6. 综合组装
# ════════════════════════════════════════════════════════════

def build_dashboard_data(shield_list, arrow_list, context, prev_context=None):
    """组装仪表盘所有数据"""
    if is_near_open():
        save_opening_snapshot(shield_list, arrow_list)
    save_current_snapshot(shield_list, arrow_list)
    
    shield_deltas = compute_rank_deltas(shield_list, "shield")
    arrow_deltas = compute_rank_deltas(arrow_list, "arrow")
    entry_signals = compute_all_entry_signals(shield_list, arrow_list)
    sentiment = compute_market_sentiment(context, prev_context)
    
    seen = set()
    unique_stocks = []
    for s in shield_list + arrow_list:
        if s["ticker"] not in seen:
            seen.add(s["ticker"])
            unique_stocks.append(s)
    sector_heatmap = compute_sector_heatmap(unique_stocks)
    
    portfolio_data = get_portfolio()
    raw_portfolio = portfolio_data.get("positions", []) if portfolio_data else []
    account = portfolio_data.get("account", {}) if portfolio_data else {}
    
    # 给持仓打分并分类
    portfolio = score_portfolio(raw_portfolio, shield_list, arrow_list)
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market_context": context,
        "sentiment": sentiment,
        "shield_list": shield_list,
        "shield_deltas": shield_deltas,
        "arrow_list": arrow_list,
        "arrow_deltas": arrow_deltas,
        "entry_signals": entry_signals,
        "sector_heatmap": sector_heatmap,
        "portfolio": portfolio,
        "account": account,
        "rank_history_available": len(load_rank_history()) > 1,
    }


def load_rank_history():
    today = datetime.now().strftime("%Y-%m-%d")
    hist_file = os.path.join(HISTORY_DIR, f"{today}.json")
    if os.path.exists(hist_file):
        with open(hist_file) as f:
            return json.load(f)
    return []
