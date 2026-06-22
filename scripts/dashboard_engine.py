#!/usr/bin/env python3
"""
Hermes Trading Dashboard Engine v3
重构：清晰分层 + V4模型集成 + 系统健康监控

布局逻辑：
┌──────────────────────────────────────────────────────────────┐
│  MARKET CONTEXT (市场环境)                                    │
├─────────────────────────┬────────────────────────────────────┤
│  🔵 蓝盾V4 (大盘股)      │  🟢 绿箭V9 (小盘彩票)              │
├─────────────────────────┴────────────────────────────────────┤
│  💼 PORTFOLIO (持仓状态)                                      │
├──────────────────────────────────────────────────────────────┤
│  📊 MODEL HEALTH (模型健康)                                   │
├──────────────────────────────────────────────────────────────┤
│  🎯 ACTIONS + ⚠️ RISKS (操作建议 + 风险)                     │
└──────────────────────────────────────────────────────────────┘
"""

import json, os, sys, time, subprocess
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
STATE_DIR = os.path.join(OUTPUT_DIR, "state")
CONFIG_PATH = os.path.join(ROOT, "config.json")
MODEL_DIR = os.path.join(ROOT, "models", "us")

os.makedirs(STATE_DIR, exist_ok=True)

with open(CONFIG_PATH) as f:
    CFG = json.load(f)


# ════════════════════════════════════════════════════════════
#  1. 市场环境数据
# ════════════════════════════════════════════════════════════

def get_market_context():
    """获取市场指数数据"""
    import yfinance as yf
    
    indices = CFG["market"]["indices"]
    result = {}
    
    for name, ticker in indices.items():
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if len(hist) >= 2:
                cur = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                pct = (cur - prev) / prev * 100
                result[name] = {
                    "price": round(cur, 2),
                    "change_pct": round(pct, 2),
                    "emoji": "🟢" if pct >= 0 else "🔴"
                }
        except:
            result[name] = {"price": 0, "change_pct": 0, "emoji": "⚪"}
    
    return result


# ════════════════════════════════════════════════════════════
#  2. 模型评分数据
# ════════════════════════════════════════════════════════════

def get_shield_scores():
    """获取蓝盾V4评分"""
    score_file = os.path.join(OUTPUT_DIR, "shield_scores.json")
    if not os.path.exists(score_file):
        return []
    
    with open(score_file) as f:
        data = json.load(f)
    
    stocks = data.get("stocks", [])
    # 按分数排序
    stocks.sort(key=lambda x: x.get("score", 0), reverse=True)
    return stocks[:CFG["scoring"]["shield"]["top_n"]]


def get_arrow_scores():
    """获取绿箭V9评分"""
    score_file = os.path.join(OUTPUT_DIR, "arrow_scores.json")
    if not os.path.exists(score_file):
        return []
    
    with open(score_file) as f:
        data = json.load(f)
    
    stocks = data.get("stocks", [])
    stocks.sort(key=lambda x: x.get("probability", 0), reverse=True)
    return stocks[:CFG["scoring"]["arrow"]["top_n"]]


# ════════════════════════════════════════════════════════════
#  3. 持仓数据
# ════════════════════════════════════════════════════════════

def get_portfolio():
    """获取持仓数据（从OpenD实时同步）"""
    # 先从OpenD同步最新持仓到portfolio.json
    try:
        sync_script = os.path.join(ROOT, "scripts", "sync_portfolio_from_opend.py")
        result = subprocess.run(
            [sys.executable, sync_script],
            capture_output=True, text=True, timeout=15,
            cwd=ROOT
        )
        if result.returncode == 0:
            print(f"[engine] 持仓同步成功")
        else:
            print(f"[engine] 持仓同步失败: {result.stderr[:200]}")
    except Exception as e:
        print(f"[engine] 持仓同步异常: {e}")
    
    portfolio_file = os.path.join(STATE_DIR, "portfolio.json")
    if not os.path.exists(portfolio_file):
        return {"large_cap": [], "small_cap": []}
    
    with open(portfolio_file) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════
#  4. 模型健康监控
# ════════════════════════════════════════════════════════════

def get_model_health():
    """获取模型健康状态"""
    health = {
        "shield": {
            "wf_sharpe": 1.203,
            "wf_annual": 0.345,
            "wf_max_dd": -0.804,
            "features": 16,
            "model_type": "XGBoost CUDA",
            "last_train": "2026-06-19"
        },
        "arrow": {
            "wf_sharpe": 0.85,
            "wf_annual": 0.42,
            "wf_max_dd": -0.45,
            "features": 50,
            "model_type": "XGBoost",
            "last_train": "2026-06-18"
        }
    }
    
    # 检查模型文件是否存在
    shield_model = os.path.join(MODEL_DIR, "blueshield_v4_cs16_xgb.json")
    arrow_model = os.path.join(MODEL_DIR, "us_v9_lottery.json")
    
    health["shield"]["model_exists"] = os.path.exists(shield_model)
    health["arrow"]["model_exists"] = os.path.exists(arrow_model)
    
    return health


def get_token_cost():
    """获取token费用"""
    try:
        result = subprocess.run(
            ["tokscale"],
            capture_output=True, text=True, timeout=10
        )
        # 解析输出
        for line in result.stdout.split("\n"):
            if "Total:" in line and "$" in line:
                # 格式: "Total: 2,379 messages, 331,930,570 tokens, $28.82"
                parts = line.split("$")
                if len(parts) >= 2:
                    cost_str = parts[-1].strip()
                    return {"total_usd": float(cost_str)}
    except:
        pass
    
    return {"total_usd": 0}


# ════════════════════════════════════════════════════════════
#  5. 排名追踪
# ════════════════════════════════════════════════════════════

def compute_rank_deltas(current_list, model_type="shield"):
    """计算排名变化"""
    opening_file = os.path.join(STATE_DIR, "opening_ranks.json")
    
    if not os.path.exists(opening_file):
        return {}
    
    with open(opening_file) as f:
        opening = json.load(f)
    
    rank_key = f"{model_type}_ranks"
    open_ranks = opening.get(rank_key, {})
    
    deltas = {}
    for i, item in enumerate(current_list):
        ticker = item.get("ticker", item.get("code", ""))
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
#  6. 生成完整数据
# ════════════════════════════════════════════════════════════

def generate_dashboard_data():
    """生成看板所有数据"""
    t0 = time.time()
    
    # 并行获取数据
    market = get_market_context()
    shield = get_shield_scores()
    arrow = get_arrow_scores()
    portfolio = get_portfolio()
    health = get_model_health()
    token = get_token_cost()
    
    # 排名变化
    shield_deltas = compute_rank_deltas(shield, "shield")
    arrow_deltas = compute_rank_deltas(arrow, "arrow")
    
    # 入场信号
    entry_signals = compute_entry_signals(shield, arrow, shield_deltas, arrow_deltas)
    
    # 风险提示
    risks = compute_risks(shield, arrow, portfolio, market)
    
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": market,
        "shield_list": shield,
        "shield_deltas": shield_deltas,
        "arrow_list": arrow,
        "arrow_deltas": arrow_deltas,
        "portfolio": portfolio,
        "health": health,
        "token": token,
        "entry_signals": entry_signals,
        "risks": risks,
        "compute_time": round(time.time() - t0, 2)
    }
    
    # 保存
    output_file = os.path.join(OUTPUT_DIR, "live_data.json")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data


def compute_entry_signals(shield, arrow, shield_deltas, arrow_deltas):
    """计算入场信号"""
    signals = []
    
    # 蓝盾信号
    for i, stock in enumerate(shield):
        ticker = stock.get("ticker", stock.get("code", ""))
        score = stock.get("score", 0)
        delta = shield_deltas.get(ticker, {})
        
        if score >= 85:
            strength = "strong" if score >= 90 else "entry"
            signals.append({
                "ticker": ticker,
                "model": "shield",
                "strength": strength,
                "score": score,
                "reason": f"蓝盾{score:.0f}分·趋势确认",
                "rank_change": delta.get("delta", 0)
            })
    
    # 绿箭信号
    for stock in arrow:
        ticker = stock.get("ticker", stock.get("code", ""))
        prob = stock.get("probability", 0)
        
        if prob >= 0.6:
            strength = "strong" if prob >= 0.8 else "entry"
            signals.append({
                "ticker": ticker,
                "model": "arrow",
                "strength": strength,
                "score": prob,
                "reason": f"绿箭{prob:.0%}概率·量化信号"
            })
    
    # 按强度排序
    strength_order = {"strong": 0, "entry": 1, "watch": 2, "none": 3}
    signals.sort(key=lambda x: strength_order.get(x["strength"], 99))
    
    return signals[:15]


def compute_risks(shield, arrow, portfolio, market):
    """计算风险提示"""
    risks = []
    
    # VIX检查
    vix = market.get("VIX", {})
    if vix.get("price", 0) > 25:
        risks.append({
            "type": "vix_high",
            "level": "high",
            "message": f"VIX {vix['price']:.1f} > 25 · 市场恐慌"
        })
    elif vix.get("price", 0) > 20:
        risks.append({
            "type": "vix_elevated",
            "level": "medium",
            "message": f"VIX {vix['price']:.1f} > 20 · 波动升高"
        })
    
    # 板块集中度
    if len(shield) >= 5:
        # 简化检查：看是否有同一sector的股票过多
        pass
    
    # 绿箭信号强度
    if arrow and arrow[0].get("probability", 0) < 0.3:
        risks.append({
            "type": "arrow_weak",
            "level": "medium",
            "message": "绿箭最高概率<30% · 信号偏弱"
        })
    
    return risks


if __name__ == "__main__":
    data = generate_dashboard_data()
    print(f"Dashboard data generated in {data['compute_time']}s")
    print(f"Shield: {len(data['shield_list'])} stocks")
    print(f"Arrow: {len(data['arrow_list'])} stocks")
    print(f"Entry signals: {len(data['entry_signals'])}")
    print(f"Risks: {len(data['risks'])}")
