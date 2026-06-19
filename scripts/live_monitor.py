#!/usr/bin/env python3
"""
Hermes 盯盘监控器 — 蓝盾V5 + 绿箭V11(待接入)
定时刷新数据 → 跑模型 → 更新HTML仪表盘

用法：
    python3 live_monitor.py                    # 单次运行
    python3 live_monitor.py --loop 300         # 每300秒循环
    python3 live_monitor.py --loop 300 --open  # 循环 + 自动打开浏览器

⚠️ V5评分脚本: scripts/us/blueshield_v5_score.py
"""

import json, sys, os, time, argparse, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── 路径 ──
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, SCRIPTS_DIR)

from recommendation_template import generate_report
from dashboard_engine import build_dashboard_data, compute_market_sentiment, load_opening_snapshot, save_opening_snapshot
from dashboard_renderer import generate_new_dashboard

# ── V9特征列表 ──
FEAT_V9 = [
    "ma5","ma5_ratio","ma20_ratio","ma60_ratio","vol5","vol20","vol_ratio",
    "ema12","ema26","macd","macd_signal","macd_hist","rsi14","k","d","j",
    "bb_upper","bb_lower","bb_width","bb_position","vol_ratio_ma5","vol_ratio_ma20",
    "adx","plus_di","minus_di","price_position","price_position_60","cmf",
    "vix_close","close_log","close_x_vol","plus_di_x_low_vol","adx_x_rsi","bb_x_vol","rsi_x_kdj","low_price",
    "price_range_norm","price_accel","oversold","trend_strength","volatility_expansion",
    "pos_60d_channel","kdj_j","bb_squeeze","rsi_trend_5d","ma5_x_ma20_cross",
    "price_vs_vwap","consecutive_up","consecutive_down","reversal_pattern"
]

# ── 美股池（扩展到200只） ──
US_UNIVERSE = [
    # 大盘科技
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AVGO","AMD","INTC",
    # 金融
    "JPM","V","MA","BAC","GS","MS","WFC","C","AXP","BLK",
    # 医疗
    "UNH","JNJ","LLY","PFE","ABBV","MRK","TMO","ABT","DHR","BMY",
    # 工业
    "CAT","HON","UPS","RTX","GE","DE","LMT","BA","MMM","EMR",
    # 消费
    "COST","WMT","HD","MCD","NKE","SBUX","TGT","LOW","TJX","ROST",
    # 能源
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL",
    # 公用
    "NEE","DUK","SO","D","AEP","SRE","EXC","XEL","WEC","ES",
    # 通信
    "DIS","CMCSA","NFLX","T","VZ","TMUS","CHTR","EA","ATVI","TTWO",
    # 地产
    "AMT","PLD","CCI","EQIX","SPG","PSA","O","WELL","DLR","AVB",
    # 材料
    "LIN","APD","SHW","ECL","DD","NEM","FCX","NUE","STLD","CF",
    # 小盘彩票池
    "MAYS","PRAX","JAZZ","HALO","ITCI","AXSM","BGNE","OCC","PCSA",
    "SBNY","FRC","PACW","WAL","EWBC","ZION","KEY","CFG","RF","HBAN",
    "DAL","AAL","UAL","LUV","ALK","JBL","ERIC","NOK","SONY","NTDOY",
    "PLTR","SOFI","HOOD","RIVN","LCID","NIO","XPEV","LI","BYDDY","BABA",
    "JD","PDD","BIDU","NIO","MNMD","CYBN","CMPS","ATAI","RENN","VNET",
    "TIGR","FUTU","YMM","DADA","KC","QFIN","FINV","LX","GDS","VNET",
]


def get_market_context():
    """获取市场概览"""
    import yfinance as yf
    try:
        sp = yf.Ticker("^GSPC").fast_info
        vix = yf.Ticker("^VIX").fast_info
        dxy = yf.Ticker("DX-Y.NYB").fast_info
        tnx = yf.Ticker("^TNX").fast_info
        return {
            "S&P 500": f"{sp.last_price:,.0f}",
            "VIX": f"{vix.last_price:.1f}",
            "美元": f"{dxy.last_price:.1f}",
            "10Y": f"{tnx.last_price:.2f}%",
        }
    except Exception as e:
        print(f"⚠️ 市场数据: {e}")
        return {}


def get_prev_market_context():
    """加载上一次的市场数据（用于计算变化）"""
    prev_file = os.path.join(OUTPUT_DIR, "prev_market.json")
    if os.path.exists(prev_file):
        try:
            with open(prev_file) as f:
                return json.load(f)
        except:
            pass
    return None


def save_market_context(context):
    """保存当前市场数据（供下次比较）"""
    prev_file = os.path.join(OUTPUT_DIR, "prev_market.json")
    with open(prev_file, "w") as f:
        json.dump(context, f, indent=2)


def load_v9_data():
    """加载V9历史OHLCV数据"""
    data_file = os.path.join(DATA_DIR, "us_all_ohlcv.json")
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return None
    print(f"📦 加载V9数据 ({os.path.getsize(data_file)/1024/1024:.0f}MB)...")
    with open(data_file) as f:
        return json.load(f)


def load_v9_model():
    """加载V9-Lottery模型（XGBoost格式）"""
    import xgboost as xgb
    model_file = os.path.join(MODELS_DIR, "us", "us_v9_lottery.json")
    if not os.path.exists(model_file):
        print(f"❌ V9模型不存在: {model_file}")
        return None
    print("🟢 加载绿箭V9-Lottery模型...")
    model = xgb.Booster()
    model.load_model(model_file)
    return model


def compute_v9_features(c, h, l, o, v, pos):
    """计算V9的50维特征"""
    import numpy as np
    
    if pos < 100 or pos > len(c):
        return None
    
    c = np.array(c[:pos], dtype=float)
    ct = float(c[-1])
    n5 = min(5, len(c))
    n20 = min(20, len(c))
    n60 = min(60, len(c))
    
    ma5 = float(np.mean(c[-n5:]))
    ma20 = float(np.mean(c[-n20:]))
    ma60 = float(np.mean(c[-n60:]))
    r5 = float(np.std(c[-n5:]))
    r20 = float(np.std(c[-n20:]))
    
    # RSI
    delta = np.diff(c[-(14+1):])
    gain = np.sum(delta[delta > 0])
    loss = abs(np.sum(delta[delta < 0]))
    rsi14 = 100 - 100 / (1 + gain / loss) if loss > 0 else 100
    
    # KDJ
    low14 = float(np.min(c[-14:]))
    high14 = float(np.max(c[-14:]))
    rsv = (ct - low14) / (high14 - low14) * 100 if high14 > low14 else 50
    k = d = j = rsv
    
    # Bollinger
    bb_mid = ma20
    bb_std = r20 if r20 > 0 else 1
    bu = bb_mid + 2 * bb_std
    bl = bb_mid - 2 * bb_std
    bb_w = (bu - bl) / bb_mid if bb_mid > 0 else 0
    
    h_arr = np.array(h[:pos], dtype=float)[-n20:]
    l_arr = np.array(l[:pos], dtype=float)[-n20:]
    low20 = float(np.min(c[-n20:]))
    high20 = float(np.max(c[-n20:]))
    low60 = float(np.min(c[-60:]))
    high60 = float(np.max(c[-60:]))
    
    pp20 = (ct - low20) / (high20 - low20) if high20 > low20 else 0.5
    pp60 = (ct - low60) / (high60 - low60) if high60 > low60 else 0.5
    
    cu = cd = 0
    for i in range(-1, -min(11, len(c)), -1):
        if abs(i) >= 1:
            if c[i] > c[i - 1]:
                cu += 1; cd = 0
            else:
                cd += 1; cu = 0
    
    ret5d = (ct - c[-6]) / c[-6] * 100 if len(c) > 5 and c[-6] != 0 else 0
    ret1d = (ct - c[-2]) / c[-2] * 100 if len(c) > 1 and c[-2] != 0 else 0
    rev = 1 if ret5d < -10 and ret1d > 0 else 0
    gp = 0
    if o is not None and len(o) >= 2:
        pc = float(c[-2])
        to = float(o[-1])
        gp = (to - pc) / pc * 100 if pc > 0 else 0
    
    m5p = float(np.mean(c[-10:-5])) if len(c) >= 10 else ma5
    m20p = float(np.mean(c[-25:-20])) if len(c) >= 25 else ma20
    
    vl5 = 0; vl20 = 0; vlr = 1.0
    if v is not None and len(v) >= 20:
        va = np.array(v[-20:], dtype=float)
        va = va[~np.isnan(va)]
        if len(va) >= 10:
            vl5 = float(np.mean(va[-5:]))
            vl20 = float(np.mean(va))
            vlr = vl5 / vl20 if vl20 > 0 else 1.0
    
    f = [0.0] * 50
    f[0] = ma5
    f[1] = ct / ma5 if ma5 > 0 else 1
    f[2] = ct / ma20 if ma20 > 0 else 1
    f[3] = ct / ma60 if ma60 > 0 else 1
    f[4] = r5 if not np.isnan(r5) else 0
    f[5] = r20 if not np.isnan(r20) else 0
    f[6] = f[4] / f[5] if f[5] > 0 else 1
    f[7] = ct; f[8] = ct
    f[12] = rsi14; f[13] = k; f[14] = d; f[15] = j
    f[16] = bu; f[17] = bl; f[18] = bb_w; f[19] = pp20
    f[20] = vlr; f[21] = vlr
    f[25] = pp20; f[26] = pp60
    f[29] = float(np.log(ct)) if ct > 0 else 0
    f[30] = ct * f[4]
    f[31] = float(gp / 10)
    f[34] = float(rsi14 / 100)
    f[35] = 1 if ct < 10 else 0
    f[36] = f[4] / ct if ct > 0 else 0
    acc = ((ma5 - m20p) - (m5p - m20p)) / ma20 if ma20 > 0 else 0
    f[37] = float(acc)
    f[38] = 1 if (rsi14 < 30 and ct < bb_mid) else 0
    f[39] = (ma5 - ma20) / ma20 if ma20 > 0 else 0
    f[40] = f[4] / f[5] if f[5] > 0 else 1
    f[41] = pp60
    f[42] = j
    bh = float(np.std(c[-60:] / np.mean(c[-60:]))) if len(c) >= 60 and np.mean(c[-60:]) > 0 else 0.1
    f[43] = 1 if bb_w < bh * 0.8 else 0
    rs5 = 100 - 100 / (1 + gain / loss) if loss > 0 else rsi14
    f[44] = rsi14 - rs5
    cr = 1 if (ma5 > ma20 and m5p <= m20p) else (-1 if (ma5 < ma20 and m5p >= m20p) else 0)
    f[45] = float(cr)
    f[46] = ct / float(np.mean(v[-5:])) if v is not None and len(v) >= 5 and np.mean(v[-5:]) > 0 else 0
    f[47] = float(cu)
    f[48] = float(cd)
    f[49] = float(rev)
    
    # 补充缺失特征
    f[9] = f[7]  # ema12 placeholder
    f[10] = f[7]  # ema26 placeholder
    f[11] = f[9] - f[10]  # macd
    f[22] = 25  # adx placeholder
    f[23] = 50  # plus_di
    f[24] = 50  # minus_di
    f[27] = 0   # cmf
    f[28] = 18  # vix_close placeholder
    f[32] = 0   # plus_di_x_low_vol
    f[33] = 0   # adx_x_rsi
    
    return f


def score_v9_lottery(v9_data, model, tickers=None):
    """绿箭V9-Lottery评分 — 使用V9数据池"""
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    
    pool = tickers if tickers else list(v9_data.keys())
    
    scored = []
    for ticker in pool:
        if ticker not in v9_data:
            continue
        
        stock = v9_data[ticker]
        c = stock.get("c", stock.get("close", []))
        h = stock.get("h", stock.get("high", []))
        l = stock.get("l", stock.get("low", []))
        o = stock.get("o", stock.get("open", []))
        v = stock.get("v", stock.get("volume", []))
        
        if len(c) < 100:
            continue
        
        price = float(c[-1])
        if price <= 0:
            continue
        
        feat = compute_v9_features(c, h, l, o, v, len(c))
        if feat is None:
            continue
        
        try:
            X = xgb.DMatrix(pd.DataFrame([feat], columns=FEAT_V9))
            prob = float(model.predict(X)[0])
        except Exception:
            continue
        
        daily_ret = (c[-1] / c[-2] - 1) * 100 if len(c) > 1 and c[-2] != 0 else 0
        
        scored.append({
            "ticker": ticker,
            "prob": round(prob, 4),
            "price": round(price, 2),
            "daily_return": round(daily_ret, 2),
            "sector": "Penny" if price < 1 else "Micro" if price < 3 else "Small" if price < 5 else "MidLo" if price < 8 else "MidHi",
        })
    
    for s in scored:
        s["ratio_bp"] = round(s["prob"] / s["price"], 4) if s["price"] > 0 else 0
    
    scored.sort(key=lambda x: x["ratio_bp"], reverse=True)
    pool_30 = scored[:30]
    top10 = sorted(pool_30, key=lambda x: x["prob"], reverse=True)[:10]
    
    return top10, scored


def score_shield_v3(tickers):
    """蓝盾V3评分 — 更精细的评分，避免满分泛滥"""
    import yfinance as yf
    import numpy as np
    
    results = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = info.get("lastPrice", 0)
            prev_close = info.get("previousClose", 1)
            
            if price <= 0:
                continue
            
            hist = t.history(period="60d")
            if hist.empty or len(hist) < 20:
                continue
            
            close = hist["Close"].values
            volume = hist["Volume"].values
            
            # RSI
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rs = avg_gain / max(avg_loss, 0.001)
            rsi = 100 - (100 / (1 + rs))
            
            # 52周
            high52 = max(close[-252:]) if len(close) >= 252 else max(close)
            low52 = min(close[-252:]) if len(close) >= 252 else min(close)
            w52 = (price - low52) / max(high52 - low52, 0.01) * 100
            
            # 均线
            ma5 = np.mean(close[-5:])
            ma20 = np.mean(close[-20:])
            ma60 = np.mean(close[-60:]) if len(close) >= 60 else ma20
            
            # 量比
            vol_recent = np.mean(volume[-5:]) if len(volume) >= 5 else 0
            vol_avg = np.mean(volume[-20:]) if len(volume) >= 20 else vol_recent
            vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1.0
            
            # ── 评分 (满分100，但条件更严格) ──
            score = 0
            signals = []
            trend_score = 0
            
            # 趋势 (30分) — 需要更强的趋势
            trend = 0
            if price > ma5: trend += 1
            if price > ma20: trend += 1
            if price > ma60: trend += 1
            if ma5 > ma20: trend += 1
            if ma20 > ma60: trend += 1
            trend_score = trend
            
            if trend >= 5:
                score += 30; signals.append("强趋势5/5")
            elif trend >= 4:
                score += 24; signals.append("趋势4/5")
            elif trend >= 3:
                score += 18; signals.append("趋势3/5")
            elif trend >= 2:
                score += 10; signals.append("趋势偏弱")
            else:
                score += 5; signals.append("趋势弱")
            
            # RSI (20分) — 更精细
            if 45 <= rsi <= 55:
                score += 20; signals.append("RSI黄金区")
            elif 40 <= rsi <= 60:
                score += 15; signals.append("RSI适中")
            elif 35 <= rsi <= 65:
                score += 10; signals.append("RSI偏高" if rsi > 55 else "RSI偏低")
            elif 30 <= rsi <= 70:
                score += 5; signals.append("RSI边缘")
            else:
                signals.append("RSI超买" if rsi > 70 else "RSI超卖")
            
            # 均线排列 (20分) — 需要完美排列
            if ma5 > ma20 > ma60 and price > ma5:
                score += 20; signals.append("完美多头")
            elif price > ma5 > ma20:
                score += 15; signals.append("站上MA5+MA20")
            elif price > ma20:
                score += 10; signals.append("站上MA20")
            elif price > ma60:
                score += 5; signals.append("站上MA60")
            
            # 52周位置 (15分) — 黄金区间加分
            if 65 <= w52 <= 85:
                score += 15; signals.append("52W黄金位")
            elif 50 <= w52 <= 90:
                score += 10; signals.append("52W适中")
            elif 30 <= w52:
                score += 5; signals.append("52W偏低")
            
            # 量能 (10分) — 放量确认
            if 1.2 <= vol_ratio <= 2.5:
                score += 10; signals.append("量能健康")
            elif 1.0 <= vol_ratio < 1.2:
                score += 5; signals.append("量能平稳")
            elif vol_ratio > 2.5:
                score += 3; signals.append("异常放量")
            
            # 波动率 (5分)
            daily_ret = abs((price / prev_close - 1) * 100) if prev_close > 0 else 0
            if daily_ret < 2:
                score += 5; signals.append("低波动")
            elif daily_ret < 4:
                score += 3; signals.append("波动适中")
            
            # 信号描述
            signal_str = " · ".join(signals[:4])  # 只显示前4个信号
            
            results.append({
                "ticker": ticker,
                "score": min(score, 100),
                "price": round(price, 2),
                "daily_return": round((price / prev_close - 1) * 100 if prev_close > 0 else 0, 2),
                "rsi": round(rsi, 1),
                "week52pct": round(w52, 1),
                "signal": signal_str,
                "sector": "",
                "trend_score": trend_score,
                "vol_ratio": round(vol_ratio, 2),
            })
        except Exception:
            continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:15]

def generate_actions(shield, arrow):
    """生成调仓建议 — 从config读取阈值"""
    with open(os.path.join(ROOT, "config.json")) as f:
        cfg = json.load(f)
    
    shield_thresholds = cfg["scoring"]["shield"]["thresholds"]
    arrow_thresholds = cfg["scoring"]["arrow"]["thresholds"]
    risk_rules = cfg["risk_rules"]
    
    actions = []
    seen = set()
    
    for s in shield[:5]:
        if s["score"] >= shield_thresholds["strong_buy"]:
            actions.append({"ticker": s["ticker"], "action": "ADD", "reason": f"蓝盾{s['score']}分 · 趋势强劲"})
            seen.add(s["ticker"])
        elif s["score"] >= shield_thresholds["buy"]:
            actions.append({"ticker": s["ticker"], "action": "BUY", "reason": f"蓝盾{s['score']}分 · 趋势确认"})
            seen.add(s["ticker"])
    
    for a in arrow[:3]:
        if a["prob"] >= arrow_thresholds["medium_prob"] and a["ticker"] not in seen:
            actions.append({"ticker": a["ticker"], "action": "BUY", "reason": f"绿箭{a['prob']*100:.0f}%概率"})
            seen.add(a["ticker"])
    
    rsi_threshold = next((r["threshold"] for r in risk_rules if r["type"] == "rsi_overbought"), 65)
    for s in shield[:10]:
        if s.get("rsi", 50) > rsi_threshold and s["ticker"] not in seen:
            actions.append({"ticker": s["ticker"], "action": "HOLD", "reason": f"RSI {s['rsi']:.0f}超买"})
            seen.add(s["ticker"])
    
    return actions


def run_once():
    """单次运行"""
    print(f"{'━' * 50}")
    print(f"📊 Hermes 盯盘监控器  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'━' * 50}")
    
    # 1. 市场概览
    print("\n📈 获取市场数据...")
    context = get_market_context()
    prev_context = get_prev_market_context()
    
    # 2. 加载V9数据
    v9_data = load_v9_data()
    if v9_data is None:
        print("❌ V9数据未就绪，等待复制完成")
        return None
    
    # 3. 加载V9模型
    v9_model = load_v9_model()
    if v9_model is None:
        return None
    
    # 4. 绿箭V9-Lottery评分
    print("\n🟢 绿箭V9-Lottery评分...")
    arrow_top10, arrow_all = score_v9_lottery(v9_data, v9_model, US_UNIVERSE)
    print(f"   评分完成：{len(arrow_all)}只候选，Top10选出")
    
    # 5. 蓝盾V3评分
    print("\n🔵 蓝盾V3评分...")
    shield_tickers = list(set([a["ticker"] for a in arrow_top10] + US_UNIVERSE[:50]))
    shield_results = score_shield_v3(shield_tickers)
    print(f"   评分完成：{len(shield_results)}只")
    
    # 6. 调仓建议
    actions = generate_actions(shield_results, arrow_top10)
    
    # 7. 组装旧格式（兼容文本报告）
    scores = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market": "US",
        "shield": shield_results,
        "arrow": arrow_top10,
        "portfolio_actions": actions,
    }
    
    # 8. 保存 JSON
    json_path = os.path.join(OUTPUT_DIR, "live_scores.json")
    with open(json_path, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    
    # 9. 文本报告
    report = generate_report(scores, context)
    print("\n" + report)
    
    report_path = os.path.join(OUTPUT_DIR, "latest_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    
    # 10. 构建仪表盘数据
    print("\n🎯 构建仪表盘数据...")
    dashboard_data = build_dashboard_data(
        shield_results, arrow_top10, context, prev_context
    )
    
    # 保存 live_data.json（供30秒轮询）
    live_data_path = os.path.join(OUTPUT_DIR, "live_data.json")
    with open(live_data_path, "w") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False, default=str)
    
    # 11. 生成 HTML 仪表盘
    print("🎨 生成 HTML 仪表盘...")
    html = generate_new_dashboard(dashboard_data)
    html_path = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(html_path, "w") as f:
        f.write(html)
    
    # 12. 保存当前市场数据（供下次比较）
    save_market_context(context)
    
    print(f"\n✅ 输出:")
    print(f"   JSON: {json_path}")
    print(f"   文本: {report_path}")
    print(f"   HTML: {html_path}")
    print(f"   数据: {live_data_path}")
    
    # 大盘情绪
    sentiment = dashboard_data.get("sentiment", {})
    print(f"   大盘: {sentiment.get('overall_label', '未知')}")
    print(f"   入场信号: {len([e for e in dashboard_data.get('entry_signals', []) if e['signals']['score'] > 0])}只有信号")
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="Hermes 盯盘监控器")
    parser.add_argument("--loop", type=int, default=0, help="循环间隔秒数")
    parser.add_argument("--open", action="store_true", help="自动打开浏览器")
    args = parser.parse_args()
    
    if args.loop > 0:
        print(f"🔄 循环模式：每{args.loop}秒刷新")
        import webbrowser
        first = True
        while True:
            try:
                run_once()
                if args.open and first:
                    webbrowser.open(os.path.join(OUTPUT_DIR, "dashboard.html"))
                    first = False
                print(f"\n⏳ 等待{args.loop}秒...")
                time.sleep(args.loop)
            except KeyboardInterrupt:
                print("\n🛑 已停止")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")
                time.sleep(30)
    else:
        run_once()


if __name__ == "__main__":
    main()
