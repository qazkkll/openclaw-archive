#!/usr/bin/env python3
"""
🦅 Falcon Alert Analyzer — 异动深度分析引擎 (v2)
=================================================
设计原则:
  - 决策权归模型(Falcon score + 止损/到期规则)
  - 新闻是给Andy的上下文参考，不改变交易决策
  - 触发条件: 价格/量异动(数字驱动)，不是新闻

L2/L3触发后:
  1. 拉FMP最新新闻 → 告诉Andy"发生了什么"
  2. FinBERT打分 → 告诉Andy"新闻偏正面还是负面"
  3. 模型推荐 → 基于价格+盈亏+持仓天数的硬规则
  4. 综合报告 = 模型推荐 + 新闻上下文(新闻不改变推荐)

用法:
    from falcon_alert_analyzer import analyze_ticker
    result = analyze_ticker("NVDA", alert_type="price_move", current_price=191.0)
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

# ── Paths ──
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
ANALYSIS_DIR = DATA_DIR / "analysis"
ENV_PATH = PROJECT_ROOT / ".env"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load .env ──
from dotenv import load_dotenv
load_dotenv(ENV_PATH)

# ── 模型硬规则(与falcon_trade_exec.py对齐) ──
STOP_LOSS = -0.15      # -15% 硬止损
HOLD_DAYS = 30          # 持有天数
PNL_WARN = -0.10        # -10% 预警线

# ── FinBERT (lazy load, cached in memory) ──
_finbert_model = None
_finbert_tokenizer = None

def _load_finbert():
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return _finbert_tokenizer, _finbert_model

def finbert_score(text: str) -> Dict[str, float]:
    """FinBERT情感打分。返回 {positive, negative, neutral, label, confidence}。"""
    import torch
    tok, model = _load_finbert()
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    probs = torch.softmax(out.logits, dim=1)[0]
    labels = ["positive", "negative", "neutral"]
    scores = {l: round(float(p), 4) for l, p in zip(labels, probs)}
    best_idx = probs.argmax().item()
    scores["label"] = labels[best_idx]
    scores["confidence"] = round(float(probs[best_idx]), 4)
    return scores


def fetch_fmp_news(ticker: str, limit: int = 5) -> List[Dict]:
    """从FMP拉取最新新闻。返回最近N条。"""
    import urllib.request
    key = os.environ.get("FMP_API_KEY", "")
    if not key:
        return []

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    url = (
        f"https://financialmodelingprep.com/stable/news/stock?"
        f"symbols={ticker}&from={start}&to={end}&limit={limit}&apikey={key}"
    )
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        if not isinstance(data, list):
            return []
        articles = []
        for a in data[:limit]:
            title = a.get("title", "")
            text = a.get("text", "")
            if title:
                articles.append({
                    "title": title,
                    "text": text[:500] if text else title,
                    "published_at": a.get("publishedDate", ""),
                    "publisher": a.get("publisher", ""),
                })
        return articles
    except Exception:
        return []


def analyze_ticker(
    ticker: str,
    alert_type: str,
    current_price: float,
    entry_price: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    score: Optional[float] = None,
    hold_days: Optional[int] = None,
    alert_level: str = "L2",
) -> Dict[str, Any]:
    """对单只ticker做深度分析。

    决策逻辑(模型规则，不依赖新闻):
      - pnl ≤ -15% → 止损
      - pnl ≤ -10% → 预警(减仓建议)
      - hold_days ≥ 30 → 到期卖出
      - 其他 → 持有

    新闻仅作为上下文:
      - 告诉Andy发生了什么
      - 不改变模型推荐
    """
    result = {
        "ticker": ticker,
        "alert_type": alert_type,
        "alert_level": alert_level,
        "current_price": current_price,
        "entry_price": entry_price,
        "pnl_pct": pnl_pct,
        "score": score,
        "hold_days": hold_days,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ═══════════════════════════════════════════════
    # 1. 模型决策(纯规则，不看新闻)
    # ═══════════════════════════════════════════════
    model_rec, model_reasoning = _model_recommendation(
        pnl_pct=pnl_pct,
        hold_days=hold_days,
        alert_type=alert_type,
    )
    result["model_recommendation"] = model_rec
    result["model_reasoning"] = model_reasoning

    # ═══════════════════════════════════════════════
    # 2. 新闻上下文(只提供信息，不改变决策)
    # ═══════════════════════════════════════════════
    news = fetch_fmp_news(ticker, limit=5)
    result["news"] = news

    if news:
        # 合并标题做整体情感
        combined_text = " | ".join(n["title"] for n in news[:3])
        sentiment = finbert_score(combined_text)
        result["sentiment"] = sentiment

        # 每条新闻单独打分
        for n in news[:3]:
            n["sentiment"] = finbert_score(n["title"])
    else:
        result["sentiment"] = {"label": "unknown", "confidence": 0, "positive": 0, "negative": 0, "neutral": 0}

    # ═══════════════════════════════════════════════
    # 3. 综合报告 = 模型推荐 + 新闻上下文
    # ═══════════════════════════════════════════════
    result["recommendation"] = model_rec  # 推荐始终=模型推荐
    result["reasoning"] = model_reasoning
    result["news_context"] = _news_summary(result["sentiment"], news)

    # 存档
    _archive_analysis(result)

    return result


def _model_recommendation(
    pnl_pct: Optional[float],
    hold_days: Optional[int],
    alert_type: str,
) -> tuple:
    """模型决策: 纯规则，不看新闻。返回 (recommendation, reasoning)。"""

    # 硬止损
    if pnl_pct is not None and pnl_pct <= STOP_LOSS * 100:
        return (
            "stop_loss",
            f"亏损{pnl_pct:+.1f}%触及{STOP_LOSS*100:.0f}%止损线，纪律止损。"
        )

    # 预警线
    if pnl_pct is not None and pnl_pct <= PNL_WARN * 100:
        return (
            "reduce",
            f"亏损{pnl_pct:+.1f}%超过{PNL_WARN*100:.0f}%预警线，建议减仓50%降低风险。"
        )

    # 到期
    if hold_days is not None and hold_days >= HOLD_DAYS:
        pnl_str = f"盈亏{pnl_pct:+.1f}%" if pnl_pct is not None else ""
        return (
            "expire",
            f"持有{hold_days}天达到{HOLD_DAYS}天持有期，{pnl_str}到期卖出。"
        )

    # 价格异动但持仓正常
    if alert_type == "price_move":
        if pnl_pct is not None and pnl_pct > 5:
            return ("hold", f"价格异动但持仓盈利{pnl_pct:+.1f}%，继续持有。")
        return ("hold", f"价格异动但未触及任何阈值，继续持有。")

    # 放量/跳空
    if alert_type in ("volume_spike", "gap"):
        return ("hold", f"{alert_type}异动，持仓未触及阈值，继续观察。")

    return ("hold", "未触发任何操作条件。")


def _news_summary(sentiment: Dict, news: List[Dict]) -> str:
    """生成新闻摘要(纯信息，不含建议)。"""
    if not news:
        return "无最新新闻。"

    sent_label = sentiment.get("label", "unknown")
    sent_conf = sentiment.get("confidence", 0)
    sent_emoji = {"positive": "🟢正面", "negative": "🔴负面", "neutral": "⚪中性"}.get(sent_label, "❓未知")

    parts = [f"新闻情感: {sent_emoji}(置信度{sent_conf:.0%})"]
    for n in news[:3]:
        parts.append(f"  • {n['title'][:80]}")

    return "\n".join(parts)


def _archive_analysis(result: Dict):
    """存档分析结果。"""
    today = datetime.now().strftime("%Y%m%d")
    archive_file = ANALYSIS_DIR / f"analysis_{today}.json"

    existing = []
    if archive_file.exists():
        try:
            with open(archive_file) as f:
                existing = json.load(f)
        except Exception:
            existing = []

    existing.append(result)

    with open(archive_file, "w") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False, default=str)


def format_analysis_telegram(result: Dict) -> str:
    """格式化分析结果为Telegram消息。"""
    lines = []
    ticker = result["ticker"]
    level = result["alert_level"]
    rec = result.get("recommendation", "hold")

    # 标题
    level_emoji = {"L1": "🟡", "L2": "🟠", "L3": "🔴"}.get(level, "⚪")
    lines.append(f"{level_emoji} **{ticker} 异动分析**")

    # 价格+盈亏
    price = result.get("current_price", 0)
    pnl = result.get("pnl_pct")
    hold = result.get("hold_days")
    pnl_str = f" | 盈亏{pnl:+.1f}%" if pnl is not None else ""
    hold_str = f" | 持{hold}天" if hold is not None else ""
    lines.append(f"💰 ${price:.2f}{pnl_str}{hold_str}")

    # 模型推荐(主要)
    rec_map = {
        "hold": "⏳ 继续持有",
        "reduce": "⚠️ 建议减仓",
        "stop_loss": "🛑 建议止损",
        "expire": "⏰ 到期卖出",
    }
    lines.append(f"\n**{rec_map.get(rec, rec)}**")
    lines.append(f"📝 {result.get('model_reasoning', '')}")

    # 新闻上下文(次要)
    news_ctx = result.get("news_context", "")
    if news_ctx:
        lines.append(f"\n📰 **新闻参考**")
        lines.append(news_ctx)

    lines.append(f"\n📁 已存档 | ⏰ {result['timestamp'][:16]}")

    return "\n".join(lines)


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Falcon Alert Analyzer")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--alert-type", default="price_move")
    parser.add_argument("--price", type=float, required=True)
    parser.add_argument("--entry", type=float, default=None)
    parser.add_argument("--pnl", type=float, default=None)
    parser.add_argument("--hold-days", type=int, default=None)
    parser.add_argument("--level", default="L2", choices=["L2", "L3"])
    args = parser.parse_args()

    result = analyze_ticker(
        ticker=args.ticker,
        alert_type=args.alert_type,
        current_price=args.price,
        entry_price=args.entry,
        pnl_pct=args.pnl,
        hold_days=args.hold_days,
        alert_level=args.level,
    )
    print(format_analysis_telegram(result))
