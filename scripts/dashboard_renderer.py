#!/usr/bin/env python3
"""
Hermes Trading Dashboard Renderer v3
清晰分层 + 深色主题 + 响应式布局

布局：
┌──────────────────────────────────────────────────────────────┐
│  📊 Hermes Trading System              [一键分析] [刷新]     │
├──────────────────────────────────────────────────────────────┤
│  📈 MARKET CONTEXT                                           │
├─────────────────────────┬────────────────────────────────────┤
│  🔵 蓝盾V4 (大盘股)      │  🟢 绿箭V9 (小盘彩票)              │
├─────────────────────────┴────────────────────────────────────┤
│  💼 PORTFOLIO                                                │
├──────────────────────────────────────────────────────────────┤
│  📊 MODEL HEALTH + 💰 TOKEN COST                             │
├──────────────────────────────────────────────────────────────┤
│  🎯 ACTIONS + ⚠️ RISKS                                      │
└──────────────────────────────────────────────────────────────┘
"""

import json, os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")


def _pct(val):
    """格式化百分比"""
    if val > 0: return f"+{val:.1f}%"
    return f"{val:.1f}%"


def _color(val):
    """获取颜色class"""
    if val > 0: return "g"
    if val < 0: return "r"
    return "d"


def _rank_delta_html(d):
    """排名变化HTML"""
    if d is None: return '<span class="rn">NEW</span>'
    trend = d.get("trend", "flat")
    delta = d.get("delta", 0)
    if trend == "new": return '<span class="rn">NEW</span>'
    if delta > 0: return f'<span class="ru">▲{delta}</span>'
    elif delta < 0: return f'<span class="rd">▼{abs(delta)}</span>'
    return '<span class="rf">→</span>'


def _traffic_light(score, thresholds):
    """交通灯信号"""
    if score >= thresholds.get("strong_buy", 95):
        return "🟢🟢", "g", "强烈买入"
    elif score >= thresholds.get("buy", 90):
        return "🟢", "g", "买入"
    elif score >= thresholds.get("watch_bullish", 85):
        return "🟡", "y", "观望"
    elif score >= thresholds.get("watch", 75):
        return "🟡", "y", "观望"
    else:
        return "🔴", "r", "回避"


def _arrow_light(prob, thresholds):
    """绿箭信号灯"""
    if prob >= thresholds.get("high_prob", 0.8):
        return "🟢🟢", "g", "高概率"
    elif prob >= thresholds.get("medium_prob", 0.6):
        return "🟢", "g", "中概率"
    elif prob >= thresholds.get("low_prob", 0.4):
        return "🟡", "y", "低概率"
    else:
        return "🔴", "r", "极低"


def generate_html(data):
    """生成完整HTML"""
    ts = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))
    market = data.get("market", {})
    shield = data.get("shield_list", [])
    shield_deltas = data.get("shield_deltas", {})
    arrow = data.get("arrow_list", [])
    arrow_deltas = data.get("arrow_deltas", {})
    portfolio = data.get("portfolio", {})
    health = data.get("health", {})
    token = data.get("token", {})
    entry_signals = data.get("entry_signals", [])
    risks = data.get("risks", [])
    
    # 从config读取阈值
    with open(os.path.join(ROOT, "config.json")) as f:
        cfg = json.load(f)
    shield_thresh = cfg["scoring"]["shield"]["thresholds"]
    arrow_thresh = cfg["scoring"]["arrow"]["thresholds"]
    
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="300">
<title>Hermes Trading System</title>
<style>
:root {{
    --bg: #0a0a0a; --c1: #111; --c2: #1a1a1a; --bd: #222;
    --tx: #e0e0e0; --dm: #888;
    --gn: #4caf50; --yl: #ffc107; --rd: #f44336; --bl: #2196f3;
    --f: 'SF Mono', 'Consolas', monospace;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--tx); font-family:var(--f); font-size:13px; padding:10px; }}
.header {{ display:flex; justify-content:space-between; align-items:center; padding:10px 15px; background:var(--c1); border-radius:6px; margin-bottom:10px; }}
.header h1 {{ font-size:16px; color:var(--bl); }}
.header .meta {{ color:var(--dm); font-size:11px; }}
.btn {{ background:var(--c2); border:1px solid var(--bd); color:var(--tx); padding:5px 12px; border-radius:4px; cursor:pointer; font-size:11px; }}
.btn:hover {{ background:var(--bd); }}
.section {{ padding:12px; background:var(--c1); border-radius:6px; margin-bottom:10px; border-left:3px solid var(--bd); }}
.section.shield {{ border-left-color:var(--bl); }}
.section.arrow {{ border-left-color:var(--gn); }}
.section.portfolio {{ border-left-color:var(--yl); }}
.section.health {{ border-left-color:var(--bl); }}
.section.actions {{ border-left-color:var(--gn); }}
.section.risks {{ border-left-color:var(--rd); }}
.section h2 {{ font-size:14px; margin-bottom:8px; color:var(--tx); }}
.section h3 {{ font-size:12px; margin-bottom:6px; color:var(--dm); }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
@media(max-width:1200px) {{ .cols {{ grid-template-columns:1fr; }} }}
.market-row {{ display:flex; gap:15px; flex-wrap:wrap; margin-bottom:8px; }}
.market-item {{ display:flex; align-items:center; gap:6px; }}
.market-item .name {{ color:var(--dm); }}
.market-item .price {{ font-weight:bold; }}
.stock-row {{ display:flex; align-items:center; padding:4px 8px; border-bottom:1px solid var(--bd); }}
.stock-row:hover {{ background:var(--c2); }}
.stock-row .rank {{ width:25px; color:var(--dm); }}
.stock-row .ticker {{ width:60px; font-weight:bold; }}
.stock-row .score {{ width:50px; text-align:right; }}
.stock-row .delta {{ width:40px; text-align:center; }}
.stock-row .signal {{ width:30px; text-align:center; }}
.ru {{ color:var(--gn); font-size:11px; }}
.rd {{ color:var(--rd); font-size:11px; }}
.rn {{ color:var(--dm); font-size:10px; }}
.rf {{ color:var(--dm); }}
.health-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:10px; }}
.health-card {{ background:var(--c2); padding:10px; border-radius:4px; }}
.health-card .label {{ color:var(--dm); font-size:11px; }}
.health-card .value {{ font-size:18px; font-weight:bold; margin:4px 0; }}
.health-card .sub {{ color:var(--dm); font-size:10px; }}
.risk-item {{ padding:6px 10px; margin:4px 0; border-radius:4px; background:var(--c2); }}
.risk-item.high {{ border-left:3px solid var(--rd); }}
.risk-item.medium {{ border-left:3px solid var(--yl); }}
.risk-item.low {{ border-left:3px solid var(--gn); }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <h1>📊 Hermes Trading System</h1>
    <div class="meta">
        <span>{ts}</span> · 
        <span>Token: ${token.get('total_usd', 0):.2f}</span> ·
        <button class="btn" onclick="location.reload()">↻ 刷新</button>
    </div>
</div>

<!-- MARKET CONTEXT -->
<div class="section">
    <h2>📈 市场环境</h2>
    <div class="market-row">
"""
    
    for name, info in market.items():
        color = _color(info.get("change_pct", 0))
        html += f"""
        <div class="market-item">
            <span class="name">{name}</span>
            <span class="price">{info.get('price', 0):,.2f}</span>
            <span class="{color}">{_pct(info.get('change_pct', 0))}</span>
        </div>"""
    
    html += """
    </div>
</div>

<!-- TWO COLUMNS: SHIELD + ARROW -->
<div class="cols">
"""
    
    # 蓝盾V4
    html += """
    <div class="section shield">
        <h2>🔵 蓝盾V4 · ML排序</h2>
        <h3>Top-15, 5天轮换, XGBoost CUDA, WF夏普1.203</h3>
"""
    for i, stock in enumerate(shield):
        ticker = stock.get("ticker", stock.get("code", "???"))
        score = stock.get("score", 0)
        delta = shield_deltas.get(ticker, {})
        light, cls, label = _traffic_light(score, shield_thresh)
        html += f"""
        <div class="stock-row">
            <span class="rank">{i+1}</span>
            <span class="ticker">{ticker}</span>
            <span class="score {cls}">{score:.2f}</span>
            <span class="delta">{_rank_delta_html(delta)}</span>
            <span class="signal">{light}</span>
        </div>"""
    
    html += """
    </div>
"""
    
    # 绿箭V9
    html += """
    <div class="section arrow">
        <h2>🟢 绿箭V9 · 量化彩票</h2>
        <h3>Top-10, 高波动高收益, 概率预测</h3>
"""
    for i, stock in enumerate(arrow):
        ticker = stock.get("ticker", stock.get("code", "???"))
        prob = stock.get("probability", 0)
        delta = arrow_deltas.get(ticker, {})
        light, cls, label = _arrow_light(prob, arrow_thresh)
        html += f"""
        <div class="stock-row">
            <span class="rank">{i+1}</span>
            <span class="ticker">{ticker}</span>
            <span class="score {cls}">{prob:.1%}</span>
            <span class="delta">{_rank_delta_html(delta)}</span>
            <span class="signal">{light}</span>
        </div>"""
    
    html += """
    </div>
</div>
"""
    
    # PORTFOLIO
    html += """
<div class="section portfolio">
    <h2>💼 持仓状态</h2>
    <div class="cols">
"""
    # 大盘持仓
    large_cap = portfolio.get("large_cap", [])
    html += """
        <div>
            <h3>大盘持仓 (蓝盾)</h3>
"""
    for pos in large_cap:
        pnl = pos.get("pnl", 0)
        html += f"""
            <div class="stock-row">
                <span class="ticker">{pos.get('ticker', '???')}</span>
                <span class="score {_color(pnl)}">{_pct(pnl)}</span>
            </div>"""
    if not large_cap:
        html += '<div style="color:var(--dm);padding:8px;">暂无持仓</div>'
    
    html += """
        </div>
"""
    
    # 小盘持仓
    small_cap = portfolio.get("small_cap", [])
    html += """
        <div>
            <h3>小盘持仓 (绿箭)</h3>
"""
    for pos in small_cap:
        pnl = pos.get("pnl", 0)
        html += f"""
            <div class="stock-row">
                <span class="ticker">{pos.get('ticker', '???')}</span>
                <span class="score {_color(pnl)}">{_pct(pnl)}</span>
            </div>"""
    if not small_cap:
        html += '<div style="color:var(--dm);padding:8px;">暂无持仓</div>'
    
    html += """
        </div>
    </div>
</div>
"""
    
    # MODEL HEALTH
    shield_h = health.get("shield", {})
    arrow_h = health.get("arrow", {})
    html += f"""
<div class="section health">
    <h2>📊 模型健康</h2>
    <div class="health-grid">
        <div class="health-card">
            <div class="label">蓝盾 WF 夏普</div>
            <div class="value g">{shield_h.get('wf_sharpe', 0):.3f}</div>
            <div class="sub">年化 {shield_h.get('wf_annual', 0)*100:.1f}% · DD {shield_h.get('wf_max_dd', 0)*100:.1f}%</div>
        </div>
        <div class="health-card">
            <div class="label">蓝盾模型</div>
            <div class="value">{shield_h.get('model_type', '???')}</div>
            <div class="sub">{shield_h.get('features', 0)}维特征 · 训练 {shield_h.get('last_train', '???')}</div>
        </div>
        <div class="health-card">
            <div class="label">绿箭 WF 夏普</div>
            <div class="value g">{arrow_h.get('wf_sharpe', 0):.3f}</div>
            <div class="sub">年化 {arrow_h.get('wf_annual', 0)*100:.1f}% · DD {arrow_h.get('wf_max_dd', 0)*100:.1f}%</div>
        </div>
        <div class="health-card">
            <div class="label">Token 费用</div>
            <div class="value">${token.get('total_usd', 0):.2f}</div>
            <div class="sub">缓存命中 98%</div>
        </div>
    </div>
</div>
"""
    
    # ACTIONS + RISKS
    html += """
<div class="cols">
    <div class="section actions">
        <h2>🎯 操作建议</h2>
"""
    for sig in entry_signals[:10]:
        strength = sig.get("strength", "none")
        emoji = "🟢🟢" if strength == "strong" else "🟢" if strength == "entry" else "🟡"
        html += f"""
        <div class="stock-row">
            <span class="signal">{emoji}</span>
            <span class="ticker">{sig.get('ticker', '???')}</span>
            <span style="flex:1;color:var(--dm);">{sig.get('reason', '')}</span>
        </div>"""
    if not entry_signals:
        html += '<div style="color:var(--dm);padding:8px;">暂无信号</div>'
    
    html += """
    </div>
    <div class="section risks">
        <h2>⚠️ 风险提示</h2>
"""
    for risk in risks:
        level = risk.get("level", "low")
        html += f"""
        <div class="risk-item {level}">
            {risk.get('message', '')}
        </div>"""
    if not risks:
        html += '<div style="color:var(--dm);padding:8px;">✅ 无风险提示</div>'
    
    html += """
    </div>
</div>

</body>
</html>
"""
    
    return html


if __name__ == "__main__":
    # 加载数据
    data_file = os.path.join(OUTPUT_DIR, "live_data.json")
    if os.path.exists(data_file):
        with open(data_file) as f:
            data = json.load(f)
    else:
        # 先生成数据
        from dashboard_engine import generate_dashboard_data
        data = generate_dashboard_data()
    
    # 生成HTML
    html = generate_html(data)
    
    # 保存
    output_file = os.path.join(OUTPUT_DIR, "dashboard.html")
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"Dashboard saved to {output_file}")
