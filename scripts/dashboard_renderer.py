#!/usr/bin/env python3
"""
Hermes Dashboard Renderer v3 — 重新整理

布局结构：
┌─────────────────────────────────────────────────────────────────┐
│  📊 Hermes · 美股盯盘                      [一键分析] [刷新]   │
├─────────────────────────────────────────────────────────────────┤
│  📈 S&P500 🟢7546+0.5% │ VIX 🟢15.6 │ ... │ 🟢利好 72°       │
├─────────────────────────────────────────────────────────────────┤
│  🗺️ 板块: [金融🟢+1.2] [科技🟡+0.3] [医疗🔴-0.5] ...         │
├───────────────────────────┬─────────────────────────────────────┤
│  🔵 蓝盾V3 · 趋势跟踪     │  🟢 绿箭V9 · 量化彩票              │
│  (15只推荐)                │  (10只推荐)                        │
├───────────────────────────┼─────────────────────────────────────┤
│  💼 大盘股持仓 (蓝盾)      │  💼 小盘股持仓 (绿箭)              │
│  (用蓝盾分数评分)          │  (用绿箭概率评分)                  │
├───────────────────────────┼─────────────────────────────────────┤
│  🎯 入场信号 · 调仓建议    │  ⚠️ 风险提示                       │
└───────────────────────────┴─────────────────────────────────────┘
"""

import json, os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
CONFIG_PATH = os.path.join(ROOT, "config.json")


def _c(color):
    return f"c-{color}" if color != "gray" else "c-gray"


def _rank_delta_html(d):
    if d is None: return '<span class="rn">NEW</span>'
    trend = d.get("trend", "flat")
    delta = d.get("delta", 0)
    open_rank = d.get("open_rank")
    if trend == "new" or open_rank is None: return '<span class="rn">NEW</span>'
    if delta > 0: return f'<span class="ru">\U0001f53a{delta}</span>'
    elif delta < 0: return f'<span class="rd">\U0001f53b{abs(delta)}</span>'
    return '<span class="rf">\u2192</span>'


def _dot(val):
    if val >= 0.7: return '<span class="dot g"></span>'
    elif val >= 0.4: return '<span class="dot y"></span>'
    return '<span class="dot r"></span>'


def _pcls(val):
    if val > 0: return "g"
    if val < 0: return "r"
    return "d"


def _pret(val):
    return f"+{val:.1f}%" if val > 0 else f"{val:.1f}%"


def _tech_light(stock):
    score = stock.get("score", 50)
    rsi = stock.get("rsi", 50)
    trend = stock.get("trend_score", 0)
    ret = stock.get("daily_return", 0)
    w52 = stock.get("week52pct", 50)
    pts = 0
    if trend >= 4: pts += 2
    elif trend >= 3: pts += 1
    if 30 <= rsi <= 60: pts += 2
    elif 20 <= rsi <= 70: pts += 1
    if 60 <= w52 <= 90: pts += 2
    elif 50 <= w52 <= 95: pts += 1
    if ret > 1: pts += 2
    elif ret > 0: pts += 1
    if score >= 90: pts += 2
    elif score >= 80: pts += 1
    if pts >= 8: return "\U0001f7e2\U0001f7e2", "g"
    elif pts >= 6: return "\U0001f7e2", "g"
    elif pts >= 4: return "\U0001f7e1", "y"
    elif pts >= 2: return "\U0001f7e0", "r"
    else: return "\U0001f534", "r"


def _get_thresholds():
    """从config.json读取阈值"""
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    return cfg["scoring"]["shield"]["thresholds"], cfg["scoring"]["arrow"]["thresholds"]


def generate_new_dashboard(data):
    """生成 Hermes 仪表盘 v3"""
    ts = data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))
    sentiment = data.get("sentiment", {})
    shield = data.get("shield_list", [])
    shield_deltas = data.get("shield_deltas", {})
    arrow = data.get("arrow_list", [])
    arrow_deltas = data.get("arrow_deltas", {})
    entry_signals = data.get("entry_signals", [])
    sector_heatmap = data.get("sector_heatmap", [])
    portfolio = data.get("portfolio", {})
    
    # 读取阈值
    shield_t, arrow_t = _get_thresholds()
    
    large_cap = portfolio.get("large_cap", [])
    small_cap = portfolio.get("small_cap", [])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="300">
<title>Hermes 盯盘仪表盘</title>
<style>
:root{{--bg:#0a0a0a;--c1:#111;--c2:#1a1a1a;--bd:#222;--tx:#e0e0e0;--dm:#888;
--gn:#4caf50;--yl:#ffc107;--rd:#f44336;--bl:#2196f3;--pp:#9c27b0;--cy:#00bcd4;--or:#ff9800;
--f:'SF Mono','Cascadia Code','Consolas','Menlo',monospace}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--tx);font-family:var(--f);padding:14px;font-size:14px;line-height:1.45}}
.hdr{{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:2px solid var(--bd);margin-bottom:10px}}
.hdr h1{{font-size:22px;color:#fff}}.hdr .info{{color:var(--dm);font-size:12px;text-align:right}}.hdr .live{{color:var(--gn);font-weight:bold}}
.btns{{display:flex;gap:8px;margin-bottom:10px;align-items:center}}
.btn{{background:linear-gradient(135deg,#1a73e8,#0d47a1);color:#fff;border:none;padding:10px 24px;border-radius:6px;font-size:15px;font-weight:bold;cursor:pointer;font-family:var(--f);transition:all .15s}}
.btn:hover{{transform:translateY(-1px)}}.btn:active{{transform:translateY(0)}}.btn.busy{{background:#555;cursor:not-allowed}}
.btn-sm{{background:var(--c2);color:var(--dm);border:1px solid var(--bd);padding:8px 14px;border-radius:4px;font-size:12px;cursor:pointer;font-family:var(--f)}}
.status{{color:var(--dm);font-size:12px;margin-left:10px}}.status.err{{color:var(--rd)}}
.temp-bar{{display:flex;gap:6px;margin-bottom:8px;align-items:stretch;flex-wrap:wrap}}
.temp-card{{flex:1;min-width:100px;background:var(--c1);border-radius:6px;padding:8px 10px;border:1px solid var(--bd);text-align:center}}
.temp-card .n{{font-size:10px;color:var(--dm)}}.temp-card .v{{font-size:18px;font-weight:bold;margin:2px 0}}.temp-card .d{{font-size:10px}}
.temp-card .tag{{font-size:9px;padding:1px 4px;border-radius:2px;display:inline-block;margin-top:2px}}
.temp-badge{{flex:0 0 120px;display:flex;flex-direction:column;justify-content:center;align-items:center;border-radius:8px;padding:8px;border:2px solid}}
.temp-badge.green{{background:#0d1f0d;border-color:var(--gn)}}.temp-badge.yellow{{background:#1f1f0d;border-color:var(--yl)}}.temp-badge.red{{background:#1f0d0d;border-color:var(--rd)}}
.temp-badge .lbl{{font-size:16px;font-weight:bold}}.temp-badge .det{{font-size:10px;color:var(--dm)}}.temp-badge .temp-num{{font-size:24px;font-weight:bold}}
.sect-bar{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;padding:8px;background:var(--c1);border-radius:6px}}
.sect-chip{{padding:4px 8px;border-radius:4px;font-size:11px;display:flex;align-items:center;gap:3px}}
.sect-chip.g{{background:#0d1f0d;border:1px solid var(--gn);color:var(--gn)}}.sect-chip.y{{background:#1f1f0d;border:1px solid var(--yl);color:var(--yl)}}.sect-chip.r{{background:#1f0d0d;border:1px solid var(--rd);color:var(--rd)}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}}@media(max-width:1200px){{.cols{{grid-template-columns:1fr}}}}
.sec{{padding:10px;background:var(--c1);border-radius:6px;border-left:3px solid var(--bd)}}
.sec h2{{font-size:14px;margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid var(--bd)}}
.sec.s-bl h2{{color:var(--bl)}}.sec.s-bl{{border-left-color:var(--bl)}}
.sec.s-ar h2{{color:var(--gn)}}.sec.s-ar{{border-left-color:var(--gn)}}
.sec.s-pf h2{{color:var(--cy)}}.sec.s-pf{{border-left-color:var(--cy)}}
.sec.s-pf2 h2{{color:var(--or)}}.sec.s-pf2{{border-left-color:var(--or)}}
.sec.s-sg h2{{color:var(--yl)}}.sec.s-sg{{border-left-color:var(--yl)}}
.sec.s-rk h2{{color:var(--rd)}}.sec.s-rk{{border-left-color:var(--rd)}}
.sct{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{text-align:left;padding:4px 5px;background:#1a1a1a;color:var(--dm);border-bottom:1px solid var(--bd);font-weight:normal;font-size:10px;white-space:nowrap}}
td{{padding:4px 5px;border-bottom:1px solid #1a1a1a;white-space:nowrap}}
tr:hover{{background:var(--c2)}}.top3{{background:#0d1f0d!important}}.b{{font-weight:bold}}
.g{{color:var(--gn)}}.y{{color:var(--yl)}}.r{{color:var(--rd)}}.d{{color:var(--dm)}}.bl{{color:var(--bl)}}
.ru{{color:var(--gn);font-weight:bold;font-size:10px}}.rd{{color:var(--rd);font-weight:bold;font-size:10px}}.rf{{color:var(--dm);font-size:10px}}.rn{{color:var(--yl);font-size:9px;font-weight:bold}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 1px;vertical-align:middle}}
.dot.g{{background:var(--gn);box-shadow:0 0 4px var(--gn)}}.dot.y{{background:var(--yl)}}.dot.r{{background:#444}}
.bar{{display:inline-block;height:6px;border-radius:3px;background:#222;vertical-align:middle;min-width:50px}}
.bar .fill{{height:100%;border-radius:3px}}.bar.gn .fill{{background:var(--gn)}}.bar.yl .fill{{background:var(--yl)}}
.ag{{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}}.ac{{padding:4px 8px;border-radius:4px;font-size:11px}}
.ac.b{{background:#0d1f0d;border:1px solid var(--gn);color:var(--gn)}}.ac.h{{background:#1f1f0d;border:1px solid var(--yl);color:var(--yl)}}
.rk{{padding:4px 8px;margin:2px 0;background:#1f0d0d;border-radius:3px;color:#ff8a80;font-size:11px}}
.ft{{text-align:center;padding:10px;color:#444;font-size:10px}}
.tl{{font-size:13px;text-align:center;white-space:nowrap}}
</style>
</head>
<body>

<div class="hdr"><h1>\U0001f4ca Hermes</h1><div class="info"><div>\U0001f1fa\U0001f1f8 \u7f8e\u80a1 \u00b7 \u84dd\u76feV3 + \u7eff\u7badV9 \u00b7 {ts}</div><div class="live">\u25cf LIVE \u00b7 <span id="cd">300</span>s</div></div></div>
<div class="btns"><button class="btn" id="abtn" onclick="doAnalyze()">\u26a1 \u4e00\u952e\u5206\u6790</button><button class="btn-sm" onclick="location.reload()">\U0001f504</button><span class="status" id="ast">\u5c31\u7eea</span></div>

<!-- Temperature -->
<div class="temp-bar">
"""
    COLOR_MAP = {"green": "gn", "yellow": "yl", "red": "rd", "gray": "bd"}
    for name, idx in sentiment.get("indices", {}).items():
        color = idx.get("color", "gray")
        css = COLOR_MAP.get(color, "bd")
        html += f'<div class="temp-card" style="border-color:var(--{css})"><div class="n">{name}</div><div class="v {_c(color)}">{idx.get("value","\u2014")}</div><div class="d {_c(color)}">{idx.get("change","")}</div><div class="tag {_c(color)}" style="background:var(--{css})">{idx.get("label","")}</div></div>\n'
    oc = sentiment.get("overall_color", "gray")
    html += f'''<div class="temp-badge {oc}"><div class="lbl">{sentiment.get("overall_label","\u672a\u77e5")}</div><div class="temp-num">{"72" if oc=="green" else "45" if oc=="yellow" else "20"}\u00b0</div><div class="det">\U0001f7e2{sentiment.get("green_count",0)} \U0001f7e1{sentiment.get("yellow_count",0)} \U0001f534{sentiment.get("red_count",0)}</div></div></div>

<!-- Sectors (top) -->
<div class="sect-bar"><span style="font-size:11px;color:var(--dm);margin-right:6px">\U0001f5fa\ufe0f</span>
'''
    for s in sector_heatmap:
        rs = f"+{s['avg_return']:.2f}%" if s['avg_return'] > 0 else f"{s['avg_return']:.2f}%"
        html += f'<div class="sect-chip {s["color"]}">{s["sector"]} {rs}</div>\n'
    html += '</div>\n'

    # Two columns: Shield + Arrow
    html += '<div class="cols">\n'
    
    # Left: Shield
    html += '<div class="sec s-bl"><h2>\U0001f535 \u84dd\u76feV3 \u00b7 \u8d8b\u52bf\u8ddf\u8e2a</h2><div class="sct"><table>\n'
    html += '<tr><th class="tl">\u706f</th><th>\u6807\u7684</th><th>\u6392\u540d</th><th>\u0394</th><th>\u5206\u6570</th><th>\u4fe1\u53f7</th><th>\u73b0\u4ef7</th><th>\u6da8\u8dcc</th><th>RSI</th></tr>\n'
    for i, s in enumerate(shield[:15]):
        t = s["ticker"]; d = shield_deltas.get(t); sc = s.get("score", 0)
        if sc >= shield_t["strong_buy"]: light, lc = "\U0001f7e2\U0001f7e2", "g"
        elif sc >= shield_t["buy"]: light, lc = "\U0001f7e2", "g"
        elif sc >= shield_t["watch"]: light, lc = "\U0001f7e1", "y"
        else: light, lc = "\U0001f534", "r"
        tech, _ = _tech_light(s); ret = s.get("daily_return", 0)
        top = ' class="top3"' if i < 3 else ""
        html += f'<tr{top}><td class="tl">{tech}</td><td class="b">{t}</td><td>{i+1}</td><td>{_rank_delta_html(d)}</td><td class="b {_c(lc)}">{sc}</td><td style="font-size:10px;white-space:normal;max-width:120px">{s.get("signal","")}</td><td>${s.get("price",0):.2f}</td><td class="{_pcls(ret)}">{_pret(ret)}</td><td>{s.get("rsi","\u2014")}</td></tr>\n'
    html += '</table></div>\n'
    
    # 大盘股持仓（在蓝盾下面）
    if large_cap:
        html += '<div style="margin-top:8px;padding-top:6px;border-top:1px solid var(--bd)"><h3 style="font-size:12px;color:var(--cy);margin-bottom:4px">\U0001f4bc \u5927\u76d8\u80a1\u6301\u4ed3</h3><table>\n'
        html += '<tr><th>\u6807\u7684</th><th>\u6301\u80a1</th><th>\u6210\u672c</th><th>\u73b0\u4ef7</th><th>\u76c8\u4e8f%</th><th>\u84dd\u76fe\u5206</th><th>\u8bc4\u7ea7</th></tr>\n'
        total_pnl = 0
        for p in large_cap:
            pc = _pcls(p["pnl_pct"]); ps = f"+{p['pnl_pct']:.1f}%" if p['pnl_pct'] > 0 else f"{p['pnl_pct']:.1f}%"
            pv = f"+${p['pnl_value']:,.0f}" if p['pnl_value'] > 0 else f"-${abs(p['pnl_value']):,.0f}"
            vc = "g" if p['pnl_value'] > 0 else "r"
            gc = {"strong": "g", "buy": "g", "watch_bullish": "y", "watch": "y", "avoid": "r", "unknown": "d"}.get(p.get("grade", "unknown"), "d")
            html += f'<tr><td class="b">{p["ticker"]}</td><td>{p["shares"]}</td><td>${p["cost"]:.1f}</td><td>${p["current"]:.1f}</td><td class="{vc}">{ps}</td><td class="b {_c(gc)}">{p.get("model_score","\u2014")}</td><td class="{gc}" style="font-size:10px">{p.get("model_label","\u2014")}</td></tr>\n'
            total_pnl += p['pnl_value']
        if large_cap:
            tc = "g" if total_pnl > 0 else "r"
            html += f'<tr style="border-top:1px solid var(--bd)"><td class="b" colspan="4">\u5c0f\u8ba1</td><td class="b {_c(tc)}">{"+" if total_pnl>0 else ""}${total_pnl:,.0f}</td><td colspan="2"></td></tr>\n'
        html += '</table></div>\n'
    
    html += '</div>\n'  # close shield section
    
    # Right: Arrow
    html += '<div class="sec s-ar"><h2>\U0001f7e2 \u7eff\u7badV9-Lottery \u00b7 \u91cf\u5316\u5f69\u7968</h2><div class="sct"><table>\n'
    html += '<tr><th class="tl">\u706f</th><th>\u6807\u7684</th><th>\u6392\u540d</th><th>\u0394</th><th>\u6982\u7387</th><th>\u786e\u5b9a\u6027</th><th>\u73b0\u4ef7</th><th>\u6da8\u8dcc</th></tr>\n'
    for i, a in enumerate(arrow[:10]):
        t = a["ticker"]; d = arrow_deltas.get(t); prob = a.get("prob", 0)
        if prob >= arrow_t["high_prob"]: light, lc = "\U0001f7e2\U0001f7e2", "g"
        elif prob >= arrow_t["medium_prob"]: light, lc = "\U0001f7e2", "g"
        elif prob >= arrow_t["low_prob"]: light, lc = "\U0001f7e1", "y"
        else: light, lc = "\U0001f534", "r"
        bw = int(prob * 100); cert = f'<div class="bar gn"><div class="fill" style="width:{bw}%"></div></div> {prob*100:.0f}%'
        ret = a.get("daily_return", 0)
        html += f'<tr><td class="tl">{light}</td><td class="b">{t}</td><td>{i+1}</td><td>{_rank_delta_html(d)}</td><td class="b {_c(lc)}">{prob*100:.1f}%</td><td>{cert}</td><td>${a.get("price",0):.2f}</td><td class="{_pcls(ret)}">{_pret(ret)}</td></tr>\n'
    html += '</table></div>\n'
    
    # 小盘股持仓（在绿箭下面）
    if small_cap:
        html += '<div style="margin-top:8px;padding-top:6px;border-top:1px solid var(--bd)"><h3 style="font-size:12px;color:var(--or);margin-bottom:4px">\U0001f4bc \u5c0f\u76d8\u80a1\u6301\u4ed3</h3><table>\n'
        html += '<tr><th>\u6807\u7684</th><th>\u6301\u80a1</th><th>\u6210\u672c</th><th>\u73b0\u4ef7</th><th>\u76c8\u4e8f%</th><th>\u7eff\u7bad%</th><th>\u8bc4\u7ea7</th></tr>\n'
        total_pnl = 0
        for p in small_cap:
            pc = _pcls(p["pnl_pct"]); ps = f"+{p['pnl_pct']:.1f}%" if p['pnl_pct'] > 0 else f"{p['pnl_pct']:.1f}%"
            pv = f"+${p['pnl_value']:,.0f}" if p['pnl_value'] > 0 else f"-${abs(p['pnl_value']):,.0f}"
            vc = "g" if p['pnl_value'] > 0 else "r"
            gc = {"strong": "g", "medium": "g", "low": "y", "very_low": "r", "unknown": "d"}.get(p.get("grade", "unknown"), "d")
            html += f'<tr><td class="b">{p["ticker"]}</td><td>{p["shares"]}</td><td>${p["cost"]:.1f}</td><td>${p["current"]:.1f}</td><td class="{vc}">{ps}</td><td class="b {_c(gc)}">{p.get("model_score","\u2014")}%</td><td class="{gc}" style="font-size:10px">{p.get("model_label","\u2014")}</td></tr>\n'
            total_pnl += p['pnl_value']
        if small_cap:
            tc = "g" if total_pnl > 0 else "r"
            html += f'<tr style="border-top:1px solid var(--bd)"><td class="b" colspan="4">\u5c0f\u8ba1</td><td class="b {_c(tc)}">{"+" if total_pnl>0 else ""}${total_pnl:,.0f}</td><td colspan="2"></td></tr>\n'
        html += '</table></div>\n'
    
    html += '</div>\n'  # close arrow section
    html += '</div>\n'  # close cols

    # Entry signals (compact)
    shown = [e for e in entry_signals if e["signals"]["score"] > 0][:8]
    if shown:
        html += '<div class="sec s-sg"><h2>\U0001f3af \u5165\u573a\u4fe1\u53f7</h2><div class="sct"><table>\n'
        html += '<tr><th>\u6807\u7684</th><th>\u8bc4\u7ea7</th><th>\u4fe1\u53f7\u6761</th><th>\u5206\u6570</th><th>\u6392\u540d\u0394</th><th class="tl">RSI</th><th class="tl">\u5747\u7ebf</th><th class="tl">\u52a8\u91cf</th><th class="tl">\u5171\u8bc6</th></tr>\n'
        for e in shown:
            sig = e["signals"]; gc = {"strong": "g", "entry": "g", "watch": "y", "none": "r"}.get(sig["grade"], "d")
            sigs = sig["signals"]; bw = int(sig["ratio"] * 100)
            html += f'<tr><td class="b">{e["ticker"]}</td><td class="{gc}">{sig["label"]}</td><td><div class="bar {"gn" if sig["ratio"]>=0.5 else "yl"}" style="width:80px"><div class="fill" style="width:{bw}%"></div></div> {sig["score"]:.1f}/{sig["max_score"]:.1f}</td><td>{_rank_delta_html(e.get("rank_delta"))}</td><td class="tl">{_dot(sigs.get("rsi",0))}</td><td class="tl">{_dot(sigs.get("ma_align",0))}</td><td class="tl">{_dot(sigs.get("momentum",0))}</td><td class="tl">{_dot(sigs.get("consensus",0))}</td></tr>\n'
        html += '</table></div></div>\n'

    # Actions + Risks
    strong = [e for e in entry_signals if e["signals"]["grade"] in ("strong", "entry")]
    if strong:
        html += '<div class="sec s-rk" style="border-left-color:var(--or)"><h2 style="color:var(--or)">\U0001f4a1 \u8c03\u4ed3\u5efa\u8bae</h2><div class="ag">'
        for e in strong[:5]:
            html += f'<div class="ac b">{e["ticker"]} \u2014 {e["signals"]["label"]}</div>'
        html += '</div></div>\n'

    html += f'<div class="ft">Powered by Hermes \u00b7 \u84dd\u76feV3 + \u7eff\u7badV9 \u00b7 {ts}</div>\n'

    html += """<script>
(function(){var s=300,el=document.getElementById("cd");function t(){if(s<=0){location.reload();return}el.textContent=s;s--;setTimeout(t,1000)}t()})();
function doAnalyze(){var b=document.getElementById("abtn"),st=document.getElementById("ast");if(b.classList.contains("busy"))return;b.classList.add("busy");b.textContent="\u23f3...";st.textContent="\u8bc4\u5206\u4e2d...";st.className="status";fetch("/api/analyze").then(function(r){return r.json()}).then(function(d){if(d.status==="started"){pollStatus()}else{st.textContent=d.message;b.classList.remove("busy");b.textContent="\u26a1 \u4e00\u952e\u5206\u6790"}}).catch(function(){st.textContent="\u274c \u672a\u8fde\u63a5";st.className="status err";b.classList.remove("busy");b.textContent="\u26a1 \u4e00\u952e\u5206\u6790"})}
function pollStatus(){fetch("/api/status").then(function(r){return r.json()}).then(function(d){if(d.running){setTimeout(pollStatus,2000)}else if(d.last_error){document.getElementById("ast").textContent="\u274c "+d.last_error;document.getElementById("abtn").classList.remove("busy");document.getElementById("abtn").textContent="\u26a1 \u4e00\u952e\u5206\u6790"}else{location.reload()}}).catch(function(){setTimeout(pollStatus,2000)})}
</script>
</body></html>"""
    return html


def generate_html_dashboard(scores, market_context=None):
    from dashboard_engine import build_dashboard_data
    shield = scores.get("shield", [])
    arrow = scores.get("arrow", [])
    data = build_dashboard_data(shield, arrow, market_context or {})
    return generate_new_dashboard(data)
