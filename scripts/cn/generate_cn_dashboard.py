#!/usr/bin/env python3
"""生成A股看板HTML — Redwood v1.0适配"""
import json, os
from datetime import datetime

ROOT = '/home/hermes/.hermes/openclaw-archive'

# 加载Redwood v1.0信号
signal_path = os.path.join(ROOT, 'signals/cn/latest_xgb.json')
if not os.path.exists(signal_path):
    print(f"❌ 信号文件不存在: {signal_path}")
    exit(1)

with open(signal_path) as f:
    signal = json.load(f)

top = signal.get('top', [])
regime = signal.get('regime', 'unknown')
position_pct = signal.get('position_pct', 0)
market = signal.get('market', {})
signal_date = signal.get('date', 'N/A')
strategy = signal.get('strategy', 'redwood-v1.0')
hold_days = signal.get('hold_days', 10)
top_n = signal.get('top_n', 15)
stop_loss = signal.get('stop_loss', -0.02)

# 市场状态颜色
regime_colors = {
    'bull': ('#10b981', '🟢 BULL 牛市', '满仓100%'),
    'cautious': ('#f59e0b', '🟡 CAUTIOUS 谨慎', '半仓50%'),
    'bear': ('#ef4444', '🔴 BEAR 熊市', '空仓0%'),
}
rc, rl, rp = regime_colors.get(regime, ('#6b7280', '⚪ UNKNOWN', '观望'))

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股 Redwood 看板</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0; 
    min-height: 100vh; padding: 16px;
}}
.header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 20px; padding-bottom: 12px;
    border-bottom: 1px solid #1e293b;
}}
.header h1 {{ font-size: 20px; color: #f8fafc; }}
.header .meta {{ font-size: 12px; color: #64748b; }}

.regime-card {{
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px; margin-bottom: 16px;
    display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px;
}}
.regime-item {{ text-align: center; }}
.regime-item .label {{ font-size: 11px; color: #64748b; text-transform: uppercase; margin-bottom: 4px; }}
.regime-item .value {{ font-size: 18px; font-weight: 700; }}
.regime-item .sub {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}

.signal-section {{
    background: #1e293b; border-radius: 12px; padding: 16px; margin-bottom: 16px;
}}
.signal-section h2 {{
    font-size: 15px; color: #94a3b8; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}}
.signal-section h2 .badge {{
    background: {rc}; color: #fff; font-size: 10px; padding: 2px 8px;
    border-radius: 10px; font-weight: 600;
}}
table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
}}
th {{
    text-align: left; padding: 8px 6px; color: #64748b; font-size: 11px;
    text-transform: uppercase; border-bottom: 1px solid #334155;
    position: sticky; top: 0; background: #1e293b;
}}
td {{
    padding: 7px 6px; border-bottom: 1px solid #1e293b;
}}
tr:hover {{ background: #334155; }}
.rank {{ color: #64748b; font-size: 11px; width: 30px; }}
.sym {{ font-weight: 600; color: #f8fafc; }}
.name {{ color: #94a3b8; font-size: 12px; }}
.industry {{ color: #64748b; font-size: 11px; }}
.price {{ color: #94a3b8; }}
.score {{ 
    font-weight: 700; 
    padding: 2px 8px; border-radius: 4px;
    display: inline-block; min-width: 50px; text-align: center;
}}
.score-gg {{ background: rgba(16,185,129,0.3); color: #10b981; }}
.score-g {{ background: rgba(16,185,129,0.15); color: #34d399; }}
.score-y {{ background: rgba(245,158,11,0.2); color: #f59e0b; }}
.score-bar {{
    display: inline-block; height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #10b981, #3b82f6);
    vertical-align: middle; margin-left: 4px;
}}
.info-card {{
    background: #1e293b; border-radius: 12px; padding: 16px;
    font-size: 12px; color: #94a3b8; line-height: 1.8;
}}
.info-card h3 {{ color: #f8fafc; font-size: 14px; margin-bottom: 8px; }}
.info-card .tag {{
    display: inline-block; background: #334155; color: #94a3b8;
    padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-right: 4px;
}}
@media (max-width: 768px) {{
    .regime-card {{ grid-template-columns: 1fr 1fr; }}
    table {{ font-size: 11px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>🇨🇳 A股 Redwood 看板</h1>
    <div class="meta">{strategy} | 数据截至 {signal_date} | 更新 {datetime.now().strftime('%m-%d %H:%M')}</div>
</div>

<!-- 市场状态 -->
<div class="regime-card">
    <div class="regime-item">
        <div class="label">市场状态</div>
        <div class="value" style="color:{rc}">{rl}</div>
        <div class="sub">三层过滤器判定</div>
    </div>
    <div class="regime-item">
        <div class="label">建议仓位</div>
        <div class="value" style="color:{rc}">{rp}</div>
        <div class="sub">Bull=100% / Cau=50% / Bear=0%</div>
    </div>
    <div class="regime-item">
        <div class="label">市场宽度</div>
        <div class="value">{market.get('breadth',0)*100:.1f}%</div>
        <div class="sub">{'✅ 宽度OK' if market.get('breadth',0) > 0.40 else '⚠️ 宽度弱' if market.get('breadth',0) > 0.30 else '❌ 宽度极弱'}</div>
    </div>
    <div class="regime-item">
        <div class="label">20日动量</div>
        <div class="value" style="color:{'#10b981' if market.get('ret20',0) > 0 else '#ef4444'}">{market.get('ret20',0)*100:+.2f}%</div>
        <div class="sub">{'✅ 正动量' if market.get('ret20',0) > 0 else '⚠️ 负动量' if market.get('ret20',0) > -0.05 else '❌ 深度回调'}</div>
    </div>
</div>

<!-- Top信号 -->
<div class="signal-section">
    <h2>
        📈 Top{top_n} 信号
        <span class="badge">{rl}</span>
        <span style="font-size:11px;color:#64748b;margin-left:auto">{hold_days}天持有 · 止损{stop_loss*100:.0f}% · {position_pct*100:.0f}%仓位</span>
    </h2>
    <div style="overflow-x:auto">
    <table>
        <thead>
            <tr>
                <th>#</th><th>信号</th><th>代码</th><th>名称</th><th>行业</th><th>价格</th><th>评分</th><th>RSI</th><th>20d</th>
            </tr>
        </thead>
        <tbody>
'''

for s in top:
    score100 = s.get('score100', 0)
    sig = s.get('signal', '-')
    sig_emoji = {'GG': '🟢🟢', 'G': '🟢', 'Y': '🟡', '-': '⚪'}.get(sig, '⚪')
    sc_class = 'score-gg' if sig == 'GG' else 'score-g' if sig == 'G' else 'score-y'
    bar_w = max(0, min(100, score100))
    rsi = s.get('rsi', 50)
    ret20 = s.get('ret20', 0)
    html += f'''            <tr>
                <td class="rank">{s['rank']}</td>
                <td>{sig_emoji}</td>
                <td class="sym">{s['sym']}</td>
                <td class="name">{s.get('name','')}</td>
                <td class="industry">{s.get('industry','')}</td>
                <td class="price">¥{s['close']:.2f}</td>
                <td><span class="score {sc_class}">{score100}</span><span class="score-bar" style="width:{bar_w}px"></span></td>
                <td>{rsi:.0f}</td>
                <td style="color:{'#10b981' if ret20>0 else '#ef4444'}">{ret20*100:+.1f}%</td>
            </tr>
'''

html += f'''        </tbody>
    </table>
    </div>
</div>

<!-- 策略说明 -->
<div class="info-card">
    <h3>🧠 Redwood v1.0 策略说明</h3>
    <p>
        <span class="tag">XGBoost</span> 25特征 · Walk-Forward验证 · 资金流+技术+宏观<br>
        <span class="tag">反转</span> A股短期反转效应，低动量/低RSI = 未来反弹概率更高<br>
        <span class="tag">资金流</span> 大/中/小/超大单净流入排名，机构在买的股票倾向继续涨<br>
        <span class="tag">市场过滤</span> 宽度>40%+20d动量>-5% = 满仓，否则半仓/空仓<br>
        <span class="tag">信号分级</span> 🟢🟢精品(≥80) / 🟢强信号(≥60) / 🟡观察(≥40)<br>
    </p>
    <p style="margin-top:8px;color:#64748b">
        <b>调仓:</b> 每{hold_days}个交易日 · Top{top_n}等权 · 止损{stop_loss*100:.0f}%<br>
        <b>数据:</b> tushare K线 + 资金流 · A股全市场(含688)<br>
    </p>
</div>

<div style="text-align:center;color:#334155;font-size:10px;margin-top:16px">
    Hermes Trading · Redwood v1.0 · 数据来源tushare · 仅供研究参考
</div>

</body>
</html>'''

# 保存
os.makedirs(os.path.join(ROOT, 'output'), exist_ok=True)
out_path = os.path.join(ROOT, 'output/cn_dashboard.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"✅ A股看板已生成: {out_path}")
print(f"   大小: {os.path.getsize(out_path)/1024:.1f}KB")
print(f"   日期: {signal_date}, {len(top)}只, {regime}")
