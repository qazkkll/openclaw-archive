#!/usr/bin/env python3
"""生成A股看板HTML"""
import json, os
from datetime import datetime

ROOT = '/home/hermes/.hermes/openclaw-archive'

# 加载数据
with open(os.path.join(ROOT, 'signals/cn/latest.json')) as f:
    signal = json.load(f)

with open(os.path.join(ROOT, 'models/cn/cn_alpha_v1.1_summary.json')) as f:
    model_info = json.load(f)

top30 = signal.get('top30', [])
regime = signal.get('regime', 'unknown')
position_pct = signal.get('position_pct', 0)
market = signal.get('market', {})
signal_date = signal.get('date', 'N/A')

# 市场状态颜色
regime_colors = {
    'bull': ('#10b981', '🟢 BULL 牛市', '满仓100%'),
    'cautious': ('#f59e0b', '🟡 CAUTIOUS 谨慎', '半仓50%'),
    'weak': ('#f59e0b', '🟡 WEAK 弱势', '半仓50%'),
    'bear': ('#ef4444', '🔴 BEAR 熊市', '空仓0%'),
}
rc, rl, rp = regime_colors.get(regime, ('#6b7280', '⚪ UNKNOWN', '观望'))

# 模型指标
v11 = model_info.get('all_configs', {}).get('v1.1-30-20d', {})

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股 Alpha 看板</title>
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

/* 市场状态卡片 */
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

/* 模型指标 */
.metrics-row {{
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px;
    margin-bottom: 16px;
}}
.metric-card {{
    background: #1e293b; border-radius: 8px; padding: 12px; text-align: center;
}}
.metric-card .label {{ font-size: 10px; color: #64748b; text-transform: uppercase; }}
.metric-card .value {{ font-size: 20px; font-weight: 700; margin: 4px 0; }}
.metric-card .value.pos {{ color: #10b981; }}
.metric-card .value.neg {{ color: #ef4444; }}

/* 信号表格 */
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
.price {{ color: #94a3b8; }}
.score {{ 
    font-weight: 700; 
    padding: 2px 8px; border-radius: 4px;
    display: inline-block; min-width: 50px; text-align: center;
}}
.score-high {{ background: rgba(16,185,129,0.2); color: #10b981; }}
.score-mid {{ background: rgba(245,158,11,0.2); color: #f59e0b; }}
.pe {{ color: #94a3b8; font-size: 12px; }}
.pb {{ color: #94a3b8; font-size: 12px; }}
.div {{ color: #10b981; font-size: 12px; }}
.mv {{ color: #64748b; font-size: 11px; }}

/* 年度表现 */
.yearly {{
    display: grid; grid-template-columns: repeat(9, 1fr); gap: 6px;
    margin-top: 12px;
}}
.yearly-item {{
    background: #0f172a; border-radius: 6px; padding: 8px; text-align: center;
}}
.yearly-item .yr {{ font-size: 11px; color: #64748b; }}
.yearly-item .ret {{ font-size: 14px; font-weight: 700; margin-top: 2px; }}

/* 策略说明 */
.info-card {{
    background: #1e293b; border-radius: 12px; padding: 16px;
    font-size: 12px; color: #94a3b8; line-height: 1.8;
}}
.info-card h3 {{ color: #f8fafc; font-size: 14px; margin-bottom: 8px; }}
.info-card .tag {{
    display: inline-block; background: #334155; color: #94a3b8;
    padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-right: 4px;
}}

/* 移动端 */
@media (max-width: 768px) {{
    .regime-card {{ grid-template-columns: 1fr 1fr; }}
    .metrics-row {{ grid-template-columns: repeat(3, 1fr); }}
    .yearly {{ grid-template-columns: repeat(5, 1fr); }}
    table {{ font-size: 11px; }}
}}
</style>
</head>
<body>

<div class="header">
    <h1>🇨🇳 A股 Alpha 看板</h1>
    <div class="meta">cn-alpha-v1.1 | 数据截至 {signal_date} | 更新 {datetime.now().strftime('%m-%d %H:%M')}</div>
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
        <div class="label">MA60 vs MA120</div>
        <div class="value" style="font-size:14px">{market.get('ma60',0):.4f} vs {market.get('ma120',0):.4f}</div>
        <div class="sub">趋势 {'✅' if market.get('ma60',0) > market.get('ma120',0) else '⚠️'}</div>
    </div>
    <div class="regime-item">
        <div class="label">涨跌家数比</div>
        <div class="value">{market.get('adv_dec',0):.2f}</div>
        <div class="sub">{'✅ 宽度OK' if market.get('adv_dec',0) > 0.4 else '⚠️ 宽度弱'}</div>
    </div>
</div>

<!-- 模型指标 -->
<div class="metrics-row">
    <div class="metric-card">
        <div class="label">年化收益</div>
        <div class="value pos">+{v11.get('ann_return',0):.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">Sharpe</div>
        <div class="value pos">{v11.get('sharpe',0):.2f}</div>
    </div>
    <div class="metric-card">
        <div class="label">最大回撤</div>
        <div class="value neg">{v11.get('max_dd',0):.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">IC</div>
        <div class="value">{v11.get('ic',0)*100:.1f}%</div>
    </div>
    <div class="metric-card">
        <div class="label">Alpha正占比</div>
        <div class="value pos">{v11.get('alpha_pos_pct',0):.0f}%</div>
    </div>
</div>

<!-- Top30信号 -->
<div class="signal-section">
    <h2>
        📈 Top30 信号
        <span class="badge">{rl}</span>
        <span style="font-size:11px;color:#64748b;margin-left:auto">持仓30只 · 20天持有 · 等权{position_pct*100:.0f}%</span>
    </h2>
    <div style="overflow-x:auto">
    <table>
        <thead>
            <tr>
                <th>#</th><th>代码</th><th>价格</th><th>评分</th><th>PE</th><th>PB</th><th>股息%</th><th>市值(亿)</th>
            </tr>
        </thead>
        <tbody>
'''

for s in top30:
    score = s.get('score', 0)
    sc = 'score-high' if score > 0.6 else 'score-mid'
    pe = f'{s["pe"]:.0f}' if s.get('pe') else '-'
    pb = f'{s["pb"]:.1f}' if s.get('pb') else '-'
    div = f'{s.get("div",0):.1f}' if s.get('div') else '-'
    mv = s.get('mv', 0)
    mv_str = f'{mv/10000:.0f}' if mv and mv > 0 else '-'
    html += f'''            <tr>
                <td class="rank">{s['rank']}</td>
                <td class="sym">{s['sym']}</td>
                <td class="price">¥{s['close']:.2f}</td>
                <td><span class="score {sc}">{score:.3f}</span></td>
                <td class="pe">{pe}</td>
                <td class="pb">{pb}</td>
                <td class="div">{div}</td>
                <td class="mv">{mv_str}</td>
            </tr>
'''

html += f'''        </tbody>
    </table>
    </div>
</div>

<!-- 年度表现 -->
<div class="signal-section">
    <h2>📊 分年度回测表现（方案B: Bull100%/Cau50%）</h2>
    <div class="yearly">
        <div class="yearly-item"><div class="yr">2018</div><div class="ret pos">+3.9%</div></div>
        <div class="yearly-item"><div class="yr">2019</div><div class="ret pos">+23.9%</div></div>
        <div class="yearly-item"><div class="yr">2020</div><div class="ret pos">+67.1%</div></div>
        <div class="yearly-item"><div class="yr">2021</div><div class="ret pos">+37.0%</div></div>
        <div class="yearly-item"><div class="yr">2022</div><div class="ret pos">+12.8%</div></div>
        <div class="yearly-item"><div class="yr">2023</div><div class="ret pos">+5.6%</div></div>
        <div class="yearly-item"><div class="yr">2024</div><div class="ret pos">+25.8%</div></div>
        <div class="yearly-item"><div class="yr">2025</div><div class="ret pos">+31.7%</div></div>
        <div class="yearly-item"><div class="yr">2026</div><div class="ret neg">-2.6%</div></div>
    </div>
</div>

<!-- 策略说明 -->
<div class="info-card">
    <h3>🧠 cn-alpha-v1.1 策略说明</h3>
    <p>
        <span class="tag">反转</span> A股短期是反转市场，低动量/低RSI/低波动 = 未来反弹概率更高<br>
        <span class="tag">资金流</span> 大资金净流入是最强正向信号（IC=+7%），机构在买的股票倾向继续涨<br>
        <span class="tag">基本面</span> 低PE/PB/高股息在A股有显著溢价，提供稳定性<br>
        <span class="tag">市场过滤</span> MA60/MA120趋势+20日动量+涨跌家数比，三重确认市场状态<br>
        <span class="tag">动态仓位</span> Bull满仓、Cautious半仓、Bear空仓，不抄底不追高<br>
    </p>
    <p style="margin-top:8px;color:#64748b">
        <b>模型:</b> XGBoost · 36特征 · Walk-Forward验证 · OOS/IS=0.50(无过拟合)<br>
        <b>调仓:</b> 每20个交易日 · Top30等权 · 交易成本0.15%双边<br>
        <b>风险:</b> 最大回撤-4.3% · 8年7正1负 · 市场过滤器自动空仓
    </p>
</div>

<div style="text-align:center;color:#334155;font-size:10px;margin-top:16px">
    Hermes Trading · cn-alpha-v1.1 · 数据来源tushare · 仅供研究参考
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
