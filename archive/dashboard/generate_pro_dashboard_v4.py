#!/usr/bin/env python3
"""Hermes Trading Dashboard v4 — Taste-Skill优化版
设计原则：
1. 反emoji策略：用颜色和文字代替emoji
2. 色彩克制：深色底+单一强调色（翠绿/暗红）
3. 数据密度：无卡片边框，用分割线和留白
4. 字体：等宽数字，无衬线
"""
import json, os
from datetime import datetime

ROOT = '/home/hermes/.hermes/openclaw-archive'
def load(p):
    try:
        with open(os.path.join(ROOT, p)) as f: return json.load(f)
    except: return {}

futu = load('output/futu_positions.json')
v6 = load('output/v6_latest.json')
v11 = load('output/v11_latest.json')
held = load('output/held_scores.json')

positions = futu.get('positions', [])
shield_pos = [p for p in positions if p.get('cost_price',0) > 10]
arrow_pos = [p for p in positions if p.get('cost_price',0) <= 10]

# 持仓评分映射
held_map = {}
for h in held.get('shield', []) + held.get('arrow', []):
    held_map[h['sym']] = h

now = datetime.now().strftime('%Y-%m-%d %H:%M')

# === 持仓行 ===
def pos_row(p):
    code = p.get('code','')
    pnl = p.get('pnl_pct', 0)
    held_days = p.get('days_held', 0)
    hold_days = p.get('hold_days', 20)
    prog = min(100, held_days/hold_days*100) if hold_days > 0 else 0
    
    # 模型评分
    hs = held_map.get(code, {})
    if hs:
        rank = hs.get('rank', 0)
        total = hs.get('total', 1)
        pct = rank / total * 100 if total > 0 else 100
        score = hs.get('score', 0)
        if pct <= 5: sig_c, sig_t = '#10b981', f'Top{pct:.0f}%'
        elif pct <= 10: sig_c, sig_t = '#10b981', f'Top{pct:.0f}%'
        elif pct <= 20: sig_c, sig_t = '#f59e0b', f'Top{pct:.0f}%'
        else: sig_c, sig_t = '#ef4444', f'Top{pct:.0f}%'
        model_cell = f'<td style="color:{sig_c};font-size:12px">{sig_t}</td>'
    else:
        model_cell = '<td style="color:#666;font-size:12px">--</td>'
    
    # 盈亏颜色
    pnl_c = '#10b981' if pnl >= 0 else '#ef4444'
    
    # 进度条
    bar_c = '#10b981' if prog >= 100 else '#f59e0b' if prog >= 50 else '#3b82f6'
    
    # 止损标记
    stop = ''
    if p.get('stop_triggered'):
        stop = '<span style="color:#ef4444;font-size:10px;font-weight:700">STOP</span>'
    elif code == 'NXTC' and pnl <= -10:
        stop = '<span style="color:#f59e0b;font-size:10px">WATCH</span>'
    
    return f'''<tr>
<td style="font-weight:600;font-size:13px">{code}</td>
<td style="font-size:12px;color:#999">${p.get('current_price',0):.2f}</td>
<td style="color:{pnl_c};font-weight:600">{pnl:+.1f}%</td>
<td style="font-size:11px;color:#666">{held_days}/{hold_days}d</td>
<td><div style="width:60px;height:3px;background:#1a1a2e;border-radius:2px"><div style="width:{prog:.0f}%;height:100%;background:{bar_c};border-radius:2px"></div></div></td>
{model_cell}
<td>{stop}</td>
</tr>'''

# === 推荐行 ===
def pick_row(i, pk, hold_days):
    sig = pk.get('signal','')
    if sig == '🟢🟢': sig_c = '#10b981'
    elif sig == '🟢': sig_c = '#10b981'
    elif sig == '🟡': sig_c = '#f59e0b'
    else: sig_c = '#ef4444'
    
    return f'''<tr>
<td style="color:#666;font-size:11px">{i+1}</td>
<td style="font-weight:600;font-size:13px">{pk['ticker']}</td>
<td style="font-size:12px">${pk['price']:.2f}</td>
<td style="font-size:12px;color:{sig_c};font-weight:600">{pk['pred_rank']:.3f}</td>
<td style="font-size:11px;color:{sig_c}">{sig}</td>
<td style="font-size:11px;color:#666">{hold_days}d</td>
</tr>'''

shield_rows = ''.join(pos_row(p) for p in shield_pos)
arrow_rows = ''.join(pos_row(p) for p in arrow_pos)
v6_rows = ''.join(pick_row(i, pk, 20) for i, pk in enumerate(v6.get('picks',[])[:10]))
v11_rows = ''.join(pick_row(i, pk, 5) for i, pk in enumerate(v11.get('picks',[])[:10]))

# 统计
total_val = sum(p.get('current_price',0)*p.get('qty',0) for p in positions)
total_pnl = sum(p.get('pnl_usd',0) for p in positions)
total_pnl_pct = (total_pnl / (total_val - total_pnl) * 100) if total_val > total_pnl else 0

vix = futu.get('vix')
vix_val = f'{vix:.1f}' if vix else '--'
vix_c = '#10b981' if vix and vix < 20 else '#f59e0b' if vix and vix < 30 else '#ef4444'

pnl_c = '#10b981' if total_pnl >= 0 else '#ef4444'

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hermes Trading</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text','Helvetica Neue',sans-serif;background:#0a0a0f;color:#e5e5e5;font-size:14px;line-height:1.5}}
.wrap{{max-width:1400px;margin:0 auto;padding:16px}}
.header{{display:flex;justify-content:space-between;align-items:baseline;padding:12px 0;border-bottom:1px solid #1a1a2e}}
.header h1{{font-size:18px;font-weight:600;letter-spacing:-0.5px;color:#fff}}
.header .meta{{font-size:11px;color:#666}}
.metrics{{display:flex;gap:24px;padding:16px 0;border-bottom:1px solid #1a1a2e}}
.metric{{text-align:left}}
.metric .label{{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:1px}}
.metric .value{{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}}
.section{{padding:16px 0}}
.section-title{{font-size:13px;font-weight:600;color:#999;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #1a1a2e}}
table{{width:100%;border-collapse:collapse}}
th{{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:0.5px;text-align:left;padding:4px 8px;border-bottom:1px solid #1a1a2e;font-weight:500}}
td{{padding:6px 8px;border-bottom:1px solid #111;font-variant-numeric:tabular-nums}}
tr:hover{{background:#111}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media(max-width:768px){{.grid{{grid-template-columns:1fr}}}}
.footer{{text-align:center;font-size:10px;color:#333;padding:16px 0;border-top:1px solid #1a1a2e;margin-top:24px}}
</style>
</head>
<body>
<div class="wrap">

<div class="header">
  <h1>Hermes Trading</h1>
  <span class="meta">{now}</span>
</div>

<div class="metrics">
  <div class="metric">
    <div class="label">Portfolio</div>
    <div class="value" style="color:#fff">${total_val:,.0f}</div>
  </div>
  <div class="metric">
    <div class="label">P&L</div>
    <div class="value" style="color:{pnl_c}">{total_pnl:+,.0f} ({total_pnl_pct:+.1f}%)</div>
  </div>
  <div class="metric">
    <div class="label">VIX</div>
    <div class="value" style="color:{vix_c}">{vix_val}</div>
  </div>
  <div class="metric">
    <div class="label">Positions</div>
    <div class="value">{len(positions)}</div>
  </div>
</div>

<div class="grid">
  <div>
    <div class="section">
      <div class="section-title">Shield / Positions</div>
      <table>
        <tr><th>Code</th><th>Price</th><th>P&L</th><th>Days</th><th>Progress</th><th>Rank</th><th></th></tr>
        {shield_rows}
      </table>
    </div>
    <div class="section">
      <div class="section-title">Arrow / Positions</div>
      <table>
        <tr><th>Code</th><th>Price</th><th>P&L</th><th>Days</th><th>Progress</th><th>Rank</th><th></th></tr>
        {arrow_rows}
      </table>
    </div>
  </div>
  <div>
    <div class="section">
      <div class="section-title">Shield / Top 10</div>
      <table>
        <tr><th>#</th><th>Code</th><th>Price</th><th>Score</th><th>Signal</th><th>Hold</th></tr>
        {v6_rows}
      </table>
    </div>
    <div class="section">
      <div class="section-title">Arrow / Top 10</div>
      <table>
        <tr><th>#</th><th>Code</th><th>Price</th><th>Score</th><th>Signal</th><th>Hold</th></tr>
        {v11_rows}
      </table>
    </div>
  </div>
</div>

<div class="footer">
  Hermes Trading System v4.0 &middot; Shield V6 + Arrow V11 &middot; Three-Layer Filter
</div>

</div>
</body>
</html>'''

out = os.path.join(ROOT, 'dashboard.html')
with open(out, 'w') as f:
    f.write(html)
print(f'Dashboard v4: {len(html):,} bytes')
print(f'  Positions: {len(positions)} | Shield: {len(shield_pos)} | Arrow: {len(arrow_pos)}')
print(f'  Portfolio: ${total_val:,.0f} | P&L: {total_pnl:+,.0f}')
