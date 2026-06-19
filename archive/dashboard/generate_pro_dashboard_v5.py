#!/usr/bin/env python3
"""Hermes Trading Dashboard v5 — Ethereal Glass Edition
Design: OLED黑底 + 玻璃态卡片 + Bento Grid布局 + 微交互动效
灵感: Linear.app / Bloomberg Terminal / Apple Finance
"""
import json, os, math
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

held_map = {}
for h in held.get('shield', []) + held.get('arrow', []):
    held_map[h['sym']] = h

now = datetime.now().strftime('%Y-%m-%d %H:%M')

# Stats
total_val = sum(p.get('current_price',0)*p.get('qty',0) for p in positions)
total_pnl = sum(p.get('pnl_usd',0) for p in positions)
total_pnl_pct = (total_pnl / (total_val - total_pnl) * 100) if total_val > total_pnl else 0

vix = futu.get('vix')
vix_val = f'{vix:.1f}' if vix else '--'
if vix and vix < 15: vix_status, vix_c = 'LOW', '#10b981'
elif vix and vix < 20: vix_status, vix_c = 'CALM', '#10b981'
elif vix and vix < 25: vix_status, vix_c = 'ELEVATED', '#f59e0b'
elif vix and vix < 30: vix_status, vix_c = 'HIGH', '#f59e0b'
else: vix_status, vix_c = 'EXTREME', '#ef4444'

pnl_c = '#10b981' if total_pnl >= 0 else '#ef4444'

# === Helpers ===
def signal_color(sig):
    if sig == '🟢🟢': return '#10b981'
    elif sig == '🟢': return '#34d399'
    elif sig == '🟡': return '#f59e0b'
    return '#ef4444'

def signal_label(sig):
    if sig == '🟢🟢': return 'STRONG'
    elif sig == '🟢': return 'BUY'
    elif sig == '🟡': return 'WATCH'
    return 'SKIP'

def pct_class(pct):
    if pct <= 5: return '#10b981', f'Top {pct:.0f}%'
    elif pct <= 10: return '#34d399', f'Top {pct:.0f}%'
    elif pct <= 20: return '#f59e0b', f'Top {pct:.0f}%'
    elif pct <= 40: return '#6b7280', f'Top {pct:.0f}%'
    return '#ef4444', f'Top {pct:.0f}%'

# === Position Card (Bento Item) ===
def pos_card(p):
    code = p.get('code','')
    name = p.get('name','')
    pnl = p.get('pnl_pct', 0)
    pnl_usd = p.get('pnl_usd', 0)
    held_days = p.get('days_held', 0)
    hold_days = p.get('hold_days', 20)
    prog = min(100, held_days/hold_days*100) if hold_days > 0 else 0
    cost = p.get('cost_price', 0)
    cur = p.get('current_price', 0)
    qty = p.get('qty', 0)
    is_shield = cost > 10
    
    # Model ranking
    hs = held_map.get(code, {})
    if hs:
        rank = hs.get('rank', 0)
        total = hs.get('total', 1)
        pct = rank / total * 100 if total > 0 else 100
        rc, rt = pct_class(pct)
        rank_html = f'<span style="color:{rc};font-size:11px;font-weight:600">{rt}</span>'
    else:
        rank_html = '<span style="color:#333;font-size:11px">--</span>'
    
    pnl_c = '#10b981' if pnl >= 0 else '#ef4444'
    bar_c = '#10b981' if prog >= 100 else '#f59e0b' if prog >= 50 else '#6366f1'
    
    # Stop warning
    stop_html = ''
    if p.get('stop_triggered'):
        stop_html = '<div class="stop-badge">STOP TRIGGERED</div>'
    elif pnl <= -8:
        stop_html = '<div class="warn-badge">APPROACHING STOP</div>'
    
    model_tag = 'SHIELD' if is_shield else 'ARROW'
    model_c = '#6366f1' if is_shield else '#10b981'
    
    return f'''
    <div class="pos-card" data-model="{'shield' if is_shield else 'arrow'}">
      <div class="pos-head">
        <div>
          <span class="pos-code">{code}</span>
          <span class="pos-model" style="color:{model_c};border-color:{model_c}30;background:{model_c}10">{model_tag}</span>
        </div>
        {rank_html}
      </div>
      <div class="pos-name">{name}</div>
      <div class="pos-metrics">
        <div class="pos-pnl" style="color:{pnl_c}">
          <span class="pnl-sign">{'+' if pnl >= 0 else ''}{pnl:.1f}%</span>
          <span class="pnl-usd">{'+' if pnl_usd >= 0 else ''}${abs(pnl_usd):,.0f}</span>
        </div>
        <div class="pos-price">
          <span class="price-cur">${cur:.2f}</span>
          <span class="price-cost">from ${cost:.2f}</span>
        </div>
      </div>
      <div class="pos-progress">
        <div class="prog-label">{held_days}/{hold_days}d</div>
        <div class="prog-bar"><div class="prog-fill" style="width:{prog:.0f}%;background:{bar_c}"></div></div>
      </div>
      {stop_html}
    </div>'''

# === Recommendation Row ===
def pick_card(i, pk, hold_days, min_score, max_score):
    sig = pk.get('signal','')
    sc = signal_color(sig)
    sl = signal_label(sig)
    ticker = pk['ticker']
    price = pk['price']
    score = pk['pred_rank']
    
    # Relative position within visible picks (min-max normalization)
    score_range = max_score - min_score if max_score > min_score else 0.01
    rel_pct = (score - min_score) / score_range * 100
    rel_pct = max(5, min(100, rel_pct))  # clamp for visual
    
    # Signal quality description
    if sig == '🟢🟢': quality = 'Top 5%'
    elif sig == '🟢': quality = 'Top 10%'
    elif sig == '🟡': quality = 'Top 20%'
    else: quality = 'Below'
    qc = sc
    
    bar_w = rel_pct
    
    return f'''
    <div class="pick-row">
      <div class="pick-rank">{i+1}</div>
      <div class="pick-info">
        <div class="pick-ticker">{ticker}</div>
        <div class="pick-price">${price:.2f}</div>
      </div>
      <div class="pick-score-bar">
        <div class="score-track"><div class="score-fill" style="width:{bar_w:.0f}%;background:{sc}"></div></div>
        <span class="score-val" style="color:{sc}">{score:.3f}</span>
      </div>
      <div class="pick-signal" style="color:{sc};border-color:{sc}30;background:{sc}10">{sl}</div>
      <div class="pick-pct" style="color:{qc}">{quality}</div>
    </div>'''

# Generate cards
shield_cards = ''.join(pos_card(p) for p in shield_pos)
arrow_cards = ''.join(pos_card(p) for p in arrow_pos)
# Score ranges for relative bar positioning
v6_scores = [pk['pred_rank'] for pk in v6.get('picks',[])[:10]]
v11_scores = [pk['pred_rank'] for pk in v11.get('picks',[])[:10]]
v6_min, v6_max = (min(v6_scores), max(v6_scores)) if v6_scores else (0, 1)
v11_min, v11_max = (min(v11_scores), max(v11_scores)) if v11_scores else (0, 1)

v6_cards = ''.join(pick_card(i, pk, 20, v6_min, v6_max) for i, pk in enumerate(v6.get('picks',[])[:10]))
v11_cards = ''.join(pick_card(i, pk, 5, v11_min, v11_max) for i, pk in enumerate(v11.get('picks',[])[:10]))

# Active signals count
v6_strong = sum(1 for pk in v6.get('picks',[])[:10] if pk.get('signal') == '🟢🟢')
v11_strong = sum(1 for pk in v11.get('picks',[])[:10] if pk.get('signal') == '🟢🟢')
active_signals = v6_strong + v11_strong

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#050508">
<title>Hermes Trading</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
  --bg: #050508;
  --bg-card: rgba(255,255,255,0.03);
  --bg-card-hover: rgba(255,255,255,0.05);
  --bg-elevated: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.06);
  --border-light: rgba(255,255,255,0.03);
  --text: #e8e8ed;
  --text-dim: #6b6b76;
  --text-muted: #3d3d47;
  --green: #10b981;
  --green-glow: rgba(16,185,129,0.15);
  --red: #ef4444;
  --red-glow: rgba(239,68,68,0.15);
  --amber: #f59e0b;
  --indigo: #6366f1;
  --indigo-glow: rgba(99,102,241,0.15);
  --radius: 16px;
  --radius-sm: 10px;
  --radius-xs: 6px;
}}

* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Inter', -apple-system, 'SF Pro Text', 'Helvetica Neue', sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}}

/* Ambient background glow */
body::before {{
  content: '';
  position: fixed;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background:
    radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.04) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 80%, rgba(16,185,129,0.03) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
}}

.wrap {{
  max-width: 1440px;
  margin: 0 auto;
  padding: 24px 32px;
  position: relative;
  z-index: 1;
}}

/* === HEADER === */
.header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 0 24px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}}
.header-left {{ display: flex; align-items: center; gap: 16px; }}
.logo {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.5px;
  color: #fff;
}}
.logo-dot {{ color: var(--indigo); }}
.header-badge {{
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.5px;
  padding: 3px 10px;
  border-radius: 20px;
  background: var(--indigo-glow);
  color: var(--indigo);
  border: 1px solid rgba(99,102,241,0.2);
}}
.header-right {{ display: flex; align-items: center; gap: 20px; }}
.timestamp {{ font-size: 12px; color: var(--text-dim); font-family: 'JetBrains Mono', monospace; }}
.live-dot {{
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 8px var(--green-glow);
  animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.4; }}
}}

/* === METRICS BAR === */
.metrics {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  margin-bottom: 24px;
  border: 1px solid var(--border);
}}
.metric {{
  background: var(--bg);
  padding: 20px 24px;
  position: relative;
  transition: background 0.3s ease;
}}
.metric:hover {{ background: rgba(255,255,255,0.02); }}
.metric-label {{
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  color: var(--text-dim);
  margin-bottom: 8px;
}}
.metric-value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.5px;
}}
.metric-sub {{
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
  font-family: 'JetBrains Mono', monospace;
}}

/* === BENTO GRID === */
.bento {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto;
  gap: 16px;
  margin-bottom: 24px;
}}
.bento-item {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  transition: all 0.4s cubic-bezier(0.32, 0.72, 0, 1);
  position: relative;
  overflow: hidden;
}}
.bento-item::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}}
.bento-item:hover {{
  background: var(--bg-card-hover);
  border-color: rgba(255,255,255,0.1);
  transform: translateY(-1px);
}}
.bento-wide {{ grid-column: span 2; }}

/* Section header inside bento */
.section-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}}
.section-title {{
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--text-dim);
  display: flex;
  align-items: center;
  gap: 10px;
}}
.section-title .dot {{
  width: 8px; height: 8px;
  border-radius: 50%;
}}
.section-count {{
  font-size: 11px;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
}}

/* === POSITION CARDS GRID === */
.pos-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}}
.pos-card {{
  background: rgba(255,255,255,0.02);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 14px 16px;
  transition: all 0.3s cubic-bezier(0.32, 0.72, 0, 1);
}}
.pos-card:hover {{
  background: rgba(255,255,255,0.04);
  border-color: var(--border);
  transform: translateY(-2px);
}}
.pos-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}}
.pos-code {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 14px;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.3px;
}}
.pos-model {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid;
}}
.pos-name {{
  font-size: 11px;
  color: var(--text-dim);
  margin-bottom: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.pos-metrics {{
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  margin-bottom: 10px;
}}
.pos-pnl {{ text-align: left; }}
.pnl-sign {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 700;
  display: block;
  letter-spacing: -0.5px;
}}
.pnl-usd {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  opacity: 0.7;
  display: block;
}}
.pos-price {{ text-align: right; }}
.price-cur {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  color: var(--text);
  display: block;
}}
.price-cost {{
  font-size: 10px;
  color: var(--text-muted);
  display: block;
}}
.pos-progress {{ margin-top: 8px; }}
.prog-label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-muted);
  margin-bottom: 4px;
}}
.prog-bar {{
  height: 3px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
}}
.prog-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 0.8s cubic-bezier(0.32, 0.72, 0, 1);
}}
.stop-badge {{
  margin-top: 8px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--red);
  background: var(--red-glow);
  padding: 4px 10px;
  border-radius: 4px;
  text-align: center;
}}
.warn-badge {{
  margin-top: 8px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--amber);
  background: rgba(245,158,11,0.1);
  padding: 4px 10px;
  border-radius: 4px;
  text-align: center;
}}

/* === PICK ROWS === */
.pick-list {{ display: flex; flex-direction: column; gap: 2px; }}
.pick-row {{
  display: grid;
  grid-template-columns: 28px 100px 1fr 60px 80px;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  border-radius: var(--radius-xs);
  transition: background 0.2s ease;
}}
.pick-row:hover {{ background: rgba(255,255,255,0.03); }}
.pick-rank {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-muted);
  text-align: center;
}}
.pick-info {{ display: flex; flex-direction: column; }}
.pick-ticker {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.3px;
}}
.pick-price {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--text-dim);
}}
.pick-score-bar {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.score-track {{
  flex: 1;
  height: 4px;
  background: rgba(255,255,255,0.06);
  border-radius: 2px;
  overflow: hidden;
}}
.score-fill {{
  height: 100%;
  border-radius: 2px;
  transition: width 1s cubic-bezier(0.32, 0.72, 0, 1);
}}
.score-val {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  min-width: 45px;
  text-align: right;
}}
.pick-signal {{
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 1px;
  padding: 3px 10px;
  border-radius: 4px;
  border: 1px solid;
  text-align: center;
}}
.pick-pct {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  text-align: right;
}}

/* === SIGNAL LEGEND === */
.legend {{
  display: flex;
  gap: 20px;
  justify-content: center;
  padding: 16px;
  border-top: 1px solid var(--border);
  margin-top: 8px;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 10px;
  color: var(--text-dim);
  letter-spacing: 0.5px;
}}
.legend-dot {{
  width: 8px; height: 8px;
  border-radius: 2px;
}}

/* === FOOTER === */
.footer {{
  text-align: center;
  padding: 20px 0;
  border-top: 1px solid var(--border);
  margin-top: 8px;
}}
.footer-text {{
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.5px;
}}
.footer-text a {{ color: var(--text-dim); text-decoration: none; }}

/* === EMPTY STATE === */
.empty {{
  text-align: center;
  padding: 32px;
  color: var(--text-muted);
  font-size: 12px;
}}

/* === ANIMATIONS === */
@keyframes fadeUp {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}
.bento-item {{
  animation: fadeUp 0.6s cubic-bezier(0.32, 0.72, 0, 1) both;
}}
.bento-item:nth-child(1) {{ animation-delay: 0.05s; }}
.bento-item:nth-child(2) {{ animation-delay: 0.1s; }}
.bento-item:nth-child(3) {{ animation-delay: 0.15s; }}
.bento-item:nth-child(4) {{ animation-delay: 0.2s; }}

.pos-card {{
  animation: fadeUp 0.5s cubic-bezier(0.32, 0.72, 0, 1) both;
}}
.pos-card:nth-child(1) {{ animation-delay: 0.1s; }}
.pos-card:nth-child(2) {{ animation-delay: 0.15s; }}
.pos-card:nth-child(3) {{ animation-delay: 0.2s; }}
.pos-card:nth-child(4) {{ animation-delay: 0.25s; }}
.pos-card:nth-child(5) {{ animation-delay: 0.3s; }}
.pos-card:nth-child(6) {{ animation-delay: 0.35s; }}

.pick-row {{
  animation: fadeUp 0.4s cubic-bezier(0.32, 0.72, 0, 1) both;
}}
.pick-row:nth-child(1) {{ animation-delay: 0.05s; }}
.pick-row:nth-child(2) {{ animation-delay: 0.08s; }}
.pick-row:nth-child(3) {{ animation-delay: 0.11s; }}
.pick-row:nth-child(4) {{ animation-delay: 0.14s; }}
.pick-row:nth-child(5) {{ animation-delay: 0.17s; }}

/* === RESPONSIVE === */
@media (max-width: 900px) {{
  .wrap {{ padding: 16px; }}
  .bento {{ grid-template-columns: 1fr; }}
  .bento-wide {{ grid-column: span 1; }}
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
  .pos-grid {{ grid-template-columns: 1fr 1fr; }}
  .pick-row {{ grid-template-columns: 24px 80px 1fr 50px 60px; gap: 8px; }}
}}
@media (max-width: 480px) {{
  .pos-grid {{ grid-template-columns: 1fr; }}
  .pick-row {{ grid-template-columns: 24px 70px 1fr 44px; }}
  .pick-pct {{ display: none; }}
  .metric-value {{ font-size: 18px; }}
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.1); border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.2); }}
</style>
</head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <div class="header">
    <div class="header-left">
      <div class="logo">HERMES<span class="logo-dot">.</span></div>
      <div class="header-badge">SHIELD V6 + ARROW V11</div>
    </div>
    <div class="header-right">
      <div class="live-dot"></div>
      <div class="timestamp">{now}</div>
    </div>
  </div>

  <!-- METRICS -->
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">Portfolio</div>
      <div class="metric-value" style="color:#fff">${total_val:,.0f}</div>
      <div class="metric-sub">{len(positions)} positions</div>
    </div>
    <div class="metric">
      <div class="metric-label">P&L</div>
      <div class="metric-value" style="color:{pnl_c}">{'+' if total_pnl >= 0 else ''}{total_pnl:,.0f}</div>
      <div class="metric-sub">{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct:.1f}% total</div>
    </div>
    <div class="metric">
      <div class="metric-label">VIX</div>
      <div class="metric-value" style="color:{vix_c}">{vix_val}</div>
      <div class="metric-sub">{vix_status}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Active Signals</div>
      <div class="metric-value" style="color:{'#10b981' if active_signals > 0 else '#6b6b76'}">{active_signals}</div>
      <div class="metric-sub">strong buys today</div>
    </div>
  </div>

  <!-- BENTO GRID -->
  <div class="bento">

    <!-- SHIELD POSITIONS -->
    <div class="bento-item">
      <div class="section-head">
        <div class="section-title">
          <span class="dot" style="background:var(--indigo)"></span>
          Shield Holdings
        </div>
        <div class="section-count">{len(shield_pos)} active</div>
      </div>
      <div class="pos-grid">
        {shield_cards if shield_cards else '<div class="empty">No shield positions</div>'}
      </div>
    </div>

    <!-- ARROW POSITIONS -->
    <div class="bento-item">
      <div class="section-head">
        <div class="section-title">
          <span class="dot" style="background:var(--green)"></span>
          Arrow Holdings
        </div>
        <div class="section-count">{len(arrow_pos)} active</div>
      </div>
      <div class="pos-grid">
        {arrow_cards if arrow_cards else '<div class="empty">No arrow positions</div>'}
      </div>
    </div>

    <!-- SHIELD TOP PICKS -->
    <div class="bento-item">
      <div class="section-head">
        <div class="section-title">
          <span class="dot" style="background:var(--indigo)"></span>
          Shield V6 — Top Picks
        </div>
        <div class="section-count">{v6.get('total', 0)} universe</div>
      </div>
      <div class="pick-list">
        {v6_cards if v6_cards else '<div class="empty">No picks today</div>'}
      </div>
    </div>

    <!-- ARROW TOP PICKS -->
    <div class="bento-item">
      <div class="section-head">
        <div class="section-title">
          <span class="dot" style="background:var(--green)"></span>
          Arrow V11 — Top Picks
        </div>
        <div class="section-count">{v11.get('total', 0)} universe</div>
      </div>
      <div class="pick-list">
        {v11_cards if v11_cards else '<div class="empty">No picks today</div>'}
      </div>
    </div>

  </div>

  <!-- LEGEND -->
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:var(--green)"></div>STRONG ≥Top5%</div>
    <div class="legend-item"><div class="legend-dot" style="background:#34d399"></div>BUY ≥Top10%</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--amber)"></div>WATCH ≥Top20%</div>
    <div class="legend-item"><div class="legend-dot" style="background:var(--red)"></div>SKIP</div>
  </div>

  <!-- FOOTER -->
  <div class="footer">
    <div class="footer-text">
      Hermes Trading System · Three-Layer Signal Filter · XGBoost Rank Models
    </div>
  </div>

</div>
</body>
</html>'''

out = os.path.join(ROOT, 'dashboard.html')
with open(out, 'w') as f:
    f.write(html)
print(f'Dashboard v5 Ethereal Glass: {len(html):,} bytes')
print(f'  Positions: {len(positions)} | Shield: {len(shield_pos)} | Arrow: {len(arrow_pos)}')
print(f'  Portfolio: ${total_val:,.0f} | P&L: {total_pnl:+,.0f}')
