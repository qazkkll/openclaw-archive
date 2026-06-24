#!/usr/bin/env python3
"""Hermes Trading Dashboard v7 — Complete Redesign
Fixes: auto-refresh, bilingual, chart axes, signal differentiation, real tracking
"""
import json, os, math
from datetime import datetime
from collections import defaultdict

ROOT = '/home/hermes/.hermes/openclaw-archive'
def load(p):
    try:
        with open(os.path.join(ROOT, p)) as f: return json.load(f)
    except: return {}

futu = load('output/futu_positions.json')
v6 = load('output/v6_latest.json')
v11 = load('output/v11_latest.json')
held = load('output/held_scores.json')
tracking = load('output/tracking_history.json')
shield_meta = load('models/us/blueshield_v8_meta.json')

# CN Redwood data
cn_signal = load('signals/cn/latest_xgb.json')
cn_prod = load('models/cn/production.json')
arrow_meta = load('models/us/arrow_v12_meta.json')

positions = futu.get('positions', [])
shield_pos = [p for p in positions if p.get('cost_price',0) > 10]
arrow_pos = [p for p in positions if p.get('cost_price',0) <= 10]
held_map = {h['sym']: h for h in held.get('shield', []) + held.get('arrow', [])}

now = datetime.now()
now_str = now.strftime('%Y-%m-%d %H:%M')

total_val = sum(p.get('current_price',0)*p.get('qty',0) for p in positions)
total_pnl = sum(p.get('pnl_usd',0) for p in positions)
total_pnl_pct = (total_pnl / (total_val - total_pnl) * 100) if total_val > total_pnl else 0

vix = futu.get('vix')
vix_val = f'{vix:.1f}' if vix else '--'
if vix and vix < 15: vix_status, vix_c = 'LOW 低波动', '#10b981'
elif vix and vix < 20: vix_status, vix_c = 'CALM 平静', '#10b981'
elif vix and vix < 25: vix_status, vix_c = 'ELEVATED 偏高', '#f59e0b'
elif vix and vix < 30: vix_status, vix_c = 'HIGH 高', '#f59e0b'
else: vix_status, vix_c = 'EXTREME 极端', '#ef4444'

pnl_c = '#10b981' if total_pnl >= 0 else '#ef4444'
win_count = sum(1 for p in positions if p.get('pnl_pct',0) > 0)
loss_count = len(positions) - win_count

# ============================================================
# SIGNAL SYSTEM — clear visual differentiation
# ============================================================
def signal_badge(sig, size='normal'):
    """Render signal as colored badge with text."""
    if sig == '🟢🟢':
        return f'<span class="sig sig-strong"><span class="sig-dot" style="background:#10b981"></span><span class="sig-dot" style="background:#10b981"></span>STRONG 强</span>'
    elif sig == '🟢':
        return f'<span class="sig sig-buy"><span class="sig-dot" style="background:#34d399"></span>BUY 买</span>'
    elif sig == '🟡':
        return f'<span class="sig sig-watch"><span class="sig-dot" style="background:#f59e0b"></span>WATCH 看</span>'
    else:
        return f'<span class="sig sig-skip"><span class="sig-dot" style="background:#ef4444"></span>SKIP 跳</span>'

def signal_color(sig):
    if sig == '🟢🟢': return '#10b981'
    elif sig == '🟢': return '#34d399'
    elif sig == '🟡': return '#f59e0b'
    return '#ef4444'

# ============================================================
# CHART HELPERS with proper axes
# ============================================================
def svg_equity_curve_curved(equity, color='#10b981', w=600, h=200, label='', y_label='Value'):
    """SVG equity curve with Y-axis labels and grid lines."""
    if not equity or len(equity) < 2:
        return '<div class="empty">No data</div>'
    
    mn, mx = min(equity), max(equity)
    rng = mx - mn if mx > mn else 1
    pad_l, pad_r, pad_t, pad_b = 50, 20, 20, 30
    chart_w, chart_h = w - pad_l - pad_r, h - pad_t - pad_b
    
    # Y-axis grid lines (5 levels)
    grid_lines = ''
    y_labels = ''
    for i in range(5):
        y = pad_t + (i / 4) * chart_h
        val = mx - (i / 4) * rng
        grid_lines += f'<line x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>'
        y_labels += f'<text x="{pad_l-6}" y="{y+4}" text-anchor="end" class="axis-label">{val:.0f}</text>'
    
    # Data points
    points = []
    for i, v in enumerate(equity):
        x = pad_l + (i / (len(equity)-1)) * chart_w
        y = pad_t + (1 - (v - mn) / rng) * chart_h
        points.append(f'{x:.1f},{y:.1f}')
    
    polyline = ' '.join(points)
    fill_poly = f'{pad_l},{pad_t+chart_h} {polyline} {pad_l+chart_w},{pad_t+chart_h}'
    
    # Start/End values
    ret_pct = (equity[-1] / equity[0] - 1) * 100
    ret_c = '#10b981' if ret_pct >= 0 else '#ef4444'
    
    return f'''<svg viewBox="0 0 {w} {h}" class="chart-svg">
      <defs><linearGradient id="g_{label}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="{color}" stop-opacity="0.2"/><stop offset="100%" stop-color="{color}" stop-opacity="0"/>
      </linearGradient></defs>
      {grid_lines}{y_labels}
      <polygon points="{fill_poly}" fill="url(#g_{label})"/>
      <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
      <text x="{pad_l}" y="{h-4}" class="axis-label">Start {equity[0]:.0f}</text>
      <text x="{w-pad_r}" y="{h-4}" class="axis-label" text-anchor="end" style="fill:{ret_c}">{equity[-1]:.0f} ({ret_pct:+.1f}%)</text>
    </svg>'''

def svg_bar_horizontal(data, labels, colors, w=500, h=250, title=''):
    """Horizontal bar chart with value labels."""
    if not data: return ''
    n = len(data)
    max_val = max(data) if data else 1
    bar_h = min(18, (h - 20) / n - 3)
    pad_l = 100
    
    bars = ''
    for i, (val, label, color) in enumerate(zip(data, labels, colors)):
        y = i * (bar_h + 3) + 10
        bar_w = (val / max_val) * (w - pad_l - 60) if max_val > 0 else 0
        bars += f'''<g>
          <text x="{pad_l-6}" y="{y+bar_h*0.75}" text-anchor="end" class="axis-label" style="font-size:10px">{label}</text>
          <rect x="{pad_l}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="2" fill="{color}" opacity="0.8"/>
          <text x="{pad_l+bar_w+4}" y="{y+bar_h*0.75}" class="axis-label" style="font-size:10px;fill:#fff">{val:.1f}</text>
        </g>'''
    return f'<svg viewBox="0 0 {w} {n*(bar_h+3)+20}" class="chart-svg">{bars}</svg>'

def svg_cumulative_by_signal(recs, w=600, h=220):
    """Cumulative return chart grouped by signal type."""
    if not recs: return ''
    
    # Group by signal, compute cumulative returns
    by_signal = defaultdict(list)
    for r in sorted(recs, key=lambda x: x['entry_date']):
        by_signal[r['signal']].append(r['return_pct'])
    
    signals = ['🟢🟢', '🟢', '🟡']
    colors = {'🟢🟢': '#10b981', '🟢': '#34d399', '🟡': '#f59e0b'}
    labels_map = {'🟢🟢': 'STRONG 强', '🟢': 'BUY 买', '🟡': 'WATCH 看'}
    
    pad_l, pad_r, pad_t, pad_b = 50, 120, 20, 30
    chart_w, chart_h = w - pad_l - pad_r, h - pad_t - pad_b
    
    # Find global min/max for Y axis
    all_cum = []
    for sig in signals:
        rets = by_signal.get(sig, [])
        if rets:
            cum = [0]
            for r in rets:
                cum.append(cum[-1] + r)
            all_cum.extend(cum)
    
    if not all_cum:
        return ''
    
    mn, mx = min(all_cum), max(all_cum)
    rng = mx - mn if mx > mn else 1
    
    # Grid
    grid = ''
    for i in range(5):
        y = pad_t + (i / 4) * chart_h
        val = mx - (i / 4) * rng
        grid += f'<line x1="{pad_l}" y1="{y}" x2="{w-pad_r}" y2="{y}" stroke="rgba(255,255,255,0.04)"/>'
        grid += f'<text x="{pad_l-6}" y="{y+4}" text-anchor="end" class="axis-label">{val:+.1f}%</text>'
    
    # Zero line
    zero_y = pad_t + (1 - (0 - mn) / rng) * chart_h
    grid += f'<line x1="{pad_l}" y1="{zero_y}" x2="{w-pad_r}" y2="{zero_y}" stroke="rgba(255,255,255,0.15)" stroke-dasharray="4"/>'
    
    # Lines
    lines = ''
    legend = ''
    max_len = max(len(by_signal.get(s, [])) for s in signals) if by_signal else 1
    
    for idx, sig in enumerate(signals):
        rets = by_signal.get(sig, [])
        if not rets:
            continue
        cum = [0]
        for r in rets:
            cum.append(cum[-1] + r)
        
        pts = []
        for i, v in enumerate(cum):
            x = pad_l + (i / max(len(cum)-1, 1)) * chart_w
            y = pad_t + (1 - (v - mn) / rng) * chart_h
            pts.append(f'{x:.1f},{y:.1f}')
        
        c = colors[sig]
        lines += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round"/>'
        
        # Legend
        ly = pad_t + idx * 20
        legend += f'<line x1="{w-pad_r+10}" y1="{ly}" x2="{w-pad_r+25}" y2="{ly}" stroke="{c}" stroke-width="2"/>'
        legend += f'<text x="{w-pad_r+30}" y="{ly+4}" class="axis-label" style="font-size:10px">{labels_map[sig]} ({len(rets)})</text>'
        legend += f'<text x="{w-pad_r+30}" y="{ly+16}" class="axis-label" style="font-size:9px;fill:{c}">{cum[-1]:+.1f}%</text>'
    
    return f'<svg viewBox="0 0 {w} {h}" class="chart-svg">{grid}{lines}{legend}</svg>'

# ============================================================
# TAB 1: TRADING
# ============================================================
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
    is_shield = cost > 10
    
    hs = held_map.get(code, {})
    if hs:
        rank = hs.get('rank', 0)
        total = hs.get('total', 1)
        pct = rank / total * 100 if total > 0 else 100
        if pct <= 5: rc, rt = '#10b981', f'Top {pct:.0f}%'
        elif pct <= 10: rc, rt = '#34d399', f'Top {pct:.0f}%'
        elif pct <= 20: rc, rt = '#f59e0b', f'Top {pct:.0f}%'
        elif pct <= 40: rc, rt = '#6b7280', f'Top {pct:.0f}%'
        else: rc, rt = '#ef4444', f'Top {pct:.0f}%'
        rank_html = f'<span style="color:{rc};font-size:10px;font-weight:700">{rt}</span>'
    else:
        rank_html = ''
    
    pnl_c = '#10b981' if pnl >= 0 else '#ef4444'
    bar_c = '#10b981' if prog >= 100 else '#f59e0b' if prog >= 50 else '#6366f1'
    
    stop_html = ''
    if p.get('stop_triggered'):
        stop_html = '<div class="stop-badge">STOP 止损</div>'
    elif pnl <= -8:
        stop_html = '<div class="warn-badge">WATCH 关注</div>'
    
    model_tag = 'SHIELD 盾' if is_shield else 'ARROW 箭'
    model_c = '#6366f1' if is_shield else '#10b981'
    
    days_left = max(0, hold_days - held_days)
    days_text = f'{days_left}d left 剩{days_left}天' if days_left > 0 else 'EXPIRED 到期'
    
    # Daily change (today_pl)
    today_pl = p.get('today_pl', 0)
    today_c = '#10b981' if today_pl >= 0 else '#ef4444'
    today_text = f'{"+" if today_pl >= 0 else ""}{today_pl:.1f}%' if today_pl else '--'
    
    return f'''<div class="pos-card">
      <div class="pos-head">
        <div><span class="pos-code">{code}</span>
        <span class="pos-model" style="color:{model_c};border-color:{model_c}30;background:{model_c}10">{model_tag}</span></div>
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
          <span class="price-cost">Entry 入${cost:.2f}</span>
        </div>
      </div>
      <div class="pos-progress">
        <div class="prog-label">{held_days}/{hold_days}d · {days_text}</div>
        <div class="prog-bar"><div class="prog-fill" style="width:{prog:.0f}%;background:{bar_c}"></div></div>
      </div>
      {stop_html}
    </div>'''

def pick_row(i, pk, hold_days, total_picks):
    """Pick row with relative signal ranking within the day's picks."""
    score = pk['pred_rank']
    # Relative signal within visible picks
    rel_pct = (i + 1) / total_picks  # 0 = best
    if rel_pct <= 0.30:
        sig, sc = '🟢🟢', '#10b981'
    elif rel_pct <= 0.70:
        sig, sc = '🟢', '#34d399'
    else:
        sig, sc = '🟡', '#f59e0b'
    
    bar_w = max(5, min(100, (score - 0.50) / 0.20 * 100))
    
    return f'''<div class="pick-row">
      <div class="pick-rank">{i+1}</div>
      <div class="pick-info"><div class="pick-ticker">{pk['ticker']}</div><div class="pick-price">${pk['price']:.2f}</div></div>
      <div class="pick-score-bar"><div class="score-track"><div class="score-fill" style="width:{bar_w:.0f}%;background:{sc}"></div></div>
      <span class="score-val" style="color:{sc}">{score:.3f}</span></div>
      {signal_badge(sig)}
      <div class="pick-hold">{hold_days}d</div>
    </div>'''

shield_cards = ''.join(pos_card(p) for p in shield_pos)
arrow_cards = ''.join(pos_card(p) for p in arrow_pos)

# ============================================================
# INSIGHT VISUALIZATIONS
# ============================================================
# 1. P&L Calendar Heatmap
daily_pnl = tracking.get('daily_pnl', {})
cal_html = ''
if daily_pnl:
    # Group by week
    weeks = defaultdict(list)
    for d in sorted(daily_pnl.keys()):
        dt = datetime.strptime(d, '%Y-%m-%d')
        week_num = dt.isocalendar()[1]
        weeks[week_num].append((d, daily_pnl[d]))
    
    # Day headers
    cal_html += '<div class="cal-header"><span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span></div>'
    
    for week_num in sorted(weeks.keys()):
        days = weeks[week_num]
        row = '<div class="cal-row">'
        # Fill empty days at start of week
        first_day = datetime.strptime(days[0][0], '%Y-%m-%d').weekday()
        for _ in range(first_day):
            row += '<div class="cal-cell empty"></div>'
        
        for d, pnl_data in days:
            avg = pnl_data['avg']
            cnt = pnl_data['count']
            # Color intensity based on return
            if avg > 2: bg = '#10b981'; tc = '#fff'
            elif avg > 1: bg = '#10b981cc'; tc = '#fff'
            elif avg > 0: bg = '#10b98160'; tc = '#e8e8ed'
            elif avg > -1: bg = '#ef444440'; tc = '#e8e8ed'
            elif avg > -2: bg = '#ef444480'; tc = '#fff'
            else: bg = '#ef4444'; tc = '#fff'
            
            day_label = d[5:]  # MM-DD
            row += f'<div class="cal-cell" style="background:{bg}" title="{d}: {avg:+.2f}% ({cnt} picks)"><span class="cal-day">{day_label}</span><span class="cal-ret" style="color:{tc}">{avg:+.1f}%</span></div>'
        
        row += '</div>'
        cal_html += row

# 2. Position Waterfall Chart
waterfall_html = '<div class="wf-container">'
sorted_pos = sorted(positions, key=lambda p: p.get('pnl_pct', 0), reverse=True)
cum_pnl = 0
for p in sorted_pos:
    pnl = p.get('pnl_pct', 0)
    pnl_usd = p.get('pnl_usd', 0)
    code = p.get('code', '')
    color = '#10b981' if pnl >= 0 else '#ef4444'
    bar_w = min(100, abs(pnl) * 5)  # Scale: 20% = full width
    
    waterfall_html += f'''<div class="wf-row">
      <div class="wf-label">{code}</div>
      <div class="wf-bar-container">
        <div class="wf-bar" style="width:{bar_w:.0f}%;background:{color}"></div>
      </div>
      <div class="wf-val" style="color:{color}">{pnl:+.1f}%</div>
      <div class="wf-usd" style="color:{color}">{'+' if pnl_usd >= 0 else ''}${pnl_usd:,.0f}</div>
    </div>'''

# Total
total_c = '#10b981' if total_pnl >= 0 else '#ef4444'
waterfall_html += f'''<div class="wf-row wf-total">
  <div class="wf-label">TOTAL</div>
  <div class="wf-bar-container"><div class="wf-bar" style="width:60%;background:{total_c}"></div></div>
  <div class="wf-val" style="color:{total_c}">{total_pnl_pct:+.1f}%</div>
  <div class="wf-usd" style="color:{total_c}">{'+' if total_pnl >= 0 else ''}${total_pnl:,.0f}</div>
</div>'''
waterfall_html += '</div>'

# 3. Radar Chart: Model Comparison
sv = shield_meta.get('validation', {})
av = arrow_meta.get('validation', {})
# 5 dimensions: Sharpe, Annual Return, Win Rate, 1/MaxDD, Features
dims = ['Sharpe', 'Return 收益', 'Win Rate 胜率', 'Risk Mgmt 风控', 'Features 特征']
shield_vals = [
    sv.get('oos_sharpe', 0) / 3,  # Normalize: 0-1 (3 = max)
    sv.get('oos_annual_return', 0) / 50,  # 50% = max
    sv.get('oos_win_rate', 0) / 100,
    1 - abs(sv.get('oos_max_dd', 0)) / 20,  # Lower DD = better
    shield_meta.get('n_features', 0) / 50  # 50 features = max
]
arrow_vals = [
    av.get('oos_sharpe', av.get('wf_avg_net', 0)) / 3,
    av.get('oos_avg_net', av.get('wf_avg_net', 0)) / 10,  # 10% per trade = max
    av.get('oos_win_rate', 0) / 100,
    0.7,  # Arrow has -10% stop loss, good risk mgmt
    arrow_meta.get('n_features', 0) / 50
]
# Clamp to 0-1
shield_vals = [max(0, min(1, v)) for v in shield_vals]
arrow_vals = [max(0, min(1, v)) for v in arrow_vals]

# SVG Radar
cx, cy, r = 150, 130, 100
n_dims = len(dims)
radar_svg = f'<svg viewBox="0 0 300 260" class="chart-svg">'

# Grid circles
for i in range(1, 5):
    gr = r * i / 4
    radar_svg += f'<circle cx="{cx}" cy="{cy}" r="{gr}" fill="none" stroke="rgba(255,255,255,0.06)"/>'

# Grid lines and labels
for i in range(n_dims):
    angle = -90 + (360 / n_dims) * i
    rad = math.radians(angle)
    x2 = cx + r * math.cos(rad)
    y2 = cy + r * math.sin(rad)
    radar_svg += f'<line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" stroke="rgba(255,255,255,0.06)"/>'
    
    lx = cx + (r + 18) * math.cos(rad)
    ly = cy + (r + 18) * math.sin(rad)
    anchor = 'middle' if abs(math.cos(rad)) < 0.3 else ('start' if math.cos(rad) > 0 else 'end')
    radar_svg += f'<text x="{lx}" y="{ly}" text-anchor="{anchor}" class="axis-label" style="font-size:9px">{dims[i]}</text>'

# Shield polygon
pts_s = []
for i in range(n_dims):
    angle = -90 + (360 / n_dims) * i
    rad = math.radians(angle)
    x = cx + r * shield_vals[i] * math.cos(rad)
    y = cy + r * shield_vals[i] * math.sin(rad)
    pts_s.append(f'{x:.1f},{y:.1f}')
radar_svg += f'<polygon points="{" ".join(pts_s)}" fill="rgba(99,102,241,0.15)" stroke="#6366f1" stroke-width="2"/>'

# Arrow polygon
pts_a = []
for i in range(n_dims):
    angle = -90 + (360 / n_dims) * i
    rad = math.radians(angle)
    x = cx + r * arrow_vals[i] * math.cos(rad)
    y = cy + r * arrow_vals[i] * math.sin(rad)
    pts_a.append(f'{x:.1f},{y:.1f}')
radar_svg += f'<polygon points="{" ".join(pts_a)}" fill="rgba(16,185,129,0.15)" stroke="#10b981" stroke-width="2"/>'

# Legend
radar_svg += f'<rect x="10" y="230" width="12" height="3" rx="1" fill="#6366f1"/>'
radar_svg += f'<text x="26" y="233" class="axis-label" style="font-size:10px">Shield V6</text>'
radar_svg += f'<rect x="100" y="230" width="12" height="3" rx="1" fill="#10b981"/>'
radar_svg += f'<text x="116" y="233" class="axis-label" style="font-size:10px">Arrow V11</text>'

radar_svg += '</svg>'
radar_html = radar_svg

# 4. Signal Performance (from tracking data)
recs_list = tracking.get('recommendations', [])
signal_perf_html = ''
if recs_list:
    sig_labels = {'🟢🟢': ('STRONG 强', '#10b981'), '🟢': ('BUY 买', '#34d399'), '🟡': ('WATCH 看', '#f59e0b')}
    for sig in ['🟢🟢', '🟢', '🟡']:
        sig_recs = [r for r in recs_list if r['signal'] == sig]
        if sig_recs:
            avg_ret = sum(r['return_pct'] for r in sig_recs) / len(sig_recs)
            wins = sum(1 for r in sig_recs if r['return_pct'] > 0)
            wr = wins / len(sig_recs) * 100
            cum = sum(r['return_pct'] for r in sig_recs)
            label, color = sig_labels[sig]
            
            # Mini equity curve for this signal
            cum_vals = [0]
            for r in sorted(sig_recs, key=lambda x: x['entry_date']):
                cum_vals.append(cum_vals[-1] + r['return_pct'])
            
            signal_perf_html += f'''<div class="sp-row">
              <div class="sp-label"><span class="sig-dot" style="background:{color}"></span>{label}</div>
              <div class="sp-stats">
                <span class="sp-count">{len(sig_recs)} picks</span>
                <span class="sp-wr" style="color:{color}">WR {wr:.0f}%</span>
                <span class="sp-avg" style="color:{color}">Avg {avg_ret:+.2f}%</span>
                <span class="sp-cum" style="color:{color}">Cum {cum:+.1f}%</span>
              </div>
            </div>'''

v6_picks = v6.get('picks',[])[:15]
v11_picks = v11.get('picks',[])[:10]
v6_rows = ''.join(pick_row(i, pk, 20, len(v6_picks)) for i, pk in enumerate(v6_picks))
v11_rows = ''.join(pick_row(i, pk, 5, len(v11_picks)) for i, pk in enumerate(v11_picks))

# ============================================================
# TAB 2: TRACKING — Real signal performance
# ============================================================
def tracking_tab():
    recs_list = tracking.get('recommendations', [])
    if not recs_list:
        return '<div class="empty">No tracking data yet 暂无追踪数据</div>'
    
    # Overall stats
    total = len(recs_list)
    wins = sum(1 for r in recs_list if r['return_pct'] > 0)
    avg_ret = sum(r['return_pct'] for r in recs_list) / total
    
    # By signal type
    signal_stats = {}
    for sig in ['🟢🟢', '🟢', '🟡']:
        sig_recs = [r for r in recs_list if r['signal'] == sig]
        if sig_recs:
            sig_wins = sum(1 for r in sig_recs if r['return_pct'] > 0)
            sig_avg = sum(r['return_pct'] for r in sig_recs) / len(sig_recs)
            signal_stats[sig] = {
                'count': len(sig_recs),
                'win_rate': sig_wins / len(sig_recs) * 100,
                'avg_return': sig_avg,
                'cum_return': sum(r['return_pct'] for r in sig_recs)
            }
    
    # Signal performance cards
    sig_cards = ''
    sig_labels = {'🟢🟢': ('STRONG 强', '#10b981'), '🟢': ('BUY 买', '#34d399'), '🟡': ('WATCH 看', '#f59e0b')}
    for sig in ['🟢🟢', '🟢', '🟡']:
        if sig in signal_stats:
            s = signal_stats[sig]
            label, color = sig_labels[sig]
            sig_cards += f'''<div class="stat-card" style="border-top:3px solid {color}">
              <div class="stat-label">{label}</div>
              <div class="stat-value" style="color:{color}">{s['count']}</div>
              <div class="stat-sub">WR {s['win_rate']:.0f}% · Avg {s['avg_return']:+.2f}%</div>
              <div class="stat-sub" style="color:{color}">Cum {s['cum_return']:+.1f}%</div>
            </div>'''
    
    # Cumulative chart by signal
    cum_chart = svg_cumulative_by_signal(recs_list, 650, 200)
    
    # Recent recommendations table
    recent = sorted(recs_list, key=lambda x: x['entry_date'], reverse=True)[:20]
    table_rows = ''
    for r in recent:
        ret_c = '#10b981' if r['return_pct'] > 0 else '#ef4444'
        status_t = 'Holding 持有' if r['status'] == 'holding' else 'Done 完成'
        status_c = '#6366f1' if r['status'] == 'holding' else '#6b7280'
        table_rows += f'''<tr>
          <td style="font-size:11px;color:#6b7280">{r['entry_date'][5:]}</td>
          <td style="font-weight:700;font-family:'JetBrains Mono',monospace">{r['ticker']}</td>
          <td>{'Shield 盾' if r['model']=='shield' else 'Arrow 箭'}</td>
          <td>{signal_badge(r['signal'])}</td>
          <td style="font-family:'JetBrains Mono',monospace">${r['entry_price']:.2f}</td>
          <td style="color:{ret_c};font-weight:700;font-family:'JetBrains Mono',monospace">{r['return_pct']:+.2f}%</td>
          <td style="color:{status_c};font-size:10px">{status_t}</td>
        </tr>'''
    
    return f'''
    <div class="tracking-summary">
      <div class="stat-card"><div class="stat-label">TOTAL PICKS 推荐总数</div><div class="stat-value">{total}</div><div class="stat-sub">This week 本周</div></div>
      <div class="stat-card"><div class="stat-label">WIN RATE 胜率</div><div class="stat-value" style="color:#10b981">{wins/total*100:.0f}%</div><div class="stat-sub">{wins}/{total}</div></div>
      <div class="stat-card"><div class="stat-label">AVG RETURN 平均收益</div><div class="stat-value" style="color:{'#10b981' if avg_ret > 0 else '#ef4444'}">{avg_ret:+.2f}%</div><div class="stat-sub">Per pick 每笔</div></div>
      {sig_cards}
    </div>
    
    <div class="bento-item" style="margin-bottom:16px">
      <div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Cumulative Return by Signal 按信号累计收益</div>
      <div class="section-count">{tracking.get('period','')}</div></div>
      {cum_chart}
    </div>
    
    <div class="bento-item">
      <div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Recent Picks 近期推荐</div>
      <div class="section-count">{len(recent)} shown</div></div>
      <table class="data-table">
        <tr><th>Date 日期</th><th>Ticker 代码</th><th>Model 模型</th><th>Signal 信号</th><th>Entry 入场</th><th>Return 收益</th><th>Status 状态</th></tr>
        {table_rows}
      </table>
    </div>
    '''

# ============================================================
# TAB 3: MODELS
# ============================================================
def model_tab():
    sf = shield_meta.get('feature_importance', {})
    sf_sorted = sorted(sf.items(), key=lambda x: x[1], reverse=True)[:15]
    sf_labels = [f[0] for f in sf_sorted]
    sf_values = [f[1] for f in sf_sorted]
    sf_colors = ['#6366f1' if any(k in l for k in ['spy','qqq','iwm','vix']) else '#10b981' if any(k in l for k in ['vol','ret','mom']) else '#f59e0b' if any(k in l for k in ['ma','bb','macd','rsi']) else '#8b5cf6' for l in sf_labels]
    
    af = arrow_meta.get('feature_importance', {})
    af_sorted = sorted(af.items(), key=lambda x: x[1], reverse=True)[:15]
    af_labels = [f[0] for f in af_sorted]
    af_values = [f[1] for f in af_sorted]
    af_colors = ['#10b981' if any(k in l for k in ['spy','qqq','iwm','vix']) else '#6366f1' if any(k in l for k in ['vol','ret','mom']) else '#f59e0b' if any(k in l for k in ['ma','bb','macd','rsi']) else '#8b5cf6' for l in af_labels]
    
    sv = shield_meta.get('validation', {})
    av = arrow_meta.get('validation', {})
    
    def feat_breakdown(meta):
        tech = meta.get('tech_features')
        macro = meta.get('macro_features')
        fund = meta.get('fund_features')
        if tech is not None:
            return tech or 0, macro or 0, fund or 0, (tech or 0)+(macro or 0)+(fund or 0)
        features = meta.get('features', [])
        t = sum(1 for f in features if any(k in f for k in ['ret','mom','vol','rsi','macd','bb','ma_','ma5','ma20','ma60','price_pos','range','trend','quality']))
        m = sum(1 for f in features if any(k in f for k in ['vix','spy','qqq','iwm']))
        fn = sum(1 for f in features if any(k in f for k in ['pe_','div_yield','beta']))
        return t, m, fn, len(features)
    
    st, sm, sf_n, stot = feat_breakdown(shield_meta)
    at, am, af_n, atot = feat_breakdown(arrow_meta)
    
    return f'''
    <div class="model-cards">
      <div class="model-card" style="border-left:3px solid #6366f1">
        <div class="model-name" style="color:#6366f1">Shield V6 蓝盾</div>
        <div class="model-algo">{shield_meta.get('algorithm','')} · {shield_meta.get('n_trees',0)} trees · {stot} features</div>
        <div class="model-stats">
          <div class="ms"><span class="ms-val">{sv.get('oos_sharpe',0):.2f}</span><span class="ms-lbl">Sharpe 夏普</span></div>
          <div class="ms"><span class="ms-val">{sv.get('oos_annual_return',0):.0f}%</span><span class="ms-lbl">Annual 年化</span></div>
          <div class="ms"><span class="ms-val">{sv.get('oos_max_dd',0):.1f}%</span><span class="ms-lbl">Max DD 回撤</span></div>
          <div class="ms"><span class="ms-val">{sv.get('oos_win_rate',0)}%</span><span class="ms-lbl">Win Rate 胜率</span></div>
        </div>
        <div class="model-feats">
          <span class="feat-tag macro">{sm} macro 宏观</span>
          <span class="feat-tag tech">{st} tech 技术</span>
          <span class="feat-tag fund">{sf_n} fund 基本</span>
        </div>
        <div class="model-universe">{shield_meta.get('universe','')} · {shield_meta.get('hold_days','')}d hold 持有</div>
      </div>
      <div class="model-card" style="border-left:3px solid #10b981">
        <div class="model-name" style="color:#10b981">Arrow V11 绿箭</div>
        <div class="model-algo">{arrow_meta.get('algorithm','')} · {arrow_meta.get('n_trees',0)} trees · {atot} features</div>
        <div class="model-stats">
          <div class="ms"><span class="ms-val">{av.get('oos_sharpe', av.get('wf_avg_net', 0)):.2f}</span><span class="ms-lbl">{'Sharpe' if av.get('oos_sharpe') else 'WF Net'}</span></div>
          <div class="ms"><span class="ms-val">{av.get('oos_avg_net', av.get('wf_avg_net', 0)):.1f}%</span><span class="ms-lbl">5d Net 净收</span></div>
          <div class="ms"><span class="ms-val">{av.get('oos_win_rate', 0)}%</span><span class="ms-lbl">Win Rate 胜率</span></div>
          <div class="ms"><span class="ms-val">{av.get('oos_big50', 0)}</span><span class="ms-lbl">Big50 大奖</span></div>
        </div>
        <div class="model-feats">
          <span class="feat-tag macro">{am} macro 宏观</span>
          <span class="feat-tag tech">{at} tech 技术</span>
        </div>
        <div class="model-universe">{arrow_meta.get('universe','')} · {arrow_meta.get('hold_days','')}d hold 持有</div>
      </div>
    </div>
    <div class="tracking-charts">
      <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield V6 Feature Importance 特征重要性</div><div class="section-count">Top 15 / {stot}</div></div>
        {svg_bar_horizontal(sf_values, sf_labels, sf_colors, 600, 320)}</div>
      <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow V11 Feature Importance 特征重要性</div><div class="section-count">Top 15 / {atot}</div></div>
        {svg_bar_horizontal(af_values, af_labels, af_colors, 600, 320)}</div>
    </div>
    <div class="bento-item" style="margin-top:16px">
      <div class="section-head"><div class="section-title"><span class="dot" style="background:#f59e0b"></span>XGBoost Hyperparameters 超参数</div></div>
      <div class="params-grid">
        <div class="param"><span class="param-key">Objective 目标</span><span class="param-val">{shield_meta.get('params',{}).get('objective','')}</span></div>
        <div class="param"><span class="param-key">Max Depth 深度</span><span class="param-val">{shield_meta.get('params',{}).get('max_depth','')}</span></div>
        <div class="param"><span class="param-key">Learning Rate 学习率</span><span class="param-val">{shield_meta.get('params',{}).get('learning_rate','')}</span></div>
        <div class="param"><span class="param-key">Subsample 采样</span><span class="param-val">{shield_meta.get('params',{}).get('subsample','')}</span></div>
        <div class="param"><span class="param-key">Colsample 列采样</span><span class="param-val">{shield_meta.get('params',{}).get('colsample_bytree','')}</span></div>
        <div class="param"><span class="param-key">Min Child 最小子</span><span class="param-val">{shield_meta.get('params',{}).get('min_child_weight','')}</span></div>
        <div class="param"><span class="param-key">Device 设备</span><span class="param-val">{shield_meta.get('params',{}).get('device','')}</span></div>
        <div class="param"><span class="param-key">Trees 树数</span><span class="param-val">{shield_meta.get('n_trees','')}</span></div>
      </div>
    </div>
    '''

# ============================================================
# CN REDWOOD TAB v3.0 — Industrial Brutalist Design
# ============================================================
def cn_redwood_tab():
    """Load the Redwood tab HTML from template file."""
    template_path = os.path.join(ROOT, 'scripts/cn/redwood_tab_template.html')
    try:
        with open(template_path) as f:
            return f.read()
    except FileNotFoundError:
        return '<div style="padding:20px;color:var(--dim)">红杉看板模板未找到，请运行 gen_xgb_signal.py 生成数据</div>'

# ASSEMBLE HTML
# ============================================================
tab1 = f'''
<div class="metrics">
  <div class="metric"><div class="metric-label">Portfolio 组合</div><div class="metric-value" style="color:#fff">${total_val:,.0f}</div><div class="metric-sub">{len(positions)} positions 持仓</div></div>
  <div class="metric"><div class="metric-label">P&L 盈亏</div><div class="metric-value" style="color:{pnl_c}">{'+' if total_pnl >= 0 else ''}{total_pnl:,.0f}</div><div class="metric-sub">{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct:.1f}% total</div></div>
  <div class="metric"><div class="metric-label">VIX 波动率</div><div class="metric-value" style="color:{vix_c}">{vix_val}</div><div class="metric-sub">{vix_status}</div></div>
  <div class="metric"><div class="metric-label">Win/Loss 盈亏比</div><div class="metric-value"><span style="color:#10b981">{win_count}</span> / <span style="color:#ef4444">{loss_count}</span></div><div class="metric-sub">Today 今日</div></div>
</div>
<div class="bento">
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield Holdings 蓝盾持仓</div><div class="section-count">{len(shield_pos)} active</div></div>
    <div class="pos-grid">{shield_cards or '<div class="empty">No positions 无持仓</div>'}</div></div>
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow Holdings 绿箭持仓</div><div class="section-count">{len(arrow_pos)} active</div></div>
    <div class="pos-grid">{arrow_cards or '<div class="empty">No positions 无持仓</div>'}</div></div>
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield V6 Top Picks 蓝盾推荐</div><div class="section-count">{v6.get('total',0)} universe</div></div>
    <div class="pick-list">{v6_rows or '<div class="empty">No picks 无推荐</div>'}</div></div>
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow V11 Top Picks 绿箭推荐</div><div class="section-count">{v11.get('total',0)} universe</div></div>
    <div class="pick-list">{v11_rows or '<div class="empty">No picks 无推荐</div>'}</div></div>
</div>
<div class="legend">
  <div class="legend-item"><span class="sig-dot" style="background:#10b981"></span><span class="sig-dot" style="background:#10b981"></span>STRONG 强 Top5%</div>
  <div class="legend-item"><span class="sig-dot" style="background:#34d399"></span>BUY 买 Top10%</div>
  <div class="legend-item"><span class="sig-dot" style="background:#f59e0b"></span>WATCH 看 Top20%</div>
  <div class="legend-item"><span class="sig-dot" style="background:#ef4444"></span>Skip 跳</div>
</div>

<!-- INSIGHTS SECTION -->
<div class="insights-title">INSIGHTS 洞察</div>

<div class="bento">
  <!-- P&L Calendar Heatmap -->
  <div class="bento-item">
    <div class="section-head"><div class="section-title"><span class="dot" style="background:#f59e0b"></span>Daily P&L Calendar 每日盈亏日历</div>
    <div class="section-count">2 weeks 两周</div></div>
    <div class="cal-grid">
      {cal_html}
    </div>
    <div class="cal-legend"><span style="color:#ef4444">■</span> Loss 亏 <span style="color:#3d3d47">■</span> Flat 平 <span style="color:#10b98140">■</span> <span style="color:#10b98180">■</span> <span style="color:#10b981">■</span> Gain 盈</div>
  </div>
  
  <!-- Position Waterfall -->
  <div class="bento-item">
    <div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>P&L Waterfall 盈亏瀑布图</div>
    <div class="section-count">By position 按持仓</div></div>
    {waterfall_html}
  </div>
</div>

<div class="bento">
  <!-- Radar Chart: Model Comparison -->
  <div class="bento-item">
    <div class="section-head"><div class="section-title"><span class="dot" style="background:#8b5cf6"></span>Model Comparison 模型对比</div></div>
    {radar_html}
  </div>
  
  <!-- Signal Performance Summary -->
  <div class="bento-item">
    <div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Signal Performance 信号表现</div>
    <div class="section-count">2 weeks</div></div>
    {signal_perf_html}
  </div>
</div>'''

tab2 = tracking_tab()
tab3 = model_tab()
tab4 = cn_redwood_tab()

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#050508">
<meta http-equiv="refresh" content="300" id="autoRefresh">
<title>Hermes Trading Intelligence</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
:root{{--bg:#050508;--bg-card:rgba(255,255,255,0.03);--bg-hover:rgba(255,255,255,0.05);--border:rgba(255,255,255,0.06);--border-lt:rgba(255,255,255,0.03);--text:#e8e8ed;--dim:#6b6b76;--muted:#3d3d47;--gn:#10b981;--rd:#ef4444;--am:#f59e0b;--ind:#6366f1;--r:16px;--rs:10px;--rx:6px}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,'SF Pro Text',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;overflow-x:hidden}}
body::before{{content:'';position:fixed;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 20% 20%,rgba(99,102,241,0.04) 0%,transparent 50%),radial-gradient(ellipse at 80% 80%,rgba(16,185,129,0.03) 0%,transparent 50%);pointer-events:none;z-index:0}}
.wrap{{max-width:1440px;margin:0 auto;padding:24px 32px;position:relative;z-index:1}}
.header{{display:flex;justify-content:space-between;align-items:center;padding:0 0 16px;border-bottom:1px solid var(--border);margin-bottom:8px}}
.header-left{{display:flex;align-items:center;gap:16px}}
.logo{{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:700;letter-spacing:-0.5px;color:#fff}}.logo-dot{{color:var(--ind)}}
.header-badge{{font-size:10px;font-weight:600;letter-spacing:0.5px;padding:3px 10px;border-radius:20px;background:rgba(99,102,241,0.1);color:var(--ind);border:1px solid rgba(99,102,241,0.2)}}
.header-right{{display:flex;align-items:center;gap:16px}}
.timestamp{{font-size:12px;color:var(--dim);font-family:'JetBrains Mono',monospace}}
.live-dot{{width:6px;height:6px;border-radius:50%;background:var(--gn);box-shadow:0 0 8px rgba(16,185,129,0.3);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}
.tab-bar{{display:flex;gap:2px;padding:16px 0 0;border-bottom:1px solid var(--border)}}
.tab{{padding:10px 20px;font-size:12px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--dim);cursor:pointer;border-bottom:2px solid transparent;transition:all 0.3s;user-select:none}}
.tab:hover{{color:var(--text);background:rgba(255,255,255,0.02)}}.tab.active{{color:#fff;border-bottom-color:var(--ind)}}
.tab-content{{display:none;padding-top:16px}}.tab-content.active{{display:block}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border-radius:var(--r);overflow:hidden;margin-bottom:16px;border:1px solid var(--border)}}
.metric{{background:var(--bg);padding:16px 20px;transition:background 0.3s}}.metric:hover{{background:rgba(255,255,255,0.02)}}
.metric-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--dim);margin-bottom:6px}}
.metric-value{{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-0.5px}}
.metric-sub{{font-size:11px;color:var(--dim);margin-top:2px;font-family:'JetBrains Mono',monospace}}
.bento{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.bento-item{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);padding:20px;backdrop-filter:blur(20px);transition:all 0.4s cubic-bezier(0.32,0.72,0,1);position:relative;overflow:hidden}}
.bento-item::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent)}}
.bento-item:hover{{background:var(--bg-hover);border-color:rgba(255,255,255,0.1)}}
.section-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}}
.section-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:var(--dim);display:flex;align-items:center;gap:10px}}
.section-title .dot{{width:8px;height:8px;border-radius:50%}}.section-count{{font-size:11px;color:var(--muted);font-family:'JetBrains Mono',monospace}}

/* SIGNAL BADGES */
.sig{{display:inline-flex;align-items:center;gap:4px;font-size:9px;font-weight:700;letter-spacing:0.5px;padding:2px 8px;border-radius:4px;border:1px solid}}
.sig-dot{{width:6px;height:6px;border-radius:50%;display:inline-block}}
.sig-strong{{color:#10b981;border-color:#10b98130;background:#10b98110}}.sig-buy{{color:#34d399;border-color:#34d39930;background:#34d39910}}
.sig-watch{{color:#f59e0b;border-color:#f59e0b30;background:#f59e0b10}}.sig-skip{{color:#ef4444;border-color:#ef444430;background:#ef444410}}

/* POSITIONS */
.pos-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}}
.pos-card{{background:rgba(255,255,255,0.02);border:1px solid var(--border-lt);border-radius:var(--rs);padding:12px 14px;transition:all 0.3s cubic-bezier(0.32,0.72,0,1)}}
.pos-card:hover{{background:rgba(255,255,255,0.04);border-color:var(--border);transform:translateY(-2px)}}
.pos-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}}
.pos-code{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#fff;letter-spacing:-0.3px}}
.pos-model{{font-size:8px;font-weight:700;letter-spacing:1px;padding:2px 6px;border-radius:4px;border:1px solid}}
.pos-name{{font-size:10px;color:var(--dim);margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.pos-metrics{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:8px}}
.pos-pnl{{text-align:left}}.pnl-sign{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;display:block;letter-spacing:-0.5px}}
.pnl-usd{{font-family:'JetBrains Mono',monospace;font-size:10px;opacity:0.7;display:block}}
.pos-price{{text-align:right}}.price-cur{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;display:block}}
.price-cost{{font-size:9px;color:var(--muted);display:block}}
.pos-progress{{margin-top:6px}}.prog-label{{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--muted);margin-bottom:3px}}
.prog-bar{{height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden}}.prog-fill{{height:100%;border-radius:2px;transition:width 0.8s cubic-bezier(0.32,0.72,0,1)}}
.stop-badge{{margin-top:6px;font-size:9px;font-weight:700;letter-spacing:1px;color:var(--rd);background:rgba(239,68,68,0.1);padding:3px 8px;border-radius:4px;text-align:center}}
.warn-badge{{margin-top:6px;font-size:9px;font-weight:700;letter-spacing:1px;color:var(--am);background:rgba(245,158,11,0.1);padding:3px 8px;border-radius:4px;text-align:center}}

/* PICKS */
.pick-list{{display:flex;flex-direction:column;gap:1px}}
.pick-row{{display:grid;grid-template-columns:24px 90px 1fr auto 40px;align-items:center;gap:10px;padding:6px 10px;border-radius:var(--rx);transition:background 0.2s}}
.pick-row:hover{{background:rgba(255,255,255,0.03)}}
.pick-rank{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);text-align:center}}
.pick-info{{display:flex;flex-direction:column}}.pick-ticker{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;color:#fff}}
.pick-price{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--dim)}}
.pick-score-bar{{display:flex;align-items:center;gap:8px}}.score-track{{flex:1;height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden}}
.score-fill{{height:100%;border-radius:2px;transition:width 1s cubic-bezier(0.32,0.72,0,1)}}
.score-val{{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;min-width:40px;text-align:right}}
.pick-hold{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);text-align:right}}

/* LEGEND */
.legend{{display:flex;gap:16px;justify-content:center;padding:12px;border-top:1px solid var(--border);margin-top:8px}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--dim);letter-spacing:0.5px}}

/* TRACKING */
.tracking-summary{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:16px}}
.stat-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--rs);padding:16px;text-align:center}}
.stat-label{{font-size:9px;font-weight:700;letter-spacing:1px;color:var(--dim);margin-bottom:6px}}
.stat-value{{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#fff}}
.stat-sub{{font-size:10px;color:var(--muted);margin-top:4px}}
.tracking-charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}

/* DATA TABLE */
.data-table{{width:100%;border-collapse:collapse}}
.data-table th{{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}}
.data-table td{{padding:6px 8px;border-bottom:1px solid var(--border-lt);font-size:12px}}

/* CHARTS */
.chart-svg{{width:100%;height:auto}}.axis-label{{font-family:'JetBrains Mono',monospace;font-size:9px;fill:var(--muted)}}

/* MODELS */
.model-cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.model-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r);padding:24px}}
.model-name{{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;margin-bottom:4px}}
.model-algo{{font-size:11px;color:var(--dim);margin-bottom:16px}}
.model-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}}
.ms{{text-align:center;padding:8px;background:rgba(255,255,255,0.02);border-radius:var(--rx)}}
.ms-val{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#fff;display:block}}
.ms-lbl{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px}}
.model-feats{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}}
.feat-tag{{font-size:9px;font-weight:600;letter-spacing:0.5px;padding:3px 8px;border-radius:4px}}
.feat-tag.macro{{background:rgba(99,102,241,0.1);color:#6366f1;border:1px solid rgba(99,102,241,0.2)}}
.feat-tag.tech{{background:rgba(16,185,129,0.1);color:#10b981;border:1px solid rgba(16,185,129,0.2)}}
.feat-tag.fund{{background:rgba(245,158,11,0.1);color:#f59e0b;border:1px solid rgba(245,158,11,0.2)}}
.model-universe{{font-size:11px;color:var(--dim)}}
.params-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.param{{padding:10px;background:rgba(255,255,255,0.02);border-radius:var(--rx);text-align:center}}
.param-key{{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px}}
.param-val{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#fff}}
.empty{{text-align:center;padding:24px;color:var(--muted);font-size:12px}}
.footer{{text-align:center;padding:16px 0;border-top:1px solid var(--border);margin-top:8px}}
.footer-text{{font-size:10px;color:var(--muted);letter-spacing:0.5px}}

/* AUTO-REFRESH INDICATOR */
.refresh-bar{{position:fixed;bottom:0;left:0;right:0;height:2px;background:rgba(99,102,241,0.3);z-index:100}}
.refresh-fill{{height:100%;background:var(--ind);width:0%;transition:width 5s linear}}

@keyframes fadeUp{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
.bento-item,.stat-card,.model-card{{animation:fadeUp 0.5s cubic-bezier(0.32,0.72,0,1) both}}

@media(max-width:900px){{.wrap{{padding:16px}}.bento,.tracking-charts,.model-cards{{grid-template-columns:1fr}}.metrics,.tracking-summary{{grid-template-columns:repeat(2,1fr)}}.pos-grid{{grid-template-columns:1fr 1fr}}.pick-row{{grid-template-columns:20px 70px 1fr auto 30px}}.params-grid{{grid-template-columns:repeat(2,1fr)}}.model-stats{{grid-template-columns:repeat(2,1fr)}}}}
::-webkit-scrollbar{{width:6px}}::-webkit-scrollbar-track{{background:transparent}}::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.1);border-radius:3px}}

/* INSIGHTS */
.insights-title{{font-size:11px;font-weight:700;letter-spacing:2px;color:var(--dim);text-transform:uppercase;margin:24px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--border)}}
.cal-header{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:6px;text-align:center;font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px}}
.cal-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:4px}}
.cal-cell{{padding:6px 4px;border-radius:4px;text-align:center;min-height:44px;display:flex;flex-direction:column;justify-content:center;cursor:default;transition:transform 0.2s}}
.cal-cell:hover{{transform:scale(1.05)}}.cal-cell.empty{{background:transparent}}
.cal-day{{font-family:'JetBrains Mono',monospace;font-size:8px;opacity:0.7}}.cal-ret{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700}}
.cal-legend{{font-size:9px;color:var(--muted);margin-top:8px;display:flex;gap:8px;justify-content:center}}
.wf-container{{display:flex;flex-direction:column;gap:4px}}
.wf-row{{display:grid;grid-template-columns:50px 1fr 55px 65px;align-items:center;gap:8px;padding:3px 0}}
.wf-label{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:#fff}}
.wf-bar-container{{height:14px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden}}
.wf-bar{{height:100%;border-radius:3px;transition:width 0.8s cubic-bezier(0.32,0.72,0,1)}}
.wf-val{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;text-align:right}}
.wf-usd{{font-family:'JetBrains Mono',monospace;font-size:10px;opacity:0.7;text-align:right}}
.wf-total{{border-top:1px solid var(--border);padding-top:8px;margin-top:4px}}
.sp-row{{display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:rgba(255,255,255,0.02);border-radius:var(--rx);margin-bottom:6px}}
.sp-label{{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600}}
.sp-stats{{display:flex;gap:12px;font-family:'JetBrains Mono',monospace;font-size:11px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="header-left"><div class="logo">HERMES<span class="logo-dot">.</span></div><div class="header-badge">INTELLIGENCE PLATFORM 智能平台</div></div>
    <div class="header-right"><div class="live-dot"></div><div class="timestamp">{now_str}</div><div id="countdown" class="timestamp" style="color:var(--ind)">5:00</div></div>
  </div>
  <div class="tab-bar">
    <div class="tab active" onclick="showTab(0)">Trading 操盘</div>
    <div class="tab" onclick="showTab(1)">Tracking 追踪</div>
    <div class="tab" onclick="showTab(2)">Models 模型</div>
    <div class="tab" onclick="showTab(3)" style="color:#dc2626">Redwood 红杉</div>
  </div>
  <div class="tab-content active" id="tab-0">{tab1}</div>
  <div class="tab-content" id="tab-1">{tab2}</div>
  <div class="tab-content" id="tab-2">{tab3}</div>
  <div class="tab-content" id="tab-3">{tab4}</div>
  <div class="footer"><div class="footer-text">Hermes Trading Intelligence · Shield V6 + Arrow V11 + Redwood v1.0 · Three-Layer Signal Filter 三层过滤</div></div>
</div>
<div class="refresh-bar"><div class="refresh-fill" id="refreshFill"></div></div>

<script>
function showTab(n){{
  document.querySelectorAll('.tab-content').forEach((s,i)=>s.classList.toggle('active',i===n));
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===n));
}}

// Auto-refresh: meta refresh每300秒自动刷新（不弹窗）
// 非工作时间通过JS移除meta refresh停止自动刷新
const fill = document.getElementById('refreshFill');
const cdEl = document.getElementById('countdown');
let countdown = 300;

function isMarketOpen() {{
  const now = new Date();
  const utc = now.getUTCHours() * 60 + now.getUTCMinutes();
  let et = (utc - 4 * 60 + 1440) % 1440;
  return et >= 570 && et <= 960 && now.getDay() >= 1 && now.getDay() <= 5;
}}

function isCNMarketOpen() {{
  const now = new Date();
  const bj = (now.getUTCHours() + 8) % 24;
  const bjMin = bj * 60 + now.getUTCMinutes();
  const day = now.getDay();
  const morning = bjMin >= 570 && bjMin <= 690;
  const afternoon = bjMin >= 780 && bjMin <= 900;
  return (morning || afternoon) && day >= 1 && day <= 5;
}}

if (!isMarketOpen() && !isCNMarketOpen()) {{
  const meta = document.getElementById('autoRefresh');
  if (meta) meta.remove();
  if (cdEl) cdEl.textContent = '休市中';
  countdown = 0;
}}

setInterval(() => {{
  if (countdown <= 0) return;
  countdown--;
  const min = Math.floor(countdown / 60);
  const sec = countdown % 60;
  if (cdEl) cdEl.textContent = min + ':' + String(sec).padStart(2, '0');
  if (fill) fill.style.width = ((1 - countdown / 300) * 100) + '%';
}}, 1000);
</script>
</body>
</html>'''

out = os.path.join(ROOT, 'dashboard.html')
with open(out, 'w') as f:
    f.write(html)
print(f'Dashboard v7: {len(html):,} bytes')
print(f'  Positions: {len(positions)} | Shield: {len(shield_pos)} | Arrow: {len(arrow_pos)}')
print(f'  Tracking: {len(tracking.get("recommendations",[]))} recommendations')
