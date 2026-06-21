#!/usr/bin/env python3
"""Hermes Trading Dashboard v6 — Triple Tab Intelligence Platform
Tab 1: Daily Trading (宏观+持仓+推荐+信号)
Tab 2: Recommendation Tracking (入场→持有→退出全周期)
Tab 3: Model Intelligence (特征+验证+信号分布)
Style: Ethereal Glass + SVG Charts + Animated Counters
"""
import json, os, math
from datetime import datetime, timedelta

ROOT = '/home/hermes/.hermes/openclaw-archive'
def load(p):
    try:
        with open(os.path.join(ROOT, p)) as f: return json.load(f)
    except: return {}

futu = load('output/futu_positions.json')
v6 = load('output/v6_latest.json')
v11 = load('output/v11_latest.json')
held = load('output/held_scores.json')
recs = load('output/recommendations.json')
shield_meta = load('models/us/blueshield_v6_meta.json')
arrow_meta = load('models/us/arrow_v11_meta.json')

positions = futu.get('positions', [])
shield_pos = [p for p in positions if p.get('cost_price',0) > 10]
arrow_pos = [p for p in positions if p.get('cost_price',0) <= 10]
held_map = {h['sym']: h for h in held.get('shield', []) + held.get('arrow', [])}

now = datetime.now().strftime('%Y-%m-%d %H:%M')
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

# Recommendation stats
shield_stats = recs.get('shield', {}).get('stats', {})
arrow_stats = recs.get('arrow', {}).get('stats', {})

# ============================================================
# SVG CHART HELPERS
# ============================================================
def svg_equity_curve(equity, color='#10b981', w=600, h=180, label=''):
    """Generate SVG equity curve chart."""
    if not equity or len(equity) < 2:
        return '<div class="empty">No data</div>'
    
    mn, mx = min(equity), max(equity)
    rng = mx - mn if mx > mn else 1
    pad = 20
    
    points = []
    for i, v in enumerate(equity):
        x = pad + (i / (len(equity)-1)) * (w - 2*pad)
        y = h - pad - ((v - mn) / rng) * (h - 2*pad)
        points.append(f'{x:.1f},{y:.1f}')
    
    polyline = ' '.join(points)
    # Fill area
    first_x, last_x = pad, pad + (w - 2*pad)
    fill_points = f'{first_x},{h-pad} {polyline} {last_x},{h-pad}'
    
    # Start/end labels
    start_val = f'{equity[0]:.0f}'
    end_val = f'{equity[-1]:.0f}'
    ret_pct = (equity[-1] / equity[0] - 1) * 100
    
    return f'''<svg viewBox="0 0 {w} {h}" class="chart-svg">
      <defs>
        <linearGradient id="grad_{label}" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{color}" stop-opacity="0.3"/>
          <stop offset="100%" stop-color="{color}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <polygon points="{fill_points}" fill="url(#grad_{label})" />
      <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <text x="{pad}" y="{h-4}" class="chart-label">{start_val}</text>
      <text x="{w-pad}" y="{h-4}" class="chart-label" text-anchor="end">{end_val} ({ret_pct:+.1f}%)</text>
    </svg>'''

def svg_bar_chart(data, labels, colors, w=500, h=200):
    """Generate SVG horizontal bar chart."""
    if not data:
        return ''
    n = len(data)
    max_val = max(data) if data else 1
    bar_h = min(20, (h - 20) / n - 4)
    
    bars = ''
    for i, (val, label, color) in enumerate(zip(data, labels, colors)):
        y = i * (bar_h + 4) + 10
        bar_w = (val / max_val) * (w - 180) if max_val > 0 else 0
        bars += f'''<g>
          <text x="0" y="{y+bar_h*0.75}" class="chart-label" style="font-size:11px">{label}</text>
          <rect x="130" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}" opacity="0.8"/>
          <text x="{135+bar_w:.1f}" y="{y+bar_h*0.75}" class="chart-value" style="font-size:10px">{val:.1f}</text>
        </g>'''
    
    return f'<svg viewBox="0 0 {w} {n*(bar_h+4)+20}" class="chart-svg">{bars}</svg>'

def svg_donut(pct, color='#10b981', size=80, label=''):
    """Generate SVG donut chart."""
    r = size/2 - 8
    circumference = 2 * math.pi * r
    offset = circumference * (1 - pct/100)
    return f'''<svg viewBox="0 0 {size} {size}" class="donut-svg">
      <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="6"/>
      <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="6"
        stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}"
        stroke-linecap="round" transform="rotate(-90 {size/2} {size/2})" class="donut-fill"/>
      <text x="{size/2}" y="{size/2}" text-anchor="middle" dominant-baseline="central" class="donut-text">{pct:.0f}%</text>
    </svg>'''

def sparkline_svg(values, color='#10b981', w=80, h=24):
    """Tiny sparkline for inline use."""
    if not values or len(values) < 2:
        return ''
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    pts = []
    for i, v in enumerate(values):
        x = (i / (len(values)-1)) * w
        y = h - ((v - mn) / rng) * h
        pts.append(f'{x:.1f},{y:.1f}')
    return f'<svg viewBox="0 0 {w} {h}" class="sparkline"><polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.5"/></svg>'

# ============================================================
# TAB 1: DAILY TRADING
# ============================================================
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
        rank_html = f'<span style="color:{rc};font-size:11px;font-weight:600">{rt}</span>'
    else:
        rank_html = '<span style="color:#333;font-size:11px">--</span>'
    
    pnl_c = '#10b981' if pnl >= 0 else '#ef4444'
    bar_c = '#10b981' if prog >= 100 else '#f59e0b' if prog >= 50 else '#6366f1'
    
    stop_html = ''
    if p.get('stop_triggered'):
        stop_html = '<div class="stop-badge">STOP</div>'
    elif pnl <= -8:
        stop_html = '<div class="warn-badge">WATCH</div>'
    
    model_tag = 'SHIELD' if is_shield else 'ARROW'
    model_c = '#6366f1' if is_shield else '#10b981'
    
    # Days remaining
    days_left = max(0, hold_days - held_days)
    days_text = f'{days_left}d left' if days_left > 0 else 'EXPIRED'
    
    return f'''<div class="pos-card" data-model="{'shield' if is_shield else 'arrow'}">
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
          <span class="price-cost">from ${cost:.2f}</span>
        </div>
      </div>
      <div class="pos-progress">
        <div class="prog-label">{held_days}/{hold_days}d · {days_text}</div>
        <div class="prog-bar"><div class="prog-fill" style="width:{prog:.0f}%;background:{bar_c}"></div></div>
      </div>
      {stop_html}
    </div>'''

def pick_row(i, pk, hold_days):
    sig = pk.get('signal','')
    sc = signal_color(sig)
    sl = signal_label(sig)
    score = pk['pred_rank']
    bar_w = max(5, min(100, (score - 0.50) / 0.20 * 100))
    
    return f'''<div class="pick-row">
      <div class="pick-rank">{i+1}</div>
      <div class="pick-info"><div class="pick-ticker">{pk['ticker']}</div><div class="pick-price">${pk['price']:.2f}</div></div>
      <div class="pick-score-bar"><div class="score-track"><div class="score-fill" style="width:{bar_w:.0f}%;background:{sc}"></div></div>
      <span class="score-val" style="color:{sc}">{score:.3f}</span></div>
      <div class="pick-signal" style="color:{sc};border-color:{sc}30;background:{sc}10">{sl}</div>
      <div class="pick-hold">{hold_days}d</div>
    </div>'''

shield_cards = ''.join(pos_card(p) for p in shield_pos)
arrow_cards = ''.join(pos_card(p) for p in arrow_pos)
v6_rows = ''.join(pick_row(i, pk, 20) for i, pk in enumerate(v6.get('picks',[])[:15]))
v11_rows = ''.join(pick_row(i, pk, 5) for i, pk in enumerate(v11.get('picks',[])[:10]))

win_count = sum(1 for p in positions if p.get('pnl_pct',0) > 0)
loss_count = len(positions) - win_count

# ============================================================
# TAB 2: RECOMMENDATION TRACKING
# ============================================================
def tracking_tab():
    s = shield_stats
    a = arrow_stats
    
    # Combined stats
    total_trades = s.get('total_trades',0) + a.get('total_trades',0)
    combined_wr = ((s.get('win_rate',0)*s.get('total_trades',0) + a.get('win_rate',0)*a.get('total_trades',0)) / total_trades) if total_trades else 0
    combined_avg = ((s.get('avg_return',0)*s.get('total_trades',0) + a.get('avg_return',0)*a.get('total_trades',0)) / total_trades) if total_trades else 0
    
    # Equity curves
    shield_eq = s.get('equity_curve', [])
    arrow_eq = a.get('equity_curve', [])
    
    # Monthly heatmap data (shield)
    monthly_html = ''
    sm = s.get('monthly', {})
    if sm:
        months = sorted(sm.keys())[-12:]  # last 12 months
        cells = ''
        for m in months:
            md = sm[m]
            ret = md['avg_return']
            wr = md['win_rate']
            if ret > 3: bg = '#10b98140'
            elif ret > 0: bg = '#10b98120'
            elif ret > -2: bg = '#ef444420'
            else: bg = '#ef444440'
            tc = '#10b981' if ret > 0 else '#ef4444'
            cells += f'''<div class="heat-cell" style="background:{bg}" title="{m}: {md['count']} trades, {ret:+.1f}% avg, {wr:.0f}% WR">
              <div class="heat-month">{m[5:]}</div>
              <div class="heat-ret" style="color:{tc}">{ret:+.1f}%</div>
              <div class="heat-wr">{wr:.0f}% WR</div>
            </div>'''
        monthly_html = f'<div class="heat-grid">{cells}</div>'
    
    # Arrow monthly
    am = a.get('monthly', {})
    arrow_monthly_html = ''
    if am:
        months = sorted(am.keys())[-12:]
        cells = ''
        for m in months:
            md = am[m]
            ret = md['avg_return']
            wr = md['win_rate']
            if ret > 5: bg = '#10b98140'
            elif ret > 0: bg = '#10b98120'
            elif ret > -3: bg = '#ef444420'
            else: bg = '#ef444440'
            tc = '#10b981' if ret > 0 else '#ef4444'
            cells += f'''<div class="heat-cell" style="background:{bg}" title="{m}: {md['count']} trades, {ret:+.1f}% avg, {wr:.0f}% WR">
              <div class="heat-month">{m[5:]}</div>
              <div class="heat-ret" style="color:{tc}">{ret:+.1f}%</div>
              <div class="heat-wr">{wr:.0f}% WR</div>
            </div>'''
        arrow_monthly_html = f'<div class="heat-grid">{cells}</div>'
    
    # Win/Loss distribution (shield)
    shield_recs_list = recs.get('shield', {}).get('recommendations', [])
    if shield_recs_list:
        returns = [r['return_pct'] for r in shield_recs_list]
        bins = [(-15,-10), (-10,-5), (-5,-2), (-2,0), (0,2), (2,5), (5,10), (10,20), (20,50)]
        bin_labels = ['-15/-10', '-10/-5', '-5/-2', '-2/0', '0/+2', '+2/+5', '+5/+10', '+10/+20', '+20/+50']
        bin_counts = []
        for lo, hi in bins:
            cnt = sum(1 for r in returns if lo <= r < hi)
            bin_counts.append(cnt)
        max_cnt = max(bin_counts) if bin_counts else 1
        dist_bars = ''
        for (lo, hi), label, cnt in zip(bins, bin_labels, bin_counts):
            bar_w = cnt / max_cnt * 100
            c = '#10b981' if lo >= 0 else '#ef4444'
            dist_bars += f'''<div class="dist-row">
              <div class="dist-label">{label}%</div>
              <div class="dist-bar"><div class="dist-fill" style="width:{bar_w:.0f}%;background:{c}"></div></div>
              <div class="dist-count">{cnt}</div>
            </div>'''
    else:
        dist_bars = ''
    
    return f'''
    <div class="tracking-summary">
      <div class="stat-card">
        <div class="stat-label">TOTAL TRADES</div>
        <div class="stat-value" data-count="{total_trades}">{total_trades}</div>
        <div class="stat-sub">Shield {s.get('total_trades',0)} + Arrow {a.get('total_trades',0)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">WIN RATE</div>
        <div class="stat-value" style="color:#10b981">{combined_wr:.1f}%</div>
        <div class="stat-sub">{svg_donut(combined_wr, '#10b981', 60)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">AVG RETURN</div>
        <div class="stat-value" style="color:#10b981">+{combined_avg:.2f}%</div>
        <div class="stat-sub">per trade</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">PROFIT FACTOR</div>
        <div class="stat-value">{s.get('profit_factor',0):.1f}x</div>
        <div class="stat-sub">Shield model</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">BEST TRADE</div>
        <div class="stat-value" style="color:#10b981">+{s.get('best_trade',0):.1f}%</div>
        <div class="stat-sub">Shield V6</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">MAX DRAWDOWN</div>
        <div class="stat-value" style="color:#ef4444">-{s.get('max_drawdown',0):.1f}%</div>
        <div class="stat-sub">Shield model</div>
      </div>
    </div>
    
    <div class="tracking-charts">
      <div class="bento-item">
        <div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield V6 Equity Curve</div>
        <div class="section-count">20-day hold · Top 15</div></div>
        {svg_equity_curve(shield_eq, '#6366f1', 700, 180, 'shield')}
      </div>
      <div class="bento-item">
        <div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow V11 Equity Curve</div>
        <div class="section-count">5-day hold · Top 5</div></div>
        {svg_equity_curve(arrow_eq, '#10b981', 700, 180, 'arrow')}
      </div>
    </div>
    
    <div class="tracking-charts">
      <div class="bento-item">
        <div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield Monthly Performance</div></div>
        {monthly_html}
      </div>
      <div class="bento-item">
        <div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow Monthly Performance</div></div>
        {arrow_monthly_html}
      </div>
    </div>
    
    <div class="bento-item" style="margin-top:16px">
      <div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Return Distribution (Shield V6)</div>
      <div class="section-count">{s.get('total_trades',0)} trades</div></div>
      <div class="dist-chart">{dist_bars}</div>
    </div>
    '''

# ============================================================
# TAB 3: MODEL INTELLIGENCE
# ============================================================
def model_tab():
    # Shield feature importance (top 15)
    sf = shield_meta.get('feature_importance', {})
    sf_sorted = sorted(sf.items(), key=lambda x: x[1], reverse=True)[:15]
    sf_labels = [f[0] for f in sf_sorted]
    sf_values = [f[1] for f in sf_sorted]
    sf_colors = ['#6366f1' if 'spy' in l or 'qqq' in l or 'iwm' in l or 'vix' in l 
                 else '#10b981' if 'vol' in l or 'ret' in l or 'mom' in l
                 else '#f59e0b' if 'ma' in l or 'bb' in l or 'macd' in l or 'rsi' in l
                 else '#8b5cf6' for l in sf_labels]
    
    # Arrow feature importance (top 15)
    af = arrow_meta.get('feature_importance', {})
    af_sorted = sorted(af.items(), key=lambda x: x[1], reverse=True)[:15]
    af_labels = [f[0] for f in af_sorted]
    af_values = [f[1] for f in af_sorted]
    af_colors = ['#10b981' if 'spy' in l or 'qqq' in l or 'iwm' in l or 'vix' in l
                 else '#6366f1' if 'vol' in l or 'ret' in l or 'mom' in l
                 else '#f59e0b' if 'ma' in l or 'bb' in l or 'macd' in l or 'rsi' in l
                 else '#8b5cf6' for l in af_labels]
    
    # Validation metrics
    sv = shield_meta.get('validation', {})
    av = arrow_meta.get('validation', {})
    
    # Feature category breakdown
    def feat_breakdown(meta):
        # Try explicit fields first
        tech = meta.get('tech_features')
        macro = meta.get('macro_features')
        fund = meta.get('fund_features')
        if tech is not None:
            total = (tech or 0) + (macro or 0) + (fund or 0)
            return tech or 0, macro or 0, fund or 0, total or meta.get('n_features', 0)
        # Infer from feature names
        features = meta.get('features', [])
        t = sum(1 for f in features if any(k in f for k in ['ret','mom','vol','rsi','macd','bb','ma_','ma5','ma20','ma60','price_pos','range','trend','quality']))
        m = sum(1 for f in features if any(k in f for k in ['vix','spy','qqq','iwm']))
        fn = sum(1 for f in features if any(k in f for k in ['pe_','div_yield','beta']))
        return t, m, fn, len(features)
    
    st, sm, sf_n, stot = feat_breakdown(shield_meta)
    at, am, af_n, atot = feat_breakdown(arrow_meta)
    
    # Signal thresholds
    thresholds_html = ''
    for model_name, meta in [('Shield V6', shield_meta), ('Arrow V11', arrow_meta)]:
        st_data = meta.get('signal_thresholds', {})
        c = '#6366f1' if 'Shield' in model_name else '#10b981'
        rows = ''
        for key, info in st_data.items():
            if isinstance(info, dict):
                label = info.get('label', key)
                thresh = info.get('threshold', info.get('percentile', ''))
                note = info.get('note', '')
                rows += f'<tr><td style="color:{c}">{label}</td><td>{thresh}</td><td style="color:#6b7280;font-size:11px">{note}</td></tr>'
        thresholds_html += f'''<div class="bento-item">
          <div class="section-head"><div class="section-title"><span class="dot" style="background:{c}"></span>{model_name} Signal Thresholds</div></div>
          <table class="meta-table"><tr><th>Signal</th><th>Threshold</th><th>Note</th></tr>{rows}</table>
        </div>'''
    
    return f'''
    <div class="model-cards">
      <div class="model-card" style="border-left:3px solid #6366f1">
        <div class="model-name" style="color:#6366f1">Shield V6</div>
        <div class="model-algo">{shield_meta.get('algorithm','')} · {shield_meta.get('n_trees',0)} trees</div>
        <div class="model-stats">
          <div class="ms"><span class="ms-val">{sv.get('oos_sharpe',0):.2f}</span><span class="ms-lbl">Sharpe</span></div>
          <div class="ms"><span class="ms-val">{sv.get('oos_annual_return',0):.0f}%</span><span class="ms-lbl">Annual</span></div>
          <div class="ms"><span class="ms-val">{sv.get('oos_max_dd',0):.1f}%</span><span class="ms-lbl">Max DD</span></div>
          <div class="ms"><span class="ms-val">{sv.get('oos_win_rate',0)}%</span><span class="ms-lbl">Win Rate</span></div>
        </div>
        <div class="model-feats">
          <span class="feat-tag macro">{sm} macro</span>
          <span class="feat-tag tech">{st} tech</span>
          <span class="feat-tag fund">{sf_n} fund</span>
          <span class="feat-tag total">{stot} total</span>
        </div>
        <div class="model-universe">{shield_meta.get('universe','')}</div>
        <div class="model-trained">Trained: {shield_meta.get('trained_on','')[:25]}</div>
      </div>
      <div class="model-card" style="border-left:3px solid #10b981">
        <div class="model-name" style="color:#10b981">Arrow V11</div>
        <div class="model-algo">{arrow_meta.get('algorithm','')} · {arrow_meta.get('n_trees',0)} trees</div>
        <div class="model-stats">
          <div class="ms"><span class="ms-val">{av.get('oos_sharpe', av.get('wf_avg_net', 0)):.2f}</span><span class="ms-lbl">{'Sharpe' if av.get('oos_sharpe') else 'WF Net'}</span></div>
          <div class="ms"><span class="ms-val">{av.get('oos_avg_net', av.get('wf_avg_net', 0)):.1f}%</span><span class="ms-lbl">5d Net</span></div>
          <div class="ms"><span class="ms-val">{av.get('oos_win_rate', 0)}%</span><span class="ms-lbl">Win Rate</span></div>
          <div class="ms"><span class="ms-val">{av.get('oos_big50', 0)}</span><span class="ms-lbl">Big50</span></div>
        </div>
        <div class="model-feats">
          <span class="feat-tag macro">{am} macro</span>
          <span class="feat-tag tech">{at} tech</span>
          <span class="feat-tag total">{atot} total</span>
        </div>
        <div class="model-universe">{arrow_meta.get('universe','')}</div>
        <div class="model-trained">Trained: {arrow_meta.get('trained_on','')[:25]}</div>
      </div>
    </div>
    
    <div class="tracking-charts">
      <div class="bento-item">
        <div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield V6 — Feature Importance</div>
        <div class="section-count">Top 15 / {stot}</div></div>
        {svg_bar_chart(sf_values, sf_labels, sf_colors, 600, 350)}
      </div>
      <div class="bento-item">
        <div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow V11 — Feature Importance</div>
        <div class="section-count">Top 15 / {atot}</div></div>
        {svg_bar_chart(af_values, af_labels, af_colors, 600, 350)}
      </div>
    </div>
    
    <div class="tracking-charts" style="margin-top:16px">
      {thresholds_html}
    </div>
    
    <div class="bento-item" style="margin-top:16px">
      <div class="section-head"><div class="section-title"><span class="dot" style="background:#f59e0b"></span>XGBoost Hyperparameters</div></div>
      <div class="params-grid">
        <div class="param"><span class="param-key">Objective</span><span class="param-val">{shield_meta.get('params',{}).get('objective','')}</span></div>
        <div class="param"><span class="param-key">Max Depth</span><span class="param-val">{shield_meta.get('params',{}).get('max_depth','')}</span></div>
        <div class="param"><span class="param-key">Learning Rate</span><span class="param-val">{shield_meta.get('params',{}).get('learning_rate','')}</span></div>
        <div class="param"><span class="param-key">Subsample</span><span class="param-val">{shield_meta.get('params',{}).get('subsample','')}</span></div>
        <div class="param"><span class="param-key">Colsample</span><span class="param-val">{shield_meta.get('params',{}).get('colsample_bytree','')}</span></div>
        <div class="param"><span class="param-key">Min Child</span><span class="param-val">{shield_meta.get('params',{}).get('min_child_weight','')}</span></div>
        <div class="param"><span class="param-key">Device</span><span class="param-val">{shield_meta.get('params',{}).get('device','')}</span></div>
        <div class="param"><span class="param-key">Trees</span><span class="param-val">{shield_meta.get('n_trees','')}</span></div>
      </div>
    </div>
    '''

# ============================================================
# HTML ASSEMBLY
# ============================================================
tab1_content = f'''
<div class="metrics">
  <div class="metric"><div class="metric-label">Portfolio</div><div class="metric-value" style="color:#fff">${total_val:,.0f}</div><div class="metric-sub">{len(positions)} positions</div></div>
  <div class="metric"><div class="metric-label">P&L</div><div class="metric-value" style="color:{pnl_c}">{'+' if total_pnl >= 0 else ''}{total_pnl:,.0f}</div><div class="metric-sub">{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct:.1f}%</div></div>
  <div class="metric"><div class="metric-label">VIX</div><div class="metric-value" style="color:{vix_c}">{vix_val}</div><div class="metric-sub">{vix_status}</div></div>
  <div class="metric"><div class="metric-label">Win/Loss</div><div class="metric-value"><span style="color:#10b981">{win_count}</span>/<span style="color:#ef4444">{loss_count}</span></div><div class="metric-sub">today</div></div>
</div>
<div class="bento">
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield Holdings</div><div class="section-count">{len(shield_pos)} active</div></div>
    <div class="pos-grid">{shield_cards if shield_cards else '<div class="empty">No positions</div>'}</div></div>
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow Holdings</div><div class="section-count">{len(arrow_pos)} active</div></div>
    <div class="pos-grid">{arrow_cards if arrow_cards else '<div class="empty">No positions</div>'}</div></div>
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#6366f1"></span>Shield V6 — Top Picks</div><div class="section-count">{v6.get('total',0)} universe</div></div>
    <div class="pick-list">{v6_rows if v6_rows else '<div class="empty">No picks</div>'}</div></div>
  <div class="bento-item"><div class="section-head"><div class="section-title"><span class="dot" style="background:#10b981"></span>Arrow V11 — Top Picks</div><div class="section-count">{v11.get('total',0)} universe</div></div>
    <div class="pick-list">{v11_rows if v11_rows else '<div class="empty">No picks</div>'}</div></div>
</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#10b981"></div>STRONG Top5%</div>
  <div class="legend-item"><div class="legend-dot" style="background:#34d399"></div>BUY Top10%</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f59e0b"></div>WATCH Top20%</div>
  <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div>Skip</div>
</div>'''

tab2_content = tracking_tab()
tab3_content = model_tab()

html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#050508">
<title>Hermes Trading Intelligence</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
:root{{--bg:#050508;--bg-card:rgba(255,255,255,0.03);--bg-card-hover:rgba(255,255,255,0.05);--border:rgba(255,255,255,0.06);--border-light:rgba(255,255,255,0.03);--text:#e8e8ed;--text-dim:#6b6b76;--text-muted:#3d3d47;--green:#10b981;--red:#ef4444;--amber:#f59e0b;--indigo:#6366f1;--purple:#8b5cf6;--radius:16px;--radius-sm:10px;--radius-xs:6px}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,'SF Pro Text',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased;overflow-x:hidden}}
body::before{{content:'';position:fixed;top:-50%;left:-50%;width:200%;height:200%;background:radial-gradient(ellipse at 20% 20%,rgba(99,102,241,0.04) 0%,transparent 50%),radial-gradient(ellipse at 80% 80%,rgba(16,185,129,0.03) 0%,transparent 50%);pointer-events:none;z-index:0}}
.wrap{{max-width:1440px;margin:0 auto;padding:24px 32px;position:relative;z-index:1}}

/* HEADER */
.header{{display:flex;justify-content:space-between;align-items:center;padding:0 0 16px;border-bottom:1px solid var(--border);margin-bottom:8px}}
.header-left{{display:flex;align-items:center;gap:16px}}
.logo{{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:700;letter-spacing:-0.5px;color:#fff}}
.logo-dot{{color:var(--indigo)}}
.header-badge{{font-size:10px;font-weight:600;letter-spacing:0.5px;padding:3px 10px;border-radius:20px;background:rgba(99,102,241,0.1);color:var(--indigo);border:1px solid rgba(99,102,241,0.2)}}
.header-right{{display:flex;align-items:center;gap:16px}}
.timestamp{{font-size:12px;color:var(--text-dim);font-family:'JetBrains Mono',monospace}}
.live-dot{{width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 8px rgba(16,185,129,0.3);animation:pulse 2s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.4}}}}

/* TABS */
.tab-bar{{display:flex;gap:2px;padding:16px 0 0;border-bottom:1px solid var(--border)}}
.tab{{padding:10px 20px;font-size:12px;font-weight:600;letter-spacing:1px;text-transform:uppercase;color:var(--text-dim);cursor:pointer;border-bottom:2px solid transparent;transition:all 0.3s ease;user-select:none}}
.tab:hover{{color:var(--text);background:rgba(255,255,255,0.02)}}
.tab.active{{color:#fff;border-bottom-color:var(--indigo)}}
.tab-content{{display:none;padding-top:16px}}.tab-content.active{{display:block}}

/* METRICS */
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border-radius:var(--radius);overflow:hidden;margin-bottom:16px;border:1px solid var(--border)}}
.metric{{background:var(--bg);padding:16px 20px;transition:background 0.3s}}.metric:hover{{background:rgba(255,255,255,0.02)}}
.metric-label{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:6px}}
.metric-value{{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-0.5px}}
.metric-sub{{font-size:11px;color:var(--text-dim);margin-top:2px;font-family:'JetBrains Mono',monospace}}

/* BENTO */
.bento{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.bento-item{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:20px;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);transition:all 0.4s cubic-bezier(0.32,0.72,0,1);position:relative;overflow:hidden}}
.bento-item::before{{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent)}}
.bento-item:hover{{background:var(--bg-card-hover);border-color:rgba(255,255,255,0.1)}}
.section-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}}
.section-title{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:var(--text-dim);display:flex;align-items:center;gap:10px}}
.section-title .dot{{width:8px;height:8px;border-radius:50%}}
.section-count{{font-size:11px;color:var(--text-muted);font-family:'JetBrains Mono',monospace}}

/* POSITION CARDS */
.pos-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}}
.pos-card{{background:rgba(255,255,255,0.02);border:1px solid var(--border-light);border-radius:var(--radius-sm);padding:12px 14px;transition:all 0.3s cubic-bezier(0.32,0.72,0,1)}}
.pos-card:hover{{background:rgba(255,255,255,0.04);border-color:var(--border);transform:translateY(-2px)}}
.pos-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:2px}}
.pos-code{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#fff;letter-spacing:-0.3px}}
.pos-model{{font-size:8px;font-weight:700;letter-spacing:1px;padding:2px 6px;border-radius:4px;border:1px solid}}
.pos-name{{font-size:10px;color:var(--text-dim);margin-bottom:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.pos-metrics{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:8px}}
.pos-pnl{{text-align:left}}.pnl-sign{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;display:block;letter-spacing:-0.5px}}
.pnl-usd{{font-family:'JetBrains Mono',monospace;font-size:10px;opacity:0.7;display:block}}
.pos-price{{text-align:right}}.price-cur{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;color:var(--text);display:block}}
.price-cost{{font-size:9px;color:var(--text-muted);display:block}}
.pos-progress{{margin-top:6px}}.prog-label{{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text-muted);margin-bottom:3px}}
.prog-bar{{height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden}}.prog-fill{{height:100%;border-radius:2px;transition:width 0.8s cubic-bezier(0.32,0.72,0,1)}}
.stop-badge{{margin-top:6px;font-size:9px;font-weight:700;letter-spacing:1px;color:var(--red);background:rgba(239,68,68,0.1);padding:3px 8px;border-radius:4px;text-align:center}}
.warn-badge{{margin-top:6px;font-size:9px;font-weight:700;letter-spacing:1px;color:var(--amber);background:rgba(245,158,11,0.1);padding:3px 8px;border-radius:4px;text-align:center}}

/* PICK ROWS */
.pick-list{{display:flex;flex-direction:column;gap:1px}}
.pick-row{{display:grid;grid-template-columns:24px 90px 1fr 50px 40px;align-items:center;gap:10px;padding:6px 10px;border-radius:var(--radius-xs);transition:background 0.2s}}
.pick-row:hover{{background:rgba(255,255,255,0.03)}}
.pick-rank{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-muted);text-align:center}}
.pick-info{{display:flex;flex-direction:column}}.pick-ticker{{font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;color:#fff;letter-spacing:-0.3px}}
.pick-price{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim)}}
.pick-score-bar{{display:flex;align-items:center;gap:8px}}.score-track{{flex:1;height:4px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden}}
.score-fill{{height:100%;border-radius:2px;transition:width 1s cubic-bezier(0.32,0.72,0,1)}}
.score-val{{font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;min-width:40px;text-align:right}}
.pick-signal{{font-size:8px;font-weight:700;letter-spacing:1px;padding:2px 8px;border-radius:4px;border:1px solid;text-align:center}}
.pick-hold{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-muted);text-align:right}}

/* LEGEND */
.legend{{display:flex;gap:16px;justify-content:center;padding:12px;border-top:1px solid var(--border);margin-top:8px}}
.legend-item{{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text-dim);letter-spacing:0.5px}}
.legend-dot{{width:7px;height:7px;border-radius:2px}}

/* TRACKING TAB */
.tracking-summary{{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:16px}}
.stat-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-sm);padding:16px;text-align:center}}
.stat-label{{font-size:9px;font-weight:700;letter-spacing:1.5px;color:var(--text-dim);margin-bottom:6px}}
.stat-value{{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#fff}}
.stat-sub{{font-size:10px;color:var(--text-muted);margin-top:4px;display:flex;justify-content:center}}
.tracking-charts{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}

/* CHART SVG */
.chart-svg{{width:100%;height:auto}}.chart-label{{font-family:'JetBrains Mono',monospace;font-size:9px;fill:var(--text-muted)}}
.chart-value{{font-family:'JetBrains Mono',monospace;fill:var(--text-dim)}}
.sparkline{{width:60px;height:16px;vertical-align:middle}}
.donut-svg{{width:60px;height:60px}}.donut-text{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;fill:#fff}}
.donut-fill{{transition:stroke-dashoffset 1.5s cubic-bezier(0.32,0.72,0,1)}}

/* HEATMAP */
.heat-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}}
.heat-cell{{padding:8px;border-radius:var(--radius-xs);text-align:center;cursor:default;transition:transform 0.2s}}
.heat-cell:hover{{transform:scale(1.05)}}
.heat-month{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim)}}
.heat-ret{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;margin:2px 0}}
.heat-wr{{font-size:9px;color:var(--text-muted)}}

/* DISTRIBUTION */
.dist-chart{{display:flex;flex-direction:column;gap:4px}}
.dist-row{{display:grid;grid-template-columns:70px 1fr 40px;align-items:center;gap:8px}}
.dist-label{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-dim);text-align:right}}
.dist-bar{{height:14px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden}}
.dist-fill{{height:100%;border-radius:3px;transition:width 0.8s ease}}
.dist-count{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--text-muted)}}

/* MODEL TAB */
.model-cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.model-card{{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);padding:24px}}
.model-name{{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;margin-bottom:4px}}
.model-algo{{font-size:11px;color:var(--text-dim);margin-bottom:16px}}
.model-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px}}
.ms{{text-align:center;padding:8px;background:rgba(255,255,255,0.02);border-radius:var(--radius-xs)}}
.ms-val{{font-family:'JetBrains Mono',monospace;font-size:16px;font-weight:700;color:#fff;display:block}}
.ms-lbl{{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px}}
.model-feats{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}}
.feat-tag{{font-size:9px;font-weight:600;letter-spacing:0.5px;padding:3px 8px;border-radius:4px}}
.feat-tag.macro{{background:rgba(99,102,241,0.1);color:#6366f1;border:1px solid rgba(99,102,241,0.2)}}
.feat-tag.tech{{background:rgba(16,185,129,0.1);color:#10b981;border:1px solid rgba(16,185,129,0.2)}}
.feat-tag.fund{{background:rgba(245,158,11,0.1);color:#f59e0b;border:1px solid rgba(245,158,11,0.2)}}
.feat-tag.total{{background:rgba(255,255,255,0.05);color:var(--text-dim);border:1px solid var(--border)}}
.model-universe{{font-size:11px;color:var(--text-dim);margin-bottom:4px}}
.model-trained{{font-size:10px;color:var(--text-muted)}}

/* META TABLE */
.meta-table{{width:100%;border-collapse:collapse}}
.meta-table th{{font-size:10px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;text-align:left;padding:4px 8px;border-bottom:1px solid var(--border)}}
.meta-table td{{padding:6px 8px;border-bottom:1px solid var(--border-light);font-size:12px;font-family:'JetBrains Mono',monospace}}

/* PARAMS */
.params-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}}
.param{{padding:10px;background:rgba(255,255,255,0.02);border-radius:var(--radius-xs);text-align:center}}
.param-key{{font-size:9px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.5px;display:block;margin-bottom:4px}}
.param-val{{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:#fff}}

/* EMPTY */
.empty{{text-align:center;padding:24px;color:var(--text-muted);font-size:12px}}

/* FOOTER */
.footer{{text-align:center;padding:16px 0;border-top:1px solid var(--border);margin-top:8px}}
.footer-text{{font-size:10px;color:var(--text-muted);letter-spacing:0.5px}}

/* ANIMATIONS */
@keyframes fadeUp{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1;transform:translateY(0)}}}}
.bento-item,.stat-card,.model-card{{animation:fadeUp 0.5s cubic-bezier(0.32,0.72,0,1) both}}
.pos-card{{animation:fadeUp 0.4s cubic-bezier(0.32,0.72,0,1) both}}
.pick-row{{animation:fadeUp 0.3s cubic-bezier(0.32,0.72,0,1) both}}

/* RESPONSIVE */
@media(max-width:900px){{.wrap{{padding:16px}}.bento,.tracking-charts,.model-cards{{grid-template-columns:1fr}}.metrics,.tracking-summary{{grid-template-columns:repeat(2,1fr)}}.pos-grid{{grid-template-columns:1fr 1fr}}.pick-row{{grid-template-columns:20px 70px 1fr 40px 30px}}.heat-grid{{grid-template-columns:repeat(4,1fr)}}.params-grid{{grid-template-columns:repeat(2,1fr)}}.model-stats{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:480px){{.pos-grid{{grid-template-columns:1fr}}.tracking-summary{{grid-template-columns:repeat(2,1fr)}}.heat-grid{{grid-template-columns:repeat(3,1fr)}}}}

::-webkit-scrollbar{{width:6px}}::-webkit-scrollbar-track{{background:transparent}}::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.1);border-radius:3px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="header-left"><div class="logo">HERMES<span class="logo-dot">.</span></div><div class="header-badge">INTELLIGENCE PLATFORM</div></div>
    <div class="header-right"><div class="live-dot"></div><div class="timestamp">{now}</div></div>
  </div>
  
  <div class="tab-bar">
    <div class="tab active" onclick="showTab(0)">Trading</div>
    <div class="tab" onclick="showTab(1)">Tracking</div>
    <div class="tab" onclick="showTab(2)">Models</div>
  </div>
  
  <div class="tab-content active" id="tab-0">{tab1_content}</div>
  <div class="tab-content" id="tab-1">{tab2_content}</div>
  <div class="tab-content" id="tab-2">{tab3_content}</div>
  
  <div class="footer"><div class="footer-text">Hermes Trading Intelligence · Shield V6 + Arrow V11 · Three-Layer Signal Filter</div></div>
</div>

<script>
function showTab(n){{
  document.querySelectorAll('.tab-content').forEach((s,i)=>{{s.classList.toggle('active',i===n)}});
  document.querySelectorAll('.tab').forEach((t,i)=>{{t.classList.toggle('active',i===n)}});
}}
</script>
</body>
</html>'''

out = os.path.join(ROOT, 'dashboard.html')
with open(out, 'w') as f:
    f.write(html)
print(f'Dashboard v6 Triple-Tab: {len(html):,} bytes')
print(f'  Positions: {len(positions)} | Shield: {len(shield_pos)} | Arrow: {len(arrow_pos)}')
print(f'  Portfolio: ${total_val:,.0f} | P&L: {total_pnl:+,.0f}')
print(f'  Tracking: {shield_stats.get("total_trades",0)} shield + {arrow_stats.get("total_trades",0)} arrow trades')
