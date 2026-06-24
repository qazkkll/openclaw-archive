#!/usr/bin/env python3
"""Hermes Trading Dashboard v3 — 表格化推荐 + 持仓分类"""
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
shield_m = load('models/us/blueshield_v8_meta.json')
arrow_m = load('models/us/arrow_v12_meta.json')

tech = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position','ret1','ret5','ret20','ret60','momentum_6m','momentum_1m','mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change','rsi14','rsi_change','macd','macd_signal','macd_hist','bb_std','bb_width','bb_pos','ret_quality']
macro_l = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60','qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60','iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
fund = ['pe_trailing','pe_forward','div_yield','beta']

def fc(n):
    if n in tech: return '#2196f3'
    if n in macro_l: return '#ffc107'
    if n in fund: return '#4caf50'
    return '#9c27b0'

def feat_bars(meta, n=15):
    fi = meta.get('feature_importance', {})
    if not fi: return ''
    s = sorted(fi.items(), key=lambda x: -x[1])[:n]
    mx = s[0][1] if s else 1
    h = ''
    for nm, v in s:
        p = v/mx*100
        c = fc(nm)
        h += f'<div style="display:flex;align-items:center;gap:4px;margin:2px 0;font-size:11px"><span style="width:75px;text-align:right;color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{nm}</span><div style="flex:1;height:10px;background:#1a1a1a;border-radius:2px;overflow:hidden"><div style="height:100%;width:{p:.0f}%;background:{c};border-radius:2px"></div></div><span style="width:30px;color:#666;font-size:10px">{v:.1f}</span></div>'
    return h

# ── 持仓HTML ──
positions = futu.get('positions', [])
shield_pos = sorted([p for p in positions if p.get('cost_price',0) > 10], key=lambda x: -x['pnl_pct'])
arrow_pos = sorted([p for p in positions if p.get('cost_price',0) <= 10], key=lambda x: -x['pnl_pct'])

# 加载持仓评分
held_scores = load('output/held_scores.json')
held_map = {}
for h in held_scores.get('shield', []) + held_scores.get('arrow', []):
    held_map[h['sym']] = h

def pos_row(p):
    pc = 'gn' if p['pnl_pct'] >= 0 else 'rd'
    held = p.get('days_held') or 0
    hd = p.get('hold_days') or 20
    prog = min(100, held/hd*100)
    exp = p.get('days_to_expiry', hd)
    bc = '#4caf50' if (exp or 0) <= 0 else '#ffc107' if prog < 50 else '#2196f3'
    sw = '<span style="color:#f44336">止损!</span>' if p.get('stop_triggered') else ''
    # 持仓评分（排名百分位）
    hs = held_map.get(p['code'], {})
    pct_c = ''
    if hs:
        rank = hs.get('rank', 0)
        total = hs.get('total', 1)
        score = hs.get('score', 0)
        pct = rank / total * 100 if total > 0 else 100
        if pct <= 10: pct_c = 'gn'; pct_l = f'前{pct}%🟢'
        elif pct <= 30: pct_c = 'yl'; pct_l = f'前{pct}%🟡'
        elif pct <= 60: pct_c = ''; pct_l = f'前{pct}%'
        else: pct_c = 'rd'; pct_l = f'前{pct}%🔴'
        model_note = f'{score:.3f} {pct_l} ({rank}/{total})'
    else:
        model_note = '<span class="dm">无评分</span>'
    return f'''<tr>
<td><b>{p["code"]}</b></td>
<td class="dm">{p["name"][:14]}</td>
<td>{p["qty"]}</td>
<td>${p["cost_price"]:.2f}</td>
<td>${p["current_price"]:.2f}</td>
<td class="{pc}"><b>{p["pnl_pct"]:+.1f}%</b></td>
<td class="{pc}">${p["pnl_usd"]:+.0f}</td>
<td class="dm">{p.get("buy_date","?")}</td>
<td><div style="display:flex;align-items:center;gap:3px"><div style="width:40px;height:4px;background:#222;border-radius:2px;overflow:hidden"><div style="height:100%;width:{prog:.0f}%;background:{bc};border-radius:2px"></div></div><span class="dm" style="font-size:9px">{held}/{hd}d</span></div></td>
<td class="{pct_c}">{model_note}</td>
<td>{sw}</td>
</tr>'''

shield_rows = ''.join(pos_row(p) for p in shield_pos)
arrow_rows = ''.join(pos_row(p) for p in arrow_pos)

# ── 推荐表格HTML ──
def picks_table(data, n, show_rsi=False):
    h = ''
    for i, pk in enumerate(data.get('picks',[])[:n]):
        sig = pk.get('signal','🔴')
        sig_c = '#4caf50' if '🟢' in sig else '#ffc107' if '🟡' in sig else '#f44336'
        if sig == '🟢🟢': adv = '🟢🟢买'
        elif sig == '🟢': adv = '🟢买'
        elif sig == '🟡': adv = '🟡观'
        else: adv = '🔴—'
        rsi = f'<td class="dm">{pk.get("rsi",0):.0f}</td>' if show_rsi else ''
        ret5 = f'<td class="{"gn" if pk.get("ret_5d",0)>=0 else "rd"}">{pk.get("ret_5d",0):+.1f}%</td>' if show_rsi else ''
        ret20 = f'<td class="{"gn" if pk.get("ret_20d",0)>=0 else "rd"}">{pk.get("ret_20d",0):+.1f}%</td>' if show_rsi else ''
        held = '✓' if any(x['code']==pk['ticker'] for x in positions) else ''
        h += f'<tr><td class="dm">{i+1}</td><td><b>{pk["ticker"]}</b>{held}</td><td>${pk["price"]:.2f}</td><td>{pk["pred_rank"]:.4f}</td><td style="color:{sig_c}">{sig}</td>{rsi}{ret5}{ret20}<td style="color:{sig_c}">{adv}</td></tr>'
    return h

v6_table = picks_table(v6, 15, show_rsi=True)
v11_table = picks_table(v11, 5)

vix = futu.get('vix')
vix_s = f'{vix:.1f}' if vix else 'N/A'
vix_c = '#4caf50' if vix and vix<20 else '#ffc107' if vix and vix<25 else '#ff9800' if vix and vix<35 else '#f44336' if vix else '#888'
tv = futu.get('total_value',0)
tp = futu.get('total_pnl',0)
tp_pct = futu.get('total_pnl_pct',0)
tn = futu.get('total_positions',0)
stops = [p for p in positions if p.get('stop_triggered')]
now = datetime.now().strftime('%Y-%m-%d %H:%M')

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta http-equiv="refresh" content="300">
<title>Hermes Trading System</title>
<style>
:root{{--bg:#0a0a0a;--c1:#111;--c2:#1a1a1a;--bd:#222;--tx:#e0e0e0;--dm:#888;--gn:#4caf50;--yl:#ffc107;--rd:#f44336;--bl:#2196f3;--or:#ff9800;--pu:#9c27b0}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--tx);font-family:'SF Mono','Consolas',monospace;font-size:13px;padding:8px}}
.hdr{{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:var(--c1);border-radius:6px;margin-bottom:8px;flex-wrap:wrap;gap:6px}}
.hdr h1{{font-size:15px;color:var(--bl);white-space:nowrap}}
.hdr .m{{color:var(--dm);font-size:11px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}}
.s{{padding:10px 12px;background:var(--c1);border-radius:6px;margin-bottom:8px;border-left:3px solid var(--bd)}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
@media(max-width:800px){{.cols{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th{{text-align:left;color:var(--dm);font-weight:500;padding:4px 6px;border-bottom:1px solid var(--bd);font-size:10px;white-space:nowrap}}
td{{padding:3px 6px;border-bottom:1px solid #151515;white-space:nowrap}}
tr:hover{{background:#151515}}
.gn{{color:var(--gn)}}.rd{{color:var(--rd)}}.yl{{color:var(--yl)}}.dm{{color:var(--dm)}}.tx{{color:var(--tx)}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:4px}}
.sc{{background:var(--c2);padding:6px;border-radius:4px;text-align:center}}
.sc .v{{font-size:18px;font-weight:bold}}.sc .l{{color:var(--dm);font-size:9px;margin-top:1px}}
.b{{display:inline-block;font-size:9px;padding:1px 5px;border-radius:8px;background:rgba(76,175,80,0.12);color:var(--gn)}}
.ri{{padding:5px 8px;margin:2px 0;border-radius:3px;background:var(--c2);font-size:11px}}
.ri.h{{border-left:3px solid var(--rd)}}.ri.m{{border-left:3px solid var(--yl)}}.ri.lo{{border-left:3px solid var(--gn)}}
.tb{{display:none;gap:3px;margin-bottom:8px;overflow-x:auto}}
.t{{padding:5px 10px;border-radius:3px;background:var(--c1);color:var(--dm);font-size:10px;cursor:pointer;white-space:nowrap;border:1px solid var(--bd)}}
.t.a{{background:var(--bl);color:white;border-color:var(--bl)}}
@media(max-width:800px){{.tb{{display:flex}}.s{{display:none}}.s.ta{{display:block}}.cols{{display:block}}.cols>.s{{display:block;margin-bottom:8px}}}}
</style>
</head>
<body>
<div class="hdr">
<h1>📊 Hermes Trading System</h1>
<div class="m">
<span>{now}</span>
<span style="color:{vix_c}">VIX {vix_s}</span>
<span>持仓{tn}只 · ${tv:,.0f} · <span class="{'gn' if tp>=0 else 'rd'}">{tp:+,.0f}({tp_pct:+.1f}%)</span></span>
</div>
</div>

<div class="tb">
<div class="t a" onclick="st('mk')">📈市场</div>
<div class="t" onclick="st('md')">🤖模型</div>
<div class="t" onclick="st('pf')">💼持仓</div>
<div class="t" onclick="st('ft')">🔬特征</div>
<div class="t" onclick="st('rk')">⚠️风险</div>
</div>

<!-- 市场 -->
<div class="s ta" id="s-mk" style="border-left-color:var(--bl)">
<h2 style="font-size:13px;margin-bottom:6px">📈 市场环境</h2>
<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:12px">
<span><span class="dm">SPX</span> 5,546 <span class="gn">+0.5%</span></span>
<span><span class="dm">VIX</span> <b style="color:{vix_c}">{vix_s}</b></span>
<span><span class="dm">NDX</span> 17,890 <span class="gn">+0.8%</span></span>
<span><span class="dm">DJI</span> 42,345 <span class="gn">+0.3%</span></span>
</div>
<div class="dm" style="margin-top:6px;font-size:10px">
VIX止损: <span class="gn">&lt;20全仓</span> · <span class="yl">20-25注意</span> · <span style="color:var(--or)">25-35减仓</span> · <span class="rd">&gt;35清仓</span>
</div>
</div>

<!-- 模型 -->
<div class="cols">
<div class="s" id="s-md" style="border-left-color:var(--bl)">
<h2 style="font-size:13px">🛡️ 蓝盾V6 <span class="b">生产中</span></h2>
<div class="sg">
<div class="sc"><div class="v gn">1.44</div><div class="l">夏普</div></div>
<div class="sc"><div class="v gn">+30.1%</div><div class="l">年化</div></div>
<div class="sc"><div class="v">60%</div><div class="l">胜率</div></div>
<div class="sc"><div class="v rd">-11.1%</div><div class="l">DD</div></div>
</div>
<div class="dm" style="margin-top:4px;font-size:10px">44维(技27+宏13+基4) · 20天Top-15 · &gt;$10 · 扫描{v6.get("total_scanned","?")}只</div>
<div style="font-size:11px;color:var(--dm);margin:8px 0 4px">📊 今日推荐 Top-15</div>
<table>
<tr><th>#</th><th>代码</th><th>价格</th><th>排名分</th><th>信号</th><th>RSI</th><th>5日</th><th>20日</th><th>建议</th></tr>
{v6_table}
</table>
</div>

<div class="s" id="s-md2" style="border-left-color:var(--gn)">
<h2 style="font-size:13px">🎯 绿箭V11 <span class="b">生产中</span></h2>
<div class="sg">
<div class="sc"><div class="v gn">2.18</div><div class="l">夏普</div></div>
<div class="sc"><div class="v gn">+5.56%</div><div class="l">净/5天</div></div>
<div class="sc"><div class="v">50%</div><div class="l">胜率</div></div>
<div class="sc"><div class="v rd">-12.1%</div><div class="l">DD</div></div>
</div>
<div class="dm" style="margin-top:4px;font-size:10px">42维(技28+宏13) · 5天Top-5 · $1-$10 · 扫描{v11.get("total_scanned","?")}只</div>
<div style="font-size:11px;color:var(--dm);margin:8px 0 4px">📊 今日推荐 Top-5</div>
<table>
<tr><th>#</th><th>代码</th><th>价格</th><th>排名分</th><th>信号</th><th>建议</th></tr>
{v11_table}
</table>
</div>
</div>

<!-- 持仓 -->
<div class="s ta" id="s-pf" style="border-left-color:var(--yl)">
<h2 style="font-size:13px;margin-bottom:4px">💼 持仓 <span class="dm">({tn}只 · ${tv:,.0f} · <span class="{'gn' if tp>=0 else 'rd'}">{tp:+,.0f}({tp_pct:+.1f}%)</span>)</span></h2>
<div style="font-size:10px;color:var(--dm);margin-bottom:6px">蓝盾20天/止损-15% · 绿箭5天/止损-10% · ⚠️未到期不卖(除非止损)</div>

<div style="font-size:11px;color:var(--bl);margin:6px 0 3px">🛡️ 蓝盾V6持仓 ({len(shield_pos)}只)</div>
<table>
<tr><th>代码</th><th>名称</th><th>数量</th><th>成本</th><th>当前</th><th>盈亏%</th><th>盈亏$</th><th>买入</th><th>进度</th><th>止损</th><th>模型</th><th></th></tr>
{shield_rows if shield_rows else '<tr><td colspan="12" class="dm" style="text-align:center;padding:8px">无蓝盾持仓</td></tr>'}
</table>

<div style="font-size:11px;color:var(--gn);margin:10px 0 3px">🎯 绿箭V11持仓 ({len(arrow_pos)}只)</div>
<table>
<tr><th>代码</th><th>名称</th><th>数量</th><th>成本</th><th>当前</th><th>盈亏%</th><th>盈亏$</th><th>买入</th><th>进度</th><th>止损</th><th>模型</th><th></th></tr>
{arrow_rows if arrow_rows else '<tr><td colspan="12" class="dm" style="text-align:center;padding:8px">无绿箭持仓</td></tr>'}
</table>
</div>

<!-- 特征 -->
<div class="cols">
<div class="s" id="s-ft" style="border-left-color:var(--pu)">
<h2 style="font-size:13px">🔬 蓝盾特征 <span style="font-size:9px;color:var(--bl)">●技</span> <span style="font-size:9px;color:var(--yl)">●宏</span> <span style="font-size:9px;color:var(--gn)">●基</span></h2>
{feat_bars(shield_m)}
</div>
<div class="s" id="s-ft2" style="border-left-color:var(--pu)">
<h2 style="font-size:13px">🔬 绿箭特征</h2>
{feat_bars(arrow_m)}
</div>
</div>

<!-- 风险 -->
<div class="s ta" id="s-rk" style="border-left-color:var(--rd)">
<h2 style="font-size:13px">⚠️ 风险预警</h2>
{''.join(f'<div class="ri h">🔴 {s["code"]} 止损! ${s["current_price"]:.2f}≤${s["stop_loss_price"]:.2f}</div>' for s in stops)}
{''.join(f'<div class="ri m">🟡 {s["code"]} 接近止损 ${s["current_price"]:.2f} vs ${s["stop_loss_price"]:.2f}</div>' for s in positions if not s.get("stop_triggered") and s["current_price"]<=s["stop_loss_price"]*1.05 and s["qty"]>0)}
{'<div class="ri lo">🟢 市场正常 VIX<20</div>' if vix and vix<20 else ''}
{'' if stops or (vix and vix>20) else '<div class="ri lo">✅ 无风险预警</div>'}
</div>

<!-- 信号说明 -->
<div class="s" style="border-left-color:var(--dm)">
<div style="font-size:11px;display:flex;gap:10px;flex-wrap:wrap">
<span><span class="gn">🟢🟢</span> Top5%精品买</span>
<span><span class="gn">🟢</span> Top10%强信号</span>
<span><span class="yl">🟡</span> Top20%观察</span>
<span><span class="rd">🔴</span> 低于中位数/VIX>30</span>
</div>
</div>

<div style="text-align:center;color:#555;font-size:9px;padding:8px">Hermes Trading System v3.0 · 蓝盾V6 + 绿箭V11 · {now}</div>
<script>
function st(n){{document.querySelectorAll('.s').forEach(s=>s.classList.remove('ta'));document.querySelectorAll('.t').forEach(t=>t.classList.remove('a'));event.target.classList.add('a');document.querySelectorAll('[id^="s-'+n+'"]').forEach(s=>s.classList.add('ta'))}}
</script>
</body>
</html>'''

out = os.path.join(ROOT, 'dashboard.html')
with open(out, 'w') as f:
    f.write(html)
print(f'✅ v3.0 Dashboard: {len(html):,} bytes')
print(f'   持仓: {tn}只 | 蓝盾: {len(shield_pos)}只 | 绿箭: {len(arrow_pos)}只')
print(f'   蓝盾推荐: {len(v6.get("picks",[]))}只 | 绿箭推荐: {len(v11.get("picks",[]))}只')
if stops:
    print(f'   ⚠️ 止损: {[s["code"] for s in stops]}')
