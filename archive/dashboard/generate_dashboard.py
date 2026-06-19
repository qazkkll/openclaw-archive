#!/usr/bin/env python3
"""
Hermes量化系统 — 综合看板生成器 v2
新增：推荐追踪标签页（活跃推荐+历史记录+统计）
"""
import json, os, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_meta(model_name):
    path = os.path.join(ROOT, 'models', 'us', f'{model_name}_meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_config():
    path = os.path.join(ROOT, 'config.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def load_recommendations():
    path = os.path.join(ROOT, 'output', 'recommendations.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'recommendations': [], 'stats': {}}

def get_vix():
    try:
        import yfinance as yf
        vix = yf.Ticker('^VIX').history(period='1d')['Close'].iloc[-1]
        return float(vix)
    except:
        return None

def vix_level(v):
    if v is None: return ('❓','未知','#666')
    if v < 15: return ('🟢','极度平静','#22c55e')
    if v < 20: return ('🟢','正常','#22c55e')
    if v < 25: return ('🟡','注意','#eab308')
    if v < 35: return ('🟠','警戒','#f97316')
    return ('🔴','恐慌','#ef4444')

def generate_html():
    shield = load_meta('blueshield_v6')
    arrow = load_meta('arrow_v11')
    config = load_config()
    recs_data = load_recommendations()
    vix_val = get_vix()
    vix_emoji, vix_text, vix_color = vix_level(vix_val)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Feature importance data
    shield_fi = shield.get('feature_importance', {}) if shield else {}
    arrow_fi = arrow.get('feature_importance', {}) if arrow else {}
    shield_sorted = sorted(shield_fi.items(), key=lambda x: -x[1])
    arrow_sorted = sorted(arrow_fi.items(), key=lambda x: -x[1])

    # Feature categories
    tech_feats = ['ma5','ma20','ma60','ma_bias20','ma_align','price_position',
        'ret1','ret5','ret20','ret60','momentum_6m','momentum_1m',
        'mom_divergence','trend_accel','vol20','vol5','vol_ratio','vol_change',
        'rsi14','rsi_change','macd','macd_signal','macd_hist',
        'bb_std','bb_width','bb_pos','ret_quality']
    macro_feats = ['vix_close','spy_ret1','spy_ret5','spy_ret20','spy_ret60',
        'qqq_ret1','qqq_ret5','qqq_ret20','qqq_ret60',
        'iwm_ret1','iwm_ret5','iwm_ret20','iwm_ret60']
    fund_feats = ['pe_trailing','pe_forward','div_yield','beta']
    extra_feats = ['price','range_pct']

    def feat_category(name):
        if name in tech_feats: return '技术面'
        if name in macro_feats: return '宏观面'
        if name in fund_feats: return '基本面'
        if name in extra_feats: return '特殊'
        return '其他'

    def feat_cat_color(name):
        cat = feat_category(name)
        colors = {'技术面':'#3b82f6','宏观面':'#f59e0b','基本面':'#10b981','特殊':'#8b5cf6'}
        return colors.get(cat, '#6b7280')

    # Signal thresholds
    shield_signals = [
        ('🟢🟢','精品买入','≥0.90','35.24%/20天','94%','17笔','马上排队'),
        ('🟢','强信号','≥0.80','26.77%/20天','84%','209笔','主力信号'),
        ('🟡','观察池','≥0.70','基准以上','—','927笔','不买，写watchlist'),
    ]
    arrow_signals = [
        ('🟢🟢','精品买入','≥0.90','16.25%/5天','54%','72笔','每只$1000'),
        ('🟢','强信号','≥0.80','16.02%/5天','59%','225笔','每只$1000'),
    ]

    # Evolution history
    evolution = [
        {'model':'蓝盾','versions':[
            {'v':'V3','desc':'公式打分110分','sharpe':'0.77','status':'已归档','note':'趋势30+动量25+MACD25'},
            {'v':'V4','desc':'5天ML分类','sharpe':'1.13(WF)','status':'已归档','note':'成本过高，交易频繁'},
            {'v':'V5','desc':'120天排名','sharpe':'1.36→0.67','status':'已归档','note':'S&P500幸存者偏差'},
            {'v':'V6','desc':'20天全市场排名','sharpe':'1.44','status':'✅ 生产中','note':'+宏观+基本面，44维'},
        ]},
        {'model':'绿箭', 'versions':[
            {'v':'V9','desc':'Lottery彩票','sharpe':'—','status':'已归档','note':'命中率1.1%，太低'},
            {'v':'V11','desc':'$1-$10排名','sharpe':'2.18','status':'✅ 生产中','note':'5天持有，净+5.56%/5天'},
        ]},
    ]

    # Key lessons
    lessons = [
        ('排名>>回归','预测"谁排前面"比"涨多少"容易10倍。V6回归测试相关性仅0.026，排名模型有效。'),
        ('宏观特征是关键突破','VIX+SPY/QQQ收益加入后，夏普+42%(0.434→0.618)。vol20+vix_close+spy_ret60是Top3特征。'),
        ('全市场验证','S&P500幸存者偏差严重，夏普从1.36降到0.666（衰减51%）。所有验证必须全市场。'),
        ('🟢🟢信号稀缺但精准','蓝盾≥0.90仅17笔但94%胜率+35%收益。信号越少越准，宁缺毋滥。'),
        ('绿箭随机买亏81%','模型选vs随机选：+95.6% vs -81%。penny stock没有模型就是送钱。'),
        ('VIX止损层','模型不管风控。VIX>25减仓，VIX>35清仓。这是独立于模型的保护层。'),
    ]

    # ============ BUILD HTML ============
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Hermes量化系统</title>
<style>
:root {{
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #e2e8f0; --dim: #94a3b8; --accent: #3b82f6;
    --green: #22c55e; --yellow: #eab308; --orange: #f97316; --red: #ef4444;
    --purple: #8b5cf6;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'SF Pro', 'Helvetica Neue', sans-serif; background:var(--bg); color:var(--text); padding:12px; max-width:900px; margin:0 auto; -webkit-font-smoothing:antialiased; }}
.header {{ text-align:center; padding:20px 0; border-bottom:1px solid var(--border); margin-bottom:16px; }}
.header h1 {{ font-size:24px; font-weight:700; }}
.header .sub {{ color:var(--dim); font-size:13px; margin-top:4px; }}
.vix-banner {{ background:var(--card); border-radius:12px; padding:16px; margin-bottom:16px; border-left:4px solid {vix_color}; display:flex; align-items:center; gap:12px; }}
.vix-banner .emoji {{ font-size:32px; }}
.vix-banner .info {{ flex:1; }}
.vix-banner .val {{ font-size:28px; font-weight:700; color:{vix_color}; }}
.vix-banner .label {{ color:var(--dim); font-size:13px; }}
.section {{ margin-bottom:20px; }}
.section-title {{ font-size:16px; font-weight:600; color:var(--accent); margin-bottom:10px; display:flex; align-items:center; gap:6px; }}
.card {{ background:var(--card); border-radius:10px; padding:14px; margin-bottom:10px; border:1px solid var(--border); }}
.card-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
.card-title {{ font-size:15px; font-weight:600; }}
.badge {{ font-size:11px; padding:2px 8px; border-radius:20px; font-weight:600; }}
.badge-live {{ background:rgba(34,197,94,0.15); color:var(--green); }}
.badge-archived {{ background:rgba(148,163,184,0.15); color:var(--dim); }}
.metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(100px,1fr)); gap:8px; margin-top:10px; }}
.metric {{ text-align:center; padding:8px 4px; background:rgba(255,255,255,0.03); border-radius:6px; }}
.metric .val {{ font-size:20px; font-weight:700; }}
.metric .lbl {{ font-size:11px; color:var(--dim); margin-top:2px; }}
.metric .val.green {{ color:var(--green); }}
.metric .val.yellow {{ color:var(--yellow); }}
.metric .val.red {{ color:var(--red); }}
.feat-bar {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:12px; }}
.feat-bar .name {{ width:100px; text-align:right; color:var(--dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.feat-bar .bar {{ flex:1; height:16px; background:rgba(255,255,255,0.05); border-radius:4px; overflow:hidden; position:relative; }}
.feat-bar .bar-fill {{ height:100%; border-radius:4px; transition:width 0.3s; }}
.feat-bar .pct {{ width:40px; font-size:11px; color:var(--dim); }}
.signal-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.signal-table th {{ text-align:left; color:var(--dim); font-weight:500; padding:6px 4px; border-bottom:1px solid var(--border); }}
.signal-table td {{ padding:6px 4px; border-bottom:1px solid rgba(255,255,255,0.05); }}
.timeline {{ position:relative; padding-left:20px; }}
.timeline::before {{ content:''; position:absolute; left:8px; top:0; bottom:0; width:2px; background:var(--border); }}
.timeline-item {{ position:relative; margin-bottom:12px; padding:10px 12px; background:var(--card); border-radius:8px; border:1px solid var(--border); }}
.timeline-item::before {{ content:''; position:absolute; left:-16px; top:14px; width:10px; height:10px; border-radius:50%; background:var(--accent); border:2px solid var(--bg); }}
.timeline-item.active::before {{ background:var(--green); box-shadow:0 0 8px rgba(34,197,94,0.5); }}
.timeline-item .ver {{ font-weight:700; font-size:14px; }}
.timeline-item .desc {{ color:var(--dim); font-size:12px; margin-top:2px; }}
.timeline-item .stats {{ font-size:12px; margin-top:4px; }}
.lesson {{ padding:10px 12px; background:var(--card); border-radius:8px; border-left:3px solid var(--purple); margin-bottom:8px; font-size:13px; }}
.lesson strong {{ color:var(--purple); }}
.cat-tag {{ display:inline-block; font-size:10px; padding:1px 6px; border-radius:10px; margin-left:4px; }}
.cat-tech {{ background:rgba(59,130,246,0.15); color:#3b82f6; }}
.cat-macro {{ background:rgba(245,158,11,0.15); color:#f59e0b; }}
.cat-fund {{ background:rgba(16,185,129,0.15); color:#10b981; }}
.cat-extra {{ background:rgba(139,92,246,0.15); color:#8b5cf6; }}
.compare-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
@media(max-width:600px) {{
    .compare-grid {{ grid-template-columns:1fr; }}
    .metric-grid {{ grid-template-columns:repeat(3,1fr); }}
}}
.tab-bar {{ display:flex; gap:4px; margin-bottom:12px; overflow-x:auto; }}
.tab {{ padding:8px 16px; border-radius:8px; background:var(--card); color:var(--dim); font-size:13px; cursor:pointer; white-space:nowrap; border:1px solid var(--border); }}
.tab.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
.section-content {{ display:none; }}
.section-content.active {{ display:block; }}
.footer {{ text-align:center; color:var(--dim); font-size:11px; padding:20px 0; border-top:1px solid var(--border); margin-top:20px; }}

/* 推荐追踪样式 */
.rec-card {{ background:var(--card); border-radius:8px; padding:10px 12px; margin-bottom:8px; border:1px solid var(--border); }}
.rec-card.active {{ border-left:3px solid var(--green); }}
.rec-card.stopped {{ border-left:3px solid var(--red); opacity:0.7; }}
.rec-card.expired {{ border-left:3px solid var(--dim); opacity:0.6; }}
.rec-header {{ display:flex; justify-content:space-between; align-items:center; }}
.rec-ticker {{ font-size:16px; font-weight:700; }}
.rec-model {{ font-size:11px; color:var(--dim); }}
.rec-prices {{ display:flex; gap:12px; margin-top:6px; font-size:12px; }}
.rec-pnl {{ font-weight:700; }}
.rec-pnl.positive {{ color:var(--green); }}
.rec-pnl.negative {{ color:var(--red); }}
.rec-meta {{ display:flex; gap:8px; margin-top:4px; font-size:11px; color:var(--dim); flex-wrap:wrap; }}
.rec-meta span {{ background:rgba(255,255,255,0.05); padding:1px 6px; border-radius:4px; }}
.stat-row {{ display:flex; justify-content:space-between; padding:4px 0; font-size:13px; border-bottom:1px solid rgba(255,255,255,0.05); }}
.stat-row .label {{ color:var(--dim); }}
</style>
</head>
<body>

<div class="header">
    <h1>🛡️ Hermes量化系统</h1>
    <div class="sub">蓝盾V6 + 绿箭V11 | 更新: {now}</div>
</div>

<!-- VIX Banner -->
<div class="vix-banner">
    <div class="emoji">{vix_emoji}</div>
    <div class="info">
        <div class="val">{f'{vix_val:.1f}' if vix_val else 'N/A'}</div>
        <div class="label">VIX恐慌指数 — {vix_text}</div>
    </div>
    <div style="text-align:right;font-size:12px;color:var(--dim)">
        {'全仓位' if vix_val and vix_val < 20 else '注意' if vix_val and vix_val < 25 else '减仓50%' if vix_val and vix_val < 35 else '清仓' if vix_val else '—'}
    </div>
</div>

<!-- Tab Navigation -->
<div class="tab-bar">
    <div class="tab active" onclick="showTab('overview')">📊 总览</div>
    <div class="tab" onclick="showTab('recommend')">🎯 推荐</div>
    <div class="tab" onclick="showTab('features')">🔬 特征</div>
    <div class="tab" onclick="showTab('signals')">🚦 信号</div>
    <div class="tab" onclick="showTab('evolution')">📈 演进</div>
    <div class="tab" onclick="showTab('lessons')">💡 教训</div>
</div>

<!-- Tab: Overview -->
<div id="tab-overview" class="section-content active">
    <div class="compare-grid">
        <!-- 蓝盾V6 -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">🛡️ 蓝盾V6</div>
                <span class="badge badge-live">✅ 生产中</span>
            </div>
            <div style="font-size:12px;color:var(--dim);margin-bottom:8px">
                XGBoost排名 | 44维特征 | 20天Top-15 | 全市场&gt;$10
            </div>
            <div class="metric-grid">
                <div class="metric"><div class="val green">1.44</div><div class="lbl">夏普比率</div></div>
                <div class="metric"><div class="val green">+30.1%</div><div class="lbl">年化收益</div></div>
                <div class="metric"><div class="val green">60%</div><div class="lbl">胜率</div></div>
                <div class="metric"><div class="val red">-11.1%</div><div class="lbl">最大回撤</div></div>
                <div class="metric"><div class="val">2.83</div><div class="lbl">Sortino</div></div>
                <div class="metric"><div class="val">2359</div><div class="lbl">股票池</div></div>
            </div>
        </div>
        <!-- 绿箭V11 -->
        <div class="card">
            <div class="card-header">
                <div class="card-title">🎯 绿箭V11</div>
                <span class="badge badge-live">✅ 生产中</span>
            </div>
            <div style="font-size:12px;color:var(--dim);margin-bottom:8px">
                XGBoost排名 | 42维特征 | 5天Top-5 | $1-$10彩票池
            </div>
            <div class="metric-grid">
                <div class="metric"><div class="val green">2.18</div><div class="lbl">夏普比率</div></div>
                <div class="metric"><div class="val green">+5.56%</div><div class="lbl">净收益/5天</div></div>
                <div class="metric"><div class="val">50%</div><div class="lbl">胜率</div></div>
                <div class="metric"><div class="val red">-12.1%</div><div class="lbl">最大回撤</div></div>
                <div class="metric"><div class="val">4%</div><div class="lbl">涨50%+率</div></div>
                <div class="metric"><div class="val">1442</div><div class="lbl">股票池</div></div>
            </div>
        </div>
    </div>

    <!-- VIX Stop-Loss Rules -->
    <div class="card" style="margin-top:10px">
        <div class="card-title" style="margin-bottom:8px">⚡ VIX止损规则</div>
        <table class="signal-table">
            <tr><th>VIX</th><th>状态</th><th>蓝盾操作</th><th>绿箭操作</th></tr>
            <tr><td>&lt;20</td><td>🟢 正常</td><td>全仓位</td><td>正常操作</td></tr>
            <tr><td>20-25</td><td>🟡 注意</td><td>收紧止损</td><td>止损收紧-8%</td></tr>
            <tr><td>25-35</td><td>🟠 警戒</td><td>减仓50%</td><td>减仓</td></tr>
            <tr><td>&gt;35</td><td>🔴 恐慌</td><td>清仓不买</td><td>暂停买入</td></tr>
        </table>
    </div>
</div>
'''

    # ============ Tab: Recommendations ============
    active_recs = [r for r in recs_data.get('recommendations', []) if r.get('status') == 'active']
    closed_recs = sorted([r for r in recs_data.get('recommendations', []) if r.get('status') != 'active'], 
                        key=lambda x: x.get('exit_date', ''), reverse=True)[:20]
    stats = recs_data.get('stats', {})

    from datetime import datetime as dt
    today = dt.now()

    html += '''<!-- Tab: Recommendations -->
<div id="tab-recommend" class="section-content">
    <!-- 信号说明 -->
    <div class="card" style="border-left:3px solid var(--accent)">
        <div class="card-title" style="margin-bottom:6px">🚦 信号含义</div>
        <div style="font-size:12px;line-height:1.6">
            <div><span style="color:var(--green)">🟢🟢 ≥0.90</span> 精品买入 — 马上下单</div>
            <div><span style="color:var(--green)">🟢 ≥0.80</span> 强信号 — 主力买入</div>
            <div><span style="color:var(--yellow)">🟡 ≥0.70</span> 观察 — 放watchlist，不买</div>
            <div><span style="color:var(--dim)">🔴 <0.70</span> 不推荐 — 模型不建议</div>
        </div>
    </div>

    <!-- 持仓纪律 -->
    <div class="card" style="border-left:3px solid var(--orange);margin-top:8px">
        <div class="card-title" style="margin-bottom:6px">⚡ 持仓纪律</div>
        <div style="font-size:12px;line-height:1.6">
            <div>🛡️ <strong>蓝盾</strong>: 持有<strong>20天</strong>，止损-15%</div>
            <div>🎯 <strong>绿箭</strong>: 持有<strong>5天</strong>，止损-10%</div>
            <div style="margin-top:4px;color:var(--orange)">⚠️ 未到期不卖（除非触发止损）</div>
            <div style="color:var(--dim)">跌了肉疼？看进度条：还剩X天到期</div>
        </div>
    </div>

    <div class="section-title" style="margin-top:16px">🎯 活跃推荐</div>
'''

    if active_recs:
        html += '    <div style="margin-bottom:8px;font-size:12px;color:var(--dim)">活跃: {}条 | 更新: {}</div>\n'.format(
            len(active_recs), recs_data.get('updated', '—'))
        
        for rec in active_recs:
            pnl = rec.get('pnl_pct', 0)
            pnl_class = 'positive' if pnl >= 0 else 'negative'
            pnl_sign = '+' if pnl >= 0 else ''
            
            # 计算持仓天数进度
            entry_str = rec.get('entry_date', '')
            expiry_str = rec.get('expiry_date', '')
            hold_days = rec.get('hold_days', 20)
            try:
                entry_dt = dt.strptime(entry_str, '%Y-%m-%d')
                expiry_dt = dt.strptime(expiry_str, '%Y-%m-%d')
                days_held = (today - entry_dt).days
                days_left = max(0, (expiry_dt - today).days)
                progress = min(100, days_held / hold_days * 100) if hold_days > 0 else 0
            except:
                days_held = 0
                days_left = hold_days
                progress = 0
            
            # 进度条颜色
            if days_left <= 0:
                bar_color = 'var(--green)'  # 到期
            elif progress < 50:
                bar_color = 'var(--yellow)'  # 早期
            else:
                bar_color = 'var(--accent)'  # 中期
            
            # 是否接近止损
            entry_price = rec.get('entry_price', 0)
            stop_loss = rec.get('stop_loss_price', 0)
            current = rec.get('current_price', 0)
            near_stop = current <= stop_loss * 1.05 if entry_price > 0 else False
            
            signal = rec.get('signal', '⚪')
            ticker = rec.get('ticker', '')
            model_name = rec.get('model_name', '')
            
            html += f'''    <div class="rec-card active" style="margin-bottom:10px;padding:12px">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <div>
                <span style="font-size:16px;font-weight:700">{ticker}</span>
                <span style="font-size:11px;color:var(--dim);margin-left:6px">{model_name}</span>
            </div>
            <div style="text-align:right">
                <span style="font-size:11px">{signal}</span>
                <span style="font-size:11px;color:var(--dim);margin-left:4px">排名{rec.get("score",0):.3f}</span>
            </div>
        </div>
        <div style="display:flex;gap:16px;margin-top:8px;font-size:13px">
            <div>入场 <strong>${entry_price:.2f}</strong></div>
            <div>当前 <strong>${current:.2f}</strong></div>
            <div class="rec-pnl {pnl_class}">{pnl_sign}{pnl:.1f}%</div>
        </div>
        <div style="display:flex;gap:16px;margin-top:4px;font-size:12px;color:var(--dim)">
            <div>止损 <span style="color:var(--red)">${stop_loss:.2f}</span></div>
            <div>到期 {expiry_str}</div>
        </div>
        <div style="margin-top:8px">
            <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px">
                <span>持仓进度 {days_held}/{hold_days}天</span>
                <span style="color:{'var(--green)' if days_left <= 0 else 'var(--text)'}">{'已到期' if days_left <= 0 else f'还剩{days_left}天'}</span>
            </div>
            <div style="height:6px;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden">
                <div style="height:100%;width:{progress:.0f}%;background:{bar_color};border-radius:3px"></div>
            </div>
        </div>
        {'<div style="margin-top:6px;font-size:11px;color:var(--orange)">⚠️ 接近止损线！考虑止损</div>' if near_stop else ''}
    </div>
'''
        html += '    </div>\n'
    else:
        html += '    <div class="card" style="text-align:center;color:var(--dim);padding:20px">暂无≥0.70的推荐<br><span style="font-size:12px">今日评分全部低于0.70（不推荐买入）</span></div>\n'

    # Recent closed
    html += '    <div class="section-title" style="margin-top:16px">📋 历史记录</div>\n'
    if closed_recs:
        html += '    <div class="card">\n'
        html += '        <table class="signal-table">\n'
        html += '            <tr><th>状态</th><th>代码</th><th>入场</th><th>出场</th><th>盈亏</th><th>持有</th></tr>\n'
        for rec in closed_recs:
            status_emoji = '🟢' if rec.get('pnl_pct', 0) > 0 else '🔴'
            status_text = {'expired':'到期','stopped':'止损'}.get(rec.get('status',''), rec.get('status',''))
            pnl = rec.get('pnl_pct', 0)
            pnl_class = 'positive' if pnl >= 0 else 'negative'
            html += f'            <tr>\n'
            html += f'                <td>{status_emoji} {status_text}</td>\n'
            html += f'                <td style="font-weight:700">{rec.get("ticker","")}</td>\n'
            html += f'                <td>${rec.get("entry_price",0):.2f}</td>\n'
            html += f'                <td>${rec.get("exit_price",0):.2f}</td>\n'
            html += f'                <td class="rec-pnl {pnl_class}">{pnl:+.1f}%</td>\n'
            html += f'                <td style="color:var(--dim)">{rec.get("hold_days",0)}天</td>\n'
            html += f'            </tr>\n'
        html += '        </table>\n'
        html += '    </div>\n'
    else:
        html += '    <div class="card" style="text-align:center;color:var(--dim);padding:16px">暂无历史记录</div>\n'

    # Stats
    if stats:
        html += '    <div class="section-title" style="margin-top:16px">📊 推荐统计</div>\n'
        html += '    <div class="card">\n'
        html += f'        <div class="stat-row"><span class="label">总推荐</span><span>{stats.get("total",0)}条</span></div>\n'
        html += f'        <div class="stat-row"><span class="label">活跃中</span><span style="color:var(--green)">{stats.get("active",0)}条</span></div>\n'
        html += f'        <div class="stat-row"><span class="label">已到期</span><span>{stats.get("expired",0)}条</span></div>\n'
        html += f'        <div class="stat-row"><span class="label">止损触发</span><span style="color:var(--red)">{stats.get("stopped",0)}条</span></div>\n'
        if stats.get('winners', 0) + stats.get('losers', 0) > 0:
            wr = stats['winners'] / (stats['winners'] + stats['losers']) * 100
            html += f'        <div class="stat-row"><span class="label">胜率</span><span style="color:var(--green)">{wr:.0f}%</span></div>\n'
        for mk in ['blueshield_v6', 'arrow_v11']:
            if mk in stats:
                ms = stats[mk]
                name = '🛡️ 蓝盾' if 'shield' in mk else '🎯 绿箭'
                html += f'        <div class="stat-row"><span class="label">{name} 平均盈亏</span><span>{ms.get("avg_pnl",0):+.1f}%</span></div>\n'
                html += f'        <div class="stat-row"><span class="label">{name} 胜率</span><span>{ms.get("win_rate",0):.0f}%</span></div>\n'
        html += '    </div>\n'

    html += '</div>\n'

    # ============ Tab: Features ============
    html += '''<!-- Tab: Features -->
<div id="tab-features" class="section-content">
    <div class="section-title">🛡️ 蓝盾V6 — 44维特征重要性</div>
    <div style="margin-bottom:6px;font-size:12px">
        <span class="cat-tag cat-tech">技术面 27</span>
        <span class="cat-tag cat-macro">宏观面 13</span>
        <span class="cat-tag cat-fund">基本面 4</span>
    </div>
    <div class="card">
'''
    if shield_sorted:
        max_imp = shield_sorted[0][1]
        for name, imp in shield_sorted[:20]:
            pct = imp / max_imp * 100
            cat = feat_category(name)
            cat_cls = {'技术面':'cat-tech','宏观面':'cat-macro','基本面':'cat-fund','特殊':'cat-extra'}.get(cat,'')
            color = feat_cat_color(name)
            html += f'        <div class="feat-bar"><div class="name">{name} <span class="cat-tag {cat_cls}">{cat}</span></div><div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div><div class="pct">{imp:.1f}%</div></div>\n'

    html += '''    </div>
    <div class="section-title" style="margin-top:16px">🎯 绿箭V11 — 42维特征重要性</div>
    <div style="margin-bottom:6px;font-size:12px">
        <span class="cat-tag cat-tech">技术面 28</span>
        <span class="cat-tag cat-macro">宏观面 13</span>
        <span class="cat-tag cat-extra">特殊 2</span>
    </div>
    <div class="card">
'''
    if arrow_sorted:
        max_imp = arrow_sorted[0][1]
        for name, imp in arrow_sorted[:20]:
            pct = imp / max_imp * 100
            cat = feat_category(name)
            cat_cls = {'技术面':'cat-tech','宏观面':'cat-macro','基本面':'cat-fund','特殊':'cat-extra'}.get(cat,'')
            color = feat_cat_color(name)
            html += f'        <div class="feat-bar"><div class="name">{name} <span class="cat-tag {cat_cls}">{cat}</span></div><div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div><div class="pct">{imp:.1f}%</div></div>\n'

    html += '''    </div>
    <div class="card" style="margin-top:10px">
        <div class="card-title" style="margin-bottom:8px">🔍 特征差异分析</div>
        <div style="font-size:13px;line-height:1.6">
            <div>• <strong style="color:#3b82f6">蓝盾Top3</strong>: vol20(14.1%) + bb_std(11.1%) + spy_ret60(10.9%)</div>
            <div>• <strong style="color:#8b5cf6">绿箭Top3</strong>: price(7.1%) + ret1(3.3%) + vol20(2.7%)</div>
            <div style="margin-top:8px">💡 绿箭更关注<strong>价格本身</strong>和<strong>短期动量</strong>（penny stock涨跌看价格敏感度）</div>
            <div>💡 蓝盾更关注<strong>波动率</strong>和<strong>宏观趋势</strong>（大盘股跟随市场节奏）</div>
        </div>
    </div>
</div>
'''

    # ============ Tab: Signals ============
    html += '''<!-- Tab: Signals -->
<div id="tab-signals" class="section-content">
    <div class="section-title">🛡️ 蓝盾V6 — 信号分级</div>
    <div class="card">
        <table class="signal-table">
            <tr><th>信号</th><th>阈值</th><th>20天收益</th><th>胜率</th><th>样本</th><th>操作</th></tr>
'''
    for emoji, label, thresh, ret, wr, sample, action in shield_signals:
        html += f'            <tr><td>{emoji} {label}</td><td>{thresh}</td><td style="color:var(--green)">{ret}</td><td>{wr}</td><td>{sample}</td><td style="color:var(--dim)">{action}</td></tr>\n'

    html += '''        </table>
    </div>
    <div class="section-title" style="margin-top:16px">🎯 绿箭V11 — 信号分级</div>
    <div class="card">
        <table class="signal-table">
            <tr><th>信号</th><th>阈值</th><th>5天收益</th><th>胜率</th><th>涨50%+</th><th>操作</th></tr>
'''
    for emoji, label, thresh, ret, wr, sample, action in arrow_signals:
        html += f'            <tr><td>{emoji} {label}</td><td>{thresh}</td><td style="color:var(--green)">{ret}</td><td>{wr}</td><td>{sample}</td><td style="color:var(--dim)">{action}</td></tr>\n'

    html += '''        </table>
    </div>
    <div class="card" style="margin-top:10px">
        <div class="card-title" style="margin-bottom:8px">📋 每日操作流程</div>
        <div style="font-size:13px;line-height:1.8">
            <div>1️⃣ <strong>04:30</strong> 评分脚本自动运行 → 生成Top-15(蓝盾) + Top-5(绿箭)</div>
            <div>2️⃣ <strong>05:00</strong> 数据更新 → VIX/SPY/QQQ最新值写入</div>
            <div>3️⃣ <strong>查看评分</strong> → 🟢🟢信号立即下单，🟢信号主力买入</div>
            <div>4️⃣ <strong>VIX检查</strong> → VIX>25减仓，VIX>35清仓</div>
            <div>5️⃣ <strong>绿箭止损</strong> → 单只亏-10%立即止损</div>
        </div>
    </div>
</div>
'''

    # ============ Tab: Evolution ============
    html += '<!-- Tab: Evolution -->\n<div id="tab-evolution" class="section-content">\n'
    for group in evolution:
        html += f'    <div class="section-title">{"🛡️" if group["model"]=="蓝盾" else "🎯"} {group["model"]}模型演进</div>\n'
        html += '    <div class="timeline">\n'
        for v in group['versions']:
            active = 'active' if '生产中' in v['status'] else ''
            html += f'        <div class="timeline-item {active}"><div class="ver">{v["v"]} — {v["desc"]}</div><div class="desc">{v["note"]}</div><div class="stats"><span style="color:{"var(--green)" if "生产" in v["status"] else "var(--dim)"}">夏普 {v["sharpe"]}</span><span style="margin-left:8px">{v["status"]}</span></div></div>\n'
        html += '    </div>\n'
    html += '</div>\n'

    # ============ Tab: Lessons ============
    html += '<!-- Tab: Lessons -->\n<div id="tab-lessons" class="section-content">\n    <div class="section-title">💡 关键教训</div>\n'
    for title, desc in lessons:
        html += f'    <div class="lesson"><strong>{title}</strong>: {desc}</div>\n'
    html += '</div>\n'

    # ============ Footer + JS ============
    html += f'''
<div class="footer">
    Hermes量化系统 v2.0 | 蓝盾V6 + 绿箭V11 | {now}
</div>

<script>
function showTab(name) {{
    document.querySelectorAll('.section-content').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('tab-'+name).classList.add('active');
    event.target.classList.add('active');
}}
</script>
</body>
</html>'''

    return html


if __name__ == '__main__':
    html = generate_html()
    out = os.path.join(ROOT, 'dashboard.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f'✅ Dashboard generated: {out}')
    print(f'   Size: {len(html):,} bytes')
