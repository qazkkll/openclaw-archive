#!/usr/bin/env python3
"""
报告引擎 v3 — 最终固定格式
命名: S1.0(主力) D1.0(防御) L1.0(彩票)
emoji: 🔵高分 🟡中分 🔴低分 ✅持仓 👀观望 ❌卖出 📦🏆
"""
import json, os, sys
from datetime import datetime

SCAN_DIR = r'/home/hermes/.hermes/openclaw-archive/data'
HOLDINGS_PATH = r'/home/hermes/.hermes/openclaw-archive/data\portfolio_root.json'
SEP = '─' * 55

def load_scan():
    candidates = []
    for f in os.listdir(SCAN_DIR):
        path = os.path.join(SCAN_DIR, f)
        mtime = os.path.getmtime(path)
        if f.startswith('daily_score') and f.endswith('.json'):
            candidates.append((mtime, path, 'us'))
        elif f.startswith('a_share') and f.endswith('.json'):
            candidates.append((mtime, path, 'a'))
    if not candidates:
        return None, None, 'unknown'
    candidates.sort(reverse=True)
    _, path, atype = candidates[0]
    with open(path, encoding='utf-8') as f:
        return json.load(f), path, atype

def load_holdings():
    if not os.path.exists(HOLDINGS_PATH):
        return {}
    with open(HOLDINGS_PATH, encoding='utf-8') as f:
        return json.load(f)

def score_badge(scr):
    if scr >= 85: return '🔵'
    if scr >= 75: return '🟡'
    return '🔴'

def holding_badge(pnl):
    if pnl > 20:  return '✅'
    if pnl > 5:   return '✅'
    if pnl > -5:  return '👀'
    return '❌'

def comment_s(sym, scr, rsi, m30, v=20):
    if scr >= 90:
        if rsi < 35: return '价值陷阱，等放量'
        return '价值王者，回调即机会'
    if scr >= 85:
        if m30 > 10:  return f'月涨{m30:.0f}%，谨慎追涨'
        if m30 < -5:  return '短线偏弱，等企稳'
        if rsi > 65:  return f'RSI={rsi}偏高，等回调'
        if rsi < 35:  return '超卖优质，反弹潜力'
        if v >= 38:   return '价值+趋势双保险'
        return '中仓试水，趋势尚可'
    if scr >= 80:
        if rsi > 60:  return 'RSI偏高，等回调入场'
        if m30 < -3:  return '暂不强，轻仓观察'
        return '刚过门槛，轻仓试水'
    return ''

def comment_d(scr, rsi):
    if rsi <= 25: return '深度超卖，快进快出'
    if rsi <= 30: return '严重超卖，等反转信号'
    if rsi <= 35: return '超卖区域，分批建仓'
    return '轻度超卖，观察确认'

def comment_holding(sym, pnl, rsi):
    return {'NVDA': f'趋势走好 +{pnl:.0f}%，止损上移$218',
            'ORCL': f'+{pnl:.0f}%趋势偏弱，分批止盈留1-2股',
            'QCOM': f'+{pnl:.0f}%横盘等催化剂，破$230走',
            'UNH': f'{pnl:+.0f}%不杀跌，等$390反弹',
            'UNP': f'{pnl:+.0f}%没到止损，等板块轮动'}.get(sym, f'{pnl:+.0f}%')

def sector_tag(sym):
    ins = ['ALL','PGR','AFL','CINF','RGA','RJF','AON','BRK-B','MKL','AMP','PRU','RNR','MET','PRU','ACGL','EG']
    eng = ['CNQ','EQT','APA','WES','HES','OXY','DVN','FANG','EOG','MPC','XOM','CVX']
    tech= ['NVDA','ADSK','GOOGL','MSFT','AAPL','CRM','ADBE','ORCL','QCOM','INTC','AMD']
    if sym in ins: return '🏦'
    if sym in eng: return '⚡'
    if sym in tech: return '💻'
    return '🏭'

def sector_name(sym):
    ins = ['ALL','PGR','AFL','CINF','RGA','RJF','AON','BRK-B','MKL','AMP','PRU','RNR','MET','PRU','ACGL','EG']
    eng = ['CNQ','EQT','APA','WES','HES','OXY','DVN','FANG','EOG','MPC','XOM','CVX']
    tech= ['NVDA','ADSK','GOOGL','MSFT','AAPL','CRM','ADBE','ORCL','QCOM','INTC','AMD']
    if sym in ins: return '🔵保险'
    if sym in eng: return '⚡能源'
    if sym in tech: return '💻科技'
    return '🏭其他'

def render(data, path='', atype='us'):
    v5 = data.get('v5', [])
    d1 = data.get('r5d', [])
    l1 = data.get('r5c', [])
    ts = data.get('timestamp', datetime.now().isoformat())
    holdings = load_holdings()
    now_dt = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = []
    lines.append(SEP)
    lines.append(f'📡 {now_dt} | 美股三模型 | S1.0(主力) D1.0(防御) L1.0(彩票)')
    lines.append(SEP)

    # S1.0 主力
    lines.append(f'\nS1.0 主力 ({len(v5)}只>=80)')
    lines.append(SEP)
    for i, r in enumerate(v5[:15]):
        sym=r['t']; scr=r['s']; price=r['p']; rsi=int(r.get('rsi',50))
        m30=r.get('m30',0); v=r.get('v',20)
        badge=score_badge(scr)
        cmt=comment_s(sym,scr,rsi,m30,v)
        holder=''
        if sym in holdings:
            h=holdings[sym]; pnl=(price/h['cost']-1)*100
            holder=f' 📦{pnl:+.0f}%'
        lines.append(f'  #{i+1:2d} {badge} {sym:<6s} {scr:2d}分  ${price:<8.2f}  V={v:2d}  RSI={rsi:2d}  M30={m30:+.1f}%{holder}')
        lines.append(f'       → {cmt}')

    # D1.0 防御
    lines.append(f'\nD1.0 防御 ({len(d1)}只>=50)')
    lines.append(SEP)
    if d1:
        for i, r in enumerate(d1[:10]):
            sym=r['t']; scr=r['s']; price=r['p']; rsi=int(r.get('rsi',50))
            cmt=comment_d(scr,rsi)
            holder=''
            if sym in holdings:
                h=holdings[sym]; pnl=(price/h['cost']-1)*100
                holder=f' 📦{pnl:+.0f}%'
            lines.append(f'  #{i+1:2d} {score_badge(scr)} {sym:<6s} {scr:2d}分  ${price:<8.2f}  RSI={rsi:2d}{holder}')
            lines.append(f'       → {cmt}')
    else:
        lines.append('  (无信号)')

    # L1.0 彩票
    lines.append(f'\nL1.0 彩票 ({len(l1)}只)')
    lines.append(SEP)
    if l1:
        for i, r in enumerate(l1[:10]):
            sym=r['t']; scr=r['s']; price=r['p']; vr=r.get('vr',1)
            lines.append(f'  #{i+1:2d} {score_badge(scr)} {sym:<6s} {scr:2d}分  ${price:<8.2f}  量比={vr:.1f}x')
    else:
        lines.append('  (无信号)')

    # 持仓
    lines.append(f'\n📦 持仓')
    lines.append(SEP)
    for sym, h in holdings.items():
        cost=h['cost']; shares=h['shares']
        found = next((x for x in v5 if x['t']==sym), None) or next((x for x in d1 if x['t']==sym), None)
        if found:
            p=found['p']; rsi=int(found.get('rsi',50)); pp=(p/cost-1)*100; pnl=(p-cost)*shares
            sig='🔵' if pp>0 else '🔴'
            act=holding_badge(pp); cmt=comment_holding(sym,pp,rsi)
            lines.append(f'  {sig} {sym:<6s} {act}  ${p:<8.2f}  {sector_tag(sym)}  {shares}股  成本${cost:<.2f}  {pp:+.1f}%(${pnl:+.0f})')
            lines.append(f'       → {cmt}')
        else:
            lines.append(f'  ⚪ {sym:<6s}  —  无实时数据  {shares}股  成本${cost:<.2f}')

    # 评述
    lines.append(f'\n评述')
    lines.append(SEP)
    if v5:
        t1=v5[0]
        lines.append(f'  🏆 榜首: {t1["t"]} {t1["s"]}分 ${t1["p"]:.2f}  RSI={int(t1.get("rsi",50))}  M30={t1.get("m30",0):+.1f}%')
        for sym, h in holdings.items():
            found = next((x for x in v5 if x['t']==sym), None)
            if found:
                pp=(found['p']/h['cost']-1)*100
                cmt=comment_holding(sym,pp,int(found.get('rsi',50)))
                lines.append(f'  📦 {sym}: S1.0={found["s"]}分  {pp:+.1f}%  → {cmt}')
            else:
                lines.append(f'  📦 {sym}: 不在候选池')
    # 行业分布
    if v5:
        secs={}
        for r in v5:
            tag=sector_name(r['t'])
            secs[tag]=secs.get(tag,0)+1
        parts=[]
        for s, c in sorted(secs.items(), key=lambda x:-x[1]):
            parts.append(f'{s}{c}只')
        lines.append(f'  {" ".join(parts)}')

    lines.append(f'\n📝 daily_score.json | 报告引擎v3')
    return '\n'.join(lines)

if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    data, path, atype = load_scan()
    if data:
        print(render(data, path, atype))
    else:
        print('⚠️ 无扫描数据')
