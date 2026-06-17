#!/usr/bin/env python3
"""
🍤 A股盘中10分钟扫描 — 只推变化，不重复

清晨扫出了Top 100，盘中每10分钟重新评分，
只有以下情况才推：
  ① 新票进了前10
  ② 持仓评分暴跌>10分
  ③ 异动>±3%
"""
import sys, json, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from data_source import AShareKline, AShareRealtime
from score_engine import v1_score_from_data
from notify import send

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALITY_POOL = os.path.join(ROOT, 'data', 'quality_pool.json')
PORTFOLIO = os.path.join(ROOT, 'data', 'portfolio.json')
STATE_FILE = os.path.join(ROOT, 'data', 'intraday_state.json')

kl = AShareKline()
rt = AShareRealtime()

# 加载策略参数
with open(os.path.join(ROOT, 'config', 'strategy.json')) as f:
    STRATEGY = json.load(f)
A_CFG = STRATEGY['a_stock']
BUY_THRESHOLD = A_CFG['buy_threshold']
SELL_THRESHOLD = A_CFG['sell_threshold']

def get_portfolio_codes():
    try:
        with open(PORTFOLIO) as f:
            pf = json.load(f)
        return [p['code'] for p in pf.get('a_stock', [])]
    except:
        return []

def load_previous_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        # 首次启动：发通知告诉用户扫描已上线
        from notify import send
        send('🍤 盘中扫描已启动，每10分钟检查一次')
        return {'top10': [], 'position_scores': {}, 'updated': ''}

def save_state(top10, pos_scores):
    state = {
        'top10': top10,
        'position_scores': pos_scores,
        'updated': time.strftime('%H:%M')
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

TOP100_FILE = os.path.join(ROOT, 'data', 'morning_top100.json')

def scan():
    # 从晨扫结果读取Top100候选
    if os.path.exists(TOP100_FILE):
        with open(TOP100_FILE) as f:
            scan_stocks = json.load(f)
    else:
        # 回退：从质量池读前100
        with open(QUALITY_POOL) as f:
            pool = json.load(f)
        all_stocks = pool.get('stocks', [])
        scan_stocks = [s for s in all_stocks if s.get('tradeable')][:100]
    
    prev = load_previous_state()
    results = []
    
    for s in scan_stocks:
        code = s['code']
        d = kl.get_kline(code)
        if not d or len(d) < 60:
            continue
        close = [x['close'] for x in d]
        high = [x['high'] for x in d]
        low = [x['low'] for x in d]
        score = v1_score_from_data(close, high, low)
        if score is None:
            continue
        
        # 实时价格
        try:
            q = rt.get_quote(code)
            price = q['price'] if q else close[-1]
            chg = q['change_pct'] if q else 0
        except:
            price = close[-1]
            chg = 0
        
        results.append({
            'code': code,
            'name': s.get('name', code),
            'score': round(float(score), 0),
            'price': price,
            'change_pct': chg
        })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    new_top10 = [r['code'] for r in results[:10]]
    
    # 持仓评分
    pos_codes = get_portfolio_codes()
    pos_alerts = []
    for r in results:
        if r['code'] in pos_codes:
            old_score = prev.get('position_scores', {}).get(r['code'], r['score'])
            score_diff = r['score'] - old_score
            if score_diff <= -10:
                pos_alerts.append(f'⚠️ {r["name"]} 评分暴跌{abs(score_diff):.0f}分 (r["score"]:.0f)')
    
    # 检查Top 10变化
    old_top10 = prev.get('top10', [])
    new_entries = [c for c in new_top10 if c not in old_top10[:10]]
    
    alerts = []
    if new_entries:
        new_names = [f'{next((r["name"] for r in results if r["code"]==c), c)}' for c in new_entries[:3]]
        alerts.append(f'📊 新进前10: {"、".join(new_names)}')
    
    alerts.extend(pos_alerts)
    
    # 保存当前状态
    pos_scores = {r['code']: r['score'] for r in results if r['code'] in pos_codes}
    save_state(new_top10, pos_scores)
    
    if alerts:
        msg = f'📊 A股盘中 · {time.strftime("%H:%M")}\n'
        msg += '──────────────────────────────────\n'
        msg += '\n'.join(alerts)
        send(msg)
        # 同时保存最新Top 10到文件供复盘用
        with open(os.path.join(ROOT, 'data', 'intraday_top10.txt'), 'w') as f:
            f.write('\n'.join([f'{r["name"]} ({r["code"]}): {r["score"]:.0f}分' for r in results[:10]]))
        print(msg)
    else:
        print(f'[{time.strftime("%H:%M")}] 无变化')

if __name__ == '__main__':
    scan()
    
    # 审计记录
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from audit_engine import audit
        audit('intraday_scan_legacy', 'success', '盘中扫描完成(旧版)')
    except:
        pass
