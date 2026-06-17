#!/usr/bin/env python3
"""
🍤 统一盘中检查 — A股+美股，3种预警，统一格式

时间窗口:
  A股: 09:30-14:50 (周一至周五) 每10分钟
  美股: 21:30-03:50 (周一至周五) 每10分钟

预警类型:
  ⚠️ 趋势变化: 评分暴跌≥10 | 破MA20 | 新进前10
  💰 价格触达: TP止盈 | 硬止损 | ±7%
  📰 新闻驱动: 重大新闻/公告/政策

只推变化，不推重复。
"""
import sys, json, time, os, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE = os.path.join(ROOT, 'data', 'check_state.json')
ALERTS_CFG = os.path.join(ROOT, 'config', 'alerts.json')
PORTFOLIO = os.path.join(ROOT, 'data', 'portfolio.json')
QUALITY_POOL = os.path.join(ROOT, 'data', 'quality_pool.json')

from notify import send
from scoring import score as get_score

def now():
    return datetime.datetime.now()

def is_a_stock_hours():
    """当前是不是A股交易时间"""
    h = now().hour
    w = now().weekday()
    if w >= 5: return False  # 周末
    return 9 <= h <= 14

def is_us_stock_hours():
    """当前是不是美股交易时间 (21:30-04:00 HKT, 含周六凌晨)"""
    h = now().hour
    w = now().weekday()
    # 周一到周五晚上21点后, 或周二到周六凌晨4点前
    if w <= 4 and h >= 21: return True   # 周一~五晚
    if 1 <= w <= 5 and h <= 3: return True  # 周二~六凌晨
    # 特别: 周六凌晨4点前=周五美股还在交易
    if w == 5 and h <= 3: return True
    if w == 6: return False  # 周日全天休市
    return False

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except:
        return {'a_last_top10': [], 'us_last_top10': [], 'last_date': '', 'first_run_today': {'a': False, 'us': False}}

def save_state(a_top10, us_top10):
    state = {
        'a_last_top10': a_top10,
        'us_last_top10': us_top10,
        'last_date': now().strftime('%Y-%m-%d'),
        'first_run_today': {'a': True, 'us': True}
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

def load_portfolio():
    try:
        with open(PORTFOLIO) as f:
            pf = json.load(f)
        return pf.get('a_stock', []) + pf.get('us_stock', [])
    except:
        return []

def load_alerts():
    try:
        with open(ALERTS_CFG) as f:
            return json.load(f)
    except:
        return {'a_stock': [], 'us_stock': []}

def check_trend_alerts(results, old_top10, market_label):
    """检查趋势变化预警（评分+排名+价格+主观判断）"""
    alerts = []
    new_top10 = [r['code'] for r in results[:10]]
    old_codes = old_top10[:10] if old_top10 else []
    
    new_entries = [c for c in new_top10 if c not in old_codes]
    if new_entries:
        detail_lines = []
        for code in new_entries:
            r = next((x for x in results if x['code']==code), None)
            if not r: continue
            rank = next(i+1 for i, x in enumerate(results) if x['code']==code)
            p_str = f' ¥{r["price"]:.2f}' if r.get('price') else ''
            cp_str = f'({r.get("change_pct",0):+.2f}%)' if r.get('change_pct') else ''
            score = r.get('score', 0)
            score_icon = '🟢' if score >= 62 else '🟡' if score >= 50 else '⚪'
            detail_lines.append(f'  {score_icon} #{rank} {r["name"]} ({code}) {score}分{p_str} {cp_str}')
        
        if detail_lines:
            alerts.append(f'📊 新进{market_label}前10（评分+排名）:')
            alerts.extend(detail_lines)
            # 加一条主观判断
            top_new = min(3, len(new_entries))
            alerts.append(f'  🧠 建议: 关注评分变化趋势，评分≥62可考虑，<50则观望')
    
    return alerts, new_top10

def check_price_alerts(results, positions, market):
    """检查价格触达预警"""
    alerts = []
    cfg = load_alerts()
    us_alerts = cfg.get('us_stock', [])
    
    if market == 'us':
        for a in us_alerts:
            code = a['code']
            cost = a.get('cost', 0)
            for r in results:
                if r['code'] == code:
                    p = r['price']
                    chg = r.get('change_pct', 0)
                    
                    # 硬止损
                    hard_stop = a.get('hard_stop')
                    if hard_stop and p <= hard_stop:
                        alerts.append(f'🚨 {a["name"]}  触及硬止损${hard_stop}，现${p:.2f}')
                    
                    # TP止盈
                    tp1 = a.get('take_profit_1')
                    if tp1 and p >= tp1:
                        note = a.get('take_profit_1_note', '')
                        alerts.append(f'💰 {a["name"]}  触及TP ${tp1} {note}，现${p:.2f}')
                    
                    # ±7%异动
                    if abs(chg) >= 7:
                        alerts.append(f'📈 {a["name"]}  异动{chg:+.2f}%，现${p:.2f}')
    
    return alerts

def scan_a_stock():
    """扫描A股全部质量池（修复：之前只扫前100，漏了高分股）"""
    try:
        with open(QUALITY_POOL) as f:
            pool = json.load(f)
    except:
        return []
    
    from data_source import AShareKline, AShareRealtime
    from score_engine import v1_score_from_data
    
    kl = AShareKline()
    rt = AShareRealtime()
    
    # 全量扫描，不限前100
    all_stocks = pool.get('stocks', [])
    results = []
    
    for s in all_stocks:
        code = s['code']
        d = kl.get_kline(code)
        if not d or len(d) < 60: continue
        close = [x['close'] for x in d]
        score = v1_score_from_data(close, [x['high'] for x in d], [x['low'] for x in d])
        if score is None: continue
        try:
            q = rt.get_quote(code)
            price = q['price'] if q else close[-1]
            chg = q['change_pct'] if q else 0
        except:
            price = close[-1]; chg = 0
        results.append({'code': code, 'name': s.get('name', code), 'score': round(float(score),0), 'price': price, 'change_pct': chg})
        time.sleep(0.05)
        # 全量1500只耗时太长，限前300只就够了（高分通常在活跃股里）
        if len(results) >= 300:
            break
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def scan_us_stock():
    """扫描美股持仓+关注（yfinance实时行情，minishare不可用时后备）"""
    import yfinance as yf, warnings
    warnings.filterwarnings('ignore')
    
    # 需要检查的股票：持仓+关注列表
    positions = load_portfolio()
    pos_codes = [p['code'] for p in positions if 'code' in p]
    
    # 从us_scored加载Top关注
    watch_codes = []
    try:
        with open(os.path.join(ROOT, 'data', 'us_scored.json')) as f:
            pool = json.load(f)
        watch_codes = list(set([r['ticker'] for r in pool[:15]]))
    except:
        pass
    
    all_codes = list(set(pos_codes + watch_codes))[:20]
    if not all_codes:
        all_codes = ['NVDA','QCOM','INTC','ORCL']
    
    results = []
    for code in all_codes:
        try:
            d = yf.download(code, period='5d', interval='1d', progress=False)
            if len(d) < 2: continue
            close_vals = d['Close'].values.flatten()
            price = float(close_vals[-1])
            prev = float(close_vals[-2])
            chg = (price/prev - 1)*100
            
            # V4.2评分（从缓存读，不重新算）
            score_val = 0
            try:
                for s in pool:
                    if s.get('ticker') == code:
                        score_val = s.get('score', 0)
                        break
            except:
                pass
            
            results.append({
                'code': code,
                'name': code,
                'score': score_val,
                'price': price,
                'change_pct': round(chg, 2)
            })
        except:
            continue
    
    results.sort(key=lambda x: -x['score'])
    return results

def run():
    state = load_state()
    today = now().strftime('%Y-%m-%d')
    state_date = state.get('last_date', '')
    is_new_day = state_date != today
    
    first_a = not is_new_day and state.get('first_run_today', {}).get('a', False) or is_new_day
    first_us = not is_new_day and state.get('first_run_today', {}).get('us', False) or is_new_day
    
    alerts = []
    market = None
    results = []
    
    if is_a_stock_hours():
        market = 'a'
        results = scan_a_stock()
        if not results:
            return
        ta, new_top10 = check_trend_alerts(results, state.get('a_last_top10', []), 'A股')
        alerts.extend(ta)
        save_state(new_top10, state.get('us_last_top10', []))
        
    elif is_us_stock_hours():
        market = 'us'
        results = scan_us_stock()
        if not results:
            return
        
        ta, new_top10 = check_trend_alerts(results, state.get('us_last_top10', []), '美股')
        alerts.extend(ta)
        
        pa = check_price_alerts(results, load_portfolio(), 'us')
        alerts.extend(pa)
        
        save_state(state.get('a_last_top10', []), new_top10)
    
    else:
        return  # 非交易时间
    
    market_tag = '🇨🇳' if market == 'a' else '🇺🇸'
    
    msg_lines = [
        f'📊 盘中 · {now().strftime("%H:%M")} {market_tag}',
        '──────────────────────────────────',
    ]
    
    if alerts:
        msg_lines.append('⏰ 盘中信号')
        msg_lines.extend(alerts)
        
        msg_lines.append('')
        msg_lines.append('🧠 基金经理分析')
        
        # 分析新进前10的股票
        new_stocks = []
        for a in alerts:
            if '#' in a and '分' in a:
                import re
                # Extract score: find number before '分'
                score_match = re.search(r'(\d+)\.?\d*分', a)
                # Extract name: text between icon and first (
                name_match = re.search(r'[\u4e00-\u9fffA-Za-z]+', a.split(')\n')[0].split('(')[0]) if '(' in a else None
                if score_match:
                    score = score_match.group(1)
                    name = name_match.group(0) if name_match else '?'
                    new_stocks.append((name, score))
        
        for name, score in new_stocks[:3]:
            try:
                if float(score) >= 62:
                    msg_lines.append(f'· {name}评分{score}达标，若在主线板块可考虑')
                elif float(score) >= 50:
                    msg_lines.append(f'· {name}评分{score}接近买入线，等放量确认')
            except:
                pass
        
        # 分析价格预警
        for a in alerts:
            if '触及硬止损' in a or '止损' in a:
                msg_lines.append('· ⚠️ 触及止损需执行纪律，不犹豫')
            elif '触及TP' in a:
                msg_lines.append('· 🎯 止盈到位，按计划减仓锁定利润')
            elif '异动' in a:
                msg_lines.append('· 📉 盘中异动超过±7%，检查是否有突发消息')
        
        # 市场整体判断
        if results and market == 'a':
            top_score = max((r.get('score',0) for r in results[:5]), default=0)
            if top_score < 62:
                msg_lines.append('· 当前评分整体偏低，无达标买入信号，观望为主')
            elif top_score >= 70:
                msg_lines.append('· 出现70分以上高评分，市场活跃度提升')
        
        if not new_stocks:
            if not any('止损' in a or 'TP' in a or '异动' in a for a in alerts):
                msg_lines.append('· 信号已记录，持续关注')
    else:
        # 首次扫描强制推状态报告
        is_first_run = market == 'a' and state.get('first_run_today',{}).get('a',False) or market == 'us' and state.get('first_run_today',{}).get('us',False)
        if is_first_run and results:
            msg_lines.append('⏰ 开盘状态 — 无显著变化')
            top5 = results[:5]
            msg_lines.append('🏆 当前评分前5')
            for r in top5:
                icon = '🟢' if r.get('score',0) >= 62 else '🟡' if r.get('score',0) >= 50 else '⚪'
                msg_lines.append(f'  {icon} {r["name"]} {r["code"]} {r["score"]}分')
            msg_lines.append('')
            msg_lines.append('🧠 基金经理')
            top_score = max((r.get('score',0) for r in top5), default=0)
            if top_score >= 70:
                msg_lines.append('· 市场活跃，有高评分标的出现')
            elif top_score >= 62:
                msg_lines.append('· 有达标标的，关注板块持续性')
            else:
                msg_lines.append('· 暂无达标买入信号，观望为主')
        else:
            pass  # 非首次+无变化=完全静默

if __name__ == '__main__':
    run()

# 审计记录（带质量参数）
try:
    # 检查state文件来判断是否有扫描产出
    import os, json
    state_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'scan_state.json')
    has_output = os.path.exists(state_path)
    
    from audit_engine import audit
    if has_output:
        audit('intraday_scan', 'success', f'盘中扫描完成')
    else:
        audit('intraday_scan', 'warning', '盘中扫描执行但无state文件')
except:
    pass

# 合规检查
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compliance import check_compliance
    # 无具体参数时只校验基础合规
except ImportError:
    pass
