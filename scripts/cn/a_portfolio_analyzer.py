"""
Portfolio Analyzer — 持仓分析工具
功能：
1. 读取持仓 + 技术面分析（MA/MACD/RSI/止损位）
2. 资金流检查（机构大单还在不在买）
3. 持仓健康度评分 + 持有/减仓/卖出建议
4. 资金流水跟踪（P&L记录）
5. 单只股票深度分析（单独问的时候用）

用法：
  python scripts/portfolio_analyzer.py            # 分析全部持仓
  python scripts/portfolio_analyzer.py --stock 600032  # 单独分析一只
  python scripts/portfolio_analyzer.py --cash -5000 "手动转出"  # 记录资金变动
"""
import json, os, sys, math, time
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORKSPACE NORTH_MONEY
ANALYSIS_LOG = os.path.join(WORKSPACE, 'data', 'portfolio_log.json')
CASH_LOG = os.path.join(WORKSPACE, 'data', 'cash_flow.json')

def ef(v):
    if v is None: return 0.0
    try: return float(v)
    except: return 0.0

def sma(arr, n):
    if not arr or len(arr) < n: return None
    return sum(arr[-n:]) / n

# 加载数据
def load_data():
    kl = json.load(open(f'{WORKSPACE}/data/a_hist_10y.parquet', 'rb'))
    daily = json.load(open(f'{WORKSPACE}/data/a1_daily.json', 'rb'))
    
    # 北向
    north = json.load(open(NORTH_MONEY, 'rb'))
    recs = north.get('records', north)
    nv = [ef(r.get('north_money', 0)) for r in recs]
    nd = [r['trade_date'] for r in recs]
    n_mom = []
    for i in range(59, len(nv)):
        s20 = sum(nv[i-19:i+1]); s60 = sum(nv[i-59:i+1])
        n_mom.append(s20 / s60 if s60 != 0 else 1.0)
    
    def nb_pct(ds):
        for i, d in enumerate(nd):
            if d == ds: break
        else: return 50
        if i < 60: return 50
        s20 = sum(nv[i-19:i+1]); s60 = sum(nv[i-59:i+1])
        m = s20 / s60 if s60 != 0 else 1.0
        return sum(1 for x in n_mom if x < m) / len(n_mom) * 100
    
    return {'kl': kl, 'daily': daily, 'nb_pct': nb_pct, 'kl_dates': sorted(kl.keys())}

# 技术面分析
def tech_analysis(code, data):
    kd = data['kl'].get(code)
    if not kd or len(kd.get('c', [])) < 60:
        return None
    
    c = kd['c']; h = kd['h']; lo = kd['l']; dates = kd.get('dates', [])
    n = len(c)
    price = c[-1]
    
    # 均线
    ma5 = sma(c, 5); ma10 = sma(c, 10); ma20 = sma(c, 20); ma60 = sma(c, 60)
    ma120 = sma(c, 120) if len(c) >= 120 else None
    
    # MACD
    def _ema(arr, p):
        k = 2 / (p + 1); r = [arr[0]]
        for v in arr[1:]: r.append(v * k + r[-1] * (1 - k))
        return r
    e12 = _ema(c, 12); e26 = _ema(c, 26)
    macd = [e12[i] - e26[i] for i in range(n)]
    signal = _ema(macd, 9)
    hist = [macd[i] - signal[i] for i in range(n)]
    
    # RSI(14)
    gains = [max(c[i]-c[i-1], 0) for i in range(1, n)]
    losses = [max(c[i-1]-c[i], 0) for i in range(1, n)]
    avg_g = sum(gains[-14:]) / 14 if len(gains) >= 14 else 0
    avg_l = sum(losses[-14:]) / 14 if len(losses) >= 14 else 0
    rsi_14 = 100 - 100 / (1 + avg_g / avg_l) if avg_l > 0 else 100
    
    # 资金流
    mf_today = data['daily'].get(dates[-1], {}).get(code, {})
    mf_score = 0
    mf_detail = "暂无数据"
    if mf_today:
        nm = ef(mf_today.get('net_mf', 0))
        be = ef(mf_today.get('buy_elg', 0)); se = ef(mf_today.get('sell_elg', 0))
        bl = ef(mf_today.get('buy_lg', 0)); sl = ef(mf_today.get('sell_lg', 0))
        tt = be + se + bl + sl
        if tt > 0:
            br = (be + bl - se - sl) / tt * 100
            mf_score = nm / 10000 * 0.4 + max(br, 0) * 0.6
            mf_detail = f"净流入{nm/10000:.0f}亿 大单净比{br:+.1f}% 评分{mf_score:.1f}"
    
    # 综合评分
    score = 50
    
    # 均线排列（+15分）
    if ma5 and ma10 and ma20 and ma60:
        if ma5 > ma10 > ma20 > ma60: score += 15
        elif price > ma20: score += 8
        elif price < ma20: score -= 5
    
    # MACD（+10分）
    if hist[-1] > 0 and hist[-1] > hist[-2]: score += 10
    elif hist[-1] < 0 and hist[-1] < hist[-2]: score -= 5
    
    # RSI（+5分）
    if 40 <= rsi_14 <= 70: score += 5
    elif rsi_14 > 80: score -= 10
    elif rsi_14 < 30: score += 3  # 超卖可能是机会
    
    # 资金流（+10分）
    if mf_score > 30: score += 10
    elif mf_score > 0: score += 3
    
    # 偏离MA20（-5分）
    if ma20 and ma20 > 0:
        dev = (price / ma20 - 1) * 100
        if dev > 20: score -= 8  # 偏离太大要回调
        elif dev < -10: score += 5  # 深跌是机会
    
    score = max(0, min(100, score))
    
    # 建议（更细化，考虑资金流）
    money_flow_ok = mf_score > 10
    
    if score >= 65 and money_flow_ok: 
        decision = '持有✅'; reason = f'技术多头+资金流支持(评分{score})'
    elif score >= 65 and not money_flow_ok: 
        decision = '持有⚠️'; reason = f'技术多头但资金流偏弱({mf_score:.1f})，注意回调'
    elif score >= 50 and money_flow_ok:
        decision = '观察👀'; reason = f'中性偏多，等待确认'
    elif score >= 50 and not money_flow_ok:
        decision = '减持❗'; reason = f'趋势不明+资金流出，建议减仓'
    elif score >= 35:
        decision = '减仓🔴'; reason = '技术面走弱，建议减仓'
    else:
        decision = '卖出🔴'; reason = '趋势恶化，建议止损'
    
    # 止损/止盈位
    stop_loss = round(ma20 * 0.92, 2) if ma20 else round(price * 0.92, 2)
    take_profit = round(ma60 * 1.15, 2) if ma60 else round(price * 1.15, 2)
    
    return {
        'code': code, 'price': price,
        'ma5': round(ma5, 2) if ma5 else None,
        'ma10': round(ma10, 2) if ma10 else None,
        'ma20': round(ma20, 2) if ma20 else None,
        'ma60': round(ma60, 2) if ma60 else None,
        'macd_hist': round(hist[-1], 2),
        'rsi': round(rsi_14, 1),
        'mf_score': round(mf_score, 1),
        'mf_detail': mf_detail,
        'm20_dev': round((price / ma20 - 1) * 100, 1) if ma20 and ma20 > 0 else 0,
        'score': score,
        'decision': decision,
        'reason': reason,
        'stop_loss': stop_loss,
        'take_profit': take_profit
    }


# 分析全部持仓
def analyze_portfolio(holdings):
    """holdings: [{code, shares, cost, name?}, ...]"""
    data = load_data()
    
    print(f"{'='*60}")
    print(f"📊 持仓分析 ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*60}")
    
    total_value = 0
    total_cost = 0
    
    for h in holdings:
        code = h['code']
        shares = h.get('shares', 0)
        cost = h.get('cost', 0)
        name = h.get('name', code)
        
        ta = tech_analysis(code, data)
        if not ta:
            print(f"\n{name} ({code}): 数据不足，跳过")
            continue
        
        market_value = ta['price'] * shares
        pl = (ta['price'] / cost - 1) * 100 if cost > 0 else 0
        total_value += market_value
        total_cost += cost * shares
        
        print(f"\n{'─'*60}")
        print(f"{name} ({code})")
        print(f"  持仓: {shares}股 | 均价: {cost:.2f} | 现价: {ta['price']:.2f} | 盈亏: {pl:+.1f}%")
        print(f"  MA5={ta['ma5']} MA10={ta['ma10']} MA20={ta['ma20']} MA60={ta['ma60']}")
        print(f"  MACD柱={ta['macd_hist']:+.2f} | RSI={ta['rsi']:.1f} | 偏MA20={ta['m20_dev']:+.1f}%")
        print(f"  资金流: {ta['mf_detail']}")
        print(f"  技术分: {ta['score']}/100")
        print(f"  建议: {ta['decision']} — {ta['reason']}")
        print(f"  止损: {ta['stop_loss']:.2f} | 止盈: {ta['take_profit']:.2f}")
    
    print(f"\n{'='*60}")
    print(f"组合总值: {total_value:,.0f}")
    print(f"总盈亏: {(total_value/total_cost-1)*100 if total_cost>0 else 0:+.1f}%")
    
    # 北向状态
    nb = data['nb_pct'](datetime.now().strftime('%Y%m%d'))
    print(f"北向动量: {nb:.0f}% ({'可买' if nb>=50 else '暂停买入'})")
    print(f"{'='*60}")


# 单只股票深度分析
def analyze_stock(code):
    data = load_data()
    ta = tech_analysis(code, data)
    if not ta:
        print(f"{code}: 数据不足")
        return
    
    name = code
    print(f"\n{'='*60}")
    print(f"📈 个股分析: {name} ({code})")
    print(f"{'='*60}")
    print(f"  现价: {ta['price']:.2f}")
    print(f"  均线: MA5={ta['ma5']} MA10={ta['ma10']} MA20={ta['ma20']} MA60={ta['ma60']}")
    print(f"  MACD柱: {ta['macd_hist']:+.2f}")
    print(f"  RSI(14): {ta['rsi']:.1f}")
    print(f"  偏离MA20: {ta['m20_dev']:+.1f}%")
    print(f"  资金流: {ta['mf_detail']}")
    print(f"")
    print(f"  技术分: {ta['score']}/100")
    print(f"  建议: {ta['decision']}")
    print(f"  原因: {ta['reason']}")
    print(f"  止损: {ta['stop_loss']:.2f} (-8%)")
    print(f"  止盈: {ta['take_profit']:.2f} (+15%)")
    
    if ta['macd_hist'] > 0:
        print(f"\n  MACD信号: 🟢 多头 (柱线{['收缩','扩张'][ta['macd_hist']>0 and 'macd_hist' in dir()]})")
    else:
        print(f"\n  MACD信号: 🔴 空头")
    
    if ta['rsi'] > 70:
        print(f"  RSI信号: ⚠️ 超买区间")
    elif ta['rsi'] < 30:
        print(f"  RSI信号: 🟢 超卖区间")
    else:
        print(f"  RSI信号: 正常区间")


# 资金流水
def record_cashflow(amount, note=""):
    """记录资金变动。amount正=入金，负=出金"""
    log = []
    if os.path.exists(CASH_LOG):
        with open(CASH_LOG, 'r', encoding='utf-8') as f:
            log = json.load(f)
    
    entry = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'amount': amount,
        'note': note,
        'running_balance': (log[-1]['running_balance'] + amount) if log else amount
    }
    log.append(entry)
    
    with open(CASH_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    
    print(f"已记录: {'入金' if amount>0 else '出金'} {abs(amount):,.0f} ({note})")
    print(f"当前余额: {entry['running_balance']:,.0f}")


if __name__ == "__main__":
    if '--cash' in sys.argv:
        idx = sys.argv.index('--cash')
        amount = float(sys.argv[idx+1])
        note = sys.argv[idx+2] if len(sys.argv) > idx+2 else ""
        record_cashflow(amount, note)
    elif '--stock' in sys.argv:
        idx = sys.argv.index('--stock')
        code = sys.argv[idx+1]
        analyze_stock(code)
    else:
        # 测试：用空持仓跑通
        print("用法:")
        print("  python scripts/portfolio_analyzer.py --stock 600032  # 分析单只")
        print("  python scripts/portfolio_analyzer.py --cash -5000 '出金'  # 记录资金")
        print("  # 持仓分析需要在代码中传入持仓数据")
        print()
        print("示例分析 600032:")
        analyze_stock('600032')
