#!/usr/bin/env python3
"""
🍤 每日收盘对比 — 我的推荐 vs 你的操作 vs 模型评分

每天A股15:30、美股04:00各跑一次。
记录三组对比数据，月底一键出报告。
"""
import sys, json, os, datetime
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_PATH = os.path.join(ROOT, 'data', 'recommendations.json')
COMP_PATH = os.path.join(ROOT, 'data', 'daily_comparison.json')
SNAP_PATH = os.path.join(ROOT, 'data', 'portfolio_snapshots.json')

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return [] if 'snapshot' not in path else {}

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_current_price(code, market):
    """获取当前股价"""
    if market == 'a':
        from data_source import AShareRealtime
        rt = AShareRealtime()
        q = rt.get_quote(code)
        return q['price'] if q else 0
    else:
        import minishare as ms
        api = ms.pro_api("Jarvne6fmgArRa46Xfon0e1kw55E6hes5IB2Fy2X0ndqnvrL48jsVOtTbf014f06")
        df = api.query('rt_us_k', ts_code=code, extFields='date')
        if df is not None and len(df) > 0:
            return float(df.iloc[0]['close'])
    return 0

def run_daily_comparison(market_label='us'):
    """每日对比"""
    recs = load_json(REC_PATH)
    today = datetime.date.today().isoformat()
    now = datetime.datetime.now().strftime('%H:%M')
    
    # 找出所有"开放中"的推荐（status=open）
    open_recs = [r for r in recs if r.get('status') == 'open']
    
    comparisons = []
    for r in open_recs:
        code = r['code']
        entry_price = r['price']
        action = r['action']
        
        current_price = get_current_price(code, r.get('market', 'us'))
        if current_price == 0:
            continue
        
        if action == 'buy':
            pnl_pct = (current_price - entry_price) / entry_price * 100
            pnl_status = '✅ 赚' if pnl_pct > 0 else ('❌ 亏' if pnl_pct < 0 else '➖ 平')
        elif action == 'sell':
            pnl_pct = (entry_price - current_price) / entry_price * 100  # 反向（卖出后跌=赚）
            pnl_status = '✅ 赚' if pnl_pct > 0 else ('❌ 亏' if pnl_pct < 0 else '➖ 平')
        else:  # hold
            pnl_pct = 0
            pnl_status = '➖ 持有中'
        
        comparisons.append({
            'date': today,
            'time': now,
            'code': code,
            'action': action,
            'entry_price': entry_price,
            'current_price': round(current_price, 2),
            'pnl_pct': round(pnl_pct, 2),
            'status': pnl_status,
            'reason': r.get('reason', '')[:50]
        })
    
    # 记录到对比文件
    all_comp = load_json(COMP_PATH)
    if today not in all_comp:
        all_comp[today] = []
    all_comp[today].append({
        'time': now,
        'market': market_label,
        'comparisons': comparisons
    })
    save_json(COMP_PATH, all_comp)
    
    # 输出简报
    # 如果有A股持仓，拉资金流向
    moneyflow_lines = []
    if market_label == 'a':
        try:
            import urllib.request
            mf_token = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
            with open(os.path.join(ROOT, 'data', 'portfolio.json')) as f:
                pf = json.load(f)
            a_positions = [p for p in pf.get('a_stock', []) if p.get('shares', 0) > 0]
            for pos in a_positions:
                code = pos['code']
                ts_code = code + '.SH' if code.startswith('6') else code + '.SZ'
                payload = json.dumps({"api_name":"moneyflow","token":mf_token,"params":{"ts_code":ts_code,"start_date":today[:4]+today[5:7]+today[8:],"end_date":today[:4]+today[5:7]+today[8:]}}).encode()
                req = urllib.request.Request('http://api.tushare.pro', data=payload, headers={'Content-Type':'application/json'})
                resp = urllib.request.urlopen(req, timeout=15)
                mf = json.loads(resp.read())
                items = mf.get('data',{}).get('items',[])
                if items:
                    elg = items[0][15] - items[0][17]  # 超大单净
                    lg = items[0][11] - items[0][13]   # 大单净
                    net = items[0][19] / 10000          # 净总额(万)
                    moneyflow_lines.append(f'{pos["name"]}: 主力净{net:+.0f}万 超大单{elg:+.0f} 大单{lg:+.0f}')
        except:
            pass
    
    print(f'📊 每日对比 · {today} {now} {market_label}')
    print('='*50)
    if moneyflow_lines:
        print('💰 资金流向:')
        for l in moneyflow_lines:
            print(f'  {l}')
        print()
    if not comparisons:
        print('  无活跃推荐')
    for c in comparisons:
        print(f'  {c["status"]} {c["action"].upper()} {c["code"]}: 入场${c["entry_price"]} → 现${c["current_price"]} ({c["pnl_pct"]:+.2f}%)')
    print()
    
    # 同时更新recommendations.json里的表现数据
    for r in recs:
        if r.get('status') == 'open':
            for c in comparisons:
                if r['code'] == c['code'] and r['action'] == c['action']:
                    r['last_price'] = c['current_price']
                    r['last_pnl'] = c['pnl_pct']
    save_json(REC_PATH, recs)
    
    return comparisons

# 也支持合并两条对比（A股+美股）
def full_day_comparison():
    print('🦐 完整每日收盘对比')
    print(f'日期: {datetime.date.today().isoformat()}')
    print()
    
    a_results = run_daily_comparison('a') if is_a_stock_day() else []
    us_results = run_daily_comparison('us') if is_us_stock_day() else []
    
    all_comps = a_results + us_results
    
    # 拼接推送消息
    if all_comps:
        from notify import send
        lines = [f'📊 收盘对比 · {datetime.date.today().isoformat()}']
        for c in all_comps:
            lines.append(f'{c["status"]} {c["action"].upper()} {c["code"]}: ${c["entry_price"]}→${c["current_price"]} ({c["pnl_pct"]:+.2f}%)')
        send('\n'.join(lines))

def is_a_stock_day():
    return datetime.datetime.now().weekday() < 5

def is_us_stock_day():
    return datetime.datetime.now().weekday() < 5

if __name__ == '__main__':
    market = sys.argv[1] if len(sys.argv) > 1 else 'us'
    if market == 'all':
        full_day_comparison()
    else:
        run_daily_comparison(market)


# 运行完成自检
import sys
if __name__ == '__main__':
    market = sys.argv[1] if len(sys.argv) > 1 else 'us'
    comparisons = run_daily_comparison(market)
    if not comparisons:
        # 没有活跃推荐是正常的，不报错
        pass
    print(f'[ok] daily_compare {market} completed')

# 审计记录
try:
    from audit_engine import audit
    audit('daily_compare', 'success', '收盘对比完成')
except:
    pass
