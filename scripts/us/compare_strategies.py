#!/usr/bin/env python3
"""
🦐 策略对比实测 — V4 vs V3 vs V2.5 vs 美股V2
============================================
逐日仿真，真实交易逻辑。

V4 A股: 全市场无行业, V1评分, 买62/卖50, 8只, 7天调仓, 5天最短持
V3 A股: 行业动量筛选前4→各5只, V1评分, 买62/卖50, 5只, 7天调仓
V2.5 A股: 行业动量筛选前4→5只, V1评分, 买62/卖48, 5只, 7天调仓
美股V2: 48只候选, V2评分(V1.6+MACD门), 买60/卖30, 5只, 20天调仓
"""

import sys, json, time
sys.path.insert(0, '.')
from scripts.score_engine import v1_score_from_data, compute_indicators

import warnings; warnings.filterwarnings('ignore')

# ===== 加载数据 =====
print("📦 加载数据...")
t0 = time.time()

A_STOCKS_FILE = '/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json'
US_STOCKS_FILE = '/home/admin/.openclaw/workspace/data/us_stock_backtest.json'
SECTOR_FILE = '/home/admin/.openclaw/workspace/data/sector_map.json'

with open(A_STOCKS_FILE) as f:
    ah = json.load(f)

# 加载行业分类
with open(SECTOR_FILE) as f:
    smap = json.load(f)

# 排除行业
EXCLUDED = {'地产基建', '农业', '交通物流'}
ETFS = {'515000','512480','512800','512880','512010','515030','510300','511010','518880','159915'}

# 所有A股
acodes = [c for c in ah if c not in ETFS and len(ah[c].get('close', [])) > 500]
adates = sorted(set(d for c in acodes for d in ah[c].get('dates', []) if '2015-01-01' <= d <= '2026-05-14'))
print(f"  A股: {len(acodes)}只, {len(adates)}天, {time.time()-t0:.1f}s")

# 日期索引
acdate_idx = {}
for c in acodes:
    ds = ah[c].get('dates', [])
    acdate_idx[c] = {d: i for i, d in enumerate(ds)} if ds else {}

def gi(code, date):
    """获取股票在某日的数据索引"""
    cm = acdate_idx.get(code, {})
    if date in cm: return cm[date]
    d = ah.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x <= date and x in cm:
                return cm[x]
    return -1

# ===== 评分函数 =====
def score_V1(code, date):
    """V1评分（统一入口）"""
    d = ah.get(code)
    if not d: return 0
    idx = gi(code, date)
    if idx < 60: return 0
    close = d['close'][:idx+1]
    high = d.get('high', close)[:idx+1]
    low = d.get('low', close)[:idx+1]
    sc = v1_score_from_data(close, high, low)
    return int(sc) if sc else 0

# ===== 仿真器 =====
def run_simulation(codes, dates, params):
    """
    通用仿真器。
    params: {buy, sell, max_pos, rebal, min_hold, pre_filter, name}
    pre_filter: None(全部) / 'sector'(行业动量) / 其他
    """
    cash = 1000000
    positions = {}
    trade_log = []
    total_buys = 0
    total_sells = 0
    rebal_counter = 0
    peak = 1000000
    
    for day_idx, date in enumerate(dates):
        rebal_counter += 1
        
        # --- 每天检查卖出 ---
        to_sell = []
        for code in list(positions.keys()):
            pos = positions[code]
            hold = day_idx - pos['buy_day']
            if hold < params['min_hold']:
                continue
            
            score = score_V1(code, date)
            if score == 0:
                # MACD门关 = 评分0分 = 卖
                to_sell.append(code)
                continue
            elif score < params['sell']:
                to_sell.append(code)
            elif (price := _get_price(code, date)) and \
                 (price - pos['buy_price']) / pos['buy_price'] * 100 < -8:
                to_sell.append(code)
        
        for code in to_sell:
            pos = positions[code]
            price = _get_price(code, date) or pos['buy_price']
            profit = (price - pos['buy_price']) / pos['buy_price'] * 100
            rev = price * pos['shares']
            cash += rev
            trade_log.append(f"🔴{date}|{params['name']}|卖出{pos['name']}({code})|{pos['shares']}股@{price:.2f}|{profit:+.1f}%")
            total_sells += 1
            del positions[code]
        
        # --- 调仓日买入 ---
        if rebal_counter >= params['rebal']:
            rebal_counter = 0
            
            # 预选候选池
            candidates = []
            for code in codes:
                if code in positions: continue
                sc = score_V1(code, date)
                if sc >= params['buy']:
                    p = _get_price(code, date)
                    if p:
                        candidates.append((code, sc, p))
            
            candidates.sort(key=lambda x: -x[1])
            
            for code, sc, price in candidates:
                if len(positions) >= params['max_pos'] or cash < 5000:
                    break
                
                per = cash / (params['max_pos'] - len(positions))
                shares = max(100, int(per / price / 100) * 100)
                cost = shares * price
                if cost > cash: shares = int(cash / price / 100) * 100; cost = shares * price
                if shares < 100: continue
                
                cash -= cost
                name = ah[code].get('name', code)
                positions[code] = {'shares': shares, 'buy_price': price, 'buy_day': day_idx, 'name': name}
                trade_log.append(f"🟢{date}|{params['name']}|买入{name}({code})|{shares}股@{price:.2f}|¥{cost:,.0f}|评分{sc}")
                total_buys += 1
        
        # 净值
        pv = 0
        for code, pos in positions.items():
            p = _get_price(code, date) or pos['buy_price']
            pv += p * pos['shares']
        total = cash + pv
        if total > peak: peak = total
    
    # 最终
    fpv = sum(positions[c].get('_price', p['buy_price']) * p['shares'] for c, p in positions.items()) if False else 0
    # 简化：用现有持仓计算
    fpv = 0
    for code, pos in positions.items():
        p = _get_price(code, dates[-1]) or pos['buy_price']
        fpv += p * pos['shares']
    
    final = cash + fpv
    ret = (final / 1000000 - 1) * 100
    days = (__import__('datetime').datetime.strptime(dates[-1], '%Y-%m-%d') - 
            __import__('datetime').datetime.strptime(dates[0], '%Y-%m-%d')).days
    ann = ((final / 1000000) ** (365/days) - 1) * 100 if days > 0 else 0
    dd = (peak - final) / peak * 100
    
    return {
        'name': params['name'],
        'final': round(final, 2),
        'return_pct': round(ret, 2),
        'annualized': round(ann, 2),
        'max_drawdown': round(dd, 2),
        'buys': total_buys,
        'sells': total_sells,
        'peak': round(peak, 2),
        'final_cash': round(cash, 2),
        'final_positions': len(positions),
        'trade_log': trade_log
    }

def _get_price(code, date):
    """获取某日收盘价"""
    idx = gi(code, date)
    if idx < 0: return None
    return ah[code]['close'][idx]

# ===== V4 A股 =====
def run_v4():
    print("\n🚀 V4 A股 直选...")
    t0 = time.time()
    params = {'buy': 62, 'sell': 50, 'max_pos': 8, 'rebal': 7, 'min_hold': 5, 'name': 'V4直选', 'pre_filter': None}
    r = run_simulation(acodes, [d for d in adates if d >= '2016-01-01'], params)
    print(f"  {r['return_pct']:+.2f}% | 年化{r['annualized']:.2f}% | 回撤{r['max_drawdown']:.1f}% | {time.time()-t0:.1f}s")
    return r

# ===== V3 A股（行业动量筛选）=====
def run_v3():
    print("\n🚀 V3 A股 行业筛选...")
    t0 = time.time()
    params = {'buy': 62, 'sell': 50, 'max_pos': 5, 'rebal': 7, 'min_hold': 5, 'name': 'V3行业', 'pre_filter': 'sector'}
    
    dates = [d for d in adates if d >= '2016-01-01']
    cash = 1000000; positions = {}; trade_log = []; total_buys = 0; total_sells = 0
    peak = 1000000; rebal_c = 0
    
    for di, date in enumerate(dates):
        rebal_c += 1
        
        # 每天检查卖出
        to_sell = []
        for code in list(positions.keys()):
            pos = positions[code]
            hold = di - pos['buy_day']
            if hold < 5: continue
            score = score_V1(code, date)
            if score < 50:
                to_sell.append(code)
            elif (price := _get_price(code, date)) and (price - pos['buy_price'])/pos['buy_price']*100 < -8:
                to_sell.append(code)
        
        for code in to_sell:
            pos = positions[code]; price = _get_price(code,date) or pos['buy_price']
            profit = (price-pos['buy_price'])/pos['buy_price']*100
            cash += price*pos['shares']
            trade_log.append(f"🔴{date}|V3|卖出{pos['name']}({code})|{profit:+.1f}%")
            total_sells += 1; del positions[code]
        
        # 调仓日
        if rebal_c >= 7:
            rebal_c = 0
            # 行业动量
            sector_mom = {}
            for sec in set(smap.get(c,'其他') for c in acodes if smap.get(c,'其他') not in EXCLUDED):
                rets = []
                for c in [c for c in acodes if smap.get(c,'其他') == sec][:20]:
                    i_now = gi(c, date); i_20 = gi(c, dates[max(0,di-20)])
                    if i_now >= 0 and i_20 >= 0:
                        p_now = ah[c]['close'][i_now]
                        p_20 = ah[c]['close'][i_20]
                        if p_20 > 0: rets.append((p_now-p_20)/p_20*100)
                if len(rets) >= 2: sector_mom[sec] = sum(rets)/len(rets)
            
            top_sectors = sorted(sector_mom.items(), key=lambda x: -x[1])[:4]
            sec_codes = [c for c in acodes if smap.get(c,'其他') in [s[0] for s in top_sectors] and
                        smap.get(c,'其他') not in EXCLUDED]
            
            candidates = []
            for code in sec_codes:
                if code in positions: continue
                sc = score_V1(code, date)
                if sc >= 62:
                    p = _get_price(code, date)
                    if p: candidates.append((code, sc, p))
            
            candidates.sort(key=lambda x: -x[1])
            for code, sc, price in candidates:
                if len(positions) >= 5 or cash < 5000: break
                per = cash/(5-len(positions))
                shares = max(100, int(per/price/100)*100)
                cost = shares*price
                if cost > cash: shares = int(cash/price/100)*100; cost = shares*price
                if shares < 100: continue
                cash -= cost
                positions[code] = {'shares': shares, 'buy_price': price, 'buy_day': di, 'name': ah[code].get('name',code)}
                trade_log.append(f"🟢{date}|V3|买入{ah[code].get('name',code)}({code})|{shares}股@{price:.2f}|评分{sc}")
                total_buys += 1
            
            pv = sum((_get_price(c,date) or p['buy_price']) * p['shares'] for c,p in positions.items())
            total = cash + pv
            if total > peak: peak = total
    
    fpv = sum((_get_price(c,dates[-1]) or p['buy_price']) * p['shares'] for c,p in positions.items())
    final = cash + fpv
    ret = (final/1000000-1)*100
    days = (__import__('datetime').datetime.strptime(dates[-1],'%Y-%m-%d') - __import__('datetime').datetime.strptime(dates[0],'%Y-%m-%d')).days
    ann = ((final/1000000)**(365/days)-1)*100 if days>0 else 0
    dd = (peak-final)/peak*100
    
    r = {'name':'V3行业','final':final,'return_pct':round(ret,2),'annualized':round(ann,2),
         'max_drawdown':round(dd,2), 'buys':total_buys,'sells':total_sells,
         'peak':peak, 'final_cash':cash, 'final_positions':len(positions)}
    print(f"  {r['return_pct']:+.2f}% | 年化{r['annualized']:.2f}% | 回撤{r['max_drawdown']:.1f}% | {time.time()-t0:.1f}s")
    return r

# ===== V2.5 A股 =====
def run_v25():
    print("\n🚀 V2.5 A股 精中选精...")
    t0 = time.time()
    
    dates = [d for d in adates if d >= '2016-01-01']
    cash = 1000000; positions = {}; trade_log = []; total_buys = 0; total_sells = 0
    peak = 1000000; rebal_c = 0
    
    for di, date in enumerate(dates):
        rebal_c += 1
        to_sell = []
        for code in list(positions.keys()):
            pos = positions[code]
            hold = di - pos['buy_day']
            if hold < 5: continue
            score = score_V1(code, date)
            if score < 48:  # V2.5卖48
                to_sell.append(code)
            elif (price := _get_price(code,date)) and (price-pos['buy_price'])/pos['buy_price']*100 < -8:
                to_sell.append(code)
        
        for code in to_sell:
            pos = positions[code]; price = _get_price(code,date) or pos['buy_price']
            profit = (price-pos['buy_price'])/pos['buy_price']*100
            cash += price*pos['shares']
            trade_log.append(f"🔴{date}|V2.5|卖出{pos['name']}({code})|{profit:+.1f}%")
            total_sells += 1; del positions[code]
        
        if rebal_c >= 7:
            rebal_c = 0
            sector_mom = {}
            for sec in set(smap.get(c,'其他') for c in acodes if smap.get(c,'其他') not in EXCLUDED):
                rets = []
                for c in [c for c in acodes if smap.get(c,'其他')==sec][:20]:
                    i_now = gi(c,date); i_20 = gi(c,dates[max(0,di-20)])
                    if i_now>=0 and i_20>=0:
                        p_now=ah[c]['close'][i_now]; p_20=ah[c]['close'][i_20]
                        if p_20>0: rets.append((p_now-p_20)/p_20*100)
                if len(rets)>=2: sector_mom[sec]=sum(rets)/len(rets)
            
            top_secs = sorted(sector_mom.items(),key=lambda x:-x[1])[:4]
            sec_codes = [c for c in acodes if smap.get(c,'其他') in [s[0] for s in top_secs] and
                        smap.get(c,'其他') not in EXCLUDED]
            
            candidates = []
            for code in sec_codes:
                if code in positions: continue
                sc=score_V1(code,date)
                if sc>=62:
                    p=_get_price(code,date)
                    if p: candidates.append((code,sc,p))
            
            candidates.sort(key=lambda x:-x[1])
            for code,sc,price in candidates:
                if len(positions)>=5 or cash<5000: break
                per=cash/(5-len(positions))
                shares=max(100,int(per/price/100)*100)
                cost=shares*price
                if cost>cash: shares=int(cash/price/100)*100; cost=shares*price
                if shares<100: continue
                cash-=cost
                positions[code]={'shares':shares,'buy_price':price,'buy_day':di,'name':ah[code].get('name',code)}
                trade_log.append(f"🟢{date}|V2.5|买入{ah[code].get('name',code)}({code})|{shares}股@{price:.2f}|评分{sc}")
                total_buys+=1
            
            pv=sum((_get_price(c,date)or p['buy_price'])*p['shares'] for c,p in positions.items())
            total=cash+pv
            if total>peak: peak=total
    
    fpv=sum((_get_price(c,dates[-1])or p['buy_price'])*p['shares'] for c,p in positions.items())
    final=cash+fpv
    ret=(final/1000000-1)*100
    days_span=(__import__('datetime').datetime.strptime(dates[-1],'%Y-%m-%d')-__import__('datetime').datetime.strptime(dates[0],'%Y-%m-%d')).days
    ann=((final/1000000)**(365/days_span)-1)*100 if days_span>0 else 0
    dd=(peak-final)/peak*100
    r={'name':'V2.5精中选精','final':final,'return_pct':round(ret,2),'annualized':round(ann,2),
       'max_drawdown':round(dd,2),'buys':total_buys,'sells':total_sells,'peak':peak,
       'final_cash':cash,'final_positions':len(positions)}
    print(f"  {r['return_pct']:+.2f}% | 年化{r['annualized']:.2f}% | 回撤{r['max_drawdown']:.1f}% | {time.time()-t0:.1f}s")
    return r

if __name__ == '__main__':
    results = []
    results.append(run_v4())
    results.append(run_v3())
    results.append(run_v25())
    
    print(f"\n{'='*60}")
    print(f"📊 策略对比 总结")
    print(f"{'='*60}")
    print(f"{'策略':<16} {'收益':>8} {'年化':>8} {'回撤':>8} {'买/卖':>10}")
    print(f"{'-'*50}")
    for r in results:
        bs = f"{r['buys']}/{r['sells']}"
        print(f"{r['name']:<16} {r['return_pct']:>+7.2f}% {r['annualized']:>7.2f}% {r['max_drawdown']:>7.2f}% {bs:>10}")
    
    # 保存
    with open('data/strategy_comparison.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存: data/strategy_comparison.json")
