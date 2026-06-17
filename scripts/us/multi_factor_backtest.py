#!/usr/bin/env python3
"""
A股多因子量化回测引擎 v3
- 8行业×3龙头=24只股票
- 3年日线数据
- 6大因子独立测试+组合
- 对比旧模型 vs 新模型 vs 买入持有
"""

import json, urllib.request, sys, math, time

RELAY = "http://47.107.99.189:8080"

# 行业龙头（主板可交易 60/00开头）
SECTORS = {
    "银行": ["SH.601398","SH.600036","SH.601939"],  # 工商银行/招商银行/建设银行
    "白酒": ["SH.600519","SZ.000858","SZ.000568"],  # 茅台/五粮液/泸州老窖
    "半导体": ["SH.603501","SH.600584","SH.603986"], # 韦尔股份/长电科技/兆易创新
    "新能源车": ["SZ.002594","SH.601633","SH.600104"], # 比亚迪/长城汽车/上汽
    "医药": ["SH.600276","SH.603259","SH.600332"],    # 恒瑞/药明康德/白云山
    "科技": ["SZ.000938","SH.600588","SH.600845"],    # 紫光股份/用友/宝信
    "光伏": ["SH.601012","SH.600438","SH.600089"],    # 隆基/通威/特变电工
    "军工": ["SH.600760","SH.600893","SH.600150"],    # 沈飞/航发动力/船舶
}

STOCKS = []
for sector, codes in SECTORS.items():
    for code in codes:
        STOCKS.append((sector, code))

print(f"📊 多因子回测启动: {len(STOCKS)} 只股票 × 8行业")

# ===== 数据获取 =====
def fetch(code, start, end):
    data = json.dumps({"code": code, "start": start, "end": end, "freq": "D"}).encode()
    req = urllib.request.Request(f"{RELAY}/history", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read()).get("data", [])

# ===== 指标计算 =====
def calc_ema(p, n):
    k = 2/(n+1); e = [p[0]]
    for v in p[1:]: e.append(v*k + e[-1]*(1-k))
    return e

def calc_sma(p, n):
    return [None]*(n-1) + [sum(p[i-n+1:i+1])/n for i in range(n-1,len(p))]

def calc_rsi_full(p):
    if len(p) < 16: return [None]*len(p)
    g, l = [], []
    for i in range(1, len(p)):
        d = p[i]-p[i-1]; g.append(max(d,0)); l.append(max(-d,0))
    r, ag, al = [None]*14, sum(g[:14])/14, sum(l[:14])/14
    for i in range(14, len(p)):
        r.append(100-100/(1+ag/al) if al else 100)
        if i < len(g): ag = (ag*13+g[i])/14; al = (al*13+l[i])/14
    return r

def full_indicators(prices):
    """返回所有指标的完整序列"""
    n = len(prices)
    rsi = calc_rsi_full(prices)
    ma5 = calc_sma(prices, 5)
    ma20 = calc_sma(prices, 20)
    ma60 = calc_sma(prices, 60)
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd = [ema12[i]-ema26[i] for i in range(n)]
    signal = calc_ema(macd, 9)
    hist = [macd[i]-signal[i] for i in range(n)]
    # 52周位置
    pos52 = [None]*251
    for i in range(251, n):
        lo, hi = min(prices[i-251:i+1]), max(prices[i-251:i+1])
        pos52.append((prices[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {
        'rsi': rsi, 'ma5': ma5, 'ma20': ma20, 'ma60': ma60,
        'macd': macd, 'signal': signal, 'hist': hist, 'pos52': pos52
    }

# ===== 因子测试 =====

def sf(arr, i):
    if 0 <= i < len(arr) and arr[i] is not None: return arr[i]
    return None

def test_factor(name, prices, ind, buy_fn, sell_fn, dates):
    """测试单个因子"""
    trades = []; in_pos = False; ep = 0
    for i in range(len(prices)):
        b = buy_fn(i, ind, prices) if not in_pos else False
        s = sell_fn(i, ind, prices) if in_pos else False
        if not in_pos and b:
            in_pos = True; ep = prices[i]; entry_date = dates[i]
        elif in_pos and s:
            in_pos = False
            pnl = (prices[i]-ep)/ep*100
            trades.append({'entry':entry_date,'exit':dates[i],'pnl':round(pnl,2)})
    if not trades: return {'name':name,'trades':0,'total_pnl':0,'win_rate':0,'avg_win':0,'avg_loss':0,'profit_factor':0,'max_dd':0}
    wins = [t for t in trades if t['pnl']>0]; losses = [t for t in trades if t['pnl']<=0]
    aw = sum(t['pnl'] for t in wins)/len(wins) if wins else 0
    al = sum(t['pnl'] for t in losses)/len(losses) if losses else 0
    # 最大回撤
    peak = 0; mdd = 0; cum = 0
    for t in trades:
        cum += t['pnl']; peak = max(peak, cum); mdd = min(mdd, cum-peak)
    return {
        'name': name, 'trades': len(trades), 'wins': len(wins),
        'win_rate': round(len(wins)/len(trades)*100,1),
        'avg_win': round(aw,2), 'avg_loss': round(al,2),
        'total_pnl': round(sum(t['pnl'] for t in trades),2),
        'profit_factor': round(abs(aw/al),2) if al else 99,
        'max_dd': round(mdd,2)
    }

# ===== 定义因子 =====
def build_factors():
    """返回买入/卖出条件函数列表"""
    return [
        # === 单因子 ===
        ("F1:M岛?柱转正(纯MACD)", 
         lambda i,ind,p: sf(ind['hist'],i) is not None and sf(ind['hist'],i) > 0 and sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) <= 0,
         lambda i,ind,p: sf(ind['hist'],i) is not None and sf(ind['hist'],i) < 0 and sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) >= 0),
        
        ("F2:RSI超卖<30买入>70卖出",
         lambda i,ind,p: sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) <= 30 and sf(ind['rsi'],i-1) is not None and sf(ind['rsi'],i-1) > 30,
         lambda i,ind,p: sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) >= 70 and sf(ind['rsi'],i-1) is not None and sf(ind['rsi'],i-1) < 70),
        
        ("F3:价>20日线买入跌破卖出(均线)",
         lambda i,ind,p: sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i) and (not sf(ind['ma20'],i-1) or p[i-1] <= sf(ind['ma20'],i-1)),
         lambda i,ind,p: sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i) and (not sf(ind['ma20'],i-1) or p[i-1] >= sf(ind['ma20'],i-1))),
        
        ("F4:均线多头(价>20>60)入场",
         lambda i,ind,p: sf(ind['ma20'],i) is not None and sf(ind['ma60'],i) is not None and p[i] > sf(ind['ma20'],i) > sf(ind['ma60'],i) and not (sf(ind['ma20'],i-1) is not None and sf(ind['ma60'],i-1) is not None and p[i-1] > sf(ind['ma20'],i-1) > sf(ind['ma60'],i-1)),
         lambda i,ind,p: sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i)),
        
        ("F5:52周低位买入高位卖出",
         lambda i,ind,p: sf(ind['pos52'],i) is not None and sf(ind['pos52'],i) < 20 and (sf(ind['pos52'],i-1) is None or sf(ind['pos52'],i-1) >= 20),
         lambda i,ind,p: sf(ind['pos52'],i) is not None and sf(ind['pos52'],i) > 80 and (sf(ind['pos52'],i-1) is None or sf(ind['pos52'],i-1) <= 80)),
        
        # === 双因子组合 ===
        ("C1:MACD金叉+价>20日(🥇新模型)",
         lambda i,ind,p: sf(ind['hist'],i) is not None and sf(ind['hist'],i) > 0 and sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) <= 0 and sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i) and not (sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) > 0 and sf(ind['ma20'],i-1) is not None and p[i-1] > sf(ind['ma20'],i-1)),
         lambda i,ind,p: (sf(ind['hist'],i) is not None and sf(ind['hist'],i) < 0 and sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) >= 0) or (sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i))),
        
        ("C2:MACD多头+RSI健康(<60)+价>20",
         lambda i,ind,p: sf(ind['hist'],i) is not None and sf(ind['hist'],i) > 0 and sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) < 60 and sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i) and not (sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) > 0 and sf(ind['rsi'],i-1) is not None and sf(ind['rsi'],i-1) < 60 and sf(ind['ma20'],i-1) is not None and p[i-1] > sf(ind['ma20'],i-1)),
         lambda i,ind,p: (sf(ind['hist'],i) is not None and sf(ind['hist'],i) < 0) or (sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) > 70) or (sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i))),
        
        ("C3:RSI超跌<35+MACD>0+价>20(🥈新模型)",
         lambda i,ind,p: sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) < 35 and sf(ind['hist'],i) is not None and sf(ind['hist'],i) > 0 and sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i) and not (sf(ind['rsi'],i-1) is not None and sf(ind['rsi'],i-1) < 35 and sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) > 0 and sf(ind['ma20'],i-1) is not None and p[i-1] > sf(ind['ma20'],i-1)),
         lambda i,ind,p: (sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) > 68) or (sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i))),
        
        # === 三因子组合 ===
        ("C4:MACD多头+价>20>60+RSI<65(全确认)",
         lambda i,ind,p: sf(ind['hist'],i) is not None and sf(ind['hist'],i) > 0 and sf(ind['ma20'],i) is not None and sf(ind['ma60'],i) is not None and p[i] > sf(ind['ma20'],i) > sf(ind['ma60'],i) and sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) < 65 and not (sf(ind['hist'],i-1) is not None and sf(ind['hist'],i-1) > 0 and sf(ind['ma20'],i-1) is not None and sf(ind['ma60'],i-1) is not None and p[i-1] > sf(ind['ma20'],i-1) > sf(ind['ma60'],i-1) and sf(ind['rsi'],i-1) is not None and sf(ind['rsi'],i-1) < 65),
         lambda i,ind,p: (sf(ind['hist'],i) is not None and sf(ind['hist'],i) < 0) or (sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i)) or (sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) > 75)),
        
        # === 旧模型（潜伏信号） ===
        ("OLD:潜伏RSI<60+价>20+涨(旧模型)",
         lambda i,ind,p: sf(ind['rsi'],i) is not None and sf(ind['rsi'],i) < 60 and sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i) and p[i] > p[i-1] and (i<2 or p[i-1] <= p[i-2]),
         lambda i,ind,p: sf(ind['rsi'],i) is not None and (sf(ind['rsi'],i) > 70 or (sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i)))),
        
        # === 多因子评分系统 ===
        ("MF:多因子评分(加权买入,跌破20出)",
         lambda i,ind,p: None,  # 特殊处理
         lambda i,ind,p: None),
    ]

# ===== 多因子评分模型 =====
def multi_factor_score(i, ind, p):
    """评分系统: 0-100"""
    score = 0
    # 1. MACD柱 (35%)
    if sf(ind['hist'],i) is not None:
        if sf(ind['hist'],i) > 0 and (sf(ind['hist'],i-1) is None or sf(ind['hist'],i-1) <= 0): score += 35
        elif sf(ind['hist'],i) > sf(ind['hist'],i-1) if sf(ind['hist'],i-1) is not None else False: score += 20
        elif sf(ind['hist'],i) > 0: score += 10
    
    # 2. 均线系统 (25%)
    if sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i): score += 10
    if sf(ind['ma5'],i) is not None and sf(ind['ma20'],i) is not None and sf(ind['ma5'],i) > sf(ind['ma20'],i): score += 10
    if sf(ind['ma20'],i) is not None and sf(ind['ma60'],i) is not None and sf(ind['ma20'],i) > sf(ind['ma60'],i): score += 5
    
    # 3. RSI (20%)
    if sf(ind['rsi'],i) is not None:
        if sf(ind['rsi'],i) < 35: score += 20
        elif sf(ind['rsi'],i) < 60: score += 15
        elif sf(ind['rsi'],i) < 70: score += 10
        else: score -= 10
    
    # 4. 52周位置 (20%)
    if sf(ind['pos52'],i) is not None:
        if sf(ind['pos52'],i) < 30: score += 20
        elif sf(ind['pos52'],i) < 60: score += 15
        elif sf(ind['pos52'],i) < 80: score += 10
        else: score += 5
    
    return max(0, min(100, score))

# ===== 主回测 =====
def run():
    all_results = []
    
    for sector, code in STOCKS:
        name_short = code.split('.')[1]
        print(f"\n{'='*55}")
        print(f"  [{sector}] {code} — 正在拉取3年数据...")
        
        data = fetch(code, "2023-05-01", "2026-05-12")
        if not data or len(data) < 100:
            print(f"  ❌ 数据不足 ({len(data) if data else 0}天)")
            continue
        
        prices = [d['收盘价'] for d in data]
        dates = [d['时间'][:10] for d in data]
        ind = full_indicators(prices)
        
        hold_pnl = (prices[-1]-prices[0])/prices[0]*100
        vol = (max(prices)-min(prices))/min(prices)*100 if min(prices) > 0 else 0
        print(f"  ✅ {len(data)}天 区间:{min(prices):.2f}~{max(prices):.2f} 持有:{hold_pnl:+.1f}%")
        
        factors = build_factors()
        stock_results = []
        
        for name, buy_fn, sell_fn in factors:
            if name == "MF:多因子评分(加权买入,跌破20出)":
                # 多因子评分系统: 评分>60买入, <40卖出
                def mf_buy(i, ind, p): return multi_factor_score(i, ind, p) >= 65 and (i==0 or multi_factor_score(i-1, ind, p) < 65)
                def mf_sell(i, ind, p): return multi_factor_score(i, ind, p) < 40 or (sf(ind['ma20'],i) is not None and p[i] < sf(ind['ma20'],i))
                res = test_factor(name, prices, ind, mf_buy, mf_sell, dates)
            else:
                res = test_factor(name, prices, ind, buy_fn, sell_fn, dates)
            
            if res['trades'] >= 2:
                stock_results.append(res)
                e = "🟢" if res['total_pnl'] > 0 else "🔴"
                bm = "🥇" if res['total_pnl'] > hold_pnl and res['trades'] >= 3 else ""
                print(f"  {bm}{e} {res['name'][:28]:<28} x{res['trades']:>2} 胜率{res['win_rate']:>5.1f}% 总{res['total_pnl']:>+7.1f}% 盈亏比{res['profit_factor']}")
        
        # 买入持有
        print(f"  {'─'*55}")
        print(f"  📊 买入持有: {hold_pnl:+.1f}%")
        
        if stock_results:
            stock_results.sort(key=lambda x: x['total_pnl']*0.6 + x['win_rate']*0.4, reverse=True)
            best = stock_results[0]
            print(f"  🏆 最佳: {best['name'][:25]} 总{best['total_pnl']:+.1f}% vs 持有{hold_pnl:+.1f}%")
        
            all_results.append({
                'sector': sector, 'stock': f'{code}',
                'hold': hold_pnl, 'results': stock_results, 'best': best
            })
        
        time.sleep(0.3)  # 避免请求过快
    
    # ===== 跨品种分析 =====
    print(f"\n\n{'='*55}")
    print(f"📋 跨品种多因子分析 (8行业×3龙头)")
    print(f"{'='*55}")
    
    # 按因子汇总
    factor_perf = {}
    for r in all_results:
        for s in r['results']:
            fn = s['name']
            # 提取因子类?
            cat = fn.split(':')[0] if ':' in fn else fn[:3]
            if fn not in factor_perf: factor_perf[fn] = {'pnls':[], 'wrs':[], 'res':[]}
            factor_perf[fn]['pnls'].append(s['total_pnl'])
            factor_perf[fn]['wrs'].append(s['win_rate'])
            factor_perf[fn]['res'].append(s)
    
    # 因子排名
    print(f"\n  {'因子':<32} {'平均收益':>7} {'胜率':>6} {'总交易':>6} {'盈亏比':>7} {'跑赢持有':>8}")
    print(f"  {'─'*32} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*8}")
    
    ranked = sorted(factor_perf.items(), key=lambda x: sum(x[1]['pnls'])/len(x[1]['pnls']), reverse=True)
    for fn, stats in ranked:
        ap = sum(stats['pnls'])/len(stats['pnls'])
        aw = sum(stats['wrs'])/len(stats['wrs'])
        tt = sum(s['trades'] for s in stats['res'])
        apf = sum(s['profit_factor'] for s in stats['res'] if isinstance(s['profit_factor'],(int,float)))/len(stats['res'])
        # 跑赢持有的次数
        beat_hold = sum(1 for i,r in enumerate(all_results) for s in r['results'] if s['name']==fn and s['total_pnl'] > r['hold'])
        total = sum(1 for r in all_results for s in r['results'] if s['name']==fn)
        beat_pct = round(beat_hold/total*100,1) if total > 0 else 0
        m = "🏆" if fn == ranked[0][0] else " "
        print(f"  {m} {fn[:30]:<30} {ap:>+6.1f}% {aw:>5.1f}% {tt:>5} {apf:>5.1f}x {beat_pct:>6.1f}%")
    
    # 新模型 vs 旧模型 vs 持有
    print(f"\n\n{'='*55}")
    print(f"📊 新旧模型对比")
    print(f"{'='*55}")
    
    new_model_names = ["C1:MACD金叉+价>20日(🥇新模型)", "C2:MACD多头+RSI健康(<60)+价>20", "C3:RSI超跌<35+MACD>0+价>20(🥈新模型)", "C4:MACD多头+价>20>60+RSI<65(全确认)", "MF:多因子评分(加权买入,跌破20出)"]
    old_model_names = ["OLD:潜伏RSI<60+价>20+涨(旧模型)", "F3:价>20日线买入跌破卖出(均线)"]
    
    for label, names in [("🔴 旧模型(潜伏信号)", old_model_names), ("🟢 新模型(多因子)", new_model_names)]:
        print(f"\n  {label}:")
        totals = []
        for fn in names:
            if fn in factor_perf:
                stats = factor_perf[fn]
                avg_pnl = sum(stats['pnls'])/len(stats['pnls'])
                avg_wr = sum(stats['wrs'])/len(stats['wrs'])
                totals.append(avg_pnl)
                print(f"    {fn[:32]:<32} 平均收益{avg_pnl:>+6.1f}% 平均胜率{avg_wr:>5.1f}%")
        
        if totals:
            avg_all = sum(totals)/len(totals)
            print(f"    {'─'*55}")
            print(f"    平均: {avg_all:+.1f}%")
    
    # 最终结论
    print(f"\n\n{'='*55}")
    print(f"✅ 最终结论")
    print(f"{'='*55}")
    print(f"")
    
    # 最优因子
    best_name, best_stats = ranked[0]
    best_avg = sum(best_stats['pnls'])/len(best_stats['pnls'])
    print(f"  🏆 最优单因子: {best_name}")
    print(f"     跨品种平均: {best_avg:+.1f}%")
    print(f"")
    
    # 新模型综合
    new_avgs = []
    for fn in new_model_names:
        if fn in factor_perf:
            new_avgs.append(sum(factor_perf[fn]['pnls'])/len(factor_perf[fn]['pnls']))
    new_avg = sum(new_avgs)/len(new_avgs) if new_avgs else 0
    
    old_avgs = []
    for fn in old_model_names:
        if fn in factor_perf:
            old_avgs.append(sum(factor_perf[fn]['pnls'])/len(factor_perf[fn]['pnls']))
    old_avg = sum(old_avgs)/len(old_avgs) if old_avgs else 0
    
    # 买入持有
    hold_avgs = [r['hold'] for r in all_results]
    hold_avg = sum(hold_avgs)/len(hold_avgs) if hold_avgs else 0
    
    print(f"  新模型平均: {new_avg:+.1f}%")
    print(f"  旧模型平均: {old_avg:+.1f}%")
    print(f"  买入持有平均: {hold_avg:+.1f}%")
    print(f"")
    
    if new_avg > old_avg:
        print(f"  ✅ 新模型比旧模型提升: {new_avg-old_avg:+.1f}%")
    else:
        print(f"  ⚠️ 旧模型比新模型: {old_avg-new_avg:+.1f}% (需进一步优化)")
    
    return all_results, factor_perf

if __name__ == "__main__":
    print("🔥 A股多因子量化回测 v3 启动")
    print(f"   数据源: {RELAY}")
    print(f"   股票数: {len(STOCKS)} 只 (8行业×3龙头)")
    print(f"   时间跨度: 2023-05 ~ 2026-05 (3年)")
    print(f"   因子数: 11个 (5单因子+4组合+1多因子+1旧模型)")
    print(f"   {'─'*55}")
    
    results, factors = run()
