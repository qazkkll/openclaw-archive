"""🦐 策略对比 v2 — 使用预计算评分，快速完成"""
import json, sys, time

with open('data/precomputed_scores.json') as f:
    pre = json.load(f)
with open('data/backtest_hist_yahoo.json') as f:
    hist = json.load(f)
with open('data/sector_map.json') as f:
    smap = json.load(f)

EXCLUDED = {'地产基建','农业','交通物流'}
ETFS = set()
codes = [c for c in hist if len(hist[c].get('close',[])) > 500 and c in pre]
adates = sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2016-01-01' <= d <= '2026-05-14'))

def pr(code, date):
    idx = hist[code]['dates'].index(date) if date in hist[code]['dates'] else -1
    return hist[code]['close'][idx] if idx >= 0 else None

def run_strategy(name, pre_filter, buy=62, sell=50, maxp=8, rebal=7, minh=5):
    """通用仿真器"""
    dates = [d for d in adates]
    cash = 1000000; pos = {}; trades = []; rb = 0; peak = 1000000
    buyn=0; seln=0
    
    for di, date in enumerate(dates):
        rb += 1
        
        # 每日卖出
        for c in list(pos.keys()):
            p = pos[c]; h = di - p['bd']
            if h < minh: continue
            sc = float(pre.get(c,{}).get(date, 0))
            if sc <= 0 or sc < sell:
                price = pr(c, date) or p['bp']
                profit = (price-p['bp'])/p['bp']*100
                cash += price * p['sh']
                trades.append(f"🔴{date}|{name}|{p['nm']}({c})|{profit:+.1f}%|评分{sc}")
                seln += 1; del pos[c]
        
        # 调仓日买入
        if rb >= rebal:
            rb = 0
            
            if pre_filter == 'sector':
                # 行业动量
                sm = {}
                all_secs = set(smap.get(c,'其他') for c in codes if smap.get(c,'其他') not in EXCLUDED)
                for sec in all_secs:
                    rets = []
                    for c in [c for c in codes if smap.get(c,'其他')==sec][:20]:
                        i20 = dates[max(0,di-20)]
                        p_n = pr(c, date); p_20 = pr(c, i20)
                        if p_n and p_20 and p_20 > 0: rets.append((p_n-p_20)/p_20*100)
                    if len(rets) >= 2: sm[sec] = sum(rets)/len(rets)
                top4 = set(s[0] for s in sorted(sm.items(),key=lambda x:-x[1])[:4])
                cand_codes = [c for c in codes if smap.get(c,'其他') in top4 and c not in pos]
            else:
                cand_codes = [c for c in codes if c not in pos]
            
            cand = []
            for c in cand_codes:
                sc = float(pre.get(c,{}).get(date, 0))
                if sc >= buy:
                    p = pr(c, date)
                    if p: cand.append((c, sc, p))
            
            cand.sort(key=lambda x: -x[1])
            for c, sc, price in cand:
                if len(pos) >= maxp or cash < 5000: break
                per = cash / (maxp - len(pos))
                sh = max(100, int(per/price/100)*100)
                cost = sh * price
                if cost > cash: sh = int(cash/price/100)*100; cost = sh*price
                if sh < 100: continue
                cash -= cost
                pos[c] = {'sh':sh,'bp':price,'bd':di,'nm':hist[c].get('name',c)}
                trades.append(f"🟢{date}|{name}|{hist[c].get('name',c)}({c})|{sh}股@{price:.2f}|¥{cost:,.0f}|评分{sc}")
                buyn += 1
        
        # 净值
        pv = sum((pr(c,date) or p['bp']) * p['sh'] for c,p in pos.items())
        total = cash + pv
        if total > peak: peak = total
    
    fpv = sum((pr(c,adates[-1]) or p['bp']) * p['sh'] for c,p in pos.items())
    final = cash + fpv
    ret = (final/1000000-1)*100
    span = (__import__('datetime').datetime.strptime(adates[-1],'%Y-%m-%d')-__import__('datetime').datetime.strptime(adates[0],'%Y-%m-%d')).days
    ann = ((final/1000000)**(365/span)-1)*100 if span>0 else 0
    dd = (peak-final)/peak*100
    
    return {'name':name,'final':round(final,2),'return_pct':round(ret,2),'annualized':round(ann,2),
            'max_drawdown':round(dd,2),'buys':buyn,'sells':seln,'cash':round(cash,2),'peak':round(peak,2)}

print("🚀 对比仿真开始...")
t0 = time.time()

results = [
    run_strategy('V4直选', None, buy=62, sell=50, maxp=8),
    run_strategy('V3行业筛选', 'sector', buy=62, sell=50, maxp=5),
    run_strategy('V2.5精中选精', 'sector', buy=62, sell=48, maxp=5),
]

print(f"\n{'='*60}")
print(f"📊 策略对比 — 2016→2026")
print(f"{'='*60}")
print(f"{'策略名':<16} {'收益':>8} {'年化':>8} {'回撤':>8} {'买/卖':>10} {'现金':>10}")
print(f"{'-'*54}")
for r in results:
    print(f"{r['name']:<16} {r['return_pct']:>+7.2f}% {r['annualized']:>7.2f}% {r['max_drawdown']:>7.2f}% {r['buys']}/{r['sells']:<7} ¥{r['cash']:>8,.0f}")
print(f"\n⏱️ 耗时: {time.time()-t0:.0f}s")

json.dump(results, open('data/strategy_comparison.json','w'), indent=2, ensure_ascii=False)
