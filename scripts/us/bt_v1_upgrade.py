#!/usr/bin/env python3
"""
V1模型升级回测引擎 v2 — 使用真实OHLCV价格
评分用作买卖信号，价格用来算真实盈亏
"""
import json, os, sys, time
sys.setrecursionlimit(10000)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print('加载数据...', flush=True)
t0 = time.time()

with open(os.path.join(ROOT, 'data', 'precomputed_scores.json')) as f:
    SCORES = json.load(f)

with open(os.path.join(ROOT, 'data', 'backtest_hist_yahoo.json')) as f:
    YAHOO = json.load(f)

# Build price lookup
PRICE = {}
for code, data in YAHOO.items():
    if isinstance(data, dict):
        dl = data.get('dates', [])
        cl = data.get('close', [])
        for i, d in enumerate(dl):
            if i < len(cl):
                PRICE[f'{d}_{code}'] = cl[i]

ALL_DATES = sorted(set(d for c in SCORES for d in SCORES[c].keys()))
print(f'  {len(SCORES)}只, {len(ALL_DATES)}天, {len(PRICE)}价格点, {time.time()-t0:.1f}s', flush=True)

BUY = 62
SELL = 50
MAX_POS = 8
COST = 0.003

def run(name, rules):
    cash = 100000.0
    pos = {}  # code -> {'cost': total, 'shares': N, 'entry_score': S, 'entry_date': D}
    
    for di, date in enumerate(ALL_DATES):
        if di < 60:
            continue
        
        # 当天评分
        daily = {}
        for code in SCORES:
            s = SCORES[code].get(date, 0)
            if s and s > 0:
                daily[code] = s
        
        ranked = sorted(daily.items(), key=lambda x: -x[1])
        
        # 卖出
        to_sell = [c for c in pos if daily.get(c, 0) < SELL]
        for code in to_sell:
            h = pos.pop(code)
            sell_price = PRICE.get(f'{date}_{code}', 0)
            if sell_price:
                proceeds = h['shares'] * sell_price * (1 - COST)
                cash += proceeds
        
        # 买入
        if len(pos) < MAX_POS:
            candidates = [(c, s) for c, s in ranked[:30] if s >= BUY and c not in pos]
            slots = MAX_POS - len(pos)
            for code, score in candidates[:slots]:
                buy_price = PRICE.get(f'{date}_{code}', 0)
                if buy_price <= 0:
                    continue
                
                # 仓位
                if rules.get('kelly'):
                    weight = min(score / 150, 0.20)
                else:
                    weight = 1.0 / MAX_POS
                
                invest = cash * weight
                shares = invest / buy_price
                cash -= invest
                
                pos[code] = {
                    'shares': shares,
                    'cost': invest,
                    'entry_score': score,
                    'entry_date': date
                }
        
        # 记录(每20天)
        if di % 20 == 0:
            val = cash + sum(
                h['shares'] * PRICE.get(f'{date}_{c}', 0) 
                for c, h in pos.items()
            )
    
    # 最终估值
    final = cash + sum(
        h['shares'] * PRICE.get(f'{ALL_DATES[-1]}_{c}', 0)
        for c, h in pos.items()
    )
    
    ret = (final / 100000 - 1) * 100
    years = len(ALL_DATES) / 245
    ann = ((final / 100000) ** (1 / max(years, 1)) - 1) * 100
    
    print(f'  {name:<20} 回报{ret:>+7.1f}%  年化{ann:>+6.1f}%  终值${final:>7.0f}', flush=True)
    
    return {'name': name, 'return': ret, 'annualized': ann, 'final': final}

print()
print('🏁 回测 (使用真实价格)')
print('─' * 60)

results = [run('V1基准(原版)', {})]
results.append(run('V1+凯利仓位', {'kelly': True}))

print()
print('📊 对比:')
for r in results:
    print(f'  {r["name"]:<20} → {r["return"]:+.1f}%')

with open(os.path.join(ROOT, 'data', 'v1_upgrade_results_v2.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n✅ 完成 ({time.time()-t0:.0f}s)')
