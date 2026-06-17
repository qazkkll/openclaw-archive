#!/usr/bin/env python3
"""
V1 参数暴力扫描 - 网格搜索最优参数组合
基于 bt_combined.py 的回测引擎，扫描关键参数组合并输出TOP10。
"""
import json, time, sys, os, itertools

ROOT = '/home/admin/.openclaw/workspace'
DATA_DIR = ROOT + '/data'

# ── 加载数据（与bt_combined.py一致） ──
Y = json.load(open(DATA_DIR + '/backtest_hist_yahoo.json'))
C = json.load(open(DATA_DIR + '/v1_scores_v2.json'))
S = json.load(open(DATA_DIR + '/sector_map.json'))

codes = [c for c in Y if c != '000001']
all_dates = sorted(set(d for s in Y for d in Y[s].get('dates',[]) if d))
sdx = {c: {d:i for i,d in enumerate(Y[c].get('dates',[]))} for c in Y}
si = all_dates.index('2016-01-04') if '2016-01-04' in all_dates else 500
print('Data: %d stocks x %d days' % (len(codes), len(all_dates)))

# ── 参数网格 ──
GRID = {
    'buy_threshold': [58, 62, 66, 70],
    'sell_threshold': [50],
    'rebalance_days': [5, 10, 15],
    'sector_top_n': [3, 4],
    'max_positions': [8, 12],
}
# 4x1x3x2x2 = 48 combinations - safe for 3.4GB RAM
KEYS = list(GRID.keys())
total = 1
for v in GRID.values(): total *= len(v)
print('Grid: %d combinations' % total)

# ── 行业动量（同bt_combined） ──
def sector_momentum(codes_subset, date, n, exclude):
    m = {}
    for c in codes_subset[:n]:
        s2 = S.get(c, 'other')
        if s2 in exclude and exclude[s2] >= 3: continue
        idx = sdx.get(c, {}).get(date, -1)
        if idx < 20: continue
        cl = Y[c].get('close', [])
        if idx >= len(cl): continue
        r = (cl[idx] - cl[idx-20]) / cl[idx-20] * 100
        m.setdefault(s2, []).append(r)
    return {s: sum(v)/len(v) for s,v in m.items() if len(v) >= 2}

# ── 单次回测 ──
def run_backtest(cfg):
    s = cfg
    I = s['initial_capital']
    cash = I
    pos = {}
    trades = 0

    for di in range(si, len(all_dates) - 1):
        date = all_dates[di]
        sm = sector_momentum(codes, date, 300, {})
        if not sm: continue
        ranked = sorted(sm.items(), key=lambda x: -x[1])
        top_sectors = {r[0] for r in ranked[:s['sector_top_n']]}
        hold_sectors = top_sectors.copy()

        # 卖出
        for c in list(pos.keys()):
            if c.endswith('_p'): continue
            s2 = S.get(c, 'other')
            sc = C.get(c, {}).get(date, 0)
            if s2 not in hold_sectors or sc < s['sell_threshold']:
                idx = sdx.get(c, {}).get(date, -1)
                if idx >= 0 and c+'_p' in pos:
                    pr = Y[c]['close'][idx]
                    bp = pos.get(c+'_p', 0)
                    if bp > 0:
                        cash += pos[c] * (1 + (pr - bp) / bp)
                if c in pos: del pos[c]
                if c+'_p' in pos: del pos[c+'_p']
                trades += 1

        # 买入（调仓日）
        if (di - si) % s['rebalance_days'] == 0:
            candidates = {}
            for c in C:
                if c in pos: continue
                s2 = S.get(c, 'other')
                if s2 not in top_sectors: continue
                sc = C[c].get(date, 0)
                if sc < s['buy_threshold']: continue
                idx = sdx.get(c, {}).get(date, -1)
                if idx < 0: continue
                pr = Y[c]['close'][idx]
                if pr <= 0: continue
                candidates.setdefault(s2, []).append((c, sc, pr))

            for s2 in top_sectors:
                ranked_c = sorted(candidates.get(s2, []), key=lambda x: -x[1])
                for c, sc, pr in ranked_c[:2]:
                    if len([k for k in pos if not k.endswith('_p')]) >= s['max_positions']: break
                    inv = min(cash * 0.15, cash * 0.95)
                    if inv < 20000: continue
                    pos[c] = inv
                    pos[c+'_p'] = pr
                    cash -= inv
                    trades += 1

    # 计算收益
    final = cash
    for c in [k for k in pos if not k.endswith('_p')]:
        if c+'_p' not in pos: continue
        idx = sdx.get(c, {}).get(all_dates[-1], -1)
        if idx >= 0:
            pr = Y[c]['close'][idx]
            bp = pos.get(c+'_p', 0)
            if pr > 0 and bp > 0:
                final += pos[c] * (1 + (pr - bp) / bp)
    ret = (final / I - 1) * 100
    years = max((len(all_dates) - si) / 245, 1)
    ann = ((final / I) ** (1/years) - 1) * 100
    return round(ret, 2), round(ann, 2), trades

# ── 扫描 ──
results = []
t0 = time.time()
keys_list = list(GRID.keys())
for i, values in enumerate(itertools.product(*(GRID[k] for k in keys_list))):
    cfg = dict(zip(keys_list, values))
    cfg['initial_capital'] = 1000000.0
    try:
        ret, ann, trades = run_backtest(cfg)
        results.append((ann, ret, trades, cfg))
        # Save partial results every 10 runs
        if (i+1) % 10 == 0:
            partial = sorted(results, key=lambda x: -x[0])
            top5 = [{'ann':r[0],'ret':r[1],'trades':r[2],'cfg':r[3]} for r in partial[:5]]
            json.dump({'progress':i+1,'total':total,'top5':top5}, open(DATA_DIR+'/bt_sweep_partial.json','w'))
    except Exception as e:
        print('  [%d/%d] ERR: %s' % (i+1, total, str(e)[:50]), flush=True)
        continue

    if (i+1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i+1) / elapsed if elapsed > 0 else 0
        eta = (total - i - 1) / rate if rate > 0 else 0
        best = max([r[0] for r in results]) if results else 0
        print('  [%d/%d] %.0fs elapsed, ETA %.0fs | best: %.1f%% ann' % (i+1, total, elapsed, eta, best), flush=True)

# ── TOP10 输出 ──
results.sort(key=lambda x: -x[0])
top10 = [{'rank': i+1, 'annualized_pct': r[0], 'total_return_pct': r[1], 'trades': r[2], 'params': r[3]}
         for i, r in enumerate(results[:10])]

out = {
    'total_combinations': total,
    'successful_runs': len(results),
    'elapsed_seconds': round(time.time() - t0),
    'top10': top10
}

with open(DATA_DIR + '/bt_sweep_results.json', 'w') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print('\n' + '='*60)
print('TOP 10 PARAMETER COMBINATIONS')
print('='*60)
for r in top10:
    print('  #%d: %.2f%% ann, %.2f%% total, %d trades - buy=%d sell=%d reb=%dd sect=%d maxpos=%d' % (
        r['rank'], r['annualized_pct'], r['total_return_pct'], r['trades'],
        r['params']['buy_threshold'], r['params']['sell_threshold'],
        r['params']['rebalance_days'], r['params']['sector_top_n'], r['params']['max_positions']))
print('='*60)
print('Results saved to data/bt_sweep_results.json')
