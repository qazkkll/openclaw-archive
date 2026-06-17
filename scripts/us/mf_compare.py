#!/usr/bin/env python3
"""
自包含对比：V1 vs V1+MF vs V1+MF+LL
使用预计算评分 + 缓存资金流，采样加速
"""
import json, os, time, sys
ROOT = '/home/admin/.openclaw/workspace'
DATA = ROOT + '/data'
CACHE = ROOT + '/data/cache'

# ── 加载数据 ──
print('Loading...', flush=True)
t0 = time.time()
Y = json.load(open(DATA + '/backtest_hist_yahoo.json'))
C = json.load(open(DATA + '/v1_scores_v2.json'))
all_dates = sorted(set(d for s in Y for d in Y[s].get('dates',[]) if d >= '20150101'))
print('Dates: %d' % len(all_dates), flush=True)

# ── 构建资金流索引 {code: {date_yyyymmdd: net_amount}} ──
mf_idx = {}
for f in os.listdir(CACHE):
    if not f.startswith('mf_') or not f.endswith('.json'): continue
    code = f[3:-5]
    data = json.load(open(CACHE+'/'+f))
    dmap = {}
    for r in data:
        dt = r.get('trade_date','')
        try: dmap[dt] = float(r.get('net_mf_amount',0))
        except: pass
    if dmap: mf_idx[code] = dmap
print('MF indexed: %d stocks' % len(mf_idx), flush=True)

# ── 构建龙虎榜索引 {code: {yyyymmdd: fd_amount}} ──
ll_idx = {}
for f in os.listdir(CACHE):
    if not f.startswith('ll_') or not f.endswith('.json'): continue
    dt = f[3:-5]
    data = json.load(open(CACHE+'/'+f))
    for r in data:
        code = r.get('ts_code','').split('.')[0]
        try: net = float(r.get('fd_amount',0))
        except: continue
        if net != 0:
            ll_idx.setdefault(code, {})[dt] = net
print('LL indexed: %d stocks' % len(ll_idx), flush=True)

# ── 取同时有评分+资金流的股票 ──
codes = sorted(set(C.keys()) & set(mf_idx.keys()))[:50]
print('Stocks: %d (%.0fs load)' % (len(codes), time.time()-t0), flush=True)

# ── 评分索引 {code: {date: score}} ──
scores = {}
for c in codes:
    sc = C.get(c, {})
    if sc: scores[c] = sc
print('Scores: %d stocks' % len(scores), flush=True)

# ── 日期索引 ──
def price_idx(code, date):
    cd = Y[code].get('dates',[])
    for i, d in enumerate(cd):
        if d >= date: return i
    return -1

# ── 回测引擎 ──
def backtest(mode, label):
    """mode: 'v1', 'v1+mf', 'v1+mf+ll'"""
    M = 1000000.0; cash=M; pos={}; trades=0
    si = all_dates.index('2016-01-04')
    sampled = range(si, len(all_dates)-1, 5)  # every 5 days
    
    for di in sampled:
        date = all_dates[di]
        # Sell
        for c in [k for k in list(pos.keys()) if not k.endswith('_p') and not k.endswith('_sc')]:
            sc = scores.get(c, {}).get(date, 0)
            sell_thresh = 50
            if sc < sell_thresh:
                idx = price_idx(c, date)
                if idx >= 0 and c+'_p' in pos:
                    pr = Y[c]['close'][idx]
                    bp = pos.get(c+'_p', 0)
                    if bp > 0: cash += pos[c] * (1 + (pr-bp)/bp)
                for k in [x for x in list(pos.keys()) if x.startswith(c)]: del pos[k]
                trades += 1
        # Buy
        if (di-si) % 10 == 0:
            cand = []
            for c in codes:
                if c in pos: continue
                bs = scores.get(c, {}).get(date, 0)
                if bs <= 0: continue
                sc = bs
                if 'mf' in mode:
                    net = mf_idx.get(c, {}).get(date.replace('-',''), 0)
                    sc += 5 if net > 0 else -5 if net < 0 else 0
                if 'll' in mode:
                    net2 = ll_idx.get(c, {}).get(date.replace('-',''), 0)
                    sc += 3 if net2 > 0 else -3 if net2 < 0 else 0
                if sc >= 62:
                    cand.append((c, sc))
            cand.sort(key=lambda x: -x[1])
            for c, sc in cand:
                active = [k for k in pos if not k.endswith('_p') and not k.endswith('_sc')]
                if len(active) >= 10: break
                idx = price_idx(c, date)
                if idx < 0: continue
                pr = Y[c]['close'][idx]
                if pr <= 0: continue
                inv = min(cash*0.15, cash*0.95)
                if inv < 20000: continue
                pos[c] = inv; pos[c+'_p'] = pr; pos[c+'_sc'] = sc; cash -= inv; trades += 1
    
    # 终值
    final = cash
    for c in [k for k in pos if not k.endswith('_p') and not k.endswith('_sc')]:
        idx = price_idx(c, all_dates[-1])
        if idx >= 0:
            pr = Y[c]['close'][idx]
            bp = pos.get(c+'_p', 0)
            if pr > 0 and bp > 0: final += pos[c] * (1 + (pr-bp)/bp)
    ret = (final/M-1)*100
    years = max(len(all_dates)/245, 1)
    ann = ((final/M)**(1/years)-1)*100
    print('  %s: ret=%.2f%% ann=%.2f%% trades=%d' % (label, ret, ann, trades), flush=True)
    return {'label':label, 'ret':ret, 'ann':ann, 'trades':trades}

rs = []
rs.append(backtest('v1', 'V1_only'))
rs.append(backtest('v1+mf', 'V1+MF'))
rs.append(backtest('v1+mf+ll', 'V1+MF+LL'))

print('\n' + '='*60)
print('FINAL')
print('='*60)
rs.sort(key=lambda x: -x['ann'])
for r in rs:
    print('  %s: %.2f%% ann, %.2f%% ret, %d trades' % (r['label'], r['ann'], r['ret'], r['trades']))
print('='*60)
json.dump(rs, open(DATA+'/mf_compare.json','w'), indent=2)
print('Saved')
