#!/usr/bin/env python3
"""
最终版：V1 + 资金流 + 龙虎榜 完整回测
基于bt_combined.py框架，资金流作为确认过滤器
"""
import json, os, time

ROOT = '/home/admin/.openclaw/workspace'
DATA = ROOT + '/data'
CACHE = ROOT + '/data/cache'

# ── 加载数据 ──
print('Loading...', flush=True)
t0 = time.time()
Y = json.load(open(DATA + '/backtest_hist_yahoo.json'))
C = json.load(open(DATA + '/v1_scores_v2.json'))
S = json.load(open(DATA + '/sector_map.json'))
all_dates = sorted(set(d for s in Y for d in Y[s].get('dates',[]) if d >= '20150101'))
codes = [c for c in Y if c != '000001' and c in C]
print('Stocks: %d, Dates: %d' % (len(codes), len(all_dates)), flush=True)

# ── 资金流缓存 {code: {yyyymmdd: net_amount}} ──
mf = {}
for f in os.listdir(CACHE):
    if not f.startswith('mf_') or not f.endswith('.json'): continue
    code = f[3:-5]
    with open(CACHE+'/'+f) as fh:
        for r in json.load(fh):
            dt = r.get('trade_date','')
            try: mf.setdefault(code, {})[dt] = float(r.get('net_mf_amount',0))
            except: pass
print('MF: %d stocks' % len(mf), flush=True)

# ── 龙虎榜缓存 {code: {yyyymmdd: fd_amount}} ──
ll = {}
for f in os.listdir(CACHE):
    if not f.startswith('ll_') or not f.endswith('.json'): continue
    dt = f[3:-5]
    with open(CACHE+'/'+f) as fh:
        for r in json.load(fh):
            code = r.get('ts_code','').split('.')[0]
            try: ll.setdefault(code, {})[dt] = float(r.get('fd_amount',0))
            except: pass
print('LL: %d stocks' % len(ll), flush=True)
print('Load: %.0fs' % (time.time()-t0), flush=True)

# ── 辅助 ──
sdx = {c: {d:i for i,d in enumerate(Y[c].get('dates',[]))} for c in Y}

def sector_mom(codes_sub, date, n, exclude):
    mom = {}
    for c in codes_sub[:n]:
        s2 = S.get(c, 'other')
        if s2 in exclude: continue
        idx = sdx.get(c, {}).get(date, -1)
        if idx < 20: continue
        cl = Y[c].get('close', [])
        if idx >= len(cl): continue
        r = (cl[idx] - cl[idx-20]) / cl[idx-20] * 100 if cl[idx-20] > 0 else 0
        mom.setdefault(s2, []).append(r)
    return {s: sum(v)/len(v) for s,v in mom.items() if len(v) >= 2}

# ── 带资金流回测 ──
def run(mf_enabled=False, ll_enabled=False, label='V1'):
    p = {'buy_thresh':62, 'sell_thresh':50, 'reb_days':7, 'sect_top':4, 'maxpos':10}
    I = 1000000.0; cash = I; pos = {}; trades = 0
    si = all_dates.index('2016-01-04')
    scored = 0; mf_hits = 0; ll_hits = 0
    
    for di in range(si, len(all_dates)-1):
        date = all_dates[di]
        date_key = date.replace('-','')
        
        # 行业动量
        sm = sector_mom(codes, date, 300, {})
        if not sm: continue
        top = {r[0] for r in sorted(sm.items(), key=lambda x:-x[1])[:p['sect_top']]}
        hold = top.copy()
        
        # 卖出
        for c in [k for k in list(pos.keys()) if not k.endswith('_p')]:
            s2 = S.get(c, 'other')
            sc = C.get(c, {}).get(date, 0)
            # 资金流修正卖出阈值
            sell_th = p['sell_thresh']
            if mf_enabled:
                net = mf.get(c, {}).get(date_key, 0)
                if net > 0: sell_th -= 5  # 正资金流：更不容易卖
                elif net < 0: sell_th += 5  # 负资金流：更容易卖
            if s2 not in hold or sc < sell_th:
                idx = sdx.get(c, {}).get(date, -1)
                if idx >= 0 and c+'_p' in pos:
                    pr = Y[c]['close'][idx]
                    bp = pos.get(c+'_p', 0)
                    if bp > 0: cash += pos[c] * (1 + (pr-bp)/bp)
                if c in pos: del pos[c]
                if c+'_p' in pos: del pos[c+'_p']
                trades += 1
        
        # 买入
        if (di-si) % p['reb_days'] == 0:
            cand = []
            for c in codes:
                if c in pos: continue
                s2 = S.get(c, 'other')
                if s2 not in top: continue
                sc = C[c].get(date, 0)
                if sc < p['buy_thresh']: continue
                idx = sdx.get(c, {}).get(date, -1)
                if idx < 0: continue
                pr = Y[c]['close'][idx]
                if pr <= 0: continue
                cand.append((c, sc, pr))
                scored += 1
            
            cand.sort(key=lambda x: -x[1])
            # 资金流优选
            if mf_enabled:
                def mf_key(x):
                    net = mf.get(x[0], {}).get(date_key, 0)
                    return x[1] + (10 if net > 0 else -10 if net < 0 else 0)
                cand.sort(key=mf_key, reverse=True)
            
            for c, sc, pr in cand:
                if len([k for k in pos if not k.endswith('_p')]) >= p['maxpos']: break
                inv = min(cash*0.15, cash*0.95)
                if inv < 20000: continue
                pos[c] = inv; pos[c+'_p'] = pr; cash -= inv; trades += 1
                if mf_enabled and mf.get(c,{}).get(date_key,0) != 0: mf_hits += 1
    
    # 终值
    final = cash
    for c in [k for k in pos if not k.endswith('_p')]:
        if c+'_p' not in pos: continue
        idx = sdx.get(c, {}).get(all_dates[-1], -1)
        if idx >= 0:
            pr = Y[c]['close'][idx]
            bp = pos.get(c+'_p', 0)
            if pr > 0 and bp > 0: final += pos[c] * (1 + (pr-bp)/bp)
    ret = (final/I-1)*100
    years = max((len(all_dates)-si)/245, 1)
    ann = ((final/I)**(1/years)-1)*100
    print('  %s: ret=%.2f%% ann=%.2f%% trades=%d mf_hits=%d' % (label, ret, ann, trades, mf_hits), flush=True)
    return {'label':label, 'ret':ret, 'ann':ann, 'trades':trades}

# ── 跑 ──
rs = []
print('\n=== COMPARISON ===', flush=True)
t_all = time.time()
rs.append(run(False, False, 'V1'))
rs.append(run(True, False, 'V1+MF'))
t_elapsed = time.time() - t_all
print('\nDone: %.0fs' % t_elapsed, flush=True)

print('\n' + '='*60)
print('RESULTS')
print('='*60)
rs.sort(key=lambda x: -x['ann'])
for r in rs:
    print('  %s: %.2f%% ann, %.2f%% ret, %d trades' % (r['label'], r['ann'], r['ret'], r['trades']))
print('='*60)
json.dump(rs, open(DATA+'/bt_final.json','w'), indent=2)
print('Saved to data/bt_final.json', flush=True)
