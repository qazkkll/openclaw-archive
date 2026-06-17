#!/usr/bin/env python3
"""
门控组合测试框架 - 短周期快速筛选 -> 长周期验证

用法:
  python3 scripts/test_gates.py                  # 跑全部组合
  python3 scripts/test_gates.py --quick          # 只跑短周期
  python3 scripts/test_gates.py --gates mf,lhb   # 指定组合
"""
import json, sys, os, time, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from score_engine import compute_indicators, v1_score, safe, v1_score_from_data

# ========== 数据加载 ==========
print("Loading data...", flush=True)
with open(f'{ROOT}/data/backtest_hist_yahoo.json', encoding='utf-8') as f: YAHOO = json.load(f)
with open(f'{ROOT}/data/v1_scores_v2.json') as f: CACHE = json.load(f)
with open(f'{ROOT}/data/sector_map.json', encoding='utf-8') as f: SMAP = json.load(f)

# Try loading 龙虎榜 data
LHB = {}
try:
    with open(f'{ROOT}/data/historical_longhu.json') as f: lhb_data = json.load(f)
    for rec in lhb_data.get('records', []):
        date, item = rec[0], rec[1]
        code = item[1] if len(item) > 1 else ''
        if code and date:
            if code not in LHB: LHB[code] = set()
            LHB[code].add(date)
    print(f"  龙虎榜 loaded: {len(LHB)} stocks")
except: print("  龙虎榜: no data")

# Try loading fund flow factor cache
FF = {}
try:
    with open(f'{ROOT}/data/tushare_factors_full.json') as f: tf = json.load(f)
    FF = tf
    print(f"  资金流 loaded: {len(FF)} stocks")
except: print("  资金流: no data")

codes = [c for c in YAHOO if c != '000001']
all_dates = sorted(set(d for s in YAHOO for d in YAHOO[s].get('dates',[]) if d))
sdx = {c:{d:i for i,d in enumerate(YAHOO[c].get('dates',[]))} for c in YAHOO}
si = all_dates.index(min(min(v.keys()) for v in CACHE.values()))
EXCLUDED = {'地\u4ea7\u57fa\u5efa', '\u519c\u4e1a', '\u4ea4\u901a\u7269\u6d41'}
print(f"Data: {len(codes)} stocks, {len(all_dates)} days")

# ========== 门控函数 ==========
def sect_mom(date, top_n=3, sample_n=300):
    """行业动量排名"""
    mom = {}
    for c in codes[:sample_n]:
        sec = SMAP.get(c, '其他')
        if sec in EXCLUDED: continue
        ci = sdx.get(c,{}).get(date,-1)
        if ci < 20: continue
        cl = YAHOO[c].get('close',[])
        if not cl or ci >= len(cl): continue
        r = (cl[ci]/cl[ci-20]-1)*100
        mom.setdefault(sec,[]).append(r)
    avg = {s:sum(v)/len(v) for s,v in mom.items() if len(v)>=2}
    if not avg: return set()
    return {r[0] for r in sorted(avg.items(),key=lambda x:-x[1])[:top_n]}

def gate_fundflow(code, date, days=5):
    """门控: 资金流累计净流入 > 0"""
    stock = FF.get(code, {})
    ff = stock.get('ff', {})
    dl = ff.get('d', [])
    try: idx = dl.index(date)
    except: return None  # no data = unknown, return None
    if idx < days: return None
    net = sum(ff['net'][idx-days+1:idx+1])
    return net > 0

def gate_longhu(code, date, lookback=5):
    """门控: 近期上过龙虎榜"""
    dates = LHB.get(code, set())
    if not dates: return None
    # Check if stock appeared on 龙虎榜 in last N days
    for d in dates:
        if d <= date and d >= all_dates[max(0, all_dates.index(date)-lookback)]:
            return True
    return False

def gate_pe(code, date, min_pe=5, max_pe=100):
    """门控: PE范围"""
    stock = FF.get(code, {})
    daily = stock.get('daily', {})
    dl = daily.get('d', [])
    try: idx = dl.index(date)
    except: return None
    pe = daily.get('pe', [0])[idx]
    if pe <= 0: return None
    return min_pe <= pe <= max_pe

def gate_turnover(code, date, min_t=0.5):
    """门控: 换手率"""
    stock = FF.get(code, {})
    daily = stock.get('daily', {})
    dl = daily.get('d', [])
    try: idx = dl.index(date)
    except: return None
    to = daily.get('to', [0])[idx]
    return to >= min_t

def gate_volratio(code, date, min_vr=0.8):
    """门控: 量比"""
    stock = FF.get(code, {})
    daily = stock.get('daily', {})
    dl = daily.get('d', [])
    try: idx = dl.index(date)
    except: return None
    vr = daily.get('vr', [0])[idx]
    return vr >= min_vr

def gate_ma_trend(code, date, short_ma=5, long_ma=10):
    """门控: 短期均线 > 长期均线 (多头排列)"""
    ci = sdx.get(code, {}).get(date, -1)
    if ci < long_ma: return None
    cl = YAHOO[code].get('close', [])
    if not cl or ci >= len(cl): return None
    ma_short = sum(cl[ci-short_ma+1:ci+1]) / short_ma
    ma_long = sum(cl[ci-long_ma+1:ci+1]) / long_ma
    return ma_short > ma_long

# ========== 回测引擎 ==========
def backtest(gates, name, start_date='2015-01-01', end_date='2026-05-14'):
    """Run V3 with specified gates"""
    try: si2 = all_dates.index(start_date)
    except: si2 = si
    try: ei2 = all_dates.index(end_date) if end_date in all_dates else len(all_dates)
    except: ei2 = len(all_dates)
    
    I = 1000000.0; cash = I; pos = {}; trades = 0
    
    for di in range(si2, min(ei2, len(all_dates)-1)):
        date = all_dates[di]
        
        # 1. Sector momentum (always on)
        top_secs = sect_mom(date, 3)
        hold_secs = top_secs.copy()
        
        # 2. Score all stocks and apply gates
        candidates = {}
        for c in codes:
            sc = CACHE[c].get(date, 0)
            if sc <= 0: continue
            sec = SMAP.get(c, '其他')
            
            # Sector gate
            if sec not in top_secs: continue
            
            # Apply additional gates
            gate_pass = True
            gate_details = []
            for gate_name in gates:
                if gate_name == 'mf':  # fund flow
                    r = gate_fundflow(c, date)
                    if r is False: gate_pass = False; break
                elif gate_name == 'lhb':  # 龙虎榜
                    r = gate_longhu(c, date)
                    if r is False: gate_pass = False; break
                elif gate_name == 'pe':  # PE filter
                    r = gate_pe(c, date)
                    if r is False: gate_pass = False; break
                elif gate_name == 'to':  # turnover
                    r = gate_turnover(c, date)
                    if r is False: gate_pass = False; break
                elif gate_name == 'vr':  # volume ratio
                    r = gate_volratio(c, date)
                    if r is False: gate_pass = False; break
                elif gate_name == 'ma':  # MA trend
                    r = gate_ma_trend(c, date)
                    if r is False: gate_pass = False; break
            
            if gate_pass:
                candidates[c] = (sc, sec)
        
        if not candidates: continue
        ranked = sorted(candidates.items(), key=lambda x:-x[1][0])
        
        # Sell
        for c in list(pos.keys()):
            sc = candidates.get(c, (0,))[0]
            sec = SMAP.get(c, '其他')
            if sc < 50 or sec not in hold_secs:
                ci = sdx.get(c,{}).get(date,-1)
                if ci >= 0:
                    pr = YAHOO[c]['close'][ci]
                    if pr > 0: cash += pos[c] * (1 + (pr - pos[c+'_p']) / pos[c+'_p'])
                del pos[c]; del pos[c+'_p']; trades += 1
        
        # Rebalance
        if (di - si2) % 7 == 0:
            ca = {}
            for c, (sc, sec) in ranked:
                if c in pos: continue
                if sc < 62: continue
                ca.setdefault(sec, []).append((c, sc))
            
            for sec in top_secs:
                cs = sorted(ca.get(sec, []), key=lambda x:-x[1])
                for c, sc in cs[:2]:
                    if len(pos) >= 10: break
                    ci = sdx.get(c,{}).get(date,-1)
                    if ci < 0: continue
                    pr = YAHOO[c]['close'][ci]
                    if pr <= 0: continue
                    inv = min(cash * 0.15, cash * 0.95)
                    if inv < 20000: continue
                    pos[c] = inv; pos[c+'_p'] = pr
                    cash -= inv; trades += 1
    
    # Final
    fin = cash
    for c in [k for k in pos if not k.endswith('_p')]:
        ci = sdx.get(c,{}).get(all_dates[-1],-1)
        if ci >= 0:
            pr = YAHOO[c]['close'][ci]
            if pr > 0: fin += pos[c] * (1 + (pr - pos[c+'_p']) / pos[c+'_p'])
    
    ret = (fin/I-1)*100
    yrs = max((ei2-si2)/245, 1)
    ann = ((fin/I)**(1/yrs)-1)*100
    return ret, ann, trades

# ========== 测试组合 ==========
GATE_COMBOS = [
    ([], 'V3_基准(3行业)'),
    (['mf'], 'V3+资金流'),
    (['lhb'], 'V3+龙虎榜'),
    (['pe'], 'V3+PE过滤'),
    (['to'], 'V3+换手率'),
    (['vr'], 'V3+量比'),
    (['ma'], 'V3+均线趋势'),
    (['mf','pe'], 'V3+资金流+PE'),
    (['mf','lhb'], 'V3+资金流+龙虎榜'),
    (['to','vr'], 'V3+换手率+量比'),
    (['pe','to','vr'], 'V3+PE+换手率+量比'),
    (['mf','pe','to','vr'], 'V3+资金流+PE+换手率+量比'),
    (['mf','lhb','pe','to','vr'], 'V3+资金流+龙虎榜+PE+换手率+量比'),
    (['mf','pe','to','vr','ma'], 'V3+资金流+PE+换手率+量比+均线'),
]

SHORT_PERIODS = [
    ('2020-2021', '2020-01-02', '2022-01-03'),
    ('2023-2024', '2023-01-03', '2025-01-02'),
    ('2020-2024', '2020-01-02', '2025-01-02'),
]
FULL_PERIOD = ('2015-2025', '2015-01-05', '2026-05-14')

print(f"\n{'='*80}")
print(f"  门控组合快速筛选 (短周期)")
print(f"{'='*80}")

results = []
for gates, name in GATE_COMBOS:
    g_name = '+'.join(gates) if gates else '无'
    total_short = 0
    years_count = 0
    
    for p_name, start_s, end_s in SHORT_PERIODS:
        t0 = time.time()
        ret, ann, trades = backtest(gates, name, start_s, end_s)
        elapsed = time.time() - t0
        total_short += ret
        years_count += 1
        print(f"  {name:35s} {p_name:12s} {ret:>+8.2f}%  {ann:>+6.2f}%/yr  {trades:>4d}tr  {elapsed:.0f}s", flush=True)
    
    avg_short = total_short / years_count if years_count else 0
    results.append((name, gates, avg_short))

# Top 5 short-cycle go to long-cycle
results.sort(key=lambda x: -x[2])
print(f"\n{'='*80}")
print(f"  Top 5 进长周期验证")
print(f"{'='*80}")

top5 = results[:5]
for name, gates, avg_short in top5:
    t0 = time.time()
    ret, ann, trades = backtest(gates, name, FULL_PERIOD[1], FULL_PERIOD[2])
    elapsed = time.time() - t0
    prefix = '+' if ret >= 0 else ''
    print(f"  {name:35s} {'FULL':>12s} {prefix}{ret:>+8.2f}%  {ann:>+6.2f}%/yr  {trades:>4d}tr  {elapsed:.0f}s")

# Save results
print(f"\n{'='*80}")
print(f"  Done")
print(f"{'='*80}")
