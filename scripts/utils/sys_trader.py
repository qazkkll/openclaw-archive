"""
trader.py v2 — A股每日选股
三因子：北向宏观 + 资金流评分 + V4确认

修正：V4是确认层不是硬门限
- V4>0 (MACD金叉) → 加分
- V4>60 → 强确认
- V4=0但资金流强 → 仍可买入（降低评级）
"""
import json, os, sys, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORKSPACE NORTH_MONEY
_cache = {}

def ensure_float(v):
    if v is None: return 0.0
    try: return float(v)
    except: return 0.0

def load_all():
    print("Loading...", flush=True)
    _cache['mf'] = json.load(open(f'{WORKSPACE}/data/a1_daily.json', 'rb'))
    _cache['kl'] = json.load(open(f'{WORKSPACE}/data/a_hist_10y.parquet', 'rb'))
    
    north = json.load(open(NORTH_MONEY, 'rb'))
    records = north.get('records', north)
    _cache['ndates'] = [r['trade_date'] for r in records]
    nv = [ensure_float(r.get('north_money', 0)) for r in records]
    _cache['nvals'] = nv
    n_mom = []
    for i in range(59, len(nv)):
        s20 = sum(nv[i-19:i+1]); s60 = sum(nv[i-59:i+1])
        n_mom.append(s20 / s60 if s60 != 0 else 1.0)
    _cache['n_mom'] = n_mom
    
    sys.path.insert(0, f'{WORKSPACE}/scripts')
    from us_score_engine import compute_indicators as ci, v1_score as v1
    _cache['ci'] = ci; _cache['v1'] = v1
    
    pool = [c for c in _cache['kl'] if (c.startswith('6') or c.startswith('0'))
            and len(c) == 6 and len(_cache['kl'][c].get('c', [])) >= 2400]
    _cache['pool'] = pool
    print(f"  {len(pool)} stocks in pool", flush=True)

def nb_pct(date_str):
    nd, nv = _cache['ndates'], _cache['nvals']
    idx = -1
    for i, d in enumerate(nd):
        if d == date_str: idx = i; break
    if idx < 60: return 50
    s20 = sum(nv[idx-19:idx+1]); s60 = sum(nv[idx-59:idx+1])
    mom = s20 / s60 if s60 != 0 else 1.0
    return sum(1 for m in _cache['n_mom'] if m < mom) / len(_cache['n_mom']) * 100

def mf_score(rec):
    """资金流评分（A1-B核心）"""
    nm = ensure_float(rec.get('net_mf', 0))
    be = ensure_float(rec.get('buy_elg', 0))
    se = ensure_float(rec.get('sell_elg', 0))
    bl = ensure_float(rec.get('buy_lg', 0))
    sl = ensure_float(rec.get('sell_lg', 0))
    tt = be + se + bl + sl
    if tt == 0: return 0
    br = (be + bl - se - sl) / tt * 100
    return nm / 10000 * 0.4 + max(br, 0) * 0.6

def v4_score(code):
    """V4评分（缓存指标）"""
    kd = _cache['kl'].get(code)
    if not kd or len(kd.get('c', [])) < 60: return 0
    ck = f'ind_{code}'
    if ck not in _cache:
        ind = _cache['ci'](kd['c'], kd['h'], kd['l'])
        _cache[ck] = ind
    ind = _cache.get(ck)
    if ind is None: return 0
    try:
        sc = _cache['v1'](ind, -1)
        return float(sc) if sc else 0
    except: return 0

def run():
    load_all()
    mf_dates = sorted(_cache['mf'].keys())
    today = mf_dates[-1]
    nb = nb_pct(today)
    
    # Use the latest available mf data
    mf_today = _cache['mf'].get(today, {})
    if not mf_today:
        for d in reversed(mf_dates):
            mf_today = _cache['mf'].get(d, {})
            if mf_today: 
                today = d
                break
    
    print(f"\nLayer 1: 北向百分位={nb:.0f}%")
    if nb > 65: print("  >> 进攻模式"); top_n = 5
    elif nb < 35: print("  >> 防守模式"); top_n = 3
    else: print("  >> 中性模式"); top_n = 5
    
    # MF scoring
    cands = [(mf_score(rec), code) for code, rec in mf_today.items() 
             if code in _cache['pool']]
    cands.sort(key=lambda x: -x[0])
    top = cands[:15]
    
    print(f"\n{'代码':>8} {'资金流':>6} {'V4':>6} {'确认':>6} {'综合':>6} {'结论':>8}")
    for sc, code in top:
        v4 = v4_score(code)
        
        # V4确认层
        if v4 > 60: conf = '强✅'; bonus = 15
        elif v4 > 0: conf = '弱⚠️'; bonus = 5
        else: conf = '无❌'; bonus = 0
        
        # 综合分：资金流 + V4确认加分
        total = sc + bonus
        
        if total > 45: conclusion = '✅买'
        elif total > 35: conclusion = '👀观'
        else: conclusion = '❌不'
        
        print(f"{code:>8} {sc:>6.1f} {v4:>5.1f} {conf:>6} {total:>5.1f} {conclusion:>8}")
    
    # Final picks
    final = []
    for sc, code in top[:top_n*2]:
        v4 = v4_score(code)
        bonus = 15 if v4 > 60 else (5 if v4 > 0 else 0)
        total = sc + bonus
        if total > 45:
            final.append((code, sc, v4, total))
    
    print(f"\n最终建议: {len(final)} 只")
    if final:
        for c, s, v, t in final:
            print(f"  {c} 资金流={s:.1f} V4={v:.1f} 综合={t:.1f}")
    else:
        print("  今日没有符合条件的买入建议")

if __name__ == "__main__":
    _cache['t0'] = time.time()
    run()
    print(f"\n耗时: {time.time()-_cache['t0']:.1f}s")
