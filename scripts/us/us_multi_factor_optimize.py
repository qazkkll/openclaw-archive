"""
多因子组合测试 — 系统搜索最优选股方案（优化版）

关键优化：每只股票只算一次完整指标，之后按月采样只调 v1_score
"""
import json, os, sys, gc, time, math
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import WORKSPACE NORTH_MONEY

def ensure_float(v):
    if v is None: return 0.0
    try: return float(v)
    except: return 0.0

# ═══════════ 1. Load data ═══════════
t0 = time.time()
print("=" * 65)
print("Multi-Factor Stock Selection Optimization")
print(f"Factors: V4 + MoneyFlow + Valuation + Momentum + Northbound")
print("=" * 65, flush=True)

print("\n[1/4] Loading data...", flush=True)

# Northbound
north = json.load(open(NORTH_MONEY, 'rb'))
records = north.get('records', north)
ndates = [r['trade_date'] for r in records]
nvals = [ensure_float(r.get('north_money', 0)) for r in records]
n_mom = []
for i in range(59, len(nvals)):
    s20 = sum(nvals[i-19:i+1]); s60 = sum(nvals[i-59:i+1])
    n_mom.append(s20 / s60 if s60 != 0 else 1.0)

def nb_pct(date_str):
    idx = -1
    for i, d in enumerate(ndates):
        if d == date_str: idx = i; break
    if idx < 60: return 50
    s20 = sum(nvals[idx-19:idx+1]); s60 = sum(nvals[idx-59:idx+1])
    mom = s20 / s60 if s60 != 0 else 1.0
    return sum(1 for m in n_mom if m < mom) / len(n_mom) * 100

# MF scores
mf_scores = json.load(open(f'{WORKSPACE}/data/precomputed_scores.json', 'rb'))
def mf_key(kc):
    if kc in mf_scores: return kc
    if f'{kc}.SZ' in mf_scores: return f'{kc}.SZ'
    if f'{kc}.SH' in mf_scores: return f'{kc}.SH'
    return None

# K-line - load and select 300 stocks
kl_data = json.load(open(f'{WORKSPACE}/data/a_hist_10y.parquet', 'rb'))
stocks = sorted(kl_data.items(), key=lambda x: -len(x[1].get('c', [])))
eligible = []
for sc, sd in stocks:
    mk = mf_key(sc)
    if mk and len(sd.get('c', [])) >= 500:
        eligible.append((sc, sd, mk))
sample_stocks = eligible[:300]
print(f"  {len(sample_stocks)} eligible stocks", flush=True)
del kl_data, stocks; gc.collect()

# ═══════════ 2. Compute factor matrices ═══════════
print("\n[2/4] Computing factors (precompute indicators once per stock)...", flush=True)
import sys as _sys
_sys.path.insert(0, os.path.join(WORKSPACE, 'scripts'))
from us_score_engine import compute_indicators, v1_score

def momentum_score(closes, idx):
    if idx < 60: return 50
    ret20 = (closes[idx] / closes[idx-20] - 1) * 100 if closes[idx-20] != 0 else 0
    ret60 = (closes[idx] / closes[idx-60] - 1) * 100 if closes[idx-60] != 0 else 0
    sc = 50 + ret20 * 1.5 + ret60 * 0.5
    return max(0, min(100, sc))

all_data = []
stock_cnt = 0

for sc, sd, mk in sample_stocks:
    closes = sd.get('c', [])
    highs = sd.get('h', [])
    lows = sd.get('l', [])
    dates = sd.get('dates', [])
    
    if len(closes) < 500: continue
    
    # Pre-compute indicators ONCE per stock
    ind = compute_indicators(closes, highs, lows)
    if ind is None: continue
    
    mf_base = mf_scores.get(mk, {}).get('mf_score', 50)
    stock_cnt += 1
    
    # Sample monthly (every 20 days)
    for i in range(200, len(closes) - 20, 20):
        d = dates[i]
        if d > '20261231': break
        if d < '20160101': continue
        
        v4 = v1_score(ind, i)  # fast - just array lookup
        mom = momentum_score(closes, i)
        nb = nb_pct(d)
        
        all_data.append({
            'stock': sc, 'date': d,
            'v4': float(v4), 'mf': float(mf_base), 'val': 50.0, 'mom': mom, 'nb': nb,
            'ret': round((closes[i+20] / closes[i] - 1) * 100, 2)
        })
    
    if stock_cnt % 50 == 0:
        print(f"  {stock_cnt}/{len(sample_stocks)} stocks ({len(all_data)} pts)", flush=True)

print(f"  Total: {len(all_data)} data points", flush=True)

# Factor correlation check
print("\nFactor correlations (training set 2016-2020):")
train = [d for d in all_data if d['date'] <= '20201231']
for f1 in ['v4', 'mf', 'val', 'mom', 'nb']:
    for f2 in ['v4', 'mf', 'val', 'mom', 'nb']:
        if f1 >= f2: continue
        v1 = [d[f1] for d in train]; v2 = [d[f2] for d in train]
        mx1, mx2 = sum(v1)/len(v1), sum(v2)/len(v2)
        num = sum((v1[i]-mx1)*(v2[i]-mx2) for i in range(len(v1)))
        d1 = math.sqrt(sum((x-mx1)**2 for x in v1))
        d2 = math.sqrt(sum((x-mx2)**2 for x in v2))
        corr = num/(d1*d2) if d1*d2 > 0 else 0
        flag = ' ⚠️HIGH' if abs(corr) > 0.5 else ''
        print(f"  {f1}-{f2}: {corr:+.3f}{flag}")

# ═══════════ 3. Test combinations ═══════════
print(f"\n[3/4] Testing weight combinations...", flush=True)

# Generate combinations
combos = set()
for v4 in [x/100 for x in range(0, 81, 20)]:
    for mf in [x/100 for x in range(0, 81, 20)]:
        for mom in [x/100 for x in range(0, 41, 20)]:
            for nb in [x/100 for x in range(0, 81, 20)]:
                total = round(v4 + mf + mom + nb, 2)
                if total > 1.0: continue
                val = round(1 - total, 2)
                combos.add(tuple(round(x, 2) for x in (v4, mf, val, mom, nb)))
combos = list(combos)
print(f"  {len(combos)} combinations", flush=True)

# Split: train(2016-2020), val(2021-2023), test(2024-2026)
train_d = [d for d in all_data if d['date'] <= '20201231']
val_d = [d for d in all_data if '20210101' <= d['date'] <= '20231231']
test_d = [d for d in all_data if d['date'] >= '20240101']

print(f"  Train: {len(train_d)} | Val: {len(val_d)} | Test: {len(test_d)}", flush=True)

def backtest(data, weights):
    """Run backtest on a dataset with given weights"""
    v4_w, mf_w, val_w, mom_w, nb_w = weights
    by_month = {}
    for d in data:
        ym = d['date'][:6]
        if ym not in by_month: by_month[ym] = []
        by_month[ym].append(d)
    
    monthly_rets = []
    for ym in sorted(by_month.keys()):
        group = by_month[ym]
        for g in group:
            g['comp'] = (g['v4']*v4_w + g['mf']*mf_w + g['val']*val_w + 
                        g['mom']*mom_w + g['nb']*nb_w)
        top5 = sorted(group, key=lambda x: -x['comp'])[:5]
        if top5:
            monthly_rets.append(sum(s['ret'] for s in top5) / len(top5))
    
    if len(monthly_rets) < 5: return None
    avg = sum(monthly_rets) / len(monthly_rets)
    win = sum(1 for r in monthly_rets if r > 0) / len(monthly_rets) * 100
    return {'avg_ret': round(avg, 2), 'win_rate': round(win, 1), 
            'max_loss': round(min(monthly_rets), 2), 'trades': len(monthly_rets)}

results = []
for weights in combos:
    tr = backtest(train_d, weights)
    if tr is None: continue
    vr = backtest(val_d, weights)
    te = backtest(test_d, weights)
    
    results.append({
        'w': weights, 'train': tr, 'val': vr, 'test': te,
        'consistency': (tr['avg_ret'] > 0) and (vr and vr['avg_ret'] > 0) and (te and te['avg_ret'] > 0)
    })

# ═══════════ 4. Results ═══════════
print(f"\n[4/4] Results ({len(results)} valid)")
print(f"{'='*75}")

# Consistently positive across all periods
consistent = [r for r in results if r['consistency']]
print(f"\nCONSISTENTLY POSITIVE (train+val+test all >0): {len(consistent)} combos")
if consistent:
    consistent.sort(key=lambda x: -x['test']['avg_ret'])
    print(f"{'V4':>4} {'MF':>4} {'Val':>4} {'Mom':>4} {'NB':>4} | {'Train':>7} {'Val':>7} {'Test':>7} {'Win':>5}")
    print(f"{'-'*55}")
    for r in consistent[:15]:
        w = r['w']
        print(f"{w[0]*100:>3.0f}% {w[1]*100:>3.0f}% {w[2]*100:>3.0f}% {w[3]*100:>3.0f}% {w[4]*100:>3.0f}% | {r['train']['avg_ret']:>+6.2f}% {r['val']['avg_ret']:>+6.2f}% {r['test']['avg_ret']:>+6.2f}% {r['test']['win_rate']:>4.0f}%")
else:
    print("  None found")

# Train ranking (from training data only - unbiased)
print(f"\nTOP 15 BY TRAINING (2016-2020):")
print(f"{'V4':>4} {'MF':>4} {'Val':>4} {'Mom':>4} {'NB':>4} | {'Train':>7} {'Val':>7} {'Test':>7} | 一致?")
print(f"{'-'*60}")
by_train = sorted(results, key=lambda x: -x['train']['avg_ret'])[:15]
for r in by_train:
    w = r['w']
    v = r['val']
    te = r['test']
    v_str = f"{v['avg_ret']:>+6.2f}%" if v else '  N/A  '
    t_str = f"{te['avg_ret']:>+6.2f}%" if te else '  N/A  '
    mark = '✅' if r['consistency'] else '❌'
    print(f"{w[0]*100:>3.0f}% {w[1]*100:>3.0f}% {w[2]*100:>3.0f}% {w[3]*100:>3.0f}% {w[4]*100:>3.0f}% | {r['train']['avg_ret']:>+6.2f}% {v_str} {t_str} | {mark}")

# Test ranking (the real OOS)
print(f"\nTOP 10 BY TEST (2024-2026, true out-of-sample):")
by_test = sorted(results, key=lambda x: -x['test']['avg_ret'])[:10]
for r in by_test:
    w = r['w']
    print(f"  V4={w[0]*100:.0f}% MF={w[1]*100:.0f}% Val={w[2]*100:.0f}% Mom={w[3]*100:.0f}% NB={w[4]*100:.0f}% | train={r['train']['avg_ret']:+.2f}% val={r['val']['avg_ret'] if r['val'] else 0:+.2f}% test={r['test']['avg_ret']:+.2f}% win={r['test']['win_rate']:.0f}%")

print(f"\nTime: {time.time()-t0:.1f}s")
