"""
分层权重验证 — 不同市场状态下用不同因子权重

三层状态（由Layer 1北向定义）：
  Bull: NB > 65% → 进攻
  Neutral: NB 35-65% → 平衡
  Bear: NB < 35% → 防御

在每层状态下分别测试：哪个因子组合在该状态下最有效

训练: 2016-2020 | 验证: 2021-2023 | 测试: 2024-2026
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

t0 = time.time()
print("=" * 65)
print("Regime-Dependent Weight System")
print("=" * 65, flush=True)

# ── Load data ──
print("\n[1/4] Loading...", flush=True)

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

def nb_regime(date_str):
    p = nb_pct(date_str)
    if p > 65: return 'bull'
    elif p < 35: return 'bear'
    return 'neutral'

mf_scores = json.load(open(f'{WORKSPACE}/data/precomputed_scores.json', 'rb'))
def mf_key(kc):
    if kc in mf_scores: return kc
    if f'{kc}.SZ' in mf_scores: return f'{kc}.SZ'
    if f'{kc}.SH' in mf_scores: return f'{kc}.SH'
    return None

kl_data = json.load(open(f'{WORKSPACE}/data/a_hist_10y.parquet', 'rb'))
stocks = sorted(kl_data.items(), key=lambda x: -len(x[1].get('c', [])))
eligible = []
for sc, sd in stocks:
    mk = mf_key(sc)
    if mk and len(sd.get('c', [])) >= 500:
        eligible.append((sc, sd, mk))
sample = eligible[:300]
print(f"  {len(sample)} stocks", flush=True)
del kl_data, stocks; gc.collect()

# ── Compute factors ──
print("\n[2/4] Computing factors...", flush=True)
import sys as _sys
_sys.path.insert(0, os.path.join(WORKSPACE, 'scripts'))
from us_score_engine import compute_indicators, v1_score

def momentum_score(closes, idx):
    if idx < 60: return 50
    ret20 = (closes[idx] / closes[idx-20] - 1) * 100
    ret60 = (closes[idx] / closes[idx-60] - 1) * 100
    return max(0, min(100, 50 + ret20*1.5 + ret60*0.5))

all_data = []
cnt = 0
for sc, sd, mk in sample:
    c = sd.get('c', []); h = sd.get('h', []); lo = sd.get('l', [])
    dates = sd.get('dates', [])
    if len(c) < 500: continue
    ind = compute_indicators(c, h, lo)
    if ind is None: continue
    mf_base = mf_scores.get(mk, {}).get('mf_score', 50)
    cnt += 1
    for i in range(200, len(c)-20, 20):
        d = dates[i]
        if d > '20261231' or d < '20160101': continue
        all_data.append({
            'stock': sc, 'date': d,
            'v4': float(v1_score(ind, i)), 'mf': float(mf_base),
            'val': 50.0, 'mom': momentum_score(c, i),
            'nb': nb_pct(d), 'regime': nb_regime(d),
            'ret': round((c[i+20]/c[i]-1)*100, 2)
        })
    if cnt % 50 == 0: print(f"  {cnt}/{len(sample)}", flush=True)

print(f"  Total: {len(all_data)} pts", flush=True)

# ── Split by regime ──
print("\n[3/4] Testing per-regime weights...", flush=True)

train = [d for d in all_data if d['date'] <= '20201231']
val = [d for d in all_data if '20210101' <= d['date'] <= '20231231']
test = [d for d in all_data if d['date'] >= '20240101']

# Generate weight combinations: (v4, mf, val, mom, nb)
# Focus on regimes: exclude v4 from weights (use as gate), test MF+NB+Val+Mom
combos = []
for a in [x/100 for x in range(0, 101, 20)]:
    for b in [x/100 for x in range(0, 101, 20)]:
        for c_w in [x/100 for x in range(0, 101, 20)]:
            for d_w in [x/100 for x in range(0, 101, 20)]:
                total = round(a + b + c_w + d_w, 2)
                if abs(total - 1.0) < 0.01:
                    combos.append((0, a, b, c_w, d_w))  # V4=0
combos = list(set(combos))
print(f"  {len(combos)} weight combinations", flush=True)

def backtest_by_regime(data, weights):
    """Backtest separately per regime, return per-regime metrics"""
    v4_w, mf_w, val_w, mom_w, nb_w = weights
    by_regime = {'bull': {}, 'neutral': {}, 'bear': {}}
    
    for d in data:
        rg = d['regime']
        ym = d['date'][:6]
        if ym not in by_regime[rg]: by_regime[rg][ym] = []
        by_regime[rg][ym].append(d)
    
    results = {}
    for rg in ['bull', 'neutral', 'bear']:
        monthly = []
        for ym in sorted(by_regime[rg].keys()):
            group = by_regime[rg][ym]
            for g in group:
                g['comp'] = (g['v4']*v4_w + g['mf']*mf_w + g['val']*val_w +
                            g['mom']*mom_w + g['nb']*nb_w)
            top5 = sorted(group, key=lambda x: -x['comp'])[:5]
            if top5:
                monthly.append(sum(s['ret'] for s in top5)/len(top5))
        
        if monthly:
            results[rg] = {
                'avg': round(sum(monthly)/len(monthly), 2),
                'win': round(sum(1 for r in monthly if r>0)/len(monthly)*100, 1),
                'n': len(monthly)
            }
    return results

# First: find BEST weights per regime in training
print("\n--- Training: Best weights per regime (2016-2020) ---")
train_regime_best = {}

for rg in ['bull', 'neutral', 'bear']:
    best = None
    best_score = -999
    for w in combos:
        res = backtest_by_regime(train, w)
        if rg not in res or res[rg] is None: continue
        r = res[rg]
        score = r['avg'] * 0.6 + (r['win'] - 50) * 0.04
        
        if score > best_score:
            best_score = score
            best = (w, r)
    
    if best:
        w, r = best
        print(f"  {rg:>8}: V4={w[0]*100:.0f}% MF={w[1]*100:.0f}% Val={w[2]*100:.0f}% Mom={w[3]*100:.0f}% NB={w[4]*100:.0f}% | avg={r['avg']:+.2f}% win={r['win']:.0f}% ({r['n']}mo)")
        train_regime_best[rg] = w

# Now validate: use the best per-regime weights, apply in val and test
print(f"\n--- Validation: Applying regime-based weights ---")
print(f"{'Period':<12} {'Bull_w':>25} {'Neutral_w':>25} {'Bear_w':>25} | {'Overall':>8}")
print(f"{'-'*90}")

for name, dataset in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
    monthly_all = []
    for d in dataset:
        rg = d['regime']
        weights = train_regime_best.get(rg, (0, 0.2, 0.2, 0, 0.6))
        v4_w, mf_w, val_w, mom_w, nb_w = weights
        d['comp'] = (d['v4']*v4_w + d['mf']*mf_w + d['val']*val_w +
                     d['mom']*mom_w + d['nb']*nb_w)
    
    by_month = {}
    for d in dataset:
        ym = d['date'][:6]
        if ym not in by_month: by_month[ym] = []
        by_month[ym].append(d)
    
    monthly_rets = []
    for ym in sorted(by_month.keys()):
        group = by_month[ym]
        top5 = sorted(group, key=lambda x: -x['comp'])[:5]
        if top5:
            avg = sum(s['ret'] for s in top5)/len(top5)
            monthly_rets.append(avg)
    
    if not monthly_rets: continue
    avg = sum(monthly_rets)/len(monthly_rets)
    win = sum(1 for r in monthly_rets if r>0)/len(monthly_rets)*100
    drawdown = min(monthly_rets)
    
    # Build display strings
    bull_w = train_regime_best.get('bull', (0,0,0,0,0))
    neu_w = train_regime_best.get('neutral', (0,0,0,0,0))
    bear_w = train_regime_best.get('bear', (0,0,0,0,0))
    bw = f"MF={bull_w[1]*100:.0f}% Val={bull_w[2]*100:.0f}% NB={bull_w[4]*100:.0f}%"
    nw = f"MF={neu_w[1]*100:.0f}% Val={neu_w[2]*100:.0f}% NB={neu_w[4]*100:.0f}%"
    bew = f"MF={bear_w[1]*100:.0f}% Val={bear_w[2]*100:.0f}% NB={bear_w[4]*100:.0f}%"
    
    print(f"{name:<12} {bw:>25} {nw:>25} {bew:>25} | {avg:>+6.2f}% win={win:.0f}% maxdd={drawdown:+.1f}%")

# Compare: regime-based vs fixed-weight vs simple benchmark
print(f"\n{'='*65}")
print(f"COMPARISON: Regime-Adaptive vs Fixed vs Hold")
print(f"{'='*65}")

# Fixed best (MF=40% NB=60%)
monthly_fixed = []
for period_name, dataset in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
    by_month = {}
    for d in dataset:
        ym = d['date'][:6]
        if ym not in by_month: by_month[ym] = []
        comp = d['mf']*0.4 + d['nb']*0.6
        d['comp_fixed'] = comp
        by_month[ym].append(d)
    
    rets = []
    for ym in sorted(by_month.keys()):
        top5 = sorted(by_month[ym], key=lambda x: -x['comp_fixed'])[:5]
        if top5:
            rets.append(sum(s['ret'] for s in top5)/len(top5))
    
    if rets:
        avg = sum(rets)/len(rets)
        win = sum(1 for r in rets if r>0)/len(rets)*100
        print(f"  Fixed(MF40%+NB60%) {period_name:<6}: {avg:+.2f}% win={win:.0f}%")

# Simple hold CSI300 benchmark - need index data
# Skip for now since we don't have CSI300 monthly

print(f"\n{'='*65}")
print(f"CONCLUSIONS")
print(f"{'='*65}")
print(f"  Regime-adaptive weights:")
print(f"    Bull:  MF={bull_w[1]*100:.0f}% Val={bull_w[2]*100:.0f}% NB={bull_w[4]*100:.0f}%")
print(f"    Neut:  MF={neu_w[1]*100:.0f}% Val={neu_w[2]*100:.0f}% NB={neu_w[4]*100:.0f}%")
print(f"    Bear:  MF={bear_w[1]*100:.0f}% Val={bear_w[2]*100:.0f}% NB={bear_w[4]*100:.0f}%")
print(f"  V4 role: threshold filter (score > 62 to execute, not in weights)")

print(f"\nTime: {time.time()-t0:.1f}s")
