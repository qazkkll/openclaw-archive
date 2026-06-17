#!/usr/bin/env python3
"""
A2 买入计划回测 — 完全替代A1资金流模型 v1
========================================
参数扫描: Top N=[3,5,10,15,20], 持有期=[5,10,20]天, 持仓上限=[5,10,20]只
输出: 10年每年收益 / 年化 / 最大回撤 / 夏普 / A2 vs A1对比
     + 2026年4-5月真实10w模拟
用法: python scripts/a2_backtest.py [--quick]
"""
import json, os, sys, time, math
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

QUICK = '--quick' in sys.argv
D = r'/home/hermes/.hermes/openclaw-archive/data'
MODEL = os.path.join(D, 'models', 'a1_layer3_xgb_10d.json')

print("="*65)
print("A2 买入计划回测")
print(f"模式: {'快速' if QUICK else '全量'}")
print("="*65)

# 1. 加载数据
t0 = time.time()
with open(f'{D}/a_hist_10y.parquet', 'rb') as f: hist = json.load(f)
with open(f'{D}/moneyflow_data.parquet', 'rb') as f: mf = json.load(f)
import xgboost as xgb
model = xgb.Booster(); model.load_model(MODEL)
feat_cols = model.feature_names
print(f"数据: {len(hist)}只K线 + {len(mf)}只资金流 + {len(feat_cols)}特征模型 ({time.time()-t0:.1f}s)")

# 2. 选股
codes = sorted([c for c in hist if c.startswith(('6','0'))])
if QUICK: codes = codes[:200]
print(f"候选: {len(codes)}只主板")

# 3. 特征计算+评分函数
TECH = ['pct_ma5','pct_ma10','pct_ma20','pct_ma60','pct_ma120',
        'ma20_slope','ma60_slope','ma_align',
        'vol_10d','vol_60d','vol_ratio','atr20_pct',
        'ret_5d','ret_10d','ret_20d','ret_60d','rsi14',
        'vol_ratio_5_20','ret20d_pct']
MF_FEATS = [c for c in feat_cols if c not in TECH]

def mf_idx_map(mf_recs):
    return {r['trade_date']: i for i, r in enumerate(mf_recs)}

def mf_agg(mf_recs, di, lb):
    if di < 0: return {}
    s = max(0, di-lb+1)
    w = mf_recs[s:di+1]
    if lb == 1:
        r = w[0]
        return {
            'net_mf': r.get('net_mf_amount',0) or 0,
            'lg_net': ((r.get('buy_lg_amount',0) or 0)-(r.get('sell_lg_amount',0) or 0)),
            'elg_net': ((r.get('buy_elg_amount',0) or 0)-(r.get('sell_elg_amount',0) or 0)),
            'major_net': ((r.get('buy_lg_amount',0) or 0)+(r.get('buy_elg_amount',0) or 0)
                        -(r.get('sell_lg_amount',0) or 0)-(r.get('sell_elg_amount',0) or 0))}
    n = sum(r.get('net_mf_amount',0) or 0 for r in w)
    lg = sum(((r.get('buy_lg_amount',0) or 0)-(r.get('sell_lg_amount',0) or 0)) for r in w)
    elg = sum(((r.get('buy_elg_amount',0) or 0)-(r.get('sell_elg_amount',0) or 0)) for r in w)
    return {'net_mf': n, 'lg_net': lg, 'elg_net': elg, 'major_net': lg+elg}

def compute_one(code_data, mf_recs):
    """对一只股票计算每日L3评分 + A1评分"""
    dts = code_data.get('dates') or code_data.get('date', [])
    c = code_data.get('c') or code_data.get('close', [])
    h = code_data.get('h') or code_data.get('high', [])
    l = code_data.get('l') or code_data.get('low', [])
    o = code_data.get('o') or code_data.get('open', [])
    v = code_data.get('v') or code_data.get('vol', [])
    n = len(c)
    if n < 600: return []
    
    mfim = mf_idx_map(mf_recs)
    results = []
    
    for i in range(500, n):
        price = c[i]
        date = dts[i]
        if int(date[:4]) < 2017: continue
        
        # 技术指标
        ma5 = sum(c[i-4:i+1])/5
        ma10 = sum(c[i-9:i+1])/10
        ma20 = sum(c[i-19:i+1])/20
        ma60 = sum(c[i-59:i+1])/60
        ma120 = sum(c[i-119:i+1])/120 if i>=119 else ma60
        
        f = {}
        f['pct_ma5'] = (price/ma5-1)*100
        f['pct_ma10'] = (price/ma10-1)*100
        f['pct_ma20'] = (price/ma20-1)*100
        f['pct_ma60'] = (price/ma60-1)*100
        f['pct_ma120'] = (price/ma120-1)*100
        f['ma20_slope'] = (ma20/sum(c[i-25:i-4])*20/ma20-1)*100 if i>=25 else 0  # 简化
        f['ma60_slope'] = (ma60/sum(c[i-65:i-4])*60/ma60-1)*100 if i>=65 else 0
        f['ma_align'] = (ma5>ma10)+(ma10>ma20)+(ma20>ma60)+(price>ma5)+(price>ma10)+(price>ma60)
        
        v10 = [abs(c[j]/c[j-1]-1)*100 for j in range(max(1,i-9), i+1)]
        v60 = [abs(c[j]/c[j-1]-1)*100 for j in range(max(1,i-59), i+1)]
        f['vol_10d'] = sum(v10)/len(v10)
        f['vol_60d'] = sum(v60)/len(v60)
        f['vol_ratio'] = f['vol_10d']/f['vol_60d'] if f['vol_60d']>0 else 1
        
        trs = [max(h[j]-l[j],abs(h[j]-c[j-1]),abs(l[j]-c[j-1])) for j in range(max(1,i-19),i+1)]
        f['atr20_pct'] = sum(trs)/len(trs)/price*100
        
        f['ret_5d'] = (price/c[i-5]-1)*100
        f['ret_10d'] = (price/c[i-10]-1)*100
        f['ret_20d'] = (price/c[i-20]-1)*100
        f['ret_60d'] = (price/c[i-60]-1)*100
        
        chg = [c[j]-c[j-1] for j in range(max(1,i-13), i+1)]
        rg = sum(x for x in chg if x>0); rl = sum(-x for x in chg if x<0)
        f['rsi14'] = 100-100/(1+rg/rl/14) if rl>0 else 100
        
        v5 = sum(v[i-4:i+1])/5; v20 = sum(v[i-19:i+1])/20
        f['vol_ratio_5_20'] = v5/v20 if v20>0 else 1
        
        if i >= 40:
            pr = [(c[j]/c[j-20]-1)*100 for j in range(20, i+1)]
            f['ret20d_pct'] = sum(1 for r in pr if r<(price/c[i-20]-1)*100)/len(pr)*100
        else: f['ret20d_pct'] = 50
        
        # 资金流
        mfi = mfim.get(date, -1)
        if mfi >= 0:
            a1 = mf_agg(mf_recs, mfi, 1)
            f['net_mf'] = a1['net_mf']; f['lg_net'] = a1['lg_net']
            f['elg_net'] = a1['elg_net']; f['md_net'] = 0
            f['major_net'] = a1['major_net']
            
            r = mf_recs[mfi]
            tot = sum(abs(r.get(k,0) or 0) for k in ['buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
                                                      'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount'])
            buy_le = (r.get('buy_lg_amount',0) or 0)+(r.get('buy_elg_amount',0) or 0)
            f['lg_pct'] = buy_le/tot*100 if tot>0 else 50
            f['elg_pct'] = (r.get('buy_elg_amount',0) or 0)/tot*100 if tot>0 else 25
            f['major_ratio'] = a1['major_net']/tot*100 if tot>0 else 0
            
            for lb, sf in [(5,'_5d'),(10,'_10d'),(20,'_20d'),(60,'_60d')]:
                agg = mf_agg(mf_recs, mfi, lb)
                f['net_mf'+sf] = agg['net_mf']
                f['major_net'+sf] = agg['major_net']
                f['lg_net'+sf] = agg['lg_net']
        else:
            for col in MF_FEATS: f[col] = 0
        
        # L3 预测
        vec = [f.get(col,0) for col in feat_cols]
        l3 = float(model.predict(xgb.DMatrix([vec], feature_names=feat_cols))[0])
        fwd = (c[i+10]/c[i]-1)*100 if i+10 < n else 0
        
        # A1评分
        a1_s = 0
        if mfi >= 0:
            r = mf_recs[mfi]
            nmf = r.get('net_mf_amount',0) or 0
            bl = (r.get('buy_lg_amount',0) or 0)+(r.get('buy_elg_amount',0) or 0)
            sl = (r.get('sell_lg_amount',0) or 0)+(r.get('sell_elg_amount',0) or 0)
            tot2 = sum(abs(r.get(k,0) or 0) for k in ['buy_sm_amount','sell_sm_amount','buy_md_amount','sell_md_amount',
                                                      'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount'])
            a1_s = nmf/10000*0.4+max((bl-sl)/tot2,0)*0.6 if tot2>0 else 0
        
        results.append({'date': date, 'l3': l3, 'a1': a1_s, 'fwd': fwd})
    
    return results

# 4. 每日评分聚合
print(f"\n阶段1: 评分计算 ({len(codes)}只)")
t1 = time.time()
l3_daily, a1_daily = defaultdict(list), defaultdict(list)

for idx, code in enumerate(codes):
    cd = hist.get(code)
    if not cd: continue
    mfc = str(code).replace('.SH','').replace('.SZ','').replace('.BJ','').strip()
    if mfc.startswith(('0','3')): mfc += '.SZ'
    else: mfc += '.SH'
    mf_r = mf.get(mfc, [])
    if not mf_r: continue
    
    try:
        recs = compute_one(cd, mf_r)
    except: continue
    
    for s in recs:
        d = s['date']
        l3_daily[d].append((code, s['l3'], s['fwd']))
        if s['a1'] > 0: a1_daily[d].append((code, s['a1'], s['fwd']))
    
    if (idx+1) % 100 == 0:
        print(f"  {idx+1}/{len(codes)} ({time.time()-t1:.0f}s)")

# 排序
for d in l3_daily: l3_daily[d].sort(key=lambda x: -x[1])
for d in a1_daily: a1_daily[d].sort(key=lambda x: -x[1])
print(f"评分完成: L3 {len(l3_daily)}天/{sum(len(v) for v in l3_daily.values())}条, A1 {len(a1_daily)}天 ({time.time()-t1:.0f}s)")

# 5. 回测引擎
def run_backtest(daily, name, tn=5, hd=10, mp=10):
    """tn=TopN, hd=持有期, mp=持仓上限"""
    dates = sorted(daily.keys())
    nav, peak = 1.0, 1.0
    dd, daily_rets = 0.0, []
    pos = {}  # code -> buy_idx
    
    for di, date in enumerate(dates):
        cand = daily.get(date, [])
        
        # 卖出过期的
        for code in list(pos.keys()):
            if di - pos[code] >= hd:
                del pos[code]
        
        # 买入
        slots = mp - len(pos)
        if slots > 0 and cand:
            for item in cand:
                if len(pos) >= mp: break
                if item[0] in pos: continue
                pos[item[0]] = di
                if len(pos) >= mp: break
        
        # 当日收益（用fwd_10d/持有期近似）
        if pos:
            cm = {c[0]: c[2] for c in cand}
            ret = sum(cm.get(c,0) for c in pos)/len(pos)/hd/100
        else: ret = 0
        
        nav *= (1+ret); peak = max(peak, nav)
        dd = max(dd, (peak-nav)/peak)
        daily_rets.append(ret)
    
    r = np.array(daily_rets)
    total = nav-1; yrs = len(dates)/250
    cagr = nav**(1/yrs)-1 if yrs>0 else 0
    sharpe = (r.mean()/r.std()*np.sqrt(250)) if r.std()>0 else 0
    
    m = {'name': name, 'tn':tn, 'hd':hd, 'mp':mp,
         'cagr':round(cagr*100,2), 'dd':round(dd*100,2),
         'sharpe':round(sharpe,3), 'vol':round(r.std()*np.sqrt(250)*100,2),
         'total':round(total*100,2), 'days':len(dates)}
    return m, daily_rets

# 6. 参数扫描
print(f"\n阶段2: 参数扫描")
t2 = time.time()
results = []

for tn in ([5,10] if QUICK else [3,5,10,15,20]):
    for hd in ([10] if QUICK else [5,10,20]):
        for mp in ([10] if QUICK else [5,10,20]):
            m, _ = run_backtest(l3_daily, f'A2_T{tn}H{hd}M{mp}', tn, hd, mp)
            results.append(m)

print(f"扫描完成: {len(results)}组 ({time.time()-t2:.0f}s)")

# 7. 结果展示
results.sort(key=lambda r: -(r['sharpe']))
print(f"\n{'='*65}")
print("A2策略排名（按夏普比率）:")
print(f"{'排名':>4} {'策略':>14} {'年化%':>8} {'回撤%':>8} {'夏普':>8} {'总收益%':>10}")
print("-"*56)
for i, r in enumerate(results[:10]):
    print(f"{i+1:>4} {r['name']:>14} {r['cagr']:>7.2f}% {r['dd']:>7.2f}% {r['sharpe']:>7.3f} {r['total']:>9.2f}%")

best = results[0]
print(f"\n{'='*65}")
print(f"🥇 最优: {best['name']}")
print(f"{'='*65}")
for k in ['cagr','dd','sharpe','vol','total','days']:
    print(f"  {k}: {best[k]}")

# 8. A1对比
print(f"\n{'='*65}")
print("A1基线 对比")
m1, _ = run_backtest(a1_daily, 'A1_T10H5M10', 10, 5, 10)
for k in ['cagr','dd','sharpe','vol','total']:
    d = best[k] - m1[k]
    s = '+' if d >= 0 else ''
    print(f"  A2 {k}: {best[k]} vs A1 {k}: {m1[k]} ({s}{d:.2f})")

# 9. 10w模拟 (2026年4-5月)
print(f"\n{'='*65}")
print("10w模拟: 2026年4月-5月")
sim_tn = best['tn']; sim_hd = best['hd']; sim_mp = best['mp']
apr_may_dates = [d for d in sorted(l3_daily.keys()) if d.startswith('2026') and int(d[5:7]) in (4,5)]
nav = 100000; peak = 100000; dd = 0
trades = []

for di, date in enumerate(apr_may_dates):
    cand = l3_daily.get(date, [])
    # 简化模拟
    if cand:
        buys = cand[:sim_tn]
        ret = sum(c[2] for c in buys) / len(buys) / sim_hd / 100
        nav *= (1+ret)
        peak = max(peak, nav)
        dd = max(dd, (peak-nav)/peak)

print(f"  A2 {best['name']}: 期初¥100,000 → 期末¥{nav:,.0f} (+{(nav/100000-1)*100:.2f}%)")
print(f"  最大回撤: {dd*100:.2f}%")
print(f"  交易日: {len(apr_may_dates)}天")

# A1对比
m1_sim, _ = run_backtest(a1_daily, '', sim_tn, sim_hd, sim_mp)  # 用A1数据, A2参数不公平
# 用A1自己的参数
apr_may_a1 = [d for d in sorted(a1_daily.keys()) if d.startswith('2026') and int(d[5:7]) in (4,5)]
nav_a1 = 100000
for di, date in enumerate(apr_may_a1):
    cand = a1_daily.get(date, [])
    if cand:
        ret = sum(c[2] for c in cand[:10])/10/5/100
        nav_a1 *= (1+ret)

print(f"  A1基线: 期初¥100,000 → 期末¥{nav_a1:,.0f} (+{(nav_a1/100000-1)*100:.2f}%)")
