#!/usr/bin/env python3
"""V4美股 · 5框架暴力扫描（内存优化版）"""
import json, itertools, numpy as np, os, sys
from bisect import bisect_right

CACHE = "/home/admin/.openclaw/workspace/data/cache"
TICKERS = [f.replace('.json','') for f in os.listdir(CACHE) if f != 'spy.json']

spy = json.load(open(f"{CACHE}/spy.json"))
SPY_DATES = spy['dates']

def get_idx(d):
    return bisect_right(SPY_DATES, d) - 1

def prep(ticker, sd_idx, ed_idx):
    """预计算指定日期范围的指标"""
    raw = json.load(open(f"{CACHE}/{ticker}.json"))['data']
    result = {}
    for i in range(60, len(raw)):
        row = raw[i]; d = row['date']
        # 只存我们关心的日期范围
        di = bisect_right(SPY_DATES, d) - 1
        if di < sd_idx - 60 or di > ed_idx + 30: continue
        
        pr = float(row['close'])
        a20 = float(np.mean([float(raw[j]['close']) for j in range(i-19,i+1)]))
        # 52周高
        hp52 = max(float(raw[j]['close']) for j in range(i-251,i+1))
        p52 = pr/hp52*100 if hp52>0 else 100
        # 多周期动量
        m15 = (pr/float(raw[i-15]['close'])-1)*100 if i>=15 else 0
        m20 = (pr/float(raw[i-20]['close'])-1)*100
        m25 = (pr/float(raw[i-25]['close'])-1)*100
        m30 = (pr/float(raw[i-30]['close'])-1)*100
        # RSI
        if i>=14:
            g=sum(max(0,float(raw[j]['close'])-float(raw[j-1]['close'])) for j in range(i-13,i+1))/14
            l=sum(max(0,float(raw[j-1]['close'])-float(raw[j]['close'])) for j in range(i-13,i+1))/14
            rsi = 100-100/(1+g/(l+0.001))
        else: rsi = 50
        result[d] = {'p':pr,'a20':a20,'p52':p52,
                     'm15':m15,'m20':m20,'m25':m25,'m30':m30,'rsi':rsi}
    return result

# 要测试的区间
SD = "2020-01-02"; ED = "2025-12-31"
SI_S = get_idx(SD); SI_E = get_idx(ED)
print(f"区间: {SD}~{ED} ({SI_E-SI_S}天)")

# 预计算所有数据
print("预计算指标...")
all_data = {}
for i, t in enumerate(TICKERS):
    try:
        all_data[t] = prep(t, SI_S, SI_E)
    except Exception as e:
        print(f"  {t}: {str(e)[:40]}", flush=True)
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(TICKERS)}", flush=True)
print(f"  {len(all_data)}只")

# ===== 框架定义 =====
FRAMEWORKS = [
    {
        'name': 'A-纯动量',
        'params': {'md':[10,15,20,25,30],'tn':[3,5,8],'hd':[10,15,20,25,30]},
        'total': 5*3*5
    },
    {
        'name': 'B-比例扣分',
        'params': {'md':[10,15,20,25,30],'tn':[3,5,8],'hd':[10,15,20,25,30],'pp':[0.3,0.5,0.7]},
        'total': 5*3*5*3
    },
    {
        'name': 'C-硬过滤',
        'params': {'md':[10,15,20,25,30],'tn':[3,5,8],'hd':[10,15,20,25,30],'th':[80,85,90,95,100]},
        'total': 5*3*5*5
    },
    {
        'name': 'D-全评分',
        'params': {'md':[10,15,20,25,30],'tn':[3,5,8],'hd':[10,15,20,25,30],'wm':[30,50],'wp':[30,50],'wr':[10,20]},
        'total': 5*3*5*2*2*2
    },
    {
        'name': 'E-双扣分',
        'params': {'md':[10,15,20,25,30],'tn':[3,5,8],'hd':[10,15,20,25,30],'p52':[0.3,0.5,0.7],'rsi':[0,0.2,0.4]},
        'total': 5*3*5*3*3
    },
]

def get_mom(vb, vp, md):
    if md == 15: return vb.get('m15', (vb['p']/vp['p']-1)*100)
    if md == 20: return vb.get('m20', (vb['p']/vp['p']-1)*100)
    if md == 25: return vb.get('m25', (vb['p']/vp['p']-1)*100)
    if md == 30: return vb.get('m30', (vb['p']/vp['p']-1)*100)
    return (vb['p']/vp['p']-1)*100

def run_fw(name, md, tn, hd, **extras):
    si_s = SI_S + max(md, 30)
    rets = []
    for si in range(si_s, SI_E - hd, hd):
        d_pr = SPY_DATES[si - md]; d_by = SPY_DATES[si]; d_sl = SPY_DATES[min(si+hd, SI_E)]
        
        cand = []
        for t, td in all_data.items():
            vb = td.get(d_by); vp = td.get(d_pr)
            if not vb or not vp or vb['p'] < 1: continue
            mom = get_mom(vb, vp, md)
            p52 = vb['p52']; rsi = vb.get('rsi', 50)
            
            if name == 'A-纯动量':
                sc = mom
            elif name == 'B-比例扣分':
                pp = extras.get('pp', 0.5)
                adj = 1 - max(0, (p52-50)/50 * pp)
                sc = mom * adj
            elif name == 'C-硬过滤':
                th = extras.get('th', 95)
                if p52 > th: continue
                sc = mom
            elif name == 'D-全评分':
                wm = extras.get('wm', 40)/100; wp = extras.get('wp', 40)/100; wr = extras.get('wr', 20)/100
                sc = max(0, 50+mom)*wm + max(0, 100-p52)*wp + (100-abs(rsi-50)*2)*wr
            elif name == 'E-双扣分':
                pp = extras.get('p52', 0.5); rp = extras.get('rsi', 0.2)
                p_adj = max(0, (p52-50)/50 * pp)
                r_adj = max(0, (rsi-70)/30 * rp) if rsi > 70 else 0
                sc = mom * (1 - p_adj) - mom * r_adj
            
            cand.append((sc, t))
        
        if len(cand) < tn: continue
        cand.sort(key=lambda x: x[0], reverse=True)
        top = cand[:tn]
        
        rr = []
        for _, t in top:
            vb = all_data[t].get(d_by); vs = all_data[t].get(d_sl)
            if vb and vs and vb['p'] > 1:
                rr.append((vs['p']/vb['p']-1)*100)
        if rr: rets.append(np.mean(rr))
    return rets

# ===== 扫描 =====
all_res = {}
for fw in FRAMEWORKS:
    name = fw['name']
    print(f"\n{'='*55}")
    print(f"{name}: {fw.get('desc','')}")
    print(f"  组合: {fw['total']}")
    print(f"{'='*55}")
    
    keys = list(fw['params'].keys())
    fw_res = []
    
    for idx, vals in enumerate(itertools.product(*fw['params'].values())):
        p = dict(zip(keys, vals))
        rets = run_fw(name, **p)
        if rets:
            total = sum(rets); avg = np.mean(rets)
            wr = 100*sum(1 for r in rets if r>0)/len(rets)
            fw_res.append((total, avg, wr, min(rets), max(rets), p))
        if (idx+1) % 50 == 0:
            sys.stdout.write(f"  {idx+1}/{fw['total']}\r"); sys.stdout.flush()
    
    fw_res.sort(key=lambda x: x[0], reverse=True)
    all_res[name] = fw_res
    
    print(f"\n🏆 {name} TOP 5:")
    print(f"{'总收益':>8s} {'均/期':>7s} {'胜率':>5s} {'最差':>7s} {'最好':>7s}  参数")
    print("-" * 60)
    for r in fw_res[:5]:
        ps = " ".join(f"{k}={v}" for k,v in r[5].items())
        print(f"{r[0]:>+8.1f}% {r[1]:>+7.2f}% {r[2]:>4.1f}% {r[3]:>+7.1f}% {r[4]:>+7.1f}%  {ps}")

# 对比所有框架的TOP1
print(f"\n{'='*55}")
print("🏆 五种框架 TOP1 对比")
print(f"{'='*55}")
print(f"{'框架':>15s} {'总收益':>8s} {'均/期':>7s} {'胜率':>5s} {'最差':>7s}")
print("-" * 50)
for name in ['A-纯动量','B-比例扣分','C-硬过滤','D-全评分','E-双扣分']:
    r = all_res[name][0]
    print(f"{name:>15s} {r[0]:>+8.1f}% {r[1]:>+7.2f}% {r[2]:>4.1f}% {r[3]:>+7.1f}%")

print("\n✅ 完成")
