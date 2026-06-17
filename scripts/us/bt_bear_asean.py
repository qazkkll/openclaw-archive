#!/usr/bin/env python3
"""A股 V4熊市逆向防守 · 全量回测 (优化版)"""
import json, warnings, numpy as np, time, sys
warnings.filterwarnings('ignore')

DATA = "/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json"
print("[1/5] 加载数据...")
sys.stdout.flush()
hist = json.load(open(DATA))

codes = [c for c in hist if len(hist[c].get('close',[])) > 500]
print(f"  合资格: {len(codes)}只")
sys.stdout.flush()

adates = sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2015-01-01'<=d<='2025-12-31'))
print(f"  交易日: {len(adates)}天")
sys.stdout.flush()

cdates = {c:{dt:i for i,dt in enumerate(hist[c]['dates'])} for c in codes}
hist_has_volume = 'volume' in hist[codes[0]] if codes else False

def gi(code, dt):
    cm = cdates.get(code)
    if cm and dt in cm: return cm[dt]
    d = hist.get(code)
    if d and d.get('dates'):
        for x in reversed(d['dates']):
            if x <= dt and x in cm: return cm[x]
    return -1

def ema(arr, p):
    if len(arr) < p: return []
    k = 2/(p+1); r = [arr[0]]
    for v in arr[1:]: r.append(v*k+r[-1]*(1-k))
    return r

print("[2/5] 预计算指标...")
sys.stdout.flush()
inds = {}
start = time.time()
for idx, code in enumerate(codes):
    try:
        d = hist[code]
        c = d['close']; n = len(c)
        e12 = ema(c,12); e26 = ema(c,26)
        ml = [e12[j]-e26[j] for j in range(min(len(e12),len(e26)))]
        sg = ema(ml,9) if len(ml)>=9 else []
        mh = [ml[j]-sg[j] for j in range(min(len(ml),len(sg)))] if sg else []
        
        # Volume data (some stocks may not have it)
        vol = d.get('volume', None)
        
        result = {}
        for i in range(60, n):
            dt = d['dates'][i]
            pr = c[i]
            
            hp52 = max(c[max(0,i-251):i+1]); lp52 = min(c[max(0,i-251):i+1])
            p52 = (pr-lp52)/(hp52-lp52)*100 if hp52>lp52 else 50
            
            ma20 = sum(c[max(0,i-19):i+1])/min(20,i+1)
            ma60 = sum(c[max(0,i-59):i+1])/min(60,i+1)
            ma5 = sum(c[max(0,i-4):i+1])/min(5,i+1)
            
            hn = mh[i-1] if i-1 < len(mh) else 0
            hp = mh[i-2] if i-2 < len(mh) else 0
            macd_cross = hn > 0 and hp <= 0
            macd_bull = hn > hp and hn > 0
            ms = 15 if macd_cross else (9 if macd_bull else (5 if hn>0 else -3))
            
            gn = sum(max(0,c[j]-c[j-1]) for j in range(i-14,i+1)) if i>=14 else 0
            ls = sum(max(0,c[j-1]-c[j]) for j in range(i-14,i+1)) if i>=14 else 0
            rsi = 100 if ls==0 else (100-100/(1+(gn/14)/(ls/14)) if ls>0 else 50)
            
            rng = max(c[max(0,i-19):i+1])-min(c[max(0,i-19):i+1])
            ae = rng/pr*100 if pr>0 else 0
            a = 20 if ae>=2.0 else 14 if ae>=1.2 else 8 if ae>=0.8 else 4 if ae>=0.5 else 2
            
            ma = 0
            if pr < ma20 and pr > ma20*0.95: ma += 8
            elif pr < ma60 and pr > ma60*0.93: ma += 6
            elif pr < ma20*0.95 and pr < ma60*0.95: ma += 4
            if ma5 > ma20: ma += 7
            
            vs = 0
            if vol:
                v5 = sum(vol[max(0,i-4):i+1])/min(5,i+1)
                v20 = sum(vol[max(0,i-19):i+1])/min(20,i+1)
                vr = v5/v20 if v20>0 else 1
                vs = 15 if vr>1.5 else 10 if vr>1.2 else 5 if vr>0.8 else 0
            
            result[dt] = {'ms':ms,'a':a,'ma':ma,'rsi':rsi,'p52':p52,'vs':vs,'gate':hn>0}
        inds[code] = result
    except: pass
    
    if (idx+1) % 200 == 0:
        elapsed = time.time()-start
        print(f"  [{idx+1}/{len(codes)}] {elapsed:.0f}s")
        sys.stdout.flush()

print(f"  完成: {len(inds)}只 ({time.time()-start:.0f}s)")
sys.stdout.flush()

# 回测函数
def bear_score(ind, w):
    if not ind['gate']: return 0
    p52 = ind['p52']
    ps = 25 if p52<15 else 20 if p52<30 else 14 if p52<45 else 8 if p52<60 else 3 if p52<75 else 0
    rsi = ind['rsi']
    rs = 20 if rsi<30 else 15 if rsi<40 else 10 if rsi<50 else 5 if rsi<60 else 0
    ms = ind['ms']; ma = ind['ma']; vs = ind['vs']
    return round(ms*(w[0]/20) + ps*(w[1]/25) + rs*(w[2]/20) + ma*(w[3]/20) + vs*(w[4]/15))

def run_one(params, year):
    w = (params['w_macd'], params['w_p52'], params['w_rsi'], params['w_ma'], params['w_vol'])
    tn = params['tn']; hd = params['hd']
    sd = f"{year}-01-02"; ed = f"{year}-12-31"
    yd = [d for d in adates if sd<=d<=ed]
    if len(yd)<60: return 0
    rets = []
    for si in range(60, len(yd)-hd, hd):
        db = yd[si]; ds = yd[min(si+hd, len(yd)-1)]
        cand = [(bear_score(inds[c].get(db,{}), w), c) for c in inds if db in inds[c]]
        cand = [(s,c) for s,c in cand if s>0]
        if len(cand)<3: continue
        cand.sort(key=lambda x:-x[0])
        prs = []
        for s,c in cand[:tn]:
            di = gi(c, ds)
            if di>=0:
                sp = hist[c]['close'][di]
                ci = gi(c, db)
                bp = hist[c]['close'][ci] if ci>=0 else 0
                if sp>0 and bp>0: prs.append((sp/bp-1)*100)
        if prs: rets.append(np.mean(prs))
    return sum(rets) if rets else 0

print("[3/5] 权重参数扫描...")
sys.stdout.flush()
BEAR_YRS = [2015, 2018, 2022, 2024]
NORM_YRS = [2016, 2017, 2019, 2020, 2021, 2023]

combos = []
for wm in [15,20,25]:
    for wp in [20,25,30]:
        for wr in [15,20,25]:
            for wma in [15,20,25]:
                for wv in [10,15,20]:
                    if abs(wm+wp+wr+wma+wv-100)<5:
                        combos.append((wm,wp,wr,wma,wv))

print(f"  参数组合: {len(combos)}个")
start = time.time()
results = []
for i, (wm,wp,wr,wma,wv) in enumerate(combos):
    p = {'w_macd':wm,'w_p52':wp,'w_rsi':wr,'w_ma':wma,'w_vol':wv,'tn':5,'hd':10}
    br = sum(run_one(p, y) for y in BEAR_YRS)
    nr = sum(run_one(p, y) for y in NORM_YRS)
    results.append((br, nr, wm, wp, wr, wma, wv))
    if (i+1)%100==0:
        print(f"  [{i+1}/{len(combos)}] {time.time()-start:.0f}s")
        sys.stdout.flush()

results.sort(key=lambda x:-x[0])
print(f"\n≡ 熊市收益TOP 8 (权重组合)")
print(f"{'熊市':>7s}  {'牛市':>7s}  {'MACD':>5s}  {'52W':>5s}  {'RSI':>5s}  {'均线':>5s}  {'量能':>5s}")
for br,nr,wm,wp,wr,wma,wv in results[:8]:
    print(f"{br:>+7.1f}%  {nr:>+7.1f}%  {wm:>4d}  {wp:>4d}  {wr:>4d}  {wma:>4d}  {wv:>4d}")

print(f"\n[4/5] 持仓+调仓参数扫描...")
sys.stdout.flush()
top_weights = [(r[2],r[3],r[4],r[5],r[6]) for r in results[:3]]
full_results = []
for wm,wp,wr,wma,wv in top_weights:
    for tn in [3,5,8]:
        for hd in [5,10,15,20]:
            p = {'w_macd':wm,'w_p52':wp,'w_rsi':wr,'w_ma':wma,'w_vol':wv,'tn':tn,'hd':hd}
            yearly = {}
            for y in BEAR_YRS+NORM_YRS:
                yearly[y] = run_one(p, y)
            total = sum(yearly.values())
            full_results.append((total, p, yearly))

full_results.sort(key=lambda x:-x[0])
print(f"{'累计':>8s}  {'权重':>20s}  {'持股':>4s}  {'调仓':>4s}  {'2015':>7s}  {'2018':>7s}  {'2022':>7s}  {'2024':>7s}")
print("-"*75)
for i,(tot,p,yr) in enumerate(full_results[:12]):
    print(f"{tot:>+8.1f}%  {p['w_macd']:>2d}/{p['w_p52']:>2d}/{p['w_rsi']:>2d}/{p['w_ma']:>2d}/{p['w_vol']:>2d}  {p['tn']:>4d}  {p['hd']:>4d}  {yr.get(2015,0):>+7.1f}% {yr.get(2018,0):>+7.1f}% {yr.get(2022,0):>+7.1f}% {yr.get(2024,0):>+7.1f}%")

print(f"\n[5/5] 对比分析...")
sys.stdout.flush()
csi = {2015:5.58,2016:-11.28,2017:21.78,2018:-25.31,2019:36.07,2020:27.21,2021:-5.20,2022:-21.63,2023:-11.38,2024:14.68}
v4 = {2016:3.47,2017:12.79,2018:0.38,2019:17.37,2020:25.83,2021:74.49,2022:-4.60,2023:22.77,2024:3.13}

best = full_results[0]
bp = best[1]; byr = best[2]
print(f"\n最佳方案 vs V4直选 vs 沪深300:")
print(f"{'年':>4s}  {'熊市逆向':>10s}  {'V4直选':>10s}  {'沪深300':>10s}")
for y in sorted(byr.keys()):
    bv = byr.get(y,0); vv = v4.get(y,0); cv = csi.get(y,0)
    beat = "✅" if bv>vv else "🟡" if bv>cv else "❌"
    print(f"{y:>4d}  {bv:>+10.1f}%  {vv:>+10.1f}%  {cv:>+10.1f}%  {beat}")

print(f"\n≡ 推荐方案TOP 3")
for i,(tot,p,yr) in enumerate(full_results[:3]):
    print(f"\n方案{i+1}:")
    print(f"  权重: MACD{p['w_macd']} 52W{p['w_p52']} RSI{p['w_rsi']} MA{p['w_ma']} VOL{p['w_vol']}")
    print(f"  持仓{p['tn']}只 调仓{p['hd']}天")
    print(f"  熊市: 2015{yr.get(2015,0):+.1f}% 2018{yr.get(2018,0):+.1f}% 2022{yr.get(2022,0):+.1f}%")

# 保存审计
import datetime
audit = {
    'timestamp': str(datetime.datetime.now()),
    'test': 'A股V4熊市逆向全量回测',
    'total_stocks': len(inds),
    'weights_tested': len(combos),
    'best_params': bp,
    'best_yearly': byr
}
with open('/home/admin/.openclaw/workspace/data/bt_bear_audit.json','w') as f:
    json.dump(audit, f, indent=2)
print(f"\n✅ 审计数据已保存")
