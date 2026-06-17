#!/usr/bin/env python3
"""
V5-S双模型全量对比 v8 — 修复净值计算
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ws = "/home/hermes/.hermes/openclaw-archive"
sys.path.insert(0, os.path.join(ws, "scripts"))

# ─── 旧模型 ───
def old_ind(c, h, l):
    n=len(c)
    def sm(a,p): return [None]*(p-1)+[sum(a[i-p+1:i+1])/p for i in range(p-1,len(a))]
    def em(a,p): k=2/(p+1); r=[a[0]]; [r.append(v*k+r[-1]*(1-k)) for v in a[1:]]; return r
    m5=sm(c,5);m20=sm(c,20);m60=sm(c,60);m120=sm(c,120)
    e12=em(c,12);e26=em(c,26);macd=[e12[i]-e26[i] for i in range(n)]
    sig=sm(macd,9)
    hst=[macd[i]-(sig[i] if i<len(sig) and sig[i] is not None else 0) for i in range(n)]
    gl=[max(c[i]-c[i-1],0) for i in range(1,n)]
    ls=[max(c[i-1]-c[i],0) for i in range(1,n)]
    rsi=[None]*14
    if len(gl)>=14:
        ag=sum(gl[:14])/14;al=sum(ls[:14])/14
        for i in range(14,n):
            rsi.append(100-100/(1+ag/al) if al>0 else 100)
            if i<len(gl): ag=(ag*13+gl[i])/14; al=(al*13+ls[i])/14
    p52=[None]*252
    for i in range(252,n):
        lo=min(c[i-251:i+1]);hi=max(c[i-251:i+1])
        p52.append((c[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return {'close':c,'ma5':m5,'ma20':m20,'ma60':m60,'ma120':m120,
            'macd':macd,'macd_signal':sig,'macd_hist':hst,'rsi':rsi,'p52':p52}

def old_sc(ind, di):
    def sf(a,i):
        if not a: return 0
        idx=i if i>=0 else len(a)+i
        return a[idx] if 0<=idx<len(a) and a[idx] is not None else 0
    c=ind['close'];p=sf(c,di)
    if p<=0: return 0
    m5=sf(ind['ma5'],di);m20=sf(ind['ma20'],di);m60=sf(ind['ma60'],di);m120=sf(ind['ma120'],di)
    tr=(10 if m5>m20 else 0)+(10 if m20>m60 else 0)+(10 if m60>m120 else 0)+(5 if p>m20 else 0)+(5 if p>m60 else 0)+(5 if p>m120 else 0)+(5 if m5>m20 and m20>m60 else 0)
    mo=15
    p5=sf(c,di-5);p20x=sf(c,di-20);p60x=sf(c,di-60)
    if p5>0 and p>p5: mo+=5
    if p20x>0 and p>p20x: mo+=5
    if p60x>0 and p>p60x: mo+=5
    p30=sf(c,di-30);m30=(p-p30)/p30*100 if p30>0 else 0; mo+=m30/10
    if m30>50: mo=max(mo-(m30-50)/5,0)
    mh=sf(ind['macd_hist'],di);mhp=sf(ind['macd_hist'],di-1);ms=8 if sf(ind['macd'],di)>sf(ind['macd_signal'],di) else 0
    if mh>0 and mhp<=0: ms+=12
    elif mh>0: ms+=5
    if mh>mhp: ms+=5
    rsi=sf(ind['rsi'],di)
    rs=5+(5 if 50<=rsi<=70 else 3 if rsi>70 else -5 if rsi<30 else 0)
    p52=sf(ind['p52'],di)
    ps=10 if 70<=p52<=100 else 7 if 50<=p52<70 else 4 if 30<=p52<50 else 0
    return max(tr+mo+ms+rs+ps,0)

from us_score_engine import v5s_calc as nc, v5s_score as ns

print("加载...", flush=True)
all_d=json.load(open(f"{ws}/data/us_hist_clean.parquet","r"))
syms=list(all_d.keys())
print(f"总池: {len(syms)}只", flush=True)

H=20; SL=-0.18; T5=5

def bt(label, key):
    print(f"\n{'#'*55}\n# {label}\n{'#'*55}")
    t0=time.time()
    pi=old_ind if key=='old' else nc
    sf=old_sc if key=='old' else ns
    
    # 预计算
    cache={}
    for i,sy in enumerate(syms):
        if i%500==0: print(f"  {i}/{len(syms)}", flush=True)
        d=all_d[sy];c=d.get("c",[]);h=d.get("h",[]);l=d.get("l",[])
        if len(c)<520: continue
        ind=pi(c,h,l)
        if ind is not None: cache[sy]=(ind,c)
    print(f"  有效池: {len(cache)}只 ({time.time()-t0:.0f}s)", flush=True)
    if not cache: return None
    
    max_n=max(len(c) for _,c in cache.values())
    all_t=[]
    
    for ti,day in enumerate(range(252, max_n - H, 5)):
        if ti%300==0: print(f"  day {ti}/~260 trades={len(all_t)}", flush=True)
        cand=[]
        for sy,(ind,c) in cache.items():
            if day>=len(c): continue
            sc=sf(ind,day)
            if sc>0: cand.append((sc,sy,c[day]))
        cand.sort(key=lambda x:-x[0])
        
        for sc,sy,bp in cand[:T5]:
            ic=cache[sy]; cc=ic[1]
            sd=min(day+H,len(cc)-1)
            for d2 in range(day+1,sd+1):
                if (cc[d2]-bp)/bp<=SL: sd=d2; sp=cc[d2]; break
            else:
                sp=cc[sd]
            ret=(sp-bp)/bp
            if ret is None or (isinstance(ret, float) and ret != ret):
                continue
            all_t.append(ret)
    
    nt=len(all_t)
    if nt==0: print("  无交易"); return None
    wr=sum(1 for r in all_t if r>0)/nt*100
    ar=sum(all_t)/nt*100
    total_ret=sum(all_t)*100
    
    # 按月复利
    nms=max(1,nt//60)
    nav=1.0
    navs=[nav]
    for i in range(0,nt,nms):
        br=all_t[i:i+nms]
        nav*=(1+sum(br)/len(br))
        navs.append(nav)
    
    fn=navs[-1]
    ann=fn**(1/5)-1
    
    # 月收益率序列
    mrs=[navs[i]/navs[i-1]-1 for i in range(1,len(navs))]
    if not mrs: mrs=[0]
    am=sum(mrs)/len(mrs)
    sm=((sum((r-am)**2 for r in mrs)/len(mrs))**0.5) if mrs else 1
    sp=am/sm*(12**0.5) if sm>1e-10 else 0
    
    pk=1.0; mdd=0
    for v in navs:
        if v>pk: pk=v
        d2=(pk-v)/pk
        if d2>mdd: mdd=d2
    
    pm=sum(1 for r in mrs if r>0)
    nm=len(mrs)-pm
    
    el=time.time()-t0
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  交易: {nt}  胜率: {wr:.1f}%  均收益: {ar:+.2f}%")
    print(f"  净值: {fn:.4f}  总收益: {(fn-1)*100:+.1f}%")
    print(f"  年化: {ann*100:+.1f}%  夏普: {sp:.2f}  回撤: {mdd*100:.1f}%")
    print(f"  正/负月: {pm}/{nm}  用时: {el:.0f}s")
    return {"t":nt,"w":wr,"a":ar,"fn":fn,"tr":(fn-1)*100,"ann":ann*100,"sp":sp,"dd":mdd*100}

r1=bt("新V5-S(2026-06-08重构版)","new")

# ── 保存 v5s_score 全量回测结果 ──
if r1:
    res = {
        "strategy_name": "V5-S v5s_score (2026-06-08重构版)",
        "annual_return": round(r1["ann"], 2),
        "sharpe": round(r1["sp"], 2),
        "max_drawdown": round(r1["dd"], 2),
        "win_rate": round(r1["w"], 2),
        "trade_count": r1["t"],
        "sl": SL,
        "hold_days": H,
        "top_n": T5,
        "total_return": round(r1["tr"], 2),
        "avg_return": round(r1["a"], 2),
        "nav": round(r1["fn"], 4),
        "pool_size": len(syms),
        "valid_pool": len([k for k in all_d if len(all_d[k].get("c",[]))>=520])
    }
    jp = f"{ws}/data/bt_v5s_score_full.json"
    json.dump(res, open(jp,"w"), indent=2, ensure_ascii=False)
    print(f"\n  ✅ 已保存: {jp}", flush=True)
    
    # 生成报告
    md = f"""# V5-S v5s_score 全量回测报告

**时间**: {time.strftime('%Y-%m-%d %H:%M')}  
**参数**: SL={SL*100:.0f}%%, H={H}, T5={T5}  
**池**: {len(syms)}只 (有效{res['valid_pool']})  

## 核心指标

| 指标 | 值 |
|:---|---:|
| 年化收益 | {res['annual_return']:+.2f}% |
| 夏普比率 | {res['sharpe']:.2f} |
| 最大回撤 | {res['max_drawdown']:.1f}% |
| 胜率 | {res['win_rate']:.1f}% |
| 交易笔数 | {res['trade_count']} |
| 总收益 | {res['total_return']:+.1f}% |
| 均收益 | {res['avg_return']:+.2f}% |
| 期末净值 | {res['nav']:.4f} |

## 参数说明

- **SL**: -{SL*100:.0f}% 止损线
- **H**: {H}天 最大持仓天数
- **T5**: 每期选Top {T5}
- 步进周期: 5天
- 回测跨度: 约5年
"""
    mp = f"{ws}/data/_out/bt_v5s_score_report.md"
    os.makedirs(f"{ws}/data/_out", exist_ok=True)
    with open(mp,"w",encoding='utf-8') as f: f.write(md)
    print(f"  ✅ 已保存: {mp}", flush=True)
print(f"\n>>> 新模型完成，开始旧模型", flush=True)
r2=bt("旧V5-S(原始版)","old")

if r1 and r2:
    print(f"\n{'='*60}")
    print(f"  📊 全量对比 (2436只)")
    print(f"{'='*60}")
    for n,k in [("交易","t"),("胜率","w"),("均收益","a"),("总收益","tr"),("年化","ann"),("夏普","sp"),("回撤","dd")]:
        a,b=r1[k],r2[k]; d=a-b
        s='🟢' if d>0 else '🔴' if d<0 else '⚪'
        print(f"  {s} {n:<6}: 新={a:.1f} 旧={b:.1f} ({d:+.1f})")
