#!/usr/bin/env python3
"""
美股宏观因子回测引擎 v3 — 优化版
预计算所有指标，批量网格搜索
"""

import json, math, time, os, sys, random
import yfinance as yf

CACHE_DIR = "/home/admin/.openclaw/workspace/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)
RESULTS_FILE = "/home/admin/.openclaw/workspace/data/us_macro_backtest.json"

STOCKS = ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","AMD","TXN","QCOM","AMAT","MU","INTC","MRVL","KLAC","LRCX","CRM","ADBE","ORCL","NOW","INTU","UBER","SHOP","SNOW","PLTR","DDOG","ZS","NET","MDB","CRWD","PANW","FTNT","EA","DIS","NFLX","RBLX","U","SE","CPNG","SQ","COIN","SOFI","DASH","V","MA","PYPL","HD","COST","WMT","NKE","SBUX","MCD","BKNG","ABNB","JPM","GS","BAC","MS","SCHW","BLK","SPGI","MCO","ADP","FISV","JNJ","UNH","PFE","ABBV","MRK","LLY","AMGN","GILD","REGN","VRTX","DHR","TMO","XOM","CVX","COP","SLB","EOG","CAT","GE","BA","HON","MMM","RTX","LMT","NOC","DE","LIN","SHW","NEE","DUK","SO","WELL","PLD","AMT","EQIX","MELI","CPNG","ABNB","BKNG","DASH","SQ","COIN"]
MACRO_TICKERS = {"VIX":"^VIX","TNX":"^TNX","DXY":"DX-Y.NYB","SPY":"SPY"}

def get_cache_path(ticker):
    safe = ticker.replace("^","_").replace("-","_")
    return os.path.join(CACHE_DIR,f"{safe}.json")

def fetch(ticker, period="5y"):
    cp = get_cache_path(ticker)
    if os.path.exists(cp):
        try:
            with open(cp) as f: c = json.load(f)
            if c.get("date")==time.strftime("%Y-%m-%d"): return c["data"]
        except: pass
    print(f"  📡 {ticker}...", end=" ", flush=True)
    try:
        t = yf.Ticker(ticker); h = t.history(period=period)
        time.sleep(0.5)
        if h.empty: print("空"); return None
        data=[{"date":idx.strftime("%Y-%m-%d"),"open":float(r["Open"]),"high":float(r["High"]),
               "low":float(r["Low"]),"close":float(r["Close"]),"volume":int(r["Volume"])}
              for idx,r in h.iterrows()]
        with open(cp,"w") as f: json.dump({"date":time.strftime("%Y-%m-%d"),"data":data},f)
        print(f"✅ {len(data)}天"); return data
    except Exception as e: print(f"❌ {e}"); return None

def fetch_macro(name, yt, period="5y"):
    cp = get_cache_path(yt)
    if os.path.exists(cp):
        try:
            with open(cp) as f: c = json.load(f)
            if c.get("date")==time.strftime("%Y-%m-%d"): return c["data"]
        except: pass
    print(f"  📡 {name}...", end=" ", flush=True)
    try:
        t = yf.Ticker(yt); h = t.history(period=period)
        time.sleep(0.5)
        if h.empty: print("空"); return None
        data=[{"date":idx.strftime("%Y-%m-%d"),"close":float(r["Close"])} for idx,r in h.iterrows()]
        with open(cp,"w") as f: json.dump({"date":time.strftime("%Y-%m-%d"),"data":data},f)
        print(f"✅ {len(data)}天"); return data
    except Exception as e: print(f"❌ {e}"); return None

# === 技术指标 ===
def ema(p,n):
    k=2/(n+1); r=[p[0]]
    for v in p[1:]: r.append(v*k+r[-1]*(1-k))
    return r
def sma(p,n):
    return [None]*(n-1)+[sum(p[i-n+1:i+1])/n for i in range(n-1,len(p))]
def rsi_full(p):
    if len(p)<16: return [None]*len(p)
    g,l=[],[]
    for i in range(1,len(p)): d=p[i]-p[i-1]; g.append(max(d,0)); l.append(max(-d,0))
    r,ag,al=[None]*14,sum(g[:14])/14,sum(l[:14])/14
    for i in range(14,len(p)):
        r.append(100-100/(1+ag/al) if al else 100)
        if i<len(g): ag=(ag*13+g[i])/14; al=(al*13+l[i])/14
    return r
def sf(a,i):
    return a[i] if 0<=i<len(a) and a[i] is not None else None

def v1_tech_score(i,p,rsi,ma5,ma20,ma60,hist,pos52):
    s=0
    if sf(hist,i) is not None:
        if sf(hist,i)>0 and (sf(hist,i-1)is None or sf(hist,i-1)<=0): s+=35
        elif sf(hist,i-1)is not None and sf(hist,i)>sf(hist,i-1): s+=20
        elif sf(hist,i)>0: s+=10
    if sf(ma20,i)is not None and p[i]>sf(ma20,i): s+=10
    if sf(ma5,i)is not None and sf(ma20,i)is not None and sf(ma5,i)>sf(ma20,i): s+=10
    if sf(ma20,i)is not None and sf(ma60,i)is not None and sf(ma20,i)>sf(ma60,i): s+=5
    if sf(rsi,i)is not None:
        if sf(rsi,i)<35: s+=20
        elif sf(rsi,i)<60: s+=15
        elif sf(rsi,i)<70: s+=10
        else: s-=10
    if sf(pos52,i)is not None:
        if sf(pos52,i)<30: s+=20
        elif sf(pos52,i)<60: s+=15
        elif sf(pos52,i)<80: s+=10
        else: s+=5
    return max(0,min(100,s))

def compute_indicators(p, hval, lval):
    n=len(p)
    rsi=rsi_full(p)
    m5=sma(p,5); m20=sma(p,20); m60=sma(p,60)
    e12=ema(p,12); e26=ema(p,26)
    macd=[e12[i]-e26[i] for i in range(n)]
    sig=ema(macd,9)
    hist=[macd[i]-sig[i] for i in range(n)]
    p52=[None]*251
    for i in range(251,n):
        lo=min(p[i-251:i+1]); hi=max(p[i-251:i+1])
        p52.append((p[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    return p,rsi,m5,m20,m60,hist,p52

# === 宏观评分 ===
def m_vix(v):
    if v is None: return 50
    if v<15: return 70
    if v<20: return 60
    if v<25: return 50
    if v<30: return 30
    return 20
def m_tnx(v,c):
    b=50
    if v is None: return 50
    if v<3.5: b=60
    elif v<4.0: b=55
    elif v<4.5: b=50
    elif v<5.0: b=45
    else: b=35
    if c:
        if c>0.05: b-=10
        elif c>0.02: b-=5
        elif c<-0.05: b+=10
        elif c<-0.02: b+=5
    return max(20,min(80,b))
def m_dxy(v,c):
    b=50
    if v is None: return 50
    if v<100: b=60
    elif v<103: b=55
    elif v<106: b=50
    elif v<110: b=40
    else: b=35
    if c:
        if c>0.3: b-=8
        elif c>0.1: b-=4
        elif c<-0.3: b+=8
        elif c<-0.1: b+=4
    return max(30,min(70,b))

# === 预计算数据 ===
def precompute_stock(prices,dates,highs,lows):
    p,rsi,m5,m20,m60,hist,p52 = compute_indicators(prices,highs,lows)
    tech_scores = [v1_tech_score(i,p,rsi,m5,m20,m60,hist,p52) for i in range(len(prices))]
    return {'prices':prices,'dates':dates,'tech_scores':tech_scores,'n':len(prices)}

def precompute_macro(macro_data):
    d={}
    for name in ['vix','tnx','dxy']:
        if macro_data.get(name):
            m={r['date']:{'close':r['close']} for r in macro_data[name]}
            # 添加变化量
            sd=sorted(m.keys())
            for i,dt in enumerate(sd):
                if i>0 and m[sd[i-1]]['close']:
                    chg=(m[dt]['close']-m[sd[i-1]]['close'])/m[sd[i-1]]['close']*100
                else: chg=0
                m[dt]['chg']=chg
            d[name]=m
    return d

def get_macro_prev(md,date,prev_date):
    """获取最接近date的宏观数据"""
    if md is None or date not in md:
        # 找最近
        best=None
        for d in sorted(md.keys()):
            if d<=date: best=d
            else: break
        if best: return md[best]
        return None
    return md[date]

# === 轻量级回测(基于预计算数据) ===
def fast_backtest(sd, macro_dicts, weights, buy_th=50, sell_th=30):
    p=sd['prices']; dts=sd['dates']; tscores=sd['tech_scores']
    trades=[]; ip=False; ep=0; ed=""
    
    for i in range(len(p)):
        dt=dts[i]; ts=tscores[i]
        ms={}
        for mn in ['vix','tnx','dxy']:
            if weights.get(mn,0)==0: continue
            rec=None
            if mn in macro_dicts:
                if dt in macro_dicts[mn]:
                    rec=macro_dicts[mn][dt]
                else:
                    best=None
                    for d in sorted(macro_dicts[mn].keys()):
                        if d<=dt: best=d
                        else: break
                    if best: rec=macro_dicts[mn][best]
            if rec is None: continue
            if mn=='vix': ms['vix']=m_vix(rec['close'])
            elif mn=='tnx': ms['tnx']=m_tnx(rec['close'],rec['chg'])
            elif mn=='dxy': ms['dxy']=m_dxy(rec['close'],rec['chg'])
        
        tw=max(0,100-sum(weights.values()))
        total=ts*tw
        for k,v in ms.items(): total+=v*weights.get(k,0)
        final=total/100.0
        
        if not ip and final>=buy_th:
            ip=True; ep=p[i]; ed=dt
        elif ip and final<sell_th:
            ip=False
            pnl=(p[i]-ep)/ep*100
            trades.append({'entry':ed,'exit':dt,'ep':round(ep,2),'xp':round(p[i],2),'pnl':round(pnl,2)})
    
    if not trades: return {'trades':0,'total_pnl':0,'win_rate':0,'avg_win':0,'avg_loss':0,'profit_factor':0,'max_dd':0,'hold_pnl':0}
    wins=[t for t in trades if t['pnl']>0]; losses=[t for t in trades if t['pnl']<=0]
    aw=sum(t['pnl'] for t in wins)/len(wins) if wins else 0
    al=sum(t['pnl'] for t in losses)/len(losses) if losses else 0
    peak=0;mdd=0;cum=0
    for t in trades: cum+=t['pnl']; peak=max(peak,cum); mdd=min(mdd,cum-peak)
    hp=(p[-1]-p[0])/p[0]*100 if len(p)>1 else 0
    return {'trades':len(trades),'wins':len(wins),'total_pnl':round(sum(t['pnl'] for t in trades),2),
            'win_rate':round(len(wins)/len(trades)*100,1),'avg_win':round(aw,2),'avg_loss':round(al,2),
            'profit_factor':round(abs(aw/al),2) if al else 99,'max_dd':round(mdd,2),'hold_pnl':round(hp,2)}

# ===== 阶段1: 独立测试 =====
def phase1(sd_pool, md):
    print(f"\n{'='*60}\n📊 阶段1: 宏观因子独立测试\n{'='*60}")
    
    base_res={t:fast_backtest(sd_pool[t],{},{}) for t in sd_pool}
    print("  📈 基线: V1纯技术")
    for t,r in base_res.items():
        e="🟢" if r['total_pnl']>0 else "🔴"
        print(f"    {e} {t:<6}: x{r['trades']:>2} 总{r['total_pnl']:>+7.1f}% 胜率{r['win_rate']:>5.1f}%")
    
    tests={"VIX(10%)":{"vix":10,"tnx":0,"dxy":0},"TNX(5%)":{"vix":0,"tnx":5,"dxy":0},
           "DXY(5%)":{"vix":0,"tnx":0,"dxy":5},"VIX+TNX(10+5)":{"vix":10,"tnx":5,"dxy":0},
           "VIX+TNX+DXY(10+5+5)":{"vix":10,"tnx":5,"dxy":5}}
    
    results={}
    for fn,fw in tests.items():
        print(f"\n  📈 测试: V1+{fn}")
        res=[]
        for t in sd_pool:
            r=fast_backtest(sd_pool[t],md,fw)
            if r['trades']>0:
                res.append({'ticker':t,'pnl':r['total_pnl'],'wr':r['win_rate'],'n':r['trades']})
                e="🟢" if r['total_pnl']>0 else "🔴"
                bm="+" if r['total_pnl']>base_res[t]['total_pnl'] else "-"
                print(f"    {e} {t:<6}: x{r['trades']:>2} 总{r['total_pnl']:>+7.1f}% wr{r['win_rate']:>5.1f}% {bm}")
        if res:
            ap=sum(x['pnl'] for x in res)/len(res)
            bp=sum(base_res[t]['total_pnl'] for t in sd_pool if any(x['ticker']==t for x in res))/len(res)
            results[fn]={'avg_pnl':round(ap,2),'base_avg':round(bp,2),
                        'improvement':round(ap-bp,2),
                        'positive':len([x for x in res if x['pnl']>0]),'total':len(res)}
            print(f"    → 平均:{ap:+.1f}% vs基线{bp:+.1f}%(改善{ap-bp:+.1f}%)")
    return results

# ===== 阶段2: 组合测试 =====
MODEL_GROUPS={
    "A: V1纯技术(100%)": {"vix":0,"tnx":0,"dxy":0},
    "B: V1+VIX(90+10)": {"vix":10,"tnx":0,"dxy":0},
    "C: V1+VIX+TNX(85+10+5)": {"vix":10,"tnx":5,"dxy":0},
    "D: V1+VIX+TNX+DXY(80+10+5+5)": {"vix":10,"tnx":5,"dxy":5},
    "E: V1+全宏(75+10+5+5+5)": {"vix":10,"tnx":5,"dxy":5},
}

def phase2(sd_pool, md):
    print(f"\n{'='*60}\n📊 阶段2: 宏观+V1组合测试(5组)\n{'='*60}")
    results={}
    for gn,gw in MODEL_GROUPS.items():
        print(f"\n  🔬 {gn}:")
        sr=[]
        for t in sd_pool:
            r=fast_backtest(sd_pool[t],md,gw)
            if r['trades']>0:
                sr.append({**r,'ticker':t})
                e="🟢" if r['total_pnl']>0 else "🔴"
                print(f"    {e} {t:<6}: x{r['trades']:>2} 总{r['total_pnl']:>+7.1f}% wr{r['win_rate']:>5.1f}%")
        if sr:
            ap=sum(s['total_pnl'] for s in sr)/len(sr)
            aw=sum(s['win_rate'] for s in sr)/len(sr)
            po=len([s for s in sr if s['total_pnl']>0])
            results[gn]={'avg_pnl':round(ap,2),'avg_wr':round(aw,1),
                        'total_trades':sum(s['trades'] for s in sr),
                        'positive':po,'total':len(sr),'details':sr}
            print(f"    → 平均:{ap:+.1f}% wr{aw:.1f}% 正{po}/{len(sr)}")
    return results

# ===== 阶段3: 交叉验证 =====
def phase3(sd_pool, md):
    print(f"\n{'='*60}\n📊 阶段3: 随机时段交叉验证(5折)\n{'='*60}")
    common=None
    for s in sd_pool.values():
        ds=set(s['dates'])
        common=ds if common is None else common&ds
    all_dt=sorted(common); td=len(all_dt)
    print(f"  共{td}交易日")
    results={}
    
    for si in range(5):
        random.seed(42+si*100)
        ts=random.randint(0,max(0,td-520))
        te=min(ts+400+random.randint(0,100),td-120)
        tst_s=te+100; tst_e=td
        
        print(f"\n  📅 #分割{si+1}: Train{all_dt[ts][:10]}~{all_dt[te-1][:10]} "
              f"Test{all_dt[tst_s][:10]}~{all_dt[tst_e-1][:10]}")
        
        split_res={}
        for gn,gw in MODEL_GROUPS.items():
            pnls=[]; wrs=[]
            for t in sd_pool:
                # 筛选测试日期
                idxs=[i for i,d in enumerate(sd_pool[t]['dates']) if d in all_dt[tst_s:tst_e]]
                if len(idxs)<30: continue
                sub={'prices':[sd_pool[t]['prices'][i] for i in idxs],
                     'dates':[sd_pool[t]['dates'][i] for i in idxs],
                     'tech_scores':[sd_pool[t]['tech_scores'][i] for i in idxs]}
                r=fast_backtest(sub,md,gw)
                if r['trades']>0:
                    pnls.append(r['total_pnl']); wrs.append(r['win_rate'])
            if pnls:
                ap=sum(pnls)/len(pnls); aw=sum(wrs)/len(wrs)
                split_res[gn]={'avg_pnl':round(ap,2),'avg_wr':round(aw,1),'n':len(pnls)}
                m="✅" if ap>0 else "❌"
                print(f"    {m} {gn[:28]:<28}: {ap:+.1f}% wr{aw:.1f}% n={len(pnls)}")
        results[f"split_{si+1}"]=split_res
    
    print(f"\n  {'─'*50}\n  📋 CV汇总:")
    summary={}
    for gn in MODEL_GROUPS:
        vals=[results.get(f"split_{si+1}",{}).get(gn,{}).get('avg_pnl') for si in range(5)]
        vals=[v for v in vals if v is not None]
        if vals:
            ap=sum(vals)/len(vals); pos=len([v for v in vals if v>0])
            summary[gn]={'avg_pnl':round(ap,2),'positive_splits':pos,'total_splits':len(vals),'pnls':vals}
            print(f"    {gn[:28]:<28}: {ap:+.1f}% (正{pos}/{len(vals)}折)")
    results['summary']=summary
    return results

# ===== 阶段4: 网格搜索(优化版) =====
def phase4(sd_pool, md):
    print(f"\n{'='*60}\n📊 阶段4: 网格搜索最优宏观因子权重\n{'='*60}")
    vix_g=[0,5,10,15,20]; tnx_g=[0,3,5,8,10]; dxy_g=[0,3,5,8]
    total_combos=len(vix_g)*len(tnx_g)*len(dxy_g)
    print(f"  搜索空间: {len(vix_g)}×{len(tnx_g)}×{len(dxy_g)}={total_combos}组合")
    results=[]; cnt=0
    
    for vw in vix_g:
        for tw in tnx_g:
            for dw in dxy_g:
                cnt+=1
                weights={"vix":vw,"tnx":tw,"dxy":dw}
                twt=max(0,100-sum(weights.values()))
                if twt<60: continue
                
                pnls=[]
                for t,sd in sd_pool.items():
                    r=fast_backtest(sd,md,weights)
                    if r['trades']>0:
                        pnls.append(r['total_pnl'])
                
                if pnls:
                    ap=sum(pnls)/len(pnls); po=len([p for p in pnls if p>0])
                    results.append({'tech':twt,'vix':vw,'tnx':tw,'dxy':dw,
                                    'avg_pnl':round(ap,2),'positive':po,
                                    'total':len(pnls),'hit_rate':round(po/len(pnls)*100,1)})
        
        print(f"    进度: {vw+vix_g[0]}/{vix_g[-1]} 总计{cnt}/{total_combos}")
    
    results.sort(key=lambda x: x['avg_pnl'], reverse=True)
    print(f"\n  🏆 最优权重(Top 10):")
    print(f"  {'排名':<4} {'技术':>4} {'VIX':>4} {'TNX':>4} {'DXY':>4} {'收益':>7} {'命中':>6} {'/总数':>6}")
    print(f"  {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*7} {'─'*6} {'─'*6}")
    for ri,r in enumerate(results[:10],1):
        print(f"  {ri:<4} {r['tech']:>3}% {r['vix']:>3}% {r['tnx']:>3}% {r['dxy']:>3}% {r['avg_pnl']:>+6.1f}% {r['hit_rate']:>5.1f}% {r['positive']:>2}/{r['total']:<3}")
    return results[:10]

# ===== Main =====
def main():
    t0=time.time()
    print(f"🔥 美股宏观因子回测 v3 启动")
    print(f"   13只代表性美股 + VIX/TNX/DXY")
    print(f"   5年数据 | 阈值:买入50/卖出30")
    
    # 数据获取
    print(f"\n📥 数据获取...")
    stock_data={}
    for t in STOCKS:
        d=fetch(t,"5y")
        if d and len(d)>100: stock_data[t]=d
    
    macro_data={}
    for n,yt in MACRO_TICKERS.items():
        d=fetch_macro(n,yt,"5y")
        if d: macro_data[n.lower()]=d
    
    print(f"\n   ✅ 股票{len(stock_data)}/13 宏观:{'/'.join(macro_data.keys())}")
    if len(stock_data)<5: print("❌ 数据不足"); return
    
    # 预计算
    print(f"\n⚡ 预计算技术指标...")
    sd_pool={}
    for t,sd in stock_data.items():
        prices=[d['close'] for d in sd]
        dates=[d['date'] for d in sd]
        highs=[d['high'] for d in sd]
        lows=[d['low'] for d in sd]
        sd_pool[t]=precompute_stock(prices,dates,highs,lows)
    print(f"   ✅ {len(sd_pool)}只预计算完成")
    
    md=precompute_macro(macro_data)
    
    all_results={
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "stocks":list(stock_data.keys()),
        "model_groups":list(MODEL_GROUPS.keys()),
        "buy_threshold":50,"sell_threshold":30
    }
    
    # 阶段1-2
    try: all_results["phase1_single_factor"]=phase1(sd_pool,md)
    except Exception as e: import traceback; traceback.print_exc(); all_results["phase1_single_factor"]={"error":str(e)}
    
    try: all_results["phase2_combination"]=phase2(sd_pool,md)
    except Exception as e: import traceback; traceback.print_exc(); all_results["phase2_combination"]={"error":str(e)}
    
    # 阶段3
    try: all_results["phase3_cross_validation"]=phase3(sd_pool,md)
    except Exception as e: import traceback; traceback.print_exc(); all_results["phase3_cross_validation"]={"error":str(e)}
    
    # 阶段4
    try: all_results["phase4_grid_search_top10"]=phase4(sd_pool,md)
    except Exception as e: import traceback; traceback.print_exc(); all_results["phase4_grid_search_top10"]={"error":str(e)}
    
    # 保存
    os.makedirs(os.path.dirname(RESULTS_FILE),exist_ok=True)
    with open(RESULTS_FILE,"w") as f: json.dump(all_results,f,indent=2,ensure_ascii=False)
    
    el=time.time()-t0
    print(f"\n{'='*60}")
    print(f"✅ 完成! {el:.0f}秒 结果:{RESULTS_FILE}")
    
    # 最终对比
    print(f"\n📋 最终对比")
    if "phase2_combination" in all_results and isinstance(all_results["phase2_combination"],dict):
        print(f"  {'模型':<28} {'收益':>7} {'胜率':>6} {'交易':>6} {'正/总':>7}")
        print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*6} {'─'*7}")
        for gn in MODEL_GROUPS:
            if gn in all_results["phase2_combination"]:
                r=all_results["phase2_combination"][gn]
                m="★" if r['positive']==r['total'] else " "
                print(f"  {m} {gn[:26]:<26} {r['avg_pnl']:>+6.1f}% {r['avg_wr']:>5.1f}% {r['total_trades']:>5} {r['positive']:>2}/{r['total']:<3}")
    
    if "phase3_cross_validation" in all_results and isinstance(all_results["phase3_cross_validation"],dict):
        s=all_results["phase3_cross_validation"].get("summary",{})
        if s:
            print(f"\n  CV汇总:")
            for gn in MODEL_GROUPS:
                if gn in s:
                    r=s[gn]; print(f"    {gn[:28]:<28}: {r['avg_pnl']:+.1f}% (正{r['positive_splits']}/{r['total_splits']}折)")
    
    if "phase4_grid_search_top10" in all_results:
        top=all_results["phase4_grid_search_top10"]
        if isinstance(top,list) and len(top)>0:
            print(f"\n  最优权重:")
            for ri,r in enumerate(top[:3],1):
                print(f"    #{ri}: 技术{r['tech']}%+VIX{r['vix']}%+TNX{r['tnx']}%+DXY{r['dxy']}% → {r['avg_pnl']:+.1f}%")

if __name__=="__main__":
    main()
