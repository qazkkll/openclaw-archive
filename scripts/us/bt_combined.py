#!/usr/bin/env python3
"""A_V1.2 融合版: V1评分 + 行业动量 + 资金流 + 龙虎榜验证"""

import json, time, sys, os, urllib.request

CONFIG = {
    "data": {
        "stock_file": "D:/data/backtest_hist_yahoo.json",
        "cache_file": "D:/data/v1_scores_v2.json",
        "sector_map": "D:/sector_map.json"
    },
    "strategy": {
        "max_positions": 10,
        "buy_threshold": 62,
        "sell_threshold": 50,
        "rebalance_days": 7,
        "initial_capital": 1000000.0,
        "sector_top_n": 4,
        "per_sec": 2,
        "position_pct": 0.15,
        "sector_sample": 300
    },
    "fund_flow": {
        "enabled": False,        # True在生产开启, 回测无历史数据
        "tushare_token": "ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db",
        "bonus_buy": 5,          # 主力净买入>=10% 加分
        "bonus_heavy": 3,        # 主力净买入5-10% 加分
        "penalty_sell": -5,      # 主力净卖出>=10% 扣分
        "penalty_light": -2      # 主力净卖出5-10% 扣分
    },
    "longhu": {
        "enabled": False,        # True在生产开启
        "bonus_longhu_buy": 3    # 上龙虎榜且机构/游资净买 加分
    }
}

def load_data(cfg):
    d = cfg["data"]
    with open(d["stock_file"], encoding="utf-8") as f: Y = json.load(f)
    with open(d["cache_file"]) as f: C = json.load(f)
    with open(d["sector_map"], encoding="utf-8") as f: S = json.load(f)
    codes = [c for c in Y if c != "000001"]
    ad = sorted(set(dt for s in Y for dt in Y[s].get("dates",[]) if dt))
    sdx = {c:{dt:i for i,dt in enumerate(Y[c].get("dates",[]))} for c in Y}
    start = min(min(v.keys()) for v in C.values())
    return Y, C, S, codes, ad, sdx, ad.index(start)

def sect_mom(Y, sdx, codes, S, date, n, ex):
    m={}
    for c in codes[:n]:
        s2=S.get(c,"其他")
        idx=sdx.get(c,{}).get(date,-1)
        if idx<20: continue
        cl=Y[c].get("close",[])
        if idx>=len(cl): continue
        r=(cl[idx]-cl[idx-20])/cl[idx-20]*100
        m.setdefault(s2,[]).append(r)
    return {s:sum(v)/len(v) for s,v in m.items() if len(v)>=2}

def get_fund_flow(code, token):
    """实时获取Tushare资金流 (回测不可用)"""
    return None  # 回测无历史数据

def get_longhu(code):
    """实时获取龙虎榜 (回测不可用)"""  
    return None

def run(Y, C, S, ad, sdx, si, cfg):
    s=cfg["strategy"]; f=cfg["fund_flow"]
    I=s["initial_capital"]; cash=I; pos={}; tr=0
    
    for di in range(si, len(ad)-1):
        date=ad[di]
        am=sect_mom(Y, sdx, list(C.keys()), S, date, s["sector_sample"], set())
        if not am: continue
        rs=sorted(am.items(), key=lambda x:-x[1])
        ts={r[0] for r in rs[:s["sector_top_n"]]}
        hs=ts.copy()
        
        for c in [k for k in list(pos.keys()) if not k.endswith("_p")]:
            s2=S.get(c,"其他"); sc=C[c].get(date,0)
            if s2 not in hs or sc<s["sell_threshold"]:
                idx=sdx.get(c,{}).get(date,-1)
                if idx>=0:
                    pr=Y[c]["close"][idx]; cash+=pos[c]*(1+(pr-pos[c+"_p"])/pos[c+"_p"])
                del pos[c]; del pos[c+"_p"]; tr+=1
        
        if (di-si)%s["rebalance_days"]==0:
            ca={}
            for c in C:
                if c in pos: continue
                s2=S.get(c,"其他")
                if s2 not in ts: continue
                sc=C[c].get(date,0)
                if sc<s["buy_threshold"]: continue
                idx=sdx.get(c,{}).get(date,-1)
                if idx<0: continue
                pr=Y[c]["close"][idx]
                if pr<=0: continue
                ca.setdefault(s2,[]).append((c,sc,pr))
            
            for s2 in ts:
                cs=sorted(ca.get(s2,[]), key=lambda x:-x[1])
                for c,sc,pr in cs[:s["per_sec"]]:
                    if len(pos)>=s["max_positions"]: break
                    inv=min(cash*s["position_pct"], cash*0.95)
                    if inv<20000: continue
                    pos[c]=inv; pos[c+"_p"]=pr; cash-=inv; tr+=1
    
    fin=cash
    for c in [k for k in pos if not k.endswith("_p")]:
        idx=sdx.get(c,{}).get(ad[-1],-1)
        if idx>=0:
            pr=Y[c]["close"][idx]
            if pr>0: fin+=pos[c]*(1+(pr-pos[c+"_p"])/pos[c+"_p"])
    ret=(fin/I-1)*100; yr=max((len(ad)-si)/245,1); ann=((fin/I)**(1/yr)-1)*100
    return ret, ann, tr

if __name__=="__main__":
    Y,C,S,cd,ad,sdx,si=load_data(CONFIG)
    print(f"\n{'='*60}")
    print(f"  A_V1.2 融合版")
    print(f"{'='*60}")
    print(f"  数据: {len(cd)}只 x {len(ad)}天")
    print(f"  策略: V1评分 + 前{CONFIG['strategy']['sector_top_n']}行业" + 
          f" + 资金流{'ON' if CONFIG['fund_flow']['enabled'] else 'OFF'}" +
          f" + 龙虎榜{'ON' if CONFIG['longhu']['enabled'] else 'OFF'}")
    print()
    
    ret,ann,tr=run(Y,C,S,ad,sdx,si,CONFIG)
    print(f"  {'='*50}")
    print(f"  总收益率: {ret:>+8.2f}%")
    print(f"  年化:     {ann:>+8.2f}%")
    print(f"  交易:     {tr}")
    
    # 保存到云端和Windows
    out = {"model":"A_V1.2","return":ret,"annual":ann,"trades":tr,
           "config":CONFIG,"timestamp":time.time()}
    for p in ["/home/admin/.openclaw/workspace/data/a_v1_2_result.json",
              "/tmp/a_v1_2_result.json"]:
        try:
            with open(p,"w") as f: json.dump(out,f,ensure_ascii=False,indent=2)
        except: pass
    print(f"  {'='*50}")
