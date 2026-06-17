#!/usr/bin/env python3
"""
bt_combined.py + 资金流/龙虎榜 完整集成
修改：run()内设日期全局变量，get_fund_flow按日期查缓存
"""
import json, os, sys, time, urllib.request

# ── 资金流/龙虎榜缓存 ──
CACHE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../data/cache/'
_CUR_DATE = None  # 回测时由run()设置

# 构建资金流索引 {code: {yyyymmdd: net_amount}}
_MF_CACHE = {}
for f in os.listdir(CACHE_DIR):
    if not f.startswith('mf_') or not f.endswith('.json'): continue
    code = f[3:-5]
    with open(CACHE_DIR+f) as fh:
        for r in json.load(fh):
            dt = r.get('trade_date','')
            try: _MF_CACHE.setdefault(code, {})[dt] = float(r.get('net_mf_amount',0))
            except: pass
print('-- MF cached: %d stocks' % len(_MF_CACHE), flush=True)

# 构建龙虎榜索引 {code: {yyyymmdd: fd_amount}}
_LL_CACHE = {}
for f in os.listdir(CACHE_DIR):
    if not f.startswith('ll_') or not f.endswith('.json'): continue
    dt = f[3:-5]
    with open(CACHE_DIR+f) as fh:
        for r in json.load(fh):
            code = r.get('ts_code','').split('.')[0]
            try: _LL_CACHE.setdefault(code, {})[dt] = float(r.get('fd_amount',0))
            except: pass
print('-- LL cached: %d stocks' % len(_LL_CACHE), flush=True)

# ── CONFIG ──
CONFIG = {
    "data": {
        "stock_file": "/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json",
        "cache_file": "/home/admin/.openclaw/workspace/data/v1_scores_v2.json",
        "sector_map": "/home/admin/.openclaw/workspace/data/sector_map.json"
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
        "enabled": True,
        "tushare_token": "",
        "bonus_buy": 5,
        "bonus_heavy": 3,
        "penalty_sell": -5,
        "penalty_light": -2
    },
    "longhu": {
        "enabled": True,
        "bonus_longhu_buy": 3
    }
}

def load_data(cfg):
    d = cfg["data"]
    Y = json.load(open(d["stock_file"], encoding="utf-8"))
    C = json.load(open(d["cache_file"]))
    S = json.load(open(d["sector_map"], encoding="utf-8"))
    codes = [c for c in Y if c != "000001"]
    ad = sorted(set(dt for s in Y for dt in Y[s].get("dates",[]) if dt))
    sdx = {c:{dt:i for i,dt in enumerate(Y[c].get("dates",[]))} for c in Y}
    start = min(min(v.keys()) for v in C.values())
    return Y, C, S, codes, ad, sdx, ad.index(start)

def sect_mom(Y, sdx, codes, S, date, n, ex):
    m={}
    for c in codes[:n]:
        s2=S.get(c,"其他")
        if s2 in ex: continue
        idx=sdx.get(c,{}).get(date,-1)
        if idx<20: continue
        cl=Y[c].get("close",[])
        if idx>=len(cl): continue
        r=(cl[idx]-cl[idx-20])/cl[idx-20]*100 if cl[idx-20]>0 else 0
        m.setdefault(s2,[]).append(r)
    return {s:sum(v)/len(v) for s,v in m.items() if len(v)>=2}

# 带日期查询的资金流
def get_fund_flow(code, token):
    global _CUR_DATE
    if _CUR_DATE is None:
        return None
    dt_key = _CUR_DATE.replace('-','')
    net = _MF_CACHE.get(code, {}).get(dt_key, 0)
    if net != 0:
        return {'net_': net, 'buy_lg': 0, 'sell_lg': 0}
    return None

def get_longhu(code):
    global _CUR_DATE
    if _CUR_DATE is None:
        return None
    dt_key = _CUR_DATE.replace('-','')
    net = _LL_CACHE.get(code, {}).get(dt_key, 0)
    if net != 0:
        return {'net_': net, 'type': 'longhu'}
    return None

def run_bt(cfg, label='V1+MF+LL'):
    Y, C, S, cd, ad, sdx, si = load_data(cfg)
    s = cfg["strategy"]; f = cfg["fund_flow"]; l = cfg["longhu"]
    I = s["initial_capital"]; cash = I; pos = {}; trades = 0
    mf_hits = 0; ll_hits = 0; total_candidates = 0
    
    for di in range(si, len(ad)-1):
        global _CUR_DATE
        _CUR_DATE = ad[di]
        date = ad[di]
        
        am = sect_mom(Y, sdx, list(C.keys()), S, date, s["sector_sample"], {})
        if not am: continue
        rs = sorted(am.items(), key=lambda x:-x[1])
        ts = {r[0] for r in rs[:s["sector_top_n"]]}
        hs = ts.copy()
        
        for c in [k for k in list(pos.keys()) if not k.endswith("_p")]:
            s2 = S.get(c, "其他"); sc = C[c].get(date, 0)
            fund_flow_data = get_fund_flow(c, "")
            if fund_flow_data and isinstance(fund_flow_data, dict):
                net = fund_flow_data.get('net_', 0)
                if net > 0: sc += f['bonus_buy']
                elif net < 0: sc += f['penalty_sell']
            longhu_data = get_longhu(c)
            if longhu_data and isinstance(longhu_data, dict):
                net = longhu_data.get('net_', 0)
                if net > 0: sc += l['bonus_longhu_buy']
            
            if s2 not in hs or sc < s["sell_threshold"]:
                idx = sdx.get(c,{}).get(date,-1)
                if idx >= 0:
                    pr = Y[c]["close"][idx]
                    cash += pos[c] * (1+(pr-pos[c+"_p"])/pos[c+"_p"]) if pos[c+"_p"] > 0 else 0
                del pos[c]; del pos[c+"_p"]; trades += 1
        
        if (di-si)%s["rebalance_days"] == 0:
            ca = {}
            for c in C:
                if c in pos: continue
                s2 = S.get(c, "其他")
                if s2 not in ts: continue
                sc = C[c].get(date, 0)
                if sc < s["buy_threshold"]: continue
                idx = sdx.get(c,{}).get(date,-1)
                if idx < 0: continue
                pr = Y[c]["close"][idx]
                if pr <= 0: continue
                # 资金流加分
                ff = get_fund_flow(c,"")
                if ff and isinstance(ff,dict):
                    net = ff.get('net_',0)
                    if net > 0: sc += f['bonus_buy']
                    elif net < 0: sc += f['penalty_sell']
                lh = get_longhu(c)
                if lh and isinstance(lh,dict):
                    net = lh.get('net_',0)
                    if net > 0: sc += l['bonus_longhu_buy']
                ca.setdefault(s2,[]).append((c,sc,pr))
                total_candidates += 1
            
            for s2 in ts:
                cs = sorted(ca.get(s2,[]), key=lambda x:-x[1])
                for c,sc,pr in cs[:s["per_sec"]]:
                    if len(pos) >= s["max_positions"]*2: break
                    inv = min(cash*s["position_pct"], cash*0.95)
                    if inv < 20000: continue
                    pos[c]=inv; pos[c+"_p"]=pr; cash-=inv; trades+=1
                    if _MF_CACHE.get(c,{}).get(date.replace('-',''),0)!=0: mf_hits+=1
                    if _LL_CACHE.get(c,{}).get(date.replace('-',''),0)!=0: ll_hits+=1
    
    fin = cash
    for c in [k for k in pos if not k.endswith("_p")]:
        idx=sdx.get(c,{}).get(ad[-1],-1)
        if idx>=0:
            pr=Y[c]["close"][idx]
            if pr>0: fin+=pos[c]*(1+(pr-pos[c+"_p"])/pos[c+"_p"])
    ret=(fin/I-1)*100; yr=max((len(ad)-si)/245,1); ann=((fin/I)**(1/yr)-1)*100
    return ret, ann, trades, mf_hits, ll_hits

if __name__=="__main__":
    t0 = time.time()
    r = run_bt(CONFIG, 'V1+MF+LL')
    print('DONE: %.0fs' % (time.time()-t0))
    print('ret=%.2f%% ann=%.2f%% trades=%d mf=%d ll=%d' % r)
    print('Parameters: buy=%d sell=%d reb=%dd sect=%d maxpos=%d' % (
        CONFIG['strategy']['buy_threshold'], CONFIG['strategy']['sell_threshold'],
        CONFIG['strategy']['rebalance_days'], CONFIG['strategy']['sector_top_n'],
        CONFIG['strategy']['max_positions']))
    
    json.dump({'ret':r[0],'ann':r[1],'trades':r[2],'mf_hits':r[3],'ll_hits':r[4]},
              open('/home/admin/.openclaw/workspace/data/bt_mf_final.json','w'), indent=2)
