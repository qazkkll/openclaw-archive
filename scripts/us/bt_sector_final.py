#!/usr/bin/env python3
"""A_V1.1: V1评分 + 行业动量预筛选 — 最终版"""
import json, time, sys, copy

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
        "stop_loss_pct": -0.08,
        "min_hold_days": 5,
        "transaction_cost": 0.003,
        "initial_capital": 1000000.0,
        "sector_top_n": 4,
        "sector_hold_n": 4,
        "per_sec": 2,
        "position_pct": 0.15,
        "exclude_sectors": [],
        "sector_sample_stocks": 300
    }
}

def load_data(cfg):
    d = cfg["data"]
    with open(d["stock_file"], encoding="utf-8") as f: YAHOO = json.load(f)
    with open(d["cache_file"]) as f: CACHE = json.load(f)
    with open(d["sector_map"], encoding="utf-8") as f: SMAP = json.load(f)
    codes = [c for c in YAHOO if c != "000001"]
    all_dates = sorted(set(dt for s in YAHOO for dt in YAHOO[s].get("dates",[]) if dt))
    stock_date_idx = {c:{dt:i for i,dt in enumerate(YAHOO[c].get("dates",[]))} for c in YAHOO}
    start = min(min(v.keys()) for v in CACHE.values())
    start_idx = all_dates.index(start)
    return YAHOO, CACHE, SMAP, codes, all_dates, stock_date_idx, start_idx

def sector_momentum(YAHOO, stock_date_idx, codes, SMAP, date, sample_n, exclude):
    mom = {}
    for code in codes[:sample_n]:
        sec = SMAP.get(code, "其他")
        if sec in exclude: continue
        idx = stock_date_idx.get(code,{}).get(date,-1)
        if idx < 20: continue
        cl = YAHOO[code].get("close",[])
        if idx >= len(cl): continue
        ret = (cl[idx]-cl[idx-20])/cl[idx-20]*100
        mom.setdefault(sec, []).append(ret)
    return {s: sum(v)/len(v) for s,v in mom.items() if len(v) >= 2}

def run_backtest_v11(YAHOO, CACHE, SMAP, all_dates, stock_date_idx, start_idx, cfg):
    s = cfg["strategy"]
    INIT = s["initial_capital"]
    cash = INIT; pos = {}; trades = 0
    
    for di in range(start_idx, len(all_dates)-1):
        date = all_dates[di]
        
        avg_mom = sector_momentum(YAHOO, stock_date_idx, list(CACHE.keys()), SMAP, date, 
                                   s["sector_sample_stocks"], set(s["exclude_sectors"]))
        if not avg_mom: continue
        ranked_sec = sorted(avg_mom.items(), key=lambda x:-x[1])
        top_secs = {r[0] for r in ranked_sec[:s["sector_top_n"]]}
        hold_secs = {r[0] for r in ranked_sec[:s["sector_hold_n"]]}
        
        # Sell: sector dropped out or score < sell
        for code in [k for k in list(pos.keys()) if not k.endswith("_p")]:
            sec = SMAP.get(code, "其他")
            sc = CACHE[code].get(date, 0)
            if sec not in hold_secs or sc < s["sell_threshold"]:
                idx = stock_date_idx.get(code,{}).get(date,-1)
                if idx >= 0:
                    pr = YAHOO[code]["close"][idx]
                    cash += pos[code] * (1 + (pr - pos[code+'_p']) / pos[code+'_p'])
                del pos[code]; del pos[code+'_p']; trades += 1
        
        if (di - start_idx) % s["rebalance_days"] == 0:
            # Score within top sectors
            cands = {}
            for code in CACHE:
                if code in pos: continue
                sec = SMAP.get(code, "其他")
                if sec not in top_secs: continue
                sc = CACHE[code].get(date, 0)
                if sc < s["buy_threshold"]: continue
                idx = stock_date_idx.get(code,{}).get(date,-1)
                if idx < 0: continue
                pr = YAHOO[code]["close"][idx]
                if pr <= 0: continue
                cands.setdefault(sec, []).append((code, sc, pr))
            
            for sec in top_secs:
                cs = sorted(cands.get(sec, []), key=lambda x:-x[1])
                for code, sc, pr in cs[:s["per_sec"]]:
                    if len(pos) >= s["max_positions"]: break
                    invest = min(cash * s["position_pct"], cash * 0.95)
                    if invest < 20000: continue
                    pos[code] = invest; pos[code+'_p'] = pr
                    cash -= invest; trades += 1
    
    final = cash
    val_keys = [k for k in pos if not k.endswith('_p')]
    for code in val_keys:
        idx = stock_date_idx.get(code,{}).get(all_dates[-1], -1)
        if idx >= 0:
            pr = YAHOO[code]["close"][idx]
            if pr > 0: final += pos[code] * (1 + (pr - pos[code+'_p']) / pos[code+'_p'])
    
    ret = (final/INIT-1)*100
    yrs = max((len(all_dates)-start_idx)/245, 1)
    ann = ((final/INIT)**(1/yrs)-1)*100
    return ret, ann, 0, 0, trades

if __name__ == "__main__":
    YAHOO, CACHE, SMAP, codes, all_dates, stock_date_idx, start_idx = load_data(CONFIG)
    
    print(f"\n{'='*60}")
    print(f"  A_V1.1 回测 (V1评分+行业动量)")
    print(f"{'='*60}")
    print(f"  数据: {len(codes)}只 x {len(all_dates)}天")
    print(f"  参数: 买>{CONFIG['strategy']['buy_threshold']} 卖<{CONFIG['strategy']['sell_threshold']}")
    print(f"        持仓{CONFIG['strategy']['max_positions']}只  前{CONFIG['strategy']['sector_top_n']}行业")
    print(f"        每行业最多{CONFIG['strategy']['per_sec']}只  再平衡{CONFIG['strategy']['rebalance_days']}天")
    print()
    
    t0 = time.time()
    ret, ann, mdd, sharpe, trades = run_backtest_v11(YAHOO, CACHE, SMAP, all_dates, stock_date_idx, start_idx, CONFIG)
    elapsed = time.time() - t0
    
    print(f"  {'='*50}")
    print(f"  总收益率:   {ret:>+8.2f}%")
    print(f"  年化收益率: {ann:>+8.2f}%")
    print(f"  交易次数:   {trades}")
    print(f"  {'='*50}")
    print(f"  用时: {elapsed:.0f}s")
