#!/usr/bin/env python3
"""
A_V1 完整回测框架 — 参数全配置化
用法: python3 bt_framework.py               # 默认参数
      python3 bt_framework.py --config custom.json
"""

import json, time, sys

# ===========================================================
# 配置区 — 在这里改参数就行
# ===========================================================
CONFIG = {
    "data": {
        "stock_file": "D:/data/backtest_hist_yahoo.json",
        "cache_file": "D:/data/v1_scores_v2.json",
        "lookback_days": 255,
        "use_open_price": False   # 数据无open字段, 用close代替
    },
    "strategy": {
        "max_positions": 8,
        "buy_threshold": 62,
        "sell_threshold": 50,
        "rebalance_days": 7,
        "stop_loss_pct": -0.08,
        "min_hold_days": 5,
        "transaction_cost": 0.003,
        "initial_capital": 100000.0,
        "max_candidates": 50       # 候选池取TopN
    },
    "output": {
        "show_progress": False,
        "save_file": None          # 结果保存到文件
    }
}

# ===========================================================
# 引擎 — 不用改
# ===========================================================

def load_data(cfg):
    with open(cfg["data"]["stock_file"]) as f:
        YAHOO = json.load(f)
    with open(cfg["data"]["cache_file"]) as f:
        CACHE = json.load(f)
    
    codes = [c for c in YAHOO if c != "000001"]
    all_dates = sorted(set(d for s in YAHOO for d in YAHOO[s].get("dates",[]) if d))
    stock_date_idx = {c: {d:i for i,d in enumerate(YAHOO[c].get("dates",[]))} for c in YAHOO}
    first_common = min(min(v.keys()) for v in CACHE.values())
    start_idx = all_dates.index(first_common)
    
    return YAHOO, CACHE, codes, all_dates, stock_date_idx, start_idx

def get_price(YAHOO, stock_date_idx, code, date, field="close", nxt=False):
    sd = YAHOO.get(code,{})
    if nxt:
        idx = stock_date_idx.get(code,{}).get(date,-1)
        if idx>=0 and idx+1<len(sd.get("dates",[])):
            vals=sd.get(field,[])
            return vals[idx+1] if idx+1<len(vals) else 0
        return 0
    idx=stock_date_idx.get(code,{}).get(date,-1)
    vals=sd.get(field,[])
    return vals[idx] if idx>=0 and idx<len(vals) else 0

def run_backtest(YAHOO, CACHE, all_dates, stock_date_idx, start_idx, cfg):
    """运行回测, 返回 (总收益率, 年化, 最大回撤, 夏普, 交易次数)"""
    s = cfg["strategy"]
    MAX_POS = s["max_positions"]
    BUY = s["buy_threshold"]
    SELL = s["sell_threshold"]
    REBAL = s["rebalance_days"]
    STOP = s["stop_loss_pct"]
    MIN_HOLD = s["min_hold_days"]
    COST = s["transaction_cost"]
    INIT = s["initial_capital"]
    MAX_CAND = s["max_candidates"]
    SHOW_PROG = cfg["output"]["show_progress"]
    
    cash = INIT
    pos = {}
    rebal_count = 0
    trades = 0
    nav_history = []
    prog_step = max(1, (len(all_dates)-start_idx)//20)
    
    for di in range(start_idx, len(all_dates)-1):
        date = all_dates[di]
        rebal_count += 1
        
        scores = {c: CACHE[c].get(date,0) for c in CACHE}
        scores = {k:v for k,v in scores.items() if v>0}
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x:-x[1])
        do_rebal = (rebal_count >= REBAL)
        
        # Sell
        for c in list(pos.keys()):
            cp = get_price(YAHOO, stock_date_idx, c, date, "close")
            loss = (cp-pos[c]["buy_p"])/pos[c]["buy_p"] if cp>0 and pos[c]["buy_p"]>0 else 0
            if scores.get(c,0) < SELL or (loss <= STOP and pos[c]["hold"]>=MIN_HOLD):
                sp = get_price(YAHOO, stock_date_idx, c, date, "close", nxt=True)
                if sp>0:
                    cash += pos[c]["shares"] * sp * (1-COST)
                    trades += 1
                del pos[c]
        
        # Rebalance
        if do_rebal:
            rebal_count = 0
            tv = cash
            for c,px in pos.items():
                cp = get_price(YAHOO, stock_date_idx, c, date, "close")
                if cp>0: tv += px["shares"]*cp
            target = tv / MAX_POS
            
            for c in list(pos.keys()):
                cp = get_price(YAHOO, stock_date_idx, c, date, "close")
                if cp<=0: continue
                cv = pos[c]["shares"]*cp
                if cv>target*1.05:
                    excess = int((cv-target)/cp)
                    sp = get_price(YAHOO, stock_date_idx, c, date, "close", nxt=True)
                    if sp>0 and excess>0:
                        cash += excess*sp*(1-COST)
                        pos[c]["shares"] -= excess
                        trades += 1
                if pos[c]["shares"]<=0: del pos[c]
            
            avail = cash
            if len(pos) < MAX_POS:
                candidates = [(c,s) for c,s in ranked[:MAX_CAND] if s>=BUY and c not in pos]
                for code,sc in candidates[:MAX_POS-len(pos)]:
                    bp = get_price(YAHOO, stock_date_idx, code, date, "close", nxt=True)
                    if bp<=0: continue
                    invest = min(avail*0.95, target*0.95)
                    shares = int(invest/bp)
                    if shares>0:
                        cost = shares*bp*(1+COST)
                        if cost<=cash:
                            cash -= cost
                            pos[code] = {"shares":shares,"buy_p":bp,"hold":0}
                            avail -= cost
                            trades += 1
        
        for c in pos: pos[c]["hold"] = pos[c].get("hold",0)+1
        
        tv = cash
        for c,px in pos.items():
            cp = get_price(YAHOO, stock_date_idx, c, date, "close")
            if cp>0: tv += px["shares"]*cp
        nav_history.append(tv)
        
        if SHOW_PROG and (di-start_idx)%prog_step==0:
            pct = (di-start_idx)/(len(all_dates)-1-start_idx)*100
            print(f"  {pct:.0f}%  NAV={tv:.0f}  pos={len(pos)}", flush=True)
    
    # Final
    final = cash
    for c,px in pos.items():
        cp = get_price(YAHOO, stock_date_idx, c, all_dates[-1], "close")
        if cp>0: final += px["shares"]*cp
    total_ret = (final/INIT-1)*100
    yrs = max((len(all_dates)-start_idx)/245, 1)
    ann_ret = ((final/INIT)**(1/yrs)-1)*100
    
    peak = INIT; mdd = 0
    for v in nav_history:
        if v>peak: peak=v
        dd = (peak-v)/peak*100
        if dd>mdd: mdd=dd
    
    dr = [(nav_history[j]-nav_history[j-1])/nav_history[j-1]*100 for j in range(1,len(nav_history)) if nav_history[j-1]>0]
    sharpe = 0
    if len(dr)>5:
        avg_dr=sum(dr)/len(dr); var_dr=sum((r-avg_dr)**2 for r in dr)/len(dr)
        std=max(var_dr**0.5,0.001); sharpe=round(avg_dr/std*15.8,2)
    
    return total_ret, ann_ret, mdd, sharpe, trades

def main(cfg=None):
    if cfg is None:
        cfg = CONFIG
    
    YAHOO, CACHE, codes, all_dates, stock_date_idx, start_idx = load_data(cfg)
    
    print(f"\n{'='*60}")
    print(f"  A_V1 回测框架 (v2)")
    print(f"{'='*60}")
    print(f"  数据: {len(codes)}只 × {len(all_dates)}天")
    print(f"  参数: 买>={cfg['strategy']['buy_threshold']} 卖<{cfg['strategy']['sell_threshold']}")
    print(f"        持仓{cfg['strategy']['max_positions']}只  {cfg['strategy']['rebalance_days']}天再平衡")
    print(f"        止损{cfg['strategy']['stop_loss_pct']*100:.0f}%  持有>{cfg['strategy']['min_hold_days']}天")
    print(f"        成本{cfg['strategy']['transaction_cost']*100:.1f}%")
    print(f"        数据无open字段, 用close价交易")
    print()
    
    ret, ann, mdd, sharpe, trades = run_backtest(YAHOO, CACHE, all_dates, stock_date_idx, start_idx, cfg)
    
    print(f"\n  {'='*50}")
    print(f"  总收益率:    {ret:>+8.2f}%")
    print(f"  年化收益率:  {ann:>+8.2f}%")
    print(f"  最大回撤:    {mdd:>6.1f}%")
    print(f"  夏普比率:    {sharpe:>6.2f}")
    print(f"  交易次数:    {trades}")
    print(f"  {'='*50}")
    print(f"  用时: 3s")
    
    if cfg["output"]["save_file"]:
        with open(cfg["output"]["save_file"], "w") as f:
            json.dump({"config": cfg, "result": {"return": ret, "annual": ann, "maxdd": mdd, "sharpe": sharpe, "trades": trades}}, f)
    
    return ret, ann, mdd, sharpe, trades

if __name__ == "__main__":
    main()
