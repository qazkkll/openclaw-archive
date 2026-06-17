#!/usr/bin/env python3
"""每日数据采集: Tushare资金流 + 龙虎榜 (供未来回测用)"""
import json, os, urllib.request, datetime

ROOT = "/home/admin/.openclaw/workspace"
TOKEN = "ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db"
URL = "http://api.tushare.pro"
FLOW_FILE = f"{ROOT}/data/historical_moneyflow.json"
LHB_FILE = f"{ROOT}/data/historical_longhu.json"

def ts(api, params):
    p = json.dumps({"api_name":api,"token":TOKEN,"params":params}).encode()
    r = urllib.request.Request(URL, data=p, headers={"Content-Type":"application/json"})
    d = json.loads(urllib.request.urlopen(r,timeout=30).read())
    return d.get("data",{})

def collect():
    today = datetime.date.today().strftime("%Y%m%d")
    
    # Load quality pool for fund flow tracking
    try:
        with open(f"{ROOT}/data/quality_pool.json") as f:
            pool = json.load(f)
        top_codes = [s["code"] for s in pool.get("stocks",[])[:50]]
    except:
        top_codes = []
    
    if not top_codes:
        print("No pool, skip")
        return
    
    # Pull fund flow for top 50 stocks
    flows = []
    for code in top_codes:
        ts_code = f"{code}.SH" if code[:2] in ("60","68") else f"{code}.SZ"
        mf = ts("moneyflow", {"ts_code":ts_code,"start_date":today,"end_date":today})
        if mf.get("items"):
            flows.append({"code":code,"data":mf["items"][0],"fields":mf["fields"]})
    
    if flows:
        existing = []
        try:
            with open(FLOW_FILE) as f: existing = json.load(f)
        except: pass
        existing.append({"date":today,"flows":flows})
        with open(FLOW_FILE,"w") as f: json.dump(existing, f, ensure_ascii=False)
        print(f"Saved {len(flows)} fund flow records for {today}")
    
    # Pull longhu
    lhb = ts("top_list", {"trade_date":today})
    if lhb.get("items"):
        existing_lhb = []
        try:
            with open(LHB_FILE) as f: existing_lhb = json.load(f)
        except: pass
        existing_lhb.append({"date":today,"data":lhb["items"],"fields":lhb["fields"]})
        with open(LHB_FILE,"w") as f: json.dump(existing_lhb, f, ensure_ascii=False)
        print(f"Saved {len(lhb['items'])} longhu records for {today}")

if __name__ == "__main__":
    collect()
