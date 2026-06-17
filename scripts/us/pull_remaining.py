import json, urllib.request, time, sys, os, multiprocessing as mp
mp.freeze_support()

TOKEN = "***"
URL = "http://api.tushare.pro"

def ts(api, params):
    p = json.dumps({"api_name":api,"token":TOKEN,"params":params}).encode()
    try:
        req = urllib.request.Request(URL, data=p, headers={"Content-Type":"application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())
    except: return None

def pull_day(date):
    r = ts("top_list", {"trade_date": date, "limit": 100})
    if not r: return []
    data = r.get("data")
    if not data: return []
    items = data.get("items",[])
    return [(date, it) for it in items] if items else []

if __name__ == "__main__":
    mp.freeze_support()
    print("Getting 2021-2022 trading days...")
    dates = []
    for y in [2021, 2022]:
        r = ts("trade_cal", {"exchange":"SSE","start_date":f"{y}0101","end_date":f"{y}1231"})
        if r and r.get("data"):
            items=r["data"].get("items",[]); fields=r["data"].get("fields",[])
            if items and fields:
                idx={f:i for i,f in enumerate(fields)}
                trading=[it[fields.index("cal_date")] for it in items if int(it[idx["is_open"]])==1]
                dates.extend(trading)
                print(f"  {y}: {len(trading)} days")
    
    print(f"Pulling {len(dates)} days with 8 workers...")
    with mp.Pool(8) as pool:
        results = pool.map(pull_day, dates)
    
    new_records = []
    for r in results: new_records.extend(r)
    print(f"New: {len(new_records)} records")
    
    # Merge with existing
    with open("D:/data/historical_longhu.json") as f: exist = json.load(f)
    exist_rec = exist.get("records",[])
    all_rec = exist_rec + new_records
    with open("D:/data/historical_longhu.json", "w") as f:
        json.dump({"total": len(all_rec), "records": all_rec}, f)
    print(f"Total: {len(all_rec)} records")
    
    yrs={}
    for r in all_rec: y=r[0][:4]; yrs[y]=yrs.get(y,0)+1
    for y in sorted(yrs): print(f"  {y}: {yrs[y]}")
