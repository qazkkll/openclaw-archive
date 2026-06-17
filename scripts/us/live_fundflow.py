#!/usr/bin/env python3
"""A_V2 资金流评分集成 — 生产中启用"""
import json, os, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

TUSHARE_TOKEN = "ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db"
TUSHARE_URL = "http://api.tushare.pro"
CACHE_FILE = os.path.join(ROOT, "data", "fundflow_cache.json")

def get_moneyflow(ts_code, trade_date):
    """获取个股资金流"""
    payload = json.dumps({
        "api_name": "moneyflow",
        "token": TUSHARE_TOKEN,
        "params": {"ts_code": ts_code, "start_date": trade_date, "end_date": trade_date}
    }).encode()
    try:
        req = urllib.request.Request(TUSHARE_URL, data=payload, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        items = data.get("data", {}).get("items", [])
        if items:
            fields = data["data"]["fields"]
            idx = {f: i for i, f in enumerate(fields)}
            item = items[0]
            lg_buy = float(item[idx["buy_lg_amount"]])
            lg_sell = float(item[idx["sell_lg_amount"]])
            elg_buy = float(item[idx["buy_elg_amount"]])
            elg_sell = float(item[idx["sell_elg_amount"]])
            net = lg_buy + elg_buy - lg_sell - elg_sell
            total = lg_buy + elg_buy + lg_sell + elg_sell
            rate = net / total * 100 if total > 0 else 0
            return {"net": net, "rate": rate}
    except:
        return None
    return None

def compute_bonus(fundflow):
    """资金流转评分修正(-5~+5)"""
    if not fundflow:
        return 0
    rate = fundflow["rate"]
    if rate > 20: return 5
    elif rate > 10: return 3
    elif rate > 5: return 2
    elif rate > 0: return 1
    elif rate > -5: return -1
    elif rate > -10: return -2
    elif rate > -20: return -3
    else: return -5

def score_with_fundflow(v1_score, code, trade_date):
    """V1评分 + 资金流修正"""
    ts_code = f"{code}.SH" if code[:2] in ("60","68") else f"{code}.SZ"
    ff = get_moneyflow(ts_code, trade_date)
    bonus = compute_bonus(ff)
    return min(max(v1_score + bonus, 0), 100), bonus, ff

def batch_top_candidates(candidates, trade_date):
    """对候选股批量获取资金流修正"""
    results = []
    for code, v1_score in candidates:
        adj_score, bonus, ff = score_with_fundflow(v1_score, code, trade_date)
        results.append({
            "code": code, "v1_score": v1_score, "adj_score": adj_score,
            "bonus": bonus, "fundflow": ff
        })
    return sorted(results, key=lambda x: -x["adj_score"])

if __name__ == "__main__":
    # 测试
    test_codes = [("000001", 65.0), ("000002", 70.0)]
    results = batch_top_candidates(test_codes, "20260529")
    for r in results:
        ff_str = f"net={r['fundflow']['net']:.0f} rate={r['fundflow']['rate']:.1f}%" if r['fundflow'] else "N/A"
        print(f"{r['code']}: V1={r['v1_score']} adj={r['adj_score']} bonus={r['bonus']} fundflow={ff_str}")
