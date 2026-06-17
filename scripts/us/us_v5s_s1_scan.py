#!/usr/bin/env python3
"""
美股V5-S扫描引擎 — 2026-06-08重构版
===================================
架构：直接对us_hist_clean.parquet全量评分 → V5-S排名 → Top N输出
评分公式：V5-S统一评分（融合原V5-S/M/L）
输出格式：原始分 + 排名(百分位)

不再做硬过滤——us_hist_clean.parquet已经是V5回测过滤后的池（2436只）
"""
import json, sys, os, time
from datetime import datetime, timezone, timedelta
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TZ = timezone(timedelta(hours=8))
WORKSPACE = "/home/hermes/.hermes/openclaw-archive/data"
BASE = WORKSPACE

now = datetime.now(TZ)
now_ts = now.isoformat()

sys.path.insert(0, os.path.join(WORKSPACE, "scripts"))
from us_score_engine import v5s_calc, v5s_score

print("加载数据...", flush=True)
data = json.load(open(f"{BASE}/us_hist_clean.parquet", "r"))
symbols = list(data.keys())
print(f"总池: {len(symbols)}只", flush=True)

results = []
t0 = time.time()

for idx, sym in enumerate(symbols):
    if idx % 500 == 0 and idx > 0:
        el = time.time() - t0
        print(f"  {idx}/{len(symbols)} 有效{len(results)}只 ({el:.0f}s)", flush=True)
    
    c = data[sym].get("c", [])
    h = data[sym].get("h", [])
    l = data[sym].get("l", [])
    if len(c) < 120: continue
    
    try:
        ind = v5s_calc(c, h, l)
        if ind is None: continue
        sc = v5s_score(ind, -1)
        if sc <= 0: continue
        
        results.append({
            "sym": sym,
            "raw": sc,
            "price": c[-1],
            "days": len(c)
        })
    except:
        continue

elapsed = time.time() - t0
print(f"\n评分完毕: {len(results)}只 ({elapsed:.0f}s)", flush=True)

if not results:
    print("\n❌ 有效池为空，无推荐")
    sys.exit(1)

# ─── 排名 ──────────────────────────────────────────────
results.sort(key=lambda x: -x["raw"])
n_total = len(results)
for i, r in enumerate(results):
    r["rank_pct"] = (n_total - i) * 100 / n_total

# ─── Top N 输出 ──────────────────────────────────────
top_n = min(30, n_total)
print(f"\n{'='*75}")
print(f" 📡 美股V5-S扫描 · {now.strftime('%Y-%m-%d %H:%M')}")
print(f" 模型: V5-S统一评分（2026-06-08重构版）")
print(f" 数据: us_hist_clean.parquet ({len(results)}只)")
print(f"{'='*75}")
print(f"{'#':<3} {'代码':<8} {'V5-S':>7} {'排名%':>6} {'价格':>8} {'天数':>5}")
print(f"{'─'*75}")

for i, r in enumerate(results[:top_n]):
    print(f"{i+1:<3} {r['sym']:<8} {r['raw']:>7.1f} {r['rank_pct']:>5.1f}% {r['price']:>8.2f} {r['days']:>5}")

# ─── 主观评价 ──────────────────────────────────────
if len(results) >= 10:
    print(f"\n{'─'*75}")
    print(f"主观判断")
    print(f"{'─'*75}")
    r1 = results[0]
    r10 = results[9]
    print(f"  Top1 {r1['sym']}: V5-S={r1['raw']:.1f}（前{r1['rank_pct']:.1f}%），价格${r1['price']:.2f}")
    print(f"  Top10 {r10['sym']}: V5-S={r10['raw']:.1f}（前{r10['rank_pct']:.1f}%），价格${r10['price']:.2f}")
    delta = (r1["raw"] - r10["raw"]) / r1["raw"] * 100
    print(f"  Top1 vs Top10: 原始分差距 {r1['raw'] - r10['raw']:.1f}分 ({delta:.1f}%)")
    
    # 头部集中度
    top5_avg = sum(r['raw'] for r in results[:5]) / 5
    all_avg = sum(r['raw'] for r in results) / len(results) if results else 0
    print(f"  Top5平均分: {top5_avg:.1f} | 全部平均分: {all_avg:.1f}")
    
    # 检查知名股票的位置
    well_known = ["GNRC","ON","MARA","COHR","IBKR","VLO","FFIV","NOK","PLUG","RIO","RIOT","ASTS",
                  "AMD","META","TSLA","TSM","PLTR","BABA","NIO","GME","COIN","MSTR","F","GM"]
    known_in_pool = [(r['sym'], r['raw'], r['rank_pct']) for r in results if r['sym'] in well_known]
    if known_in_pool:
        print(f"\n  // 知名股票位置 //")
        for sym, sc, rp in sorted(known_in_pool, key=lambda x: -x[1])[:10]:
            print(f"  {sym}: V5-S={sc:.1f}（前{rp:.1f}%）")

# ─── 写入决策历史 ──────────────────────────────────
record = {
    "type": "us_v5s_scan",
    "time": now_ts,
    "model": "V5-S统一评分(2026-06-08重构版)",
    "data_source": "us_hist_clean.parquet",
    "pool_size": len(symbols),
    "active_pool": len(results),
    "top10": [{"sym": r["sym"], "v5s_score": round(r["raw"], 1), "rank_pct": round(r["rank_pct"], 1), "price": round(r["price"], 2)} for r in results[:10]],
    "sender_id": "Andi Yang",
    "sender_name": "Andi Yang",
    "source_channel": "telegram",
}

dec_path = r"/home/hermes/.hermes/openclaw-archive/data\decision_history.jsonl"
try:
    with open(dec_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n✅ 已写入 decision_history.jsonl")
except Exception as e:
    print(f"⚠️ 写入失败: {e}")

print(f"\n总耗时: {time.time()-t0:.0f}s")
