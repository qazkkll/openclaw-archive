#!/usr/bin/env python3
"""看看硬过滤掉了哪些熟悉的股票"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ws = "C:\\Users\\admin\\.openclaw\\workspace"
data = json.load(open(os.path.join(ws, "data", "us_hist_clean.parquet"), "r"))

MIN_PRICE = 3
MIN_AVG_VALUE = 200000
MIN_VOL = 0.05
MIN_DAYS = 500

# 知名股票列表（可能有遗漏，先扫一些常见票）
well_known = [
    "MARA", "RIOT", "COIN", "MSTR",  # 比特币概念
    "AMC", "GME", "BB", "KOSS",  # meme
    "PLTR", "SOFI", "HOOD", "RIVN", "LCID",  # 热门成长
    "AAL", "CCL", "NCLH", "DAL", "UAL",  # 航空邮轮
    "SNAP", "PINS", "DASH", "UBER", "LYFT",  # 互联网
    "F", "GM", "CHWY", "WISH", "AFRM",  # 其他热门
    "NIO", "XPEV", "LI", "BABA", "JD", "BIDU",  # 中概
    "TSLA", "AMZN", "GOOGL", "META", "AAPL", "MSFT", "NVDA",  # 巨无霸（肯定不过滤）
    "AMD", "INTC", "MU", "QCOM", "TSM", "ARM",  # 半导体
    "GME", "AMC", "BBBYQ",  # meme
    "PLUG", "FCEL", "BE", "ENPH", "SEDG",  # 新能源
    "RBLX", "U", "ZM", "DOCU", "NET", "CRWD",  # SaaS
    "SPCE", "ASTR", "RKLB", "ASTS", "GSAT",  # 航天/卫星
    "CGC", "TLRY", "ACB",  # 大麻
    "BYND", "TTD", "SQ", "PYPL", "SHOP",  # 支付/电商
]

# 先查us_scored.json里知道的票
scored = json.load(open(os.path.join(ws, "data", "us_scored.json"), "r"))
scored_syms = set(r["ticker"] for r in scored)

print(f"us_scored.json 含 {len(scored)} 只")
print()

hist_syms = set(data.keys())
both = scored_syms & hist_syms
print(f"同时存在: {len(both)} 只")

# 检查知名票哪些被过滤了
filtered_cnt = 0
for sym in sorted(set(well_known)):
    if sym not in hist_syms:
        print(f"  不在us_hist_clean: {sym}")
        continue
    d = data[sym]
    c = d.get("c", [])
    h = d.get("h", [])
    l = d.get("l", [])
    v = d.get("v", [])
    
    if len(c) < MIN_DAYS:
        print(f"  ❌ {sym}: 数据不足({len(c)}天 < {MIN_DAYS})")
        filtered_cnt += 1
        continue
    
    price = c[-1]
    avg_price_60 = sum(c[-60:]) / 60
    avg_vol_60 = sum(v[-60:]) / 60
    avg_value = avg_price_60 * avg_vol_60
    
    close_60 = c[-60:]
    rets = [(close_60[i]-close_60[i-1])/close_60[i-1] for i in range(1, len(close_60))]
    vol_60 = (sum(r**2 for r in rets)/len(rets)*252)**0.5 if rets else 0
    
    reasons = []
    if price < MIN_PRICE:
        reasons.append(f"price=${price:.2f}<${MIN_PRICE}")
    if avg_value < MIN_AVG_VALUE:
        reasons.append(f"avg_value=${avg_value:.0f}<${MIN_AVG_VALUE}")
    if vol_60 < MIN_VOL:
        reasons.append(f"vol={vol_60*100:.1f}%<{MIN_VOL*100}%")
    
    if reasons:
        print(f"  ❌ {sym}: {'; '.join(reasons)} (价格=${price:.2f}, 日均=${avg_value:.0f}, 波动率={vol_60*100:.1f}%)")
        filtered_cnt += 1
    else:
        print(f"  ✅ {sym}: 通过 (价格=${price:.2f}, 日均=${avg_value:.0f})")

print(f"\n优秀股票过滤总计: {filtered_cnt}只")
