#!/usr/bin/env python3
"""A1今日推荐存档（基于最近可用数据）"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

WORKSPACE = "/home/hermes/.hermes/openclaw-archive"
now_ts = "2026-06-08T16:01:00+08:00"

# 用上周五资金流
import tushare as ts
pro = ts.pro_api()

df = pro.moneyflow(trade_date='20260605')
print(f"资金流数据: {len(df)}条")

with open(os.path.join(WORKSPACE, "data", "stock_info.json"), 'r', encoding='utf-8') as f:
    info = json.load(f)

rows = []
for _, row in df.iterrows():
    code = row['ts_code']
    clean = code.replace('.SH','').replace('.SZ','').replace('.BJ','')
    if not (clean.startswith('60') or clean.startswith('00')):
        continue
    si = info.get(clean, {})
    name = si.get('name', '')
    if 'ST' in name or '*' in name:
        continue
    
    net_mf = row['net_mf_amount']
    buy_lg = row['buy_lg_amount'] + row['buy_elg_amount']
    sell_lg = row['sell_lg_amount'] + row['sell_elg_amount']
    total_vol = (row['buy_sm_amount'] + row['sell_sm_amount'] +
                 row['buy_md_amount'] + row['sell_md_amount'] +
                 row['buy_lg_amount'] + row['sell_lg_amount'] +
                 row['buy_elg_amount'] + row['sell_elg_amount'])
    big_net_ratio = (buy_lg - sell_lg) / total_vol if total_vol > 0 else 0
    score = net_mf / 10000 * 0.4 + max(big_net_ratio, 0) * 0.6
    
    rows.append({
        'code': clean,
        'name': name,
        'industry': si.get('industry', '?'),
        'score': round(score, 4),
        'net_mf': net_mf,
        'big_net_ratio': round(big_net_ratio * 100, 2),
        'buy_lg': round(buy_lg, 0),
        'sell_lg': round(sell_lg, 0),
    })

rows.sort(key=lambda x: -x['score'])

# 取Top10
top10 = rows[:10]
top5 = rows[:5]

print("\n=== A1 资金流模型 1.0 今日推荐 ===")
print(f"数据: 2026-06-05 收盘资金流（今日data未出，16:30后重跑）")
print(f"{'#':<3} {'代码':<7} {'名称':<10} {'行业':<12} {'评分':<8} {'净流入(万)':<12} {'大单净占比%':<8}")
print('-' * 65)
for i, r in enumerate(top10):
    mf_str = f"{r['net_mf']/10000:.1f}亿" if abs(r['net_mf']) >= 10000 else f"{r['net_mf']:.0f}万"
    print(f"{i+1:<3} {r['code']:<7} {r['name']:<10} {r['industry']:<12} {r['score']:<8.4f} {mf_str:<12} {r['big_net_ratio']:<8.2f}")

# 主观评价
print("\n=== 主观评价 ===")
print(f"Top1 {top5[0]['name']}({top5[0]['code']}): "
      f"净流入{top5[0]['net_mf']/10000:.1f}亿, "
      f"大单净{top5[0]['big_net_ratio']:+.2f}%")
print(f"  大单买{top5[0]['buy_lg']:.0f}万 vs 大单卖{top5[0]['sell_lg']:.0f}万")

# 检查大单流向
if top5[0]['big_net_ratio'] < 0:
    print(f"  ⚠️ 评分第一但大单在出，信号纯度低")
elif top5[0]['big_net_ratio'] > 3:
    print(f"  ✅ 大单强推，信号最纯")

# ─── 写入决策历史 ──────────────────────────
record = {
    'type': 'daily_recommendation',
    'time': now_ts,
    'data_source': 'tushare_20260605',
    'is_spot': False,
    'top10': [{'code': r['code'], 'name': r['name'], 'score': r['score'], 'net_mf': r['net_mf']} for r in top10],
    'top5_subjective': [{'code': r['code'], 'name': r['name'], 'judgment': 
        f"信号纯" if r['big_net_ratio'] > 3 else
        f"大单在出" if r['big_net_ratio'] < 0 else
        f"中性"
    } for r in top5],
    'total_candidates': len(rows),
    'sender_id': 'Andi Yang',
    'sender_name': 'Andi Yang',
    'source_channel': 'telegram',
}

with open(os.path.join(WORKSPACE, "data", "decision_history.jsonl"), 'a', encoding='utf-8') as f:
    f.write(json.dumps(record, ensure_ascii=False) + '\n')
print("\n✅ 已写入 decision_history.jsonl")
