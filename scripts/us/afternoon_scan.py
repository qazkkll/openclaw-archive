#!/usr/bin/env python3
"""
🍤 A股下午新推荐 — 14:30跑一次，推送新的Top 10
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from data_source import AShareKline, AShareRealtime
from score_engine import v1_score_from_data
from notify import send

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALITY_POOL = os.path.join(ROOT, 'data', 'quality_pool.json')
kl = AShareKline()
rt = AShareRealtime()

with open(QUALITY_POOL) as f:
    pool = json.load(f)

main_board = [s for s in pool.get('stocks', []) if s.get('tradeable')][:100]

results = []
for s in main_board:
    code = s['code']
    d = kl.get_best(code)
    if not d or len(d) < 60: continue
    close = [x['close'] for x in d]
    score = v1_score_from_data(close, [x['high'] for x in d], [x['low'] for x in d])
    if score is None: continue
    try:
        q = rt.get_quote(code)
        price = q['price'] if q else close[-1]
        chg = q['change_pct'] if q else 0
        name = q['name'] if q else s.get('name', code)
    except:
        price = close[-1]; chg = 0; name = s.get('name', code)
    results.append({'code': code, 'name': name, 'score': round(float(score),0), 'price': price, 'chg': chg})
    time.sleep(0.08)

results.sort(key=lambda x: x['score'], reverse=True)

lines = [f'📊 A股午后推荐 · {time.strftime("%H:%M")}', '']
buyable = [r for r in results if r['score'] >= 62]
watch = [r for r in results if 50 <= r['score'] < 62]
top = (buyable or watch)[:10]

if buyable:
    lines.append(f'🏆 下午新候选 ({len(buyable)}只过买入线)')
else:
    lines.append(f'👀 下午接近买入线 ({len(watch)}只)')
for r in top[:10]:
    sign = '🟢' if r['score'] >= 62 else '🟡'
    lines.append(f'  {sign} {r["name"]} ({r["code"]}): {r["score"]:.0f}分 ¥{r["price"]:.2f} ({r["chg"]:+.2f}%)')

report = '\n'.join(lines)
send(report)
print(report)
