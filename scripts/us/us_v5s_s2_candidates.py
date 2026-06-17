"""
S2 候选池生成：
1. 只筛主板（60/00开头）
2. 净流入 ÷ 前20日均成交额 → 资金冲击比
3. MA20以上 + 成交量正常 + 非ST
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = 'D:\\openclaw-workspace'

with open(WORKSPACE + '/data/a1_daily.json', 'rb') as f:
    raw = json.load(f)
with open(WORKSPACE + '/data/stock_info.json', 'rb') as f:
    info = json.load(f)
with open(WORKSPACE + '/data/a_hist_10y.parquet', 'rb') as f:
    hist = json.load(f)

# Get latest date
date_keys = sorted([k for k in raw if len(k)==8 and k.isdigit()], reverse=True)
latest = date_keys[0]
daily = raw[latest]

def get_ma20_and_vol(code, hist_data):
    d = hist_data.get(code)
    if not d or not d.get('c'): return None, None
    closes = d['c']
    vals = d.get('v', None) or d.get('vol', None)
    if not vals or len(closes) < 20 or len(vals) < 20: return None, None
    ma20 = sum(closes[-20:]) / 20
    cur = closes[-1]
    avg_vol_20 = sum(vals[-20:]) / 20
    cur_vol = vals[-1]
    return ma20, cur_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

candidates = []
for code, d in daily.items():
    if not isinstance(d, dict): continue
    # 只主板
    if not (code.startswith('6') or code.startswith('0')): continue
    # 跳过ST
    inf = info.get(code, {})
    name = inf.get('name', '')
    if 'ST' in name or '退' in name: continue
    
    net = d.get('net_mf', 0) or 0
    bl = d.get('buy_lg', 0) or 0
    bel = d.get('buy_elg', 0) or 0
    sl = d.get('sell_lg', 0) or 0
    sel = d.get('sell_elg', 0) or 0
    total_buy = bl + bel
    total_sell = sl + sel
    total = total_buy + total_sell
    big_ratio = total_buy / total * 100 if total > 0 else 0
    
    # 资金冲击比 = 净流入 / 20日均成交额
    # 先用stock_info里的price作为参考
    # 实际操作需要历史成交量，这里先从hist取
    
    ma20, vol_ratio = get_ma20_and_vol(code, hist)
    if ma20 is None: continue  # 数据不够跳过
    
    # 计算20日均成交额（简单近似：均价*均量）
    d_hist = hist.get(code)
    if d_hist and d_hist.get('c') and d_hist.get('v'):
        closes20 = d_hist['c'][-20:]
        vols20 = d_hist['v'][-20:]
        avg_price_20 = sum(closes20) / 20
        avg_vol_20 = sum(vols20) / 20
        avg_turnover_20 = avg_price_20 * avg_vol_20  # 金额
        
        impact_ratio = net / avg_turnover_20 * 100 if avg_turnover_20 > 0 else 0
    else:
        impact_ratio = 0
    
    # 收盘价
    cur_price = d_hist['c'][-1]
    dev_from_ma20 = (cur_price / ma20 - 1) * 100
    
    # 条件过滤
    above_ma20 = dev_from_ma20 > 0  # MA20上方
    vol_ok = vol_ratio > 0.3  # 成交量不低于30%日均（宽松）
    
    candidates.append({
        'code': code,
        'name': name,
        'industry': inf.get('industry', ''),
        'net': net,
        'big_ratio': big_ratio,
        'impact_ratio': impact_ratio,
        'ma20_dev': dev_from_ma20,
        'vol_ratio': vol_ratio,
        'above_ma20': above_ma20,
        'vol_ok': vol_ok
    })

# 排序方式1：按大单占比（A1原逻辑）
by_big_ratio = sorted(candidates, key=lambda x: -x['big_ratio'])

# 排序方式2：按资金冲击比（净流入标准化的）
by_impact = sorted([c for c in candidates if c['net'] > 0], key=lambda x: -x['impact_ratio'])

# 排序方式3：综合（冲击比+大单占比+趋势）
def composite_score(c):
    s = 0
    s += c['impact_ratio'] * 2  # 冲击比权重
    s += c['big_ratio'] * 0.5   # 大单占比
    if c['above_ma20']: s += 10
    if c['vol_ok']: s += 5
    return s

by_composite = sorted(candidates, key=lambda x: -composite_score(x))

print(f'主板有效标的: {len(candidates)}')
print(f'正净流入: {len([c for c in candidates if c["net"] > 0])}')
print()

# 输出三种排名
print('=== S2候选 A组：大单占比排名（A1原逻辑，仅主板）===')
print(' #   代码  名称       行业       大单比   净流入   MA20偏离  冲击比')
print('-' * 70)
for i, c in enumerate(by_big_ratio[:20]):
    net_s = f'{c["net"]/1e4:.0f}万' if abs(c["net"]) < 1e8 else f'{c["net"]/1e8:.2f}亿'
    ma20_s = f'{c["ma20_dev"]:+.1f}%'
    imp_s = f'{c["impact_ratio"]:.4f}%'
    print(f'{i+1:>2}  {c["code"]:>6}  {c["name"]:<10}{c["industry"]:<10}  {c["big_ratio"]:>5.1f}%  {net_s:>8}  {ma20_s:>7}  {imp_s}')

print()
print('=== S2候选 B组：资金冲击比排名（净流入/日均成交额）===')
print(' #   代码  名称       行业       冲击比    净流入   大单比  MA20偏离')
print('-' * 70)
for i, c in enumerate(by_impact[:20]):
    net_s = f'{c["net"]/1e4:.0f}万' if abs(c["net"]) < 1e8 else f'{c["net"]/1e8:.2f}亿'
    ma20_s = f'{c["ma20_dev"]:+.1f}%'
    imp_s = f'{c["impact_ratio"]:.4f}%'
    print(f'{i+1:>2}  {c["code"]:>6}  {c["name"]:<10}{c["industry"]:<10}  {imp_s:>9}  {net_s:>8}  {c["big_ratio"]:>5.1f}%  {ma20_s:>7}')

print()
print('=== S2候选 C组：综合评分排名（冲击比+大单比+趋势）===')
print(' #   代码  名称       行业       综分  冲击比   净流入   大单比  MA20偏离')
print('-' * 80)
by_composite_sorted = sorted(candidates, key=lambda x: -composite_score(x))
for i, c in enumerate(by_composite_sorted[:20]):
    net_s = f'{c["net"]/1e4:.0f}万' if abs(c["net"]) < 1e8 else f'{c["net"]/1e8:.2f}亿'
    ma20_s = f'{c["ma20_dev"]:+.1f}%'
    imp_s = f'{c["impact_ratio"]:.4f}%'
    score = composite_score(c)
    print(f'{i+1:>2}  {c["code"]:>6}  {c["name"]:<10}{c["industry"]:<10}  {score:>5.1f}  {imp_s:>8}  {net_s:>8}  {c["big_ratio"]:>5.1f}%  {ma20_s:>7}')

# Also save for backtest
with open(WORKSPACE + '/data/us_v5s_s2_candidates_latest.json', 'w', encoding='utf-8') as f:
    json.dump({
        'date': latest,
        'by_big_ratio': by_big_ratio[:50],
        'by_impact': by_impact[:50],
        'by_composite': by_composite_sorted[:50]
    }, f, ensure_ascii=False, default=str)
print(f'\n已保存到 data/s2_candidates_latest.json')

# Print my subjective picks
print()
print('=== 📋 我的S2推选（综合+主观过滤）===')
print('标准：资金冲击比>0 + MA20上方 + 大单比有参考意义')
print(' #   代码  名称       行业       冲击比   净流入  大单比  MA20偏离  vol比')
print('-' * 80)

# Top picks that satisfy all criteria
picks = [c for c in by_composite_sorted 
         if c['net'] > 0 and c['above_ma20'] and c['big_ratio'] > 40]
picks.sort(key=lambda x: -x['impact_ratio'])
for i, c in enumerate(picks[:10]):
    net_s = f'{c["net"]/1e4:.0f}万' if abs(c["net"]) < 1e8 else f'{c["net"]/1e8:.2f}亿'
    imp_s = f'{c["impact_ratio"]:.4f}%'
    print(f'{i+1:>2}  {c["code"]:>6}  {c["name"]:<10}{c["industry"]:<10}  {imp_s:>8}  {net_s:>8}  {c["big_ratio"]:>5.1f}%  {c["ma20_dev"]:>+5.1f}%  {c["vol_ratio"]:>.1f}x')
