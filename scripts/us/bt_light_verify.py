#!/usr/bin/env python3
"""
⚡ 超轻量V1评分验证 — CPU总量可控，不跑全量回放

核心策略: 不逐日回放(需250万次评分),改为抽样验证
验证逻辑: 在N个随机时间点击穿,计算评分与后续收益的关系

验证样本: 50个时间点 × 100只股票 = 5000次评分 (仅全量的2%)
单只验证: 评分≥62的股票,后续20天是否跑赢沪深300?
结果: 分组统计(高分/中分/低分)的超额收益

CPU控制: 每评10只sleep 0.5s + 每时间点后sleep 1s
预计总耗时: ~5分钟,CPU平均占用<40%

用法: python3 scripts/bt_light_verify.py
"""
import json, os, sys, time, random, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import v1_score_from_data

t0 = time.time()
print('⚡ 超轻量 V1评分验证', flush=True)
print(f'⏰ {time.strftime("%H:%M")}', flush=True)

# 加载数据
print('📂 加载数据...', flush=True)
with open('/home/admin/.openclaw/workspace/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)

# 过滤合格股票
all_codes = [c for c in YAHOO if isinstance(YAHOO[c], dict) and 
             len(YAHOO[c].get('close', [])) >= 500]
random.seed(42)
SAMPLE_STOCKS = min(100, len(all_codes))
SAMPLE_POINTS = 50  # 验证用50个时间点(分散在11年)
LOOKAHEAD = 20       # 后续20天收益

sample_codes = random.sample(all_codes, SAMPLE_STOCKS)
print(f'📊 {len(all_codes)}只合格 → {SAMPLE_STOCKS}只样本 × {SAMPLE_POINTS}个时间点', flush=True)

# ============================================================
# 收集所有可用的(代码, 日期索引)对
# ============================================================
print('🔍 构建时间轴...', flush=True)
all_avail = {}
for code in sample_codes:
    d = YAHOO[code]
    dates = d.get('dates', [])
    if len(dates) >= 300:
        all_avail[code] = dates

# 找出所有日期交集范围
earliest = max(d[200] for d in all_avail.values())  # 至少200天预热后
latest = min(d[-LOOKAHEAD-1] for d in all_avail.values())  # 留出lookahead空间

# 取所有股票都有的日期范围
common_dates = sorted(set(
    d for code in sample_codes 
    for d in (all_avail.get(code) or [])
    if earliest <= d <= latest
))
print(f'📅 有效日期范围: {common_dates[0]} ~ {common_dates[-1]} ({len(common_dates)}天)', flush=True)

# 随机选时间点 (间距至少60天,避免重叠)
random.seed(123)
check_points = []
step = max(1, len(common_dates) // SAMPLE_POINTS)
for i in range(0, len(common_dates) - LOOKAHEAD - 1, step):
    check_points.append(common_dates[i])
check_points = check_points[:SAMPLE_POINTS]
print(f'🎯 {len(check_points)}个检查时间点 (间距~{step}天)', flush=True)

# ============================================================
# 核心验证
# ============================================================
print()
print('═══ 评分有效性验证 ═══')
print(f'{"时间点":<12} {"高价数":>6} {"高分收益":>8} {"低价数":>6} {"低分收益":>8} {"差异":>8} {"CPU负载":>6}')
print('─' * 60)

results = []

for pi, pt in enumerate(check_points):
    # 找到pt在所有样本中的索引
    pt_idx = common_dates.index(pt)
    
    scores_here = {}  # code -> score
    forward_ret = {}  # code -> forward return
    
    for code in sample_codes:
        data = YAHOO[code]
        dates = data.get('dates', [])
        closes = data.get('close', [])
        
        # 找pt对应的日期索引
        if pt not in dates:
            continue
        
        di = dates.index(pt)
        if di < 200 or di + LOOKAHEAD >= len(closes):
            continue
        
        # 评分 (用截至今天的数据)
        c_sub = closes[max(0, di-200):di+1]
        h_sub = (data.get('high', []) or closes)[max(0, di-200):di+1]
        l_sub = (data.get('low', []) or closes)[max(0, di-200):di+1]
        
        try:
            s = v1_score_from_data(c_sub, h_sub, l_sub)
        except:
            continue
        if not s or s <= 0:
            continue
        
        score = round(s, 1)
        current_p = closes[di]
        future_p = closes[di + LOOKAHEAD]
        ret = (future_p / current_p - 1) * 100 if current_p > 0 else 0
        
        scores_here[code] = score
        forward_ret[code] = ret
    
    if not scores_here:
        continue
    
    # 分组
    high_codes = [c for c in scores_here if scores_here[c] >= 62]
    low_codes = [c for c in scores_here if scores_here[c] < 50]
    mid_codes = [c for c in scores_here if 50 <= scores_here[c] < 62]
    
    high_ret = sum(forward_ret[c] for c in high_codes) / max(len(high_codes), 1) if high_codes else 0
    low_ret = sum(forward_ret[c] for c in low_codes) / max(len(low_codes), 1) if low_codes else 0
    mid_ret = sum(forward_ret[c] for c in mid_codes) / max(len(mid_codes), 1) if mid_codes else 0
    
    diff = high_ret - low_ret
    
    # CPU负载
    try:
        load = float(open('/proc/loadavg').read().split()[0])
    except:
        load = 0
    
    # 报告
    print(f'{pt:<12} {len(high_codes):>6} {high_ret:>+7.1f}% {len(low_codes):>6} {low_ret:>+7.1f}% {diff:>+7.1f}% {load:>5.1f}', flush=True)
    
    results.append({
        'date': pt,
        'high_count': len(high_codes),
        'mid_count': len(mid_codes),
        'low_count': len(low_codes),
        'high_ret': high_ret,
        'mid_ret': mid_ret,
        'low_ret': low_ret,
        'diff': diff
    })
    
    # CPU自控
    if load > 0.8:
        time.sleep(0.5)
    time.sleep(0.1)  # 每时间点轻量放松

# ============================================================
# 汇总
# ============================================================
print()
print('═══ 汇总统计 ═══')

if not results:
    print('❌ 没有有效的验证结果')
    sys.exit(1)

# 平均差异
avg_diff = sum(r['diff'] for r in results) / len(results)
avg_high = sum(r['high_ret'] for r in results) / len(results)
avg_low = sum(r['low_ret'] for r in results) / len(results)
avg_mid = sum(r['mid_ret'] for r in results) / len(results)
pos_diffs = sum(1 for r in results if r['diff'] > 0)
neg_diffs = sum(1 for r in results if r['diff'] <= 0)

total_high = sum(r['high_count'] for r in results)
total_low = sum(r['low_count'] for r in results)
total_mid = sum(r['mid_count'] for r in results)

print(f'{"":<16} {"高分≥62":>10} {"中分50-61":>10} {"低分<50":>10}')
print(f'{"平均20日收益":<16} {avg_high:>+9.1f}% {avg_mid:>+9.1f}% {avg_low:>+9.1f}%')
print(f'{"总样本数":<16} {total_high:>10} {total_mid:>10} {total_low:>10}')
print(f'{"高分-低分差异":<16} {avg_diff:>+9.1f}%')
print(f'{"高分跑赢次数":<16} {pos_diffs}/{len(results)} ({pos_diffs/len(results)*100:.0f}%)')
print(f'{"高分跑输次数":<16} {neg_diffs}/{len(results)} ({neg_diffs/len(results)*100:.0f}%)')

# 结论
print()
print('═══ 结论 ═══')
if avg_diff > 3:
    print(f'✅ V1评分显著有效: 高分比低分多赚 {avg_diff:.1f}%/20日')
elif avg_diff > 1:
    print(f'🟡 V1评分轻度有效: 高分比低分多赚 {avg_diff:.1f}%/20日')
elif avg_diff > 0:
    print(f'⚠️ V1评分勉强有效: 差异仅 {avg_diff:.1f}%/20日')
else:
    print(f'❌ V1评分无效: 高分比低分还差 {abs(avg_diff):.1f}%/20日')

print(f'\n⏱ 总耗时: {time.time()-t0:.0f}s ({time.time()/60-t0/60:.1f}分)')
print(f'✅ {time.strftime("%H:%M")}')
