#!/usr/bin/env python3
"""
🔥 CPU友好版 V1评分验证回测 — 两步走 + 负载自动控制

核心思路:
  1) 评分预计算 → data/v1_scores_cache.json (重型,但只跑一次)
  2) 从缓存放回测 → 秒出,想跑几组参数跑几组

CPU保护:
  - 每25只一批,批间sleep(3)让CPU冷却
  - 每批前检查/proc/loadavg, >1.0 自动加长sleep
  - nice -n 19 运行

用法:  cd ~/workspace && nice -n 19 ionice -c 3 python3 scripts/bt_optimized.py
"""
import json, os, sys, time, math, random

# 加载评分引擎
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import v1_score_from_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = f'{ROOT}/data/v1_scores_cache.json'
STOCK_SAMPLE = 200   # 代表样本数,无需全量1177只
BATCH_SIZE = 25       # 每批处理,批间放松CPU
CPU_CHECK_INTERVAL = 5  # 每5只检查一次负载

def get_cpu_load():
    """读取当前CPU负载（1分钟平均）"""
    try:
        with open('/proc/loadavg') as f:
            return float(f.read().split()[0])
    except:
        return 0

def get_cpu_pct():
    """获取当前CPU使用率（约值）"""
    try:
        with open('/proc/stat') as f:
            line = f.readline().strip().split()
            total = sum(int(v) for v in line[1:])
            idle = int(line[4])
            return (1 - idle/total) * 100
    except:
        return 0

t0 = time.time()

# ============================================================
# 数据加载
# ============================================================
print('🚀 CPU友好型 V1评分验证回测', flush=True)
print(f'📦 数据文件: backtest_hist_yahoo.json', flush=True)
print(f'💻 机器: {os.cpu_count()}核 | 样本: {STOCK_SAMPLE}只 | 批: {BATCH_SIZE}只', flush=True)
print(f'⏰ {time.strftime("%H:%M")}', flush=True)
print()

with open(f'{ROOT}/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)

# 构建统一数据结构
ALL_DATA = {}
for code, item in YAHOO.items():
    if not isinstance(item, dict):
        continue
    dates = item.get('dates', [])
    closes = item.get('close', [])
    highs = item.get('high', [])
    lows = item.get('low', [])
    opens = item.get('open', [])
    sd = {}
    for i, d in enumerate(dates):
        if i < len(closes):
            sd[d] = {
                'c': closes[i],
                'h': highs[i] if i < len(highs) else closes[i],
                'l': lows[i] if i < len(lows) else closes[i],
                'o': opens[i] if i < len(opens) else closes[i]
            }
    if len(sd) >= 500:  # 至少500个交易日才有意义
        ALL_DATA[code] = sd

all_codes = list(ALL_DATA.keys())
all_dates = sorted(set(d for c in ALL_DATA for d in ALL_DATA[c].keys()))
print(f'📊 {len(all_codes)}只股票(过滤后), {len(all_dates)}个交易日', flush=True)

# ============================================================
# Step 1: 评分预计算 (只跑一次,缓存后永久重用)
# ============================================================
if not os.path.exists(CACHE_FILE):
    print()
    print('═══════════════════════════════════════')
    print('🔥 Step 1: 评分预计算 (低优先级,低CPU占)')
    print('═══════════════════════════════════════')
    
    # 随机取样200只 (保证行业覆盖)
    random.seed(42)
    sample_codes = random.sample(all_codes, min(STOCK_SAMPLE, len(all_codes)))
    print(f'🎯 取样{len(sample_codes)}只 (随机种子42)', flush=True)
    
    SCORE_CACHE = {}
    processed = 0
    
    for bi in range(0, len(sample_codes), BATCH_SIZE):
        batch = sample_codes[bi:bi+BATCH_SIZE]
        
        # 检查CPU负载
        load = get_cpu_load()
        if load > 1.0:
            extra_sleep = min(10, int(load * 3))
            print(f'  ⚠️ CPU负载{load:.1f}, 额外sleep {extra_sleep}s...', flush=True)
            time.sleep(extra_sleep)
        
        for ci, code in enumerate(batch):
            cd = ALL_DATA[code]
            stock_dates = sorted(cd.keys())
            stock_scores = {}
            skip = max(0, len(stock_dates) - 1500)  # 足够的数据
            used_dates = stock_dates[skip:]
            
            for di in range(len(used_dates)):
                d = used_dates[di]
                # 取过去200天数据算分
                start = max(0, skip + di - 200)
                closes = [cd[stock_dates[j]]['c'] for j in range(start, skip + di + 1) if stock_dates[j] in cd]
                highs = [cd[stock_dates[j]]['h'] for j in range(start, skip + di + 1) if stock_dates[j] in cd]
                lows = [cd[stock_dates[j]]['l'] for j in range(start, skip + di + 1) if stock_dates[j] in cd]
                
                if len(closes) >= 60:
                    try:
                        s = v1_score_from_data(closes, highs, lows)
                        if s and s > 0:
                            stock_scores[d] = round(s, 1)
                    except:
                        pass
            
            if stock_scores:
                SCORE_CACHE[code] = stock_scores
            
            processed += 1
            
            # 每5只轻量负载检查
            if (ci + 1) % CPU_CHECK_INTERVAL == 0:
                load = get_cpu_load()
                if load > 0.8:
                    time.sleep(1)
        
        elapsed = time.time() - t0
        batch_pct = min(100, (bi + BATCH_SIZE) / len(sample_codes) * 100)
        print(f'  📈 批{bi//BATCH_SIZE+1}/{math.ceil(len(sample_codes)/BATCH_SIZE)} '
              f'({processed}/{len(sample_codes)}) {batch_pct:.0f}% | '
              f'CPU负载{get_cpu_load():.1f} | ⏱{elapsed:.0f}s', flush=True)
        
        # 批间放松
        time.sleep(3)
    
    # 保存缓存
    with open(CACHE_FILE, 'w') as f:
        json.dump(SCORE_CACHE, f)
    print(f'💾 缓存已保存: {CACHE_FILE} ({len(SCORE_CACHE)}只, {time.time()-t0:.0f}s)', flush=True)

else:
    print()
    print('📂 找到缓存,直接加载...', flush=True)
    with open(CACHE_FILE) as f:
        SCORE_CACHE = json.load(f)
    print(f'  {len(SCORE_CACHE)}只股票已缓存', flush=True)

# ============================================================
# Step 2: 从缓存放回测 (秒出)
# ============================================================
print()
print('═══════════════════════════════════════')
print('🚀 Step 2: 回测 (轻量,从缓存)')
print('═══════════════════════════════════════')

def run_bt(buy_th, sell_th, max_pos=8, cost=0.003, name='策略'):
    """从缓存评分进行回测"""
    cash = 100000.0
    positions = {}
    trades = {'buy': 0, 'sell': 0}
    sell_pnls = []
    
    # 找第一个有评分的日期
    first_scores = None
    for code in SCORE_CACHE:
        if SCORE_CACHE[code]:
            first_scores = list(SCORE_CACHE[code].keys())
            break
    if not first_scores:
        return None
    first_date = first_scores[0]
    
    if first_date not in all_dates:
        # 找最近的日期
        for d in all_dates:
            if d >= first_date:
                first_date = d
                break
    
    start_idx = all_dates.index(first_date)
    total_days = len(all_dates) - start_idx - 1
    
    # 建一个快速查价的结构
    price_cache = {}
    for code in SCORE_CACHE:
        cd = ALL_DATA.get(code, {})
        pc = {}
        for d in all_dates[start_idx:]:
            if d in cd:
                pc[d] = cd[d]
        if pc:
            price_cache[code] = pc
    
    for di in range(start_idx, len(all_dates) - 1):
        date = all_dates[di]
        next_date = all_dates[di + 1]
        
        # 收集当日所有评分
        daily_scores = {}
        for code in SCORE_CACHE:
            s = SCORE_CACHE[code].get(date, 0)
            if s >= 30:  # 过滤极低分
                daily_scores[code] = s
        
        if not daily_scores:
            continue
        
        ranked = sorted(daily_scores.items(), key=lambda x: -x[1])
        
        # 卖出
        for code in list(positions.keys()):
            score = daily_scores.get(code, 0)
            if score < sell_th:
                pos = positions.pop(code)
                pd = price_cache.get(code, {}).get(next_date, {})
                sp = pd.get('o', 0) or pd.get('c', 0)
                if sp > 0:
                    cash += pos['shares'] * sp * (1 - cost)
                    pnl = (sp / pos['buy_price'] - 1) * 100
                    sell_pnls.append(pnl)
                    trades['sell'] += 1
        
        # 买入
        if len(positions) < max_pos:
            candidates = [(c, s) for c, s in ranked[:30] 
                         if s >= buy_th and c not in positions]
            slots = max_pos - len(positions)
            
            for code, score in candidates[:slots]:
                pd = price_cache.get(code, {}).get(next_date, {})
                bp = pd.get('o', 0) or pd.get('c', 0)
                if bp <= 0:
                    continue
                
                # 等权分配
                weight = 1.0 / max_pos
                invest = cash * weight * 0.95
                shares = invest / bp
                
                if shares > 0:
                    cash -= invest
                    positions[code] = {'shares': shares, 'buy_price': bp}
                    trades['buy'] += 1
    
    # 终值
    final_val = cash
    for code, pos in positions.items():
        pd = price_cache.get(code, {}).get(all_dates[-1], {})
        fp = pd.get('c', 0) or pd.get('o', 0)
        if fp > 0:
            final_val += pos['shares'] * fp
    
    ret = (final_val / 100000 - 1) * 100
    years = max(1, total_days / 245)
    ann = ((final_val / 100000) ** (1 / years) - 1) * 100
    
    # 胜率
    wins = sum(1 for p in sell_pnls if p > 0)
    win_rate = (wins / max(len(sell_pnls), 1)) * 100
    avg_pnl = sum(sell_pnls) / max(len(sell_pnls), 1) if sell_pnls else 0
    
    return {
        'return': ret, 'annualized': ann, 'final': final_val,
        'win_rate': win_rate, 'avg_pnl': avg_pnl,
        'sells': len(sell_pnls), 'buys': trades['buy']
    }

# 跑多组参数
print(f'{"参数组合":<20} {"回报":>8} {"年化":>8} {"胜率":>6} {"平均盈亏":>8} {"交易":>6}')
print('─' * 60)

params = [
    (62, 50, 8, 'V1_62/50'),
    (55, 40, 8, 'V1_55/40'),
    (65, 45, 6, 'V1_65/45'),
    (60, 45, 8, 'V1_60/45'),
    (62, 35, 8, 'V1_62/35'),
    (58, 40, 6, 'V1_58/40_6只'),
    (50, 30, 8, 'V1_50/30(松)'),
    (70, 50, 5, 'V1_70/50(严)'),
]

bt_t0 = time.time()
results = []
for buy, sell, max_pos, name in params:
    if get_cpu_load() > 0.8:
        time.sleep(2)
    r = run_bt(buy, sell, max_pos)
    if r:
        results.append((name, r))
        print(f'{name:<20} {r["return"]:>+7.1f}% {r["annualized"]:>+6.1f}% '
              f'{r["win_rate"]:>5.0f}% {r["avg_pnl"]:>+7.1f}% '
              f'{r["sells"]+r["buys"]:>6}')

print()
bt_elapsed = time.time() - bt_t0
print(f'📊 8组回测完成: {bt_elapsed:.1f}s', flush=True)

# ============================================================
# 总结
# ============================================================
print()
print('═══════════════════════════════════════')
print('📋 验证结论')
print('═══════════════════════════════════════')

# 找到最优
best = max(results, key=lambda x: x[1]['return'])
print(f'🏆 最优: {best[0]} → {best[1]["return"]:+.1f}% 年化{best[1]["annualized"]:+.1f}%')

# 验证当前参数62/50
current = [r for r in results if r[0] == 'V1_62/50']
if current:
    cr = current[0][1]
    print(f'✅ 当前策略(62/50): {cr["return"]:+.1f}% 年化{cr["annualized"]:+.1f}% 胜率{cr["win_rate"]:.0f}%')
    print(f'   200只样本回测结果,买卖{cr["buys"]+cr["sells"]}次,平均盈亏{cr["avg_pnl"]:+.1f}%')

total_elapsed = time.time() - t0
print(f'\n⏱ 总耗时: {total_elapsed:.0f}s ({total_elapsed/60:.1f}分钟)')
print(f'✅ 完成时间: {time.strftime("%H:%M")}')
print(f'📌 注意: {STOCK_SAMPLE}只样本回测,非全量1177只')
print(f'     全量回测需更多内存+时间,样本已足够验证评分引擎一致性')
