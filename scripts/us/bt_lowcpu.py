#!/usr/bin/env python3
"""
⚡ CPU自控 V1评分验证回测 v3

方案:
1. 不自降优先级(nice) — 否则Gateway占80%CPU后脚本几乎抢不到时间
2. 改为Python内自控: 每10只股票检查一次CPU负载,
   超过阈值自动sleep更长时间
3. 逐只处理,不用ALL_DATA重建
4. 先预评分 → 缓存,再用缓存快速回测

用法: python3 scripts/bt_lowcpu.py
"""
import json, os, sys, time, math, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import v1_score_from_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = f'{ROOT}/data/v1_scores_cache.json'
SAMPLE_SIZE = 100
BATCH_SIZE = 20
MIN_PRICE = 2.0

DUTY_CYCLE = 1.5  # 每只股票后sleep(秒),硬限CPU在~50%

def cpu_ok(force_sleep=False):
    """CPU占空比控制: force_sleep=true强制放松"""
    if force_sleep:
        time.sleep(DUTY_CYCLE + 0.5)
        return
    # 轻量负载检查
    try:
        load = float(open('/proc/loadavg').read().split()[0])
        ncpu = os.cpu_count() or 2
        if load > ncpu * 0.7:
            time.sleep(min(3, load))
    except:
        pass

t0 = time.time()
print('⚡ CPU自控 V1验证 v3', flush=True)
print(f'⏰ {time.strftime("%H:%M")}', flush=True)

# 加载数据
print('📂 加载数据...', flush=True)
with open(f'{ROOT}/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)

all_codes = [c for c in YAHOO if isinstance(YAHOO[c], dict) and 
             len(YAHOO[c].get('close', [])) >= 500]
random.seed(42)
sample = random.sample(all_codes, min(SAMPLE_SIZE, len(all_codes)))
print(f'📊 {len(all_codes)}只合格 → 取样{len(sample)}只', flush=True)

# ============================================================
# Step 1: 评分预计算
# ============================================================
print()
print('═══ Step 1: 评分预计算 ═══', flush=True)

SCORE_CACHE = {}
processed = 0
score_elapsed = 0

for bi in range(0, len(sample), BATCH_SIZE):
    batch = sample[bi:bi+BATCH_SIZE]
    cpu_ok()  # 批前检查
    
    for code in batch:
        data = YAHOO[code]
        dates = data.get('dates', [])
        closes = data.get('close', [])
        highs = data.get('high', [])
        lows = data.get('low', [])
        
        if not all([dates, closes]) or closes[-1] < MIN_PRICE:
            processed += 1
            continue
        
        stock_scores = {}
        
        # 评分: 从200日开始
        for di in range(min(200, len(closes)-1), len(dates)):
            c_sub = closes[max(0, di-200):di+1]
            h_sub = highs[max(0, di-200):di+1] if len(highs) > di else None
            l_sub = lows[max(0, di-200):di+1] if len(lows) > di else None
            
            if len(c_sub) >= 60:
                try:
                    s = v1_score_from_data(c_sub, h_sub or c_sub, l_sub or c_sub)
                    if s and s > 0:
                        stock_scores[dates[di]] = round(s, 1)
                except:
                    continue
        
        if len(stock_scores) >= 100:
            SCORE_CACHE[code] = stock_scores
        
        processed += 1
        cpu_ok()  # 每只后放松CPU
        
        # 每5只报告
        if processed % 5 == 0:
            print(f'  📈 {processed}/{len(sample)} | scores:{len(stock_scores)} | ⏱{time.time()-t0:.0f}s', flush=True)
    
    batch_elapsed = time.time() - t0 - score_elapsed
    score_elapsed = time.time() - t0
    print(f'  ✅ 批{bi//BATCH_SIZE+1}/{math.ceil(len(sample)/BATCH_SIZE)} ({batch_elapsed:.0f}s)', flush=True)

print(f'💾 评分缓存: {len(SCORE_CACHE)}只有效评分', flush=True)

if len(SCORE_CACHE) >= 5:
    with open(CACHE_FILE, 'w') as f:
        json.dump(SCORE_CACHE, f)
    print(f'  已保存至 {CACHE_FILE}', flush=True)

# ============================================================
# Step 2: 回测
# ============================================================
print()
print('═══ Step 2: 回测 ═══', flush=True)

# 收集所有日期的并集
all_dates = sorted(set(d for c in SCORE_CACHE for d in SCORE_CACHE[c]))
print(f'📅 {len(all_dates)}个交易日覆盖', flush=True)

def run_bt(buy_th, sell_th, max_pos, name):
    cash = 100000.0
    pos = {}
    sell_pnls = []
    trades = {'buy': 0, 'sell': 0}
    
    # 有效起始日期
    first_vals = [list(v.keys())[0] for v in SCORE_CACHE.values() if v]
    if not first_vals:
        return 0, 0, 0, 0, 0
    start = all_dates.index(sorted(first_vals)[0])
    
    for di in range(start, len(all_dates) - 1):
        date = all_dates[di]
        ndate = all_dates[di+1]
        
        # 当天评分
        scores = {}
        for code in SCORE_CACHE:
            s = SCORE_CACHE[code].get(date, 0)
            if s >= sell_th - 10:  # 稍宽过滤
                scores[code] = s
        
        if not scores:
            continue
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        # 卖出
        for c in list(pos.keys()):
            if scores.get(c, 0) < sell_th:
                p = pos.pop(c)
                nd = YAHOO.get(c, {})
                ndates = nd.get('dates', [])
                if ndate in ndates:
                    ix = ndates.index(ndate)
                    sp = nd.get('close', [0])[ix]
                    if sp > 0:
                        cash += p['s'] * sp * 0.997
                        pnl = (sp / p['bp'] - 1) * 100
                        sell_pnls.append(pnl)
                        trades['sell'] += 1
        
        # 买入
        if len(pos) < max_pos:
            slots = max_pos - len(pos)
            cand = [(c, s) for c, s in ranked[:20] if s >= buy_th and c not in pos]
            for code, sc in cand[:slots]:
                nd = YAHOO.get(code, {})
                ndates = nd.get('dates', [])
                if ndate in ndates:
                    ix = ndates.index(ndate)
                    bp = nd.get('close', [0])[ix]
                    if bp <= 0: continue
                    w = 1.0 / max_pos
                    invest = cash * w * 0.95
                    shares = invest / bp
                    if shares > 0:
                        cash -= invest
                        pos[code] = {'s': shares, 'bp': bp}
                        trades['buy'] += 1
        
        # 定期CPU检查
        if (di - start) % 200 == 0:
            cpu_ok()
    
    # 最终估值
    final = cash
    for c, p in pos.items():
        nd = YAHOO.get(c, {})
        ndates = nd.get('dates', [])
        ld = all_dates[-1]
        if ld in ndates:
            fp = nd.get('close', [0])[ndates.index(ld)]
            if fp > 0:
                final += p['s'] * fp
    
    ret = (final / 100000 - 1) * 100
    years = max(1, (len(all_dates) - start) / 245)
    ann = ((final/100000)**(1/years)-1)*100
    
    wins = sum(1 for pnl in sell_pnls if pnl > 0)
    wr = (wins / max(len(sell_pnls), 1)) * 100
    avg_pnl = sum(sell_pnls) / max(len(sell_pnls), 1) if sell_pnls else 0
    
    return ret, ann, wr, avg_pnl, trades['buy'] + trades['sell']

# 参数组合
params = [
    (62, 50, 8, 'V1_62/50'),
    (55, 40, 8, 'V1_55/40'),
    (65, 45, 6, 'V1_65/45'),
    (60, 45, 8, 'V1_60/45'),
    (58, 40, 6, 'V1_58/40'),
    (62, 35, 8, 'V1_62/35'),
    (50, 30, 8, 'V1_50/30'),
    (70, 50, 5, 'V1_70/50'),
]

print(f'{"参数":<20} {"回报":>8} {"年化":>8} {"胜率":>6} {"平均盈亏":>7} {"交易":>5}')
print('─' * 58)
for buy, sell, mp, name in params:
    cpu_ok()
    r = run_bt(buy, sell, mp, name)
    print(f'{name:<20} {r[0]:>+7.1f}% {r[1]:>+6.1f}% {r[2]:>5.0f}% {r[3]:>+6.1f}% {r[4]:>5}', flush=True)

# 对照
first_p = YAHOO.get('000001', {}).get('close', [0])[200]
last_p = YAHOO.get('000001', {}).get('close', [0])[-1]
if first_p and last_p:
    bh = (last_p/first_p - 1)*100
    print(f'📊 沪深300买入持有: {bh:+.1f}%')

print(f'\n⏱ 总耗时: {time.time()-t0:.0f}s')
print(f'✅ {time.strftime("%H:%M")}')
