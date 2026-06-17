#!/usr/bin/env python3
"""
🦐 交易实测 v2 — 优化版
========================
先一次性预计算所有股票所有日期的V1评分（5分钟）
再逐日读取预计算数据跑仿真（1分钟）

用法: nohup python3 scripts/trading_simulation_v2.py > logs/sim_run.log 2>&1 &
"""

import sys, json, time, math, os
sys.path.insert(0, '.')
from scripts.score_engine import compute_indicators, v1_score

WORKSPACE = '.'
CAPITAL = 1000000
BUY_THRESHOLD = 62
SELL_THRESHOLD = 50  
MAX_POS = 8
MIN_HOLD = 5
REBALANCE = 7

START_YEAR = 2016

# ===== 1. 加载 =====
print("📦 加载数据...")
t0 = time.time()
with open('data/backtest_hist_yahoo.json') as f:
    hist = json.load(f)
codes = [c for c in hist if len(hist[c].get('close', [])) > 500]
print(f"  {len(codes)}只股票, {time.time()-t0:.1f}s")

# 交易日列表
all_dates = sorted(set(d for c in codes for d in hist[c].get('dates', [])
                       if '2015-01-01' <= d <= '2026-05-14'))
print(f"  {len(all_dates)}个交易日")

# ===== 2. 预计算评分 =====
print(f"\n⚙️ 预计算V1评分（{START_YEAR}年后）...")
t0 = time.time()

cache_file = 'data/precomputed_scores.json'
if os.path.exists(cache_file):
    print("  🔄 使用缓存评分文件...")
    with open(cache_file) as f:
        precomputed = json.load(f)
    print(f"  ✅ 已加载 {len(precomputed)}只的缓存评分")
else:
    precomputed = {}
    processed = 0
    start_date = f'{START_YEAR}-01-01'
    
    for code in codes:
        d = hist[code]
        dates = d['dates']
        close = d['close']
        high = d.get('high', close)
        low = d.get('low', close)
        
        # 预计算全部指标（一次调用）
        ind = compute_indicators(close, high, low)
        if ind is None: continue
        
        # 找起始索引
        si = 0
        for i, dt in enumerate(dates):
            if dt >= start_date and i >= 60:
                si = i
                break
        
        if si == 0: continue
        
        # 逐日评分（从预计算的ind读取）
        scores = {}
        for idx in range(si, len(dates)):
            sc = v1_score(ind, idx)
            scores[dates[idx]] = sc
        
        if scores:
            precomputed[code] = scores
        
        processed += 1
        if processed % 200 == 0:
            print(f"  📊 {processed}/{len(codes)}...", end='\r')
    
    # 保存缓存
    with open(cache_file, 'w') as f:
        json.dump(precomputed, f)
    print(f"\n  ✅ 预计算完成: {len(precomputed)}只, 耗时{time.time()-t0:.0f}s, 已缓存到{cache_file}")

# ===== 3. 仿真（修正版：仅调仓日操作）=====
print(f"\n🚀 仿真: {START_YEAR}年 → {all_dates[-1]}")
t0 = time.time()

dates = [d for d in all_dates if d >= f'{START_YEAR}-01-01']
cash = CAPITAL
positions = {}  # {code: {shares, buy_price, buy_day, name}}
trade_log = []
total_buys = 0
total_sells = 0
peak_val = CAPITAL

for day_idx, date in enumerate(dates):
    # 只在调仓日操作（每7天）
    if day_idx % REBALANCE != 0:
        continue
    
    # ----- 卖出 -----
    to_sell = []
    for code in list(positions.keys()):
        pos = positions[code]
        hold = day_idx - pos['buy_day']
        if hold < MIN_HOLD:
            continue  # 未满最短持有期
        
        score = precomputed.get(code, {}).get(date, 0)
        idx = hist[code]['dates'].index(date) if date in hist[code]['dates'] else -1
        if idx < 0: continue
        price = hist[code]['close'][idx]
        profit = (price - pos['buy_price']) / pos['buy_price'] * 100
        
        should_sell = False
        reason = ''
        
        if score < SELL_THRESHOLD:
            should_sell = True
            reason = f'评分{int(score)}分<{SELL_THRESHOLD}'
        elif profit < -8:
            should_sell = True
            reason = f'止损({profit:.1f}%)'
        
        if should_sell:
            rev = price * pos['shares']
            cash += rev
            trade_log.append(f"🔴{date}|卖出{pos['name']}({code})|{pos['shares']}股@{price:.2f}|{profit:+.1f}%|{reason}")
            to_sell.append(code)
            total_sells += 1
    
    for code in to_sell:
        del positions[code]
    
    # ----- 买入（等权分配）-----
    empty = MAX_POS - len(positions)
    if empty > 0:
        # 计算目标等权金额
        target_total = cash + sum(
            hist[code]['close'][hist[code]['dates'].index(date) if date in hist[code]['dates'] else -1] * p['shares']
            if date in hist[code]['dates'] else p['buy_price'] * p['shares']
            for code, p in positions.items()
        )
        target_per_pos = target_total / MAX_POS
        
        # 候选
        candidates = []
        for code in codes:
            sc = precomputed.get(code, {}).get(date, 0)
            if sc >= BUY_THRESHOLD and code not in positions:
                idx = hist[code]['dates'].index(date) if date in hist[code]['dates'] else -1
                if idx >= 0:
                    candidates.append((code, sc, hist[code]['close'][idx]))
        
        candidates.sort(key=lambda x: -x[1])
        
        for code, sc, price in candidates:
            if len(positions) >= MAX_POS: break
            
            # 等权买入: 目标金额 - 已有金额
            target_buy = max(0, target_per_pos - 0)  # 新仓
            per = min(target_buy, cash * 0.5)  # 单笔不超现金一半
            
            shares = max(100, int(per / price / 100) * 100)
            cost = shares * price
            if cost > cash:
                shares = int(cash / price / 100) * 100
                cost = shares * price
            if shares < 100 or cost <= 0: continue
            
            cash -= cost
            pos_pct = cost / target_total * 100
            positions[code] = {'shares': shares, 'buy_price': price, 'buy_day': day_idx, 'name': hist[code].get('name', code)}
            trade_log.append(f"🟢{date}|买入{hist[code].get('name',code)}({code})|{shares}股@{price:.2f}|¥{cost:,.0f}({pos_pct:.1f}%)|评分{int(sc)}")
            total_buys += 1
    
    # 净值
    pv = sum(
        hist[code]['close'][hist[code]['dates'].index(date)] * p['shares']
        if date in hist[code]['dates'] else p['buy_price'] * p['shares']
        for code, p in positions.items()
    )
    total = cash + pv
    if total > peak_val: peak_val = total# ===== 4. 报告 =====
print(f"\n{'='*60}")
print(f"📊 交易实测 报告")
print(f"{'='*60}")
print(f"期间: {dates[0]} → {dates[-1]} ({len(dates)}天)")

# 最终价值
final_pv = 0
for code, pos in positions.items():
    idx = hist[code]['dates'].index(dates[-1]) if dates[-1] in hist[code]['dates'] else -1
    price = hist[code]['close'][idx] if idx >= 0 else pos['buy_price']
    val = price * pos['shares']
    final_pv += val
    profit = (price - pos['buy_price']) / pos['buy_price'] * 100
    print(f"  持仓 {pos['name']}({code}): {pos['shares']}股 @{price:.2f} ¥{val:,.0f} ({profit:+.2f}%)")

final_v = cash + final_pv
ret = (final_v / CAPITAL - 1) * 100

days = (__import__('datetime').datetime.strptime(dates[-1], '%Y-%m-%d') 
      - __import__('datetime').datetime.strptime(dates[0], '%Y-%m-%d')).days
ann = ((final_v / CAPITAL) ** (365/days) - 1) * 100 if days > 0 else 0
dd = (peak_val - final_v) / peak_val * 100

print(f"\n初始: ¥{CAPITAL:,}")
print(f"最终: ¥{final_v:,.0f}")
print(f"收益: {ret:+.2f}%")
print(f"年化: {ann:.2f}%")
print(f"峰值: ¥{peak_val:,.0f}")
print(f"回撤: {dd:.2f}%")
print(f"买入: {total_buys}次 | 卖出: {total_sells}次")
print(f"末持仓: {len(positions)}只 | 现金: ¥{cash:,.0f}")
print(f"\n⏱️ 仿真耗时: {time.time()-t0:.0f}s")

# 保存
with open('data/trading_simulation_v2_log.txt', 'w') as f:
    f.write(f"🦐 交易实测 v2\n{START_YEAR}年 → {dates[-1]}\n")
    f.write(f"收益: {ret:+.2f}% | 年化: {ann:.2f}% | 回撤: {dd:.2f}%\n")
    f.write(f"{'='*60}\n")
    for e in trade_log:
        f.write(e + '\n')

print(f"\n💾 交易日志: data/trading_simulation_v2_log.txt ({len(trade_log)}条)")
print(f"💾 评分缓存: data/precomputed_scores.json")
print(f"\n{'='*60}")
print(f"  与V4回测对比:")
print(f"  回测算的累计+273.91% | 这边仿真结果应该接近")
print(f"{'='*60}")
