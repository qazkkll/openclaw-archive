#!/usr/bin/env python3
"""
🦐 交易实测 v1 — A股 V4直选 逐日仿真
====================================
模拟从指定日期开始，用¥1,000,000做真实交易决策。
每天:
  - 检查持仓评分 → 决定卖出/持有
  - 每7天调仓(或快速轮动时调整)
  - 评分≥62买入, <50卖出
  - 最大8只等权, 最短持5天
  - 输出每笔交易的理由

用法: python3 scripts/trading_simulation.py [起始年份]
示例: python3 scripts/trading_simulation.py 2024
"""

import sys, json, time, math
sys.path.insert(0, '.')
from scripts.score_engine import compute_indicators, v1_score, v1_score_from_data

# ===== 配置 =====
BUY_THRESHOLD = 62
SELL_THRESHOLD = 50
MAX_POSITIONS = 8
MIN_HOLD_DAYS = 5
REBALANCE_INTERVAL = 7
CAPITAL = 1000000

# ===== 加载数据 =====
print("📦 加载数据...")
t0 = time.time()
with open('data/backtest_hist_yahoo.json') as f:
    hist = json.load(f)
print(f"  共 {len(hist)} 只股票, 耗时 {time.time()-t0:.1f}s")

# 建立索引
codes = [c for c in hist if len(hist[c].get('close', [])) > 500]
cdates = {}
for c in codes:
    dates = hist[c].get('dates', [])
    cdates[c] = {dt: i for i, dt in enumerate(dates)} if dates else {}

all_dates = sorted(set(d for c in codes for d in hist[c].get('dates', [])
                       if '2015-01-01' <= d <= '2026-05-14'))
print(f"  候选股: {len(codes)} | 交易日: {len(all_dates)}")

# ===== 评分函数 =====
def score_stock_at(code, target_date):
    """返回(score, macd_pass) 在 target_date 的V1评分"""
    d = hist.get(code)
    if not d: return 0, False
    idx = cdates.get(code, {}).get(target_date)
    if idx is None or idx < 60: return 0, False
    
    close = d['close'][:idx+1]
    high = d.get('high', close)[:idx+1]
    low = d.get('low', close)[:idx+1]
    
    score_obj = v1_score_from_data(close, high, low)
    if not score_obj: return 0, False
    score = score_obj.get('total', 0) if isinstance(score_obj, dict) else score_obj
    macd_pass = (compute_indicators(close, high, low) or {}).get('macdHist', 0) > 0
    return score, macd_pass

# ===== 仿真 =====
def simulate(start_year=2016):
    start_date = f'{start_year}-01-01'
    
    # 过滤日期
    dates = [d for d in all_dates if d >= start_date]
    print(f"\n🚀 交易实测: {start_date} → {dates[-1]}（{len(dates)}个交易日）")
    print(f"💰 初始: ¥{CAPITAL:,} | 策略: V4直选\n")
    
    cash = CAPITAL
    positions = {}  # {code: {shares, buy_price, buy_date, name}}
    trade_log = []
    last_rebalance_day = -REBALANCE_INTERVAL
    
    # 统计
    total_buys = 0
    total_sells = 0
    peak_value = CAPITAL
    
    for day_idx, date in enumerate(dates):
        # 检查现有持仓评分
        to_sell = []
        for code in list(positions.keys()):
            pos = positions[code]
            score, _ = score_stock_at(code, date)
            hold_days = day_idx - pos.get('buy_day', day_idx)
            
            # 当前价格
            idx = cdates.get(code, {}).get(date)
            if idx is None: continue
            price = hist[code]['close'][idx]
            profit_pct = (price - pos['buy_price']) / pos['buy_price'] * 100
            
            should_sell = False
            reason = ''
            
            if score < SELL_THRESHOLD and hold_days >= MIN_HOLD_DAYS:
                should_sell = True
                reason = f'评分{score}低于卖出线{SELL_THRESHOLD}'
            elif profit_pct < -8 and hold_days >= MIN_HOLD_DAYS:
                should_sell = True
                reason = f'触发止损({profit_pct:.1f}%)'
            
            if should_sell:
                revenue = price * pos['shares']
                cash += revenue
                trade_log.append(f"🔴 {date} 卖出 {pos['name']}({code}) "
                               f"{pos['shares']}股@{price:.2f} "
                               f"盈亏{profit_pct:+.2f}% | {reason}")
                to_sell.append(code)
                total_sells += 1
        
        for code in to_sell:
            del positions[code]
        
        # 每7天调仓
        days_since_rebalance = day_idx - last_rebalance_day
        need_rebalance = days_since_rebalance >= REBALANCE_INTERVAL
        
        if need_rebalance:
            last_rebalance_day = day_idx
            
            # 获取全市场评分
            scored = []
            for code in codes:
                s, mp = score_stock_at(code, date)
                if s >= BUY_THRESHOLD and mp:
                    # 获取价格
                    idx = cdates[code].get(date, -1)
                    if idx >= 0:
                        price = hist[code]['close'][idx]
                        name = hist[code].get('name', code)
                        scored.append((code, name, s, price))
            
            scored.sort(key=lambda x: -x[2])
            
            # 买入
            empty_slots = MAX_POSITIONS - len(positions)
            if empty_slots > 0 and scored and cash > 50000:
                per_pos = min(cash * 0.12, cash / empty_slots)
                
                for code, name, cs, price in scored[:empty_slots * 2]:
                    if cash < 10000: break
                    
                    # 忽略已持仓的
                    if code in positions: continue
                    if len(positions) >= MAX_POSITIONS: break
                    
                    shares = max(100, int(per_pos / price / 100) * 100)
                    cost = shares * price
                    
                    if cost > cash:
                        shares = int(cash / price / 100) * 100
                        cost = shares * price
                    
                    if shares >= 100 and cost <= cash:
                        cash -= cost
                        positions[code] = {
                            'shares': shares, 'buy_price': price,
                            'buy_date': date, 'buy_day': day_idx,
                            'name': name
                        }
                        pct = cost / CAPITAL * 100
                        trade_log.append(f"🟢 {date} 买入 {name}({code}) "
                                       f"{shares}股@{price:.2f} "
                                       f"金额¥{cost:,.0f}({pct:.1f}%) "
                                       f"评分{cs}")
                        total_buys += 1
        
        # 计算每日总资产
        pos_value = 0
        for code, pos in positions.items():
            idx = cdates.get(code, {}).get(date)
            if idx is not None:
                pos_value += hist[code]['close'][idx] * pos['shares']
            else:
                pos_value += pos['buy_price'] * pos['shares']
        
        total = cash + pos_value
        if total > peak_value:
            peak_value = total
    
    # ===== 报告 =====
    print(f"\n{'='*60}")
    print(f"📊 交易实测 报告 ({start_date} → {dates[-1]})")
    print(f"{'='*60}")
    
    final_pos_value = 0
    for code, pos in positions.items():
        last_idx = cdates.get(code, {}).get(dates[-1])
        if last_idx is not None:
            price = hist[code]['close'][last_idx]
            val = price * pos['shares']
            final_pos_value += val
            profit = (price - pos['buy_price']) / pos['buy_price'] * 100
            print(f"  持仓 {pos['name']}({code}): {pos['shares']}股 "
                  f"@{price:.2f} ¥{val:,.0f} ({profit:+.2f}%)")
    
    final_total = cash + final_pos_value
    total_return = (final_total / CAPITAL - 1) * 100
    years = (dates[-1] != start_date)
    if years:
        days = (__import__('datetime').datetime.strptime(dates[-1], '%Y-%m-%d')
              - __import__('datetime').datetime.strptime(start_date, '%Y-%m-%d')).days
        ann = ((final_total / CAPITAL) ** (365/days) - 1) * 100 if days > 0 else 0
    else:
        ann = total_return
    
    print(f"\n初始: ¥{CAPITAL:,}")
    print(f"最终: ¥{final_total:,.0f}")
    print(f"收益: {total_return:+.2f}%")
    print(f"年化: {ann:.2f}%")
    print(f"峰值: ¥{peak_value:,.0f}")
    print(f"回撤: {((peak_value - final_total)/peak_value)*100:.2f}%")
    print(f"买入: {total_buys}次 | 卖出: {total_sells}次")
    print(f"末持仓: {len(positions)}只 | 现金: ¥{cash:,.0f}")
    
    # 保存交易日志
    with open('data/trading_simulation_log.txt', 'w') as f:
        f.write(f"🦐 交易实测: {start_date} → {dates[-1]}\n")
        f.write(f"收益: {total_return:+.2f}% | 年化: {ann:.2f}%\n")
        f.write(f"{'='*60}\n")
        for entry in trade_log:
            f.write(entry + '\n')
    
    print(f"\n交易日志: {len(trade_log)}条 → data/trading_simulation_log.txt")

if __name__ == '__main__':
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    simulate(year)
