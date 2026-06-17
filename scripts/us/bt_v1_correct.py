#!/usr/bin/env python3
"""
V1正确回测引擎 — 无前视偏差
规则: 当天收盘算评分 → 决定次日开盘买卖
数据: backtest_hist_yahoo.json (OHLCV原始数据)
"""
import json, os, sys, time, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print('加载数据...', flush=True)
t0 = time.time()

with open(f'{ROOT}/data/backtest_hist_yahoo.json') as f:
    YAHOO = json.load(f)

# Build structured data: {code: {date: {close, high, low, open}}}
print(' 重建数据结构...', flush=True)
ALL_DATA = {}
for code, item in YAHOO.items():
    if not isinstance(item, dict):
        continue
    dates = item.get('dates', [])
    closes = item.get('close', [])
    highs = item.get('high', [])
    lows = item.get('low', [])
    opens = item.get('open', [])
    
    stock_data = {}
    for i, d in enumerate(dates):
        if i < len(closes):
            stock_data[d] = {
                'close': closes[i],
                'high': highs[i] if i < len(highs) else closes[i],
                'low': lows[i] if i < len(lows) else closes[i],
                'open': opens[i] if i < len(opens) else closes[i],
            }
    if stock_data:
        ALL_DATA[code] = stock_data

# Get all unique dates across all stocks
all_dates_set = set()
for code in ALL_DATA:
    all_dates_set.update(ALL_DATA[code].keys())
all_dates = sorted(all_dates_set)
print(f'  {len(ALL_DATA)}只股票, {len(all_dates)}个交易日, {time.time()-t0:.1f}s', flush=True)

def compute_v1_score(code, date_idx, lookback=120):
    """用截至date_idx的数据算V1评分"""
    code_data = ALL_DATA.get(code)
    if not code_data:
        return 0
    
    date = all_dates[date_idx]
    if date not in code_data:
        return 0
    
    # 获取过去lookback天的数据
    start = max(0, date_idx - lookback)
    closes = []
    highs = []
    lows = []
    
    for i in range(start, date_idx + 1):
        d = all_dates[i]
        if d in code_data:
            closes.append(code_data[d]['close'])
            highs.append(code_data[d]['high'])
            lows.append(code_data[d]['low'])
    
    if len(closes) < 60:
        return 0
    
    price = closes[-1]
    if price <= 0:
        return 0
    
    # === V1评分因子 ===
    
    # 1. MACD门控
    ema12 = 0
    ema26 = 0
    k = 2/13  # EMA12
    k26 = 2/27  # EMA26
    
    for i, c in enumerate(closes):
        if i == 0:
            ema12 = c
            ema26 = c
        else:
            ema12 = c * k + ema12 * (1 - k)
            ema26 = c * k26 + ema26 * (1 - k26)
    
    macd = ema12 - ema26
    signal = macd
    # 简化版: macd为正才给分
    if macd < 0:
        return 0
    
    # 2. 均线系统(25%)
    ma20 = sum(closes[-20:]) / 20
    ma50 = sum(closes[-50:]) / 50
    ma_score = 0
    if price > ma20:
        ma_score += 12.5
    if price > ma50:
        ma_score += 12.5
    
    # 3. RSI(20%)
    delta_sum = 0
    up_sum = 0
    for i in range(max(1, len(closes)-14), len(closes)):
        diff = closes[i] - closes[i-1]
        if diff > 0:
            up_sum += diff
        delta_sum += abs(diff)
    
    rsi = 50
    if delta_sum > 0:
        rsi = 100 * up_sum / delta_sum
    
    rsi_score = 0
    if rsi > 50:
        rsi_score = 20 * min(rsi / 100, 1)
    
    # 4. 52周位置(20%)
    high_52w = max(closes[-252:]) if len(closes) >= 252 else max(closes)
    low_52w = min(closes[-252:]) if len(closes) >= 252 else min(closes)
    
    pos52 = 50
    if high_52w > low_52w:
        pos52 = ((price - low_52w) / (high_52w - low_52w)) * 100
    
    pos_score = 0
    if pos52 > 30 and pos52 < 85:
        pos_score = 20 * (1 - abs(pos52 - 57.5) / 57.5)
    elif pos52 >= 85:
        pos_score = 5
    
    # 5. 动量(35%)
    mom20 = (price / closes[-20] - 1) * 100 if len(closes) >= 20 else 0
    mom60 = (price / closes[-60] - 1) * 100 if len(closes) >= 60 else 0
    
    mom_score = 0
    if mom20 > 0 and mom60 > 0:
        mom_score = min(35, max(0, (mom20 + mom60) / 2))
    
    # 总分
    total = ma_score + rsi_score + pos_score + mom_score
    return min(100, total)

def run_backtest(buy_th=62, sell_th=50, max_pos=8, cost_rate=0.003):
    """完整回测"""
    cash = 100000.0
    positions = {}  # code -> {'shares': N, 'buy_price': P, 'buy_date': D}
    trades = []
    daily_values = []
    
    print(f'  回测 B{buy_th}_S{sell_th}...', end=' ', flush=True)
    t_start = time.time()
    
    # 需要至少200天预热
    start_idx = 200
    total = 0
    
    for di in range(start_idx, len(all_dates) - 1):
        date = all_dates[di]      # 今天
        next_date = all_dates[di + 1]  # 明天(买卖执行日)
        total += 1
        
        # 计算所有股票的今日评分
        scores = {}
        for code in ALL_DATA:
            s = compute_v1_score(code, di)
            if s > 0:
                scores[code] = s
        
        if not scores:
            continue
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        # === 卖出检查 (用今天评分决定明天卖出) ===
        sell_codes = []
        for code in list(positions.keys()):
            current_score = scores.get(code, 0)
            if current_score < sell_th:
                sell_codes.append(code)
        
        # 执行卖出(明天开盘价)
        for code in sell_codes:
            pos = positions.pop(code)
            sell_data = ALL_DATA.get(code, {}).get(next_date)
            if sell_data:
                sell_price = sell_data['open']
                proceeds = pos['shares'] * sell_price * (1 - cost_rate)
                cash += proceeds
                pnl = (sell_price / pos['buy_price'] - 1) * 100
                trades.append({
                    'date': next_date, 'code': code, 'type': 'sell',
                    'pnl_pct': pnl, 'hold_days': di - pos.get('buy_idx', di)
                })
        
        # === 买入检查 ===
        if len(positions) < max_pos:
            candidates = [(c, s) for c, s in ranked[:30] if s >= buy_th and c not in positions]
            slots = max_pos - len(positions)
            
            for code, score in candidates[:slots]:
                buy_data = ALL_DATA.get(code, {}).get(next_date)
                if not buy_data:
                    continue
                
                buy_price = buy_data['open']
                if buy_price <= 0:
                    continue
                
                weight = 1.0 / max_pos
                invest = cash * weight * 0.95  # 留5%现金
                shares = invest / buy_price
                
                if shares > 0:
                    cash -= invest
                    positions[code] = {
                        'shares': shares,
                        'buy_price': buy_price,
                        'buy_date': next_date,
                        'buy_idx': di + 1,
                        'entry_score': score
                    }
                    trades.append({
                        'date': next_date, 'code': code, 'type': 'buy',
                        'price': buy_price, 'score': score, 'value': invest
                    })
        
        # 每20天记录价值
        if total % 20 == 0:
            val = cash
            for c, p in positions.items():
                pd = ALL_DATA.get(c, {}).get(next_date)
                if pd:
                    val += p['shares'] * pd['open']
            daily_values.append({'date': next_date, 'value': val})
    
    # 最终估值
    final_date = all_dates[-1]
    final_val = cash
    for c, p in positions.items():
        pd = ALL_DATA.get(c, {}).get(final_date)
        if pd:
            final_val += p['shares'] * pd['close']
    
    ret = (final_val / 100000 - 1) * 100
    years = (len(all_dates) - start_idx) / 245
    ann = ((final_val / 100000) ** (1 / max(years, 1)) - 1) * 100 if final_val > 0 else -100
    
    # 最大回撤
    peak = 100000
    max_dd = 0
    for dv in daily_values:
        if dv['value'] > peak:
            peak = dv['value']
        dd = (dv['value'] / peak - 1) * 100
        max_dd = min(max_dd, dd)
    
    # 胜率
    sell_trades = [t for t in trades if t['type'] == 'sell']
    wins = sum(1 for t in sell_trades if t.get('pnl_pct', 0) > 0)
    win_rate = wins / max(len(sell_trades), 1) * 100
    
    elapsed = time.time() - t_start
    print(f'回报{ret:+.1f}% 年化{ann:+.1f}% 回撤{max_dd:.1f}% 胜率{win_rate:.0f}% 交易{len(trades)} {elapsed:.0f}s', flush=True)
    
    return {
        'return': ret, 'annualized': ann, 'max_dd': max_dd,
        'win_rate': win_rate, 'trades': len(trades), 'final': final_val
    }

print()
print('🏁 V1正确回测（无前视偏差，次日开盘买卖）')
print('═' * 60)
print()

# 基准测试
results = []
results.append(('V1基准 62/50', run_backtest(62, 50)))

print()
print('📊 汇总')
print('─' * 60)
print(f'{"策略":<20} {"回报":>8} {"年化":>8} {"回撤":>8} {"胜率":>6} {"交易":>6}')
print('─' * 60)
for name, r in results:
    print(f'{name:<20} {r["return"]:>+7.1f}% {r["annualized"]:>+6.1f}% {r["max_dd"]:>6.1f}% {r["win_rate"]:>5.0f}% {r["trades"]:>6}')

# 对比买入持有
print()
print('对比: 买入持有沪深300')
first_p = ALL_DATA.get('000001', {}).get(all_dates[start_idx], {}).get('close', 0)
last_p = ALL_DATA.get('000001', {}).get(all_dates[-1], {}).get('close', 0)
if first_p and last_p:
    bh = (last_p/first_p - 1) * 100
    print(f'  买入持有: {bh:+.1f}%')

print(f'\n总耗时: {time.time()-t0:.0f}s')
print(f'完成时间: {time.strftime("%H:%M")}')
