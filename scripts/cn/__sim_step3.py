"""
Step 3: 绿箭V7.5 交易详细分析
- 每个买入的完整追踪（最高涨幅、收益分布）
- 被错过的机会（买了但没拿住的）
- 没买的股票中未来涨超30%的
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

print('Step 3: 详细分析', flush=True)

# 加载缓存
with open('_green_arrow_cache.json', 'r') as f:
    cache = json.load(f)
may_dates = cache['dates']
all_day_scores = cache['scores']
price_db = cache['prices']

print(f'日期: {len(may_dates)}天, 有价格股票: {len(price_db)}', flush=True)

# ======== 重新跑模拟（为了拿完整数据） ========
BUY_SCORE = 85
MAX_PER_DAY = 5
AMOUNT = 1000
HOLD_DAYS = 4
STOP_LOSS = -0.10
TRIGGER_UP = 0.30
WIN_SMALL = 0.30
WIN_BIG = 1.00

# 记录每笔买入的完整追踪
all_buys = []  # {sym, buy_date, buy_price, score, sell_date, sell_price, reason, max_ret, ret}
bid_sell_map = {}

class Analyzer:
    def __init__(self):
        self.holdings = {}
        self.trades = []
        self.all_positions = {}  # sym_buyDate -> full tracking
        
    def buy(self, sym, date, price, score):
        key = f'{sym}_{date}'
        self.holdings[key] = {
            'sym': sym, 'buy_date': date, 'buy_price': price,
            'score': score, 'max_price': price, 'max_ret': 0.0,
            'exit_date': None, 'exit_price': None, 'exit_reason': None,
            'ret': 0.0
        }
        self.trades.append({'date': date, 'sym': sym, 'action': 'buy', 'price': price, 'score': score})
        
    def update(self, date):
        """逐日更新，记录最高价"""
        exits = []
        for key, pos in list(self.holdings.items()):
            if pos['exit_date'] is not None:
                continue
            dp = price_db.get(pos['sym'], {}).get(date)
            if dp is None:
                continue
            high = dp['high']
            # 更新最高价
            if high > pos['max_price']:
                pos['max_price'] = high
                pos['max_ret'] = (high - pos['buy_price']) / pos['buy_price']
            
            low, close = dp['low'], dp['close']
            buy_price = pos['buy_price']
            
            # 条件1: 单日急涨 >=30%
            if high >= buy_price * (1 + TRIGGER_UP):
                pos['exit_date'] = date
                pos['exit_price'] = high
                pos['exit_reason'] = 'daily_pop'
                pos['ret'] = (high - buy_price) / buy_price
                exits.append(key)
                self.trades.append({'date': date, 'sym': pos['sym'], 'action': 'sell', 'price': high, 'reason': 'daily_pop'})
                continue
            
            # 条件2: 止损
            if low <= buy_price * (1 + STOP_LOSS):
                pos['exit_date'] = date
                pos['exit_price'] = close
                pos['exit_reason'] = 'stop_loss'
                pos['ret'] = (close - buy_price) / buy_price
                exits.append(key)
                self.trades.append({'date': date, 'sym': pos['sym'], 'action': 'sell', 'price': close, 'reason': 'stop_loss'})
                continue
            
            # 条件3: T+4到期
            hold_idx = may_dates.index(pos['buy_date']) + HOLD_DAYS
            hold_until = may_dates[hold_idx] if hold_idx < len(may_dates) else may_dates[-1]
            if date >= hold_until:
                pos['exit_date'] = date
                pos['exit_price'] = close
                pos['exit_reason'] = 'hold_expiry'
                pos['ret'] = (close - buy_price) / buy_price
                exits.append(key)
                self.trades.append({'date': date, 'sym': pos['sym'], 'action': 'sell', 'price': close, 'reason': 'hold_expiry'})
                continue
        
        for key in exits:
            if key in self.holdings:
                self.all_positions[key] = self.holdings.pop(key)
    
    def force_sell(self, date):
        for key in list(self.holdings.keys()):
            pos = self.holdings[key]
            if pos['exit_date'] is not None:
                continue
            dp = price_db.get(pos['sym'], {}).get(date)
            if dp:
                close = dp['close']
                pos['exit_date'] = date
                pos['exit_price'] = close
                pos['exit_reason'] = 'sim_end'
                pos['ret'] = (close - pos['buy_price']) / pos['buy_price']
                self.all_positions[key] = self.holdings.pop(key)

    def get_best_price(self, sym, buy_date, sell_date):
        """在买入到卖出期间的最高价（用来判断如果没提前卖能涨多少）"""
        buy_idx = may_dates.index(buy_date)
        sell_idx = may_dates.index(sell_date) if sell_date in may_dates else len(may_dates)-1
        
        pmap = price_db.get(sym, {})
        max_price = 0
        for d in may_dates[buy_idx:sell_idx+1]:
            dp = pmap.get(d)
            if dp and dp['high'] > max_price:
                max_price = dp['high']
        if max_price > 0:
            pos = self.all_positions.get(f'{sym}_{buy_date}')
            if pos:
                return (max_price - pos['buy_price']) / pos['buy_price']
        return 0

    def get_peak_up_to(self, sym, start_date, end_date):
        """从start_date到end_date的最高涨幅（从买入价算）"""
        pos = self.all_positions.get(f'{sym}_{start_date}')
        if not pos:
            return 0
        buy_price = pos['buy_price']
        if buy_price <= 0:
            return 0
        
        start_idx = may_dates.index(start_date)
        end_idx = may_dates.index(end_date) if end_date in may_dates else len(may_dates)-1
        
        pmap = price_db.get(sym, {})
        max_ret = 0
        for d in may_dates[start_idx:end_idx+1]:
            dp = pmap.get(d)
            if dp and dp['high'] > buy_price:
                ret = (dp['high'] - buy_price) / buy_price
                if ret > max_ret:
                    max_ret = ret
        return max_ret

# 运行分析
analyzer = Analyzer()
for date in may_dates:
    records = all_day_scores.get(date, [])
    qualified = [r for r in records if r['score'] >= BUY_SCORE]
    candidates = [r for r in qualified 
                  if f'{r["sym"]}_{date}' not in analyzer.holdings]
    candidates = candidates[:MAX_PER_DAY]
    
    for r in candidates:
        sym, score = r['sym'], r['score']
        dp = price_db.get(sym, {}).get(date)
        if dp and dp['close'] > 0:
            analyzer.buy(sym, date, dp['close'], score)
    
    analyzer.update(date)

analyzer.force_sell(may_dates[-1])

# ======== 分析1: 每笔交易的完整追踪 ========
print('\n=== 全部100笔交易分析 ===', flush=True)
positions = list(analyzer.all_positions.values())
positions.sort(key=lambda x: x['buy_date'])

# 大奖/小奖统计
big_wins = [p for p in positions if p['ret'] >= WIN_BIG]
small_wins = [p for p in positions if p['ret'] >= WIN_SMALL and p['ret'] < WIN_BIG]
breakeven = [p for p in positions if p['ret'] >= 0 and p['ret'] < WIN_SMALL]
losses = [p for p in positions if p['ret'] < 0]

print(f'  大奖(>=+100%): {len(big_wins)}笔')
print(f'  小奖(+30%~99%): {len(small_wins)}笔')
print(f'  打平(0~30%): {len(breakeven)}笔')
print(f'  亏损: {len(losses)}笔')

if big_wins:
    print('\n  大奖名单:')
    for p in sorted(big_wins, key=lambda x: -x['ret']):
        print(f'  {p["sym"]:<8s} {p["buy_date"]} +{p["ret"]*100:.1f}% (max_hit: {p["max_ret"]*100:.1f}%) exit={p["exit_reason"]}')

print('\n  最佳小奖（前5）:')
for p in sorted(small_wins, key=lambda x: -x['ret'])[:5]:
    print(f'  {p["sym"]:<8s} {p["buy_date"]} +{p["ret"]*100:.1f}% (max_hit: {p["max_ret"]*100:.1f}%) exit={p["exit_reason"]}')

# ======== 分析2: 买到的股票最高能涨多少 ========
print('\n=== 买入后全程最高涨幅（含未卖出时的峰值）===', flush=True)
# 对每笔买入，看从买入到月底的最高价
max_potentials = []
for p in positions:
    peak = analyzer.get_peak_up_to(p['sym'], p['buy_date'], may_dates[-1])
    max_potentials.append({'sym': p['sym'], 'buy_date': p['buy_date'], 
                           'buy_price': p['buy_price'], 'actual_ret': p['ret'],
                           'peak_ret': peak*100, 'exit_reason': p['exit_reason']})

# 被止损但后来涨了的
saved_but_would_win = [m for m in max_potentials if m['peak_ret'] >= 30 and m['actual_ret'] < 30]
print(f'  止损/提前卖但后续涨超30%的: {len(saved_but_would_win)}笔')
for m in sorted(saved_but_would_win, key=lambda x: -x['peak_ret'])[:10]:
    print(f'  {m["sym"]:<8s} {m["buy_date"]} 实际:{m["actual_ret"]*100:.1f}% 峰值:{m["peak_ret"]:.1f}% exit={m["exit_reason"]}')

# 错过的30%+收益
missed_30 = [m for m in max_potentials if m['peak_ret'] >= 30 and m['actual_ret'] < 30]
print(f'\n  "错失30%+"的买入（峰值>=30%但实际<30%）：{len(missed_30)}笔')

# ======== 分析3: 没买的股票中那些涨了的 ========
print('\n=== 当天评分>=85但没买到（被前5名挤掉）且后续涨超30%的 ===', flush=True)

all_bought_syms = {p['sym'] for p in positions}
all_bought_keys = {f'{p["sym"]}_{p["buy_date"]}' for p in positions}

missed_opportunities = []  # (sym, date, score, peak_ret到月底)
for date in may_dates:
    records = all_day_scores.get(date, [])
    bought_that_day = {f'{p["sym"]}_{p["buy_date"]}' for p in positions if p['buy_date']==date}
    
    # 当天评分>=85但没买到的
    for r in records:
        sym = r['sym']; score = r['score']
        key = f'{sym}_{date}'
        if key in bought_that_day or key in all_bought_keys:
            continue
        if score < 85:
            continue
        
        # 买入价
        dp = price_db.get(sym, {}).get(date)
        if not dp or dp['close'] <= 0:
            continue
        buy_price = dp['close']
        
        # 从买入日到月底的最高涨幅
        buy_idx = may_dates.index(date)
        max_ret = 0
        pmap = price_db.get(sym, {})
        for d in may_dates[buy_idx:]:
            dp2 = pmap.get(d)
            if dp2 and dp2['high'] > buy_price:
                ret = (dp2['high'] - buy_price) / buy_price
                if ret > max_ret:
                    max_ret = ret
        
        if max_ret >= WIN_SMALL:
            missed_opportunities.append({
                'sym': sym, 'date': date, 'score': score,
                'peak_ret': max_ret*100, 'buy_price': buy_price
            })

print(f'  被遗漏但后续涨超30%的: {len(missed_opportunities)}笔')
if missed_opportunities:
    for m in sorted(missed_opportunities, key=lambda x: -x['peak_ret'])[:15]:
        print(f'  {m["sym"]:<8s} {m["date"]} score={m["score"]:.0f} 峰值+{m["peak_ret"]:.1f}% (成本${m["buy_price"]:.2f})')

# ======== 分析4: 每日候选池对比 ========
print('\n=== 日收益分布（买了的 vs 当天候选全买）===', flush=True)

# 汇总
biggest_win = max(positions, key=lambda x: x['ret'])
print(f'\n  最大收益: {biggest_win["sym"]} +{biggest_win["ret"]*100:.1f}% (买入{biggest_win["buy_date"]}, {biggest_win["exit_reason"]})')

biggest_loss = min(positions, key=lambda x: x['ret'])
print(f'  最大亏损: {biggest_loss["sym"]} {biggest_loss["ret"]*100:.1f}% (买入{biggest_loss["buy_date"]}, {biggest_loss["exit_reason"]})')

biggest_peak = max(max_potentials, key=lambda x: x['peak_ret'])
print(f'  买入后最高峰值: {biggest_peak["sym"]} +{biggest_peak["peak_ret"]:.1f}% (实际+{biggest_peak["actual_ret"]*100:.1f}%)')

# 止损分布
for threshold in [-30, -20, -15, -10, -5, 0]:
    count = sum(1 for p in positions if p['ret']*100 <= threshold)
    print(f'  亏损>{threshold}%: {count}笔')

# 收益分布
for threshold in [30, 50, 75, 100, 150]:
    count = sum(1 for p in positions if p['ret']*100 >= threshold)
    print(f'  收益>{threshold}%: {count}笔')
