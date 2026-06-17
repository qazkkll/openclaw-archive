"""
Step 2: 绿箭V7.5 交易模拟引擎
- 逐日买入评分>=85的前5只（未持仓）
- 真实价格跟踪
- 止损-10% / 单日急涨>=30%卖出 / T+4到期
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')
from collections import defaultdict

t0 = time.time()

BUY_SCORE = 85
MAX_PER_DAY = 5
AMOUNT = 1000
HOLD_DAYS = 4
STOP_LOSS = -0.10
TRIGGER_UP = 0.30
WIN_SMALL = 0.30
WIN_BIG = 1.00

# 1. 加载Step 1缓存的评分数据（如果缓存存在）
print('Step 2: 交易模拟引擎', flush=True)
print('='*50, flush=True)

cache_path = '_green_arrow_cache.json'
if os.path.exists(cache_path):
    print(f'加载缓存...', flush=True)
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    may_dates = cache['dates']
    all_day_scores = cache['scores']
    price_db = cache['prices']
    print(f'缓存加载完成: {len(may_dates)}天', flush=True)
else:
    print('缓存不存在，需要先运行 Step 1', flush=True)
    sys.exit(1)

print(f'日期: {may_dates[0]} ~ {may_dates[-1]}', flush=True)
print(f'有价格的股票: {len(price_db)}', flush=True)

# ======== Portfolio引擎 ========
class Portfolio:
    def __init__(self):
        self.holdings = {}
        self.trades = []
        self.prizes = []

    def buy(self, sym, date, price, score):
        hold_idx = may_dates.index(date) + HOLD_DAYS
        hold_until = may_dates[hold_idx] if hold_idx < len(may_dates) else may_dates[-1]

        self.holdings[sym] = {
            'buy_date': date,
            'buy_price': price,
            'hold_until': hold_until,
            'score': score,
            'exit_date': None,
            'exit_reason': None,
            'exit_price': None,
            'prize': None,
            'days_held': 0
        }

        self.trades.append({
            'date': date, 'sym': sym, 'action': 'buy',
            'price': price, 'score': score, 'value': AMOUNT
        })

    def update(self, date):
        """逐日更新持仓，检查退出条件"""
        exitable = []
        for sym, pos in list(self.holdings.items()):
            if pos['exit_date'] is not None:
                continue
            pos['days_held'] += 1

            pmap = price_db.get(sym, {})
            dp = pmap.get(date)
            if not dp:
                continue

            buy_price = pos['buy_price']
            if buy_price <= 0:
                continue

            high, low, close = dp['high'], dp['low'], dp['close']

            # 条件1: 单日急涨>=30% -> 最高价卖出
            if high >= buy_price * (1 + TRIGGER_UP):
                pos['exit_date'] = date
                pos['exit_price'] = high
                pos['exit_reason'] = 'daily_pop'
                ret = (high - buy_price) / buy_price
                pos['prize'] = 'big' if ret >= WIN_BIG else ('small' if ret >= WIN_SMALL else None)
                exitable.append(sym)
                self.trades.append({
                    'date': date, 'sym': sym, 'action': 'sell',
                    'price': high, 'reason': 'daily_pop',
                    'value': AMOUNT * (high / buy_price)
                })
                if pos['prize']:
                    self.prizes.append({'sym': sym, 'date': pos['buy_date'], 'prize': pos['prize'], 'ret': round(ret*100,1)})
                continue

            # 条件2: 跌幅>=10% -> 收盘价卖出
            if low <= buy_price * (1 + STOP_LOSS):
                pos['exit_date'] = date
                pos['exit_price'] = close
                pos['exit_reason'] = 'stop_loss'
                pos['prize'] = None
                exitable.append(sym)
                self.trades.append({
                    'date': date, 'sym': sym, 'action': 'sell',
                    'price': close, 'reason': 'stop_loss',
                    'value': AMOUNT * (close / buy_price)
                })
                continue

            # 条件3: T+4到期
            if date >= pos['hold_until']:
                pos['exit_date'] = date
                pos['exit_price'] = close
                pos['exit_reason'] = 'hold_expiry'
                ret = (close - buy_price) / buy_price
                pos['prize'] = 'big' if ret >= WIN_BIG else ('small' if ret >= WIN_SMALL else None)
                exitable.append(sym)
                self.trades.append({
                    'date': date, 'sym': sym, 'action': 'sell',
                    'price': close, 'reason': f'expiry({ret*100:.1f}%)',
                    'value': AMOUNT * (close / buy_price)
                })
                if pos['prize']:
                    self.prizes.append({'sym': sym, 'date': pos['buy_date'], 'prize': pos['prize'], 'ret': round(ret*100,1)})
                continue

            # 未退出：检查中奖状态
            ret = (close - buy_price) / buy_price
            if ret >= WIN_BIG and pos['prize'] is None:
                pos['prize'] = 'big'
            elif ret >= WIN_SMALL and pos['prize'] is None:
                pos['prize'] = 'small'

        for sym in exitable:
            if sym in self.holdings:
                del self.holdings[sym]

    def is_holding(self, sym):
        return sym in self.holdings and self.holdings[sym]['exit_date'] is None

    def summary(self):
        total_value = sum(t['value'] for t in self.trades if t['action'] == 'sell')
        total_cost = sum(t['value'] for t in self.trades if t['action'] == 'buy')
        buys = len([t for t in self.trades if t['action'] == 'buy'])
        sells = len([t for t in self.trades if t['action'] == 'sell'])
        profit = total_value - total_cost
        roi = profit / total_cost * 100 if total_cost > 0 else 0

        stop_losses = [t for t in self.trades if t.get('reason','') == 'stop_loss']
        daily_pops = [t for t in self.trades if t.get('reason','') == 'daily_pop']
        hold_expiries = [t for t in self.trades if t.get('reason','').startswith('expiry')]
        sim_ends = [t for t in self.trades if t.get('reason','') == 'sim_end']

        prize_small = sum(1 for p in self.prizes if p['prize'] == 'small')
        prize_big = sum(1 for p in self.prizes if p['prize'] == 'big')

        return {
            'buys': buys, 'sells': sells,
            'total_cost': total_cost, 'total_value': total_value,
            'profit': profit, 'roi': roi,
            'prize_small': prize_small, 'prize_big': prize_big,
            'stop_losses': len(stop_losses),
            'daily_pops': len(daily_pops),
            'hold_expiries': len(hold_expiries),
            'sim_ends': len(sim_ends),
            'win_rate': ((prize_small+prize_big)/buys*100) if buys > 0 else 0
        }

# ======== 运行 ========
portfolio = Portfolio()

for i, date in enumerate(may_dates):
    records = all_day_scores.get(date, [])

    # 更新持仓（先更新再买入，当天买入不卖）
    portfolio.update(date)

    if not records:
        continue

    # 筛选>=85的候选
    qualified = [r for r in records if r['score'] >= BUY_SCORE]
    if not qualified:
        continue

    # 去重（跳过已持仓的）
    candidates = [r for r in qualified if not portfolio.is_holding(r['sym'])]
    if not candidates:
        continue

    top5 = candidates[:MAX_PER_DAY]

    buys_made = 0
    for r in top5:
        sym, score = r['sym'], r['score']
        dp = price_db.get(sym, {}).get(date)
        if not dp or dp['close'] <= 0:
            continue
        price = dp['close']
        portfolio.buy(sym, date, price, score)
        buys_made += 1

    # 显示进度
    holding = [s for s, p in portfolio.holdings.items() if p['exit_date'] is None]
    today_sells = len([t for t in portfolio.trades if t['action']=='sell' and t['date']==date])
    print(f'  [{i+1:2d}/{len(may_dates)}] {date}: {buys_made}笔买入, 已平{today_sells}笔, 持仓{len(holding)}只')

# 最后一天强制平仓剩余持仓
print('\n强制平仓剩余持仓...', flush=True)
last_date = may_dates[-1]
force_sell_count = 0
for sym in list(portfolio.holdings.keys()):
    pos = portfolio.holdings[sym]
    if pos['exit_date'] is None:
        dp = price_db.get(sym, {}).get(last_date)
        if dp:
            close = dp['close']
            buy_price = pos['buy_price']
            ret = (close - buy_price) / buy_price if buy_price > 0 else 0
            if ret >= WIN_BIG:
                pos['prize'] = 'big'
            elif ret >= WIN_SMALL:
                pos['prize'] = 'small'
            portfolio.trades.append({
                'date': last_date, 'sym': sym, 'action': 'sell',
                'price': close, 'reason': 'sim_end',
                'value': AMOUNT * (close / buy_price)
            })
            pos['exit_date'] = last_date
            pos['exit_price'] = close
            if pos['prize']:
                portfolio.prizes.append({'sym': sym, 'date': pos['buy_date'], 'prize': pos['prize'], 'ret': round(ret*100,1)})
            force_sell_count += 1

print(f'强制平仓: {force_sell_count}笔', flush=True)

# ======== 输出结果 ========
summary = portfolio.summary()
elapsed = time.time() - t0

print('\n' + '='*55)
print('     绿箭V7.5 精确回测 -- 2026年5月')
print('     (真实价格 + 真实模型评分)')
print('='*55)
print()
print(f'  模拟天数: {len(may_dates)}个交易日')
print(f'  买入笔数: {summary["buys"]}')
print(f'  卖出笔数: {summary["sells"]}')
print(f'  总投资: ${summary["total_cost"]:,.0f}')
print()
print(f'  最终价值: ${summary["total_value"]:,.0f}')
print(f'  净利润: ${summary["profit"]:+,.0f}')
print(f'  收益率: {summary["roi"]:+.1f}%')
print()
print(f'  大奖(+100%+): {summary["prize_big"]}笔')
print(f'  小奖(+30%~99%): {summary["prize_small"]}笔')
print(f'  中奖率: {summary["win_rate"]:.1f}%')
print()
print(f'  止损(-10%): {summary["stop_losses"]}笔')
print(f'  单日急涨: {summary["daily_pops"]}笔')
print(f'  T+4到期: {summary["hold_expiries"]}笔')
print(f'  强制平仓: {summary["sim_ends"]}笔')
print()
print(f'  耗时: {elapsed:.1f}秒')
print()

# 显示获奖明细
if portfolio.prizes:
    print('  --- 获奖明细 ---')
    for p in portfolio.prizes[:20]:
        label = 'BIG' if p['prize']=='big' else 'SMALL'
        print(f'  {label:5s} {p["sym"]:<8s} {p["date"]} +{p["ret"]:.1f}%')
    if len(portfolio.prizes) > 20:
        print(f'  ... 共{len(portfolio.prizes)}笔')

# 显示止损
sl_trades = [t for t in portfolio.trades if t.get('reason','')=='stop_loss']
if sl_trades:
    print('\n  --- 止损明细 ---')
    for t in sl_trades[:10]:
        buy_t = [x for x in portfolio.trades if x['action']=='buy' and x['sym']==t['sym']]
        if buy_t:
            ret = (t['price'] - buy_t[0]['price']) / buy_t[0]['price'] * 100
            print(f'  {t["sym"]:<8s} {t["date"]} ${t["price"]:.2f} ({ret:+.1f}%)')
    if len(sl_trades) > 10:
        print(f'  ... 共{len(sl_trades)}笔')

print('\n' + '='*55)
print('  Step 2 完成')
print('='*55)
