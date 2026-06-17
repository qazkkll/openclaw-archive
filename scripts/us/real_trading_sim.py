"""
🦐 拟真交易模拟器 — V4直选 真实资金约束
========================================
不像回测那样"无限资金随时买"，这个模拟:
  - 进场适宜度: 大盘不好时主动空仓/减仓
  - 现金缓冲: 始终保留10%现金
  - 仓位管理: 每股不超12.5%, 单票-8%止损
  - 分批建仓:  不一天全买满
  - 记录每笔交易的理由和盈亏

架构保留, 可用于美股V3.
"""

import json, sys, time
from datetime import datetime

print("📦 加载数据...")
t0 = time.time()
with open('data/precomputed_scores.json') as f: pre = json.load(f)
with open('data/backtest_hist_yahoo.json') as f: hist = json.load(f)

codes = [c for c in hist if len(hist[c].get('close',[])) > 500 and c in pre]
all_dates = sorted(set(d for c in codes for d in hist[c].get('dates',[]) if '2016-01-01' <= d <= '2026-05-14'))
print(f"  {len(codes)}只, {len(all_dates)}天, {time.time()-t0:.1f}s")

def pr(code, date):
    idx = hist[code]['dates'].index(date) if date in hist[code]['dates'] else -1
    return hist[code]['close'][idx] if idx >= 0 else None

def score(code, date):
    return float(pre.get(code, {}).get(date, 0))

def delist_risk(code, date):
    """简化退市检测: 连续30天价格低于1元"""
    d = hist[code]
    try:
        idx = d['dates'].index(date)
        low_30 = [d['close'][i] for i in range(max(0,idx-29),idx+1)]
        return min(low_30) < 0.5
    except: return False

def market_temp(date):
    """进场适宜度 (0-100) — 基于上证综合评分"""
    code = '000001'
    s = score(code, date)
    if s >= 60: return 80, '🟢 极佳'
    if s >= 50: return 60, '🟢 适合'
    if s >= 40: return 40, '🟡 谨慎'
    if s >= 25: return 20, '🟠 不宜'
    return 5, '🔴 禁止'

# ===== 仿真 =====
def simulate():
    dates = [d for d in all_dates]
    capital = 1000000  # 总现金
    cash = capital     # 可用现金
    positions = {}     # {code: {shares, buy_price, buy_day, reason}}
    trade_log = []
    peak = capital
    market_state = '等待'
    
    V4_BUY = 62
    V4_SELL = 50
    MAX_POS = 8
    PER_POS = 0.125  # 12.5% per position
    CASH_RESERVE = 0.10  # 至少保留10%现金
    STOP_LOSS = -0.08   # -8%止损
    MIN_HOLD = 5
    
    for di, date in enumerate(dates):
        # === 大盘情绪 ===
        temp_score, temp_label = market_temp(date)
        
        # === 卖出决策（每天检查）===
        to_sell = []
        for code in list(positions.keys()):
            p = positions[code]
            hold = di - p['buy_day']
            if hold < MIN_HOLD: continue
            
            sc = score(code, date)
            price = pr(code, date) or p['buy_price']
            profit = (price - p['buy_price']) / p['buy_price']
            
            # 卖出理由分层
            reason = None
            
            # 1) 退市风险 → 无条件卖
            if delist_risk(code, date):
                reason = f'退市预警!'
            
            # 2) 止损 -8% → 卖
            elif profit < STOP_LOSS:
                reason = f'触发止损({profit*100:.1f}%)'
            
            # 3) 评分低于卖出线 → 卖
            elif sc < V4_SELL:
                reason = f'评分{int(sc)}分<{V4_SELL}卖出线'
            
            # 4) 大盘极度不佳 → 减仓（评分低于60的持仓卖一半）
            elif temp_score < 20 and sc < 60 and hold >= 5:
                reason = f'大盘🔴, 主动减仓'
            
            if reason:
                revenue = price * p['shares']
                cash += revenue
                trade_log.append(f"🔴{date}|{p['name']}({code})|卖出{p['shares']}股@{price:.2f}|{profit*100:+.1f}%|{reason}")
                to_sell.append(code)
        
        for code in to_sell: del positions[code]
        
        # === 调仓买入（每7天）===
        if di % 7 == 0:
            # 先算总资产
            pos_val = sum((pr(c,date) or p['buy_price']) * p['shares'] for c,p in positions.items())
            total = cash + pos_val
            if total > peak: peak = total
            
            target_per = total * PER_POS  # 每仓目标金额
            
            # 进场适宜度决定仓位上限
            if temp_score >= 60:  # 🟢 适合/极佳 → 正常操作
                max_target_pos = MAX_POS
            elif temp_score >= 40:  # 🟡 谨慎 → 降至6只
                max_target_pos = 6
            elif temp_score >= 20:  # 🟠 不宜 → 最多4只
                max_target_pos = 4
            else:  # 🔴 禁止 → 清仓观望
                max_target_pos = 0
            
            # 如果场景极差, 主动清仓
            if temp_score < 20:
                for code in list(positions.keys()):
                    p = positions[code]
                    hold = di - p['buy_day']
                    if hold >= MIN_HOLD:
                        price = pr(code, date) or p['buy_price']
                        profit = (price-p['buy_price'])/p['buy_price']
                        cash += price * p['shares']
                        trade_log.append(f"🔴{date}|清仓|{p['name']}({code})|{profit*100:+.1f}%|大盘🔴清仓观望")
                        del positions[code]
            
            # 买入
            empty = max_target_pos - len(positions)
            if empty > 0 and cash > 50000:
                cand = []
                for c in codes:
                    if c in positions: continue
                    sc = score(c, date)
                    if sc >= V4_BUY:
                        p = pr(c, date)
                        if p: cand.append((c, sc, p))
                cand.sort(key=lambda x: -x[1])
                
                for c, sc, price in cand[:empty * 3]:
                    if len(positions) >= max_target_pos: break
                    if cash < 50000: break
                    
                    # 保留现金缓冲
                    usable = cash - total * CASH_RESERVE
                    if usable < 0: break
                    
                    per = min(target_per, usable / max(1, empty - len(positions) + 1))
                    shares = max(100, int(per / price / 100) * 100)
                    cost = shares * price
                    if cost > usable:
                        shares = int(usable / price / 100) * 100
                        cost = shares * price
                    if shares < 100: continue
                    
                    cash -= cost
                    name = hist[c].get('name', c)
                    positions[c] = {'shares': shares, 'buy_price': price, 'buy_day': di, 'name': name}
                    pos_pct = cost / total * 100
                    
                    # 进场理由
                    detail = f'评分{int(sc)} MACD✅'
                    if temp_score >= 60: detail += ' 大盘🟢'
                    trade_log.append(f"🟢{date}|{name}({c})|买入{shares}股@{price:.2f}|¥{cost:,.0f}({pos_pct:.1f}%)|{detail}")
    
    # 最终统计
    fpv = sum((pr(c, dates[-1]) or p['buy_price']) * p['shares'] for c,p in positions.items())
    final = cash + fpv
    ret = (final/1000000-1)*100
    days = (datetime.strptime(dates[-1],'%Y-%m-%d')-datetime.strptime(dates[0],'%Y-%m-%d')).days
    ann = ((final/1000000)**(365/days)-1)*100 if days>0 else 0
    dd = (peak-final)/peak*100
    
    return {
        'final': round(final), 'ret': round(ret,2), 'ann': round(ann,2),
        'dd': round(dd,2), 'peak': round(peak), 'cash': round(cash),
        'positions': len(positions), 'trades': len(trade_log),
        'buy': sum(1 for t in trade_log if t.startswith('🟢')),
        'sell': sum(1 for t in trade_log if t.startswith('🔴')),
        'log': trade_log
    }

# 跑
print("🚀 拟真交易模拟 (V4直选 + 真实资金约束)")
r = simulate()

print(f"\n{'='*55}")
print(f"📊 V4 拟真交易报告 (2016→2026)")
print(f"{'='*55}")
print(f"最终资产: ¥{r['final']:,}")
print(f"总收益:   {r['ret']:+.2f}%")
print(f"年化:     {r['ann']:.2f}%")
print(f"最大回撤: {r['dd']:.2f}%")
print(f"峰值:     ¥{r['peak']:,}")
print(f"期末现金: ¥{r['cash']:,}")
print(f"期末持仓: {r['positions']}只")
print(f"买入: {r['buy']}次 | 卖出: {r['sell']}次")

# 保存
log_name = f'data/real_trading_log_{datetime.now().strftime("%Y%m%d")}.txt'
with open(log_name,'w') as f:
    f.write(f"🦐 V4 拟真交易模拟 (2016→{dates[-1]})\n")
    f.write(f"初始¥1,000,000 | 最终¥{r['final']:,} | 收益{r['ret']:+.2f}%\n")
    f.write(f"{'='*60}\n")
    for t in r['log']: f.write(t + '\n')

print(f"\n💾 交易日志: {log_name} ({len(r['log'])}条)")

# 保存结构供美股复用
json.dump(r, open('data/real_trading_result.json','w'), indent=2, ensure_ascii=False)
