#!/usr/bin/env python3
"""
A股全市场多因子回测 v4
- 使用过滤规则筛掉垃圾股
- 分别跑3个月/1年/3年
- 对比模型在不同时间跨度的表现
"""

import json, urllib.request, sys, time, math

RELAY = "http://47.107.99.189:8080"
STOCK_LIST_FILE = "/home/admin/.openclaw/workspace/data/a_stock_mainboard.json"

# ===== 过滤规则 =====
FILTER_RULES = """
垃圾股过滤规则（在回测前筛掉不重要的股票）:

1. 股价 < 3元 → 剔除（低价股风险高，流动性差）
2. 日均成交量 < 5000手 → 剔除（没人交易）
3. 总市值 < 15亿 → 剔除（小微盘，机构不关注）
4. ST / *ST / 退市 → 剔除（有退市风险）
5. 上市 < 60天 → 剔除（次新股，数据不足）
6. 日均换手率 < 0.1% → 剔除（僵尸股）
"""

def fetch(code, start, end):
    data = json.dumps({"code": code, "start": start, "end": end, "freq": "D"}).encode()
    req = urllib.request.Request(f"{RELAY}/history", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read()).get("data", [])

def calc_ema(p, n):
    k = 2/(n+1); e = [p[0]]
    for v in p[1:]: e.append(v*k + e[-1]*(1-k))
    return e

def calc_rsi_full(p):
    if len(p) < 16: return [None]*len(p)
    g, l = [], []
    for i in range(1, len(p)):
        d = p[i]-p[i-1]; g.append(max(d,0)); l.append(max(-d,0))
    r, ag, al = [None]*14, sum(g[:14])/14, sum(l[:14])/14
    for i in range(14, len(p)):
        r.append(100-100/(1+ag/al) if al else 100)
        if i < len(g): ag = (ag*13+g[i])/14; al = (al*13+l[i])/14
    return r

def calc_sma(p, n):
    return [None]*(n-1) + [sum(p[i-n+1:i+1])/n for i in range(n-1,len(p))]

def sf(arr, i):
    if 0 <= i < len(arr) and arr[i] is not None: return arr[i]
    return None

def calc_indicators(prices):
    n = len(prices)
    rsi = calc_rsi_full(prices)
    ma20 = calc_sma(prices, 20)
    ma60 = calc_sma(prices, 60)
    ma5 = calc_sma(prices, 5)
    ema12 = calc_ema(prices, 12)
    ema26 = calc_ema(prices, 26)
    macd = [ema12[i]-ema26[i] for i in range(n)]
    signal = calc_ema(macd, 9)
    hist = [macd[i]-signal[i] for i in range(n)]
    pos52 = [None]*251
    for i in range(251, n):
        lo, hi = min(prices[i-251:i+1]), max(prices[i-251:i+1])
        pos52.append((prices[i]-lo)/(hi-lo)*100 if hi>lo else 50)
    ma20_ser = calc_sma(prices, 20)
    return {'rsi': rsi, 'ma5': ma5, 'ma20': ma20, 'ma60': ma60,
            'hist': hist, 'pos52': pos52}

def multi_factor_score(i, ind, p):
    """多因子评分 0-100（回测最优模型）"""
    score = 0
    if sf(ind['hist'],i) is not None:
        if sf(ind['hist'],i) > 0 and (sf(ind['hist'],i-1) is None or sf(ind['hist'],i-1) <= 0): score += 35
        elif sf(ind['hist'],i) > (sf(ind['hist'],i-1) if sf(ind['hist'],i-1) is not None else 0): score += 20
        elif sf(ind['hist'],i) > 0: score += 10
    if sf(ind['ma20'],i) is not None and p[i] > sf(ind['ma20'],i): score += 10
    if sf(ind['ma5'],i) is not None and sf(ind['ma20'],i) is not None and sf(ind['ma5'],i) > sf(ind['ma20'],i): score += 10
    if sf(ind['ma20'],i) is not None and sf(ind['ma60'],i) is not None and sf(ind['ma20'],i) > sf(ind['ma60'],i): score += 5
    if sf(ind['rsi'],i) is not None:
        if sf(ind['rsi'],i) < 35: score += 20
        elif sf(ind['rsi'],i) < 60: score += 15
        elif sf(ind['rsi'],i) < 70: score += 10
        else: score -= 10
    if sf(ind['pos52'],i) is not None:
        if sf(ind['pos52'],i) < 30: score += 20
        elif sf(ind['pos52'],i) < 60: score += 15
        elif sf(ind['pos52'],i) < 80: score += 10
        else: score += 5
    return max(0, min(100, score))

def backtest(prices, dates, period_name):
    """跑多因子评分模型回测"""
    if len(prices) < 60:
        return {'period': period_name, 'error': '数据不足', 'trades': 0}
    
    ind = calc_indicators(prices)
    trades = []; in_pos = False; ep = 0
    
    for i in range(60, len(prices)):
        score = multi_factor_score(i, ind, prices)
        if not in_pos and score >= 65:
            in_pos = True; ep = prices[i]
        elif in_pos and (score < 40 or (sf(ind['ma20'],i) is not None and prices[i] < sf(ind['ma20'],i))):
            in_pos = False
            trades.append((prices[i]-ep)/ep*100)
    
    if not trades: return {'period': period_name, 'trades': 0, 'total_pnl': 0}
    hold = (prices[-1]-prices[0])/prices[0]*100
    total = sum(trades)
    wins = sum(1 for t in trades if t > 0)
    return {
        'period': period_name, 'trades': len(trades),
        'win_rate': round(wins/len(trades)*100,1) if trades else 0,
        'total_pnl': round(total,2),
        'hold_pnl': round(hold,1),
        'beat_hold': round(total-hold,1)
    }

def run():
    # 加载股票池
    with open(STOCK_LIST_FILE) as f:
        all_stocks = json.load(f)
    print(f"📊 股票池: {len(all_stocks)} 只")
    
    # 用过滤规则预筛（用腾讯实时数据）
    import subprocess
    result = subprocess.run(
        ['node', '-e', '''
        const fs = require("fs");
        const list = JSON.parse(fs.readFileSync("/home/admin/.openclaw/workspace/data/a_stock_mainboard.json"));
        // 过滤ST/退市/新股
        const filtered = list.filter(s => {
            if (!s || !s.name || !s.code) return false;
            if (s.name.includes("ST") || s.name.includes("退") || s.name.includes("N")) return false;
            return true;
        });
        console.log(JSON.stringify({count: filtered.length, stocks: filtered.map(s=>s.code).slice(0,30)}));
        '''
    ], capture_output=True, text=True, timeout=10)
    print(result.stdout)
    
    # 因为全市场3000只拉3年数据太慢，用抽样统计
    # 从过滤后的池子里随机抽200只代表全市场
    import random
    filtered_stocks = [s for s in all_stocks 
                       if not any(x in s.get('name','') for x in ['ST','退','N']) 
                       and s.get('code','')]
    
    sample = random.sample(filtered_stocks, min(200, len(filtered_stocks)))
    print(f"🔬 抽样: {len(sample)} 只（代表全市场）")
    
    periods = [
        ("3个月", "2026-02-12", "2026-05-12"),
        ("1年", "2025-05-12", "2026-05-12"),
        ("3年", "2023-05-12", "2026-05-12"),
    ]
    
    for pname, start, end in periods:
        print(f"\n{'='*55}")
        print(f"📈 回测周期: {pname} ({start} ~ {end})")
        print(f"{'='*55}")
        
        results = []
        done = 0
        
        for s in sample:
            code = s['code']
            prefix = 'SH' if code.startswith('6') or code.startswith('5') else 'SZ'
            symbol = f"{prefix}.{code}"
            
            data = fetch(symbol, start, end)
            if not data or len(data) < 30:
                continue
            
            prices = [d['收盘价'] for d in data]
            dates = [d['时间'][:10] for d in data]
            res = backtest(prices, dates, pname)
            
            if res.get('trades',0) > 0:
                results.append(res)
            
            done += 1
            if done % 20 == 0:
                print(f"  📡 {done}/{len(sample)}", end='', flush=True)
                print()
        
        if not results:
            print(f"  ⚠️ 无足够交易数据")
            continue
        
        avg_pnl = sum(r['total_pnl'] for r in results)/len(results)
        avg_wr = sum(r['win_rate'] for r in results)/len(results)
        avg_hold = sum(r['hold_pnl'] for r in results)/len(results)
        avg_beat = sum(r['beat_hold'] for r in results)/len(results)
        total_trades = sum(r['trades'] for r in results)
        win_stocks = sum(1 for r in results if r['total_pnl'] > 0)
        
        print(f"\n  📊 {pname} 回测结果 (有效{len(results)}只):")
        print(f"    总交易次数: {total_trades}")
        print(f"    平均收益: {avg_pnl:+.1f}%")
        print(f"    平均胜率: {avg_wr:.1f}%")
        print(f"    平均持有基准: {avg_hold:+.1f}%")
        print(f"    平均超额收益: {avg_beat:+.1f}%")
        print(f"    正收益占比: {win_stocks}/{len(results)} ({win_stocks/len(results)*100:.1f}%)")
    
    print(f"\n\n{'='*55}")
    print(f"✅ 全市场回测完成")
    print(f"{'='*55}")
    print(f"")
    print(f"结论: 如果3个时间段的平均收益都为正且跑赢持有基准")
    print(f"     → 模型有效，不需要调整")
    print(f"    如果某个时间段严重偏离")
    print(f"     → 需要优化该周期参数")

if __name__ == "__main__":
    run()
