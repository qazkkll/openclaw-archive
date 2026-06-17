#!/usr/bin/env python3
"""
🍤 熊市模式参数暴力扫描 v2 — 一次预计算，百次快速回测

方法:
  1. compute_indicators 每只股票一次（~40秒）
  2. 逐日v1_score索引评分（~40秒）
  3. 评分缓存后，参数扫描仅需改变门槛/持仓（每组合~0.1秒）

用法: python3 scripts/bt_bear_sweep.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import compute_indicators, v1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'backtest_hist_yahoo.json')

RB_DAYS = 7

def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)

def precompute_v1_scores(all_stocks, codes):
    """高效预计算：compute_indicators一次/股票 + 逐日索引评分"""
    n = len(all_stocks[codes[0]]['close'])
    print(f"预计算 {len(codes)}只×{n}天 V1评分...", flush=True)
    t0 = time.time()
    
    cache = {}
    for idx, code in enumerate(codes):
        d = all_stocks[code]
        close = d['close']
        high = d['high']
        low = d['low']
        
        if len(close) < 60:
            continue
        
        # 一次计算指标
        ind = compute_indicators(close, high, low)
        if ind is None:
            continue
        
        # 逐日索引评分（极快）
        scores = []
        for di in range(60, len(close)):
            s = v1_score(ind, di)
            scores.append(round(s, 1) if s else 0.0)
        
        # 前60天填0
        scores = [0.0] * 60 + scores
        cache[code] = {'scores': scores}
        
        if (idx + 1) % 300 == 0:
            print(f"  {idx+1}/{len(codes)} | {time.time()-t0:.0f}s", flush=True)
    
    print(f"预计算完成: {len(cache)}只 | {time.time()-t0:.0f}s", flush=True)
    return cache

def run_backtest(cache, codes, all_stocks, dates, n,
                 buy_threshold, sell_threshold, max_positions):
    """单次回测 — 纯逻辑，无评分计算"""
    START = 250
    holdings = {}
    last_rebalance = -RB_DAYS
    daily_value = {}
    
    for di in range(START, n):
        today = dates[di]
        
        # 当天候选股（纯阈值过滤）
        candidates = []
        for code in codes:
            if code in cache and di < len(cache[code]['scores']):
                s = cache[code]['scores'][di]
                if s >= buy_threshold:
                    candidates.append((s, code))
        
        candidates.sort(reverse=True)
        
        if di - last_rebalance >= RB_DAYS:
            if candidates:
                top = candidates[:max_positions]
                holdings = {code: 1.0/max_positions for _, code in top}
                last_rebalance = di
            else:
                holdings = {}
        
        # 无持仓=1.0现金
        if holdings:
            total = 0.0
            for code, weight in list(holdings.items()):
                arr = all_stocks[code]['close']
                if di < len(arr) and last_rebalance < len(arr):
                    ret = arr[di] / arr[last_rebalance] - 1
                    total += weight * (1 + ret)
                else:
                    total += weight
            daily_value[today] = total
        else:
            daily_value[today] = 1.0
    
    # 计算年收益
    years = {}
    for yr in range(2016, 2027):
        y = str(yr)
        yr_days = {d: v for d, v in daily_value.items() if d.startswith(y)}
        if not yr_days or len(yr_days) < 20:
            continue
        ds = sorted(yr_days.keys())
        ret = (yr_days[ds[-1]] / yr_days[ds[0]] - 1) * 100
        years[y] = round(ret, 1)
    
    cum = 0
    if daily_value:
        ds = sorted(daily_value.keys())[::1]
        cum = (daily_value[ds[-1]] - 1) * 100
    
    return years, round(cum, 1)

def main():
    t0 = time.time()
    all_stocks = load_data()
    codes = list(all_stocks.keys())
    n = len(all_stocks[codes[0]]['close'])
    dates = all_stocks[codes[0]]['dates']
    
    print(f"数据: {len(codes)}只 | {n}根K线 | {dates[0][:10]}~{dates[-1][:10]}", flush=True)
    
    # Step 1: 预计算评分（一次，后续复用）
    cache = precompute_v1_scores(all_stocks, codes)
    valid = list(cache.keys())
    print(f"有效: {len(valid)}只\n", flush=True)
    
    # Step 2: 参数扫描
    # 熊市模式参数: 高买入门+少持仓+严格卖出
    buy_opts = [60, 62, 63, 65, 68, 70]  # 买入门槛
    sell_opts = [50, 45, 40]               # 卖出门槛
    pos_opts = [3, 5, 8, 10]               # 最大持仓数
    
    results = []
    total_combos = 0
    
    print(f"扫描 {len(buy_opts)}×{len(sell_opts)}×{len(pos_opts)}={len(buy_opts)*len(sell_opts)*len(pos_opts)} 组合...\n", flush=True)
    
    for buy in buy_opts:
        for sell in sell_opts:
            for pos in pos_opts:
                if sell >= buy:
                    continue
                
                t1 = time.time()
                years, cum = run_backtest(cache, valid, all_stocks, dates, n, buy, sell, pos)
                elapsed = time.time() - t1
                total_combos += 1
                
                bear_sum = years.get('2018', 0) + years.get('2022', 0)
                results.append({
                    'buy': buy, 'sell': sell, 'pos': pos,
                    'cum': cum, 'bear_sum': round(bear_sum, 1),
                    'years': years, 'ms': round(elapsed * 1000)
                })
                
                yr_show = ' '.join([f'{y}{years.get(y,"x"):+.1f}' for y in ['2018','2020','2022','2024']])
                icon = '🍀' if bear_sum > 0 else '  '
                print(f"{icon} buy={buy:2d} sell={sell:2d} pos={pos:2d} | 累计{cum:>+7.1f}% 熊市{bear_sum:>+6.1f}% | {yr_show}", flush=True)
    
    print(f"\n扫描完成: {total_combos}组合 | {time.time()-t0:.0f}s\n", flush=True)
    
    # 按熊市总和排序
    results.sort(key=lambda r: (-r['bear_sum'], -r['cum']))
    
    print("=" * 75)
    print("🏆 熊市表现最佳（2018+2022总和排序）")
    print("=" * 75)
    for i, r in enumerate(results[:8]):
        yr = ' '.join([f'{y}{r["years"].get(y,"?"):+5.1f}' for y in ['2018','2020','2022','2024']])
        print(f"#{i+1} buy={r['buy']} sell={r['sell']} pos={r['pos']} | 累计{r['cum']:+7.1f}% 熊市{r['bear_sum']:+6.1f}% | {yr}")
    
    print(f"\n{'=' * 75}")
    print("🏆 综合最佳（累计收益排序）")
    print("=" * 75)
    results.sort(key=lambda r: -r['cum'])
    for i, r in enumerate(results[:8]):
        yr = ' '.join([f'{y}{r["years"].get(y,"?"):+5.1f}' for y in ['2018','2020','2022','2024']])
        print(f"#{i+1} buy={r['buy']} sell={r['sell']} pos={r['pos']} | 累计{r['cum']:+7.1f}% 熊市{r['bear_sum']:+6.1f}% | {yr}")
    
    # 保存
    out_path = os.path.join(ROOT, 'data', 'bt_bear_sweep.json')
    results.sort(key=lambda r: (-r['bear_sum'], -r['cum']))
    save = {
        'run_time': time.strftime('%Y-%m-%d %H:%M'),
        'data_source': 'backtest_hist_yahoo.json (Yahoo Finance)',
        'stocks': len(valid),
        'total_combos': total_combos,
        'ranking_by_bear_performance': results[:15],
        'note': 'bear_sum = 2018年收益 + 2022年收益（两个典型熊市年份）'
    }
    with open(out_path, 'w') as f:
        json.dump(save, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

if __name__ == '__main__':
    main()
