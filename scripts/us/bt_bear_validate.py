#!/usr/bin/env python3
"""
🍤 熊市模式回测验证 v2 — 预计算指标加速版

优化: 先对所有股票一次性计算完所有指标，再逐日回测
数据源: data/backtest_hist_yahoo.json (Yahoo Finance, 1177只)
数据范围: 2015-2025

用法: python3 scripts/bt_bear_validate.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import compute_indicators, bear_score
from functools import lru_cache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, 'data', 'backtest_hist_yahoo.json')

REBALANCE_DAYS = 20
MAX_POSITIONS = 3

def run():
    t0 = time.time()
    
    # 加载数据
    print("加载数据...", flush=True)
    with open(DATA_PATH) as f:
        all_stocks = json.load(f)
    
    codes = list(all_stocks.keys())
    n = len(all_stocks[codes[0]]['close'])
    dates = all_stocks[codes[0]]['dates']
    
    print(f"数据源: backtest_hist_yahoo.json (Yahoo Finance)", flush=True)
    print(f"股票: {len(codes)}只 | K线: {n}根/只", flush=True)
    print(f"时间: {dates[0][:10]} ~ {dates[-1][:10]}", flush=True)
    
    # 预计算所有股票的指标（20-30分钟）
    print("\n预计算指标（每只股票一次compute_indicators）...", flush=True)
    precalc = {}  # code -> {scores_per_day, p52_per_day, vol_ratio_per_day, ...}
    
    for idx, code in enumerate(codes):
        d = all_stocks[code]
        close = d['close']
        high = d['high']
        low = d['low']
        
        if len(close) < 60:
            continue
        
        ind = compute_indicators(close, high, low)
        if ind is None:
            continue
        
        # 预计算每天评分
        scores = []
        for di in range(len(close)):
            s = bear_score(ind, di)
            scores.append(s)
        
        precalc[code] = {'scores': scores}
        
        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(codes)} | {time.time()-t0:.0f}s", flush=True)
    
    print(f"预计算完成: {len(precalc)}只有效股票 | {time.time()-t0:.0f}s", flush=True)
    
    # 回测
    print("\n开始回测...", flush=True)
    START_DAY = 250  # 预热期
    
    positions = {}
    holdings = {}
    last_rebalance = -REBALANCE_DAYS
    daily_value = {}
    yearly_returns = {}
    
    for di in range(START_DAY, n):
        today = dates[di]
        year = today[:4]
        
        if year not in yearly_returns:
            yearly_returns[year] = []
        
        # 当天所有评分>=0的股票
        candidates = []
        for code, p in precalc.items():
            if di >= len(p['scores']):
                continue
            s = p['scores'][di]
            if s > 0:
                candidates.append((s, code))
        
        candidates.sort(reverse=True)
        
        # 每20天调仓
        if di - last_rebalance >= REBALANCE_DAYS:
            top3 = candidates[:MAX_POSITIONS]
            if top3 and top3[0][0] > 0:
                holdings = {code: 1.0/MAX_POSITIONS for _, code in top3}
                positions = {code: di for code in holdings.keys()}
                last_rebalance = di
        
        # 计算当天组合市值
        if holdings:
            total = 0.0
            for code, weight in list(holdings.items()):
                arr = all_stocks[code]['close']
                if code in precalc and di < len(arr) and last_rebalance < len(arr):
                    ret = arr[di] / arr[last_rebalance] - 1
                    total += weight * (1 + ret)
                else:
                    total += weight
            daily_value[today] = total
        else:
            daily_value[today] = 1.0
    
    # 按年统计
    results = {}
    cum = 1.0
    
    for year in sorted(yearly_returns.keys()):
        yr_days = {d: v for d, v in daily_value.items() if d.startswith(year)}
        if not yr_days:
            continue
        ds = sorted(yr_days.keys())
        if len(ds) < 2:
            continue
        
        start_v = yr_days[ds[0]]
        end_v = yr_days[ds[-1]]
        ret = end_v / start_v - 1
        cum *= (1 + ret)
        results[year] = round(ret * 100, 1)
    
    total_ret = round((cum - 1) * 100, 1)
    nyears = len(results)
    ann_ret = round((cum ** (1/nyears) - 1) * 100, 1) if nyears > 0 else 0
    win_years = sum(1 for v in results.values() if v > 0)
    
    # 输出
    print(f"\n{'='*60}")
    print(f"熊市模式回测结果 (bear_score, v2)")
    print(f"数据源: Yahoo Finance (Yfinance) | {len(precalc)}只股票 | {n}根K线")
    print(f"参数: {REBALANCE_DAYS}天调仓 | 等权前{MAX_POSITIONS}只")
    print(f"运行时间: {time.time()-t0:.0f}s")
    print(f"{'='*60}")
    print(f"\n{'年份':<8} {'年收益':>10}")
    print('-' * 22)
    for year in sorted(results.keys()):
        print(f'{year:<8} {results[year]:>+8.1f}%')
    print('-' * 22)
    print(f'累计: {total_ret:+.1f}%')
    print(f'年化: {ann_ret:+.1f}%')
    print(f'胜率: {win_years}/{nyears}')
    
    # 保存
    out = {
        'script': 'bt_bear_validate.py v2',
        'run_time': time.strftime('%Y-%m-%d %H:%M'),
        'data_source': 'backtest_hist_yahoo.json (Yahoo Finance)',
        'stocks': len(precalc),
        'model': 'bear_score (V4熊市逆向, 无volume因子)',
        'parameters': {
            'rebalance_days': REBALANCE_DAYS,
            'max_positions': MAX_POSITIONS,
            'factors': 'MACD20/52W30/RSI15/均线15/量能20(无数据→分摊)'
        },
        'yearly_returns': results,
        'total_return_pct': total_ret,
        'annualized_pct': ann_ret,
        'win_years': f'{win_years}/{nyears}',
    }
    
    out_path = os.path.join(ROOT, 'data', 'bt_bear_validate.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

if __name__ == '__main__':
    run()
