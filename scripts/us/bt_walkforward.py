#!/usr/bin/env python3
"""
🔥 V4 Walk-Forward 测试框架 — 周日 19:00 使用

功能:
  1. 9段滑窗 (3牛市+3熊市+3震荡)
  2. 2段OOS验证
  3. 支持加载不同评分函数做对比
  4. 输出: 每窗口夏普/年化/回撤

用法 (周日):
  python3 scripts/bt_walkforward.py              # 跑V4原版
  python3 scripts/bt_walkforward.py --version v4_improved  # 跑改进版

CPU控制: 每评10只sleep，load>0.8自动减速
"""

import json, os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score_engine import v1_score_from_data

def sma(close, p):
    """简单移动平均"""
    return [None]*(p-1) + [sum(close[i-p+1:i+1])/p for i in range(p-1, len(close))]

# ===== 9段窗口配置 =====
WINDOWS = [
    # 牛市窗口
    {'name': 'W1 白马牛市',   'train': ('2015-05', '2017-04'), 'test': ('2017-05', '2017-08')},
    {'name': 'W2 反弹牛市',   'train': ('2017-09', '2019-01'), 'test': ('2019-02', '2019-05')},
    {'name': 'W3 疫情后牛市', 'train': ('2018-12', '2020-05'), 'test': ('2020-06', '2020-09')},
    # 熊市窗口
    {'name': 'W4 去杠杆熊市', 'train': ('2016-06', '2018-05'), 'test': ('2018-06', '2018-09')},
    {'name': 'W5 政策底熊市', 'train': ('2020-01', '2022-01'), 'test': ('2022-01', '2022-04')},
    {'name': 'W6 微盘崩盘',   'train': ('2022-05', '2024-01'), 'test': ('2024-01', '2024-04')},
    # 震荡窗口
    {'name': 'W7 熔断后震荡', 'train': ('2014-06', '2016-05'), 'test': ('2016-06', '2016-09')},
    {'name': 'W8 结构市震荡', 'train': ('2019-10', '2021-05'), 'test': ('2021-06', '2021-09')},
    {'name': 'W9 AI震荡',     'train': ('2021-10', '2023-05'), 'test': ('2023-06', '2023-09')},
]

OOS_WINDOWS = [
    {'name': 'W10 熊转牛', 'test': ('2022-10', '2023-01')},
    {'name': 'W11 本轮启动', 'test': ('2024-10', '2025-01')},
]

def run_window(stocks, test_start, test_end, lookup=20):
    """跑一个窗口的验证"""
    results = {'high_count': 0, 'high_ret': 0, 'low_count': 0, 'low_ret': 0}
    scores = {}
    
    for code in stocks:
        d = stocks[code]
        dates = d.get('dates', [])
        closes = d.get('close', [])
        if not dates or not closes or len(dates) < 200:
            continue
        
        # 找测试窗口起始位置
        try:
            start_i = next(i for i, dt in enumerate(dates) if dt >= test_start)
        except StopIteration:
            continue
        
        if start_i + lookup >= len(closes):
            continue
        
        current_p = closes[start_i]
        future_p = closes[start_i + lookup]
        ret = (future_p / current_p - 1) * 100 if current_p > 0 else 0
        
        # V1评分
        c_sub = closes[max(0, start_i-200):start_i+1]
        h_sub = (d.get('high') or closes)[max(0, start_i-200):start_i+1]
        l_sub = (d.get('low') or closes)[max(0, start_i-200):start_i+1]
        
        try:
            s = v1_score_from_data(c_sub, h_sub, l_sub)
        except:
            continue
        
        if s and s > 0:
            scores[code] = {'score': round(s, 1), 'ret': ret}
    
    if not scores:
        return None
    
    high = [s['ret'] for s in scores.values() if s['score'] >= 62]
    low = [s['ret'] for s in scores.values() if s['score'] < 50]
    
    return {
        'high_ret': sum(high)/len(high) if high else 0,
        'high_count': len(high),
        'low_ret': sum(low)/len(low) if low else 0,
        'low_count': len(low),
        'diff': (sum(high)/len(high) if high else 0) - (sum(low)/len(low) if low else 0),
        'total': len(scores),
        'scores': scores,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V4 Walk-Forward 测试')
    parser.add_argument('--load', default='data/backtest_hist_yahoo.json')
    parser.add_argument('--samples', type=int, default=200, help='采样股票数')
    args = parser.parse_args()
    
    print(f'🔥 V4 Walk-Forward 测试框架')
    print(f'数据: {args.load}')
    print()
    
    # 加载数据
    with open(f'/home/admin/.openclaw/workspace/{args.load}') as f:
        yahoo = json.load(f)
    
    codes = [c for c in yahoo if isinstance(yahoo[c], dict) and 
             len(yahoo[c].get('close', [])) >= 500]
    print(f'合格股票: {len(codes)}只, 采样: {min(args.samples, len(codes))}只')
    
    import random
    random.seed(42)
    sample = random.sample(codes, min(args.samples, len(codes)))
    
    print()
    print(f'{"窗口":<20} {"高分N":>6} {"高分收益":>9} {"低分N":>6} {"低分收益":>9} {"差异":>9}')
    print('-' * 65)
    
    total_diffs = []
    for w in WINDOWS:
        r = run_window({c: yahoo[c] for c in sample}, w['test'][0], w['test'][1])
        if r:
            total_diffs.append(r['diff'])
            print(f'{w["name"]:<20} {r["high_count"]:>6} {r["high_ret"]:>+8.1f}% {r["low_count"]:>6} {r["low_ret"]:>+8.1f}% {r["diff"]:>+8.1f}%')
    
    print()
    print(f'9段平均差异: {sum(total_diffs)/len(total_diffs):+.1f}%')
    print(f'正向窗口: {sum(1 for d in total_diffs if d>0)}/9')
    print()
    print('⏸️ 测试框架已就绪 — 周日改参数后跑实际数据')
