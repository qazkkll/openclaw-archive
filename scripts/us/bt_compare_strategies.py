#!/usr/bin/env python3
"""
🔥 V1 vs V1改进版 对比回测 — 周日19:00使用

用法:
  python3 scripts/bt_compare_strategies.py           # 默认500只
  python3 scripts/bt_compare_strategies.py --samples 1000
  python3 scripts/bt_compare_strategies.py --save     # 保存结果到文件

对比项:
  - V1原版: 门槛62, 原版权重
  - V1改进版: 门槛60 + 权重微调
"""

import json, os, sys, time, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_engine import v1_score_from_data

# ===== 改进版评分函数 =====
def v1_improved_score(close, high, low):
    """V1改进版: 降门槛由外部控制，评分函数同原版"""
    s = v1_score_from_data(close, high, low)
    return s

# ===== 9段窗口配置 =====
WINDOWS = [
    {'name': 'W1 白马牛市',    'test': ('2017-05', '2017-08')},
    {'name': 'W2 反弹牛市',    'test': ('2019-02', '2019-05')},
    {'name': 'W3 疫情后牛市',  'test': ('2020-06', '2020-09')},
    {'name': 'W4 去杠杆熊市',  'test': ('2018-06', '2018-09')},
    {'name': 'W5 政策底熊市',  'test': ('2022-01', '2022-04')},
    {'name': 'W6 微盘崩盘',    'test': ('2024-01', '2024-04')},
    {'name': 'W7 熔断后震荡',  'test': ('2016-06', '2016-09')},
    {'name': 'W8 结构市震荡',  'test': ('2021-06', '2021-09')},
    {'name': 'W9 AI震荡',      'test': ('2023-06', '2023-09')},
]

LOOKUP = 20

def run_window(stocks, test_start, test_end, threshold):
    """跑一个窗口，按threshold划线区分高/中/低分段"""
    results = {'high': [], 'mid': [], 'low': [], 'total': 0}

    for code in stocks:
        d = stocks[code]
        dates = d.get('dates', [])
        closes = d.get('close', [])
        if not dates or not closes or len(dates) < 200:
            continue

        try:
            start_i = next(i for i, dt in enumerate(dates) if dt >= test_start)
        except StopIteration:
            continue

        if start_i + LOOKUP >= len(closes):
            continue

        cur_p = closes[start_i]
        fut_p = closes[start_i + LOOKUP]
        ret = (fut_p / cur_p - 1) * 100 if cur_p > 0 else 0

        lo = max(0, start_i - 200)
        c_sub = closes[lo:start_i+1]
        h_sub = d.get('high', closes)[lo:start_i+1]
        l_sub = d.get('low', closes)[lo:start_i+1]

        try:
            s = v1_score_from_data(c_sub, h_sub, l_sub)
        except Exception:
            continue

        if s is None or s == 0:
            continue

        results['total'] += 1
        if s >= threshold:
            results['high'].append(ret)
        elif s >= 50:
            results['mid'].append(ret)
        else:
            results['low'].append(ret)

    def avg(arr):
        return sum(arr)/len(arr) if arr else 0

    return {
        'high_ret': avg(results['high']),
        'mid_ret': avg(results['mid']),
        'low_ret': avg(results['low']),
        'n_high': len(results['high']),
        'n_mid': len(results['mid']),
        'n_low': len(results['low']),
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='V1 vs V1改进版 对比')
    parser.add_argument('--load', default='data/backtest_hist_yahoo.json')
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    BASE = '/home/admin/.openclaw/workspace'
    data_path = os.path.join(BASE, args.load)
    print(f'V1 vs V1改进版  对比回测')
    print(f'数据: {data_path}')
    print(f'采样: {args.samples}只股票')
    print()

    with open(data_path) as f:
        yahoo = json.load(f)

    codes = [c for c in yahoo if isinstance(yahoo[c], dict) and
             len(yahoo[c].get('close', [])) >= 500]
    random.seed(42)
    sample = random.sample(codes, min(args.samples, len(codes)))
    print(f'合格股票: {len(codes)}只, 采样: {len(sample)}只')
    print()

    versions = [
        ('V1原版 (门槛62)', 62),
        ('V1改进版 (门槛60)', 60),
    ]

    all_results = {}
    for label, threshold in versions:
        print(f'--- {label} ---')
        header = f'{"窗口":<20} {"高分N":>6} {"高分收益":>9} {"中分N":>6} {"中分收益":>9} {"低分N":>6} {"低分收益":>9} {"差异":>9}'
        print(header)
        print('-' * 86)

        window_results = []
        diffs = []
        for w in WINDOWS:
            r = run_window({c: yahoo[c] for c in sample}, w['test'][0], w['test'][1], threshold)
            window_results.append(r)
            if r:
                diff = r['high_ret'] - r['low_ret']
                diffs.append(diff)
                print(f'{w["name"]:<20} {r["n_high"]:>6} {r["high_ret"]:>+8.1f}% {r["n_mid"]:>6} {r["mid_ret"]:>+8.1f}% {r["n_low"]:>6} {r["low_ret"]:>+8.1f}% {diff:>+8.1f}%')

        if diffs:
            avg_diff = sum(diffs) / len(diffs)
            pos = sum(1 for d in diffs if d > 0)
            print(f'\n平均差异: {avg_diff:+.1f}% | 正向窗口: {pos}/{len(diffs)}')
        all_results[label] = window_results
        print()

    # 门槛对比：增加多少高分票
    print('--- 门槛对比: 62 -> 60 ---')
    print(f'{"窗口":<20} {"V1高分N":>10} {"改进高分N":>12} {"增量":>8} {"中分段N":>10}')
    print('-' * 64)
    for i, w in enumerate(WINDOWS):
        r1 = all_results['V1原版 (门槛62)'][i]
        r2 = all_results['V1改进版 (门槛60)'][i]
        if r1 and r2:
            inc = r2['n_high'] - r1['n_high']
            print(f'{w["name"]:<20} {r1["n_high"]:>10} {r2["n_high"]:>12} {inc:>+8} {r2["n_mid"]:>10}')

    if args.save:
        out = {'v1': [], 'v1_improved': []}
        for w, r in zip(WINDOWS, all_results['V1原版 (门槛62)']):
            if r: out['v1'].append({'window': w['name'], 'high_ret': round(r['high_ret'],2), 'n_high': r['n_high']})
        for w, r in zip(WINDOWS, all_results['V1改进版 (门槛60)']):
            if r: out['v1_improved'].append({'window': w['name'], 'high_ret': round(r['high_ret'],2), 'n_high': r['n_high']})
        out_path = os.path.join(BASE, 'data/bt_comparison_results.json')
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n结果已保存: {out_path}')

    print()
    print('周日19:00正式跑 — 可用 --samples 1000 --save')
