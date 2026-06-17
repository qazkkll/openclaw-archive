#!/usr/bin/env python3
"""
V4.1 参数暴力扫描 · 140只质量池 · 12年回测
扫描参数组合 → TOP10排序输出
"""
import json, os, warnings, time
import numpy as np
from itertools import product

warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
OUTPUT = "/home/admin/.openclaw/workspace/data/bt_v41_sweep_results.json"

# ── 加载候选池 ──
print("=" * 85)
print("V4.1 参数暴力扫描 · 140只质量池 · 12年(2014-2025)")
print("=" * 85)

pool_data = json.load(open(UNIVERSE))
tickers = pool_data['tickers']
print(f"候选池: {len(tickers)}只")

# ── 加载基准数据 ──
import yfinance as yf
spy_raw = yf.download('SPY', start="2013-06-01", end="2026-06-01", progress=False)
spy_close = spy_raw['Close'].squeeze()
spy_dates = list(spy_raw.index)
spy_dates_str = [d.strftime('%Y-%m-%d') for d in spy_dates]

def get_benchmark(d):
    for offset in range(5):
        if d in spy_dates_str:
            idx = spy_dates_str.index(d)
            return float(spy_close.iloc[idx])
        # Next day
        dd = d[:8] + str(int(d[8:10]) + offset + 1).zfill(2)
        if dd in spy_dates_str:
            idx = spy_dates_str.index(dd)
            return float(spy_close.iloc[idx])
    return None

# ── 加载所有股票数据 ──
print("\n加载140只股票数据...")
def get_metric(raw):
    result = {}
    n = len(raw)
    for i in range(60, n):
        row = raw[i]; d = row['date']; pr = float(row['close'])
        hp52 = max(float(raw[j]['close']) for j in range(max(0, i-251), i+1))
        p52 = pr/hp52*100 if hp52>0 else 100
        m = {}
        for p in [15, 20, 25, 30]:
            if i >= p:
                m[p] = (pr/float(raw[i-p]['close'])-1)*100
        result[d] = {'p': pr, 'p52': p52, **m}
    return result

loaded = {}
for t in tickers:
    fpath = f"{CACHE}/{t}.json"
    if os.path.exists(fpath):
        try:
            data = json.load(open(fpath))['data']
            if len(data) > 200:
                loaded[t] = get_metric(data)
        except:
            pass
print(f"  加载: {len(loaded)}只")

# ── 获取所有交易日 ──
all_dates = set()
for td in loaded.values():
    all_dates.update(td.keys())
dates_list = sorted(d for d in all_dates if '2014-01-01' <= d <= '2025-12-31')
print(f"  交易日: {len(dates_list)}天 ({dates_list[0]}~{dates_list[-1]})")

# ── 参数空间 ──
PARAM_GRID = {
    'deduct_start': [30, 40, 50, 60],       # 扣分起始点
    'deduct_coeff': [0.3, 0.5, 0.7, 0.9],    # 扣分系数
    'md': [15, 20, 25, 30],                   # 动量周期
    'tn': [3, 5, 8],                          # 持仓数
    'hd': [10, 15, 20, 25, 30],               # 调仓周期
}

total_combos = 1
for v in PARAM_GRID.values():
    total_combos *= len(v)
print(f"\n参数空间: {total_combos}个组合")
print(f"  扣分起始: {PARAM_GRID['deduct_start']}")
print(f"  扣分系数: {PARAM_GRID['deduct_coeff']}")
print(f"  动量周期: {PARAM_GRID['md']}日")
print(f"  持仓数: {PARAM_GRID['tn']}只")
print(f"  调仓周期: {PARAM_GRID['hd']}天")

# ── 扫描函数 ──
def run_backtest(params, years=range(2014, 2026)):
    """Run backtest for one parameter set, return total return, yearly returns, max drawdown"""
    ds = params['deduct_start']
    dc = params['deduct_coeff']
    md = params['md']
    tn = params['tn']
    hd = params['hd']
    
    yearly_rets = []
    
    for y in years:
        sd = f"{y}-01-02"
        ed = f"{y}-12-31"
        yr_dates = [d for d in dates_list if sd <= d <= ed]
        
        if len(yr_dates) < 60:
            yearly_rets.append(0)
            continue
        
        rets = []
        for si in range(hd, len(yr_dates) - hd, hd):
            d_buy = yr_dates[si]
            d_sell = yr_dates[min(si + hd, len(yr_dates) - 1)]
            d_mom = yr_dates[max(0, si - md)]
            
            cand = []
            for t, td in loaded.items():
                vb = td.get(d_buy)
                vp = td.get(d_mom)
                if not vb or not vp or vb['p'] < 1:
                    continue
                
                momentum = (vb['p'] / vp['p'] - 1) * 100 if md not in vb else vb[md]
                p52 = vb['p52']
                
                # 比例扣分
                deduction = max(0, (p52 - ds) / (100 - ds)) * dc
                score = momentum * (1 - min(deduction, 1))
                bp = vb['p']  # store buy price
                
                cand.append((score, t, bp))
            
            if len(cand) < tn:
                continue
            
            cand.sort(key=lambda x: x[0], reverse=True)
            
            pr = []
            for _, t, bp in cand[:tn]:
                vs = loaded[t].get(d_sell)
                if vs and bp > 0:
                    pr.append((vs['p'] / bp - 1) * 100)
            
            if pr:
                rets.append(np.mean(pr))
        
        if rets:
            yearly_rets.append(sum(rets))
    
    # 累计收益
    cumulative = sum(yearly_rets)
    nyears = len([r for r in yearly_rets if r != 0])
    if nyears == 0:
        return cumulative, yearly_rets, 0, 0, 0, 0, 0
    
    # 年化
    annualized = ((1 + cumulative / 100) ** (1 / nyears) - 1) * 100 if cumulative > -100 else 0
    
    # 胜率
    win_years = sum(1 for r in yearly_rets if r > 0)
    yr_win_rate = win_years / nyears * 100
    
    # 夏普（用逐年标准差）
    if len(yearly_rets) > 2 and np.std(yearly_rets) > 0:
        sharpe = np.mean(yearly_rets) / np.std(yearly_rets) * (12 ** 0.5)
    else:
        sharpe = 0
    
    # 最大回撤（逐年累积）
    cum_val = 100
    peak = 100
    mdd = 0
    for r in yearly_rets:
        if r != 0:
            cum_val *= (1 + r / 100)
            if cum_val > peak:
                peak = cum_val
            dd = (peak - cum_val) / peak * 100
            if dd > mdd:
                mdd = dd
    
    # 综合评分: 年化×1.0 + 夏普×3.0 - 最大回撤×0.5
    composite = annualized * 1.0 + sharpe * 3.0 - mdd * 0.5
    
    return cumulative, yearly_rets, annualized, yr_win_rate, sharpe, mdd, composite


# ── 执行暴力扫描 ──
print(f"\n开始扫描...")
start_time = time.time()

results = []
keys = list(PARAM_GRID.keys())
count = 0

for ds in PARAM_GRID['deduct_start']:
    for dc in PARAM_GRID['deduct_coeff']:
        for md in PARAM_GRID['md']:
            for tn in PARAM_GRID['tn']:
                for hd in PARAM_GRID['hd']:
                    params = {
                        'deduct_start': ds,
                        'deduct_coeff': dc,
                        'md': md,
                        'tn': tn,
                        'hd': hd
                    }
                    
                    cumulative, yearly_rets, annualized, yr_win, sharpe, mdd, composite = run_backtest(params)
                    
                    results.append({
                        'params': params,
                        'cumulative': round(cumulative, 2),
                        'annualized': round(annualized, 2),
                        'win_rate': round(yr_win, 1),
                        'sharpe': round(sharpe, 2),
                        'max_drawdown': round(mdd, 1),
                        'composite': round(composite, 2),
                        'yearly': {str(2014+i): round(yearly_rets[i], 2) if i < len(yearly_rets) else 0 
                                  for i in range(12)}
                    })
                    
                    count += 1
                    if count % 100 == 0 or count == total_combos:
                        elapsed = time.time() - start_time
                        rate = count / elapsed if elapsed > 0 else 0
                        print(f"  {count}/{total_combos} ({count/total_combos*100:.0f}%) | {rate:.0f}/s | elapsed: {elapsed:.0f}s")

elapsed = time.time() - start_time
print(f"\n✅ 扫描完成！{count}个组合，耗时{elapsed:.0f}s")

# ── 排序 ──
by_cumulative = sorted(results, key=lambda x: x['cumulative'], reverse=True)
by_sharpe = sorted(results, key=lambda x: x['sharpe'], reverse=True)
by_composite = sorted(results, key=lambda x: x['composite'], reverse=True)
by_mdd = sorted(results, key=lambda x: x['max_drawdown'])

# ── 输出TOP榜 ──
print(f"\n{'=' * 100}")
print("🏆 TOP 10 by 累计收益")
print(f"{'=' * 100}")
print(f"{'#':>3s}  {'累计':>8s}  {'年化':>7s}  {'夏普':>6s}  {'回撤':>6s}  {'胜率':>5s}  {'综合':>6s}  {'参数':>45s}")
print(f"{'-' * 100}")
for i, r in enumerate(by_cumulative[:15]):
    p = r['params']
    param_str = f"ds={p['deduct_start']} dc={p['deduct_coeff']} md={p['md']} tn={p['tn']} hd={p['hd']}"
    print(f"{i+1:>3d}  {r['cumulative']:>+8.1f}%  {r['annualized']:>+6.1f}%  {r['sharpe']:>6.2f}  {r['max_drawdown']:>5.1f}%  {r['win_rate']:>4.0f}%  {r['composite']:>6.1f}  {param_str}")

print(f"\n{'=' * 100}")
print("🏆 TOP 10 by 夏普比率")
print(f"{'=' * 100}")
print(f"{'#':>3s}  {'累计':>8s}  {'年化':>7s}  {'夏普':>6s}  {'回撤':>6s}  {'胜率':>5s}  {'综合':>6s}  {'参数':>45s}")
print(f"{'-' * 100}")
for i, r in enumerate(by_sharpe[:15]):
    p = r['params']
    param_str = f"ds={p['deduct_start']} dc={p['deduct_coeff']} md={p['md']} tn={p['tn']} hd={p['hd']}"
    print(f"{i+1:>3d}  {r['cumulative']:>+8.1f}%  {r['annualized']:>+6.1f}%  {r['sharpe']:>6.2f}  {r['max_drawdown']:>5.1f}%  {r['win_rate']:>4.0f}%  {r['composite']:>6.1f}  {param_str}")

print(f"\n{'=' * 100}")
print("🏆 TOP 10 by 综合评分")
print(f"{'=' * 100}")
print(f"{'#':>3s}  {'累计':>8s}  {'年化':>7s}  {'夏普':>6s}  {'回撤':>6s}  {'胜率':>5s}  {'综合':>6s}  {'参数':>45s}")
print(f"{'-' * 100}")
for i, r in enumerate(by_composite[:15]):
    p = r['params']
    param_str = f"ds={p['deduct_start']} dc={p['deduct_coeff']} md={p['md']} tn={p['tn']} hd={p['hd']}"
    print(f"{i+1:>3d}  {r['cumulative']:>+8.1f}%  {r['annualized']:>+6.1f}%  {r['sharpe']:>6.2f}  {r['max_drawdown']:>5.1f}%  {r['win_rate']:>4.0f}%  {r['composite']:>6.1f}  {param_str}")

# ── 各因素影响分析 ──
print(f"\n{'=' * 100}")
print("各因素对收益的影响分析")
print(f"{'=' * 100}")

# For each dimension, show the best, average, and worst
for dim_name, dim_key, dim_vals in [
    ("扣分起始点(deduct_start)", 'deduct_start', PARAM_GRID['deduct_start']),
    ("扣分系数(deduct_coeff)", 'deduct_coeff', PARAM_GRID['deduct_coeff']),
    ("动量周期", 'md', PARAM_GRID['md']),
    ("持仓数", 'tn', PARAM_GRID['tn']),
    ("调仓周期", 'hd', PARAM_GRID['hd']),
]:
    print(f"\n{dim_name}:")
    for v in dim_vals:
        group = [r for r in results if r['params'][dim_key] == v]
        avg_cum = np.mean([r['cumulative'] for r in group])
        avg_sharpe = np.mean([r['sharpe'] for r in group])
        avg_mdd = np.mean([r['max_drawdown'] for r in group])
        best_cum = max(r['cumulative'] for r in group)
        print(f"  {v:>5}: avg({avg_cum:+.1f}%) 夏普{avg_sharpe:.2f} 回撤{avg_mdd:.1f}%  max(+{best_cum:.1f}%)")

# ── 保存结果 ──
output = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'total_combinations': total_combos,
    'parameter_space': PARAM_GRID,
    'top_by_cumulative': [{
        'rank': i+1,
        'params': r['params'],
        'cumulative': r['cumulative'],
        'annualized': r['annualized'],
        'sharpe': r['sharpe'],
        'max_drawdown': r['max_drawdown'],
        'win_rate': r['win_rate'],
        'composite': r['composite']
    } for i, r in enumerate(by_cumulative[:20])],
    'top_by_sharpe': [{
        'rank': i+1,
        'params': r['params'],
        'cumulative': r['cumulative'],
        'annualized': r['annualized'],
        'sharpe': r['sharpe'],
        'max_drawdown': r['max_drawdown'],
        'win_rate': r['win_rate'],
        'composite': r['composite']
    } for i, r in enumerate(by_sharpe[:20])],
    'top_by_composite': [{
        'rank': i+1,
        'params': r['params'],
        'cumulative': r['cumulative'],
        'annualized': r['annualized'],
        'sharpe': r['sharpe'],
        'max_drawdown': r['max_drawdown'],
        'win_rate': r['win_rate'],
        'composite': r['composite']
    } for i, r in enumerate(by_composite[:20])],
    'factor_analysis': {}
}

for dim_name, dim_key, dim_vals in [
    ("扣分起始点", 'deduct_start', PARAM_GRID['deduct_start']),
    ("扣分系数", 'deduct_coeff', PARAM_GRID['deduct_coeff']),
    ("动量周期", 'md', PARAM_GRID['md']),
    ("持仓数", 'tn', PARAM_GRID['tn']),
    ("调仓周期", 'hd', PARAM_GRID['hd']),
]:
    output['factor_analysis'][dim_key] = {}
    for v in dim_vals:
        group = [r for r in results if r['params'][dim_key] == v]
        output['factor_analysis'][dim_key][str(v)] = {
            'avg_cumulative': round(np.mean([r['cumulative'] for r in group]), 2),
            'avg_sharpe': round(np.mean([r['sharpe'] for r in group]), 2),
            'avg_max_drawdown': round(np.mean([r['max_drawdown'] for r in group]), 2),
            'best_cumulative': round(max(r['cumulative'] for r in group), 2)
        }

with open(OUTPUT, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n✅ 全量结果已保存: {OUTPUT}")
