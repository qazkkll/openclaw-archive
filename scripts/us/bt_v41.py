#!/usr/bin/env python3
"""
小钳轮动 V4.1 美股 · 140只质量池 · 12年全量回测
对比: V4.1 / V3纯动量 / SPY / QQQ

V4.1参数:
  - 扣分起始: p52=40%
  - 扣分系数: 0.5
  - 动量周期: 25日
  - 持仓: 前5只
  - 调仓: 20天
"""
import json, os, sys, warnings
import numpy as np
from bisect import bisect_right

warnings.filterwarnings('ignore')

CACHE = "/home/admin/.openclaw/workspace/data/cache"
UNIVERSE = "/home/admin/.openclaw/workspace/data/sp500_universe.json"
OUTPUT = "/home/admin/.openclaw/workspace/data/bt_v41_results.json"

# ── 1. 加载候选池 ──
print("=" * 95)
print("V4.1 美股 · 140只质量池 · 12年全量回测 (2014-2025)")
print("=" * 95)

pool_data = json.load(open(UNIVERSE))
tickers = pool_data['tickers']
print(f"\n候选池: {len(tickers)}只 (S&P 500质量筛选)")
print(f"筛选标准: ROE>15%, 毛利率>40%, 市值>100亿, 负债率<1, PE在0-40")

# ── 2. 加载SPY/QQQ基准 ──
print("\n加载基准指数...")
import yfinance as yf
spy_raw = yf.download('SPY', start="2013-06-01", end="2026-06-01", progress=False)
spy_close = spy_raw['Close'].squeeze()
spy_dates = [d.strftime('%Y-%m-%d') for d in spy_raw.index]

qqq_raw = yf.download('QQQ', start="2013-06-01", end="2026-06-01", progress=False)
qqq_close = qqq_raw['Close'].squeeze()
qqq_dates = [d.strftime('%Y-%m-%d') for d in qqq_raw.index]

print(f"  SPY: {len(spy_dates)}天 ({spy_dates[0]}~{spy_dates[-1]})")
print(f"  Q关注: {len(qqq_dates)}天 ({qqq_dates[0]}~{qqq_dates[-1]})")

# ── 3. 加载所有股票数据 ──
print("\n加载140只股票数据...")

def get_metric(raw):
    """预计算每只股票的指标: 25日动量, 15日动量, 52周位置, 收盘价"""
    result = {}
    n = len(raw)
    for i in range(60, n):
        row = raw[i]
        d = row['date']
        pr = float(row['close'])
        
        # 52周高点和位置
        hp52 = max(float(raw[j]['close']) for j in range(max(0, i-251), i+1))
        p52 = pr / hp52 * 100 if hp52 > 0 else 100
        
        # 动量
        m15 = (pr / float(raw[i-15]['close']) - 1) * 100 if i >= 15 else 0
        m25 = (pr / float(raw[i-25]['close']) - 1) * 100 if i >= 25 else 0
        m20 = (pr / float(raw[i-20]['close']) - 1) * 100 if i >= 20 else 0
        
        result[d] = {'p': pr, 'p52': p52, 'm15': m15, 'm25': m25, 'm20': m20}
    return result

loaded = {}
failed = []
for t in tickers:
    fpath = f"{CACHE}/{t}.json"
    if os.path.exists(fpath):
        try:
            data = json.load(open(fpath))
            raw = data['data']
            if len(raw) > 200:
                loaded[t] = get_metric(raw)
            else:
                failed.append((t, "too_short", len(raw)))
        except Exception as e:
            failed.append((t, str(e)[:30], 0))
    else:
        failed.append((t, "no_cache", 0))

print(f"  成功加载: {len(loaded)}只")
if failed:
    print(f"  加载失败: {len(failed)}只")
    for t, r, n in failed[:10]:
        print(f"    {t}: {r}" + (f" (len={n})" if n else ""))

# ── 4. 回测函数 ──

def get_spy_price(d):
    """Get SPY close price for a given date"""
    for offset in range(5):
        dt = f"{d[:8]}{int(d[8:10])-offset:02d}" if d[8:10].isdigit() else d
        if dt in spy_dates:
            idx = spy_dates.index(dt)
            return float(spy_close.iloc[idx])
    return None

def get_qqq_price(d):
    """Get QQQ close price for a given date"""
    for offset in range(5):
        dt = f"{d[:8]}{int(d[8:10])-offset:02d}" if d[8:10].isdigit() else d
        if dt in qqq_dates:
            idx = qqq_dates.index(dt)
            return float(qqq_close.iloc[idx])
    return None

def run_v41_year(all_data, year, md=25, tn=5, hd=20, 
                 deduct_start=40, deduct_coeff=0.5):
    """
    V4.1 比例扣分动量追涨
    score = md_momentum × (1 - max(0, (p52 - deduct_start) / (100-deduct_start) × deduct_coeff))
    """
    sd = f"{year}-01-02"
    ed = f"{year}-12-31"
    
    # Find date range
    all_dates = set()
    for t, td in all_data.items():
        all_dates.update(td.keys())
    all_dates = sorted(d for d in all_dates if sd <= d <= ed)
    
    if len(all_dates) < 60:
        return [], []
    
    # Convert to sorted list for iteration
    dates = sorted(all_dates)
    
    rets = []
    spy_rets = []
    qqq_rets = []
    
    for si in range(hd, len(dates) - hd, hd):
        d_buy = dates[si]
        d_sell = dates[min(si + hd, len(dates) - 1)]
        d_mom = dates[max(0, si - md)]
        
        cand = []
        for t, td in all_data.items():
            vb = td.get(d_buy)
            vp = td.get(d_mom)
            if not vb or not vp or vb['p'] < 1:
                continue
            
            # 25日动量
            momentum = vb.get(f'm{md}', (vb['p'] / vp['p'] - 1) * 100)
            p52 = vb['p52']
            
            # V4.1 比例扣分
            deduction = max(0, (p52 - deduct_start) / (100 - deduct_start)) * deduct_coeff
            score = momentum * (1 - min(deduction, 1))
            
            cand.append((score, t, vb['p']))
        
        if len(cand) < tn:
            continue
        
        cand.sort(key=lambda x: x[0], reverse=True)
        
        # 买入前tn只
        period_rets = []
        for sc, t, bp in cand[:tn]:
            vs = all_data[t].get(d_sell)
            if vs and vs['p'] > 0 and bp > 0:
                period_rets.append((vs['p'] / bp - 1) * 100)
        
        if period_rets:
            rets.append(np.mean(period_rets))
        
            # SPY/QQQ同期收益
            sp = get_spy_price(d_buy)
            ss = get_spy_price(d_sell)
            if sp and ss and sp > 0:
                spy_rets.append((ss / sp - 1) * 100)
            
            qp = get_qqq_price(d_buy)
            qs = get_qqq_price(d_sell)
            if qp and qs and qp > 0:
                qqq_rets.append((qs / qp - 1) * 100)
    
    return rets, spy_rets, qqq_rets


def run_v3_year(all_data, year, md=20, tn=5, hd=20):
    """
    V3 纯动量（基准）：前5 = 20日动量最高的5只，无过滤
    """
    sd = f"{year}-01-02"
    ed = f"{year}-12-31"
    
    all_dates = set()
    for t, td in all_data.items():
        all_dates.update(td.keys())
    all_dates = sorted(d for d in all_dates if sd <= d <= ed)
    
    if len(all_dates) < 60:
        return []
    
    dates = sorted(all_dates)
    rets = []
    
    for si in range(hd, len(dates) - hd, hd):
        d_buy = dates[si]
        d_sell = dates[min(si + hd, len(dates) - 1)]
        
        cand = []
        for t, td in all_data.items():
            vb = td.get(d_buy)
            if not vb or vb['p'] < 1:
                continue
            
            momentum = vb.get(f'm{md}', 0)
            cand.append((momentum, t, vb['p']))
        
        if len(cand) < tn:
            continue
        
        cand.sort(key=lambda x: x[0], reverse=True)
        
        period_rets = []
        for m, t, bp in cand[:tn]:
            vs = all_data[t].get(d_sell)
            if vs and vs['p'] > 0 and bp > 0:
                period_rets.append((vs['p'] / bp - 1) * 100)
        
        if period_rets:
            rets.append(np.mean(period_rets))
    
    return rets


def calc_stats(yearly_rets):
    """计算汇总统计"""
    if not yearly_rets:
        return {}
    
    cumulative = sum(yearly_rets)
    nyears = len(yearly_rets)
    
    # 年华
    annualized = ((1 + cumulative / 100) ** (1 / nyears) - 1) * 100 if cumulative > -100 else 0
    
    # 胜率
    win_rate = sum(1 for r in yearly_rets if r > 0) / len(yearly_rets) * 100
    
    # 夏普（月收益序列近似）
    # 这里用逐年标准差近似
    if len(yearly_rets) > 2:
        std = np.std(yearly_rets)
        sharpe = (np.mean(yearly_rets) / std) * (12 ** 0.5) if std > 0 else 0
    else:
        sharpe = 0
    
    # 最大回撤（对重复益计算）
    cum_values = []
    val = 100
    for r in yearly_rets:
        val *= (1 + r / 100)
        cum_values.append(val)
    peak = 100
    mdd = 0
    for v in cum_values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        if dd > mdd:
            mdd = dd
    
    return {
        'cumulative': round(cumulative, 2),
        'annualized': round(annualized, 2),
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(mdd, 1)
    }


# ── 5. 执行回测 ──
YEARS = list(range(2014, 2026))

v41_years = []
v3_years = []
spy_years = []
qqq_years = []

print(f"\n{'=' * 95}")
print(f"{'年份':>6s}  {'V4.1选股':>10s}  {'V3纯动量':>10s}  {'SPY':>8s}  {'QQQ':>8s}  {'V4超额QQQ':>10s}  {'V4超额SPY':>10s}")
print(f"{'=' * 95}")

for y in YEARS:
    # V4.1
    rets, spy_rets_period, qqq_rets_period = run_v41_year(loaded, y)
    v41_total = sum(rets) if rets else 0
    
    # V3
    v3_rets = run_v3_year(loaded, y)
    v3_total = sum(v3_rets) if v3_rets else 0
    
    # SPY/QQQ年度
    spy_yr = sum(spy_rets_period) if spy_rets_period else 0
    qqq_yr = sum(qqq_rets_period) if qqq_rets_period else 0
    
    v41_years.append(v41_total)
    v3_years.append(v3_total)
    spy_years.append(spy_yr)
    qqq_years.append(qqq_yr)
    
    excess_qqq = v41_total - qqq_yr
    excess_spy = v41_total - spy_yr
    
    line = f"{y:>6d}  {v41_total:>+10.1f}%  {v3_total:>+10.1f}%  {spy_yr:>+8.1f}%  {qqq_yr:>+8.1f}%  {excess_qqq:>+10.1f}%  {excess_spy:>+10.1f}%"
    print(line)

print(f"{'=' * 95}")

# 汇总
v41_stats = calc_stats(v41_years)
v3_stats = calc_stats(v3_years)
spy_yr_total = sum(spy_years)
qqq_yr_total = sum(qqq_years)

print(f"\n{'=' * 95}")
print("汇总统计")
print(f"{'=' * 95}")
print(f"{'指标':>20s}  {'V4.1':>10s}  {'V3纯动量':>10s}  {'SPY':>10s}  {'QQQ':>10s}")
print(f"{'-' * 65}")
print(f"{'12年累计':>20s}  {sum(v41_years):>+10.1f}%  {sum(v3_years):>+10.1f}%  {sum(spy_years):>+10.1f}%  {sum(qqq_years):>+10.1f}%")
print(f"{'年化收益率':>20s}  {v41_stats['annualized']:>+9.1f}%  {v3_stats['annualized']:>+9.1f}%  --  --")
print(f"{'胜率(年)':>20s}  {v41_stats['win_rate']:>9.1f}%  {v3_stats['win_rate']:>9.1f}%  --  --")
print(f"{'夏普(年化)':>20s}  {v41_stats['sharpe']:>10.2f}  {v3_stats['sharpe']:>10.2f}  --  --")
print(f"{'最大回撤':>20s}  {v41_stats['max_drawdown']:>9.1f}%  {v3_stats['max_drawdown']:>9.1f}%  --  --")

# ├─ 跑赢/跑输 SPY 次数
v41_vs_spy = sum(1 for i in range(12) if v41_years[i] > spy_years[i])
v41_vs_qqq = sum(1 for i in range(12) if v41_years[i] > qqq_years[i])
v3_vs_spy = sum(1 for i in range(12) if v3_years[i] > spy_years[i])

print(f"\n{'跑赢SPY年数':>20s}  {v41_vs_spy}/12         {v3_vs_spy}/12         --            --")
print(f"{'跑赢QQQ年数':>20s}  {v41_vs_qqq}/12         --                --            --")

# ── 6. 因子贡献度分解 ──
print(f"\n{'=' * 95}")
print("因子贡献度分解 (逐一去掉看影响)")
print(f"{'=' * 95}")

baseline_year_rets = v41_years
baseline_total = sum(baseline_year_rets)

factors = [
    ("去掉扣分(纯动量)",      dict(deduct_coeff=0.0)),           # 相当于 V3 但用25日
    ("扣分从50%起(原始)",     dict(deduct_start=50, deduct_coeff=0.7)),  # V4原始参数
    ("换20日动量",             dict(md=20)),
    ("换10天调仓",             dict(hd=10)),
    ("换3只持仓",              dict(tn=3)),
    ("换8只持仓",              dict(tn=8)),
    ("扣分系数0.7(原始)",      dict(deduct_coeff=0.7)),
]

print(f"\n{'因子变体':>25s}  {'12年累计':>10s}  {'vs基准':>10s}")
print(f"{'-' * 50}")
print(f"{'V4.1基准':>25s}  {baseline_total:>+10.1f}%  {'--':>10s}")

for fname, params in factors:
    # Merge with default V4.1 params
    p = {'md': 25, 'tn': 5, 'hd': 20, 'deduct_start': 40, 'deduct_coeff': 0.5}
    p.update(params)
    
    total = 0
    for y in YEARS:
        rets, _, _ = run_v41_year(loaded, y, **p)
        total += sum(rets) if rets else 0
    
    diff = total - baseline_total
    print(f"{fname:>25s}  {total:>+10.1f}%  {diff:>+10.1f}%")

# ── 7. 选股能力验证（每期平均收益 vs 全池等权） ──
print(f"\n{'=' * 95}")
print("选股能力验证 (V4.1选股每期收益 vs 全池等权)")
print(f"{'=' * 95}")

total_v41_period = []
total_pool_period = []

for y in YEARS:
    sd = f"{y}-01-02"
    ed = f"{y}-12-31"
    
    all_dates = set()
    for t, td in loaded.items():
        all_dates.update(td.keys())
    all_dates = sorted(d for d in all_dates if sd <= d <= ed)
    
    if len(all_dates) < 60:
        continue
    
    dates = sorted(all_dates)
    
    for si in range(20, len(dates) - 20, 20):
        d_buy = dates[si]
        d_sell = dates[min(si + 20, len(dates) - 1)]
        d_mom = dates[max(0, si - 25)]
        
        # V4.1选股
        cand = []
        all_returns = []
        for t, td in loaded.items():
            vb = td.get(d_buy)
            vp = td.get(d_mom)
            vs = td.get(d_sell)
            if not vb or not vp or not vs or vb['p'] < 1:
                continue
            
            momentum = vb.get('m25', (vb['p'] / vp['p'] - 1) * 100)
            p52 = vb['p52']
            deduction = max(0, (p52 - 40) / 60) * 0.5
            score = momentum * (1 - min(deduction, 1))
            
            ret = (vs['p'] / vb['p'] - 1) * 100
            all_returns.append(ret)
            cand.append((score, ret))
        
        if len(cand) < 5:
            continue
        
        cand.sort(key=lambda x: x[0], reverse=True)
        v41_avg = np.mean([r for _, r in cand[:5]])
        pool_avg = np.mean(all_returns)
        
        total_v41_period.append(v41_avg)
        total_pool_period.append(pool_avg)

if total_v41_period:
    v41_avg_period = np.mean(total_v41_period)
    pool_avg_period = np.mean(total_pool_period)
    print(f"\n  V4.1选股平均收益/期: {v41_avg_period:+.2f}%")
    print(f"  全池等权平均收益/期: {pool_avg_period:+.2f}%")
    print(f"  选股超额: {v41_avg_period - pool_avg_period:+.2f}%/期")
    print(f"  选股胜率: {sum(1 for a, p in zip(total_v41_period, total_pool_period) if a > p)/len(total_v41_period)*100:.1f}%")

# ── 8. 累积曲线数据 ──
print(f"\n{'=' * 95}")
print("累积收益曲线")
print(f"{'=' * 95}")
print(f"{'年份':>6s}  {'V4.1累积':>10s}  {'V3累积':>10s}  {'SPY累积':>10s}  {'QQQ累积':>10s}")
print(f"{'-' * 55}")

cum_v41 = 0; cum_v3 = 0; cum_spy = 0; cum_qqq = 0
for i, y in enumerate(YEARS):
    cum_v41 += v41_years[i]
    cum_v3 += v3_years[i]
    cum_spy += spy_years[i]
    cum_qqq += qqq_years[i]
    
    ei = "🏆" if v41_years[i] > spy_years[i] and v41_years[i] > qqq_years[i] else \
         "✅" if v41_years[i] > spy_years[i] else "🟡" if v41_years[i] > qqq_years[i] else ""
    print(f"{y:>6d}  {cum_v41:>+10.1f}%  {cum_v3:>+10.1f}%  {cum_spy:>+10.1f}%  {cum_qqq:>+10.1f}%  {ei}")

# ── 9. 保存结果 ──
results = {
    'model': 'V4.1',
    'date': '2026-05-19',
    'pool_size': len(loaded),
    'parameters': {
        'momentum_days': 25,
        'top_n': 5,
        'hold_days': 20,
        'deduct_start_p52': 40,
        'deduct_coeff': 0.5
    },
    'years': {str(y): {
        'v41': round(v41_years[i], 2),
        'v3': round(v3_years[i], 2),
        'spy': round(spy_years[i], 2),
        'qqq': round(qqq_years[i], 2)
    } for i, y in enumerate(YEARS)},
    'cumulative': {
        'v41': round(cum_v41, 2),
        'v3': round(cum_v3, 2),
        'spy': round(cum_spy, 2),
        'qqq': round(cum_qqq, 2)
    },
    'stats': {
        'v41': v41_stats,
        'v3': v3_stats
    },
    'beat_spy_rate': f"{v41_vs_spy}/12",
    'beat_qqq_rate': f"{v41_vs_qqq}/12"
}

with open(OUTPUT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ 结果已保存: {OUTPUT}")
print("✅ 完成")
