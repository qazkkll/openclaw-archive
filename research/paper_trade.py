#!/usr/bin/env python3
"""A股模型V2 模拟验证"""
import json, pandas as pd, numpy as np
import os

os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=== A股模型V2 模拟验证 ===\n")

# 加载信号
with open('research/v2_signal.json') as f:
    signal = json.load(f)

top15 = pd.DataFrame(signal['top15'])
signal_date = int(signal['date'].replace('-',''))

print(f"信号日期: {signal['date']}")
print(f"模型 IC: {signal['wf']['ic']:.4f}, 多空: {signal['wf']['ls']:.4f}")
print(f"\nTop15建仓:")
for _, s in top15.iterrows():
    print(f"  {s['sym']} {s['name']}: ¥{s['close']:.2f} ({s['industry']})")

# 加载 features
print(f"\n加载特征数据...")
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date_int'] = df['date'].astype(int)
print(f"  {len(df):,}行, 日期: {df['date'].min()} → {df['date'].max()}")
print(f"  股票数: {df['sym'].nunique()}")

# 信号日和验证日
avail_dates = sorted(df['date_int'].unique())
sd_idx = next((i for i, d in enumerate(avail_dates) if d >= signal_date), None)
if sd_idx is None:
    print("信号日之后无数据"); exit(1)

eval_dates = avail_dates[sd_idx+1:min(sd_idx+22, len(avail_dates))]
print(f"  验证期: {eval_dates[0]} → {eval_dates[-1]} ({len(eval_dates)}天)")

# 建仓价
codes = set(top15['sym'].tolist())
entry_prices = dict(zip(top15['sym'], top15['close'].astype(float)))

# 逐日跟踪
print(f"\n{'='*70}")
print(f"{'日期':>10} {'组合收益':>10} {'持仓胜率':>10} {'最佳':>15} {'最差':>15}")
print(f"{'='*70}")

results = []
for date in eval_dates:
    day = df[(df['date_int'] == date) & (df['sym'].isin(codes))]
    
    rets = []
    details = {}
    for code in codes:
        ep = entry_prices[code]
        stock = day[day['sym'] == code]
        if len(stock) > 0:
            cp = stock.iloc[0]['close']
            ret = (cp - ep) / ep
            rets.append(ret)
            details[code] = ret
    
    if rets:
        cum_ret = np.mean(rets)
        wr = len([r for r in rets if r > 0]) / len(rets) * 100
        best = max(details.items(), key=lambda x: x[1])
        worst = min(details.items(), key=lambda x: x[1])
        print(f"{date:>10} {cum_ret*100:>+9.2f}% {wr:>9.0f}% {best[0]} {best[1]*100:>+8.1f}% {worst[0]} {worst[1]*100:>+8.1f}%")
        results.append({'date': date, 'cum_ret': cum_ret, 'wr': wr})

# 最终个股详情
print(f"\n{'='*70}")
last_date = eval_dates[-1]
last_day = df[(df['date_int'] == last_date) & (df['sym'].isin(codes))]

print(f"📈 个股表现 ({signal['date']} → {last_date}):")
stock_results = []
for _, s in top15.iterrows():
    code = s['sym']
    ep = float(s['close'])
    stock = last_day[last_day['sym'] == code]
    if len(stock) > 0:
        cp = stock.iloc[0]['close']
        ret = (cp - ep) / ep
        stock_results.append({'code': code, 'name': s['name'], 'entry': ep, 'exit': float(cp), 'ret': ret, 'industry': s['industry']})
    else:
        stock_results.append({'code': code, 'name': s['name'], 'entry': ep, 'exit': None, 'ret': 0, 'industry': s['industry']})

stock_results.sort(key=lambda x: x['ret'], reverse=True)
winners = [s for s in stock_results if s['ret'] > 0]
avg_ret = np.mean([s['ret'] for s in stock_results])

print(f"  胜率: {len(winners)}/15 = {len(winners)/15*100:.0f}%")
print(f"  平均收益: {avg_ret*100:+.2f}%")
print()
for s in stock_results:
    emoji = "🟢" if s['ret'] > 0 else "🔴"
    exit_p = f"¥{s['exit']:.2f}" if s['exit'] else "停牌"
    print(f"  {emoji} {s['code']} {s['name']}: ¥{s['entry']:.2f} → {exit_p} ({s['ret']*100:+.1f}%) [{s['industry']}]")

# 基准对比
bench_start = df[df['date_int'] == avail_dates[sd_idx]]
bench_end = df[df['date_int'] == last_date]
merged = bench_start[['sym','close']].merge(bench_end[['sym','close']], on='sym', suffixes=('_s','_e'))
if len(merged) > 0:
    bench_ret = ((merged['close_e'] - merged['close_s'])/merged['close_s']).mean()
    alpha = avg_ret - bench_ret
    print(f"\n📊 基准对比:")
    print(f"  模型组合: {avg_ret*100:+.2f}%")
    print(f"  全市场:   {bench_ret*100:+.2f}%")
    print(f"  Alpha:    {alpha*100:+.2f}%")

# 保存
with open('research/paper_trade_result.json', 'w') as f:
    json.dump({
        'signal_date': signal['date'],
        'eval_date': str(last_date),
        'days': len(eval_dates),
        'portfolio_return': round(avg_ret*100, 2),
        'win_rate': round(len(winners)/15*100, 0),
        'benchmark_return': round(bench_ret*100, 2) if len(merged) > 0 else None,
        'alpha': round(alpha*100, 2) if len(merged) > 0 else None,
        'stocks': [{k: v for k, v in s.items()} for s in stock_results]
    }, f, indent=2, default=str)
print(f"\n✅ 结果已保存 research/paper_trade_result.json")
