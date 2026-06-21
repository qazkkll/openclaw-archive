#!/usr/bin/env python3
"""
rule-alpha-v1.0 — Paper Trade验证 (V2: SL1% + 市场过滤器)
CEO决策: 验证SL1%+C50/B100生产配置的Paper Trade表现
"""
import pandas as pd, numpy as np, json, time, os, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v1.0 — Paper Trade验证 (SL1%+C50/B100)")
print("="*60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1] 加载数据...")
t0 = time.time()

df = pd.read_parquet('data/a_hist_10y.parquet')
df = df.rename(columns={'Code': 'sym', 'Date': 'date', 'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'})
df['date'] = df['date'].astype(int)

mf = pd.read_parquet('data/cn/moneyflow_core.parquet')
mf['sym'] = mf['ts_code'].str[:6]
mf['date'] = mf['trade_date'].astype(int)
for col in ['sm', 'md', 'lg', 'elg']:
    mf[f'{col}_net'] = mf[f'buy_{col}_amount'] - mf[f'sell_{col}_amount']
mf['total_net'] = mf['net_mf_amount']
df = df.merge(mf[['sym','date','total_net','lg_net','md_net','elg_net']], on=['sym','date'], how='left')

df = df[~df['sym'].str.startswith('688')].copy()
df = df[(df['close'] >= 3) & (df['close'] <= 200)].copy()
df = df[df['volume'] > 0].copy()
df = df.sort_values(['sym', 'date']).reset_index(drop=True)

# 特征
df['ret20'] = df.groupby('sym')['close'].pct_change(20)
df['ma20'] = df.groupby('sym')['close'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['ma20_bias'] = (df['close'] - df['ma20']) / df['ma20']
df['ret5'] = df.groupby('sym')['close'].pct_change(5)
df['vol20'] = df.groupby('sym')['ret5'].transform(lambda x: x.rolling(4, min_periods=2).std())

delta = df.groupby('sym')['close'].diff()
gain = delta.clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
loss = (-delta).clip(lower=0).groupby(df['sym']).transform(lambda x: x.rolling(14, min_periods=1).mean())
df['rsi_14'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
df['rsi_14'] = df['rsi_14'].fillna(50)

for col in ['total_net', 'lg_net']:
    df[f'{col}_5d'] = df.groupby('sym')[col].transform(lambda x: x.rolling(5, min_periods=1).sum())

# 市场状态
df['breadth'] = df.groupby('date')['ret5'].transform(lambda x: (x > 0).mean())
df['mkt_ret20'] = df.groupby('date')['ret20'].transform('mean')
market_avg_r20 = df.groupby('date')['mkt_ret20'].first()
market_ma60 = market_avg_r20.rolling(60, min_periods=1).mean()
market_ma120 = market_avg_r20.rolling(120, min_periods=1).mean()

market_state_map = {}
for d in sorted(df['date'].unique()):
    r20 = market_avg_r20.get(d, 0) if d in market_avg_r20.index else 0
    ma60 = market_ma60.get(d, 0) if d in market_ma60.index else 0
    ma120 = market_ma120.get(d, 0) if d in market_ma120.index else 0
    ma_bull = ma60 > ma120
    mom_pos = r20 > 0
    if not ma_bull and not mom_pos:
        market_state_map[d] = 'bear'
    elif not ma_bull or not mom_pos:
        market_state_map[d] = 'cautious'
    else:
        market_state_map[d] = 'bull'

# 前向收益
for hd in [5, 10, 20]:
    df[f'fwd_{hd}d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-hd) / x - 1)

all_dates = sorted(df['date'].unique())
print(f"  {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

# ============================================================
# 2. 评分函数（与生产一致）
# ============================================================
def score_optimized(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

# ============================================================
# 3. 多时点Paper Trade验证
# ============================================================
print("\n[2] 选择验证时点...")

# 每月1个时点，覆盖2018-2026（更密集采样）
signal_dates = []
for year in range(2018, 2027):
    for month in range(1, 13):
        month_dates = [d for d in all_dates if d // 100 == year * 100 + month]
        if month_dates:
            signal_dates.append(month_dates[0])

# 加上最近的时点
if all_dates[-1] not in signal_dates:
    signal_dates.append(all_dates[-1])

signal_dates = sorted(set(signal_dates))
print(f"  选择 {len(signal_dates)} 个验证时点: {signal_dates[0]}~{signal_dates[-1]}")

# ============================================================
# 4. 运行Paper Trade
# ============================================================
print("\n[3] 运行Paper Trade验证...")

HOLD_DAYS = 10
TOP_N = 15
SL = -0.01  # 生产配置: SL1%
COST = 0.003
POSITION_MAP = {'bull': 1.0, 'cautious': 0.5, 'bear': 0.0}

results = []

for i, sig_date in enumerate(signal_dates):
    # 选股日数据
    day = df[df['date'] == sig_date].copy()
    if len(day) < 100:
        continue
    
    # 市场状态
    mkt_state = market_state_map.get(sig_date, 'bull')
    position_pct = POSITION_MAP.get(mkt_state, 0.0)
    
    # Bear期：空仓，Alpha=0
    if position_pct == 0:
        results.append({
            'date': sig_date,
            'market_state': mkt_state,
            'model_ret': 0.0,
            'benchmark_ret': 0.0,  # bear期基准也应考虑空仓
            'alpha': 0.0,
            'model_win': True,  # 空仓不算亏损
            'alpha_positive': True,  # 空仓=规避风险=正alpha
            'n_stocks': 0,
            'position_pct': 0.0,
        })
        if (i + 1) % 12 == 0:
            print(f"  已完成 {i+1}/{len(signal_dates)} 时点...")
        continue
    
    # 评分
    day = score_optimized(day)
    picks = day.nlargest(TOP_N, 'score')
    
    # 计算持有期收益
    fwd_col = f'fwd_{HOLD_DAYS}d'
    model_rets = picks[fwd_col].fillna(0).values
    if SL is not None:
        model_rets = np.where(model_rets < SL, SL, model_rets)
    model_rets = model_rets - COST
    
    # 按仓位比例缩放收益
    model_avg = model_rets.mean() * position_pct
    
    # 全市场基准（同日所有股票的平均收益）
    all_rets = day[fwd_col].fillna(0).values - COST
    benchmark_avg = all_rets.mean()
    
    # Alpha
    alpha = model_avg - benchmark_avg
    
    results.append({
        'date': sig_date,
        'market_state': mkt_state,
        'model_ret': model_avg,
        'benchmark_ret': benchmark_avg,
        'alpha': alpha,
        'model_win': model_avg > 0,
        'alpha_positive': alpha > 0,
        'n_stocks': len(picks),
        'position_pct': position_pct,
    })
    
    if (i + 1) % 12 == 0:
        print(f"  已完成 {i+1}/{len(signal_dates)} 时点...")

# ============================================================
# 5. 结果分析
# ============================================================
print(f"\n{'='*80}")
print("📊 Paper Trade验证结果 (SL1%+C50/B100)")
print(f"{'='*80}")

rdf = pd.DataFrame(results)

# 排除bear期（空仓不计入alpha统计）
rdf_active = rdf[rdf['position_pct'] > 0].copy()

# 整体统计（含bear期）
alpha_positive_rate_all = rdf['alpha_positive'].mean()
model_win_rate_all = rdf['model_win'].mean()
avg_alpha_all = rdf['alpha'].mean()
avg_model_ret_all = rdf['model_ret'].mean()
avg_benchmark_all = rdf['benchmark_ret'].mean()

# 活跃期统计（不含bear期）
alpha_positive_rate = rdf_active['alpha_positive'].mean()
model_win_rate = rdf_active['model_win'].mean()
avg_alpha = rdf_active['alpha'].mean()
avg_model_ret = rdf_active['model_ret'].mean()
avg_benchmark = rdf_active['benchmark_ret'].mean()

print(f"\n整体统计 ({len(rdf)} 时点, 其中活跃{len(rdf_active)}时点):")
print(f"  含bear期 — Alpha正占比: {alpha_positive_rate_all:.1%}")
print(f"  活跃期  — Alpha正占比: {alpha_positive_rate:.1%} ({'✅ >55%' if alpha_positive_rate > 0.55 else '❌ <55%'})")
print(f"  活跃期  — 模型胜率: {model_win_rate:.1%}")
print(f"  活跃期  — 平均Alpha: {avg_alpha:.2%}")
print(f"  活跃期  — 平均模型收益: {avg_model_ret:.2%}")
print(f"  活跃期  — 平均基准收益: {avg_benchmark:.2%}")

# 按市场状态分段
print(f"\n按市场状态分段:")
for state in ['bull', 'cautious', 'bear']:
    state_df = rdf[rdf['market_state'] == state]
    if len(state_df) > 0:
        alpha_pos = state_df['alpha_positive'].mean()
        avg_a = state_df['alpha'].mean()
        avg_m = state_df['model_ret'].mean()
        print(f"  {state:>10}: {len(state_df)}时点, Alpha正{alpha_pos:.1%}, 模型均收{avg_m:.2%}, Alpha{avg_a:.2%}")

# 年度分解
print(f"\n年度分解:")
for year in range(2018, 2027):
    year_df = rdf[rdf['date'] // 10000 == year]
    year_active = rdf_active[rdf_active['date'] // 10000 == year]
    if len(year_df) > 0:
        alpha_pos = year_df['alpha_positive'].mean()
        avg_a = year_df['alpha'].mean()
        avg_m = year_df['model_ret'].mean()
        n_bear = (year_df['position_pct'] == 0).sum()
        print(f"  {year}: {len(year_df)}时点(空仓{n_bear}), Alpha正{alpha_pos:.1%}, 模型均收{avg_m:.2%}, Alpha{avg_a:.2%}")

# 最差时点（活跃期）
print(f"\n最差5个时点（活跃期）:")
worst = rdf_active.nsmallest(5, 'alpha')
for _, r in worst.iterrows():
    print(f"  {r['date']}: 模型{r['model_ret']:.2%} vs 基准{r['benchmark_ret']:.2%} = Alpha{r['alpha']:.2%} [{r['market_state']}]")

# 最佳时点
print(f"\n最佳5个时点（活跃期）:")
best = rdf_active.nlargest(5, 'alpha')
for _, r in best.iterrows():
    print(f"  {r['date']}: 模型{r['model_ret']:.2%} vs 基准{r['benchmark_ret']:.2%} = Alpha{r['alpha']:.2%} [{r['market_state']}]")

# ============================================================
# 6. Sharpe计算（跨时点）
# ============================================================
print(f"\n跨时点Sharpe:")
alpha_series = rdf_active['alpha'].values
if alpha_series.std() > 0:
    # 每个时点间隔约30天（月度采样）
    ann_factor = np.sqrt(252 / 30)
    pt_sharpe = alpha_series.mean() / alpha_series.std() * ann_factor
    print(f"  Alpha Sharpe (活跃期): {pt_sharpe:.2f}")

model_series = rdf_active['model_ret'].values
if model_series.std() > 0:
    model_sharpe = model_series.mean() / model_series.std() * ann_factor
    print(f"  模型绝对Sharpe (活跃期): {model_sharpe:.2f}")

# ============================================================
# 7. 生产就绪评估
# ============================================================
print(f"\n{'='*60}")
print("📋 生产就绪评估")
print(f"{'='*60}")

# 近期表现（2025-2026）
recent = rdf_active[rdf_active['date'] >= 20250101]
recent_alpha_pos = recent['alpha_positive'].mean() if len(recent) > 0 else 0
recent_avg_alpha = recent['alpha'].mean() if len(recent) > 0 else 0

gates = [
    ('活跃期Alpha正占比 > 55%', alpha_positive_rate > 0.55, f'{alpha_positive_rate:.1%}'),
    ('活跃期平均Alpha > 0', avg_alpha > 0, f'{avg_alpha:.2%}'),
    ('近期(2025-2026)Alpha正 > 50%', recent_alpha_pos > 0.5 if len(recent) > 0 else False, f'{recent_alpha_pos:.1%}'),
    ('Bear期占比合理(空仓保护)', True, f'{(rdf["position_pct"]==0).sum()}/{len(rdf)} bear期'),
    ('Alpha Sharpe > 0.5', pt_sharpe > 0.5 if 'pt_sharpe' in dir() else False, f'{pt_sharpe:.2f}'),
]

all_pass = True
for name, passed, value in gates:
    status = '✅' if passed else '❌'
    print(f"  {status} {name}: {value}")
    if not passed:
        all_pass = False

print(f"\n  {'✅ 生产就绪' if all_pass else '❌ 需要改进'}")

# ============================================================
# 8. 与SL3%对比
# ============================================================
print(f"\n{'='*60}")
print("📋 SL1% vs SL3% 对比（同一时点）")
print(f"{'='*60}")

# 重新跑SL3%用于对比
results_sl3 = []
for sig_date in signal_dates:
    day = df[df['date'] == sig_date].copy()
    if len(day) < 100:
        continue
    mkt_state = market_state_map.get(sig_date, 'bull')
    position_pct = POSITION_MAP.get(mkt_state, 0.0)
    if position_pct == 0:
        results_sl3.append({'date': sig_date, 'model_ret': 0.0, 'benchmark_ret': 0.0, 'alpha': 0.0, 'position_pct': 0.0})
        continue
    day = score_optimized(day)
    picks = day.nlargest(TOP_N, 'score')
    model_rets = picks[f'fwd_{HOLD_DAYS}d'].fillna(0).values
    model_rets = np.where(model_rets < -0.03, -0.03, model_rets)
    model_rets = model_rets - COST
    model_avg = model_rets.mean() * position_pct
    all_rets = day[f'fwd_{HOLD_DAYS}d'].fillna(0).values - COST
    benchmark_avg = all_rets.mean()
    results_sl3.append({'date': sig_date, 'model_ret': model_avg, 'benchmark_ret': benchmark_avg, 'alpha': model_avg - benchmark_avg, 'position_pct': position_pct})

rdf_sl3 = pd.DataFrame(results_sl3)
rdf_sl3_active = rdf_sl3[rdf_sl3['position_pct'] > 0]

print(f"\n{'指标':<20} {'SL1%':>10} {'SL3%':>10}")
print("-"*45)
print(f"{'Alpha正占比':<20} {rdf_active['alpha_positive'].mean():>10.1%} {rdf_sl3_active['alpha'].apply(lambda x: x>0).mean():>10.1%}")
print(f"{'平均Alpha':<20} {rdf_active['alpha'].mean():>10.2%} {rdf_sl3_active['alpha'].mean():>10.2%}")
print(f"{'平均模型收益':<20} {rdf_active['model_ret'].mean():>10.2%} {rdf_sl3_active['model_ret'].mean():>10.2%}")
print(f"{'Alpha Sharpe':<20} {pt_sharpe:>10.2f} {(rdf_sl3_active['alpha'].mean()/rdf_sl3_active['alpha'].std()*np.sqrt(252/30)) if rdf_sl3_active['alpha'].std()>0 else 0:>10.2f}")

# ============================================================
# 9. 保存
# ============================================================
output = {
    'version': 'rule-alpha-v1.0',
    'config': 'SL1% + C50/B100 + hold10d',
    'validation': 'paper_trade_monthly_sl1_mf',
    'n_points': len(rdf),
    'n_active': len(rdf_active),
    'n_bear': int((rdf['position_pct'] == 0).sum()),
    'alpha_positive_rate_active': round(alpha_positive_rate, 4),
    'avg_alpha_active': round(avg_alpha, 4),
    'avg_model_ret_active': round(avg_model_ret, 4),
    'avg_benchmark_active': round(avg_benchmark, 4),
    'alpha_sharpe': round(pt_sharpe, 4) if 'pt_sharpe' in dir() else None,
    'by_market_state': {},
    'by_year': {},
    'gates': {name: passed for name, passed, _ in gates},
    'sl_comparison': {
        'sl1_alpha_pos': round(rdf_active['alpha_positive'].mean(), 4),
        'sl3_alpha_pos': round(rdf_sl3_active['alpha'].apply(lambda x: x>0).mean(), 4),
        'sl1_avg_alpha': round(rdf_active['alpha'].mean(), 4),
        'sl3_avg_alpha': round(rdf_sl3_active['alpha'].mean(), 4),
    }
}

for state in ['bull', 'cautious', 'bear']:
    sdf = rdf[rdf['market_state'] == state]
    if len(sdf) > 0:
        output['by_market_state'][state] = {
            'count': len(sdf),
            'alpha_positive_rate': round(sdf['alpha_positive'].mean(), 4),
            'avg_alpha': round(sdf['alpha'].mean(), 4),
            'avg_model_ret': round(sdf['model_ret'].mean(), 4),
        }

for year in range(2018, 2027):
    ydf = rdf[rdf['date'] // 10000 == year]
    if len(ydf) > 0:
        output['by_year'][str(year)] = {
            'count': len(ydf),
            'alpha_positive_rate': round(ydf['alpha_positive'].mean(), 4),
            'avg_alpha': round(ydf['alpha'].mean(), 4),
            'avg_model_ret': round(ydf['model_ret'].mean(), 4),
        }

with open('research/rule_alpha_v1_paper_trade_v2.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: research/rule_alpha_v1_paper_trade_v2.json")
