#!/usr/bin/env python3
"""
rule-alpha-v2.1 因子衰减分析
检查各因子在不同时期的IC变化，识别因子是否在衰减
"""
import pandas as pd, numpy as np, json, time, os, datetime
import warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("="*60)
print("rule-alpha-v2.1 因子衰减分析")
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

# 未来收益
df['fwd_10d'] = df.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

print(f"  {len(df):,}行, {df['sym'].nunique()}只, {time.time()-t0:.0f}秒")

# ============================================================
# 2. 因子定义
# ============================================================
factors = {
    'reversal': lambda d: (-d['ret20'].fillna(0)).clip(-0.3, 0.3),
    'flow_rank': lambda d: d['total_net_5d'].fillna(0).rank(pct=True),
    'low_vol': lambda d: 1 - d['vol20'].fillna(d['vol20'].median()).rank(pct=True),
    'rsi_oversold': lambda d: (d['rsi_14'].fillna(50) < 35).astype(float),
    'lg_flow': lambda d: d['lg_net_5d'].fillna(0).rank(pct=True),
    'ma_bias': lambda d: (-d['ma20_bias'].fillna(0)).clip(-0.2, 0.2),
}

# ============================================================
# 3. 按年计算IC
# ============================================================
print("\n[2] 按年计算因子IC...")

df['year'] = df['date'] // 10000
years = sorted(df['year'].unique())

ic_by_year = {}
for year in years:
    df_year = df[df['year'] == year].copy()
    if len(df_year) < 1000:
        continue
    
    ic_by_year[year] = {}
    for factor_name, factor_fn in factors.items():
        try:
            factor_values = factor_fn(df_year)
            valid_mask = factor_values.notna() & df_year['fwd_10d'].notna()
            if valid_mask.sum() < 100:
                ic_by_year[year][factor_name] = np.nan
                continue
            
            # Rank IC (Spearman)
            from scipy import stats
            ic, _ = stats.spearmanr(factor_values[valid_mask], df_year['fwd_10d'][valid_mask])
            ic_by_year[year][factor_name] = ic
        except Exception as e:
            ic_by_year[year][factor_name] = np.nan

# ============================================================
# 4. 结果汇总
# ============================================================
print("\n" + "="*80)
print("📊 因子IC年度变化")
print("="*80)

# 打印表格
print(f"\n{'年份':>6}", end='')
for factor_name in factors.keys():
    print(f" {factor_name:>14}", end='')
print()
print("-" * (6 + 15 * len(factors)))

for year in years:
    if year not in ic_by_year:
        continue
    print(f"{year:>6}", end='')
    for factor_name in factors.keys():
        ic = ic_by_year[year].get(factor_name, np.nan)
        if np.isnan(ic):
            print(f" {'N/A':>14}", end='')
        else:
            # Color coding
            if ic > 0.05:
                marker = "✅"
            elif ic > 0:
                marker = "🟡"
            else:
                marker = "❌"
            print(f" {marker}{ic:>10.4f}", end='')
    print()

# 计算平均IC和趋势
print(f"\n{'平均':>6}", end='')
for factor_name in factors.keys():
    ics = [ic_by_year[y].get(factor_name, np.nan) for y in years if y in ic_by_year]
    ics = [x for x in ics if not np.isnan(x)]
    if ics:
        avg_ic = np.mean(ics)
        print(f" {avg_ic:>14.4f}", end='')
    else:
        print(f" {'N/A':>14}", end='')
print()

# 计算最近3年vs前3年的变化
print(f"\n📊 因子衰减分析:")
for factor_name in factors.keys():
    early_ics = [ic_by_year[y].get(factor_name, np.nan) for y in years[:3] if y in ic_by_year]
    late_ics = [ic_by_year[y].get(factor_name, np.nan) for y in years[-3:] if y in ic_by_year]
    
    early_ics = [x for x in early_ics if not np.isnan(x)]
    late_ics = [x for x in late_ics if not np.isnan(x)]
    
    if early_ics and late_ics:
        early_avg = np.mean(early_ics)
        late_avg = np.mean(late_ics)
        change = late_avg - early_avg
        change_pct = change / abs(early_avg) * 100 if early_avg != 0 else 0
        
        status = "✅稳定" if abs(change_pct) < 30 else ("⚠️衰减" if change < 0 else "📈增强")
        print(f"  {factor_name:<15}: 前3年={early_avg:.4f}, 后3年={late_avg:.4f}, 变化={change_pct:+.1f}% {status}")

# ============================================================
# 5. 组合因子IC
# ============================================================
print(f"\n📊 组合因子IC（v2.1评分函数）:")

def score_v1(day):
    s = day.copy()
    s['score'] = 0.0
    s['score'] += (-s['ret20'].fillna(0)).clip(-0.3, 0.3) * 3
    s['score'] += s['total_net_5d'].fillna(0).rank(pct=True) * 2
    s['score'] += (1 - s['vol20'].fillna(s['vol20'].median()).rank(pct=True)) * 2
    s['score'] += (s['rsi_14'].fillna(50) < 35).astype(float) * 1.5
    s['score'] += s['lg_net_5d'].fillna(0).rank(pct=True) * 1
    s['score'] += (-s['ma20_bias'].fillna(0)).clip(-0.2, 0.2) * 1
    return s

for year in years:
    if year not in ic_by_year:
        continue
    df_year = df[df['year'] == year].copy()
    if len(df_year) < 1000:
        continue
    
    scored = score_v1(df_year)
    valid_mask = scored['score'].notna() & df_year['fwd_10d'].notna()
    
    if valid_mask.sum() < 100:
        continue
    
    from scipy import stats
    ic, _ = stats.spearmanr(scored['score'][valid_mask], df_year['fwd_10d'][valid_mask])
    
    status = "✅" if ic > 0.05 else ("🟡" if ic > 0 else "❌")
    print(f"  {year}: IC={ic:.4f} {status}")

# ============================================================
# 6. 保存结果
# ============================================================
output = {
    'experiment': 'rule-alpha-v2.1-factor-decay',
    'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
    'ic_by_year': {str(k): v for k, v in ic_by_year.items()},
    'factors': list(factors.keys()),
}

with open('research/rule_alpha_v2_1_factor_decay.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n结果已保存: research/rule_alpha_v2_1_factor_decay.json")
print("="*60)
print("CEO决策: 因子衰减分析完成")
print("="*60)
