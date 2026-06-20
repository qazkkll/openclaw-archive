#!/usr/bin/env python3
"""CEO诊断：分析模型在熊市失败的根因"""
import pandas as pd, numpy as np
import os, json
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

print("=== CEO 诊断报告 ===\n")

# 1. 数据质量诊断
df = pd.read_parquet('data/cn/features_v2.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date_int'] = df['date'].dt.strftime('%Y%m%d').astype(int)

print("1. 数据质量检查")
print(f"   总行数: {len(df):,}")
print(f"   股票数: {df['sym'].nunique()}")
print(f"   日期范围: {df['date'].min()} → {df['date'].max()}")

# 检查异常收益（单日>30%可能是涨跌停或复权问题）
df['daily_ret'] = df.groupby('sym')['close'].pct_change()
extreme = df[df['daily_ret'].abs() > 0.2]
print(f"   单日|收益|>20%的记录: {len(extreme)} ({len(extreme)/len(df)*100:.2f}%)")
if len(extreme) > 0:
    print(f"   最大单日涨幅: {extreme['daily_ret'].max()*100:.1f}%")
    print(f"   最大单日跌幅: {extreme['daily_ret'].min()*100:.1f}%")

# 2. 动量因子IC分析
print("\n2. 动量因子方向分析 (A股关键发现)")
print("   根据研究：A股短期价格动量是反转信号（负IC）")

features = ['r1','r5','r10','r20','d5','d10','d20','vol5','vol20','atr_pct',
    'vol_r','rsi14','macd','macd_sig','macd_hist','log_circ_mv',
    'turnover_20','sm_net_5','sm_net_20','md_net_5','md_net_20',
    'lg_net_5','lg_net_20','elg_net_5','elg_net_20','total_net_5','total_net_20']

# 取每个月末计算IC
monthly_dates = sorted(df[df['date'].dt.day >= 25]['date_int'].unique())
# 只取每月最后一个交易日
all_dates = sorted(df['date_int'].unique())
monthly_ends = []
for ym in sorted(set(d // 100 for d in all_dates)):
    m_dates = [d for d in all_dates if d // 100 == ym]
    if m_dates:
        monthly_ends.append(m_dates[-1])

ics = []
for md in monthly_ends[:-1]:  # 去掉最后一个月（没有未来收益）
    m_df = df[df['date_int'] == md].copy()
    if len(m_df) < 100:
        continue
    
    # 未来20天收益
    md_idx = all_dates.index(md)
    if md_idx + 20 >= len(all_dates):
        continue
    future_date = all_dates[md_idx + 20]
    future_df = df[df['date_int'] == future_date][['sym','close']].rename(columns={'close':'future_close'})
    
    merged = m_df.merge(future_df, on='sym')
    merged['fwd_ret'] = (merged['future_close'] - merged['close']) / merged['close']
    merged = merged.dropna(subset=['fwd_ret'])
    
    if len(merged) < 50:
        continue
    
    # 计算每个特征的Rank IC
    row = {'date': md}
    for feat in features:
        valid = merged.dropna(subset=[feat, 'fwd_ret'])
        if len(valid) > 50:
            row[feat] = valid[feat].corr(valid['fwd_ret'], method='spearman')
    ics.append(row)

ic_df = pd.DataFrame(ics)
print(f"\n   特征Rank IC (全时段均值):")
for feat in features:
    if feat in ic_df.columns:
        mean_ic = ic_df[feat].mean()
        icir = mean_ic / ic_df[feat].std() if ic_df[feat].std() > 0 else 0
        direction = "⚠️反转" if mean_ic < 0 else "✅正向"
        print(f"   {feat:>15}: IC={mean_ic:+.4f} ICIR={icir:+.3f} {direction}")

# 3. 市场状态分析
print("\n3. 市场状态分析")

# 计算全市场月度收益（代理大盘）
market_monthly = []
for md in monthly_ends[:-1]:
    m_df = df[df['date_int'] == md]
    md_idx = all_dates.index(md)
    if md_idx + 20 >= len(all_dates):
        continue
    future_date = all_dates[md_idx + 20]
    future_df = df[df['date_int'] == future_date][['sym','close']].rename(columns={'close':'future_close'})
    merged = m_df[['sym','close']].merge(future_df, on='sym')
    merged['fwd_ret'] = (merged['future_close'] - merged['close']) / merged['close']
    market_monthly.append({'date': md, 'market_ret': merged['fwd_ret'].mean()})

mm_df = pd.DataFrame(market_monthly)
mm_df['rolling_12m'] = mm_df['market_ret'].rolling(12).sum()

bull = mm_df[mm_df['rolling_12m'] > 0.2]
bear = mm_df[mm_df['rolling_12m'] < -0.1]
osc = mm_df[(mm_df['rolling_12m'] >= -0.1) & (mm_df['rolling_12m'] <= 0.2)]

print(f"   牛市(>20%): {len(bull)}月 ({len(bull)/len(mm_df)*100:.0f}%)")
print(f"   熊市(<-10%): {len(bear)}月 ({len(bear)/len(mm_df)*100:.0f}%)")
print(f"   震荡: {len(osc)}月 ({len(osc)/len(mm_df)*100:.0f}%)")

# 4. 负Alpha期分析
print("\n4. 负Alpha时期分析")
with open('research/paper_trade_sim.json') as f:
    sim = json.load(f)

neg_alpha = [p for p in sim['periods'] if p['alpha'] < 0]
pos_alpha = [p for p in sim['periods'] if p['alpha'] > 0]
print(f"   正Alpha: {len(pos_alpha)}期, 均值{np.mean([p['alpha'] for p in pos_alpha])*100:+.2f}%")
print(f"   负Alpha: {len(neg_alpha)}期, 均值{np.mean([p['alpha'] for p in neg_alpha])*100:+.2f}%")
print(f"\n   负Alpha时期详情:")
for p in neg_alpha:
    print(f"   {p['signal_date']}: 模型{p['port_ret']*100:+.1f}% 基准{p['bench_ret']*100:+.1f}% Alpha{p['alpha']*100:+.1f}%")

# 5. 小盘流动性风险
print("\n5. 市值分布分析")
latest = df[df['date_int'] == all_dates[-1]]
if 'circ_mv' in latest.columns:
    latest = latest.dropna(subset=['circ_mv'])
    print(f"   持仓股票平均市值: {latest['circ_mv'].mean():.0f}万")
    print(f"   中位数市值: {latest['circ_mv'].median():.0f}万")
    print(f"   <50亿: {(latest['circ_mv'] < 500000).sum()}只 ({(latest['circ_mv'] < 500000).sum()/len(latest)*100:.0f}%)")
    print(f"   >100亿: {(latest['circ_mv'] > 1000000).sum()}只 ({(latest['circ_mv'] > 1000000).sum()/len(latest)*100:.0f}%)")

print("\n=== 诊断完成 ===")
