#!/usr/bin/env python3
"""
analyst_count 因子 IC 验证
用analyst_historical的历史数据，Walk-Forward计算analyst_count的IC/ICIR
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"

# 加载数据
print("加载数据...")
master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
master["date"] = master["date"].astype(str)
dates = sorted(master["date"].unique())

analyst_data = json.load(open(DATA_DIR / "analyst_historical.json"))

# 构建analyst_count的时间序列
print("构建analyst_count历史...")
# 对每个交易日，获取每只股票的analyst_count
# 用PIT方式：只用date之前已有的数据
all_counts = {}
for ticker, records in analyst_data.items():
    for r in records:
        d = r.get("date", "")
        n = r.get("numAnalystsEps")
        if d and n is not None:
            if ticker not in all_counts:
                all_counts[ticker] = {}
            all_counts[ticker][d] = int(n)

print(f"  覆盖: {len(all_counts)} 只股票")

# Walk-Forward IC计算
# 用60天调仓周期，每60天计算一次截面IC
# IC = Spearman rank correlation(factor_rank, forward_return)
HOLD_DAYS = 60
lookback = 252  # 用于计算IC的窗口

# 生成调仓日（每60天）
rebal_dates = dates[::HOLD_DAYS]
print(f"  调仓日: {len(rebal_dates)} 个")

# 构建价格矩阵
price_pivot = master.pivot_table(index="date", columns="ticker", values="close")
price_pivot = price_pivot.sort_index()

ics = []
for i, rebal_date in enumerate(rebal_dates[:-1]):  # 最后一个没法算forward return
    next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else None
    if not next_rebal:
        continue
    
    # 找rebal_date对应的analyst_count
    count_vals = {}
    for ticker in master[master["date"] == rebal_date]["ticker"].unique():
        if ticker not in all_counts:
            continue
        # PIT: 取rebal_date之前最近的值
        prior_dates = [d for d in all_counts[ticker].keys() if d <= rebal_date]
        if prior_dates:
            latest = max(prior_dates)
            count_vals[ticker] = all_counts[ticker][latest]
    
    if len(count_vals) < 50:
        continue
    
    # 计算forward return (rebal_date -> next_rebal)
    fwd_ret = {}
    for ticker in count_vals:
        try:
            p0 = price_pivot.loc[rebal_date, ticker]
            p1 = price_pivot.loc[next_rebal, ticker]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret[ticker] = (p1 - p0) / p0
        except:
            continue
    
    # 取交集
    common = set(count_vals.keys()) & set(fwd_ret.keys())
    if len(common) < 30:
        continue
    
    factor_rank = pd.Series({t: count_vals[t] for t in common}).rank(pct=True)
    return_rank = pd.Series({t: fwd_ret[t] for t in common}).rank(pct=True)
    
    ic = factor_rank.corr(return_rank, method="spearman")
    ics.append({"date": rebal_date, "ic": ic, "n": len(common)})

ic_df = pd.DataFrame(ics)
print(f"\n=== analyst_count IC 验证 ===")
print(f"样本: {len(ic_df)} 个调仓窗口")
print(f"平均IC: {ic_df['ic'].mean():.4f}")
print(f"IC标准差: {ic_df['ic'].std():.4f}")
icir = ic_df['ic'].mean() / ic_df['ic'].std() if ic_df['ic'].std() > 0 else 0
print(f"ICIR: {icir:.4f}")
t_stat = ic_df['ic'].mean() / (ic_df['ic'].std() / np.sqrt(len(ic_df))) if ic_df['ic'].std() > 0 else 0
print(f"t-stat: {t_stat:.3f}")
win = (ic_df['ic'] > 0).mean()
print(f"IC>0 胜率: {win:.1%}")

# 也测试：analyst_count作为质量过滤器的效果
# 高覆盖 vs 低覆盖 的平均forward return
print(f"\n=== 高覆盖 vs 低覆盖 分析 ===")
high_rets = []
low_rets = []
for _, row in ic_df.iterrows():
    d = row["date"]
    next_idx = list(ic_df["date"]).index(d) + 1
    if next_idx >= len(rebal_dates):
        continue
    next_d = rebal_dates[next_idx]
    
    count_vals = {}
    fwd_ret = {}
    for ticker in master[master["date"] == d]["ticker"].unique():
        if ticker not in all_counts:
            continue
        prior_dates_list = [dd for dd in all_counts[ticker].keys() if dd <= d]
        if prior_dates_list:
            count_vals[ticker] = all_counts[ticker][max(prior_dates_list)]
        try:
            p0 = price_pivot.loc[d, ticker]
            p1 = price_pivot.loc[next_d, ticker]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                fwd_ret[ticker] = (p1 - p0) / p0
        except:
            continue
    
    common = set(count_vals.keys()) & set(fwd_ret.keys())
    if len(common) < 30:
        continue
    
    median_count = np.median([count_vals[t] for t in common])
    high = [fwd_ret[t] for t in common if count_vals[t] >= median_count]
    low = [fwd_ret[t] for t in common if count_vals[t] < median_count]
    
    if high and low:
        high_rets.append(np.mean(high))
        low_rets.append(np.mean(low))

if high_rets:
    print(f"高覆盖(>中位数)平均60天收益: {np.mean(high_rets):.2%}")
    print(f"低覆盖(<中位数)平均60天收益: {np.mean(low_rets):.2%}")
    print(f"差异(high-low): {np.mean(high_rets) - np.mean(low_rets):.2%}")

# 结论
print(f"\n=== 结论 ===")
if abs(icir) >= 0.05 and abs(t_stat) >= 1.96:
    print(f"✅ analyst_count 有效 (ICIR={icir:.4f}, t={t_stat:.3f})，建议纳入评分")
elif abs(icir) >= 0.05:
    print(f"⚠️ analyst_count 边缘有效 (ICIR={icir:.4f}, t={t_stat:.3f})，可考虑纳入")
else:
    print(f"❌ analyst_count 无效 (ICIR={icir:.4f}, t={t_stat:.3f})，不纳入评分")
