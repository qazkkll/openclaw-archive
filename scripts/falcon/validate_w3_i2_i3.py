#!/usr/bin/env python3
"""
验证W3/I2/I3参数 — 用Falcon ScoringEngine生成历史评分序列
W3: 信号退化阈值 (entry_score, degrade_score, days_held)
I2: VIX regime有效性
I3: 信号过期阈值 (多少天后评分预测力衰减)
"""
import sys, json, os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# 加入scripts路径 (falcon_system是包, 从scripts目录导入)
PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from falcon_system.core.data_manager import DataManager
from falcon_system.engine.scorer import ScoringEngine

# ══════════════════════════════════════════════
# 1. 生成历史评分序列
# ══════════════════════════════════════════════

print("=" * 70)
print("Falcon历史评分验证 (W3/I2/I3)")
print("=" * 70)

dm = DataManager()
engine = ScoringEngine(dm)

# Monkey-patch: 跳过实时价格覆盖(历史回测不需要, 而且yfinance在WSL2不可用)
engine._override_with_realtime_prices = lambda signals: None

# 加载价格数据获取可用日期
master = dm.load_master_prices()
all_dates = sorted(master["date"].unique())
print(f"可用交易日: {len(all_dates)}天 ({all_dates[0]} ~ {all_dates[-1]})")

# 每10天一个评分日(最近200天, 约20个)
score_dates = []
for i in range(len(all_dates) - 1, max(0, len(all_dates) - 200), -10):
    score_dates.append(all_dates[i])
score_dates.reverse()
print(f"评分日期: {len(score_dates)}个 (每10天一次)")

# 价格矩阵(用于计算未来收益)
price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()

# VIX数据 — 优先从parquet读, 否则用SPY 20日波动率做proxy
try:
    if "vix" in master.columns:
        vix_series = master[master["ticker"] == "VIX"][["date", "close"]].set_index("date")["close"]
        vix_series.index = pd.to_datetime(vix_series.index)
        print(f"VIX数据(parquet): {len(vix_series)}天")
    else:
        raise ValueError("no vix column")
except:
    # 用SPY 20日滚动波动率做proxy
    try:
        spy = price_pivot.get("SPY")
        if spy is not None:
            spy_returns = spy.pct_change()
            vix_proxy = spy_returns.rolling(20).std() * np.sqrt(252) * 100
            vix_series = vix_proxy.dropna()
            print(f"VIX proxy(SPY vol): {len(vix_series)}天")
        else:
            vix_series = pd.Series()
            print("⚠️ 无VIX数据")
    except:
        vix_series = pd.Series()
        print("⚠️ VIX数据获取失败")

# 批量生成评分
print(f"\n--- 开始批量评分 ({len(score_dates)}个日期) ---")
all_scores = {}  # {date: {ticker: score}}

for i, sdate in enumerate(score_dates):
    try:
        result = engine.score(target_date=sdate, universe="spx")
        date_scores = {}
        for sig in result.signals:
            date_scores[sig.ticker] = {
                "score": sig.score,
                "rank_pct": sig.rank_pct,
                "signal": sig.signal_type,
                "close": sig.close,
            }
        all_scores[sdate] = date_scores
        if (i + 1) % 10 == 0:
            print(f"  已评分: {i+1}/{len(score_dates)} ({sdate}, {len(date_scores)}只)")
    except Exception as e:
        print(f"  跳过 {sdate}: {e}")

print(f"成功评分: {len(all_scores)}个日期")

# ══════════════════════════════════════════════
# 2. W3验证: 信号退化→价格表现
# ══════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("W3: 信号退化验证")
print("=" * 70)

# 构建每只股票的评分时间序列
ticker_history = {}  # {ticker: [(date, score), ...]}
for sdate, scores in sorted(all_scores.items()):
    for ticker, info in scores.items():
        if ticker not in ticker_history:
            ticker_history[ticker] = []
        ticker_history[ticker].append((sdate, info["score"]))

# 找退化案例: 入场score高 → 后续score大幅下降
degradation_cases = []
sorted_score_dates = sorted(all_scores.keys())

for ticker, history in ticker_history.items():
    history.sort(key=lambda x: x[0])
    
    for i in range(len(history)):
        entry_date, entry_score = history[i]
        
        # 只看"买入信号"级别的评分
        if entry_score < 0.50:
            continue
        
        # 找后续评分
        for j in range(i + 1, len(history)):
            later_date, later_score = history[j]
            days_between = (datetime.strptime(later_date, "%Y-%m-%d") - 
                          datetime.strptime(entry_date, "%Y-%m-%d")).days
            
            if days_between < 3:
                continue
            
            score_drop = entry_score - later_score
            
            # 计算后续14天价格表现
            if ticker in price_pivot.columns:
                future_prices = price_pivot[ticker].loc[
                    price_pivot.index > later_date
                ].dropna().head(14)
                
                current_prices = price_pivot[ticker].loc[
                    price_pivot.index <= later_date
                ].dropna()
                
                if len(future_prices) >= 5 and len(current_prices) > 0:
                    p0 = current_prices.iloc[-1]
                    future_pnl_14d = (future_prices.iloc[-1] - p0) / p0
                    future_pnl_7d = (future_prices.iloc[min(6, len(future_prices)-1)] - p0) / p0 if len(future_prices) >= 3 else np.nan
                    
                    degradation_cases.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "entry_score": entry_score,
                        "degrade_date": later_date,
                        "degrade_score": later_score,
                        "score_drop": score_drop,
                        "days_between": days_between,
                        "future_7d_pnl": future_pnl_7d,
                        "future_14d_pnl": future_pnl_14d,
                    })
            break  # 只取第一次退化

print(f"退化案例总数: {len(degradation_cases)}")

if degradation_cases:
    df_deg = pd.DataFrame(degradation_cases)
    
    # 测试不同退化阈值
    print(f"\n--- 不同退化阈值对比 ---")
    print(f"{'退化到<':>10s} | {'案例数':>6s} | {'14天均盈亏':>10s} | {'14天胜率':>8s} | {'7天均盈亏':>9s} | {'对照组(未退化)':>14s}")
    print("-" * 80)
    
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50]:
        subset = df_deg[df_deg["degrade_score"] < thresh]
        # 对照组: 评分保持在阈值以上的同期股票
        control = df_deg[df_deg["degrade_score"] >= thresh]
        
        if len(subset) < 3:
            print(f"  <{thresh:.2f}    | {len(subset):6d} |       数据不足")
            continue
        
        avg_14d = subset["future_14d_pnl"].mean()
        win_14d = (subset["future_14d_pnl"] > 0).mean()
        avg_7d = subset["future_7d_pnl"].mean()
        
        ctrl_14d = control["future_14d_pnl"].mean() if len(control) > 0 else np.nan
        
        print(f"  <{thresh:.2f}    | {len(subset):6d} | {avg_14d:+10.2%} | {win_14d:+8.1%} | {avg_7d:+9.2%} | {ctrl_14d:+14.2%}")
    
    # 测试不同持有天数阈值
    print(f"\n--- 不同持有天数阈值 (退化到<0.40) ---")
    for min_days in [3, 5, 7, 10, 14]:
        subset = df_deg[(df_deg["degrade_score"] < 0.40) & (df_deg["days_between"] >= min_days)]
        if len(subset) < 3:
            print(f"  >={min_days}天  | {len(subset):6d} | 数据不足")
            continue
        avg_14d = subset["future_14d_pnl"].mean()
        win_14d = (subset["future_14d_pnl"] > 0).mean()
        print(f"  >={min_days:2d}天  | {len(subset):6d} | {avg_14d:+10.2%} | {win_14d:+8.1%}")
    
    # 最优W3参数
    print(f"\n--- W3最优参数推荐 ---")
    best_thresh = None
    best_improvement = -999
    
    for thresh in [0.30, 0.35, 0.40, 0.45]:
        degraded = df_deg[df_deg["degrade_score"] < thresh]
        not_degraded = df_deg[df_deg["degrade_score"] >= thresh]
        if len(degraded) < 5 or len(not_degraded) < 5:
            continue
        # 改善度 = 退化组卖出后避免的亏损 vs 不退化组的收益
        improvement = not_degraded["future_14d_pnl"].mean() - degraded["future_14d_pnl"].mean()
        if improvement > best_improvement:
            best_improvement = improvement
            best_thresh = thresh
    
    if best_thresh:
        print(f"  推荐阈值: degrade_score < {best_thresh}")
        print(f"  理由: 退化组vs未退化组14天盈亏差异 = {best_improvement:+.2%}")
    else:
        print(f"  结论: 数据不足或退化无显著预测力, 建议暂不启用W3")

# ══════════════════════════════════════════════
# 3. I3验证: 信号衰减曲线
# ══════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("I3: 信号衰减曲线 (多少天后评分预测力消失)")
print("=" * 70)

# 对每个评分日, 计算Top5股票在1/3/5/7/14/30天后的收益
decay_results = {d: [] for d in [1, 3, 5, 7, 14, 21, 30]}

for sdate, scores in sorted(all_scores.items()):
    # 取Top5
    sorted_picks = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
    
    for ticker, info in sorted_picks:
        if ticker not in price_pivot.columns:
            continue
        
        entry_score = info["score"]
        if entry_score < 0.50:
            continue
        
        # 获取评分日的价格
        if sdate not in price_pivot.index:
            continue
        entry_price = price_pivot[ticker].loc[sdate]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        
        # 计算不同持有期的收益
        future_all = price_pivot[ticker].loc[price_pivot.index > sdate].dropna()
        
        for days in [1, 3, 5, 7, 14, 21, 30]:
            if len(future_all) >= days:
                exit_price = future_all.iloc[days - 1]
                pnl = (exit_price - entry_price) / entry_price
                decay_results[days].append({
                    "date": sdate,
                    "ticker": ticker,
                    "score": entry_score,
                    "pnl": pnl,
                })

print(f"\n--- Top5信号在不同持有期的表现 ---")
print(f"{'持有天数':>8s} | {'样本数':>6s} | {'平均盈亏':>8s} | {'胜率':>6s} | {'Sharpe':>7s} | {'IC(评分vs收益)':>14s}")
print("-" * 70)

ic_by_days = {}
for days in [1, 3, 5, 7, 14, 21, 30]:
    data = decay_results[days]
    if not data:
        continue
    df_d = pd.DataFrame(data)
    avg_pnl = df_d["pnl"].mean()
    win_rate = (df_d["pnl"] > 0).mean()
    sharpe = avg_pnl / df_d["pnl"].std() if df_d["pnl"].std() > 0 else 0
    # IC: score和pnl的rank相关
    if len(df_d) > 10:
        ic = df_d["score"].corr(df_d["pnl"])
    else:
        ic = np.nan
    
    ic_by_days[days] = ic
    print(f"  {days:6d}天 | {len(data):6d} | {avg_pnl:+8.2%} | {win_rate:+6.1%} | {sharpe:+7.3f} | {ic:+14.3f}")

# 找IC衰减到不显著的点
print(f"\n--- I3信号衰减结论 ---")
if ic_by_days:
    # IC > 0.05认为还有预测力
    last_significant = max(d for d, ic in ic_by_days.items() if ic > 0.05) if any(ic > 0.05 for ic in ic_by_days.values()) else 0
    # IC降到初始一半的天数
    initial_ic = ic_by_days.get(1, ic_by_days.get(3, 0))
    half_life = min(d for d, ic in ic_by_days.items() if ic < initial_ic * 0.5) if initial_ic > 0 else 0
    
    print(f"  IC>0.05的最大持有天数: {last_significant}天")
    print(f"  IC半衰期: {half_life}天")
    print(f"  推荐I3过期阈值: {max(last_significant, 7)}天")

# ══════════════════════════════════════════════
# 4. I2验证: VIX regime有效性
# ══════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("I2: VIX regime有效性验证")
print("=" * 70)

if not vix_series.empty:
    # 对每个评分日, 查VIX值, 分regime, 看后续收益
    regime_results = {
        "bull": [],      # VIX < 20
        "neutral": [],   # 20-25
        "bear": [],      # 25-30
        "extreme_bear": [],  # > 30
    }
    
    for sdate, scores in sorted(all_scores.items()):
        # 查VIX
        try:
            vix_date = pd.to_datetime(sdate)
            # 找最近的VIX值
            nearby_vix = vix_series.loc[vix_series.index <= vix_date]
            if len(nearby_vix) == 0:
                continue
            vix_val = float(nearby_vix.iloc[-1])
        except:
            continue
        
        # 分regime
        if vix_val < 20:
            regime = "bull"
        elif vix_val < 25:
            regime = "neutral"
        elif vix_val < 30:
            regime = "bear"
        else:
            regime = "extreme_bear"
        
        # Top5后续收益
        sorted_picks = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)[:5]
        for ticker, info in sorted_picks:
            if ticker not in price_pivot.columns:
                continue
            if sdate not in price_pivot.index:
                continue
            
            entry_price = price_pivot[ticker].loc[sdate]
            if pd.isna(entry_price) or entry_price <= 0:
                continue
            
            future = price_pivot[ticker].loc[price_pivot.index > sdate].dropna().head(30)
            if len(future) >= 14:
                pnl_14d = (future.iloc[13] - entry_price) / entry_price
                pnl_30d = (future.iloc[-1] - entry_price) / entry_price
                regime_results[regime].append({
                    "date": sdate,
                    "ticker": ticker,
                    "vix": vix_val,
                    "score": info["score"],
                    "pnl_14d": pnl_14d,
                    "pnl_30d": pnl_30d,
                })
    
    print(f"\n--- VIX regime下Top5信号的30天表现 ---")
    print(f"{'Regime':>15s} | {'VIX范围':>10s} | {'案例数':>6s} | {'14天均盈亏':>10s} | {'30天均盈亏':>10s} | {'胜率':>6s}")
    print("-" * 75)
    
    for regime, label, vix_range in [
        ("bull", "🟢 Bull", "<20"),
        ("neutral", "🟡 Neutral", "20-25"),
        ("bear", "🟠 Bear", "25-30"),
        ("extreme_bear", "🔴 Extreme", ">30"),
    ]:
        data = regime_results[regime]
        if not data:
            print(f"  {label:15s} | {vix_range:>10s} |      0 |        N/A |        N/A |    N/A")
            continue
        df_r = pd.DataFrame(data)
        avg_14 = df_r["pnl_14d"].mean()
        avg_30 = df_r["pnl_30d"].mean()
        win_30 = (df_r["pnl_30d"] > 0).mean()
        print(f"  {label:15s} | {vix_range:>10s} | {len(data):6d} | {avg_14:+10.2%} | {avg_30:+10.2%} | {win_30:+6.1%}")
    
    # 检验: extreme_bear是否显著差于bull
    bull_data = pd.DataFrame(regime_results["bull"])["pnl_30d"] if regime_results["bull"] else pd.Series()
    bear_data = pd.DataFrame(regime_results["extreme_bear"])["pnl_30d"] if regime_results["extreme_bear"] else pd.Series()
    
    if len(bull_data) > 5 and len(bear_data) > 5:
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(bull_data, bear_data)
        print(f"\n  t检验 (bull vs extreme_bear): t={t_stat:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"  ✅ VIX regime对收益有显著影响 (p<0.05)")
        else:
            print(f"  ⚠️ VIX regime对收益影响不显著 (p={p_val:.4f})")
    else:
        print(f"\n  数据不足, 无法做统计检验")
else:
    print("  ⚠️ 无VIX数据, 跳过I2验证")

# ══════════════════════════════════════════════
# 5. 综合结论
# ══════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("综合结论")
print("=" * 70)

conclusions = {
    "W3": {
        "status": "NEEDS_DATA",
        "note": "见上方退化阈值分析",
    },
    "I2": {
        "status": "NEEDS_DATA",
        "note": "见上方VIX regime分析",
    },
    "I3": {
        "status": "NEEDS_DATA",
        "note": "见上方信号衰减曲线",
    },
}

# 保存结果
output_file = PROJECT_ROOT / "data" / "falcon" / "w3_i2_i3_validation.json"
with open(output_file, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "score_dates_count": len(all_scores),
        "degradation_cases": len(degradation_cases),
        "conclusions": conclusions,
    }, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存: {output_file}")
