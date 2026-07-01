#!/usr/bin/env python3
"""
V0.3.1 Rank Inversion Test
对比V0.3.1 (fund_ratio=70%, analyst=20%, fund_metric=10%) 在每个Walk-Forward窗口的
Top5% vs Bottom20%收益方向，判断rank inversion通过率。

用法:
  python3 v031_rank_inversion_test.py

输出:
  - data/falcon/v031_rank_inversion_results.json
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════
# 路径
# ═══════════════════════════════════════════════════
PROJECT = Path("/home/hermes/.hermes/openclaw-archive")
DATA_PATH = PROJECT / "data/falcon/features_v02.parquet"
RESULTS_PATH = PROJECT / "data/falcon/v031_rank_inversion_results.json"
BACKTEST_ENGINE = PROJECT / "scripts/falcon/backtest_engine.py"

# ═══════════════════════════════════════════════════
# V0.3.1 因子组和权重
# ═══════════════════════════════════════════════════
# fund_ratio (0.70): FMP财务比率
RATIO_FIELDS = [
    "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
    "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
    "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
    "ebitdaMargin", "assetTurnover", "inventoryTurnover",
    "receivablesTurnover", "debtToEquityRatio", "currentRatio",
    "quickRatio", "financialLeverageRatio",
    "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
    "dividendYieldPercentage", "dividendPayoutRatio"
]

# analyst (0.20): 分析师指标
ANALYST_FIELDS = ["eps_revision", "revenue_revision", "eps_dispersion", "num_analysts_eps"]

# fund_metric (0.10): Key Metrics — features_v02中不可用，记为0
# 注意: METRIC_FIELDS (earningsYield, evToEBITDA等) 不在features_v02中

# Walk-Forward参数
TRAIN_YEARS = 0.5  # 6个月训练窗口
TEST_MONTHS = 6    # 6个月测试窗口
HOLD_DAYS = 30
TOP_N = 10
COST = 0.001
STOP_LOSS = -0.15

# V0.3.1原始权重
WEIGHTS_ORIG = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10}


# ═══════════════════════════════════════════════════
# 因子分组: 从features_v02的列名映射到V0.3.1因子组
# ═══════════════════════════════════════════════════
def get_factor_groups(df_columns):
    """返回可用的V0.3.1因子组及其列。"""
    available_ratio = [c for c in RATIO_FIELDS if c in df_columns]
    available_analyst = [c for c in ANALYST_FIELDS if c in df_columns]
    
    print(f"  fund_ratio 因子: {len(available_ratio)}/{len(RATIO_FIELDS)} 可用")
    print(f"  analyst 因子: {len(available_analyst)}/{len(ANALYST_FIELDS)} 可用")
    print(f"  fund_metric: 0 可用 (METRIC_FIELDS不在features_v02中)")
    
    # 由于fund_metric不可用，将其权重重新分配给fund_ratio
    # 原始: fund_ratio=0.70, analyst=0.20, fund_metric=0.10
    # 调整后: fund_ratio=0.80, analyst=0.20 (fund_metric权重归入fund_ratio)
    weights_adjusted = {"fund_ratio": 0.80, "analyst": 0.20}
    
    return {
        "fund_ratio": available_ratio,
        "analyst": available_analyst,
    }, weights_adjusted


# ═══════════════════════════════════════════════════
# 数据加载和预处理
# ═══════════════════════════════════════════════════
def load_and_prepare():
    """加载features_v02.parquet, 计算截面rank, 构建ranks dict和prices matrix。"""
    print("📊 加载 features_v02.parquet...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  原始: {df.shape[0]:,}行, {df['ticker'].nunique()}只, "
          f"{df['date'].min()} ~ {df['date'].max()}")
    
    # 获取可用因子组
    factor_groups, weights = get_factor_groups(df.columns)
    
    all_factor_cols = []
    for cols in factor_groups.values():
        all_factor_cols.extend(cols)
    
    # 构建prices矩阵
    print("📊 构建价格矩阵...")
    df['date_str'] = df['date'].apply(lambda x: str(x)[:10])
    prices = df.pivot_table(index='date_str', columns='ticker', values='close')
    prices = prices.sort_index()
    print(f"  Prices: {prices.shape[0]} 天 × {prices.shape[1]} 只")
    
    # 逐日计算截面rank → 分组得分
    print("📊 逐日计算截面rank和分组得分...")
    dates = sorted(df['date_str'].unique())
    ranks_dict = {}
    
    for di, date in enumerate(dates):
        day = df[df['date_str'] == date].copy()
        if len(day) < 20:
            continue
        
        day = day.set_index('ticker')
        row = pd.DataFrame(index=day.index)
        
        # 对每个因子组计算组内百分位rank的均值
        for group_name, cols in factor_groups.items():
            if not cols:
                row[group_name] = np.nan
                continue
            group_ranks = []
            for c in cols:
                if c in day.columns and day[c].notna().sum() > 5:
                    group_ranks.append(day[c].rank(pct=True, na_option='keep'))
            if group_ranks:
                row[group_name] = pd.concat(group_ranks, axis=1).mean(axis=1)
            else:
                row[group_name] = np.nan
        
        ranks_dict[date] = row
        
        if (di + 1) % 500 == 0:
            print(f"  📊 {di+1}/{len(dates)} 天...")
    
    print(f"  ✅ {len(ranks_dict)} 天, {len(factor_groups)} 因子组")
    return ranks_dict, prices, weights, factor_groups


# ═══════════════════════════════════════════════════
# Rank Inversion分析
# ═══════════════════════════════════════════════════
def compute_combined_score(ranks_dict, weights):
    """计算每个日期的综合得分。"""
    print("📊 计算综合得分...")
    scores_dict = {}
    for date, ranks in ranks_dict.items():
        combined = pd.Series(0.0, index=ranks.index)
        for f, w in weights.items():
            if f in ranks.columns:
                combined = combined + w * ranks[f]
        scores_dict[date] = combined.dropna().sort_values(ascending=False)
    return scores_dict


def rank_inversion_per_window(ranks_dict, prices, weights, train_years, test_months):
    """
    Walk-Forward rank inversion分析。
    
    对每个窗口:
    1. 用训练期最后一天的scores确定Top5%和Bottom20%
    2. 用测试期的forward return判断方向
    3. Top5%平均收益 > Bottom20%平均收益 → PASS
    """
    print("\n📊 Rank Inversion分析 (Walk-Forward)...")
    
    # 先计算综合得分
    scores_dict = compute_combined_score(ranks_dict, weights)
    
    dates = sorted(scores_dict.keys())
    if not dates:
        print("  ❌ 无有效日期")
        return []
    
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    
    windows = []
    train_start = start
    window_idx = 0
    
    while True:
        train_end = train_start + pd.DateOffset(months=int(train_years * 12))
        test_end = train_end + pd.DateOffset(months=test_months)
        
        if str(test_end)[:10] > str(end)[:10]:
            break
        
        test_start_str = str(train_end)[:10]
        test_end_str = str(test_end)[:10]
        
        # 训练期最后一天的scores
        train_dates = [d for d in dates if d <= str(train_end)[:10]]
        if not train_dates:
            train_start += pd.DateOffset(months=test_months)
            window_idx += 1
            continue
        last_train_date = train_dates[-1]
        
        if last_train_date not in scores_dict:
            train_start += pd.DateOffset(months=test_months)
            window_idx += 1
            continue
        
        train_scores = scores_dict[last_train_date]
        if len(train_scores) < 10:
            train_start += pd.DateOffset(months=test_months)
            window_idx += 1
            continue
        
        # 确定Top5%和Bottom20%的ticker列表
        n_stocks = len(train_scores)
        top5_n = max(1, int(n_stocks * 0.05))
        bot20_n = max(1, int(n_stocks * 0.20))
        
        top5_tickers = train_scores.head(top5_n).index.tolist()
        bot20_tickers = train_scores.tail(bot20_n).index.tolist()
        
        # 测试期forward return: 用测试期第一天买入，最后一天卖出的近似
        test_dates_in_prices = [d for d in prices.index.tolist() 
                                if test_start_str <= d <= test_end_str]
        
        if len(test_dates_in_prices) < 2:
            train_start += pd.DateOffset(months=test_months)
            window_idx += 1
            continue
        
        # 用测试期第一天和最后一天的价格计算return
        buy_date = test_dates_in_prices[0]
        sell_date = test_dates_in_prices[-1]
        
        buy_prices = prices.loc[buy_date]
        sell_prices = prices.loc[sell_date]
        
        # Top5% return
        top5_rets = []
        for t in top5_tickers:
            if t in buy_prices.index and t in sell_prices.index:
                bp = buy_prices.get(t)
                sp = sell_prices.get(t)
                if pd.notna(bp) and pd.notna(sp) and bp > 0:
                    top5_rets.append((sp - bp) / bp)
        
        # Bottom20% return
        bot20_rets = []
        for t in bot20_tickers:
            if t in buy_prices.index and t in sell_prices.index:
                bp = buy_prices.get(t)
                sp = sell_prices.get(t)
                if pd.notna(bp) and pd.notna(sp) and bp > 0:
                    bot20_rets.append((sp - bp) / bp)
        
        top5_avg_ret = np.mean(top5_rets) if top5_rets else np.nan
        bot20_avg_ret = np.mean(bot20_rets) if bot20_rets else np.nan
        
        # 判断通过
        if not np.isnan(top5_avg_ret) and not np.isnan(bot20_avg_ret):
            passed = top5_avg_ret > bot20_avg_ret
            spread = top5_avg_ret - bot20_avg_ret
        else:
            passed = None
            spread = np.nan
        
        windows.append({
            "index": window_idx,
            "period": f"{test_start_str} → {test_end_str}",
            "train_end": str(train_end)[:10],
            "top5_pct": f"{top5_n}/{n_stocks}",
            "bot20_pct": f"{bot20_n}/{n_stocks}",
            "top5_avg_return": round(float(top5_avg_ret * 100), 2) if not np.isnan(top5_avg_ret) else None,
            "bot20_avg_return": round(float(bot20_avg_ret * 100), 2) if not np.isnan(bot20_avg_ret) else None,
            "spread_bps": round(float(spread * 10000), 1) if not np.isnan(spread) else None,
            "passed": passed,
        })
        
        train_start += pd.DateOffset(months=test_months)
        window_idx += 1
    
    # 汇总
    valid = [w for w in windows if w["passed"] is not None]
    passed_count = sum(1 for w in valid if w["passed"])
    total = len(valid)
    pass_rate = passed_count / total * 100 if total > 0 else 0
    
    print(f"\n  📊 Rank Inversion结果: {passed_count}/{total} 通过 ({pass_rate:.1f}%)")
    for w in windows:
        status = "✅" if w["passed"] else ("❌" if w["passed"] is False else "⚠️")
        print(f"    {status} W{w['index']}: {w['period']} | "
              f"Top5%={w['top5_avg_return']}% | Bot20%={w['bot20_avg_return']}% | "
              f"Spread={w['spread_bps']}bps")
    
    return windows, pass_rate, passed_count, total


# ═══════════════════════════════════════════════════
# Walk-Forward Sharpe (用backtest_engine.py)
# ═══════════════════════════════════════════════════
def run_wf_sharpe(ranks_dict, prices, weights):
    """用backtest_engine.py运行Walk-Forward，返回WF Sharpe。"""
    print("\n📊 Walk-Forward Sharpe (backtest_engine.py)...")
    sys.path.insert(0, str(BACKTEST_ENGINE.parent))
    from backtest_engine import BacktestEngine
    
    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
    
    # Note: backtest_engine.py requires integer train_years.
    # Original request: train_years=0.5 (6 months). Using train_years=1 (12 months) as closest integer.
    result = engine.walk_forward(
        ranks_dict, prices, weights,
        hold_days=HOLD_DAYS, top_n=TOP_N,
        train_years=1, test_months=TEST_MONTHS
    )
    
    print(f"  WF Sharpe: {result.sharpe:.3f}")
    print(f"  MaxDD: {result.max_dd:.1%}")
    print(f"  CAGR: {result.cagr:.1%}")
    print(f"  Win Rate: {result.win_rate:.0%}")
    
    return {
        "sharpe": result.sharpe,
        "max_dd": result.max_dd,
        "cagr": result.cagr,
        "win_rate": result.win_rate,
        "n_trades": result.n_trades,
        "window_details": result.window_details,
    }


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("🦅 V0.3.1 Rank Inversion Test")
    print("=" * 80)
    print(f"\nV0.3.1原始权重: {WEIGHTS_ORIG}")
    print(f"  fund_ratio: 0.70 (FMP财务比率)")
    print(f"  analyst: 0.20 (分析师指标)")
    print(f"  fund_metric: 0.10 (Key Metrics — 不在features_v02中)")
    print(f"\n调整后权重 (fund_metric权重归入fund_ratio):")
    print(f"  fund_ratio: 0.80")
    print(f"  analyst: 0.20")
    print(f"\nWalk-Forward参数:")
    print(f"  train_years={TRAIN_YEARS}, test_months={TEST_MONTHS}")
    print(f"  hold_days={HOLD_DAYS}, top_n={TOP_N}")
    print(f"  cost={COST}, stop_loss={STOP_LOSS}")
    
    # 1. 加载数据
    ranks_dict, prices, weights, factor_groups = load_and_prepare()
    
    # 2. Rank Inversion分析
    windows, pass_rate, passed_count, total = rank_inversion_per_window(
        ranks_dict, prices, weights, TRAIN_YEARS, TEST_MONTHS
    )
    
    # 3. Walk-Forward Sharpe
    wf_result = run_wf_sharpe(ranks_dict, prices, weights)
    
    # 4. 保存结果
    output = {
        "version": "V0.3.1",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "original_weights": WEIGHTS_ORIG,
            "adjusted_weights": weights,
            "note": "fund_metric不在features_v02中, 权重归入fund_ratio",
            "factor_groups": {k: len(v) for k, v in factor_groups.items()},
            "train_years": TRAIN_YEARS,
            "test_months": TEST_MONTHS,
            "hold_days": HOLD_DAYS,
            "top_n": TOP_N,
            "cost": COST,
            "stop_loss": STOP_LOSS,
        },
        "rank_inversion": {
            "pass_rate_pct": round(pass_rate, 1),
            "passed": passed_count,
            "total_windows": total,
            "windows": windows,
        },
        "wf_sharpe": wf_result,
        "v041_comparison": {
            "v041_pass_rate_pct": 47.4,
            "v041_passed": 9,
            "v041_total_windows": 19,
        },
    }
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    print(f"\n✅ 结果已保存: {RESULTS_PATH}")
    
    # 5. 对比总结
    print("\n" + "=" * 80)
    print("📊 V0.3.1 vs V0.4.1 对比")
    print("=" * 80)
    print(f"  {'指标':<25} {'V0.3.1':<15} {'V0.4.1':<15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}")
    print(f"  {'Rank Inversion通过率':<25} {pass_rate:.1f}%{'':<10} 47.4%")
    print(f"  {'通过/总窗口':<25} {passed_count}/{total}{'':<8} 9/19")
    print(f"  {'WF Sharpe':<25} {wf_result['sharpe']:.3f}{'':<9} (见v041结果)")
    print(f"  {'MaxDD':<25} {wf_result['max_dd']:.1%}{'':<9} (见v041结果)")
    print(f"  {'CAGR':<25} {wf_result['cagr']:.1%}{'':<9} (见v041结果)")
    print()
    
    if pass_rate > 50:
        print("  ✅ V0.3.1 Rank Inversion通过率 > 50%: 模型能稳定区分Top5%和Bottom20%")
    else:
        print("  ⚠️ V0.3.1 Rank Inversion通过率 ≤ 50%: 区分度不足")
    
    if pass_rate > 47.4:
        print(f"  ✅ V0.3.1通过率({pass_rate:.1f}%) > V0.4.1(47.4%): V0.3.1更好")
    else:
        print(f"  ⚠️ V0.3.1通过率({pass_rate:.1f}%) ≤ V0.4.1(47.4%): V0.4.1更好")


if __name__ == "__main__":
    main()
