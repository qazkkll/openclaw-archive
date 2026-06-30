"""
T3.1 动态线性模型：IC自适应权重的线性模型

Walk-Forward验证:
  - 训练窗口: 5年(从2016开始)
  - 测试窗口: 6个月
  - 每月重算因子权重(基于最近6个月IC)
  - 权重 = recent_IC / sum(|recent_IC|)  (signed IC, normalized by abs sum)
  - score = sum(weights[f] * factor_rank[f] for f in factors)

使用 backtest_engine.py 回测，禁止自行实现回测逻辑。
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", message=".*constant.*")
warnings.filterwarnings("ignore", message=".*correlation coefficient.*")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, BacktestResult


# ═══════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════

DATA_PATH = PROJECT_ROOT / "data" / "falcon" / "training_data_v04.parquet"
IC_PATH = PROJECT_ROOT / "data" / "falcon" / "v04_ic_analysis.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "falcon" / "v04_dynamic_linear_results.json"

# Walk-Forward参数
TRAIN_YEARS = 5
TEST_MONTHS = 6
HOLD_DAYS = 30
TOP_N = 10
COST = 0.001
STOP_LOSS = -0.15

# IC计算参数
IC_LOOKBACK_MONTHS = 6  # 最近6个月IC
IC_MIN_DATES = 40       # 最少需要40个交易日的IC数据
MIN_FACTOR_COVERAGE = 0.3  # 因子在训练期的最低覆盖率(用于过滤低质量因子)


# ═══════════════════════════════════════════════════════════════════
#  数据加载与预处理
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """加载训练数据和IC分析结果。"""
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} cols")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Tickers: {df['ticker'].nunique()}, Dates: {df['date'].nunique()}")

    with open(IC_PATH) as f:
        ic_data = json.load(f)

    # Factor columns: exclude price/volume/returns/meta
    exclude = {
        'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap',
        'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d', 'fwd_ret_30d',
        'fmp_covered', 'analyst_covered',
        'news_avg_sentiment', 'news_sentiment_vol', 'news_neg_ratio',
        'news_pos_ratio', 'news_article_count', 'news_confidence_avg',
    }
    factor_cols = [c for c in df.columns if c not in exclude]
    print(f"  Factor columns: {len(factor_cols)}")

    return df, ic_data, factor_cols


def compute_cross_sectional_ranks(df, factor_cols):
    """
    对每个日期的每个因子做截面percentile ranking。
    返回: {date_str: DataFrame(ticker→factor_ranks)}
    """
    print("Computing cross-sectional ranks...")
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: str(x))

    rank_dict = {}
    dates = sorted(df['date_str'].unique())

    for d in dates:
        day_df = df[df['date_str'] == d].copy()
        if len(day_df) < 20:
            continue
        day_df = day_df.set_index('ticker')

        ranks = pd.DataFrame(index=day_df.index)
        for f in factor_cols:
            if f in day_df.columns:
                ranks[f] = day_df[f].rank(pct=True, na_option='keep')
            else:
                ranks[f] = np.nan

        rank_dict[d] = ranks

    print(f"  Built ranks for {len(rank_dict)} dates")
    return rank_dict


def build_prices_df(df):
    """
    构建价格矩阵: index=date_str, columns=ticker, values=close
    """
    print("Building prices matrix...")
    prices = df.pivot_table(index=df['date'].apply(lambda x: str(x)),
                           columns='ticker',
                           values='close')
    prices = prices.sort_index()
    print(f"  Prices: {prices.shape[0]} dates × {prices.shape[1]} tickers")
    return prices


# ═══════════════════════════════════════════════════════════════════
#  IC计算
# ═══════════════════════════════════════════════════════════════════

def compute_monthly_ic(df, factor_cols):
    """
    计算每个月的IC (基于该月所有交易日的截面IC均值)。

    返回: {month_str: {factor: ic_value}}
    month_str format: 'YYYY-MM'
    """
    print("Computing monthly IC...")
    df = df.copy()
    df['date_str'] = df['date'].apply(lambda x: str(x))
    df['month'] = df['date'].apply(lambda x: x.strftime('%Y-%m'))

    months = sorted(df['month'].unique())
    monthly_ic = {}

    for m in months:
        month_data = df[df['month'] == m]
        dates_in_month = sorted(month_data['date_str'].unique())

        ic_dict = {}
        for f in factor_cols:
            daily_ics = []
            for d in dates_in_month:
                day_data = month_data[month_data['date_str'] == d][[f, 'fwd_ret_30d']].dropna()
                if len(day_data) < 20:
                    continue
                try:
                    ic, _ = spearmanr(day_data[f].values, day_data['fwd_ret_30d'].values)
                    if not np.isnan(ic):
                        daily_ics.append(ic)
                except Exception:
                    pass

            if len(daily_ics) >= 5:
                ic_dict[f] = float(np.mean(daily_ics))

        if ic_dict:
            monthly_ic[m] = ic_dict

    print(f"  Computed monthly IC for {len(months)} months")
    return monthly_ic


def compute_factor_coverage(df, factor_cols, start_date, end_date):
    """
    计算因子在指定时间窗口内的覆盖率(非NaN比例)。
    返回: {factor: coverage_ratio}
    """
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    subset = df[mask]

    if len(subset) == 0:
        return {f: 0.0 for f in factor_cols}

    coverage = {}
    for f in factor_cols:
        if f in subset.columns:
            coverage[f] = float(subset[f].notna().mean())
        else:
            coverage[f] = 0.0
    return coverage


def compute_dynamic_weights(monthly_ic, factor_cols, target_month,
                            lookback_months=6, training_coverage=None):
    """
    基于最近lookback_months个月的IC计算动态权重。

    权重 = recent_IC / sum(|recent_IC|)
    使用signed IC，绝对值归一化确保权重和为1。

    如果提供了training_coverage，只使用覆盖率>阈值的因子。
    """
    months = sorted(monthly_ic.keys())
    target_idx = months.index(target_month) if target_month in months else -1

    if target_idx < lookback_months:
        lookback_months = target_idx

    if lookback_months <= 0:
        return None

    recent_months = months[target_idx - lookback_months:target_idx]

    # Average IC over recent months
    ic_sums = {}
    ic_counts = {}
    for m in recent_months:
        for f, ic in monthly_ic[m].items():
            ic_sums[f] = ic_sums.get(f, 0) + ic
            ic_counts[f] = ic_counts.get(f, 0) + 1

    avg_ic = {}
    for f in ic_sums:
        if ic_counts[f] >= 3:
            avg_ic[f] = ic_sums[f] / ic_counts[f]

    if not avg_ic:
        return None

    # Filter to available factors with adequate coverage
    available = {}
    for f, ic in avg_ic.items():
        if f not in factor_cols:
            continue
        # Check coverage if provided
        if training_coverage is not None:
            if training_coverage.get(f, 0) < MIN_FACTOR_COVERAGE:
                continue
        available[f] = ic

    if not available:
        return None

    # Compute weights: signed IC / sum(|IC|)
    abs_sum = sum(abs(v) for v in available.values())
    if abs_sum < 1e-8:
        n = len(available)
        return {f: 1.0 / n for f in available}

    weights = {f: v / abs_sum for f, v in available.items()}
    return weights


# ═══════════════════════════════════════════════════════════════════
#  Walk-Forward 验证
# ═══════════════════════════════════════════════════════════════════

def run_walk_forward(df, rank_dict, prices, factor_cols, monthly_ic):
    """
    Walk-Forward验证:
    - 训练窗口: 5年
    - 测试窗口: 6个月
    - 每个窗口: 基于训练期IC计算权重 → 在测试期回测
    - 训练期内做因子覆盖率过滤

    使用 backtest_engine.py 的 run() 方法。
    """
    print("\n" + "=" * 60)
    print("Walk-Forward Validation")
    print("=" * 60)

    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)

    all_dates = sorted(prices.index.tolist())
    start_date = pd.Timestamp(all_dates[0])
    end_date = pd.Timestamp(all_dates[-1])

    windows = []
    train_start = start_date
    window_idx = 0

    while True:
        train_end = train_start + pd.DateOffset(years=TRAIN_YEARS)
        test_end = train_end + pd.DateOffset(months=TEST_MONTHS)

        if test_end > end_date:
            break

        test_start_str = str(train_end.date())
        test_end_str = str(test_end.date())
        train_end_month = train_end.strftime('%Y-%m')

        print(f"\n--- Window {window_idx}: Test {test_start_str} → {test_end_str} ---")

        # Compute factor coverage in training period
        train_start_date = train_start.date()
        train_end_date = train_end.date()
        training_coverage = compute_factor_coverage(df, factor_cols, train_start_date, train_end_date)
        low_cov_factors = {f: c for f, c in training_coverage.items() if c < MIN_FACTOR_COVERAGE}
        if low_cov_factors:
            print(f"    Filtered out {len(low_cov_factors)} low-coverage factors: {list(low_cov_factors.keys())[:5]}...")

        # Compute weights based on training period IC
        weights = compute_dynamic_weights(monthly_ic, factor_cols, train_end_month,
                                          lookback_months=IC_LOOKBACK_MONTHS,
                                          training_coverage=training_coverage)

        if weights is None:
            print(f"    ⚠️ No valid weights, skipping window")
            windows.append({
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "error": "No valid weights",
            })
            window_idx += 1
            train_start += pd.DateOffset(months=TEST_MONTHS)
            continue

        n_factors = len(weights)
        top3 = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f"    Factors with weights: {n_factors}")
        print(f"    Top 3: {[(f, round(w, 4)) for f, w in top3]}")

        # Run backtest for this window
        try:
            result, baseline = engine.run(
                rank_dict, prices, weights,
                hold_days=HOLD_DAYS, top_n=TOP_N,
                start_date=test_start_str, end_date=test_end_str,
                run_baseline=True
            )

            print(f"    Sharpe={result.sharpe:.3f}  MaxDD={result.max_dd:.1%}  "
                  f"CAGR={result.cagr:.1%}  WR={result.win_rate:.0%}  "
                  f"Trades={result.n_trades}")
            if baseline:
                print(f"    Baseline: Sharpe={baseline.sharpe:.3f}")

            window_info = {
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "sharpe": result.sharpe,
                "max_dd": result.max_dd,
                "cagr": result.cagr,
                "win_rate": result.win_rate,
                "n_trades": result.n_trades,
                "n_factors": n_factors,
                "weights": {f: round(w, 4) for f, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:10]},
            }
            if baseline:
                window_info["baseline_sharpe"] = baseline.sharpe

            windows.append(window_info)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            windows.append({
                "index": window_idx,
                "period": f"{test_start_str} → {test_end_str}",
                "error": str(e),
            })

        window_idx += 1
        train_start += pd.DateOffset(months=TEST_MONTHS)

    return windows


def aggregate_results(windows):
    """汇总Walk-Forward结果。"""
    valid = [w for w in windows if "sharpe" in w]

    if not valid:
        return {"error": "No valid windows"}

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]
    trades = [w["n_trades"] for w in valid]

    agg_sharpe = float(np.mean(sharpes))
    agg_max_dd = float(np.min(dds))
    agg_cagr = float(np.mean(cagrs))
    agg_wr = float(np.mean(wrs))
    total_trades = sum(trades)

    positive_sharpe_pct = sum(1 for s in sharpes if s > 0) / len(sharpes)

    recent = valid[-3:] if len(valid) >= 3 else valid
    recent_sharpes = [w["sharpe"] for w in recent]

    return {
        "wf_sharpe": round(agg_sharpe, 3),
        "wf_max_dd": round(agg_max_dd, 4),
        "wf_cagr": round(agg_cagr, 4),
        "wf_win_rate": round(agg_wr, 3),
        "wf_total_trades": total_trades,
        "wf_n_windows": len(valid),
        "wf_failed_windows": len(windows) - len(valid),
        "wf_positive_sharpe_pct": round(positive_sharpe_pct, 3),
        "wf_sharpe_std": round(float(np.std(sharpes)), 3),
        "wf_recent_sharpes": [round(s, 3) for s in recent_sharpes],
        "wf_all_sharpes": [round(s, 3) for s in sharpes],
        "wf_all_cagrs": [round(c, 4) for c in cagrs],
        "wf_all_max_dds": [round(d, 4) for d in dds],
        "wf_all_win_rates": [round(w, 3) for w in wrs],
        "v031_baseline_sharpe": 1.161,
        "improvement_vs_v031": round(agg_sharpe - 1.161, 3),
    }


# ═══════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("T3.1 Dynamic Linear Model — IC-Adaptive Weights")
    print("=" * 60)

    # 1. Load data
    df, ic_data, factor_cols = load_data()

    # 2. Compute cross-sectional ranks
    rank_dict = compute_cross_sectional_ranks(df, factor_cols)

    # 3. Build prices matrix
    prices = build_prices_df(df)

    # 4. Compute monthly IC
    monthly_ic = compute_monthly_ic(df, factor_cols)

    # 5. Walk-Forward validation
    windows = run_walk_forward(df, rank_dict, prices, factor_cols, monthly_ic)

    # 6. Aggregate results
    summary = aggregate_results(windows)

    # 7. Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"WF Sharpe:     {summary['wf_sharpe']:.3f}")
    print(f"WF MaxDD:      {summary['wf_max_dd']:.1%}")
    print(f"WF CAGR:       {summary['wf_cagr']:.1%}")
    print(f"WF Win Rate:   {summary['wf_win_rate']:.0%}")
    print(f"WF Windows:    {summary['wf_n_windows']} ({summary['wf_failed_windows']} failed)")
    print(f"Positive Sharpe: {summary['wf_positive_sharpe_pct']:.0%}")
    print(f"Sharpe Std:    {summary['wf_sharpe_std']:.3f}")
    print(f"Recent Sharpes: {summary['wf_recent_sharpes']}")
    print(f"\n--- Comparison with V0.3.1 ---")
    print(f"V0.3.1 Sharpe: {summary['v031_baseline_sharpe']:.3f}")
    print(f"V0.4.0 Sharpe: {summary['wf_sharpe']:.3f}")
    print(f"Improvement:   {summary['improvement_vs_v031']:+.3f}")

    # 8. Save results
    output = {
        "model": "T3.1 Dynamic Linear Model (IC-Adaptive Weights)",
        "params": {
            "train_years": TRAIN_YEARS,
            "test_months": TEST_MONTHS,
            "hold_days": HOLD_DAYS,
            "top_n": TOP_N,
            "cost": COST,
            "stop_loss": STOP_LOSS,
            "ic_lookback_months": IC_LOOKBACK_MONTHS,
            "ic_min_dates": IC_MIN_DATES,
            "min_factor_coverage": MIN_FACTOR_COVERAGE,
        },
        "summary": summary,
        "windows": windows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
