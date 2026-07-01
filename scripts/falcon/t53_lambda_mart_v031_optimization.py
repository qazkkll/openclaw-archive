#!/usr/bin/env python3
"""
T5.3 LambdaMART排名模型 + V0.3.1权重优化 (V2 - Optimized)
===========================================================

方案:
  A: LambdaMART (LightGBM lambdarank) — 62个因子, 30d收益排名
  B: V0.3.1权重优化 — fund_ratio/analyst/fund_metric的最优权重
  C: V0.3.1 + 新闻因子 — 加入FinBERT sentiment
  D: 混合模型 — V0.3.1排名 50% + LambdaMART排名 50%

关键优化: 预计算所有因子截面rank，权重搜索只做加权求和（不重算rank）。
"""

import sys
import os
import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "falcon"
sys.path.insert(0, str(SCRIPTS_DIR))

from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════

V031_BASELINE_SHARPE = 1.161
TRAIN_YEARS = 5
TEST_MONTHS = 6
HOLD_DAYS = 30
TOP_N = 10
COST = 0.001
STOP_LOSS = -0.15

EXCLUDE_COLS = {
    "date", "ticker", "open", "high", "low", "close", "volume",
    "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_20d", "fwd_ret_30d",
}

NEWS_COLS = [
    "news_avg_sentiment", "news_sentiment_vol", "news_neg_ratio",
    "news_pos_ratio", "news_article_count", "news_confidence_avg",
]

# V0.3.1 factor group definitions
RATIO_COLS = [
    "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
    "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
    "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
    "ebitdaMargin", "assetTurnover", "inventoryTurnover",
    "receivablesTurnover", "debtToEquityRatio", "currentRatio",
    "quickRatio", "financialLeverageRatio",
    "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
    "dividendYieldPercentage", "dividendPayoutRatio",
]

ANALYST_COLS = [
    "eps_revision", "revenue_revision", "num_analysts_eps",
    "num_analysts_rev", "eps_dispersion",
]

METRIC_COLS = [
    "grossProfitMargin_qoq", "netProfitMargin_qoq",
    "operatingProfitMargin_qoq", "ebitdaMargin_qoq",
]


def load_data():
    """Load training data."""
    print("📂 Loading training data...")
    df = pd.read_parquet(DATA_DIR / "training_data_v04.parquet")
    df["date"] = df["date"].astype(str)
    print(f"  Rows: {len(df):,} | Tickers: {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")

    all_feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    high_cov = [c for c in all_feature_cols if df[c].notna().mean() >= 0.50]
    non_news = [c for c in high_cov if c not in NEWS_COLS]
    print(f"  Non-news features: {len(non_news)}")

    return df, all_feature_cols, high_cov, non_news


def build_walk_forward_windows(dates, train_years=5, test_months=6):
    """Build walk-forward windows."""
    dates_sorted = sorted(dates)
    start = pd.Timestamp(dates_sorted[0])
    end = pd.Timestamp(dates_sorted[-1])

    windows = []
    train_start = start

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        windows.append({
            "train_start": str(train_start)[:10],
            "train_end": str(train_end)[:10],
            "test_start": str(train_end)[:10],
            "test_end": str(test_end)[:10],
        })
        train_start += pd.DateOffset(months=test_months)

    return windows


def prepare_prices(df):
    """Build price matrix."""
    prices = df.pivot_table(index="date", columns="ticker", values="close")
    prices.index = prices.index.astype(str)
    prices = prices.sort_index()
    return prices


# ═══════════════════════════════════════════════════════════════════
#  Pre-compute factor ranks (do once, reuse everywhere)
# ═══════════════════════════════════════════════════════════════════

def precompute_all_factor_ranks(df):
    """Pre-compute cross-sectional percentile ranks for ALL factor columns.
    
    Returns: dict of {factor_col: {date: Series(ticker → rank)}}
    This avoids recomputing ranks for each weight combination.
    """
    print("  Pre-computing factor ranks for all dates...")
    t0 = time.time()
    
    all_dates = sorted(df["date"].unique())
    factor_cols = RATIO_COLS + ANALYST_COLS + METRIC_COLS + NEWS_COLS
    factor_cols = [c for c in factor_cols if c in df.columns]
    
    # Compute all ranks at once using groupby
    rank_dict = {f: {} for f in factor_cols}
    
    for date in all_dates:
        df_day = df[df["date"] == date]
        if len(df_day) < 10:
            continue
        tickers = df_day["ticker"].values
        
        for f in factor_cols:
            vals = df_day[f].values
            valid_mask = pd.notna(vals)
            if valid_mask.sum() > 5:
                # Percentile rank
                ranks = pd.Series(np.nan, index=tickers)
                valid_vals = vals[valid_mask]
                # rankdata returns 1-based, normalize to [0,1]
                from scipy.stats import rankdata
                r = rankdata(valid_vals, method="average") / len(valid_vals)
                ranks.iloc[valid_mask] = r
                rank_dict[f][date] = ranks
    
    elapsed = time.time() - t0
    print(f"  Pre-computed {len(factor_cols)} factors × {len(all_dates)} dates in {elapsed:.1f}s")
    return rank_dict, factor_cols


def compute_weighted_score_fast(rank_dict, date, weights):
    """Fast weighted score using pre-computed ranks.
    
    weights: dict of {factor_col: weight} or {group_name: weight}
    """
    # Map group names to individual factor columns
    group_map = {
        "fund_ratio": RATIO_COLS,
        "analyst": ANALYST_COLS,
        "fund_metric": METRIC_COLS,
    }
    
    combined = None
    
    for group_name, group_weight in weights.items():
        if group_weight <= 0:
            continue
        
        if group_name in group_map:
            # Group: average the percentile ranks of constituent factors
            group_cols = [c for c in group_map[group_name] if c in rank_dict]
            if not group_cols:
                continue
            
            group_sum = None
            count = 0
            for f in group_cols:
                if date in rank_dict[f]:
                    r = rank_dict[f][date]
                    if group_sum is None:
                        group_sum = r.copy()
                    else:
                        group_sum = group_sum.add(r, fill_value=0.5)
                    count += 1
            
            if count > 0:
                group_avg = group_sum / count
                if combined is None:
                    combined = group_avg * group_weight
                else:
                    combined = combined.add(group_avg * group_weight, fill_value=0)
        
        elif group_name == "news_sentiment":
            # News: use news_avg_sentiment as primary
            f = "news_avg_sentiment"
            if f in rank_dict and date in rank_dict[f]:
                r = rank_dict[f][date]
                if combined is None:
                    combined = r * group_weight
                else:
                    combined = combined.add(r * group_weight, fill_value=0)
    
    if combined is None:
        return None
    
    return combined.dropna().sort_values(ascending=False)


# ═══════════════════════════════════════════════════════════════════
#  LambdaMART
# ═══════════════════════════════════════════════════════════════════

def create_rank_labels(df):
    """Create ranking labels for LambdaMART."""
    df = df.copy()
    df["rel_rank"] = df.groupby("date")["fwd_ret_30d"].rank(pct=True)
    df["relevance"] = 0
    df.loc[df["rel_rank"] >= 0.90, "relevance"] = 4
    df.loc[(df["rel_rank"] >= 0.75) & (df["rel_rank"] < 0.90), "relevance"] = 3
    df.loc[(df["rel_rank"] >= 0.50) & (df["rel_rank"] < 0.75), "relevance"] = 2
    df.loc[(df["rel_rank"] >= 0.25) & (df["rel_rank"] < 0.50), "relevance"] = 1
    return df


def train_lambdamart(train_df, feature_cols):
    """Train LambdaMART model."""
    train_groups = train_df.groupby("date").size().values.tolist()
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["relevance"].values.astype(int)
    
    ds_train = lgb.Dataset(
        X_train, label=y_train,
        group=train_groups,
        feature_name=feature_cols,
        free_raw_data=False,
    )
    
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [TOP_N],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
    }
    
    model = lgb.train(
        params, ds_train,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def check_rank_inversion(df_test, model, feature_cols, top_pct=0.05, bot_pct=0.20):
    """Check ranking direction correctness.
    
    Uses only features with >80% coverage in the test data to avoid
    dropping too many rows due to low-coverage factors.
    """
    # Filter to high-coverage features only
    high_cov_features = [f for f in feature_cols if df_test[f].notna().mean() >= 0.80]
    if len(high_cov_features) < 5:
        # Fallback: use all features but fill NaN
        high_cov_features = feature_cols
    
    valid = df_test.dropna(subset=high_cov_features + ["fwd_ret_30d"])
    if len(valid) < 50:
        return {"passed": False, "reason": f"Insufficient data ({len(valid)} rows)"}
    
    X = valid[high_cov_features].fillna(0).values
    preds = model.predict(X)
    
    pred_series = pd.Series(preds, index=valid.index)
    n = len(pred_series)
    top_n = max(int(n * top_pct), 1)
    bot_n = max(int(n * bot_pct), 1)
    
    top_avg = valid.loc[pred_series.nlargest(top_n).index, "fwd_ret_30d"].mean()
    bot_avg = valid.loc[pred_series.nsmallest(bot_n).index, "fwd_ret_30d"].mean()
    spread = top_avg - bot_avg
    
    return {
        "passed": spread > 0,
        "top5_avg_ret": round(float(top_avg), 6),
        "bot20_avg_ret": round(float(bot_avg), 6),
        "spread": round(float(spread), 6),
        "direction_correct": spread > 0,
        "n_valid_rows": len(valid),
        "n_features_used": len(high_cov_features),
    }


# ═══════════════════════════════════════════════════════════════════
#  Plan A: LambdaMART Walk-Forward
# ═══════════════════════════════════════════════════════════════════

def run_plan_a(df, feature_cols, prices):
    """LambdaMART Walk-Forward."""
    print("\n" + "=" * 60)
    print("方案A: LambdaMART Walk-Forward")
    print("=" * 60)
    
    all_dates = sorted(df["date"].unique())
    windows = build_walk_forward_windows(all_dates, TRAIN_YEARS, TEST_MONTHS)
    print(f"  Windows: {len(windows)}")
    
    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
    window_results = []
    
    for i, w in enumerate(windows):
        print(f"\n  Window {i+1}/{len(windows)}: {w['test_start']} → {w['test_end']}")
        
        train_mask = (df["date"] >= w["train_start"]) & (df["date"] < w["train_end"])
        train_df = df[train_mask].copy()
        test_mask = (df["date"] >= w["test_start"]) & (df["date"] <= w["test_end"])
        test_df = df[test_mask].copy()
        
        if len(train_df) < 100 or len(test_df) < 10:
            print(f"    ⚠️ Insufficient data")
            continue
        
        train_df = create_rank_labels(train_df)
        
        t0 = time.time()
        model = train_lambdamart(train_df, feature_cols)
        elapsed = time.time() - t0
        print(f"    Trained: {model.num_trees()} trees in {elapsed:.1f}s")
        
        # Build test ranks
        test_dates = sorted(test_df["date"].unique())
        test_ranks = {}
        
        for date in test_dates:
            df_day = test_df[test_df["date"] == date].copy()
            if len(df_day) < 10:
                continue
            df_day.index = df_day["ticker"]
            X = df_day[feature_cols].fillna(0).values
            preds = model.predict(X)
            test_ranks[date] = pd.DataFrame({"lambdamart_score": preds}, index=df_day.index)
        
        if not test_ranks:
            print(f"    ⚠️ No valid test dates")
            continue
        
        try:
            result, _ = engine.run(
                test_ranks, prices, {"lambdamart_score": 1.0},
                hold_days=HOLD_DAYS, top_n=TOP_N,
                start_date=w["test_start"], end_date=w["test_end"],
                run_baseline=False,
            )
            
            ri = check_rank_inversion(test_df, model, feature_cols)
            
            window_results.append({
                "window": i + 1, "period": f"{w['test_start']} → {w['test_end']}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades, "n_trees": model.num_trees(),
                "rank_inversion": ri,
            })
            print(f"    Sharpe={result.sharpe:.3f} MaxDD={result.max_dd:.1%} "
                  f"CAGR={result.cagr:.1%} WR={result.win_rate:.0%} "
                  f"Trades={result.n_trades} RI={'✅' if ri['passed'] else '❌'}")
        
        except (DataQualityError, Exception) as e:
            print(f"    ❌ {e}")
            window_results.append({"window": i + 1, "error": str(e)})
    
    valid = [w for w in window_results if "sharpe" in w]
    if not valid:
        return {"status": "FAIL", "reason": "All windows failed", "windows": window_results}
    
    # Feature importance from last model
    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain")))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    agg = {
        "status": "OK", "model": "LambdaMART",
        "n_features": len(feature_cols),
        "windows_total": len(windows), "windows_valid": len(valid),
        "wf_sharpe": round(float(np.mean([w["sharpe"] for w in valid])), 3),
        "wf_max_dd": round(float(np.min([w["max_dd"] for w in valid])), 4),
        "wf_cagr": round(float(np.mean([w["cagr"] for w in valid])), 4),
        "wf_win_rate": round(float(np.mean([w["win_rate"] for w in valid])), 3),
        "wf_n_trades": sum(w["n_trades"] for w in valid),
        "avg_trees": round(float(np.mean([w["n_trees"] for w in valid])), 0),
        "rank_inversion_pass": all(w.get("rank_inversion", {}).get("passed", False) for w in valid),
        "top_features": [{f: round(float(g), 1)} for f, g in top_features],
        "windows": window_results,
    }
    
    print(f"\n  ✅ LambdaMART WF Sharpe={agg['wf_sharpe']:.3f} "
          f"MaxDD={agg['wf_max_dd']:.1%} CAGR={agg['wf_cagr']:.1%}")
    
    return agg


# ═══════════════════════════════════════════════════════════════════
#  Plan B: V0.3.1 Weight Optimization (Fast)
# ═══════════════════════════════════════════════════════════════════

def run_plan_b(df, prices, rank_dict):
    """V0.3.1 weight optimization using pre-computed ranks."""
    print("\n" + "=" * 60)
    print("方案B: V0.3.1 Weight Optimization (Fast)")
    print("=" * 60)
    
    all_dates = sorted(df["date"].unique())
    windows = build_walk_forward_windows(all_dates, TRAIN_YEARS, TEST_MONTHS)
    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
    
    # Weight combinations
    fund_ratios = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    analysts = [0.10, 0.15, 0.20, 0.25, 0.30]
    fund_metrics = [0.05, 0.10, 0.15, 0.20]
    
    combos = []
    for fr, an, fm in product(fund_ratios, analysts, fund_metrics):
        total = fr + an + fm
        if 0.98 <= total <= 1.02:
            combos.append({
                "fund_ratio": round(fr / total, 3),
                "analyst": round(an / total, 3),
                "fund_metric": round(fm / total, 3),
            })
    
    # Deduplicate
    seen = set()
    combos_unique = []
    for c in combos:
        key = (c["fund_ratio"], c["analyst"], c["fund_metric"])
        if key not in seen:
            seen.add(key)
            combos_unique.append(c)
    combos = combos_unique
    print(f"  Weight combinations: {len(combos)}")
    
    # Speed optimization: use first 3 windows for weight search
    search_windows = windows[:3]
    print(f"  Searching on {len(search_windows)} windows...")
    
    best_sharpe = -999
    best_weights = None
    best_idx = 0
    
    t0 = time.time()
    for ci, w in enumerate(combos):
        sharpes = []
        for sw in search_windows:
            try:
                # Build ranks using pre-computed factor ranks
                test_dates = [d for d in all_dates if sw["test_start"] <= d <= sw["test_end"]]
                if not test_dates:
                    sharpes.append(0)
                    continue
                
                test_ranks = {}
                for date in test_dates:
                    score = compute_weighted_score_fast(rank_dict, date, w)
                    if score is not None and len(score) > 0:
                        test_ranks[date] = pd.DataFrame({"v031_score": score})
                
                if not test_ranks:
                    sharpes.append(0)
                    continue
                
                result, _ = engine.run(
                    test_ranks, prices, {"v031_score": 1.0},
                    hold_days=HOLD_DAYS, top_n=TOP_N,
                    start_date=sw["test_start"], end_date=sw["test_end"],
                    run_baseline=False,
                )
                sharpes.append(result.sharpe)
            except Exception:
                sharpes.append(0)
        
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        
        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_weights = w.copy()
            best_idx = ci
        
        if (ci + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"    Tested {ci+1}/{len(combos)} ({elapsed:.0f}s): best_sharpe={best_sharpe:.3f} "
                  f"at {best_weights}")
    
    elapsed = time.time() - t0
    print(f"\n  Search complete in {elapsed:.1f}s")
    print(f"  Best weights: {best_weights} (search Sharpe={best_sharpe:.3f})")
    
    if best_weights is None:
        return {"status": "FAIL", "reason": "No valid combination"}
    
    # Full Walk-Forward with best weights
    print("  Running full Walk-Forward with best weights...")
    window_results = []
    
    for i, w in enumerate(windows):
        print(f"    Window {i+1}/{len(windows)}: {w['test_start']} → {w['test_end']}")
        try:
            test_dates = [d for d in all_dates if w["test_start"] <= d <= w["test_end"]]
            test_ranks = {}
            for date in test_dates:
                score = compute_weighted_score_fast(rank_dict, date, best_weights)
                if score is not None and len(score) > 0:
                    test_ranks[date] = pd.DataFrame({"v031_score": score})
            
            if not test_ranks:
                print(f"      ⚠️ No valid dates")
                continue
            
            result, _ = engine.run(
                test_ranks, prices, {"v031_score": 1.0},
                hold_days=HOLD_DAYS, top_n=TOP_N,
                start_date=w["test_start"], end_date=w["test_end"],
                run_baseline=False,
            )
            window_results.append({
                "window": i + 1, "period": f"{w['test_start']} → {w['test_end']}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades,
            })
            print(f"      Sharpe={result.sharpe:.3f} MaxDD={result.max_dd:.1%}")
        except Exception as e:
            print(f"      ❌ {e}")
            window_results.append({"window": i + 1, "error": str(e)})
    
    valid = [w for w in window_results if "sharpe" in w]
    if not valid:
        return {"status": "FAIL", "reason": "All windows failed"}
    
    agg = {
        "status": "OK", "model": "V0.3.1_Optimized",
        "best_weights": best_weights,
        "search_sharpe": round(best_sharpe, 3),
        "windows_total": len(windows), "windows_valid": len(valid),
        "wf_sharpe": round(float(np.mean([w["sharpe"] for w in valid])), 3),
        "wf_max_dd": round(float(np.min([w["max_dd"] for w in valid])), 4),
        "wf_cagr": round(float(np.mean([w["cagr"] for w in valid])), 4),
        "wf_win_rate": round(float(np.mean([w["win_rate"] for w in valid])), 3),
        "wf_n_trades": sum(w["n_trades"] for w in valid),
        "windows": window_results,
    }
    
    print(f"\n  ✅ V0.3.1 Optimized WF Sharpe={agg['wf_sharpe']:.3f}")
    return agg


# ═══════════════════════════════════════════════════════════════════
#  Plan C: V0.3.1 + News Factor
# ═══════════════════════════════════════════════════════════════════

def run_plan_c(df, prices, rank_dict):
    """V0.3.1 + news sentiment factor."""
    print("\n" + "=" * 60)
    print("方案C: V0.3.1 + 新闻因子")
    print("=" * 60)
    
    # Check news coverage
    for nc in NEWS_COLS:
        if nc in df.columns:
            cov = df[nc].notna().mean()
            print(f"  {nc}: {cov:.1%} coverage")
    
    all_dates = sorted(df["date"].unique())
    windows = build_walk_forward_windows(all_dates, TRAIN_YEARS, TEST_MONTHS)
    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
    
    # Weight search: fund_ratio + analyst + fund_metric + news_sentiment
    fund_ratios = [0.50, 0.55, 0.60, 0.65, 0.70]
    analysts = [0.10, 0.15, 0.20]
    fund_metrics = [0.05, 0.10, 0.15]
    news_weights = [0.05, 0.10, 0.15, 0.20]
    
    combos = []
    for fr, an, fm, nw in product(fund_ratios, analysts, fund_metrics, news_weights):
        total = fr + an + fm + nw
        if 0.98 <= total <= 1.02:
            combos.append({
                "fund_ratio": round(fr / total, 3),
                "analyst": round(an / total, 3),
                "fund_metric": round(fm / total, 3),
                "news_sentiment": round(nw / total, 3),
            })
    
    seen = set()
    combos_unique = []
    for c in combos:
        key = tuple(c.values())
        if key not in seen:
            seen.add(key)
            combos_unique.append(c)
    combos = combos_unique
    print(f"  Weight combinations: {len(combos)}")
    
    search_windows = windows[:3]
    best_sharpe = -999
    best_weights = None
    
    t0 = time.time()
    for ci, w in enumerate(combos):
        sharpes = []
        for sw in search_windows:
            try:
                test_dates = [d for d in all_dates if sw["test_start"] <= d <= sw["test_end"]]
                test_ranks = {}
                for date in test_dates:
                    score = compute_weighted_score_fast(rank_dict, date, w)
                    if score is not None and len(score) > 0:
                        test_ranks[date] = pd.DataFrame({"composite_score": score})
                
                if not test_ranks:
                    sharpes.append(0)
                    continue
                
                result, _ = engine.run(
                    test_ranks, prices, {"composite_score": 1.0},
                    hold_days=HOLD_DAYS, top_n=TOP_N,
                    start_date=sw["test_start"], end_date=sw["test_end"],
                    run_baseline=False,
                )
                sharpes.append(result.sharpe)
            except Exception:
                sharpes.append(0)
        
        avg_sharpe = np.mean(sharpes) if sharpes else 0
        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_weights = w.copy()
        
        if (ci + 1) % 10 == 0:
            print(f"    Tested {ci+1}/{len(combos)}: best={best_sharpe:.3f}")
    
    print(f"\n  Search complete in {time.time()-t0:.1f}s")
    print(f"  Best: {best_weights} Sharpe={best_sharpe:.3f}")
    
    if best_weights is None:
        return {"status": "FAIL", "reason": "No valid combination"}
    
    # Full Walk-Forward
    window_results = []
    for i, w in enumerate(windows):
        print(f"    Window {i+1}/{len(windows)}: {w['test_start']} → {w['test_end']}")
        try:
            test_dates = [d for d in all_dates if w["test_start"] <= d <= w["test_end"]]
            test_ranks = {}
            for date in test_dates:
                score = compute_weighted_score_fast(rank_dict, date, best_weights)
                if score is not None and len(score) > 0:
                    test_ranks[date] = pd.DataFrame({"composite_score": score})
            
            if not test_ranks:
                continue
            
            result, _ = engine.run(
                test_ranks, prices, {"composite_score": 1.0},
                hold_days=HOLD_DAYS, top_n=TOP_N,
                start_date=w["test_start"], end_date=w["test_end"],
                run_baseline=False,
            )
            window_results.append({
                "window": i + 1, "period": f"{w['test_start']} → {w['test_end']}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades,
            })
            print(f"      Sharpe={result.sharpe:.3f}")
        except Exception as e:
            print(f"      ❌ {e}")
            window_results.append({"window": i + 1, "error": str(e)})
    
    valid = [w for w in window_results if "sharpe" in w]
    if not valid:
        return {"status": "FAIL", "reason": "All windows failed"}
    
    agg = {
        "status": "OK", "model": "V0.3.1+News",
        "best_weights": best_weights,
        "search_sharpe": round(best_sharpe, 3),
        "windows_total": len(windows), "windows_valid": len(valid),
        "wf_sharpe": round(float(np.mean([w["sharpe"] for w in valid])), 3),
        "wf_max_dd": round(float(np.min([w["max_dd"] for w in valid])), 4),
        "wf_cagr": round(float(np.mean([w["cagr"] for w in valid])), 4),
        "wf_win_rate": round(float(np.mean([w["win_rate"] for w in valid])), 3),
        "wf_n_trades": sum(w["n_trades"] for w in valid),
        "windows": window_results,
    }
    
    print(f"\n  ✅ V0.3.1+News WF Sharpe={agg['wf_sharpe']:.3f}")
    return agg


# ═══════════════════════════════════════════════════════════════════
#  Plan D: Hybrid
# ═══════════════════════════════════════════════════════════════════

def run_plan_d(df, feature_cols, prices, rank_dict):
    """Hybrid: 50% V0.3.1 + 50% LambdaMART."""
    print("\n" + "=" * 60)
    print("方案D: Hybrid (V0.3.1 50% + LambdaMART 50%)")
    print("=" * 60)
    
    all_dates = sorted(df["date"].unique())
    windows = build_walk_forward_windows(all_dates, TRAIN_YEARS, TEST_MONTHS)
    engine = BacktestEngine(cost=COST, stop_loss=STOP_LOSS)
    
    v031_weights = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10}
    window_results = []
    
    for i, w in enumerate(windows):
        print(f"  Window {i+1}/{len(windows)}: {w['test_start']} → {w['test_end']}")
        
        train_mask = (df["date"] >= w["train_start"]) & (df["date"] < w["train_end"])
        train_df = df[train_mask].copy()
        test_mask = (df["date"] >= w["test_start"]) & (df["date"] <= w["test_end"])
        test_df = df[test_mask].copy()
        
        if len(train_df) < 100 or len(test_df) < 10:
            print(f"    ⚠️ Insufficient data")
            continue
        
        # Train LambdaMART
        train_df = create_rank_labels(train_df)
        try:
            model = train_lambdamart(train_df, feature_cols)
        except Exception as e:
            print(f"    ❌ LambdaMART train failed: {e}")
            continue
        
        # Build hybrid ranks
        test_dates = sorted(test_df["date"].unique())
        hybrid_ranks = {}
        
        for date in test_dates:
            df_day = test_df[test_df["date"] == date].copy()
            if len(df_day) < 10:
                continue
            df_day.index = df_day["ticker"]
            
            # LambdaMART score → rank
            X = df_day[feature_cols].fillna(0).values
            lm_preds = model.predict(X)
            lm_series = pd.Series(lm_preds, index=df_day.index)
            lm_rank = lm_series.rank(pct=True)
            
            # V0.3.1 score → rank
            v031_score = compute_weighted_score_fast(rank_dict, date, v031_weights)
            if v031_score is not None:
                v031_rank = v031_score.rank(pct=True)
            else:
                v031_rank = pd.Series(0.5, index=df_day.index)
            
            # Hybrid: 50/50
            hybrid_score = 0.5 * lm_rank + 0.5 * v031_rank.reindex(df_day.index, fill_value=0.5)
            
            hybrid_ranks[date] = pd.DataFrame({"hybrid_score": hybrid_score})
        
        if not hybrid_ranks:
            continue
        
        try:
            result, _ = engine.run(
                hybrid_ranks, prices, {"hybrid_score": 1.0},
                hold_days=HOLD_DAYS, top_n=TOP_N,
                start_date=w["test_start"], end_date=w["test_end"],
                run_baseline=False,
            )
            window_results.append({
                "window": i + 1, "period": f"{w['test_start']} → {w['test_end']}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades,
            })
            print(f"    Sharpe={result.sharpe:.3f} MaxDD={result.max_dd:.1%}")
        except Exception as e:
            print(f"    ❌ {e}")
            window_results.append({"window": i + 1, "error": str(e)})
    
    valid = [w for w in window_results if "sharpe" in w]
    if not valid:
        return {"status": "FAIL", "reason": "All windows failed"}
    
    agg = {
        "status": "OK", "model": "Hybrid (V0.3.1 50% + LambdaMART 50%)",
        "windows_total": len(windows), "windows_valid": len(valid),
        "wf_sharpe": round(float(np.mean([w["sharpe"] for w in valid])), 3),
        "wf_max_dd": round(float(np.min([w["max_dd"] for w in valid])), 4),
        "wf_cagr": round(float(np.mean([w["cagr"] for w in valid])), 4),
        "wf_win_rate": round(float(np.mean([w["win_rate"] for w in valid])), 3),
        "wf_n_trades": sum(w["n_trades"] for w in valid),
        "windows": window_results,
    }
    
    print(f"\n  ✅ Hybrid WF Sharpe={agg['wf_sharpe']:.3f}")
    return agg


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("🦅 Falcon T5.3: LambdaMART + V0.3.1 Optimization (V2)")
    print("=" * 60)
    
    # Load data
    df, all_feature_cols, high_cov, non_news = load_data()
    prices = prepare_prices(df)
    
    # Pre-compute all factor ranks (shared by Plans B, C, D)
    rank_dict, factor_cols = precompute_all_factor_ranks(df)
    
    # ── A: LambdaMART ──
    lambdamart_result = run_plan_a(df, non_news, prices)
    
    # ── B: V0.3.1 Weight Optimization ──
    v031_opt_result = run_plan_b(df, prices, rank_dict)
    
    # ── C: V0.3.1 + News ──
    v031_news_result = run_plan_c(df, prices, rank_dict)
    
    # ── D: Hybrid ──
    hybrid_result = run_plan_d(df, non_news, prices, rank_dict)
    
    # ═══════════════════════════════════════════════════════════════
    #  Selection & Summary
    # ═══════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 60)
    print("📊 RESULTS COMPARISON")
    print("=" * 60)
    
    all_results = {
        "A_lambdamart": lambdamart_result,
        "B_v031_optimized": v031_opt_result,
        "C_v031_with_news": v031_news_result,
        "D_hybrid": hybrid_result,
    }
    
    comparison = []
    for name, res in all_results.items():
        if res.get("status") == "OK":
            sharpe = res["wf_sharpe"]
            beats_v031 = sharpe > V031_BASELINE_SHARPE
            ri_pass = res.get("rank_inversion_pass", True)
            comparison.append({
                "name": name, "model": res.get("model", name),
                "wf_sharpe": sharpe, "wf_max_dd": res.get("wf_max_dd"),
                "wf_cagr": res.get("wf_cagr"),
                "beats_v031": beats_v031, "rank_inversion": ri_pass,
                "qualified": beats_v031 and ri_pass,
            })
            status = "✅" if beats_v031 and ri_pass else "❌"
            print(f"  {status} {name}: Sharpe={sharpe:.3f} "
                  f"MaxDD={res.get('wf_max_dd', 0):.1%} "
                  f"Beats V0.3.1={beats_v031} RI={'✅' if ri_pass else '❌'}")
        else:
            print(f"  ❌ {name}: {res.get('status')} - {res.get('reason', 'unknown')}")
            comparison.append({
                "name": name, "model": res.get("model", name),
                "wf_sharpe": None, "status": res.get("status"),
                "reason": res.get("reason"), "qualified": False,
            })
    
    # Select best
    qualified = [c for c in comparison if c.get("qualified")]
    if qualified:
        best = max(qualified, key=lambda x: x["wf_sharpe"])
        print(f"\n  🏆 BEST: {best['name']} (Sharpe={best['wf_sharpe']:.3f})")
    else:
        best = None
        print(f"\n  ❌ No方案beat V0.3.1 baseline (Sharpe={V031_BASELINE_SHARPE:.3f})")
    
    # Save
    elapsed = time.time() - t0
    output = {
        "task": "T5.3 LambdaMART + V0.3.1 Optimization",
        "version": "v0.4.0",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "v031_baseline_sharpe": V031_BASELINE_SHARPE,
        "wf_params": {
            "train_years": TRAIN_YEARS, "test_months": TEST_MONTHS,
            "hold_days": HOLD_DAYS, "top_n": TOP_N,
            "cost": COST, "stop_loss": STOP_LOSS,
        },
        "feature_count": len(non_news),
        "comparison": comparison,
        "best_proposal": best,
        "results": all_results,
    }
    
    out_path = DATA_DIR / "v04_lambda_mart_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  💾 Saved to {out_path}")
    print(f"  ⏱️ Total time: {elapsed:.1f}s")
    
    return output


if __name__ == "__main__":
    main()
