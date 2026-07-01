#!/usr/bin/env python3
"""
🦅 Falcon V0.4.4: Fixed Rank Inversion Check
=============================================
V0.4.4审计发现BLOCKER: check_rank_inversion()只检查Sharpe退化稳定性，
不是真正的Top5% vs Bottom20%收益比较。

本脚本:
  1. 使用V0.4.4最佳配置 (fund_ratio=0.45 + gc_baseline=0.20 + qoq=0.20 + cashflow=0.15)
  2. 实现真正的Rank Inversion检查: 对每个Walk-Forward窗口计算Top5%股票的
     平均前瞻收益 vs Bottom20%股票的平均前瞻收益
  3. 使用backtest_engine.py回测

Walk-Forward 参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出: data/falcon/v044_fixed_ri_results.json
"""
import sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, DataQualityError

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "v044_fixed_ri_results.json"

# ═══════════════════════════════════════════════════
#  因子组定义 (从v044_factor_expansion.py复用)
# ═══════════════════════════════════════════════════

EXCLUDE_COLS = {
    'ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap',
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'ma_cross_5_20', 'ma_cross_20_60',
    'price_position', 'ret1', 'ret5', 'ret10', 'ret20', 'ret30', 'ret60', 'ret90',
    'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
    'vol20', 'vol5', 'vol_ratio', 'vol_change', 'vol_regime',
    'rsi14', 'rsi_change', 'rsi_zone',
    'macd', 'macd_signal', 'macd_hist', 'macd_roc',
    'bb_std', 'bb_width', 'bb_pos',
    'ret_quality', 'range_ratio', 'avg_body', 'vwap_drift', 'dd_60', 'ud_vol_ratio', 'beta',
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin', 'ebitdaMargin',
    'assetTurnover', 'inventoryTurnover', 'receivablesTurnover',
    'debtToEquityRatio', 'currentRatio', 'quickRatio', 'financialLeverageRatio',
    'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
    'dividendYieldPercentage', 'dividendPayoutRatio',
    'eps_revision', 'revenue_revision', 'num_analysts_eps', 'num_analysts_rev',
    'eps_dispersion', 'fmp_covered', 'analyst_covered',
    'grossProfitMargin_qoq', 'netProfitMargin_qoq', 'operatingProfitMargin_qoq', 'ebitdaMargin_qoq',
}

FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_assetGrowth',
        'g_bookValueperShareGrowth', 'g_debtGrowth',
    ],
    'analyst': [
        'a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps',
    ],
    'income': [
        'i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
        'i_revenue_growth_yoy', 'i_gross_margin_delta',
    ],
    'qoq': [
        'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
        'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
    ],
    'cashflow': [
        'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
    ],
}


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载特征和价格数据。"""
    print("📂 加载数据...")
    t0 = time.time()
    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    print(f"  ✅ Features: {df.shape[0]}行 × {df.shape[1]}列, {df['ticker'].nunique()}只")

    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")
    print(f"  ⏱️ 加载耗时: {time.time()-t0:.1f}秒")
    return df, price_pivot


# ═══════════════════════════════════════════════════
#  截面百分位排名 (从v044_factor_expansion.py复用)
# ═══════════════════════════════════════════════════

def compute_cross_sectional_ranks(df, factor_cols):
    """计算截面百分位排名。"""
    print("📊 计算截面百分位排名...")
    t0 = time.time()

    flip_factors = {
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
        'b_debt_to_equity', 'b_net_debt_to_assets', 'm_netDebtToEBITDA',
        'm_capexToRevenue', 'm_capexToDepreciation',
        'm_researchAndDevelopementToRevenue', 'm_stockBasedCompensationToRevenue',
        'c_capex_intensity',
        'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
        'a_eps_dispersion',
        'm_cashConversionCycle',
    }

    from scipy.stats import rankdata

    dates = sorted(df['date'].unique())
    ranks = {}

    for date in dates:
        day_df = df[df['date'] == date].copy()
        if len(day_df) < 10:
            continue

        tickers = day_df['ticker'].values
        rank_df = pd.DataFrame(index=tickers)

        for col in factor_cols:
            if col not in day_df.columns:
                continue
            vals = day_df[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue

            ranks_raw = np.full_like(vals, np.nan)
            if valid.sum() > 0:
                ranks_raw[valid] = rankdata(vals[valid], method='average') / valid.sum()

            if col in flip_factors:
                mask = ~np.isnan(ranks_raw)
                ranks_raw[mask] = 1.0 - ranks_raw[mask]

            rank_df[col] = ranks_raw

        ranks[date] = rank_df

    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)}天排名计算完成 ({elapsed:.0f}秒)")
    return ranks


def compute_group_ranks(ranks, factor_groups):
    """将因子组的排名合并为组级排名(等权平均)。"""
    print("📊 计算因子组排名...")
    for date in list(ranks.keys()):
        df = ranks[date]
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                df[group_name] = df[available].mean(axis=1)
        ranks[date] = df
    print(f"  ✅ 因子组排名已添加: {list(factor_groups.keys())}")
    return ranks


# ═══════════════════════════════════════════════════
#  Growth Composite (从v044_factor_expansion.py复用)
# ═══════════════════════════════════════════════════

def add_growth_composite(ranks):
    """添加growth_composite (gc_baseline)。"""
    gc_baseline = lambda d: (
        d.get('fund_growth', 0) * 0.60 +
        d.get('analyst', 0) * 0.25 +
        d.get('income', 0) * 0.15
    )

    for date in ranks:
        df = ranks[date]
        try:
            df['gc_baseline'] = gc_baseline(df.to_dict('series'))
        except Exception:
            df['gc_baseline'] = np.nan
        ranks[date] = df

    print("  ✅ gc_baseline (growth_composite) 已添加")
    return ranks


# ═══════════════════════════════════════════════════
#  真正的 Rank Inversion 检查
# ═══════════════════════════════════════════════════

def compute_combined_scores(ranks, date, weights):
    """计算因子组合分数 (与BacktestEngine._get_scores相同逻辑)。
    
    如果exact date不在ranks中，使用最近的可用日期。
    """
    if date in ranks:
        r = ranks[date]
    else:
        # 找到最近的可用日期 (不超过date)
        rank_dates = sorted(ranks.keys())
        candidates = [d for d in rank_dates if d <= date]
        if not candidates:
            return None
        r = ranks[candidates[-1]]
    
    available = [f for f in weights if f in r.columns and weights[f] > 0]
    if not available:
        return None
    combined = pd.Series(0.0, index=r.index)
    for f in available:
        combined = combined + weights[f] * r[f]
    return combined.dropna().sort_values(ascending=False)


def check_real_rank_inversion(ranks, prices, weights, windows):
    """真正的Rank Inversion检查: Top5% vs Bottom20% 前瞻收益。
    
    对每个Walk-Forward窗口:
    1. 在test_start日期计算模型分数
    2. 获取Top5%和Bottom20%的股票
    3. 计算这些股票从test_start到test_end的平均前瞻收益
    4. 检查Top5%收益 > Bottom20%收益
    
    Returns:
        dict: {passed, per_window, overall_stats}
    """
    print("\n🔍 真正的Rank Inversion检查 (Top5% vs Bottom20%)...")
    
    results_per_window = []
    valid_count = 0
    pass_count = 0
    
    for w in windows:
        if "error" in w:
            results_per_window.append({
                "window": w["index"],
                "period": w["period"],
                "status": "SKIPPED",
                "reason": "Window failed backtest"
            })
            continue
        
        # 解析test_start和test_end
        period = w["period"]
        try:
            parts = period.split(" → ")
            test_start_str = parts[0].strip()
            test_end_str = parts[1].strip()
        except (IndexError, ValueError):
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "ERROR",
                "reason": "Cannot parse period"
            })
            continue
        
        # 在test_start日期计算分数
        scores = compute_combined_scores(ranks, test_start_str, weights)
        if scores is None or len(scores) < 20:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": f"Insufficient scores at {test_start_str} (got {len(scores) if scores is not None else 0})"
            })
            continue
        
        # 计算前瞻收益: price[test_end] / price[test_start] - 1
        # 使用test_start到test_end之间的价格变化
        # 找到test_start之后最近的价格日期
        price_dates = sorted(prices.index.astype(str))
        
        # 找test_start对应的价格日期
        start_candidates = [d for d in price_dates if d >= test_start_str]
        end_candidates = [d for d in price_dates if d >= test_end_str]
        
        if not start_candidates or not end_candidates:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": "No price data for period"
            })
            continue
        
        actual_start = start_candidates[0]
        actual_end = end_candidates[0]
        
        if actual_start not in prices.index or actual_end not in prices.index:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": "Price dates not in index"
            })
            continue
        
        start_prices = prices.loc[actual_start]
        end_prices = prices.loc[actual_end]
        
        # 计算每只股票的前瞻收益
        common_tickers = scores.index.intersection(start_prices.index).intersection(end_prices.index)
        valid_start = start_prices[common_tickers]
        valid_end = end_prices[common_tickers]
        
        # 过滤掉价格为0或NaN的
        mask = (valid_start > 0) & valid_end.notna() & valid_start.notna()
        valid_tickers = common_tickers[mask]
        
        if len(valid_tickers) < 20:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": f"Insufficient valid prices ({len(valid_tickers)})"
            })
            continue
        
        fwd_returns = (valid_end[valid_tickers] / valid_start[valid_tickers]) - 1
        
        # Top5%: 得分最高的5%股票
        n_top5 = max(1, int(len(scores) * 0.05))
        top5_tickers = scores.nlargest(n_top5).index
        top5_tickers = [t for t in top5_tickers if t in fwd_returns.index]
        
        # Bottom20%: 得分最低的20%股票
        n_bot20 = max(1, int(len(scores) * 0.20))
        bot20_tickers = scores.nsmallest(n_bot20).index
        bot20_tickers = [t for t in bot20_tickers if t in fwd_returns.index]
        
        if len(top5_tickers) == 0 or len(bot20_tickers) == 0:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": "No tickers in top/bottom groups"
            })
            continue
        
        top5_ret = float(fwd_returns[top5_tickers].mean())
        bot20_ret = float(fwd_returns[bot20_tickers].mean())
        ri_passed = top5_ret > bot20_ret
        
        valid_count += 1
        if ri_passed:
            pass_count += 1
        
        results_per_window.append({
            "window": w["index"],
            "period": period,
            "status": "PASS" if ri_passed else "FAIL",
            "top5_pct_count": len(top5_tickers),
            "top5_avg_return": round(top5_ret, 4),
            "bottom20_pct_count": len(bot20_tickers),
            "bottom20_avg_return": round(bot20_ret, 4),
            "spread": round(top5_ret - bot20_ret, 4),
            "sharpe": w["sharpe"],
        })
        
        mark = "✅" if ri_passed else "❌"
        print(f"    {mark} W{w['index']}: {period} | "
              f"Top5%={top5_ret:+.2%} Bottom20%={bot20_ret:+.2%} "
              f"Spread={top5_ret-bot20_ret:+.2%}")
    
    # 汇总
    overall_passed = pass_count > valid_count * 0.5 if valid_count > 0 else False
    
    all_top5_rets = [r["top5_avg_return"] for r in results_per_window if "top5_avg_return" in r]
    all_bot20_rets = [r["bottom20_avg_return"] for r in results_per_window if "bottom20_avg_return" in r]
    all_spreads = [r["spread"] for r in results_per_window if "spread" in r]
    
    summary = {
        "passed": overall_passed,
        "method": "Top5% vs Bottom20% forward returns",
        "valid_windows": valid_count,
        "pass_windows": pass_count,
        "pass_rate": round(pass_count / valid_count, 3) if valid_count > 0 else 0,
        "avg_top5_return": round(float(np.mean(all_top5_rets)), 4) if all_top5_rets else None,
        "avg_bottom20_return": round(float(np.mean(all_bot20_rets)), 4) if all_bot20_rets else None,
        "avg_spread": round(float(np.mean(all_spreads)), 4) if all_spreads else None,
        "per_window": results_per_window,
    }
    
    print(f"\n  📊 Rank Inversion Summary:")
    print(f"     Method: Top5% vs Bottom20% forward returns")
    print(f"     Valid windows: {valid_count}")
    print(f"     Pass: {pass_count}/{valid_count} ({summary['pass_rate']:.0%})")
    if all_top5_rets:
        print(f"     Avg Top5% return: {summary['avg_top5_return']:+.2%}")
        print(f"     Avg Bottom20% return: {summary['avg_bottom20_return']:+.2%}")
        print(f"     Avg spread: {summary['avg_spread']:+.2%}")
    print(f"     Overall: {'✅ PASS' if overall_passed else '❌ FAIL'}")
    
    return summary


# ═══════════════════════════════════════════════════
#  Walk-Forward (使用backtest_engine)
# ═══════════════════════════════════════════════════

def run_walk_forward(ranks, prices, weights, train_years=0.5, test_months=6,
                     hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15):
    """运行Walk-Forward, 返回(result_dict, window_details)。"""
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    dates = sorted(ranks.keys())
    if not dates:
        return None, []

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    windows = []
    idx = 0

    train_months = int(train_years * 12)
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
        tss = str(train_end)[:10]
        tes = str(test_end)[:10]
        try:
            result, baseline = engine.run(
                ranks, prices, weights, hold_days, top_n,
                start_date=tss, end_date=tes, run_baseline=True
            )
            windows.append({
                "index": idx, "period": f"{tss} → {tes}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades, "n_days": len(result.daily_equity),
                "baseline_sharpe": baseline.sharpe if baseline else None,
            })
        except (DataQualityError, Exception) as e:
            windows.append({"index": idx, "period": f"{tss} → {tes}", "error": str(e)[:200]})
        idx += 1
        train_start += pd.DateOffset(months=test_months)

    if not windows:
        return None, []

    valid = [w for w in windows if "sharpe" in w]
    if not valid:
        return {"error": "All windows failed", "windows": windows}, windows

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]

    result = {
        "sharpe": round(float(np.mean(sharpes)), 3),
        "max_dd": round(float(np.min(dds)), 4),
        "cagr": round(float(np.mean(cagrs)), 4),
        "win_rate": round(float(np.mean(wrs)), 3),
        "n_trades": sum(w["n_trades"] for w in valid),
        "n_windows": len(valid),
        "warnings": [],
        "status": "PASS",
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  JSON序列化辅助
# ═══════════════════════════════════════════════════

def serialize(obj):
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4.4: Fixed Rank Inversion Check")
    print("=" * 80)

    # ─── 1. 加载数据 ───
    df, price_pivot = load_data()

    # ─── 2. 确定可用PIT因子 ───
    all_pit_cols = []
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        all_pit_cols.extend(available)
        print(f"  {group}: {len(available)}/{len(cols)} factors")
    print(f"  Total PIT factors: {len(all_pit_cols)}")

    # ─── 3. 计算截面百分位排名 ───
    ranks = compute_cross_sectional_ranks(df, all_pit_cols)

    # ─── 4. 计算因子组排名 ───
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)

    # ─── 5. 添加growth composite ───
    ranks = add_growth_composite(ranks)

    # ═══════════════════════════════════════════════
    #  V0.4.4最佳配置
    # ═══════════════════════════════════════════════
    v044_weights = {
        "fund_ratio": 0.45,
        "gc_baseline": 0.20,
        "qoq": 0.20,
        "cashflow": 0.15,
    }

    print(f"\n{'='*60}")
    print(f"📊 V0.4.4 Walk-Forward (真正Rank Inversion)")
    print(f"{'='*60}")
    print(f"  配置: {v044_weights}")
    print(f"  参数: train_years=0.5, test_months=6, hold_days=30, top_n=10")

    # ─── 6. Walk-Forward ───
    wf_result, wf_windows = run_walk_forward(
        ranks, price_pivot, v044_weights,
        train_years=0.5, test_months=6, hold_days=30, top_n=10,
        cost=0.001, stop_loss=-0.15
    )

    if wf_result:
        print(f"\n  Walk-Forward结果:")
        print(f"    Sharpe: {wf_result['sharpe']:.3f}")
        print(f"    MaxDD: {wf_result['max_dd']:.1%}")
        print(f"    CAGR: {wf_result['cagr']:.1%}")
        print(f"    Win Rate: {wf_result['win_rate']:.0%}")
        print(f"    Windows: {wf_result['n_windows']}")
        
        # 打印每个窗口详情
        valid_wins = [w for w in wf_windows if "sharpe" in w]
        print(f"\n  窗口详情:")
        for w in valid_wins:
            mark = "✅" if w["sharpe"] > 0 else "❌"
            print(f"    {mark} W{w['index']}: {w['period']}  "
                  f"Sharpe={w['sharpe']:.3f}  MaxDD={w['max_dd']:.1%}  "
                  f"WR={w['win_rate']:.0%}  Trades={w['n_trades']}")
    else:
        print("  ❌ Walk-Forward失败")
        return

    # ═══════════════════════════════════════════════
    #  真正的Rank Inversion检查
    # ═══════════════════════════════════════════════
    ri_result = check_real_rank_inversion(ranks, price_pivot, v044_weights, wf_windows)

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.4 Fixed Rank Inversion Check",
            "description": "Re-run V0.4.4 with real Rank Inversion check (Top5% vs Bottom20% forward returns)",
            "config": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
            },
            "weights": v044_weights,
            "features": "features_v04_1.parquet",
            "pit_factors": len(all_pit_cols),
            "factor_groups": {k: len(v) for k, v in FACTOR_GROUPS.items()},
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "walk_forward": serialize(wf_result) if wf_result else None,
        "window_details": serialize(wf_windows),
        "rank_inversion": serialize(ri_result),
        "old_rank_inversion": {
            "description": "旧的Sharpe退化检查 (从v044_factor_expansion.py复制)",
            "method": "Sharpe degradation (recent vs early windows)",
            "verdict": "BLOCKER - not a real RI check",
        },
        "verdict": {
            "wf_sharpe": wf_result["sharpe"] if wf_result else 0,
            "ri_passed": ri_result["passed"],
            "ri_pass_rate": ri_result.get("pass_rate", 0),
            "ri_avg_spread": ri_result.get("avg_spread"),
            "overall": "PASS" if (wf_result and ri_result["passed"]) else "FAIL",
            "comparison_with_v043": {
                "v043_sharpe": 2.007,
                "v044_sharpe": wf_result["sharpe"] if wf_result else 0,
                "improvement_pct": round(
                    (wf_result["sharpe"] - 2.007) / 2.007 * 100, 1
                ) if wf_result else 0,
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved: {OUTPUT_PATH}")
    print(f"⏱️ Total time: {(time.time()-t0)/60:.1f} minutes")

    # Final summary
    print(f"\n{'='*80}")
    print(f"📋 Final Summary")
    print(f"{'='*80}")
    print(f"  V0.4.4 weights: {v044_weights}")
    if wf_result:
        print(f"  WF Sharpe: {wf_result['sharpe']:.3f}")
        print(f"  WF MaxDD: {wf_result['max_dd']:.1%}")
    print(f"  Rank Inversion: {'✅ PASS' if ri_result['passed'] else '❌ FAIL'}")
    print(f"    Method: Top5% vs Bottom20% forward returns")
    print(f"    Pass rate: {ri_result.get('pass_rate', 0):.0%} ({ri_result.get('pass_windows', 0)}/{ri_result.get('valid_windows', 0)})")
    if ri_result.get("avg_spread") is not None:
        print(f"    Avg spread: {ri_result['avg_spread']:+.2%}")
    print(f"  V0.4.3 baseline Sharpe: 2.007")
    if wf_result:
        improvement = (wf_result['sharpe'] - 2.007) / 2.007 * 100
        print(f"  Improvement vs V0.4.3: {improvement:+.1f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
