#!/usr/bin/env python3
"""
🦅 Falcon V0.4.5: Portfolio Construction Optimization
======================================================
测试组合构建方法、股票数量、止损水平的最优配置。

V0.4.4配置:
  fund_ratio=0.45 + growth_composite=0.20 + qoq=0.20 + cashflow=0.15
  WF Sharpe: 2.122, RI=PASS

测试维度:
  1. 组合构建: 等权/市值加权/风险平价
  2. 股票数量: 5/10/15/20
  3. 止损水平: -10%/-15%/-20%

Walk-Forward 参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出: data/falcon/v045_portfolio_optimization_results.json
"""
import sys, json, time, warnings, math
from pathlib import Path
from datetime import datetime
from itertools import product

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
OUTPUT_PATH = DATA_DIR / "v045_portfolio_optimization_results.json"

# ═══════════════════════════════════════════════════
#  因子组定义 (与v044一致)
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
    'fund_metric': [
        'm_earningsYield', 'm_evToEBITDA', 'm_evToFreeCashFlow', 'm_evToSales',
        'm_freeCashFlowYield', 'm_returnOnEquity', 'm_returnOnAssets',
        'm_returnOnCapitalEmployed', 'm_returnOnInvestedCapital', 'm_returnOnTangibleAssets',
        'm_incomeQuality', 'm_grahamNumber', 'm_cashConversionCycle',
        'm_capexToRevenue', 'm_capexToDepreciation',
        'm_researchAndDevelopementToRevenue', 'm_stockBasedCompensationToRevenue',
        'm_netDebtToEBITDA', 'm_operatingReturnOnAssets',
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
    'balance': [
        'b_cash_to_assets', 'b_net_debt_to_assets', 'b_equity_ratio', 'b_debt_to_equity',
    ],
    'cashflow': [
        'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
    ],
    'income': [
        'i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
        'i_revenue_growth_yoy', 'i_gross_margin_delta',
    ],
    'qoq': [
        'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
        'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
    ],
}

# V0.4.5 权重 (来自V0.4.4优化)
V045_WEIGHTS = {
    "fund_ratio": 0.45,
    "growth_composite": 0.20,
    "qoq": 0.20,
    "cashflow": 0.15,
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
#  截面百分位排名
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
#  组合因子: growth_composite
# ═══════════════════════════════════════════════════

def add_growth_composite(ranks):
    """添加growth_composite因子 (V0.4.5权重)。"""
    gc_func = lambda d: (
        d.get('fund_growth', 0) * 0.50 +
        d.get('analyst', 0) * 0.20 +
        d.get('income', 0) * 0.15 +
        d.get('qoq', 0) * 0.15
    )

    for date in ranks:
        df = ranks[date]
        try:
            df['growth_composite'] = gc_func(df.to_dict('series'))
        except Exception:
            df['growth_composite'] = np.nan
        ranks[date] = df

    print(f"  ✅ growth_composite added")
    return ranks


# ═══════════════════════════════════════════════════
#  PortfolioEngine: 支持不同组合构建方法的回测引擎
# ═══════════════════════════════════════════════════

class PortfolioEngine(BacktestEngine):
    """支持不同组合构建方法的回测引擎。
    
    继承BacktestEngine，重写_simulate方法支持:
    - equal: 等权 (默认，与父类一致)
    - mcap: 市值加权 (用收盘价作为市值代理)
    - risk_parity: 风险平价 (用20日收益率标准差的倒数加权)
    """
    
    def __init__(self, cost=0.001, stop_loss=-0.15,
                 portfolio_method='equal', price_data=None):
        """
        Args:
            cost: 单边交易成本
            stop_loss: 止损线
            portfolio_method: 'equal', 'mcap', 'risk_parity'
            price_data: DataFrame(date×ticker) 用于计算市值/波动率
        """
        super().__init__(cost=cost, stop_loss=stop_loss)
        self.portfolio_method = portfolio_method
        self.price_data = price_data  # 完整价格数据
        
        # 预计算波动率 (risk_parity用, 向量化)
        self._rolling_vol = None
        if portfolio_method == 'risk_parity' and price_data is not None:
            returns = price_data.pct_change()
            self._rolling_vol = returns.rolling(window=20, min_periods=5).std()
    
    def _get_portfolio_weights(self, picks, date_str, pr):
        """计算持仓权重。
        
        Args:
            picks: 选中的股票列表
            date_str: 当前日期字符串
            pr: 当前价格Series
            
        Returns:
            dict: {ticker: weight_fraction} 权重总和=1
        """
        if self.portfolio_method == 'equal':
            n = len(picks)
            return {t: 1.0 / n for t in picks}
        
        elif self.portfolio_method == 'mcap':
            # 市值加权: 用收盘价作为市值代理
            weights = {}
            total = 0.0
            for t in picks:
                if t in pr.index and pd.notna(pr[t]) and pr[t] > 0:
                    weights[t] = float(pr[t])
                    total += weights[t]
                else:
                    weights[t] = 0.0
            
            if total > 0:
                return {t: w / total for t, w in weights.items()}
            else:
                n = len(picks)
                return {t: 1.0 / n for t in picks}
        
        elif self.portfolio_method == 'risk_parity':
            # 风险平价: 用20日波动率的倒数加权 (向量化)
            weights = {}
            total = 0.0
            for t in picks:
                if self._rolling_vol is not None and date_str in self._rolling_vol.index and t in self._rolling_vol.columns:
                    vol = self._rolling_vol.loc[date_str, t]
                    vol = float(vol) if pd.notna(vol) and vol > 0 else 1.0
                else:
                    vol = 1.0
                inv_vol = 1.0 / max(vol, 1e-8)
                weights[t] = inv_vol
                total += inv_vol
            
            if total > 0:
                return {t: w / total for t, w in weights.items()}
            else:
                n = len(picks)
                return {t: 1.0 / n for t in picks}
        
        else:
            n = len(picks)
            return {t: 1.0 / n for t in picks}
    
    def _simulate(self, ranks, prices, all_dates_in_prices, weights, hold_days, top_n):
        """重写模拟方法，支持不同组合构建方法。"""
        cash = 1.0
        portfolio = {}  # ticker -> (entry_idx, entry_price, cash_allocated)
        equity_list = []
        trade_list = []
        rebalance_count = 0

        for i, date in enumerate(all_dates_in_prices):
            if date not in prices.index:
                if equity_list:
                    equity_list.append(equity_list[-1])
                continue

            pr = prices.loc[date]
            
            # ── 止损(每天检查) ──
            to_close = []
            for t, (ei, ep, alloc) in portfolio.items():
                if t in pr.index and pd.notna(pr[t]) and ep > 0:
                    pnl = (pr[t] - ep) / ep
                    if pnl <= self.stop_loss:
                        cash += alloc * pr[t] / ep * (1 - self.cost)
                        trade_list.append({"pnl": pnl, "reason": "stop_loss", "date": date})
                        to_close.append(t)
            for t in to_close:
                del portfolio[t]
            
            # ── 调仓检查 ──
            should_rebalance = False
            sell_tickers = []
            
            for t, (ei, ep, alloc) in list(portfolio.items()):
                if (i - ei) >= hold_days:
                    sell_tickers.append(t)
            
            if sell_tickers or len(portfolio) == 0:
                should_rebalance = True
            
            # ── 卖出到期持仓 ──
            for t in sell_tickers:
                if t in portfolio:
                    ei, ep, alloc = portfolio.pop(t)
                    if t in pr.index and pd.notna(pr[t]) and ep > 0:
                        pnl = (pr[t] - ep) / ep
                        cash += alloc * pr[t] / ep * (1 - self.cost)
                        trade_list.append({"pnl": pnl, "reason": "rebalance", "date": date})
            
            # ── 买入新持仓 ──
            if should_rebalance and len(portfolio) == 0 and cash > 0.01:
                scores = self._get_scores(ranks, date, weights)
                if scores is not None:
                    picks = scores.head(top_n).index.tolist()
                    picks = [t for t in picks if t in pr.index and pd.notna(pr[t]) and pr[t] > 0]
                    if picks:
                        # 使用组合构建方法计算权重
                        port_weights = self._get_portfolio_weights(picks, date, pr)
                        
                        buy_cost = 0.0
                        for t in picks:
                            alloc = cash * port_weights[t]
                            portfolio[t] = (i, pr[t], alloc)
                            buy_cost += alloc * self.cost
                        
                        cash = 0.0
                        cash -= buy_cost
                        rebalance_count += 1
            
            # ── 日频净值 ──
            pv = cash
            for t, (_, ep, alloc) in portfolio.items():
                if t in pr.index and pd.notna(pr[t]) and ep > 0:
                    pv += alloc * pr[t] / ep
            equity_list.append(pv)
        
        return self._compute_result(equity_list, trade_list, rebalance_count, all_dates_in_prices[:len(equity_list)])


# ═══════════════════════════════════════════════════
#  Rank Inversion 检查
# ═══════════════════════════════════════════════════

def check_rank_inversion(windows):
    """检查排名反转。"""
    valid = [w for w in windows if "sharpe" in w]
    if len(valid) < 2:
        return {"passed": False, "reason": "Too few windows"}

    recent = valid[-3:] if len(valid) >= 3 else valid
    early = valid[:3] if len(valid) >= 3 else valid

    recent_avg = np.mean([w["sharpe"] for w in recent])
    early_avg = np.mean([w["sharpe"] for w in early])
    neg_recent = sum(1 for w in recent if w["sharpe"] < 0)

    passed = True
    reason = "OK"

    if neg_recent >= 2:
        passed = False
        reason = f"Recent {neg_recent}/3 windows negative"
    elif early_avg > 0 and recent_avg < early_avg * 0.3:
        passed = False
        reason = f"Severe degradation: early={early_avg:.2f} → recent={recent_avg:.2f}"

    return {
        "passed": passed,
        "recent_avg_sharpe": round(float(recent_avg), 3),
        "early_avg_sharpe": round(float(early_avg), 3),
        "negative_recent_windows": neg_recent,
        "reason": reason,
    }


# ═══════════════════════════════════════════════════
#  Walk-Forward
# ═══════════════════════════════════════════════════

def run_wf(ranks, prices, factor_weights, train_years=0.5, test_months=6,
           hold_days=30, top_n=10, cost=0.001, stop_loss=-0.15,
           portfolio_method='equal', price_data=None):
    """运行Walk-Forward, 返回(result_dict, window_details)。"""
    engine = PortfolioEngine(
        cost=cost, stop_loss=stop_loss,
        portfolio_method=portfolio_method,
        price_data=price_data,
    )
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
                ranks, prices, factor_weights, hold_days, top_n,
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

    ri = check_rank_inversion(windows)

    result = {
        "sharpe": round(float(np.mean(sharpes)), 3),
        "max_dd": round(float(np.min(dds)), 4),
        "cagr": round(float(np.mean(cagrs)), 4),
        "win_rate": round(float(np.mean(wrs)), 3),
        "n_trades": sum(w["n_trades"] for w in valid),
        "n_windows": len(valid),
        "rank_inversion": ri,
        "warnings": [],
        "status": "PASS",
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  Candidate Tracker
# ═══════════════════════════════════════════════════

class CandidateTracker:
    """跟踪所有候选方案。"""
    def __init__(self):
        self.candidates = []

    def add(self, name, config, res, category=""):
        if res and "sharpe" in res:
            self.candidates.append({
                "name": name,
                "category": category,
                "config": config,
                "sharpe": res["sharpe"],
                "max_dd": res["max_dd"],
                "cagr": res["cagr"],
                "win_rate": res["win_rate"],
                "n_windows": res["n_windows"],
                "rank_inversion_passed": res["rank_inversion"]["passed"],
                "rank_inversion_detail": res["rank_inversion"],
            })

    def best(self):
        """返回最佳(RI通过, Sharpe最高)。"""
        ri_passed = [c for c in self.candidates if c["rank_inversion_passed"]]
        pool = ri_passed if ri_passed else self.candidates
        if not pool:
            return None
        return max(pool, key=lambda x: x["sharpe"])

    def best_in_category(self, category):
        """返回某类别的最佳。"""
        filtered = [c for c in self.candidates if c["category"] == category and c["rank_inversion_passed"]]
        if not filtered:
            filtered = [c for c in self.candidates if c["category"] == category]
        if not filtered:
            return None
        return max(filtered, key=lambda x: x["sharpe"])


# ═══════════════════════════════════════════════════
#  TEST 1: 组合构建方法
# ═══════════════════════════════════════════════════

def test1_portfolio_methods(ranks, prices, tracker, train_years=0.5):
    """测试不同组合构建方法。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 1: Portfolio Construction Methods")
    print(f"{'='*60}")

    methods = [
        ("equal", "等权 (baseline)"),
        ("mcap", "市值加权"),
        ("risk_parity", "风险平价"),
    ]

    for method, desc in methods:
        print(f"\n  --- {method}: {desc} ---")
        res, _ = run_wf(
            ranks, prices, V045_WEIGHTS,
            train_years=train_years,
            portfolio_method=method,
            price_data=prices,
        )
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(
                f"method_{method}",
                {"portfolio_method": method, "top_n": 10, "stop_loss": -0.15},
                res,
                category="portfolio_method",
            )
            ri_str = "PASS" if res["rank_inversion"]["passed"] else "FAIL"
            print(f"    Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
                  f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}  RI={ri_str}")

    best = tracker.best_in_category("portfolio_method")
    if best:
        print(f"\n  🏆 Best method: {best['name']} (Sharpe={best['sharpe']:.3f})")


# ═══════════════════════════════════════════════════
#  TEST 2: 股票数量
# ═══════════════════════════════════════════════════

def test2_stock_count(ranks, prices, best_method, tracker, train_years=0.5):
    """测试不同股票数量。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 2: Stock Count (method={best_method})")
    print(f"{'='*60}")

    counts = [5, 10, 15, 20]

    for n in counts:
        print(f"\n  --- top_n={n} ---")
        res, _ = run_wf(
            ranks, prices, V045_WEIGHTS,
            train_years=train_years,
            top_n=n,
            portfolio_method=best_method,
            price_data=prices,
        )
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(
                f"count_{n}",
                {"portfolio_method": best_method, "top_n": n, "stop_loss": -0.15},
                res,
                category="stock_count",
            )
            ri_str = "PASS" if res["rank_inversion"]["passed"] else "FAIL"
            print(f"    Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
                  f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}  RI={ri_str}")

    best = tracker.best_in_category("stock_count")
    if best:
        best_n = best["config"]["top_n"]
        print(f"\n  🏆 Best count: top_n={best_n} (Sharpe={best['sharpe']:.3f})")
        return best_n
    return 10


# ═══════════════════════════════════════════════════
#  TEST 3: 止损水平
# ═══════════════════════════════════════════════════

def test3_stop_loss(ranks, prices, best_method, best_n, tracker, train_years=0.5):
    """测试不同止损水平。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 3: Stop Loss (method={best_method}, top_n={best_n})")
    print(f"{'='*60}")

    stop_losses = [-0.10, -0.15, -0.20]

    for sl in stop_losses:
        print(f"\n  --- stop_loss={sl:.0%} ---")
        res, _ = run_wf(
            ranks, prices, V045_WEIGHTS,
            train_years=train_years,
            top_n=best_n,
            stop_loss=sl,
            portfolio_method=best_method,
            price_data=prices,
        )
        if res:
            res["status"] = "PASS" if res["rank_inversion"]["passed"] else "WARN"
            tracker.add(
                f"sl_{abs(int(sl*100))}pct",
                {"portfolio_method": best_method, "top_n": best_n, "stop_loss": sl},
                res,
                category="stop_loss",
            )
            ri_str = "PASS" if res["rank_inversion"]["passed"] else "FAIL"
            print(f"    Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
                  f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}  RI={ri_str}")

    best = tracker.best_in_category("stop_loss")
    if best:
        best_sl = best["config"]["stop_loss"]
        print(f"\n  🏆 Best stop_loss: {best_sl:.0%} (Sharpe={best['sharpe']:.3f})")
        return best_sl
    return -0.15


# ═══════════════════════════════════════════════════
#  TEST 4: 最终验证
# ═══════════════════════════════════════════════════

def test4_final_validation(ranks, prices, best_method, best_n, best_sl, train_years=0.5):
    """最终Walk-Forward + 详细窗口。"""
    print(f"\n{'='*60}")
    print(f"📊 TEST 4: Final Walk-Forward Validation")
    print(f"{'='*60}")
    print(f"  Config: method={best_method}, top_n={best_n}, stop_loss={best_sl:.0%}")
    print(f"  Weights: {V045_WEIGHTS}")

    res, wins = run_wf(
        ranks, prices, V045_WEIGHTS,
        train_years=train_years,
        top_n=best_n,
        stop_loss=best_sl,
        portfolio_method=best_method,
        price_data=prices,
    )
    if res:
        print(f"\n  Sharpe={res['sharpe']:.3f}  MaxDD={res['max_dd']:.1%}  "
              f"CAGR={res['cagr']:.1%}  WR={res['win_rate']:.0%}")
        print(f"  Windows={res['n_windows']}  RI={'PASS' if res['rank_inversion']['passed'] else 'FAIL'}")

        valid_wins = [w for w in wins if "sharpe" in w]
        for w in valid_wins:
            ri_mark = "✅" if w["sharpe"] > 0 else "❌"
            baseline_str = f"  Base={w['baseline_sharpe']:.3f}" if w.get('baseline_sharpe') else ""
            print(f"    {ri_mark} W{w['index']}: {w['period']}  "
                  f"Sharpe={w['sharpe']:.3f}  MaxDD={w['max_dd']:.1%}  "
                  f"WR={w['win_rate']:.0%}  Trades={w['n_trades']}{baseline_str}")
        return res, wins
    return None, []


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
    print("🦅 Falcon V0.4.5: Portfolio Construction Optimization")
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

    # ─── 5. 添加growth_composite ───
    ranks = add_growth_composite(ranks)

    # ═══════════════════════════════════════════════
    #  追踪器
    # ═══════════════════════════════════════════════
    tracker = CandidateTracker()

    # ═══════════════════════════════════════════════
    #  TEST 1: 组合构建方法
    # ═══════════════════════════════════════════════
    test1_portfolio_methods(ranks, price_pivot, tracker, train_years=0.5)

    # 确定最优方法
    best_method_candidate = tracker.best_in_category("portfolio_method")
    best_method = best_method_candidate["config"]["portfolio_method"] if best_method_candidate else "equal"
    print(f"\n  → Selected method: {best_method}")

    # ═══════════════════════════════════════════════
    #  TEST 2: 股票数量
    # ═══════════════════════════════════════════════
    best_n = test2_stock_count(ranks, price_pivot, best_method, tracker, train_years=0.5)
    print(f"\n  → Selected top_n: {best_n}")

    # ═══════════════════════════════════════════════
    #  TEST 3: 止损水平
    # ═══════════════════════════════════════════════
    best_sl = test3_stop_loss(ranks, price_pivot, best_method, best_n, tracker, train_years=0.5)
    print(f"\n  → Selected stop_loss: {best_sl:.0%}")

    # ═══════════════════════════════════════════════
    #  TEST 4: 最终验证
    # ═══════════════════════════════════════════════
    final_result, final_windows = test4_final_validation(
        ranks, price_pivot, best_method, best_n, best_sl, train_years=0.5
    )

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.5 Portfolio Construction Optimization",
            "config": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "cost": 0.001,
                "factor_weights": V045_WEIGHTS,
            },
            "features": "features_v04_1.parquet",
            "pit_factors": len(all_pit_cols),
            "factor_groups": {k: len(v) for k, v in FACTOR_GROUPS.items()},
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "test1_portfolio_methods": {
            "description": "Equal vs Market Cap vs Risk Parity",
            "best_method": best_method,
            "candidates": serialize([
                c for c in tracker.candidates if c["category"] == "portfolio_method"
            ]),
        },
        "test2_stock_count": {
            "description": "top_n: 5/10/15/20",
            "best_top_n": best_n,
            "candidates": serialize([
                c for c in tracker.candidates if c["category"] == "stock_count"
            ]),
        },
        "test3_stop_loss": {
            "description": "stop_loss: -10%/-15%/-20%",
            "best_stop_loss": best_sl,
            "candidates": serialize([
                c for c in tracker.candidates if c["category"] == "stop_loss"
            ]),
        },
        "final_optimal_config": {
            "portfolio_method": best_method,
            "top_n": best_n,
            "stop_loss": best_sl,
            "factor_weights": V045_WEIGHTS,
        },
        "final_validation": {
            "result": serialize(final_result) if final_result else None,
            "window_details": serialize(final_windows),
        },
        "all_candidates": serialize(tracker.candidates),
        "comparison_with_v044": {
            "v044_sharpe": 2.122,
            "v044_config": {
                "fund_ratio": 0.45,
                "growth_composite": 0.20,
                "qoq": 0.20,
                "cashflow": 0.15,
                "portfolio_method": "equal",
                "top_n": 10,
                "stop_loss": -0.15,
            },
            "v045_best_sharpe": final_result["sharpe"] if final_result else 0,
            "v045_best_config": {
                "portfolio_method": best_method,
                "top_n": best_n,
                "stop_loss": best_sl,
            },
            "improvement_pct": round(
                (final_result["sharpe"] - 2.122) / 2.122 * 100, 1
            ) if final_result else 0,
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
    print(f"  Factor weights: {V045_WEIGHTS}")
    print(f"  Optimal portfolio method: {best_method}")
    print(f"  Optimal stock count: {best_n}")
    print(f"  Optimal stop loss: {best_sl:.0%}")
    if final_result:
        print(f"  Final WF Sharpe: {final_result['sharpe']:.3f}")
        print(f"  Final WF MaxDD: {final_result['max_dd']:.1%}")
        print(f"  Rank Inversion: {'PASS' if final_result['rank_inversion']['passed'] else 'FAIL'}")
    print(f"  V0.4.4 baseline Sharpe: 2.122")
    if final_result:
        improvement = (final_result['sharpe'] - 2.122) / 2.122 * 100
        print(f"  Improvement vs V0.4.4: {improvement:+.1f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
