#!/usr/bin/env python3
"""
🦅 Falcon V0.4.6: 2026年逐日持仓回放
=======================================
从2026-01-01到今天，逐日模拟：
- 每周一：重新评分，调仓（卖出到期/止损的，买入新Top10）
- 每天：检查止损、VIX、到期
- 输出每日持仓状态

用法:
    python3 scripts/falcon/v046_ytd_simulation.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
IC_WEIGHTS_PATH = DATA_DIR / "factor_ic_weights.json"

# V0.4.6 Factor groups (same as falcon_score.py)
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
    'analyst': ['a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps'],
    'income': ['i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
               'i_revenue_growth_yoy', 'i_gross_margin_delta'],
    'qoq': ['r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq'],
    'cashflow': ['c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield'],
}

FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity', 'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}

V046_WEIGHTS = {"fund_ratio": 0.45, "growth_composite": 0.20, "qoq": 0.20, "cashflow": 0.15}
GC_WEIGHTS = {"fund_growth": 0.60, "analyst": 0.25, "income": 0.15}

TOP_N = 10
HOLD_DAYS = 30
STOP_LOSS = -0.15
VIX_THRESHOLD = 25
COST = 0.001  # 0.1% per trade


def compute_group_score_ic(day, group_cols, flip_set, ic_weights):
    available = [c for c in group_cols if c in day.columns and day[c].notna().sum() > 5]
    if not available:
        return pd.Series(0.5, index=day.index)
    
    ranks = pd.DataFrame(index=day.index)
    for col in available:
        r = day[col].rank(pct=True)
        if col in flip_set:
            r = 1 - r
        ranks[col] = r
    
    if ic_weights:
        weights = {col: max(0, ic_weights.get(col, 0)) for col in available}
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
            score = pd.Series(0.0, index=day.index)
            for col in available:
                score += weights[col] * ranks[col]
            return score
    
    return ranks.mean(axis=1)


def score_date(features, date_str, ic_data):
    day = features[features["date"] == date_str].copy()
    if len(day) < 10:
        return None
    day.index = day["ticker"].values
    
    ic_weights_by_group = ic_data.get('weights', {})
    
    scores = {}
    scores["fund_ratio"] = compute_group_score_ic(
        day, FACTOR_GROUPS["fund_ratio"], FLIP_FACTORS, ic_weights_by_group.get('fund_ratio', {}))
    
    fg = compute_group_score_ic(day, FACTOR_GROUPS["fund_growth"], FLIP_FACTORS, ic_weights_by_group.get('fund_growth', {}))
    an = compute_group_score_ic(day, FACTOR_GROUPS["analyst"], FLIP_FACTORS, ic_weights_by_group.get('analyst', {}))
    inc = compute_group_score_ic(day, FACTOR_GROUPS["income"], FLIP_FACTORS, ic_weights_by_group.get('income', {}))
    scores["growth_composite"] = GC_WEIGHTS["fund_growth"] * fg + GC_WEIGHTS["analyst"] * an + GC_WEIGHTS["income"] * inc
    
    scores["qoq"] = compute_group_score_ic(day, FACTOR_GROUPS["qoq"], FLIP_FACTORS, ic_weights_by_group.get('qoq', {}))
    scores["cashflow"] = compute_group_score_ic(day, FACTOR_GROUPS["cashflow"], FLIP_FACTORS, ic_weights_by_group.get('cashflow', {}))
    
    falcon_score = sum(V046_WEIGHTS[f] * scores[f] for f in V046_WEIGHTS)
    
    result = day[["ticker", "close"]].copy()
    result["falcon_score"] = falcon_score
    result["rank_pct"] = falcon_score.rank(pct=True)
    return result.sort_values("falcon_score", ascending=False)


def main():
    print("🦅 Falcon V0.4.6: 2026年逐日持仓回放")
    print("=" * 60)
    
    # Load data
    print("📂 加载数据...")
    features = pd.read_parquet(FEATURES_PATH)
    features["date"] = features["date"].astype(str)
    
    prices = pd.read_parquet(PRICES_PATH)
    if "date" in prices.columns:
        prices["date"] = prices["date"].astype(str)
    
    # Load IC weights (use latest available)
    with open(IC_WEIGHTS_PATH) as f:
        ic_data = json.load(f)
    print(f"  IC权重: {ic_data.get('computed_at', 'unknown')}")
    
    # Get all trading dates in 2026
    all_dates = sorted(features["date"].unique())
    dates_2026 = [d for d in all_dates if d >= "2026-01-01"]
    print(f"  2026年交易日: {len(dates_2026)} 天 ({dates_2026[0]} ~ {dates_2026[-1]})")
    
    # VIX data
    vix_map = {}
    try:
        vix_path = PROJECT_ROOT / "data" / "us" / "vix_10y.parquet"
        if vix_path.exists():
            vix_raw = pd.read_parquet(vix_path)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_close = vix_raw[("Close", "^VIX")]
            elif "Close" in vix_raw.columns:
                vix_close = vix_raw["Close"]
            else:
                vix_close = vix_raw.iloc[:, 0]
            for d, v in zip(vix_raw.index, vix_close):
                ds = str(d)[:10]
                vix_map[ds] = float(v)
    except Exception:
        pass
    
    # Build price lookup: {ticker: {date: close}}
    price_lookup = {}
    for _, row in prices.iterrows():
        t = row.get("ticker", row.get("Ticker", ""))
        d = str(row.get("date", ""))[:10]
        c = row.get("close", row.get("Close", 0))
        if t and d:
            if t not in price_lookup:
                price_lookup[t] = {}
            price_lookup[t][d] = float(c)
    
    # Simulate
    print("\n📊 开始逐日回放...")
    print("-" * 80)
    
    portfolio = {}  # {ticker: {"entry_date": str, "entry_price": float, "shares": float}}
    cash = 100000.0
    initial_capital = 100000.0
    daily_nav = []
    trade_log = []
    weekly_scored = {}  # {date: DataFrame}
    
    for date in dates_2026:
        vix = vix_map.get(date, 20)
        is_monday = pd.Timestamp(date).dayofweek == 0
        
        # Check existing positions
        sells = []
        for ticker, pos in list(portfolio.items()):
            current_price = price_lookup.get(ticker, {}).get(date)
            if current_price is None:
                continue
            
            hold_days = (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days
            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            
            # Stop loss
            if pnl_pct <= STOP_LOSS:
                sells.append((ticker, "止损", pnl_pct, current_price, hold_days))
            # Expired (30 days)
            elif hold_days >= HOLD_DAYS:
                sells.append((ticker, "到期", pnl_pct, current_price, hold_days))
        
        # Execute sells
        for ticker, reason, pnl_pct, price, hold_days in sells:
            pos = portfolio.pop(ticker)
            proceeds = pos["shares"] * price * (1 - COST)
            cash += proceeds
            trade_log.append({
                "date": date, "action": "SELL", "ticker": ticker, "reason": reason,
                "price": price, "pnl_pct": round(pnl_pct * 100, 1), "hold_days": hold_days
            })
        
        # Weekly rebalance (Monday)
        new_picks = []
        if is_monday:
            scored = score_date(features, date, ic_data)
            if scored is not None:
                weekly_scored[date] = scored
                top = scored.head(TOP_N)
                
                # Buy new picks (not already held)
                for _, row in top.iterrows():
                    ticker = row["ticker"]
                    if ticker not in portfolio and vix < VIX_THRESHOLD:
                        buy_price = price_lookup.get(ticker, {}).get(date)
                        if buy_price and buy_price > 0:
                            alloc = initial_capital / TOP_N  # equal weight
                            shares = alloc / buy_price
                            cost_total = alloc * (1 + COST)
                            if cost_total <= cash:
                                cash -= cost_total
                                portfolio[ticker] = {
                                    "entry_date": date, "entry_price": buy_price, "shares": shares
                                }
                                new_picks.append(ticker)
                                trade_log.append({
                                    "date": date, "action": "BUY", "ticker": ticker,
                                    "price": buy_price, "score": round(row["falcon_score"], 3)
                                })
        
        # Calculate NAV
        port_value = 0
        for ticker, pos in portfolio.items():
            cp = price_lookup.get(ticker, {}).get(date, pos["entry_price"])
            port_value += pos["shares"] * cp
        
        nav = cash + port_value
        daily_nav.append({"date": date, "nav": nav, "cash": cash, "positions": len(portfolio), "vix": vix})
        
        # Print summary for notable days
        if is_monday or sells:
            held_tickers = list(portfolio.keys())
            pnl_total = (nav - initial_capital) / initial_capital * 100
            sell_str = " ".join([f"🔴{t}({r})" for t, r, _, _, _ in sells]) if sells else ""
            buy_str = " ".join([f"🟢{t}" for t in new_picks]) if new_picks else ""
            print(f"{date} | NAV=${nav:,.0f} ({pnl_total:+.1f}%) | 持仓{len(portfolio)} | "
                  f"VIX={vix:.0f} | {sell_str} {buy_str}")
    
    # Final summary
    print("\n" + "=" * 80)
    nav_df = pd.DataFrame(daily_nav)
    nav_df["return"] = nav_df["nav"].pct_change()
    
    final_nav = nav_df["nav"].iloc[-1]
    total_return = (final_nav - initial_capital) / initial_capital * 100
    sharpe = nav_df["return"].mean() / nav_df["return"].std() * np.sqrt(252) if nav_df["return"].std() > 0 else 0
    
    # Max drawdown
    cum = nav_df["nav"].values
    max_dd = (cum / np.maximum.accumulate(cum) - 1).min() * 100
    
    print(f"\n📊 2026年YTD表现:")
    print(f"  起始资金: ${initial_capital:,.0f}")
    print(f"  最终NAV: ${final_nav:,.0f}")
    print(f"  总收益: {total_return:+.1f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.1f}%")
    print(f"  交易次数: {len(trade_log)}")
    print(f"  当前持仓: {len(portfolio)} 只")
    
    # Current holdings
    print(f"\n📋 当前持仓:")
    print("-" * 60)
    current_holdings = []
    for ticker, pos in sorted(portfolio.items()):
        cp = price_lookup.get(ticker, {}).get(dates_2026[-1], pos["entry_price"])
        pnl = (cp - pos["entry_price"]) / pos["entry_price"] * 100
        hold_days = (pd.Timestamp(dates_2026[-1]) - pd.Timestamp(pos["entry_date"])).days
        current_holdings.append({
            "ticker": ticker, "entry_date": pos["entry_date"], "entry_price": pos["entry_price"],
            "current_price": cp, "pnl_pct": pnl, "hold_days": hold_days
        })
        print(f"  {ticker:6s} | 买入{pos['entry_date']} @${pos['entry_price']:.2f} | "
              f"现价${cp:.2f} | {'🟢' if pnl > 0 else '🔴'}{pnl:+.1f}% | 持有{hold_days}天")
    
    # Today's latest score
    print(f"\n🏆 今日最新评分 ({dates_2026[-1]}):")
    print("-" * 60)
    latest_scored = score_date(features, dates_2026[-1], ic_data)
    if latest_scored is not None:
        for i, (_, row) in enumerate(latest_scored.head(TOP_N).iterrows()):
            held = "📌" if row["ticker"] in portfolio else "  "
            signal = "🟢🟢" if row["falcon_score"] >= 0.55 and row["rank_pct"] >= 0.95 else \
                     "🟢" if row["falcon_score"] >= 0.55 and row["rank_pct"] >= 0.80 else \
                     "🟡" if row["falcon_score"] >= 0.50 else "🔴"
            print(f"  {held}{signal} {row['ticker']:6s} Score={row['falcon_score']:.3f} "
                  f"Pct={row['rank_pct']:.1%} Close=${row['close']:.2f}")
    
    # Save results
    output = {
        "version": "V0.4.6",
        "period": f"{dates_2026[0]} ~ {dates_2026[-1]}",
        "total_return_pct": round(total_return, 1),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 1),
        "total_trades": len(trade_log),
        "current_holdings": current_holdings,
        "trade_log": trade_log[-20:],  # last 20 trades
    }
    out_path = DATA_DIR / "v046_ytd_simulation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n📁 结果已保存: {out_path}")


if __name__ == "__main__":
    main()
