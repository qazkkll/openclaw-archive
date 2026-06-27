#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — 灵活信号驱动调仓 + 全量FMP因子 + Futu成本
支持5种调仓策略, 动态成本模型
"""
import pandas as pd, numpy as np, json, time
from pathlib import Path

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")

# ═══════════════════════════════════════════════════
# Futu动态成本模型
# ═══════════════════════════════════════════════════
def futu_cost(price, action="buy"):
    """Futu美股实际费用: 佣金+平台费+SEC费。"""
    per_share = 0.0049 + 0.005  # 佣金+平台费
    min_fee = 0.99 + 1.0  # 最低
    cost = max(min_fee, per_share * 100) / (price * 100)  # 假设100股
    if action == "sell":
        cost += 0.0000278  # SEC fee
    return cost


# ═══════════════════════════════════════════════════
# 特征定义 (全量FMP)
# ═══════════════════════════════════════════════════
TECH_FIELDS = ["rsi14", "macd_hist", "momentum_1m", "vol20", "bb_pos",
               "ma_align", "ret_quality", "dd_60", "ud_vol_ratio"]

# FMP Ratios (20个)
RATIO_FIELDS = ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
                "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
                "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
                "ebitdaMargin", "assetTurnover", "inventoryTurnover",
                "receivablesTurnover", "debtToEquityRatio", "currentRatio",
                "quickRatio", "financialLeverageRatio",
                "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
                "dividendYieldPercentage", "dividendPayoutRatio"]

# Key Metrics (23个)
METRIC_FIELDS = ["earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
                 "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
                 "returnOnCapitalEmployed", "returnOnInvestedCapital",
                 "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
                 "cashConversionCycle", "capexToRevenue", "capexToDepreciation",
                 "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
                 "netDebtToEBITDA", "operatingReturnOnAssets"]

# Financial Growth (18个)
GROWTH_FIELDS = ["revenueGrowth", "grossProfitGrowth", "ebitgrowth",
                 "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
                 "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
                 "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
                 "receivablesGrowth", "inventoryGrowth", "assetGrowth",
                 "bookValueperShareGrowth", "debtGrowth"]

# Analyst (3个)
ANALYST_FIELDS = ["eps_revision", "revenue_revision", "eps_dispersion"]

ALL_FMP_FIELDS = RATIO_FIELDS + METRIC_FIELDS + GROWTH_FIELDS


def get_pit(quarterly_data, date):
    """Point-in-time: 返回date之前最新季度数据。"""
    if not quarterly_data:
        return {}
    latest = {}
    for q in quarterly_data:
        if isinstance(q, dict) and q.get("date", "") <= date:
            latest = q
    return latest


def get_pit_insider(insider_data, date, lookback_days=90):
    """Point-in-time insider: 过去90天的净买卖。"""
    if not insider_data:
        return {}
    from datetime import datetime, timedelta
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
    except:
        return {}
    start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    net_buy_shares = 0
    net_buy_value = 0
    n_buy = 0
    n_sell = 0
    ceo_buy = 0
    
    for t in insider_data:
        td = t.get("date", "")
        if td < start or td > date:
            continue
        acq = t.get("acq_disp", "")
        shares = t.get("shares", 0) or 0
        price = t.get("price", 0) or 0
        owner = t.get("owner", "").lower()
        
        if acq == "A":
            net_buy_shares += shares
            net_buy_value += shares * price
            n_buy += 1
            if "ceo" in owner or "chief executive" in owner:
                ceo_buy += shares
        elif acq == "D":
            net_buy_shares -= shares
            net_buy_value -= shares * price
            n_sell += 1
    
    return {
        "insider_net_shares": net_buy_shares,
        "insider_net_value": net_buy_value,
        "insider_buy_count": n_buy,
        "insider_sell_count": n_sell,
        "insider_net_count": n_buy - n_sell,
        "insider_ceo_buy": ceo_buy,
    }


def get_pit_dcf(dcf_data, price_target_data):
    """DCF估值偏离度。"""
    dcf = dcf_data.get("dcf")
    stock_price = dcf_data.get("price")
    if dcf and stock_price and stock_price > 0:
        dcf_upside = (dcf - stock_price) / stock_price
    else:
        dcf_upside = None
    
    pt_consensus = price_target_data.get("targetConsensus")
    if pt_consensus and stock_price and stock_price > 0:
        pt_upside = (pt_consensus - stock_price) / stock_price
    else:
        pt_upside = None
    
    return {"dcf_upside": dcf_upside, "pt_upside": pt_upside}


# ═══════════════════════════════════════════════════
# 预计算PIT rank (全量因子)
# ═══════════════════════════════════════════════════
def precompute_pit_ranks(master, fmp_hist, ana_hist, metrics_hist,
                          growth_hist, insider_hist, dcf_data, pt_data):
    """预计算全量PIT截面rank。"""
    print("📊 预计算全量PIT rank...")
    dates = sorted(master["date"].unique())
    ranks_dict = {}
    
    for date in dates:
        day = master[master["date"] == date].copy()
        if len(day) < 10:
            continue
        day.index = day["ticker"].values
        row = day[["ticker"]].copy()
        
        # ── Tech rank (K线) ──
        tech_r = []
        for f in TECH_FIELDS:
            if f in day.columns and day[f].notna().sum() > 5:
                row[f"t_{f}"] = day[f].rank(pct=True)
                tech_r.append(f"t_{f}")
        row["tech"] = row[tech_r].mean(axis=1) if tech_r else 0.5
        
        # ── FMP Ratios rank ──
        for f in RATIO_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                pit = get_pit(fmp_hist.get(t, []), date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"r_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── Key Metrics rank ──
        for f in METRIC_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                pit = get_pit(metrics_hist.get(t, []), date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"m_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── Financial Growth rank ──
        for f in GROWTH_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                pit = get_pit(growth_hist.get(t, []), date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"g_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── Analyst rank ──
        for f in ANALYST_FIELDS:
            vals = {}
            for t in day["ticker"].values:
                pit = get_pit(ana_hist.get(t, []), date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 5:
                row[f"a_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── Insider rank ──
        insider_fields = ["insider_net_shares", "insider_net_count", "insider_ceo_buy"]
        for f in insider_fields:
            vals = {}
            for t in day["ticker"].values:
                pit = get_pit_insider(insider_hist.get(t, []), date)
                v = pit.get(f)
                if v is not None and v != 0:
                    vals[t] = v
            if len(vals) > 5:
                row[f"i_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── DCF/PT rank ──
        for f in ["dcf_upside", "pt_upside"]:
            vals = {}
            for t in day["ticker"].values:
                pit = get_pit_dcf(dcf_data.get(t, {}), pt_data.get(t, {}))
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"d_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── 分组得分 ──
        r_cols = [c for c in row.columns if c.startswith("r_")]
        m_cols = [c for c in row.columns if c.startswith("m_")]
        g_cols = [c for c in row.columns if c.startswith("g_")]
        a_cols = [c for c in row.columns if c.startswith("a_")]
        i_cols = [c for c in row.columns if c.startswith("i_")]
        d_cols = [c for c in row.columns if c.startswith("d_")]
        
        row["fund_ratio"] = row[r_cols].mean(axis=1) if r_cols else 0.5
        row["fund_metric"] = row[m_cols].mean(axis=1) if m_cols else 0.5
        row["fund_growth"] = row[g_cols].mean(axis=1) if g_cols else 0.5
        row["analyst"] = row[a_cols].mean(axis=1) if a_cols else 0.5
        row["insider"] = row[i_cols].mean(axis=1) if i_cols else 0.5
        row["valuation"] = row[d_cols].mean(axis=1) if d_cols else 0.5
        
        ranks_dict[date] = row.set_index("ticker")[[
            "tech", "fund_ratio", "fund_metric", "fund_growth",
            "analyst", "insider", "valuation"
        ]]
    
    print(f"✅ 全量PIT rank: {len(ranks_dict)} 天, 7个因子组")
    return ranks_dict


# ═══════════════════════════════════════════════════
# 灵活调仓引擎
# ═══════════════════════════════════════════════════
def backtest_flexible(ranks_dict, price_pivot, dates, regime_above,
                      weights, strategy="fixed", params=None, top_n=5):
    """
    灵活调仓引擎。
    
    strategy:
      "fixed"     — 固定周期调仓 (params: hold_days)
      "signal"    — 信号驱动: 持仓排名跌出阈值就换 (params: rank_threshold, check_every)
      "hybrid"    — 固定检查+信号退出 (params: check_every, rank_threshold)
      "event"     — 事件驱动: FMP新数据发布时调仓 (params: ~)
      "adaptive"  — 自适应: 波动率高→缩短hold, 低→延长 (params: base_hold, vol_factor)
    
    weights: dict of factor_name -> weight
    """
    if params is None:
        params = {}
    
    cost = params.get("cost", 0.001)  # 默认0.1% per side (Futu)
    stop_loss = params.get("stop_loss", -0.15)
    bear_alloc = params.get("bear_alloc", 0.50)
    
    cash = 100000.0
    portfolio = {}  # ticker -> (entry_idx, entry_price, shares)
    values = []
    trades = []
    rebalance_count = 0
    
    def get_scores(date):
        if date not in ranks_dict:
            return None
        r = ranks_dict[date]
        combined = sum(w * r[f] for f, w in weights.items() if f in r.columns)
        return combined.dropna().sort_values(ascending=False)
    
    for i, date in enumerate(dates):
        if date not in price_pivot.index:
            continue
        pr = price_pivot.loc[date]
        above = regime_above.loc[date] if date in regime_above.index else 1
        alloc = bear_alloc if above == 0 else 1.0
        
        # ── 止损 ──
        to_close = []
        for t, (ei, ep, sh) in portfolio.items():
            if t in pr and not pd.isna(pr[t]):
                pnl = (pr[t] - ep) / ep
                if pnl <= stop_loss:
                    price = pr[t]
                    futu_sell = futu_cost(price, "sell")
                    cash += sh * price * (1 - futu_sell)
                    trades.append({"pnl": pnl, "reason": "止损", "date": date})
                    to_close.append(t)
        for t in to_close:
            del portfolio[t]
        
        # ── 调仓逻辑 ──
        should_rebalance = False
        sell_tickers = []
        
        if strategy == "fixed":
            hold_days = params.get("hold_days", 30)
            # 到期全部卖出
            for t, (ei, ep, sh) in list(portfolio.items()):
                if (i - ei) >= hold_days:
                    sell_tickers.append(t)
            if sell_tickers or len(portfolio) == 0:
                should_rebalance = True
        
        elif strategy == "signal":
            check_every = params.get("check_every", 5)
            rank_threshold = params.get("rank_threshold", 0.5)
            if i % check_every == 0 or len(portfolio) == 0:
                scores = get_scores(date)
                if scores is not None:
                    for t in list(portfolio.keys()):
                        if t in scores.index:
                            rank = scores.index.get_loc(t) / len(scores)
                            if rank > rank_threshold:
                                sell_tickers.append(t)
                    should_rebalance = True
        
        elif strategy == "hybrid":
            check_every = params.get("check_every", 20)
            rank_threshold = params.get("rank_threshold", 0.3)
            hold_min = params.get("hold_min", 10)
            if i % check_every == 0 or len(portfolio) == 0:
                scores = get_scores(date)
                if scores is not None:
                    for t, (ei, ep, sh) in list(portfolio.items()):
                        rank = scores.index.get_loc(t) / len(scores) if t in scores.index else 1.0
                        if rank > rank_threshold and (i - ei) >= hold_min:
                            sell_tickers.append(t)
                    should_rebalance = True
        
        elif strategy == "adaptive":
            base_hold = params.get("base_hold", 30)
            vol_factor = params.get("vol_factor", 2.0)
            # 根据市场波动率调整hold期
            mkt_vol = regime_above.get(date, 0.15) if hasattr(regime_above, 'get') else 0.15
            # 简化: 用近20天的市场波动率
            recent_dates = [d for d in dates[max(0,i-20):i] if d in price_pivot.index]
            if len(recent_dates) > 5:
                recent_prices = price_pivot.loc[recent_dates]
                mkt_ret = recent_prices.pct_change(fill_method=None).mean(axis=1)
                mkt_vol = mkt_ret.std() * np.sqrt(252)
            adaptive_hold = int(base_hold * (1 + (mkt_vol - 0.15) * vol_factor))
            adaptive_hold = max(10, min(90, adaptive_hold))
            
            for t, (ei, ep, sh) in list(portfolio.items()):
                if (i - ei) >= adaptive_hold:
                    sell_tickers.append(t)
            if sell_tickers or len(portfolio) == 0:
                should_rebalance = True
        
        # ── 执行卖出 ──
        for t in sell_tickers:
            if t in portfolio and t in pr and not pd.isna(pr[t]):
                ei, ep, sh = portfolio.pop(t)
                price = pr[t]
                futu_sell = futu_cost(price, "sell")
                cash += sh * price * (1 - futu_sell)
                pnl = (price - ep) / ep
                trades.append({"pnl": pnl, "reason": "调仓", "date": date})
        
        # ── 买入 ──
        if should_rebalance and len(portfolio) == 0 and cash > 100:
            scores = get_scores(date)
            if scores is not None:
                deploy = cash * alloc
                reserve = cash - deploy
                picks = scores.head(top_n).index.tolist()
                per = deploy / len(picks) if picks else 0
                for t in picks:
                    if t in pr and not pd.isna(pr[t]) and pr[t] > 0:
                        price = pr[t]
                        futu_buy = futu_cost(price, "buy")
                        sh = (per * (1 - futu_buy)) / price
                        portfolio[t] = (i, price, sh)
                cash = reserve
                rebalance_count += 1
        
        # ── 记录净值 ──
        pv = cash
        for t, (_, ep, sh) in portfolio.items():
            pv += sh * (pr[t] if t in pr and not pd.isna(pr[t]) else ep)
        values.append(pv)
    
    if len(values) < 20:
        return None
    
    v = np.array(values, dtype=np.float64)
    rets = np.diff(v) / np.where(v[:-1] > 0, v[:-1], 1)
    std = np.std(rets)
    if std == 0:
        return None
    
    sr = np.mean(rets) / std * np.sqrt(252)
    tr = (v[-1]/v[0]-1)*100
    pk = np.maximum.accumulate(v)
    dd = ((pk-v)/pk).max()*100
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / len(trades) * 100 if trades else 0
    
    return {"sharpe": round(sr, 3), "dd": round(dd, 2), "ret": round(tr, 2),
            "wr": round(wr, 1), "trades": len(trades), "rebalances": rebalance_count,
            "avg_cost_pct": round(np.mean([futu_cost(p) for p in [50]]) * 100, 3)}
