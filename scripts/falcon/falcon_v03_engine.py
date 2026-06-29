#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — 灵活信号驱动调仓 + 全量FMP因子 + Futu成本
支持5种调仓策略, 动态成本模型
"""
import pandas as pd, numpy as np, json, time
from pathlib import Path
from bisect import bisect_right
from datetime import datetime, timedelta

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

# FMP Premium Earnings (4个)
EARNINGS_FIELDS = ["earnings_surprise", "earnings_surprise_2q", "earnings_beat_count_4q", "earnings_price_reaction"]

# FMP Premium Grade Sentiment (4个)
GRADE_FIELDS = ["grade_upgrade_ratio_90d", "grade_downgrade_ratio_90d", "grade_momentum_90d", "grade_target_raised_90d"]

# ═══════════════════════════════════════════════════
# 三大报表因子 (V0.3.2新增, 2026-06-30)
# 来源: fmp_balance_sheet.json / fmp_cashflow.json / fmp_income_stmt.json
# ═══════════════════════════════════════════════════

BALANCE_FIELDS = [
    "debt_to_equity",       # totalDebt / totalStockholdersEquity (低=好)
    "cash_to_assets",       # cashAndCashEquivalents / totalAssets (高=好)
    "net_debt_to_assets",   # netDebt / totalAssets (低=好)
    "equity_ratio",         # totalStockholdersEquity / totalAssets (高=好)
]

CASHFLOW_FIELDS = [
    "fcf_margin",           # freeCashFlow / revenue (高=好, 需income配对)
    "ocf_margin",           # operatingCashFlow / revenue (高=好, 需income配对)
    "capex_intensity",      # capitalExpenditure / revenue (低=好, 需income配对)
    "buyback_yield",        # abs(commonStockRepurchased) / totalAssets (高=好)
    "fcf_to_income",        # freeCashFlow / netIncome (高=好, 盈利质量)
]

INCOME_FIELDS = [
    "revenue_growth_yoy",   # 同比收入增长 (高=好)
    "gross_margin",         # grossProfit / revenue (高=好)
    "operating_margin",     # operatingIncome / revenue (高=好)
    "net_margin",           # netIncome / revenue (高=好)
    "gross_margin_delta",   # 本期 - 去年同期毛利率 (高=好, 趋势)
    "ebitda_margin",        # ebitda / revenue (高=好)
]

ALL_FMP_FIELDS = RATIO_FIELDS + METRIC_FIELDS + GROWTH_FIELDS


def build_pit_index_statements(stmt_data, use_filing_date=False):
    """为三大报表构建PIT索引。

    Args:
        stmt_data: dict, ticker -> list of quarterly records
        use_filing_date: 若True，用record['filingDate']作为数据可用日(收入报表有filingDate)
                         若False，用date + FILING_DELAY_DAYS (资产负债表/现金流量表无filingDate)

    Returns:
        dict: ticker -> (avail_dates, entries)
    """
    idx = {}
    for ticker, records in stmt_data.items():
        if not records:
            idx[ticker] = ([], [])
            continue
        pairs = []
        for r in records:
            if not isinstance(r, dict) or not r.get("date"):
                continue
            if use_filing_date and r.get("filingDate"):
                avail = r["filingDate"]
            else:
                try:
                    qdate = datetime.strptime(r["date"], "%Y-%m-%d")
                    avail = (qdate + timedelta(days=FILING_DELAY_DAYS)).strftime("%Y-%m-%d")
                except ValueError:
                    continue
            pairs.append((avail, r))
        pairs.sort(key=lambda x: x[0])
        idx[ticker] = ([p[0] for p in pairs], [p[1] for p in pairs])
    return idx


def compute_statement_factors(ticker, date, balance_idx, cashflow_idx, income_idx,
                               cashflow_income_map):
    """计算三大报表衍生因子。返回 dict: factor_name -> float。

    balance_idx/cashflow_idx/income_idx: ticker -> (avail_dates, entries)
    cashflow_income_map: ticker -> {quarter_date: income_record} 用于配对cashflow和income
    """
    factors = {}

    # ── Income statement factors ──
    ad, en = income_idx.get(ticker, ([], []))
    entry = get_pit_from_index(ad, en, date)
    if entry:
        rev = entry.get("revenue")
        gp = entry.get("grossProfit")
        oi = entry.get("operatingIncome")
        ni = entry.get("netIncome")
        ebitda = entry.get("ebitda")
        coR = entry.get("costOfRevenue")

        if rev and rev > 0:
            if gp is not None:
                factors["gross_margin"] = gp / rev
            if oi is not None:
                factors["operating_margin"] = oi / rev
            if ni is not None:
                factors["net_margin"] = ni / rev
            if ebitda is not None:
                factors["ebitda_margin"] = ebitda / rev

        # YoY revenue growth (same quarter last year)
        qdate = entry.get("date", "")
        if qdate and rev and rev > 0:
            yoy_qdate = _prev_year_quarter(qdate)
            yoy_map = cashflow_income_map.get(ticker, {})
            # Try income_idx directly for YoY
            prev_entry = _get_yoy_entry(*income_idx.get(ticker, ([], [])), yoy_qdate=yoy_qdate)
            if prev_entry:
                prev_rev = prev_entry.get("revenue")
                if prev_rev and prev_rev > 0:
                    factors["revenue_growth_yoy"] = (rev - prev_rev) / abs(prev_rev)
                # Gross margin delta
                prev_gp = prev_entry.get("grossProfit")
                if prev_gp is not None and prev_rev and prev_rev > 0 and "gross_margin" in factors:
                    prev_gm = prev_gp / prev_rev
                    factors["gross_margin_delta"] = factors["gross_margin"] - prev_gm

    # ── Balance sheet factors ──
    ad_b, en_b = balance_idx.get(ticker, ([], []))
    entry_b = get_pit_from_index(ad_b, en_b, date)
    if entry_b:
        td = entry_b.get("totalDebt")
        te = entry_b.get("totalStockholdersEquity")
        ta = entry_b.get("totalAssets")
        cash = entry_b.get("cashAndCashEquivalents")
        nd = entry_b.get("netDebt")

        if te and te > 0 and td is not None:
            factors["debt_to_equity"] = td / te
        if ta and ta > 0:
            if cash is not None:
                factors["cash_to_assets"] = cash / ta
            if nd is not None:
                factors["net_debt_to_assets"] = nd / ta
            if te is not None:
                factors["equity_ratio"] = te / ta

    # ── Cashflow factors (needs income for revenue pairing) ──
    ad_c, en_c = cashflow_idx.get(ticker, ([], []))
    entry_c = get_pit_from_index(ad_c, en_c, date)
    if entry_c:
        ocf = entry_c.get("operatingCashFlow")
        capex = entry_c.get("capitalExpenditure") or 0
        fcf = entry_c.get("freeCashFlow")
        divs = entry_c.get("dividendsPaid") or 0
        buyback = entry_c.get("commonStockRepurchased") or 0

        # Pair with income statement for margin calculations
        cf_qdate = entry_c.get("date", "")
        paired_rev = None
        paired_ni = None
        if cf_qdate:
            # Find income record with same or closest quarter date
            ad_i, en_i = income_idx.get(ticker, ([], []))
            paired_entry = _get_paired_income(ad_i, en_i, cf_qdate)
            if paired_entry:
                paired_rev = paired_entry.get("revenue")
                paired_ni = paired_entry.get("netIncome")

        if paired_rev and paired_rev > 0:
            if fcf is not None:
                factors["fcf_margin"] = fcf / paired_rev
            if ocf is not None:
                factors["ocf_margin"] = ocf / paired_rev
            factors["capex_intensity"] = abs(capex) / paired_rev

        if paired_ni and paired_ni > 0 and fcf is not None:
            factors["fcf_to_income"] = fcf / paired_ni

        # Buyback yield (relative to total assets)
        ta_b = entry_b.get("totalAssets") if entry_b else None
        if ta_b and ta_b > 0 and buyback:
            factors["buyback_yield"] = abs(buyback) / ta_b

    return factors


def _prev_year_quarter(qdate):
    """从 '2024-06-30' 得到 '2023-06-30'。"""
    try:
        year = int(qdate[:4])
        return f"{year - 1}{qdate[4:]}"
    except:
        return ""


def _get_yoy_entry(avail_dates, entries, yoy_qdate):
    """找到去年同期的entry (按quarter date匹配)。"""
    if not avail_dates or not yoy_qdate:
        return None
    # 遍历entries找date匹配yoy_qdate的
    for e in entries:
        if e.get("date", "") == yoy_qdate:
            return e
    # fallback: 找最接近的
    best = None
    best_gap = 999
    for e in entries:
        ed = e.get("date", "")
        if not ed:
            continue
        try:
            gap = abs((datetime.strptime(ed, "%Y-%m-%d") - datetime.strptime(yoy_qdate, "%Y-%m-%d")).days)
            if gap < best_gap:
                best_gap = gap
                best = e
        except:
            continue
    return best if best_gap < 45 else None  # 同季度gap应<45天


def _get_paired_income(avail_dates, entries, cf_qdate):
    """找到与cashflow record同季度的income record。"""
    if not avail_dates or not cf_qdate:
        return None
    # 直接匹配quarter date
    for e in entries:
        if e.get("date", "") == cf_qdate:
            return e
    # fallback: 找最接近的
    best = None
    best_gap = 999
    for e in entries:
        ed = e.get("date", "")
        if not ed:
            continue
        try:
            gap = abs((datetime.strptime(ed, "%Y-%m-%d") - datetime.strptime(cf_qdate, "%Y-%m-%d")).days)
            if gap < best_gap:
                best_gap = gap
                best = e
        except:
            continue
    return best if best_gap < 45 else None


# FMP PIT延迟修正: FMP date = 财报季末日, 实际数据在SEC filing后才可用
# 美股10-Q/10-K平均发布延迟: 大公司~33天, 小公司~45天
FILING_DELAY_DAYS = 33


# ═══════════════════════════════════════════════════
# 高性能PIT查找 (bisect二分, O(log n))
# ═══════════════════════════════════════════════════

def build_pit_index(quarterly_data):
    """预计算avail_date并排序，供bisect查找。
    返回: (avail_dates: list[str], entries: list[dict]) — 按avail_date升序排列
    """
    if not quarterly_data:
        return ([], [])
    pairs = []
    for q in quarterly_data:
        if not isinstance(q, dict) or not q.get("date"):
            continue
        try:
            qdate = datetime.strptime(q["date"], "%Y-%m-%d")
            avail = (qdate + timedelta(days=FILING_DELAY_DAYS)).strftime("%Y-%m-%d")
        except ValueError:
            continue
        pairs.append((avail, q))
    pairs.sort(key=lambda x: x[0])
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def get_pit_from_index(avail_dates, entries, date):
    """O(log n) PIT查找: 返回date之前已发布的最新条目。"""
    if not avail_dates:
        return {}
    idx = bisect_right(avail_dates, date) - 1
    if idx < 0:
        return {}
    return entries[idx]


def build_insider_index(insider_data):
    """预排序insider数据供bisect查找。返回 (dates, entries)。"""
    if not insider_data:
        return ([], [])
    pairs = [(t.get("date", ""), t) for t in insider_data if t.get("date")]
    pairs.sort(key=lambda x: x[0])
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def get_pit(quarterly_data, date):
    """Point-in-time: 返回date之前已发布(available)的最新季度数据。

    关键修正(2026-06-29): FMP的date字段是财报期结束日(季末), 不是发布日。
    数据在季末+33天后才真正可用。因此:
    - query date = "2024-07-15" 时, 只能用 avail_date <= "2024-07-15" 的数据
    - 季末日="2024-06-29" 的数据, avail_date = "2024-08-01"(6/29+33天)
    - 所以 2024-07-15 时该数据不可用, 只能用上一季度(2024-03-30+33=2024-05-02)
    """
    from datetime import datetime, timedelta
    if not quarterly_data:
        return {}
    latest = {}
    for q in quarterly_data:
        if not isinstance(q, dict) or not q.get("date"):
            continue
        # 计算数据实际可用日 = 季末日 + FILING_DELAY_DAYS
        try:
            qdate = datetime.strptime(q["date"], "%Y-%m-%d")
            avail = (qdate + timedelta(days=FILING_DELAY_DAYS)).strftime("%Y-%m-%d")
        except ValueError:
            continue
        if avail <= date:
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
                          growth_hist, insider_hist, dcf_data, pt_data,
                          earnings_hist=None, grades_hist=None):
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
        
        # ── Earnings rank (FMP Premium) ──
        if earnings_hist:
            from extract_fmp_premium_features import extract_earnings_features
            for f in EARNINGS_FIELDS:
                vals = {}
                for t in day["ticker"].values:
                    er = extract_earnings_features(earnings_hist.get(t, []), date)
                    v = er.get(f)
                    if v is not None:
                        vals[t] = v
                if len(vals) > 5:
                    row[f"e_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── Grade Sentiment rank (FMP Premium) ──
        if grades_hist:
            from extract_fmp_premium_features import extract_grade_features
            for f in GRADE_FIELDS:
                vals = {}
                for t in day["ticker"].values:
                    gr = extract_grade_features(grades_hist.get(t, []), date)
                    v = gr.get(f)
                    if v is not None:
                        vals[t] = v
                if len(vals) > 5:
                    row[f"s_{f}"] = pd.Series(vals).rank(pct=True)
        
        # ── 分组得分 ──
        r_cols = [c for c in row.columns if c.startswith("r_")]
        m_cols = [c for c in row.columns if c.startswith("m_")]
        g_cols = [c for c in row.columns if c.startswith("g_")]
        a_cols = [c for c in row.columns if c.startswith("a_")]
        i_cols = [c for c in row.columns if c.startswith("i_")]
        d_cols = [c for c in row.columns if c.startswith("d_")]
        e_cols = [c for c in row.columns if c.startswith("e_")]
        s_cols = [c for c in row.columns if c.startswith("s_")]
        
        row["fund_ratio"] = row[r_cols].mean(axis=1) if r_cols else 0.5
        row["fund_metric"] = row[m_cols].mean(axis=1) if m_cols else 0.5
        row["fund_growth"] = row[g_cols].mean(axis=1) if g_cols else 0.5
        row["analyst"] = row[a_cols].mean(axis=1) if a_cols else 0.5
        row["insider"] = row[i_cols].mean(axis=1) if i_cols else 0.5
        row["valuation"] = row[d_cols].mean(axis=1) if d_cols else 0.5
        row["earnings"] = row[e_cols].mean(axis=1) if e_cols else 0.5
        row["grade_sentiment"] = row[s_cols].mean(axis=1) if s_cols else 0.5
        
        ranks_dict[date] = row.set_index("ticker")[[
            "tech", "fund_ratio", "fund_metric", "fund_growth",
            "analyst", "insider", "valuation", "earnings", "grade_sentiment"
        ]]
    
    print(f"✅ 全量PIT rank: {len(ranks_dict)} 天, 9个因子组")
    return ranks_dict


def precompute_pit_ranks_fast(master, fmp_hist, ana_hist, metrics_hist,
                               growth_hist, insider_hist, dcf_data, pt_data,
                               earnings_hist=None, grades_hist=None):
    """高性能版PIT rank: 用bisect替代线性扫描，10-20x加速。
    
    用法与precompute_pit_ranks完全一致，结果也完全一致。
    """
    print("📊 预计算全量PIT rank (bisect加速版)...")
    
    # ── 预建索引 (一次性O(n log n)) ──
    fmp_idx = {}   # ticker -> (avail_dates, entries)
    ana_idx = {}
    met_idx = {}
    grw_idx = {}
    ins_idx = {}
    
    all_tickers = set(master["ticker"].unique())
    for t in all_tickers:
        fmp_idx[t] = build_pit_index(fmp_hist.get(t, []))
        ana_idx[t] = build_pit_index(ana_hist.get(t, []))
        met_idx[t] = build_pit_index(metrics_hist.get(t, []))
        grw_idx[t] = build_pit_index(growth_hist.get(t, []))
        ins_idx[t] = build_insider_index(insider_hist.get(t, []))
    
    # earnings/grades索引
    earn_idx = {}
    grade_idx = {}
    if earnings_hist:
        from extract_fmp_premium_features import load_fmp_premium_earnings
        for t in all_tickers:
            earn_idx[t] = build_pit_index(earnings_hist.get(t, []))
    if grades_hist:
        for t in all_tickers:
            # grades不需要PIT延迟(90天窗口筛选), 但排序仍有助bisect
            records = grades_hist.get(t, [])
            if records:
                pairs = [(r.get("date", ""), r) for r in records if r.get("date")]
                pairs.sort(key=lambda x: x[0])
                grade_idx[t] = ([p[0] for p in pairs], [p[1] for p in pairs])
            else:
                grade_idx[t] = ([], [])
    
    print(f"  ✅ 索引建好: {len(all_tickers)}只, {time.time():.0f}")
    
    # ── 逐日rank (用bisect查找) ──
    dates = sorted(master["date"].unique())
    ranks_dict = {}
    
    for di, date in enumerate(dates):
        day = master[master["date"] == date]
        if len(day) < 10:
            continue
        tickers = day["ticker"].values
        day_data = day.set_index("ticker")
        row = day_data[[]].copy()
        
        # Tech rank (K线) — 直接从master取
        tech_r = []
        for f in TECH_FIELDS:
            if f in day_data.columns and day_data[f].notna().sum() > 5:
                row[f"t_{f}"] = day_data[f].rank(pct=True)
                tech_r.append(f"t_{f}")
        row["tech"] = row[tech_r].mean(axis=1) if tech_r else 0.5
        
        # FMP Ratios rank — bisect查找
        for f in RATIO_FIELDS:
            vals = {}
            for t in tickers:
                ad, en = fmp_idx.get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"r_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Key Metrics rank — bisect查找
        for f in METRIC_FIELDS:
            vals = {}
            for t in tickers:
                ad, en = met_idx.get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"m_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Financial Growth rank — bisect查找
        for f in GROWTH_FIELDS:
            vals = {}
            for t in tickers:
                ad, en = grw_idx.get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"g_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Analyst rank — bisect查找
        for f in ANALYST_FIELDS:
            vals = {}
            for t in tickers:
                ad, en = ana_idx.get(t, ([], []))
                pit = get_pit_from_index(ad, en, date)
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 5:
                row[f"a_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Insider rank — bisect查找
        insider_fields = ["insider_net_shares", "insider_net_count", "insider_ceo_buy"]
        for f in insider_fields:
            vals = {}
            for t in tickers:
                ad, en = ins_idx.get(t, ([], []))
                # insider用90天窗口, 需要特殊处理
                pit = _get_insider_from_index_fast(ad, en, date)
                v = pit.get(f)
                if v is not None and v != 0:
                    vals[t] = v
            if len(vals) > 5:
                row[f"i_{f}"] = pd.Series(vals).rank(pct=True)
        
        # DCF/PT rank (不变, 本身就是O(1))
        for f in ["dcf_upside", "pt_upside"]:
            vals = {}
            for t in tickers:
                pit = get_pit_dcf(dcf_data.get(t, {}), pt_data.get(t, {}))
                v = pit.get(f)
                if v is not None:
                    vals[t] = v
            if len(vals) > 10:
                row[f"d_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Earnings rank (FMP Premium) — bisect查找
        if earnings_hist:
            from extract_fmp_premium_features import extract_earnings_features
            for f in EARNINGS_FIELDS:
                vals = {}
                for t in tickers:
                    ad, en = earn_idx.get(t, ([], []))
                    # earnings用PIT延迟, 但特征提取需要多条记录
                    # 用线性扫描取date前的记录 (只做一次筛选)
                    available = [e for a, e in zip(ad, en) if a <= date]
                    if available:
                        er = _extract_earnings_from_available(available, f)
                        if er is not None:
                            vals[t] = er
                if len(vals) > 5:
                    row[f"e_{f}"] = pd.Series(vals).rank(pct=True)
        
        # Grade Sentiment rank (FMP Premium) — 线性扫描(90天窗口)
        if grades_hist:
            for f in GRADE_FIELDS:
                vals = {}
                for t in tickers:
                    ad, en = grade_idx.get(t, ([], []))
                    # grades用90天窗口, 筛选cutoff <= date <= record_date
                    from datetime import datetime as dt2, timedelta as td2
                    try:
                        dt_obj = dt2.strptime(date, "%Y-%m-%d")
                        cutoff = (dt_obj - td2(days=90)).strftime("%Y-%m-%d")
                    except:
                        continue
                    recent = [e for a, e in zip(ad, en) if cutoff <= a <= date]
                    if recent:
                        from extract_fmp_premium_features import _grade_snapshot_features, _grade_snapshot_features_trend
                        if len(recent) >= 2:
                            gr = _grade_snapshot_features_trend(recent)
                        else:
                            gr = _grade_snapshot_features(recent[0])
                        v = gr.get(f)
                        if v is not None:
                            vals[t] = v
                if len(vals) > 5:
                    row[f"s_{f}"] = pd.Series(vals).rank(pct=True)
        
        # 分组得分
        r_cols = [c for c in row.columns if c.startswith("r_")]
        m_cols = [c for c in row.columns if c.startswith("m_")]
        g_cols = [c for c in row.columns if c.startswith("g_")]
        a_cols = [c for c in row.columns if c.startswith("a_")]
        i_cols = [c for c in row.columns if c.startswith("i_")]
        d_cols = [c for c in row.columns if c.startswith("d_")]
        e_cols = [c for c in row.columns if c.startswith("e_")]
        s_cols = [c for c in row.columns if c.startswith("s_")]
        
        row["fund_ratio"] = row[r_cols].mean(axis=1) if r_cols else 0.5
        row["fund_metric"] = row[m_cols].mean(axis=1) if m_cols else 0.5
        row["fund_growth"] = row[g_cols].mean(axis=1) if g_cols else 0.5
        row["analyst"] = row[a_cols].mean(axis=1) if a_cols else 0.5
        row["insider"] = row[i_cols].mean(axis=1) if i_cols else 0.5
        row["valuation"] = row[d_cols].mean(axis=1) if d_cols else 0.5
        row["earnings"] = row[e_cols].mean(axis=1) if e_cols else 0.5
        row["grade_sentiment"] = row[s_cols].mean(axis=1) if s_cols else 0.5
        
        ranks_dict[date] = row[[ 
            "tech", "fund_ratio", "fund_metric", "fund_growth",
            "analyst", "insider", "valuation", "earnings", "grade_sentiment"
        ]]
        
        # 进度打印 (每500天)
        if (di + 1) % 500 == 0:
            print(f"  📊 {di+1}/{len(dates)} 天...")
    
    print(f"✅ 全量PIT rank (bisect): {len(ranks_dict)} 天, 9个因子组")
    return ranks_dict


def _get_insider_from_index_fast(dates, entries, date):
    """insider用90天窗口的bisect版本。"""
    if not dates:
        return {}
    from datetime import datetime as dt2, timedelta as td2
    try:
        dt_obj = dt2.strptime(date, "%Y-%m-%d")
        start = (dt_obj - td2(days=90)).strftime("%Y-%m-%d")
    except:
        return {}
    
    # bisect找到start的位置
    idx_start = bisect_right(dates, start) - 1
    if idx_start < 0:
        idx_start = 0
    
    net_buy_shares = 0
    n_buy = 0
    n_sell = 0
    ceo_buy = 0
    
    for i in range(idx_start, len(dates)):
        if dates[i] > date:
            break
        t = entries[i]
        acq = t.get("acq_disp", "")
        shares = t.get("shares", 0) or 0
        owner = t.get("owner", "").lower()
        if acq == "A":
            net_buy_shares += shares
            n_buy += 1
            if "ceo" in owner or "chief executive" in owner:
                ceo_buy += shares
        elif acq == "D":
            net_buy_shares -= shares
            n_sell += 1
    
    return {
        "insider_net_shares": net_buy_shares,
        "insider_net_count": n_buy - n_sell,
        "insider_ceo_buy": ceo_buy,
    }


def _extract_earnings_from_available(available, field):
    """从已筛选的available earnings记录中提取单个字段。"""
    if not available:
        return None
    # available按avail_date升序, 取最新的
    latest = available[-1]
    eps_actual = latest.get("epsActual")
    eps_est = latest.get("epsEstimated")
    rev_actual = latest.get("revenueActual")
    rev_est = latest.get("revenueEstimated")
    
    if field == "earnings_surprise":
        if eps_actual is not None and eps_est is not None and eps_est != 0:
            return (eps_actual - eps_est) / abs(eps_est)
    elif field == "earnings_surprise_2q":
        surprises = []
        for r in available[-2:]:
            ea = r.get("epsActual")
            ee = r.get("epsEstimated")
            if ea is not None and ee is not None and ee != 0:
                surprises.append((ea - ee) / abs(ee))
        if surprises:
            return sum(surprises) / len(surprises)
    elif field == "earnings_beat_count_4q":
        beats = 0
        total = 0
        for r in available[-4:]:
            ea = r.get("epsActual")
            ee = r.get("epsEstimated")
            if ea is not None and ee is not None:
                total += 1
                if ea > ee:
                    beats += 1
        if total > 0:
            return beats / total
    elif field == "earnings_price_reaction":
        if rev_actual is not None and rev_est is not None and rev_est != 0:
            return (rev_actual - rev_est) / abs(rev_est)
    return None


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
