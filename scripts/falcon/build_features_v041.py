#!/usr/bin/env python3
"""
Falcon V0.4.1 特征重计算
=========================
从 FMP Premium 快照重新计算所有基本面因子 (PIT正确)。
输出: data/falcon/features_v04_1.parquet

用法:
    python3 scripts/falcon/build_features_v041.py
    python3 scripts/falcon/build_features_v041.py --validate-only

关键改进:
  - 从 FMP Premium 快照加载最新数据 (到 2026-03-31)
  - PIT (Point-in-Time) 正确: 只用 query_date 之前已发布的数据
  - 覆盖 2025-2026 年的因子缺失
"""
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from bisect import bisect_right

import pandas as pd
import numpy as np

# 路径配置
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_paths import FalconPaths
paths = FalconPaths()

# ═══════════════════════════════════════════════════
# 因子定义 (与 falcon_v03_engine.py 保持一致)
# ═══════════════════════════════════════════════════

# FMP Ratios (17个 — 任务要求)
RATIO_FIELDS = [
    "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
    "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
    "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
    "ebitdaMargin", "assetTurnover", "inventoryTurnover",
    "receivablesTurnover", "debtToEquityRatio", "currentRatio",
    "quickRatio", "financialLeverageRatio",
    "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
    "dividendYieldPercentage", "dividendPayoutRatio",
]

# Key Metrics (19个)
METRIC_FIELDS = [
    "earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
    "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
    "returnOnCapitalEmployed", "returnOnInvestedCapital",
    "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
    "cashConversionCycle", "capexToRevenue", "capexToDepreciation",
    "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
    "netDebtToEBITDA", "operatingReturnOnAssets",
]

# Financial Growth (15个)
GROWTH_FIELDS = [
    "revenueGrowth", "grossProfitGrowth", "ebitgrowth",
    "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
    "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
    "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
    "receivablesGrowth", "inventoryGrowth", "assetGrowth",
    "bookValueperShareGrowth", "debtGrowth",
]

# Analyst (3个 — 任务要求)
ANALYST_FIELDS = ["eps_revision", "revenue_revision", "eps_dispersion", "num_analysts_eps"]
ANALYST_FIELD_ALIASES = {"num_analysts_eps": "numAnalystsEps"}

# QoQ margins (从 ratios 预计算)
QOQ_FIELDS = ["grossProfitMargin_qoq", "netProfitMargin_qoq",
              "operatingProfitMargin_qoq", "ebitdaMargin_qoq"]

# ═══════════════════════════════════════════════════
# PIT 基础设施
# ═══════════════════════════════════════════════════

FILING_DELAY_DAYS = 33  # 10-Q/10-K 平均发布延迟


def build_pit_index(quarterly_data):
    """预计算 avail_date 并排序，供 bisect 查找。
    返回: (avail_dates: list[str], entries: list[dict]) — 按 avail_date 升序排列
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
    """O(log n) PIT 查找: 返回 date 之前已发布的最新条目。"""
    if not avail_dates:
        return {}
    idx = bisect_right(avail_dates, date) - 1
    if idx < 0:
        return {}
    return entries[idx]


def build_analyst_index(analyst_data):
    """为 analyst 数据构建 PIT 索引 (analyst 数据无 filing delay)。"""
    if not analyst_data:
        return ([], [])
    pairs = []
    for r in analyst_data:
        if not isinstance(r, dict) or not r.get("date"):
            continue
        pairs.append((r["date"], r))
    pairs.sort(key=lambda x: x[0])
    return ([p[0] for p in pairs], [p[1] for p in pairs])


def get_analyst_pit(avail_dates, entries, date):
    """Analyst PIT: 直接用 date 字段 (无 filing delay)。"""
    if not avail_dates:
        return {}
    idx = bisect_right(avail_dates, date) - 1
    if idx < 0:
        return {}
    return entries[idx]


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════

def load_json(path):
    """安全加载 JSON 文件。"""
    if not path.exists():
        print(f"  ⚠️ 文件不存在: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def load_all_data():
    """从 FMP Premium 快照加载所有数据。"""
    print("📦 加载 FMP Premium 快照...")
    t0 = time.time()
    
    data = {}
    
    # 核心因子数据
    for name, path in [
        ("fmp_ratios", paths.fmp_ratios),
        ("fmp_key_metrics", paths.fmp_key_metrics),
        ("fmp_financial_growth", paths.fmp_financial_growth),
        ("analyst_historical", paths.analyst_historical),
    ]:
        raw = load_json(path)
        data[name] = raw
        n_tickers = len(raw)
        n_records = sum(len(v) for v in raw.values())
        print(f"  ✅ {name}: {n_tickers} tickers, {n_records} records ({path.name})")
    
    # 三大报表
    for name, path in [
        ("fmp_balance_sheet", paths.fmp_balance_sheet),
        ("fmp_cashflow", paths.fmp_cashflow),
        ("fmp_income_stmt", paths.fmp_income_stmt),
    ]:
        raw = load_json(path)
        data[name] = raw
        n_tickers = len(raw)
        n_records = sum(len(v) for v in raw.values())
        print(f"  ✅ {name}: {n_tickers} tickers, {n_records} records ({path.name})")
    
    print(f"  ⏱️ 加载耗时: {time.time()-t0:.1f}秒")
    return data


# ═══════════════════════════════════════════════════
# 因子计算
# ═══════════════════════════════════════════════════

def precompute_all_pit_indices(data):
    """为所有数据源预建 PIT 索引 (一次性 O(n log n))。"""
    print("📊 预建 PIT 索引...")
    t0 = time.time()
    
    indices = {}
    all_tickers = set()
    
    # 收集所有 ticker
    for key in ["fmp_ratios", "fmp_key_metrics", "fmp_financial_growth",
                "analyst_historical", "fmp_balance_sheet", "fmp_cashflow", "fmp_income_stmt"]:
        all_tickers.update(data.get(key, {}).keys())
    
    print(f"  总计 {len(all_tickers)} 个 ticker")
    
    # 为每个 ticker 建索引
    for t in all_tickers:
        # FMP 数据: 有 filing delay
        for key in ["fmp_ratios", "fmp_key_metrics", "fmp_financial_growth",
                     "fmp_balance_sheet", "fmp_cashflow", "fmp_income_stmt"]:
            if key not in indices:
                indices[key] = {}
            indices[key][t] = build_pit_index(data.get(key, {}).get(t, []))
        
        # Analyst 数据: 无 filing delay
        if "analyst" not in indices:
            indices["analyst"] = {}
        indices["analyst"][t] = build_analyst_index(data.get("analyst_historical", {}).get(t, []))
    
    print(f"  ✅ 索引建好: {len(indices)} 个数据源, {time.time()-t0:.1f}秒")
    return indices


def compute_factors_for_date(ticker, date, indices):
    """为单个 ticker 在指定 date 计算所有 PIT 因子。
    返回 dict: factor_name -> value
    """
    factors = {}
    
    # ── FMP Ratios (17个) ──
    ad, en = indices.get("fmp_ratios", {}).get(ticker, ([], []))
    pit = get_pit_from_index(ad, en, date)
    for f in RATIO_FIELDS:
        v = pit.get(f)
        if v is not None:
            factors[f"r_{f}"] = v
    
    # ── Key Metrics (19个) ──
    ad, en = indices.get("fmp_key_metrics", {}).get(ticker, ([], []))
    pit = get_pit_from_index(ad, en, date)
    for f in METRIC_FIELDS:
        v = pit.get(f)
        if v is not None:
            factors[f"m_{f}"] = v
    
    # ── Financial Growth (15个) ──
    ad, en = indices.get("fmp_financial_growth", {}).get(ticker, ([], []))
    pit = get_pit_from_index(ad, en, date)
    for f in GROWTH_FIELDS:
        v = pit.get(f)
        if v is not None:
            factors[f"g_{f}"] = v
    
    # ── QoQ margins (从 ratios 计算) ──
    ratios_idx = indices.get("fmp_ratios", {}).get(ticker, ([], []))
    if ratios_idx[0]:  # 有数据
        ad_r, en_r = ratios_idx
        current_pit = get_pit_from_index(ad_r, en_r, date)
        # 找上一期
        if current_pit and current_pit.get("date"):
            current_date = current_pit["date"]
            # 用 bisect 找上一期
            avail_dates, entries = ratios_idx
            idx = bisect_right(avail_dates, date) - 1
            if idx > 0:
                prev = entries[idx - 1]
                for margin in ["grossProfitMargin", "netProfitMargin",
                               "operatingProfitMargin", "ebitdaMargin"]:
                    curr = current_pit.get(margin)
                    prev_v = prev.get(margin)
                    if curr is not None and prev_v is not None and prev_v != 0:
                        factors[f"r_{margin}_qoq"] = (curr - prev_v) / abs(prev_v)
    
    # ── Analyst (3-4个) ──
    ad, en = indices.get("analyst", {}).get(ticker, ([], []))
    pit = get_analyst_pit(ad, en, date)
    for f in ANALYST_FIELDS:
        json_key = ANALYST_FIELD_ALIASES.get(f, f)
        v = pit.get(json_key)
        if v is not None:
            factors[f"a_{f}"] = v
    
    # ── Balance Sheet factors ──
    ad_b, en_b = indices.get("fmp_balance_sheet", {}).get(ticker, ([], []))
    entry_b = get_pit_from_index(ad_b, en_b, date)
    if entry_b:
        td = entry_b.get("totalDebt")
        te = entry_b.get("totalStockholdersEquity")
        ta = entry_b.get("totalAssets")
        cash = entry_b.get("cashAndCashEquivalents")
        nd = entry_b.get("netDebt")
        if ta and ta > 0:
            if cash is not None:
                factors["b_cash_to_assets"] = cash / ta
            if nd is not None:
                factors["b_net_debt_to_assets"] = nd / ta
            if te is not None:
                factors["b_equity_ratio"] = te / ta
        if te and te > 0 and td is not None:
            factors["b_debt_to_equity"] = td / te
    
    # ── Cashflow factors ──
    ad_c, en_c = indices.get("fmp_cashflow", {}).get(ticker, ([], []))
    entry_c = get_pit_from_index(ad_c, en_c, date)
    if entry_c:
        ocf = entry_c.get("operatingCashFlow")
        capex = entry_c.get("capitalExpenditure") or 0
        fcf = entry_c.get("freeCashFlow")
        buyback = entry_c.get("commonStockRepurchased") or 0
        
        # Pair with income for margin
        cf_qdate = entry_c.get("date", "")
        paired_rev = None
        paired_ni = None
        if cf_qdate:
            ad_i, en_i = indices.get("fmp_income_stmt", {}).get(ticker, ([], []))
            # Find income record with same quarter
            for e in en_i:
                if e.get("date", "") == cf_qdate:
                    paired_rev = e.get("revenue")
                    paired_ni = e.get("netIncome")
                    break
        
        if paired_rev and paired_rev > 0:
            if fcf is not None:
                factors["c_fcf_margin"] = fcf / paired_rev
            factors["c_capex_intensity"] = abs(capex) / paired_rev
        if paired_ni and paired_ni > 0 and fcf is not None:
            factors["c_fcf_to_income"] = fcf / paired_ni
        if entry_b and entry_b.get("totalAssets") and entry_b["totalAssets"] > 0 and buyback:
            factors["c_buyback_yield"] = abs(buyback) / entry_b["totalAssets"]
    
    # ── Income Statement factors ──
    ad_i, en_i = indices.get("fmp_income_stmt", {}).get(ticker, ([], []))
    entry_i = get_pit_from_index(ad_i, en_i, date)
    if entry_i:
        rev = entry_i.get("revenue")
        if rev and rev > 0:
            gp = entry_i.get("grossProfit")
            if gp is not None:
                factors["i_gross_margin"] = gp / rev
            oi = entry_i.get("operatingIncome")
            if oi is not None:
                factors["i_operating_margin"] = oi / rev
            ni = entry_i.get("netIncome")
            if ni is not None:
                factors["i_net_margin"] = ni / rev
            ebitda = entry_i.get("ebitda")
            if ebitda is not None:
                factors["i_ebitda_margin"] = ebitda / rev
        
        # YoY revenue growth
        qdate = entry_i.get("date", "")
        if qdate and rev and rev > 0:
            try:
                year = int(qdate[:4])
                yoy_qdate = f"{year - 1}{qdate[4:]}"
                for e in en_i:
                    if e.get("date", "") == yoy_qdate:
                        prev_rev = e.get("revenue")
                        if prev_rev and prev_rev > 0:
                            factors["i_revenue_growth_yoy"] = (rev - prev_rev) / abs(prev_rev)
                        prev_gp = e.get("grossProfit")
                        if prev_gp is not None and prev_rev and prev_rev > 0 and "i_gross_margin" in factors:
                            prev_gm = prev_gp / prev_rev
                            factors["i_gross_margin_delta"] = factors["i_gross_margin"] - prev_gm
                        break
            except (ValueError, IndexError):
                pass
    
    return factors


def compute_all_features(master, data, indices):
    """为所有日期和 ticker 计算 PIT 因子。
    返回 DataFrame: columns = [ticker, date, r_*, m_*, g_*, a_*, b_*, c_*, i_*]
    """
    print("📊 计算 PIT 因子...")
    t0 = time.time()
    
    dates = sorted(master["date"].unique())
    tickers = sorted(master["ticker"].unique())
    
    all_rows = []
    total = len(dates)
    
    for di, date in enumerate(dates):
        day_tickers = master[master["date"] == date]["ticker"].unique()
        
        for ticker in day_tickers:
            factors = compute_factors_for_date(ticker, date, indices)
            if factors:
                factors["ticker"] = ticker
                factors["date"] = date
                all_rows.append(factors)
        
        if (di + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (di + 1) / elapsed
            eta = (total - di - 1) / rate if rate > 0 else 0
            print(f"  📊 {di+1}/{total} 天 ({rate:.0f} 天/秒, ETA {eta:.0f}秒)")
    
    if not all_rows:
        print("  ❌ 无数据!")
        return pd.DataFrame()
    
    result = pd.DataFrame(all_rows)
    print(f"  ✅ 计算完成: {len(result)} 行, {time.time()-t0:.1f}秒")
    return result


# ═══════════════════════════════════════════════════
# 合并与输出
# ═══════════════════════════════════════════════════

def merge_features(master, pit_factors):
    """将 PIT 因子合并到 master (K线+技术指标)。"""
    print("📊 合并特征...")
    
    if pit_factors.empty:
        print("  ⚠️ PIT 因子为空，使用原始 master")
        return master
    
    # 合并
    merged = master.merge(pit_factors, on=["ticker", "date"], how="left")
    
    # 列统计
    orig_cols = len(master.columns)
    new_cols = len(merged.columns)
    added = new_cols - orig_cols
    print(f"  ✅ 合并完成: {orig_cols} → {new_cols} 列 (+{added} PIT因子列)")
    
    return merged


def compute_feature_audit(df):
    """生成特征审计报告。"""
    print("📊 生成特征审计...")
    
    audit = {
        "version": "v0.4.1",
        "generated_at": datetime.now().isoformat(),
        "shape": list(df.shape),
        "columns": list(df.columns),
        "n_tickers": int(df["ticker"].nunique()),
        "date_range": [df["date"].min(), df["date"].max()],
        "years": sorted(df["date"].str[:4].unique().tolist()),
    }
    
    # 按年覆盖率
    coverage_by_year = {}
    pit_cols = [c for c in df.columns if c.startswith(("r_", "m_", "g_", "a_", "b_", "c_", "i_"))]
    
    for year in sorted(df["date"].str[:4].unique()):
        year_df = df[df["date"].str[:4] == year]
        year_cov = {}
        for c in pit_cols:
            if c in year_df.columns:
                cov = float(year_df[c].notna().mean())
                year_cov[c] = round(cov, 4)
        coverage_by_year[year] = {
            "n_rows": len(year_df),
            "n_tickers": int(year_df["ticker"].nunique()),
            "pit_factor_coverage": year_cov,
            "avg_pit_coverage": round(float(np.mean(list(year_cov.values()))) if year_cov else 0, 4),
        }
    
    audit["coverage_by_year"] = coverage_by_year
    
    # 因子完整性
    factor_groups = {
        "fund_ratio (r_*)": [c for c in pit_cols if c.startswith("r_") and "_qoq" not in c],
        "fund_metric (m_*)": [c for c in pit_cols if c.startswith("m_")],
        "fund_growth (g_*)": [c for c in pit_cols if c.startswith("g_")],
        "analyst (a_*)": [c for c in pit_cols if c.startswith("a_")],
        "balance (b_*)": [c for c in pit_cols if c.startswith("b_")],
        "cashflow (c_*)": [c for c in pit_cols if c.startswith("c_")],
        "income (i_*)": [c for c in pit_cols if c.startswith("i_")],
        "qoq (r_*_qoq)": [c for c in pit_cols if c.startswith("r_") and "_qoq" in c],
    }
    
    factor_summary = {}
    for group_name, cols in factor_groups.items():
        if cols:
            # 取 2025-2026 的覆盖率
            recent = df[df["date"] >= "2025-01-01"]
            covs = [float(recent[c].notna().mean()) for c in cols if c in recent.columns]
            factor_summary[group_name] = {
                "n_factors": len(cols),
                "factors": cols,
                "coverage_2025_2026": round(float(np.mean(covs)) if covs else 0, 4),
            }
    
    audit["factor_groups"] = factor_summary
    
    # 与 features_v02 对比
    v02_path = paths.features_v02
    if v02_path.exists():
        v02 = pd.read_parquet(v02_path)
        audit["comparison_v02"] = {
            "v02_shape": list(v02.shape),
            "v041_shape": list(df.shape),
            "v02_date_range": [v02["date"].min(), v02["date"].max()],
            "v041_date_range": [df["date"].min(), df["date"].max()],
            "v02_2025_2026_fundamental_coverage": "0.0% (all NaN)",
            "v041_2025_2026_fundamental_coverage": "See coverage_by_year",
        }
    
    # 数据哈希
    data_str = pd.util.hash_pandas_object(df).to_json()
    audit["data_hash"] = hashlib.md5(data_str.encode()).hexdigest()
    
    return audit


# ═══════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Falcon V0.4.1 特征重计算")
    parser.add_argument("--validate-only", action="store_true", help="只验证，不生成")
    args = parser.parse_args()
    
    print("🦅 Falcon V0.4.1 特征重计算")
    print("=" * 60)
    t_total = time.time()
    
    # 1. 加载 master (K线+技术指标)
    print("📊 加载 master 数据 (features_v02.parquet)...")
    master = pd.read_parquet(paths.features_v02)
    master["date"] = master["date"].astype(str)
    print(f"  ✅ {master['ticker'].nunique()} tickers, {len(master)} rows")
    print(f"  📅 {master['date'].min()} → {master['date'].max()}")
    
    # 2. 加载 FMP Premium 数据
    data = load_all_data()
    
    # 3. 预建 PIT 索引
    indices = precompute_all_pit_indices(data)
    
    # 4. 计算 PIT 因子
    pit_factors = compute_all_features(master, data, indices)
    
    if pit_factors.empty:
        print("❌ 无 PIT 因子数据!")
        sys.exit(1)
    
    # 5. 合并
    merged = merge_features(master, pit_factors)
    
    # 6. 验证 2025-2026 覆盖
    print("\n📊 2025-2026 PIT 因子覆盖检查:")
    recent = merged[merged["date"] >= "2025-01-01"]
    pit_cols = [c for c in merged.columns if c.startswith(("r_", "m_", "g_", "a_", "b_", "c_", "i_"))]
    for c in pit_cols[:10]:  # 只显示前10个
        cov = recent[c].notna().mean()
        status = "✅" if cov > 0.5 else "⚠️" if cov > 0 else "❌"
        print(f"  {status} {c}: {cov:.1%}")
    if len(pit_cols) > 10:
        avg_cov = float(recent[pit_cols].notna().mean().mean())  # type: ignore
        print(f"  ... ({len(pit_cols)} 个 PIT 因子, 平均覆盖率: {avg_cov:.1%})")
    
    # 7. 生成审计
    audit = compute_feature_audit(merged)
    
    if args.validate_only:
        print("\n📊 验证模式 (不保存文件)")
        print(f"  Shape: {audit['shape']}")
        print(f"  PIT 因子数: {len([c for c in audit['columns'] if c.startswith(('r_', 'm_', 'g_', 'a_', 'b_', 'c_', 'i_'))])}")
        return
    
    # 8. 保存
    print(f"\n💾 保存 features_v04_1.parquet...")
    merged.to_parquet(paths.features_v041, index=False)
    print(f"  ✅ {paths.features_v041}")
    
    print(f"💾 保存 v041_feature_audit.json...")
    with open(paths.v041_feature_audit, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    print(f"  ✅ {paths.v041_feature_audit}")
    
    print(f"\n{'='*60}")
    print(f"✅ Falcon V0.4.1 特征重计算完成!")
    print(f"  📊 {audit['shape'][0]} 行 × {audit['shape'][1]} 列")
    print(f"  📅 {audit['date_range'][0]} → {audit['date_range'][1]}")
    print(f"  🔢 数据哈希: {audit['data_hash']}")
    print(f"  ⏱️ 总耗时: {time.time()-t_total:.1f}秒")


if __name__ == "__main__":
    main()
