#!/usr/bin/env python3
"""
Falcon Features PIT Builder
============================
从FMP Premium JSON数据包提取基本面因子，用PIT(Point-in-Time)对齐到日频，
补全features_v02.parquet中2016-2021和2025-2026缺失的基本面数据。

数据源: data/fmp_premium/data/raw/*.json (139K文件, 5.7GB)
输出: data/falcon/features_v02_pit.parquet

PIT规则:
- income_statement的filingDate是唯一可靠的"数据可用日期"
- ratios/key_metrics没有filingDate → 用同一ticker同一period的income filingDate推断
- 如果找不到filingDate → 用保守估计(季度末+60天)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

FMP_DIR = Path("data/fmp_premium/data/raw")
FEATURES_PATH = Path("data/falcon/features_v02.parquet")
OUTPUT_PATH = Path("data/falcon/features_v02_pit.parquet")

# 需要从ratios提取的列 (对应features_v02.parquet中的列名)
RATIO_COLS = {
    "priceToEarningsRatio": "priceToEarningsRatio",
    "priceToBookRatio": "priceToBookRatio",
    "priceToSalesRatio": "priceToSalesRatio",
    "priceToFreeCashFlowRatio": "priceToFreeCashFlowRatio",
    "enterpriseValueMultiple": "enterpriseValueMultiple",
    "grossProfitMargin": "grossProfitMargin",
    "netProfitMargin": "netProfitMargin",
    "operatingProfitMargin": "operatingProfitMargin",
    "ebitdaMargin": "ebitdaMargin",
    "assetTurnover": "assetTurnover",
    "inventoryTurnover": "inventoryTurnover",
    "receivablesTurnover": "receivablesTurnover",
    "debtToEquityRatio": "debtToEquityRatio",
    "currentRatio": "currentRatio",
    "quickRatio": "quickRatio",
    "financialLeverageRatio": "financialLeverageRatio",
    "freeCashFlowOperatingCashFlowRatio": "freeCashFlowOperatingCashFlowRatio",
    "operatingCashFlowRatio": "operatingCashFlowRatio",
    "dividendYieldPercentage": "dividendYieldPercentage",
    "dividendPayoutRatio": "dividendPayoutRatio",
}


def load_json(path):
    """Load JSON file, return empty list on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return []


def get_filing_dates(symbol):
    """从income_statement获取filingDate，作为PIT锚点。"""
    # Try quarterly first (most precise)
    q_path = FMP_DIR / f"income_statement_quarter_period-quarter_symbol-{symbol}_limit-120.json"
    q_data = load_json(q_path)
    
    filing_map = {}  # (fiscalYear, period) -> filingDate
    for r in q_data:
        fy = r.get("fiscalYear")
        period = r.get("period")
        fd = r.get("filingDate")
        if fy and period and fd:
            try:
                filing_map[(str(fy), period)] = datetime.strptime(fd, "%Y-%m-%d")
            except ValueError:
                pass
    
    # Also try annual
    a_path = FMP_DIR / f"income_statement_annual_period-annual_symbol-{symbol}_limit-30.json"
    a_data = load_json(a_path)
    for r in a_data:
        fy = r.get("fiscalYear")
        fd = r.get("filingDate")
        if fy and fd:
            try:
                filing_map[(str(fy), "FY")] = datetime.strptime(fd, "%Y-%m-%d")
            except ValueError:
                pass
    
    return filing_map


def parse_ratios(symbol, filing_map):
    """解析ratios数据，用filingDate做PIT对齐。返回[(available_date, {col: val}), ...]"""
    records = []
    
    for period_type in ["annual", "quarterly"]:
        limit = 30 if period_type == "annual" else 120
        suffix = "annual" if period_type == "annual" else "quarter"
        path = FMP_DIR / f"ratios_{period_type}_period-{suffix}_symbol-{symbol}_limit-{limit}.json"
        data = load_json(path)
        
        for r in data:
            fy = str(r.get("fiscalYear", ""))
            period = r.get("period", "FY" if period_type == "annual" else "")
            date_str = r.get("date", "")
            
            if not date_str:
                continue
            
            # 确定可用日期: 优先用filingDate
            avail_date = None
            key = (fy, period) if period else (fy, "FY")
            if key in filing_map:
                avail_date = filing_map[key]
            elif period_type == "annual":
                # 年报保守估计: 财年结束+90天
                try:
                    avail_date = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=90)
                except ValueError:
                    continue
            else:
                # 季报保守估计: 季度结束+60天
                try:
                    avail_date = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=60)
                except ValueError:
                    continue
            
            # 提取需要的列
            row = {}
            for src_col, dst_col in RATIO_COLS.items():
                val = r.get(src_col)
                if val is not None and val != "":
                    try:
                        row[dst_col] = float(val)
                    except (ValueError, TypeError):
                        pass
            
            # QoQ变化 (用于grossProfitMargin_qoq等)
            if period_type == "quarterly":
                for col in ["grossProfitMargin", "netProfitMargin", "operatingProfitMargin", "ebitdaMargin"]:
                    val = r.get(col)
                    if val is not None:
                        try:
                            row[f"{col}_qoq"] = float(val)
                        except (ValueError, TypeError):
                            pass
            
            if row:
                records.append((avail_date, row))
    
    return records


def parse_analyst_estimates(symbol):
    """解析analyst_estimates，获取EPS/Revenue revision和分析师数量。"""
    records = []
    
    for period_type in ["annual"]:
        limit = 10
        path = FMP_DIR / f"analyst_estimates_period-{period_type}_symbol-{symbol}_limit-{limit}_page-0.json"
        data = load_json(path)
        
        for r in data:
            date_str = r.get("date")
            if not date_str:
                continue
            
            try:
                avail_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            
            row = {}
            eps_avg = r.get("epsAvg")
            rev_avg = r.get("revenueAvg")
            num_eps = r.get("numAnalystsEps")
            num_rev = r.get("numAnalystsRevenue")
            
            if eps_avg is not None:
                try:
                    row["eps_revision"] = float(eps_avg)
                except (ValueError, TypeError):
                    pass
            if rev_avg is not None:
                try:
                    row["revenue_revision"] = float(rev_avg)
                except (ValueError, TypeError):
                    pass
            if num_eps is not None:
                try:
                    row["num_analysts_eps"] = float(num_eps)
                except (ValueError, TypeError):
                    pass
            if num_rev is not None:
                try:
                    row["num_analysts_rev"] = float(num_rev)
                except (ValueError, TypeError):
                    pass
            
            if row:
                records.append((avail_date, row))
    
    return records


def parse_grades(symbol):
    """解析grades_historical，计算grade sentiment分数。"""
    path = FMP_DIR / f"grades_historical_symbol-{symbol}.json"
    data = load_json(path)
    
    records = []
    for r in data:
        date_str = r.get("date")
        if not date_str:
            continue
        
        try:
            avail_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue
        
        strong_buy = r.get("analystRatingsStrongBuy", 0) or 0
        buy = r.get("analystRatingsBuy", 0) or 0
        hold = r.get("analystRatingsHold", 0) or 0
        sell = r.get("analystRatingsSell", 0) or 0
        strong_sell = r.get("analystRatingsStrongSell", 0) or 0
        
        total = strong_buy + buy + hold + sell + strong_sell
        if total > 0:
            # Grade score: (strong_buy*2 + buy*1 - sell*1 - strong_sell*2) / total
            score = (strong_buy * 2 + buy * 1 - sell * 1 - strong_sell * 2) / total
            records.append((avail_date, {"analyst_covered": 1.0}))
    
    return records


def build_pit_series(symbol, trading_dates):
    """
    为单个ticker构建PIT基本面时间序列。
    
    Args:
        symbol: ticker
        trading_dates: 该ticker的交易日列表 (sorted)
    
    Returns:
        dict of {col_name: pd.Series(index=trading_dates)}
    """
    # 1. 获取filingDate映射
    filing_map = get_filing_dates(symbol)
    
    # 2. 解析各数据源
    ratio_records = parse_ratios(symbol, filing_map)
    analyst_records = parse_analyst_estimates(symbol)
    grade_records = parse_grades(symbol)
    
    # 3. 合并所有记录，按available_date排序
    all_records = ratio_records + analyst_records + grade_records
    all_records.sort(key=lambda x: x[0])
    
    if not all_records:
        return {}
    
    # 4. 对每个交易日，找到最近可用的数据 (forward-fill)
    result_cols = defaultdict(lambda: np.full(len(trading_dates), np.nan))
    
    # 收集所有列名
    all_col_names = set()
    for _, row in all_records:
        all_col_names.update(row.keys())
    
    # Forward-fill: 对每个交易日，找<=该日的最新数据
    record_idx = 0
    current_data = {}
    
    for i, td in enumerate(trading_dates):
        # 推进到<=td的最新记录
        while record_idx < len(all_records) and all_records[record_idx][0] <= td:
            current_data.update(all_records[record_idx][1])
            record_idx += 1
        
        for col in all_col_names:
            if col in current_data:
                result_cols[col][i] = current_data[col]
    
    return {col: pd.Series(vals, index=trading_dates) for col, vals in result_cols.items()}


def main():
    print("=" * 60)
    print("Falcon Features PIT Builder")
    print("=" * 60)
    
    # 1. 读取现有features
    print("\n[1/5] Loading features_v02.parquet...")
    feat = pd.read_parquet(FEATURES_PATH)
    feat["date"] = pd.to_datetime(feat["date"])
    tickers = feat["ticker"].unique().tolist()
    print(f"  {len(tickers)} tickers, {len(feat)} rows")
    print(f"  Date range: {feat['date'].min()} ~ {feat['date'].max()}")
    
    # 2. 对每个ticker构建PIT基本面数据
    print(f"\n[2/5] Building PIT fundamental data for {len(tickers)} tickers...")
    
    # 基本面列 (需要补全的)
    fundamental_cols = list(RATIO_COLS.values()) + [
        "grossProfitMargin_qoq", "netProfitMargin_qoq",
        "operatingProfitMargin_qoq", "ebitdaMargin_qoq",
        "eps_revision", "revenue_revision",
        "num_analysts_eps", "num_analysts_rev",
    ]
    
    # 创建结果DataFrame (先复制原features)
    result = feat.copy()
    
    # 初始化新列为NaN (如果不存在)
    for col in fundamental_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    processed = 0
    errors = 0
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(tickers)} ({processed} ok, {errors} err)...")
        
        try:
            # 获取该ticker的交易日
            mask = result["ticker"] == ticker
            ticker_dates = result.loc[mask, "date"].sort_values().values
            
            if len(ticker_dates) == 0:
                continue
            
            # 转换为datetime
            trading_dates = pd.to_datetime(ticker_dates)
            
            # 构建PIT数据
            pit_data = build_pit_series(ticker, trading_dates)
            
            # 写入result
            for col, series in pit_data.items():
                if col in result.columns:
                    result.loc[mask, col] = series.values
            
            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR {ticker}: {e}")
    
    print(f"  Done: {processed} ok, {errors} errors")
    
    # 3. 验证补全效果
    print("\n[3/5] Validating coverage...")
    result["year"] = result["date"].dt.year
    
    key_cols = ["priceToEarningsRatio", "grossProfitMargin", "debtToEquityRatio", "currentRatio"]
    for col in key_cols:
        if col in result.columns:
            cov = result.groupby("year")[col].apply(lambda x: (x.notna() & (x != 0)).mean())
            print(f"  {col}:")
            for y, v in zip(cov.index, cov.values):
                marker = "✅" if v > 0.5 else "⚠️" if v > 0 else "❌"
                print(f"    {y}: {v:.1%} {marker}")
    
    # 4. 保存
    print(f"\n[4/5] Saving to {OUTPUT_PATH}...")
    result = result.drop(columns=["year"])
    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Saved: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f}MB)")
    
    # 5. 总结
    print("\n[5/5] Summary:")
    print(f"  Original: {FEATURES_PATH} ({os.path.getsize(FEATURES_PATH) / 1024 / 1024:.1f}MB)")
    print(f"  Updated:  {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.1f}MB)")
    
    # 对比NaN比例
    orig = pd.read_parquet(FEATURES_PATH)
    for col in key_cols:
        if col in orig.columns and col in result.columns:
            orig_nan = orig[col].isna().mean()
            new_nan = result[col].isna().mean()
            print(f"  {col}: NaN {orig_nan:.1%} -> {new_nan:.1%}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
