#!/usr/bin/env python3
"""
Falcon V0.4.6 数据重建脚本
=========================
重建所有缺失的模型关联文件:
  1. fmp_ratios_historical.json (FMP 财务比率, 季度)
  2. fmp_key_metrics.json (FMP 关键指标, 季度)
  3. fmp_financial_growth.json (FMP 财务增长, 季度)
  4. us_prices_daily.parquet (10年日线价格)
  5. features_v04_1.parquet (V0.4.6 因子矩阵)

用法: python3 scripts/falcon/rebuild_all_10yr.py
"""
import sys
import os
import json
import time
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# ─── 配置 ───────────────────────────────────────────
DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

FMP_KEY = "185VX9wJgwR7ZwQsLIfEbUzc066hfpLN"
FMP_BASE = "https://financialmodelingprep.com/stable"

# 从 fmp_balance_sheet.json 获取 476 ticker 列表
with open(DATA_DIR / "fmp_balance_sheet.json") as f:
    TICKERS = sorted(json.load(f).keys())
print(f"📋 Universe: {len(TICKERS)} tickers")


# ─── FMP 数据下载 ────────────────────────────────────

def fetch_fmp_stable(endpoint, ticker, period="quarter", limit=40):
    """从 FMP stable API 拉取数据"""
    url = f"{FMP_BASE}/{endpoint}?symbol={ticker}&period={period}&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        return None


def download_fmp_file(endpoint, out_name, fields=None, batch_size=50, delay=0.15):
    """批量下载 FMP 数据, 支持断点续传"""
    out_file = SNAPSHOTS_DIR / f"{out_name}.json"
    
    # 加载已有数据
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
        print(f"  📂 已有 {len(existing)} 只缓存")
    
    # 找缺失 ticker
    missing = [t for t in TICKERS if t not in existing or not existing[t]]
    if not missing:
        print(f"  ✅ 已完整, 跳过")
        return
    
    print(f"  📥 下载 {len(missing)} 只 (已有 {len(existing)})...")
    data = dict(existing)
    
    def fetch_one(ticker):
        result = fetch_fmp_stable(endpoint, ticker)
        if result and isinstance(result, list) and fields:
            filtered = []
            for d in result:
                row = {"date": d.get("date", "")}
                if "filingDate" in d:
                    row["filingDate"] = d.get("filingDate", "")
                for f in fields:
                    row[f] = d.get(f)
                filtered.append(row)
            filtered.sort(key=lambda x: x["date"])
            return ticker, filtered
        return ticker, result if isinstance(result, list) else []
    
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_one, t): t for t in missing}
        for f in as_completed(futures):
            t, result = f.result()
            data[t] = result
            done += 1
            if done % 50 == 0:
                print(f"    {done}/{len(missing)}...")
            time.sleep(delay)  # rate limit
    
    with open(out_file, "w") as f:
        json.dump(data, f)
    has = sum(1 for v in data.values() if v)
    print(f"  ✅ {out_name}: {has}/{len(TICKERS)} 有数据")


def download_ratios():
    """下载 fmp_ratios_historical.json"""
    fields = [
        "priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
        "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
        "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
        "ebitdaMargin", "assetTurnover", "inventoryTurnover",
        "receivablesTurnover", "debtToEquityRatio", "currentRatio",
        "quickRatio", "financialLeverageRatio",
        "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
        "dividendYieldPercentage", "dividendPayoutRatio",
    ]
    print("\n📥 [1/5] fmp_ratios_historical.json")
    download_fmp_file("ratios", "fmp_ratios_historical", fields=fields)


def download_key_metrics():
    """下载 fmp_key_metrics.json"""
    fields = [
        "earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
        "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
        "returnOnCapitalEmployed", "returnOnInvestedCapital",
        "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
        "cashConversionCycle", "capexToRevenue", "capexToDepreciation",
        "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
        "netDebtToEBITDA", "operatingReturnOnAssets",
    ]
    print("\n📥 [2/5] fmp_key_metrics.json")
    download_fmp_file("key-metrics", "fmp_key_metrics", fields=fields)


def download_financial_growth():
    """下载 fmp_financial_growth.json"""
    fields = [
        "revenueGrowth", "grossProfitGrowth", "ebitgrowth",
        "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
        "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
        "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
        "receivablesGrowth", "inventoryGrowth", "assetGrowth",
        "bookValueperShareGrowth", "debtGrowth",
    ]
    print("\n📥 [3/5] fmp_financial_growth.json")
    download_fmp_file("financial-growth", "fmp_financial_growth", fields=fields)


# ─── 价格数据下载 (yfinance) ────────────────────────

def download_prices_10yr():
    """下载 10 年日线价格"""
    print("\n📥 [4/5] us_prices_daily.parquet (10年)")
    
    try:
        import yfinance as yf
    except ImportError:
        print("  ❌ yfinance 未安装, pip install yfinance")
        return
    
    out_file = DATA_DIR / "us_prices_daily.parquet"
    
    # 检查已有数据
    existing_dates = set()
    existing_df = None
    if out_file.exists():
        existing_df = pd.read_parquet(out_file)
        existing_dates = set(existing_df["date"].unique())
        print(f"  📂 已有 {existing_df['ticker'].nunique()} 只, {len(existing_dates)} 天")
    
    # 10年日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10 + 30)  # 多30天buffer
    
    all_frames = []
    batch_size = 20  # yfinance 批量下载
    
    for i in range(0, len(TICKERS), batch_size):
        batch = TICKERS[i:i + batch_size]
        try:
            data = yf.download(
                batch,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            
            if len(batch) == 1:
                ticker = batch[0]
                if not data.empty:
                    df = data.reset_index()
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    df = df.rename(columns={"Date": "date", "Close": "close", "Adj Close": "adj_close"})
                    df["ticker"] = ticker
                    all_frames.append(df[["date", "ticker", "close", "adj_close"]].dropna())
            else:
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            sub = data[ticker].reset_index()
                            sub = sub.rename(columns={"Date": "date", "Close": "close", "Adj Close": "adj_close"})
                            sub["ticker"] = ticker
                            sub = sub[["date", "ticker", "close", "adj_close"]].dropna(subset=["close"])
                            if not sub.empty:
                                all_frames.append(sub)
                    except Exception:
                        pass
            
            print(f"    {min(i + batch_size, len(TICKERS))}/{len(TICKERS)}...")
        except Exception as e:
            print(f"    ❌ batch {i}-{i+batch_size}: {e}")
    
    if not all_frames:
        print("  ❌ 无数据下载成功")
        return
    
    new_df = pd.concat(all_frames, ignore_index=True)
    new_df["date"] = pd.to_datetime(new_df["date"]).dt.strftime("%Y-%m-%d")
    
    # 合并已有数据
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        combined = new_df
    
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    combined.to_parquet(out_file, index=False)
    
    print(f"  ✅ us_prices_daily.parquet: {combined['ticker'].nunique()} 只, "
          f"{combined['date'].min()} → {combined['date'].max()}, "
          f"{len(combined)} 行")


# ─── 特征构建 ────────────────────────────────────────

def build_features():
    """构建 features_v04_1.parquet"""
    print("\n📥 [5/5] features_v04_1.parquet (构建因子矩阵)")
    build_script = Path("/home/hermes/.hermes/openclaw-archive/scripts/falcon/build_features_v041.py")
    
    if not build_script.exists():
        print(f"  ❌ 脚本不存在: {build_script}")
        return
    
    import subprocess
    result = subprocess.run(
        [sys.executable, str(build_script)],
        capture_output=True, text=True, timeout=600
    )
    
    if result.returncode == 0:
        # 提取最后几行输出
        lines = result.stdout.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")
    else:
        print(f"  ❌ 构建失败 (exit {result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")


# ─── 主流程 ──────────────────────────────────────────

def main():
    t0 = time.time()
    
    download_ratios()
    download_key_metrics()
    download_financial_growth()
    download_prices_10yr()
    build_features()
    
    elapsed = time.time() - t0
    print(f"\n⏱️ 总耗时: {elapsed/60:.1f} 分钟")
    
    # 验证所有文件
    print("\n📋 最终文件状态:")
    files = {
        "fmp_ratios_historical.json": SNAPSHOTS_DIR / "fmp_ratios_historical.json",
        "fmp_key_metrics.json": SNAPSHOTS_DIR / "fmp_key_metrics.json",
        "fmp_financial_growth.json": SNAPSHOTS_DIR / "fmp_financial_growth.json",
        "us_prices_daily.parquet": DATA_DIR / "us_prices_daily.parquet",
        "features_v04_1.parquet": DATA_DIR / "features_v04_1.parquet",
    }
    for name, path in files.items():
        if path.exists():
            size = path.stat().st_size
            if size > 1024 * 1024:
                print(f"  ✅ {name}: {size / 1024 / 1024:.1f} MB")
            else:
                print(f"  ✅ {name}: {size / 1024:.1f} KB")
        else:
            print(f"  ❌ {name}: 缺失")


if __name__ == "__main__":
    main()
