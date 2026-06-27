#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — 全量FMP因子 + 幸存者偏差修正
拉取: insider/key_metrics/financial_growth/dcf/price_target/earnings
"""
import json, os, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(Path("/home/hermes/.hermes/openclaw-archive/.env"))
FMP_KEY = os.environ.get("FMP_API_KEY", "")
DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")


def fetch_json(url, timeout=15):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except:
        return None


def fetch_insider(ticker):
    """Insider交易: 净买卖金额/次数, CEO/CFO交易。"""
    url = f"https://financialmodelingprep.com/stable/insider-trading/search?symbol={ticker}&limit=100&apikey={FMP_KEY}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []
    result = []
    for d in data:
        result.append({
            "date": d.get("transactionDate", ""),
            "filingDate": d.get("filingDate", ""),
            "type": d.get("transactionType", ""),
            "acq_disp": d.get("acquisitionOrDisposition", ""),
            "shares": d.get("securitiesTransacted", 0),
            "price": d.get("price", 0),
            "owner": d.get("reportingName", ""),
            "ownerType": d.get("typeOfOwner", ""),
        })
    return result


def fetch_key_metrics(ticker):
    """Key Metrics: 47字段/季度 (ROE/ROA/EV/FCF yield等)。"""
    url = f"https://financialmodelingprep.com/stable/key-metrics?symbol={ticker}&period=quarter&limit=40&apikey={FMP_KEY}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []
    fields = ["earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
              "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
              "returnOnCapitalEmployed", "returnOnInvestedCapital",
              "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
              "cashConversionCycle", "capexToRevenue", "capexToDepreciation",
              "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
              "netDebtToEBITDA", "operatingReturnOnAssets", "interestBurden",
              "taxBurden", "marketCap", "enterpriseValue"]
    result = []
    for d in data:
        row = {"date": d.get("date", "")}
        for f in fields:
            row[f] = d.get(f)
        result.append(row)
    result.sort(key=lambda x: x["date"])
    return result


def fetch_financial_growth(ticker):
    """Financial Growth: 收入/利润/FCF增长率。"""
    url = f"https://financialmodelingprep.com/stable/financial-growth?symbol={ticker}&period=quarter&limit=40&apikey={FMP_KEY}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []
    fields = ["revenueGrowth", "grossProfitGrowth", "ebitgrowth",
              "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
              "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
              "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
              "tenYOperatingCFGrowthPerShare", "fiveYOperatingCFGrowthPerShare",
              "threeYOperatingCFGrowthPerShare", "receivablesGrowth",
              "inventoryGrowth", "assetGrowth", "bookValueperShareGrowth",
              "debtGrowth"]
    result = []
    for d in data:
        row = {"date": d.get("date", "")}
        for f in fields:
            row[f] = d.get(f)
        result.append(row)
    result.sort(key=lambda x: x["date"])
    return result


def fetch_dcf(ticker):
    """DCF公允价值。"""
    url = f"https://financialmodelingprep.com/stable/discounted-cash-flow?symbol={ticker}&apikey={FMP_KEY}"
    data = fetch_json(url)
    if isinstance(data, list) and data:
        return {"date": data[0].get("date", ""), "dcf": data[0].get("dcf"), "price": data[0].get("Stock Price")}
    elif isinstance(data, dict):
        return {"date": data.get("date", ""), "dcf": data.get("dcf"), "price": data.get("Stock Price")}
    return {}


def fetch_price_target(ticker):
    """分析师价格目标共识。"""
    url = f"https://financialmodelingprep.com/stable/price-target-consensus?symbol={ticker}&apikey={FMP_KEY}"
    data = fetch_json(url)
    if isinstance(data, list) and data:
        return data[0]
    return {}


def fetch_income_stmt(ticker):
    """利润表: 收入/利润/毛利率等。"""
    url = f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&period=quarter&limit=40&apikey={FMP_KEY}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []
    fields = ["revenue", "costOfRevenue", "grossProfit", "operatingIncome",
              "netIncome", "ebitda", "epsdiluted", "weightedAverageShsOut"]
    result = []
    for d in data:
        row = {"date": d.get("date", ""), "filingDate": d.get("filingDate", "")}
        for f in fields:
            row[f] = d.get(f)
        result.append(row)
    result.sort(key=lambda x: x["date"])
    return result


def fetch_cashflow(ticker):
    """现金流表: OCF/FCF/CapEx。"""
    url = f"https://financialmodelingprep.com/stable/cash-flow-statement?symbol={ticker}&period=quarter&limit=40&apikey={FMP_KEY}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []
    fields = ["operatingCashFlow", "capitalExpenditure", "freeCashFlow",
              "dividendsPaid", "commonStockRepurchased"]
    result = []
    for d in data:
        row = {"date": d.get("date", "")}
        for f in fields:
            row[f] = d.get(f)
        result.append(row)
    result.sort(key=lambda x: x["date"])
    return result


def fetch_balance_sheet(ticker):
    """资产负债表。"""
    url = f"https://financialmodelingprep.com/stable/balance-sheet-statement?symbol={ticker}&period=quarter&limit=40&apikey={FMP_KEY}"
    data = fetch_json(url)
    if not isinstance(data, list):
        return []
    fields = ["totalAssets", "totalLiabilities", "totalStockholdersEquity",
              "cashAndCashEquivalents", "totalDebt", "netDebt",
              "workingCapital", "totalInvestments"]
    result = []
    for d in data:
        row = {"date": d.get("date", "")}
        for f in fields:
            row[f] = d.get(f)
        result.append(row)
    result.sort(key=lambda x: x["date"])
    return result


def main():
    import pandas as pd
    # 用已有的476只(后面再处理幸存者偏差)
    feat = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    tickers = sorted(feat["ticker"].unique())
    print(f"📥 全量FMP因子拉取: {len(tickers)} 只")

    endpoints = {
        "insider": fetch_insider,
        "key_metrics": fetch_key_metrics,
        "financial_growth": fetch_financial_growth,
        "income_stmt": fetch_income_stmt,
        "cashflow": fetch_cashflow,
        "balance_sheet": fetch_balance_sheet,
    }

    for name, func in endpoints.items():
        out_file = DATA_DIR / f"fmp_{name}.json"
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"📂 {name}: 已有 {len(existing)} 只缓存")
            # 补拉缺失的
            missing = [t for t in tickers if t not in existing]
            if not missing:
                continue
            print(f"  补拉 {len(missing)} 只...")
        else:
            existing = {}
            missing = tickers
            print(f"📥 拉取 {name}: {len(missing)} 只...")

        data = dict(existing)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(func, t): t for t in missing}
            done = 0
            for f in as_completed(futures):
                t = futures[f]
                try:
                    data[t] = f.result()
                except:
                    data[t] = []
                done += 1
                if done % 100 == 0:
                    print(f"  {name}: {done}/{len(missing)}...")
        with open(out_file, "w") as f:
            json.dump(data, f)
        has = sum(1 for v in data.values() if v)
        print(f"✅ {name}: {has}/{len(tickers)} 有数据")

    # DCF和price_target (单条数据, 不是历史)
    for name, func in [("dcf", fetch_dcf), ("price_target", fetch_price_target)]:
        out_file = DATA_DIR / f"fmp_{name}.json"
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            missing = [t for t in tickers if t not in existing]
            if not missing:
                print(f"📂 {name}: 已有 {len(existing)} 只缓存")
                continue
        else:
            existing = {}
            missing = tickers

        print(f"📥 拉取 {name}: {len(missing)} 只...")
        data = dict(existing)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(func, t): t for t in missing}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    data[t] = f.result()
                except:
                    data[t] = {}
        with open(out_file, "w") as f:
            json.dump(data, f)
        has = sum(1 for v in data.values() if v)
        print(f"✅ {name}: {has}/{len(tickers)} 有数据")

    # 汇总
    print(f"\n📊 全量因子统计:")
    for name in list(endpoints.keys()) + ["dcf", "price_target"]:
        f = DATA_DIR / f"fmp_{name}.json"
        if f.exists():
            with open(f) as fh:
                d = json.load(fh)
            has = sum(1 for v in d.values() if v)
            print(f"  {name}: {has}/{len(tickers)} 只")


if __name__ == "__main__":
    main()
