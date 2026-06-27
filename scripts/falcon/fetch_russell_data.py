#!/usr/bin/env python3
"""
🦅 Falcon Russell扩展 — 拉取小盘股数据
Phase 1: 抽样500只pilot
Phase 2: 扩展到2000只
"""
import json, os, time, urllib.request, random, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(Path("/home/hermes/.hermes/openclaw-archive/.env"))
MASSIVE_KEY = os.environ.get("MASSIVE_API_KEY", "")
FMP_KEY = os.environ.get("FMP_API_KEY", "")
DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")
CONFIG_DIR = Path("/home/hermes/.hermes/openclaw-archive/config")


def fetch_massive_daily(ticker, from_date="2022-01-01", to_date="2024-12-31"):
    """拉取Massive API日K数据。"""
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?apiKey={MASSIVE_KEY}&limit=5000"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        results = data.get("results", [])
        if not results:
            return None
        rows = []
        for bar in results:
            ts = bar.get("t", 0) / 1000
            from datetime import datetime
            date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            rows.append({
                "date": date,
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
                "vwap": bar.get("vw"),
            })
        return rows
    except:
        return None


def fetch_fmp_ratios(ticker, limit=40):
    """FMP Ratios历史。"""
    url = f"https://financialmodelingprep.com/stable/ratios?symbol={ticker}&period=quarter&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if not isinstance(data, list):
            return []
        fields = ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
                  "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
                  "grossProfitMargin", "netProfitMargin", "operatingProfitMargin",
                  "ebitdaMargin", "assetTurnover", "inventoryTurnover",
                  "receivablesTurnover", "debtToEquityRatio", "currentRatio",
                  "quickRatio", "financialLeverageRatio",
                  "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
                  "dividendYieldPercentage", "dividendPayoutRatio"]
        result = []
        for q in data:
            row = {"date": q.get("date", "")}
            for f in fields:
                row[f] = q.get(f)
            result.append(row)
        result.sort(key=lambda x: x["date"])
        return result
    except:
        return []


def fetch_fmp_metrics(ticker, limit=40):
    """FMP Key Metrics历史。"""
    url = f"https://financialmodelingprep.com/stable/key-metrics?symbol={ticker}&period=quarter&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if not isinstance(data, list):
            return []
        fields = ["earningsYield", "evToEBITDA", "evToFreeCashFlow", "evToSales",
                  "freeCashFlowYield", "returnOnEquity", "returnOnAssets",
                  "returnOnCapitalEmployed", "returnOnInvestedCapital",
                  "returnOnTangibleAssets", "incomeQuality", "grahamNumber",
                  "cashConversionCycle", "capexToRevenue",
                  "researchAndDevelopementToRevenue", "stockBasedCompensationToRevenue",
                  "netDebtToEBITDA", "operatingReturnOnAssets"]
        result = []
        for q in data:
            row = {"date": q.get("date", "")}
            for f in fields:
                row[f] = q.get(f)
            result.append(row)
        result.sort(key=lambda x: x["date"])
        return result
    except:
        return []


def fetch_fmp_growth(ticker, limit=40):
    """FMP Growth历史。"""
    url = f"https://financialmodelingprep.com/stable/financial-growth?symbol={ticker}&period=quarter&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if not isinstance(data, list):
            return []
        fields = ["revenueGrowth", "grossProfitGrowth", "ebitgrowth",
                  "operatingIncomeGrowth", "netIncomeGrowth", "epsdilutedGrowth",
                  "freeCashFlowGrowth", "fiveYRevenueGrowthPerShare",
                  "threeYRevenueGrowthPerShare", "assetGrowth", "bookValueperShareGrowth"]
        result = []
        for q in data:
            row = {"date": q.get("date", "")}
            for f in fields:
                row[f] = q.get(f)
            result.append(row)
        result.sort(key=lambda x: x["date"])
        return result
    except:
        return []


def fetch_fmp_analyst(ticker, limit=40):
    """FMP Analyst历史。"""
    url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol={ticker}&period=quarter&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
        if not isinstance(data, list):
            return []
        result = []
        for q in data:
            row = {
                "date": q.get("date", ""),
                "epsAvg": q.get("epsAvg"),
                "epsHigh": q.get("epsHigh"),
                "epsLow": q.get("epsLow"),
                "revenueAvg": q.get("revenueAvg"),
                "numAnalystsEps": q.get("numAnalystsEps"),
                "numAnalystsRevenue": q.get("numAnalystsRevenue"),
            }
            result.append(row)
        result.sort(key=lambda x: x["date"])
        for i in range(1, len(result)):
            eps_now = result[i].get("epsAvg")
            eps_prev = result[i-1].get("epsAvg")
            if eps_now and eps_prev and eps_prev != 0:
                result[i]["eps_revision"] = (eps_now - eps_prev) / abs(eps_prev)
            rev_now = result[i].get("revenueAvg")
            rev_prev = result[i-1].get("revenueAvg")
            if rev_now and rev_prev and rev_prev != 0:
                result[i]["revenue_revision"] = (rev_now - rev_prev) / abs(rev_prev)
            eps_high = result[i].get("epsHigh")
            eps_low = result[i].get("epsLow")
            if eps_now and eps_now != 0 and eps_high is not None and eps_low is not None:
                result[i]["eps_dispersion"] = (eps_high - eps_low) / abs(eps_now)
        return result
    except:
        return []


def main():
    import pandas as pd
    t0 = time.time()

    # ── Phase 0: 获取候选ticker ──
    print("Phase 0: 获取候选小盘股ticker...")
    url = f"https://financialmodelingprep.com/stable/stock-list?apikey={FMP_KEY}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    all_symbols = sorted(set(d["symbol"] for d in data if re.match(r"^[A-Z]{1,5}$", d["symbol"])))

    existing = set(pd.read_parquet(DATA_DIR / "features_v02.parquet")["ticker"].unique())
    candidates = [s for s in all_symbols if s not in existing]
    print(f"  候选: {len(candidates)} 只")

    # ── Phase 1: 批量验证Massive可用性 ──
    print("\nPhase 1: 批量验证Massive API (2000只)...")
    random.seed(42)
    sample = random.sample(candidates, min(2000, len(candidates)))

    valid_tickers = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = {ex.submit(fetch_massive_daily, t, "2024-01-01", "2024-01-10"): t for t in sample}
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            try:
                result = f.result()
                if result and len(result) > 0:
                    valid_tickers.append(t)
            except:
                pass
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(sample)}: {len(valid_tickers)} 有效...")

    print(f"  ✅ Massive可用: {len(valid_tickers)}/{len(sample)}")

    # 取前2000只(或全部)
    target_tickers = valid_tickers[:2000]
    print(f"  目标: {len(target_tickers)} 只")

    # ── Phase 2: 拉取3年日K数据 ──
    print(f"\nPhase 2: 拉取 {len(target_tickers)} 只日K数据 (2022-2024)...")
    price_data = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_massive_daily, t, "2022-01-01", "2024-12-31"): t for t in target_tickers}
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            try:
                result = f.result()
                if result and len(result) > 200:  # 至少200天数据
                    price_data[t] = result
            except:
                pass
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(target_tickers)}: {len(price_data)} 有足够数据...")

    print(f"  ✅ 有足够日K数据: {len(price_data)} 只")

    # 保存价格数据
    price_file = DATA_DIR / "russell_prices.json"
    with open(price_file, "w") as f:
        json.dump(price_data, f)
    print(f"  💾 保存: {price_file}")

    # ── Phase 3: 拉取FMP基本面数据 ──
    valid_price_tickers = list(price_data.keys())
    print(f"\nPhase 3: 拉取 {len(valid_price_tickers)} 只FMP基本面数据...")

    for name, func in [("fmp_ratios_russell", fetch_fmp_ratios),
                        ("fmp_metrics_russell", fetch_fmp_metrics),
                        ("fmp_growth_russell", fetch_fmp_growth),
                        ("fmp_analyst_russell", fetch_fmp_analyst)]:
        out_file = DATA_DIR / f"{name}.json"
        if out_file.exists():
            with open(out_file) as f:
                existing_data = json.load(f)
            missing = [t for t in valid_price_tickers if t not in existing_data]
            if not missing:
                print(f"  📂 {name}: 已有 {len(existing_data)} 只缓存")
                continue
            print(f"  补拉 {name}: {len(missing)} 只...")
            data = dict(existing_data)
        else:
            data = {}
            missing = valid_price_tickers
            print(f"  📥 拉取 {name}: {len(missing)} 只...")

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
                if done % 200 == 0:
                    print(f"    {name}: {done}/{len(missing)}...")

        with open(out_file, "w") as f:
            json.dump(data, f)
        has = sum(1 for v in data.values() if v)
        print(f"  ✅ {name}: {has}/{len(data)} 有数据")

    print(f"\n⏱️ {time.time()-t0:.0f}秒")


if __name__ == "__main__":
    main()
