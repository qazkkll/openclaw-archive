#!/usr/bin/env python3
"""
🦅 Falcon V0.2.1 — 修复前视偏差
拉取FMP历史季度数据, 回测只用point-in-time数据
"""
import json, os, time, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(Path("/home/hermes/.hermes/openclaw-archive/.env"))
FMP_KEY = os.environ.get("FMP_API_KEY", "")

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")

FMP_FIELDS = ["priceToEarningsRatio", "priceToBookRatio", "priceToSalesRatio",
              "priceToFreeCashFlowRatio", "enterpriseValueMultiple",
              "grossProfitMargin", "netProfitMargin", "operatingProfitMargin", "ebitdaMargin",
              "assetTurnover", "inventoryTurnover", "receivablesTurnover",
              "debtToEquityRatio", "currentRatio", "quickRatio", "financialLeverageRatio",
              "freeCashFlowOperatingCashFlowRatio", "operatingCashFlowRatio",
              "dividendYieldPercentage", "dividendPayoutRatio"]


def fetch_fmp_ratios_historical(ticker, limit=40):
    """拉取FMP全部历史季度数据, 返回[{date, field1, field2, ...}, ...]"""
    url = f"https://financialmodelingprep.com/stable/ratios?symbol={ticker}&period=quarter&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    result = []
    for q in data:
        row = {"date": q.get("date", "")}
        for field in FMP_FIELDS:
            row[field] = q.get(field)
        # QoQ趋势: 需要上一季度, 但先存原始数据, 后面算
        result.append(row)
    # 按日期排序
    result.sort(key=lambda x: x["date"])
    # 计算QoQ
    for i in range(1, len(result)):
        for field in ["grossProfitMargin", "netProfitMargin", "operatingProfitMargin", "ebitdaMargin"]:
            curr = result[i].get(field)
            prev = result[i-1].get(field)
            if curr is not None and prev is not None and prev != 0:
                result[i][f"{field}_qoq"] = (curr - prev) / abs(prev)
    return result


def fetch_fmp_analyst_historical(ticker, limit=40):
    """拉取FMP分析师历史估计, 返回[{date, epsAvg, revenueAvg, ...}, ...]"""
    url = f"https://financialmodelingprep.com/stable/analyst-estimates?symbol={ticker}&period=quarter&limit={limit}&apikey={FMP_KEY}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception:
        return []
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
    # 计算EPS修正(当前vs上一季度)
    for i in range(1, len(result)):
        eps_now = result[i].get("epsAvg")
        eps_prev = result[i-1].get("epsAvg")
        if eps_now and eps_prev and eps_prev != 0:
            result[i]["eps_revision"] = (eps_now - eps_prev) / abs(eps_prev)
        rev_now = result[i].get("revenueAvg")
        rev_prev = result[i-1].get("revenueAvg")
        if rev_now and rev_prev and rev_prev != 0:
            result[i]["revenue_revision"] = (rev_now - rev_prev) / abs(rev_prev)
        # EPS分歧度
        eps_high = result[i].get("epsHigh")
        eps_low = result[i].get("epsLow")
        if eps_now and eps_now != 0 and eps_high is not None and eps_low is not None:
            result[i]["eps_dispersion"] = (eps_high - eps_low) / abs(eps_now)
    return result


def main():
    import pandas as pd
    uni = pd.read_csv(DATA_DIR.parent.parent / "config" / "universe_sp500.csv")
    # 用features_v02的tickers(已验证476只)
    feat = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    tickers = sorted(feat["ticker"].unique())
    print(f"📥 拉取 {len(tickers)} 只FMP历史季度(limit=40)...")

    # FMP Ratios 历史
    fmp_hist_file = DATA_DIR / "fmp_ratios_historical.json"
    fmp_data = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_fmp_ratios_historical, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            try:
                fmp_data[t] = f.result()
            except:
                fmp_data[t] = []
            done += 1
            if done % 100 == 0:
                print(f"  ratios: {done}/{len(tickers)}...")
    with open(fmp_hist_file, "w") as f:
        json.dump(fmp_data, f)
    has = sum(1 for v in fmp_data.values() if v)
    avg_q = sum(len(v) for v in fmp_data.values()) / max(has, 1)
    print(f"✅ FMP Ratios: {has}/{len(tickers)} 有数据, 平均{avg_q:.0f}季度/只")

    # FMP Analyst 历史
    ana_hist_file = DATA_DIR / "analyst_historical.json"
    ana_data = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_fmp_analyst_historical, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            t = futures[f]
            try:
                ana_data[t] = f.result()
            except:
                ana_data[t] = []
            done += 1
            if done % 100 == 0:
                print(f"  analyst: {done}/{len(tickers)}...")
    with open(ana_hist_file, "w") as f:
        json.dump(ana_data, f)
    has_a = sum(1 for v in ana_data.values() if v)
    print(f"✅ Analyst: {has_a}/{len(tickers)} 有数据")

    # 统计
    # 看2022-01-03时有多少季度数据可用
    sample_t = "AAPL"
    if sample_t in fmp_data:
        q2022 = [q for q in fmp_data[sample_t] if q["date"] <= "2022-01-03"]
        print(f"\n{sample_t} 2022-01-03时可用: {len(q2022)} 季度")
        if q2022:
            latest = q2022[-1]
            print(f"  最新: {latest['date']}, PE={latest.get('priceToEarningsRatio', 'N/A')}")


if __name__ == "__main__":
    main()
