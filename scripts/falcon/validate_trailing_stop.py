#!/usr/bin/env python3
"""W1: 验证追踪止损参数 — 基于Falcon历史价格数据"""
import pandas as pd
import numpy as np

prices = pd.read_parquet("/home/hermes/.hermes/openclaw-archive/data/falcon/us_prices_daily.parquet")
prices['date'] = pd.to_datetime(prices['date'])

np.random.seed(42)
tickers = prices['ticker'].unique()
sample_tickers = np.random.choice(list(tickers), 50, replace=False)

dates = sorted(prices['date'].unique())
buy_dates = []
for year in range(2020, 2027):
    for month in [1, 4, 7, 10]:
        candidates = [d for d in dates if d.year == year and d.month == month]
        if candidates:
            buy_dates.append(candidates[0])

results = {
    "no_trail": [],
    "10_8": [],
    "5_5": [],
    "15_10": [],
    "10_5": [],
    "10_12": [],
}

for buy_date in buy_dates:
    for sym in sample_tickers:
        stock = prices.loc[
            (prices['ticker'] == sym) & (prices['date'] >= buy_date)
        ].head(30)
        if len(stock) < 10:
            continue
        entry_price = stock.iloc[0]['close']
        if entry_price <= 0:
            continue
        price_series = stock['close'].values

        for label, (activate, distance) in {
            "no_trail": (999, 0),
            "10_8": (0.10, 0.08),
            "5_5": (0.05, 0.05),
            "15_10": (0.15, 0.10),
            "10_5": (0.10, 0.05),
            "10_12": (0.10, 0.12),
        }.items():
            h = entry_price
            sold = False
            for day_price in price_series[1:]:
                if day_price > h:
                    h = day_price
                profit = (h - entry_price) / entry_price
                if activate < 999 and profit >= activate:
                    trail_price = h * (1 - distance)
                    if day_price <= trail_price:
                        pnl = (day_price - entry_price) / entry_price
                        results[label].append(pnl)
                        sold = True
                        break
                if (day_price - entry_price) / entry_price <= -0.15:
                    results[label].append(-0.15)
                    sold = True
                    break
            if not sold:
                final_pnl = (price_series[-1] - entry_price) / entry_price
                results[label].append(final_pnl)

print(f"模拟买入: {len(buy_dates)}季度 x 50只")
print(f"\n{'策略':20s} | {'样本':>5s} | {'平均盈亏':>8s} | {'胜率':>6s} | {'最大亏损':>8s} | {'Sharpe':>7s}")
print("-" * 75)

for label, pnls in results.items():
    if not pnls:
        continue
    arr = np.array(pnls)
    avg = arr.mean()
    win = (arr > 0).mean()
    worst = arr.min()
    sharpe = avg / arr.std() if arr.std() > 0 else 0
    name = {
        "no_trail": "无追踪止损",
        "10_8": "10%/8% (当前)",
        "5_5": "5%/5% (激进)",
        "15_10": "15%/10% (保守)",
        "10_5": "10%/5% (紧追踪)",
        "10_12": "10%/12% (宽松)",
    }.get(label, label)
    print(f"{name:20s} | {len(arr):5d} | {avg:+8.2%} | {win:+6.1%} | {worst:+8.2%} | {sharpe:+7.3f}")

valid = {k: v for k, v in results.items() if v}
if valid:
    best = max(valid.keys(), key=lambda k: np.array(valid[k]).mean() / (np.array(valid[k]).std() or 1))
    print(f"\nSharpe最优: {best}")
    best_pnl = max(valid.keys(), key=lambda k: np.array(valid[k]).mean())
    print(f"平均盈亏最优: {best_pnl}")
