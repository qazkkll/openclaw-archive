#!/usr/bin/env python3
"""
🦅 Falcon 分钟级择时回测
=========================
对比三种入场策略：
  1. 基线: 开盘价全仓买入 (不择时)
  2. VWAP择时: 等价格跌破VWAP后买入
  3. 分批建仓: 基于入场评分分3批进

退出: 统一用固定持有期，只比较入场差异。
数据: Alpaca分钟K线 (含VWAP, trade_count)

用法:
  python3 falcon_timing_backtest.py                    # 默认3个月，top-5
  python3 falcon_timing_backtest.py --months 1         # 1个月
  python3 falcon_timing_backtest.py --top-n 3          # 每日选3只
"""

import sys, json, os, time, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Alpaca ──
from dotenv import load_dotenv
load_dotenv()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# ── Falcon engine ──
FALCON_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FALCON_DIR))
from falcon_v03_engine import (
    get_pit, RATIO_FIELDS, METRIC_FIELDS, GROWTH_FIELDS,
    ANALYST_FIELDS, TECH_FIELDS, futu_cost,
)

# ═══════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════
DATA_DIR = FALCON_DIR.parent.parent / "data" / "falcon"
HOLD_DAYS = 30
STOP_LOSS = -0.15
COST_PER_TRADE = 0.001  # 0.1% per side

# Falcon权重 (SPX最优)
WEIGHTS = {
    "fund_ratio": 0.7,
    "analyst": 0.2,
    "fund_metric": 0.1,
    "tech": 0.0,
}

# 入场评分权重
ENTRY_WEIGHTS = {
    "price_position": 0.30,   # 价格在日内区间的位置
    "vwap_deviation": 0.30,   # 偏离VWAP
    "volume_confirm": 0.20,   # 成交量确认
    "momentum": 0.10,         # 分钟动量
    "spread_quality": 0.10,   # 价差质量(用trade_count近似)
}


# ═══════════════════════════════════════════
# Step 1: 生成Falcon日频信号
# ═══════════════════════════════════════════
def load_falcon_data():
    """加载Falcon全量数据。"""
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)

    data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_historical.json"),
        ("analyst_historical", "analyst_historical.json"),
        ("fmp_key_metrics", "fmp_key_metrics.json"),
        ("fmp_financial_growth", "fmp_financial_growth.json"),
    ]:
        f = DATA_DIR / fname
        data[name] = json.load(open(f)) if f.exists() else {}

    return master, data


def compute_daily_signals(master, data, target_date):
    """计算某日的Falcon排名，返回top-N ticker列表。"""
    dates = sorted(master["date"].unique())
    available = [d for d in dates if d <= target_date]
    if not available:
        return []
    date = available[-1]

    day = master[master["date"] == date].copy()
    if len(day) < 10:
        return []
    day.index = day["ticker"].values

    # FMP Ratios rank
    r_vals = {}
    for t in day["ticker"].values:
        pit = get_pit(data.get("fmp_ratios_historical", {}).get(t, []), date)
        scores = []
        for f in RATIO_FIELDS:
            v = pit.get(f)
            if v is not None:
                scores.append(v)
        if scores:
            r_vals[t] = np.mean(scores)

    # Key Metrics rank
    m_vals = {}
    for t in day["ticker"].values:
        pit = get_pit(data.get("fmp_key_metrics", {}).get(t, []), date)
        scores = []
        for f in METRIC_FIELDS:
            v = pit.get(f)
            if v is not None:
                scores.append(v)
        if scores:
            m_vals[t] = np.mean(scores)

    # Analyst rank
    a_vals = {}
    for t in day["ticker"].values:
        pit = get_pit(data.get("analyst_historical", {}).get(t, []), date)
        scores = []
        for f in ANALYST_FIELDS:
            v = pit.get(f)
            if v is not None:
                scores.append(v)
        if scores:
            a_vals[t] = np.mean(scores)

    # Rank each group
    r_rank = pd.Series(r_vals).rank(pct=True) if r_vals else pd.Series(dtype=float)
    m_rank = pd.Series(m_vals).rank(pct=True) if m_vals else pd.Series(dtype=float)
    a_rank = pd.Series(a_vals).rank(pct=True) if a_vals else pd.Series(dtype=float)

    # Combine
    combined = pd.Series(dtype=float)
    all_tickers = set(r_rank.index) | set(m_rank.index) | set(a_rank.index)
    for t in all_tickers:
        score = (WEIGHTS["fund_ratio"] * r_rank.get(t, 0.5) +
                 WEIGHTS["fund_metric"] * m_rank.get(t, 0.5) +
                 WEIGHTS["analyst"] * a_rank.get(t, 0.5))
        combined[t] = score

    return combined.sort_values(ascending=False).index.tolist()


# ═══════════════════════════════════════════
# Step 2: 拉取分钟K线
# ═══════════════════════════════════════════
def fetch_minute_bars(client, symbols, start, end):
    """批量拉取分钟K线。Alpaca每次最多~20只。"""
    all_bars = {}
    batch_size = 15  # 保守

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = client.get_stock_bars(req)
            df = bars.df
            for sym in batch:
                if sym in df.index.get_level_values(0):
                    sub = df.loc[sym].copy()
                    sub.index = pd.to_datetime(sub.index)
                    all_bars[sym] = sub
        except Exception as e:
            print(f"  ⚠️ 批次 {batch[:3]}... 失败: {e}")
        time.sleep(0.3)  # 限流

    return all_bars


# ═══════════════════════════════════════════
# Step 3: 入场评分
# ═══════════════════════════════════════════
def compute_entry_score(bar_idx, bars_df, lookback=5):
    """
    计算入场评分 (0-100)。
    bar_idx: 当前bar在df中的位置
    bars_df: 当天的分钟K线
    """
    if bar_idx < lookback:
        return 50  # 数据不足，给中性分

    row = bars_df.iloc[bar_idx]
    price = row["close"]
    vwap = row["vwap"]

    # 当天数据 (从开盘到现在)
    day_bars = bars_df.iloc[:bar_idx + 1]
    day_high = day_bars["high"].max()
    day_low = day_bars["low"].min()
    day_open = day_bars.iloc[0]["open"]

    if day_high == day_low:
        return 50  # 没有波动

    # 1. 价格位置 (0=最低, 1=最高) → 越低越好
    price_pos = (price - day_low) / (day_high - day_low)
    price_score = max(0, min(100, (1 - price_pos) * 100))  # 反转：低=高分

    # 2. VWAP偏离 → 低于VWAP越好
    vwap_dev = (price - vwap) / vwap if vwap > 0 else 0
    vwap_score = max(0, min(100, (0.01 - vwap_dev) * 5000))  # -1%→100, +1%→0
    vwap_score = max(0, min(100, vwap_score))

    # 3. 成交量确认 → 相对今天之前的量
    avg_vol = day_bars["volume"].mean()
    recent_vol = day_bars.iloc[-lookback:]["volume"].mean()
    vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0
    vol_score = max(0, min(100, vol_ratio * 50))  # 2x均量→100

    # 4. 分钟动量 → 最近5根bar方向
    if bar_idx >= lookback:
        price_5ago = bars_df.iloc[bar_idx - lookback]["close"]
        mom = (price - price_5ago) / price_5ago if price_5ago > 0 else 0
        # 企稳/小幅回升好于急跌
        mom_score = max(0, min(100, (mom + 0.01) * 5000))
    else:
        mom_score = 50

    # 5. 交易密度 (trade_count近似流动性)
    tc = row.get("trade_count", 100)
    avg_tc = day_bars["trade_count"].mean() if "trade_count" in day_bars.columns else 100
    tc_ratio = tc / avg_tc if avg_tc > 0 else 1.0
    spread_score = max(0, min(100, tc_ratio * 50))

    # 加权
    score = (ENTRY_WEIGHTS["price_position"] * price_score +
             ENTRY_WEIGHTS["vwap_deviation"] * vwap_score +
             ENTRY_WEIGHTS["volume_confirm"] * vol_score +
             ENTRY_WEIGHTS["momentum"] * mom_score +
             ENTRY_WEIGHTS["spread_quality"] * spread_score)

    return round(score, 1)


# ═══════════════════════════════════════════
# Step 4: 入场策略模拟
# ═══════════════════════════════════════════
def simulate_baseline(day_bars):
    """基线: 开盘价全仓。"""
    open_price = day_bars.iloc[0]["open"]
    return {
        "strategy": "baseline_open",
        "entries": [{"time": day_bars.index[0], "price": open_price, "pct": 1.0}],
        "avg_price": open_price,
        "entry_score": None,
    }


def simulate_vwap_timing(day_bars, max_wait_bars=60, fallback_pct=0.7):
    """
    VWAP择时: 等价格跌破VWAP后买入。
    如果到max_wait_bars都没跌破，在fallback_pct时间点用市价买入。
    """
    entry_price = None
    entry_time = None
    entry_bar = None

    # 只看开盘后前4小时 (240 bars), 排除盘前和收盘后
    market_bars = day_bars.iloc[:240] if len(day_bars) > 240 else day_bars

    for i in range(min(5, len(market_bars)), len(market_bars)):
        row = market_bars.iloc[i]
        # 价格跌破VWAP
        if row["close"] < row["vwap"]:
            entry_price = row["close"]
            entry_time = market_bars.index[i]
            entry_bar = i
            break

    # Fallback: 如果始终没跌破VWAP
    if entry_price is None:
        fallback_bar = min(int(len(market_bars) * fallback_pct), len(market_bars) - 1)
        entry_price = market_bars.iloc[fallback_bar]["close"]
        entry_time = market_bars.index[fallback_bar]
        entry_bar = fallback_bar

    return {
        "strategy": "vwap_timing",
        "entries": [{"time": entry_time, "price": entry_price, "pct": 1.0}],
        "avg_price": entry_price,
        "entry_bar": entry_bar,
    }


def simulate_phased_entry(day_bars):
    """
    分批建仓: 基于入场评分分3批。
    批次1 (30%): score ≥ 60
    批次2 (40%): score 持续 ≥ 50, +30min后
    批次3 (30%): 确认趋势, 收盘前1小时
    """
    market_bars = day_bars.iloc[:240] if len(day_bars) > 240 else day_bars
    entries = []
    batch1_done = False
    batch2_done = False
    batch2_eligible_bar = None

    for i in range(min(5, len(market_bars)), len(market_bars)):
        score = compute_entry_score(i, market_bars)

        # 批次1: score ≥ 60 → 30%仓位
        if not batch1_done and score >= 60:
            entries.append({
                "time": market_bars.index[i],
                "price": market_bars.iloc[i]["close"],
                "pct": 0.3,
                "score": score,
                "batch": 1,
            })
            batch1_done = True
            batch2_eligible_bar = i + 30  # 30分钟后才能批次2
            continue

        # 批次2: 批次1完成后30分钟 + score ≥ 50 → 40%仓位
        if batch1_done and not batch2_done and batch2_eligible_bar:
            if i >= batch2_eligible_bar and score >= 50:
                entries.append({
                    "time": market_bars.index[i],
                    "price": market_bars.iloc[i]["close"],
                    "pct": 0.4,
                    "score": score,
                    "batch": 2,
                })
                batch2_done = True
                continue

        # 批次3: 批次2完成 + 确认趋势(价格在VWAP上方) → 30%仓位
        if batch2_done and len(entries) == 2:
            row = market_bars.iloc[i]
            if row["close"] > row["vwap"]:
                entries.append({
                    "time": market_bars.index[i],
                    "price": market_bars.iloc[i]["close"],
                    "pct": 0.3,
                    "score": score,
                    "batch": 3,
                })
                break

    # 如果批次没完成，用收盘价补齐
    if not entries:
        # 批次1都没触发 → 用收盘价全仓
        close_price = market_bars.iloc[-1]["close"]
        entries = [{"time": market_bars.index[-1], "price": close_price,
                     "pct": 1.0, "score": 0, "batch": 1}]
    elif len(entries) == 1:
        # 批次2/3没触发 → 用当前收盘价补齐剩余70%
        close_price = market_bars.iloc[-1]["close"]
        remaining = 1.0 - entries[0]["pct"]
        entries.append({"time": market_bars.index[-1], "price": close_price,
                         "pct": remaining, "score": 0, "batch": "fallback"})
    elif len(entries) == 2:
        close_price = market_bars.iloc[-1]["close"]
        remaining = 1.0 - sum(e["pct"] for e in entries)
        if remaining > 0.01:
            entries.append({"time": market_bars.index[-1], "price": close_price,
                             "pct": remaining, "score": 0, "batch": "fallback"})

    # 加权平均入场价
    total_pct = sum(e["pct"] for e in entries)
    avg_price = sum(e["price"] * e["pct"] for e in entries) / total_pct if total_pct > 0 else 0

    return {
        "strategy": "phased_entry",
        "entries": entries,
        "avg_price": avg_price,
        "n_batches": len([e for e in entries if isinstance(e.get("batch"), int)]),
    }


# ═══════════════════════════════════════════
# Step 5: 计算持有期收益
# ═══════════════════════════════════════════
def compute_hold_return(entry_price, future_bars, hold_days=30, stop_loss=-0.15):
    """
    给定入场价，计算hold_days后的出场价和收益。
    简化: 用hold_days后的日频收盘价。
    future_bars: 未来N天的分钟K线 (可能跨多天)
    """
    if future_bars is None or len(future_bars) == 0:
        return None

    # 按天分组
    future_bars = future_bars.copy()
    future_bars["date"] = future_bars.index.date
    trading_days = sorted(future_bars["date"].unique())

    if len(trading_days) < 1:
        return None

    # 止损检查: 逐bar检查
    for _, bar in future_bars.iterrows():
        pnl = (bar["low"] - entry_price) / entry_price
        if pnl <= stop_loss:
            exit_price = entry_price * (1 + stop_loss + 0.005)  # 止损滑点0.5%
            return {
                "exit_price": exit_price,
                "return_pct": stop_loss + 0.005,
                "hold_days_actual": 0,
                "exit_reason": "stop_loss",
            }

    # 到期退出: 用第hold_days天的收盘价
    exit_day_idx = min(hold_days - 1, len(trading_days) - 1)
    exit_day = trading_days[exit_day_idx]
    exit_bars = future_bars[future_bars["date"] == exit_day]
    if len(exit_bars) > 0:
        exit_price = exit_bars.iloc[-1]["close"]  # 最后一根bar
    else:
        exit_price = entry_price

    ret = (exit_price - entry_price) / entry_price
    return {
        "exit_price": exit_price,
        "return_pct": ret,
        "hold_days_actual": exit_day_idx + 1,
        "exit_reason": "hold_expire",
    }


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Falcon 分钟级择时回测")
    parser.add_argument("--months", type=int, default=1, help="回测月数 (默认1)")
    parser.add_argument("--top-n", type=int, default=5, help="每日选股数 (默认5)")
    parser.add_argument("--hold", type=int, default=30, help="持有天数 (默认30)")
    parser.add_argument("--stop-loss", type=float, default=-0.15, help="止损 (默认-15%%)")
    args = parser.parse_args()

    print("🦅 Falcon 分钟级择时回测")
    print("=" * 70)

    # ── 1. 加载Falcon数据 ──
    print("\n📊 加载Falcon数据...")
    master, falcon_data = load_falcon_data()
    all_dates = sorted(master["date"].unique())
    print(f"  ✅ {master['ticker'].nunique()} 只, {len(all_dates)} 天")

    # ── 2. 确定回测日期范围 ──
    # Falcon数据到2024-12-31, Alpaca有到2025年1月的分钟数据
    # 信号日取数据末尾N个交易日, 收益通过Alpaca分钟数据计算(不需要master data)
    n_signal_days = args.months * 22  # 每月约22个交易日
    signal_dates = all_dates[-n_signal_days:]
    print(f"\n📅 回测期间: {signal_dates[0]} → {signal_dates[-1]} ({len(signal_dates)} 个信号日)")

    # ── 3. 生成全部Falcon信号 ──
    print("\n🔍 生成Falcon日频信号...")
    daily_picks = {}
    all_tickers = set()
    for d in signal_dates:
        picks = compute_daily_signals(master, falcon_data, d)[:args.top_n]
        daily_picks[d] = picks
        all_tickers.update(picks)

    print(f"  ✅ {len(daily_picks)} 天, {len(all_tickers)} 只不重复股票")

    # ── 4. 拉取分钟K线 ──
    print(f"\n📡 拉取分钟K线 (Alpaca API)...")
    alpaca_key = os.environ.get("APCA_API_KEY_ID", "")
    alpaca_secret = os.environ.get("APCA_API_SECRET_KEY", "")
    client = StockHistoricalDataClient(alpaca_key, alpaca_secret)

    # 拉取整个回测期间的分钟数据 (signal_dates前后各加buffer)
    fetch_start = datetime.strptime(signal_dates[0], "%Y-%m-%d") - timedelta(days=3)
    fetch_end = datetime.strptime(signal_dates[-1], "%Y-%m-%d") + timedelta(days=args.hold + 10)
    # 确保不超出现有数据范围
    max_end = datetime(2025, 1, 31)  # Alpaca数据边界
    if fetch_end > max_end:
        fetch_end = max_end

    all_tickers_list = sorted(all_tickers)
    print(f"  需要拉取 {len(all_tickers_list)} 只股票, "
          f"{fetch_start.strftime('%Y-%m-%d')} → {fetch_end.strftime('%Y-%m-%d')}")

    minute_data = fetch_minute_bars(client, all_tickers_list, fetch_start, fetch_end)
    print(f"  ✅ 成功 {len(minute_data)}/{len(all_tickers_list)} 只")

    # ── 5. 模拟三种策略 ──
    print(f"\n⏱️ 模拟入场策略 (hold={args.hold}d, SL={args.stop_loss*100:.0f}%)...")
    print("-" * 70)

    results = {
        "baseline_open": [],
        "vwap_timing": [],
        "phased_entry": [],
    }
    daily_details = []

    for date in signal_dates:
        picks = daily_picks.get(date, [])
        if not picks:
            continue

        date_dt = datetime.strptime(date, "%Y-%m-%d")
        day_str = date_dt.strftime("%Y-%m-%d")

        trade_record = {"date": date, "stocks": []}

        for ticker in picks:
            if ticker not in minute_data:
                continue

            bars = minute_data[ticker]
            # 当天的bars (转为美东时间)
            bars_et = bars.tz_convert("US/Eastern") if bars.index.tz else bars.tz_localize("UTC").tz_convert("US/Eastern")
            day_bars = bars_et[bars_et.index.strftime("%Y-%m-%d") == day_str]

            if len(day_bars) < 30:
                continue  # 数据不足

            # 未来的bars (计算收益用)
            future_start = date_dt + timedelta(days=1)
            future_end = date_dt + timedelta(days=args.hold + 5)
            future_mask = (bars.index >= pd.Timestamp(future_start, tz="UTC")) & \
                          (bars.index <= pd.Timestamp(future_end, tz="UTC"))
            future_bars = bars[future_mask]

            # 模拟三种入场
            baseline = simulate_baseline(day_bars)
            vwap = simulate_vwap_timing(day_bars)
            phased = simulate_phased_entry(day_bars)

            # 计算每种策略的收益
            stock_result = {"ticker": ticker, "date": date}

            for strat_name, strat_result in [
                ("baseline_open", baseline),
                ("vwap_timing", vwap),
                ("phased_entry", phased),
            ]:
                hold_result = compute_hold_return(
                    strat_result["avg_price"], future_bars,
                    hold_days=args.hold, stop_loss=args.stop_loss,
                )
                if hold_result:
                    # 扣除交易成本
                    net_ret = hold_result["return_pct"] - COST_PER_TRADE * 2  # 双边
                    entry = strat_result["entries"][0] if strat_result["entries"] else {}
                    stock_result[strat_name] = {
                        "avg_entry": round(strat_result["avg_price"], 2),
                        "exit": round(hold_result["exit_price"], 2),
                        "gross_ret": round(hold_result["return_pct"] * 100, 2),
                        "net_ret": round(net_ret * 100, 2),
                        "exit_reason": hold_result["exit_reason"],
                    }
                    results[strat_name].append(net_ret)
                else:
                    stock_result[strat_name] = None

            trade_record["stocks"].append(stock_result)

        daily_details.append(trade_record)

    # ── 6. 统计 ──
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)

    def calc_stats(returns, name):
        if not returns:
            return {"name": name, "trades": 0}
        r = np.array(returns)
        wins = (r > 0).sum()
        losses = (r < 0).sum()
        avg = r.mean() * 100
        med = np.median(r) * 100
        std = r.std() * 100
        total = ((1 + r).prod() - 1) * 100
        # 按天聚合的Sharpe (简化)
        daily_rets = []
        for d in daily_details:
            day_r = [s.get(name, {}).get("net_ret", 0) / 100
                     for s in d["stocks"] if s.get(name)]
            if day_r:
                daily_rets.append(np.mean(day_r))
        if daily_rets:
            dr = np.array(daily_rets)
            sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        else:
            sharpe = 0

        return {
            "name": name,
            "trades": len(r),
            "win_rate": round(wins / len(r) * 100, 1),
            "avg_ret": round(avg, 2),
            "median_ret": round(med, 2),
            "std": round(std, 2),
            "total_ret": round(total, 2),
            "sharpe": round(sharpe, 3),
            "avg_win": round(r[r > 0].mean() * 100, 2) if wins > 0 else 0,
            "avg_loss": round(r[r < 0].mean() * 100, 2) if losses > 0 else 0,
        }

    stats = {}
    for strat_name in ["baseline_open", "vwap_timing", "phased_entry"]:
        stats[strat_name] = calc_stats(results[strat_name], strat_name)

    # 打印对比表
    header = f"{'指标':<16} {'开盘全仓':>12} {'VWAP择时':>12} {'分批建仓':>12}"
    print(header)
    print("-" * 60)

    metrics = [
        ("交易次数", "trades", "d"),
        ("胜率%", "win_rate", ".1f"),
        ("平均收益%", "avg_ret", ".2f"),
        ("中位收益%", "median_ret", ".2f"),
        ("收益波动%", "std", ".2f"),
        ("累计收益%", "total_ret", ".2f"),
        ("Sharpe", "sharpe", ".3f"),
        ("平均盈利%", "avg_win", ".2f"),
        ("平均亏损%", "avg_loss", ".2f"),
    ]

    for label, key, fmt in metrics:
        vals = []
        for s in ["baseline_open", "vwap_timing", "phased_entry"]:
            v = stats[s].get(key, 0)
            vals.append(f"{v:{fmt}}")
        print(f"{label:<16} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")

    # ── 7. 择时增益分析 ──
    print("\n" + "=" * 70)
    print("📈 择时增益 (vs 基线)")
    print("=" * 70)

    for strat in ["vwap_timing", "phased_entry"]:
        s = stats[strat]
        b = stats["baseline_open"]
        if s["trades"] > 0 and b["trades"] > 0:
            entry_gain = b["avg_ret"] - s["avg_ret"]  # 负=择时入场更好
            # 实际计算: 对比每笔交易的入场价差
            entry_diffs = []
            for d in daily_details:
                for stock in d["stocks"]:
                    base = stock.get("baseline_open", {})
                    comp = stock.get(strat, {})
                    if base and comp:
                        diff = (base["avg_entry"] - comp["avg_entry"]) / base["avg_entry"] * 100
                        entry_diffs.append(diff)

            if entry_diffs:
                ed = np.array(entry_diffs)
                print(f"\n  {strat}:")
                print(f"    入场价差: 均值 {ed.mean():+.2f}%, 中位 {np.median(ed):+.2f}%")
                print(f"    (正=择时入场更便宜)")
                print(f"    Sharpe变化: {b['sharpe']:.3f} → {s['sharpe']:.3f}")

    # ── 8. 入场评分分布 ──
    if any(s["name"] == "phased_entry" and s["trades"] > 0 for s in stats.values()):
        print("\n" + "=" * 70)
        print("🎯 入场评分分布 (分批建仓策略)")
        print("=" * 70)

        all_scores = []
        for d in daily_details:
            for stock in d["stocks"]:
                pe = stock.get("phased_entry")
                if pe and "entries" in pe:
                    for e in pe["entries"]:
                        if "score" in e:
                            all_scores.append(e["score"])

        if all_scores:
            sc = np.array(all_scores)
            print(f"  评分均值: {sc.mean():.1f}")
            print(f"  评分中位: {np.median(sc):.1f}")
            print(f"  ≥70 (直接入场): {(sc >= 70).sum()} / {len(sc)} ({(sc >= 70).mean()*100:.0f}%)")
            print(f"  60-70 (等待): {((sc >= 60) & (sc < 70)).sum()} / {len(sc)}")
            print(f"  <60 (放弃): {(sc < 60).sum()} / {len(sc)}")

    # ── 9. 保存详细结果 ──
    output = {
        "config": {
            "months": args.months,
            "top_n": args.top_n,
            "hold_days": args.hold,
            "stop_loss": args.stop_loss,
            "signal_dates": [signal_dates[0], signal_dates[-1]],
            "n_stocks": len(all_tickers),
        },
        "summary": stats,
        "daily": daily_details,
    }

    out_file = DATA_DIR / "timing_backtest_result.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n📁 详细结果: {out_file}")


if __name__ == "__main__":
    main()
