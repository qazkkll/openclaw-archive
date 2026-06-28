#!/usr/bin/env python3
"""
🦅 Falcon 2026年1-6月逐月回测
$100万初始资金, Top5选股, 30天持有, -15%止损
"""
import pandas as pd, numpy as np, json, sys, time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible, futu_cost

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")

# ═══════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════
INITIAL_CAPITAL = 1_000_000.0
WEIGHTS = {"fund_ratio": 0.70, "analyst": 0.20, "fund_metric": 0.10}
PARAMS = {"hold_days": 30, "stop_loss": -0.15, "bear_alloc": 0.50, "cost": 0.001}
TOP_N = 5
START_DATE = "2026-01-02"
END_DATE = "2026-06-26"

# ═══════════════════════════════════════════════════
# 加载数据
# ═══════════════════════════════════════════════════
print("=" * 80)
print("🦅 Falcon 2026年1-6月 回测")
print(f"   初始资金: ${INITIAL_CAPITAL:,.0f}")
print(f"   策略: Top{TOP_N}, hold={PARAMS['hold_days']}d, SL={PARAMS['stop_loss']:.0%}")
print(f"   权重: Fund={WEIGHTS['fund_ratio']:.0%} Ana={WEIGHTS['analyst']:.0%} Metric={WEIGHTS['fund_metric']:.0%}")
print("=" * 80)

t0 = time.time()

# 加载价格数据
print("\n📊 加载价格数据...")
master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
master["date"] = master["date"].astype(str)
master = master[(master["date"] >= START_DATE) & (master["date"] <= END_DATE)]
print(f"   {master['ticker'].nunique()} 只, {master['date'].nunique()} 天")

# 价格矩阵
price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()

# SPY基准
spy_data = None
if "SPY" in price_pivot.columns:
    spy_prices = price_pivot["SPY"].dropna()
    spy_ret = spy_prices.pct_change().fillna(0)
    spy_nav = INITIAL_CAPITAL * (1 + spy_ret).cumprod()
    print(f"   SPY: {spy_prices.iloc[0]:.2f} → {spy_prices.iloc[-1]:.2f} ({(spy_prices.iloc[-1]/spy_prices.iloc[0]-1)*100:.1f}%)")

# 加载FMP数据
print("\n📊 加载FMP基本面数据...")
fmp_data = {}
for name, fname in [
    ("fmp_ratios_historical", "fmp_ratios_historical.json"),
    ("analyst_historical", "analyst_historical.json"),
    ("fmp_key_metrics", "fmp_key_metrics.json"),
    ("fmp_financial_growth", "fmp_financial_growth.json"),
]:
    f = DATA_DIR / fname
    fmp_data[name] = json.load(open(f)) if f.exists() else {}
    print(f"   ✅ {name}: {len(fmp_data[name])} tickers")

fmp_data["fmp_insider"] = {}
fmp_data["fmp_dcf"] = {}
fmp_data["fmp_price_target"] = {}

# ═══════════════════════════════════════════════════
# 预计算PIT rank
# ═══════════════════════════════════════════════════
print("\n📊 预计算PIT rank...")
ranks_dict = precompute_pit_ranks(
    master,
    fmp_data["fmp_ratios_historical"],
    fmp_data["analyst_historical"],
    fmp_data["fmp_key_metrics"],
    fmp_data["fmp_financial_growth"],
    fmp_data["fmp_insider"],
    fmp_data["fmp_dcf"],
    fmp_data["fmp_price_target"],
)

# Regime: SPX等权MA200
dates_sorted = sorted(ranks_dict.keys())
all_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
all_price = (1 + all_ret).cumprod()
ma200 = all_price.rolling(200, min_periods=100).mean()
regime_above = (all_price > ma200).astype(int)

# ═══════════════════════════════════════════════════
# 回测引擎（改进版，按月报告）
# ═══════════════════════════════════════════════════
print("\n📊 运行回测...")

cash = INITIAL_CAPITAL
portfolio = {}  # ticker -> (entry_idx, entry_price, shares)
trades = []
daily_nav = []
monthly_snapshots = []

def get_scores(date):
    if date not in ranks_dict:
        return None
    r = ranks_dict[date]
    combined = sum(w * r[f] for f, w in WEIGHTS.items() if f in r.columns)
    return combined.dropna().sort_values(ascending=False)

for i, date in enumerate(dates_sorted):
    if date not in price_pivot.index:
        continue
    pr = price_pivot.loc[date]
    above = regime_above.loc[date] if date in regime_above.index else 1
    alloc = PARAMS["bear_alloc"] if above == 0 else 1.0

    # ── 止损 ──
    to_close = []
    for t, (ei, ep, sh) in portfolio.items():
        if t in pr and not pd.isna(pr[t]):
            pnl_pct = (pr[t] - ep) / ep
            if pnl_pct <= PARAMS["stop_loss"]:
                sell_price = pr[t]
                cost = futu_cost(sell_price, "sell")
                proceeds = sh * sell_price * (1 - cost)
                cash += proceeds
                trades.append({
                    "date": date, "ticker": t, "action": "止损",
                    "entry_price": ep, "exit_price": sell_price,
                    "pnl_pct": pnl_pct, "pnl_usd": proceeds - sh * ep,
                    "days_held": i - ei,
                })
                to_close.append(t)
    for t in to_close:
        del portfolio[t]

    # ── 调仓（固定30天） ──
    hold_days = PARAMS["hold_days"]
    sell_tickers = [t for t, (ei, _, _) in portfolio.items() if (i - ei) >= hold_days]

    for t in sell_tickers:
        if t in portfolio and t in pr and not pd.isna(pr[t]):
            _, ep, sh = portfolio.pop(t)
            sell_price = pr[t]
            cost = futu_cost(sell_price, "sell")
            proceeds = sh * sell_price * (1 - cost)
            cash += proceeds
            pnl_pct = (sell_price - ep) / ep
            trades.append({
                "date": date, "ticker": t, "action": "到期",
                "entry_price": ep, "exit_price": sell_price,
                "pnl_pct": pnl_pct, "pnl_usd": proceeds - sh * ep,
                "days_held": hold_days,
            })

    # ── 买入 ──
    if (sell_tickers or len(portfolio) == 0) and cash > 1000:
        scores = get_scores(date)
        if scores is not None:
            deploy = cash * alloc
            existing = set(portfolio.keys())
            candidates = [t for t in scores.head(TOP_N * 2).index if t not in existing]
            picks = candidates[:TOP_N]
            if picks:
                per_stock = deploy / len(picks)
                for t in picks:
                    if t in pr and not pd.isna(pr[t]) and pr[t] > 0:
                        buy_price = pr[t]
                        cost = futu_cost(buy_price, "buy")
                        shares = (per_stock * (1 - cost)) / buy_price
                        portfolio[t] = (i, buy_price, shares)
                        trades.append({
                            "date": date, "ticker": t, "action": "买入",
                            "entry_price": buy_price, "exit_price": 0,
                            "pnl_pct": 0, "pnl_usd": 0, "days_held": 0,
                        })
                cash -= deploy

    # ── 每日NAV ──
    port_value = sum(
        sh * (pr[t] if t in pr and not pd.isna(pr[t]) else ep)
        for t, (_, ep, sh) in portfolio.items()
    )
    nav = cash + port_value
    daily_nav.append({"date": date, "nav": nav, "cash": cash, "positions": len(portfolio)})

    # ── 月末快照 ──
    if date[5:7] != (dates_sorted[i+1] if i+1 < len(dates_sorted) else "")[5:7]:
        monthly_snapshots.append({
            "month": date[:7],
            "nav": nav,
            "return_pct": (nav / INITIAL_CAPITAL - 1) * 100,
            "positions": len(portfolio),
            "cash_pct": cash / nav * 100,
        })

# ═══════════════════════════════════════════════════
# 结果分析
# ═══════════════════════════════════════════════════
elapsed = time.time() - t0
nav_df = pd.DataFrame(daily_nav)
nav_df["date"] = pd.to_datetime(nav_df["date"])
nav_df = nav_df.set_index("date")

# 基本指标
final_nav = nav_df["nav"].iloc[-1]
total_return = (final_nav / INITIAL_CAPITAL - 1) * 100
daily_ret = nav_df["nav"].pct_change().dropna()
sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0
peak = nav_df["nav"].cummax()
drawdown = (nav_df["nav"] - peak) / peak
max_dd = drawdown.min() * 100

# 交易统计
buy_trades = [t for t in trades if t["action"] == "买入"]
sell_trades = [t for t in trades if t["action"] in ("到期", "止损")]
win_trades = [t for t in sell_trades if t["pnl_pct"] > 0]
loss_trades = [t for t in sell_trades if t["pnl_pct"] <= 0]
stop_loss_trades = [t for t in sell_trades if t["action"] == "止损"]

win_rate = len(win_trades) / len(sell_trades) * 100 if sell_trades else 0
avg_win = np.mean([t["pnl_pct"] for t in win_trades]) * 100 if win_trades else 0
avg_loss = np.mean([t["pnl_pct"] for t in loss_trades]) * 100 if loss_trades else 0
avg_hold = np.mean([t["days_held"] for t in sell_trades]) if sell_trades else 0

# SPY对比
spy_total = 0
if "SPY" in price_pivot.columns:
    spy_start = price_pivot["SPY"].dropna().iloc[0]
    spy_end = price_pivot["SPY"].dropna().iloc[-1]
    spy_total = (spy_end / spy_start - 1) * 100

# ═══════════════════════════════════════════════════
# 输出报告
# ═══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("📊 Falcon 2026年1-6月回测结果")
print("=" * 80)

print(f"\n💰 总体表现")
print(f"   初始资金:   ${INITIAL_CAPITAL:>12,.0f}")
print(f"   最终净值:   ${final_nav:>12,.0f}")
print(f"   总收益:     {total_return:>+10.2f}%")
print(f"   年化Sharpe: {sharpe:>10.2f}")
print(f"   最大回撤:   {max_dd:>10.2f}%")
print(f"   SPY同期:    {spy_total:>+10.2f}%")
print(f"   超额收益:   {total_return - spy_total:>+10.2f}%")

print(f"\n📈 交易统计")
print(f"   买入次数:   {len(buy_trades):>6}")
print(f"   卖出次数:   {len(sell_trades):>6}")
print(f"   胜率:       {win_rate:>6.1f}%")
print(f"   平均盈利:   {avg_win:>+6.2f}%")
print(f"   平均亏损:   {avg_loss:>+6.2f}%")
print(f"   平均持仓:   {avg_hold:>6.1f}天")
print(f"   止损次数:   {len(stop_loss_trades):>6} ({len(stop_loss_trades)/len(sell_trades)*100:.1f}% of sells)")

print(f"\n📅 逐月表现")
print(f"   {'月份':<10} {'净值':>12} {'月收益':>10} {'累计收益':>10} {'持仓':>6} {'现金%':>8}")
print(f"   {'-'*58}")
prev_nav = INITIAL_CAPITAL
for snap in monthly_snapshots:
    month_ret = (snap["nav"] / prev_nav - 1) * 100
    print(f"   {snap['month']:<10} ${snap['nav']:>11,.0f} {month_ret:>+9.2f}% {snap['return_pct']:>+9.2f}% {snap['positions']:>5} {snap['cash_pct']:>7.1f}%")
    prev_nav = snap["nav"]

# Top10交易
print(f"\n🏆 Top10盈利交易")
profitable = sorted([t for t in sell_trades if t["pnl_pct"] > 0], key=lambda x: x["pnl_pct"], reverse=True)[:10]
for t in profitable:
    print(f"   {t['date']} {t['ticker']:<6} {t['pnl_pct']*100:>+6.2f}% ${t['pnl_usd']:>+10,.0f} ({t['days_held']}d) {t['action']}")

print(f"\n💀 Top10亏损交易")
losing = sorted([t for t in sell_trades if t["pnl_pct"] < 0], key=lambda x: x["pnl_pct"])[:10]
for t in losing:
    print(f"   {t['date']} {t['ticker']:<6} {t['pnl_pct']*100:>+6.2f}% ${t['pnl_usd']:>+10,.0f} ({t['days_held']}d) {t['action']}")

# 持仓快照
print(f"\n📋 当前持仓 ({len(portfolio)}只)")
current_date = dates_sorted[-1]
pr = price_pivot.loc[current_date]
for t, (ei, ep, sh) in sorted(portfolio.items(), key=lambda x: x[1][0]):
    cur = pr[t] if t in pr and not pd.isna(pr[t]) else ep
    pnl = (cur - ep) / ep * 100
    val = sh * cur
    print(f"   {t:<6} 入${ep:>8.2f} → 现${cur:>8.2f} ({pnl:>+6.2f}%) × {sh:.0f}股 = ${val:>10,.0f}")

print(f"\n⏱️ 耗时: {elapsed:.0f}秒")

# 保存结果
output = {
    "config": {"initial_capital": INITIAL_CAPITAL, "weights": WEIGHTS, "params": PARAMS, "top_n": TOP_N},
    "performance": {
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "spy_return_pct": round(spy_total, 2),
        "excess_return_pct": round(total_return - spy_total, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": len(sell_trades),
        "stop_losses": len(stop_loss_trades),
    },
    "monthly": monthly_snapshots,
    "final_nav": round(final_nav, 2),
}
output_path = DATA_DIR / "backtest_2026_h1.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\n💾 结果已保存: {output_path}")
