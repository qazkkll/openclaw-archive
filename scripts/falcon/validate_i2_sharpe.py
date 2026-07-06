#!/usr/bin/env python3
"""I2 Sharpe验证 — 快速版(最近2年, ~100评分日)"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from falcon_system.core.data_manager import DataManager
from falcon_system.engine.scorer import ScoringEngine

t0 = time.time()
dm = DataManager()
engine = ScoringEngine(dm)
engine._override_with_realtime_prices = lambda signals: None

master = dm.load_master_prices()
price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
all_dates = sorted(master["date"].unique())

vix = pd.read_parquet("data/us/vix_10y.parquet")
vix_series = vix.set_index("date")["close"]
vix_series.index = pd.to_datetime(vix_series.index.astype(str))

# 最近2年每5天
score_dates = all_dates[-500::5]
print(f"评分日期: {len(score_dates)}个 ({score_dates[0]}~{score_dates[-1]})")

regime_pnls = {"bull": [], "neutral": [], "bear": [], "extreme_bear": []}

for i, sdate in enumerate(score_dates):
    try:
        result = engine.score(target_date=sdate, universe="spx")
    except:
        continue
    vix_dt = pd.to_datetime(sdate)
    nearby = vix_series.loc[vix_series.index <= vix_dt]
    if len(nearby)==0: continue
    vix_val = float(nearby.iloc[-1])
    if vix_val<20: regime="bull"
    elif vix_val<25: regime="neutral"
    elif vix_val<30: regime="bear"
    else: regime="extreme_bear"
    
    top5 = sorted(result.signals, key=lambda x: x.score, reverse=True)[:5]
    for sig in top5:
        if sig.ticker not in price_pivot.columns or sdate not in price_pivot.index:
            continue
        p0 = price_pivot[sig.ticker].loc[sdate]
        if pd.isna(p0) or p0<=0: continue
        future = price_pivot[sig.ticker].loc[price_pivot.index > sdate].dropna()
        if len(future)>=30:
            pnl = (future.iloc[29]-p0)/p0
            regime_pnls[regime].append(pnl)
    if (i+1)%20==0:
        print(f"  [{i+1}/{len(score_dates)}] {time.time()-t0:.0f}s")

elapsed = time.time()-t0
total = sum(len(v) for v in regime_pnls.values())
print(f"\n完成: {elapsed:.0f}s, {total}样本")

print(f"\n{'='*75}")
print(f"I2 Sharpe验证 — 单只股票30天持有")
print("="*75)
print(f"{'Regime':>15s} | {'n':>5s} | {'均值':>8s} | {'std':>8s} | {'Sharpe':>8s} | {'年化Sharpe':>10s} | {'胜率':>6s} | {'盈亏比':>6s}")
print("-"*80)

for regime, label in [("bull","🟢Bull(<20)"),("neutral","🟡Neutral(20-25)"),("bear","🟠Bear(25-30)"),("extreme_bear","🔴Extreme(>30)")]:
    pnls = np.array(regime_pnls[regime])
    if len(pnls)==0:
        print(f"  {label:15s} |     0 | N/A"); continue
    m = pnls.mean(); s = pnls.std()
    sh = m/s if s>0 else 0
    ann = sh * np.sqrt(12)
    win = (pnls>0).mean()
    w = pnls[pnls>0].mean() if (pnls>0).any() else 0
    l = pnls[pnls<0].mean() if (pnls<0).any() else -1
    wl = abs(w/l) if l!=0 else np.nan
    print(f"  {label:15s} | {len(pnls):5d} | {m:+8.2%} | {s:8.2%} | {sh:+8.3f} | {ann:+10.3f} | {win:+6.1%} | {wl:6.2f}")

# 等权组合(5只均值)
print(f"\n{'='*75}")
print(f"I2 Sharpe验证 — 等权组合(5只均值, 更接近实盘)")
print("="*75)
print(f"{'Regime':>15s} | {'组合数':>6s} | {'均值':>8s} | {'std':>8s} | {'Sharpe':>8s} | {'年化Sharpe':>10s}")
print("-"*70)

for regime, label in [("bull","🟢Bull(<20)"),("neutral","🟡Neutral(20-25)"),("bear","🟠Bear(25-30)"),("extreme_bear","🔴Extreme(>30)")]:
    pnls = np.array(regime_pnls[regime])
    if len(pnls)<5: continue
    n = len(pnls)//5
    port = np.array([pnls[i*5:(i+1)*5].mean() for i in range(n)])
    m = port.mean(); s = port.std()
    sh = m/s if s>0 else 0
    ann = sh * np.sqrt(12)
    print(f"  {label:15s} | {n:6d} | {m:+8.2%} | {s:8.2%} | {sh:+8.3f} | {ann:+10.3f}")

# SPY对照
spy = price_pivot.get("SPY")
if spy is not None:
    spy30 = spy.dropna().pct_change(30).dropna()
    sm=spy30.mean(); ss=spy30.std()
    print(f"\n--- SPY 30天基准 --- mean={sm:+.2%}, std={ss:.2%}, Sharpe={sm/ss:.3f}, 年化={sm/ss*np.sqrt(12):.3f}")
