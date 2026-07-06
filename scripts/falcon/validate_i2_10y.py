#!/usr/bin/env python3
"""I2: 10年VIX regime验证 — Falcon ScoringEngine全量历史评分"""
import sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from falcon_system.core.data_manager import DataManager
from falcon_system.engine.scorer import ScoringEngine

print("=" * 70)
print("I2 VIX Regime 10年验证")
print("=" * 70)
t0 = time.time()

dm = DataManager()
engine = ScoringEngine(dm)
engine._override_with_realtime_prices = lambda signals: None

master = dm.load_master_prices()
price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
all_dates = sorted(master["date"].unique())
print(f"交易日: {len(all_dates)}天 ({all_dates[0]} ~ {all_dates[-1]})")

# VIX
vix = pd.read_parquet("data/us/vix_10y.parquet")
vix_series = vix.set_index("date")["close"]
vix_series.index = pd.to_datetime(vix_series.index.astype(str))
print(f"VIX: {len(vix_series)}天")

# 10年每周评分 (~520个日期)
score_dates = all_dates[::5]
print(f"评分日期: {len(score_dates)}个 (每5天)")

regime_data = {"bull": [], "neutral": [], "bear": [], "extreme_bear": []}
errors = 0

for i, sdate in enumerate(score_dates):
    try:
        result = engine.score(target_date=sdate, universe="spx")
    except Exception:
        errors += 1
        continue

    vix_dt = pd.to_datetime(sdate)
    nearby = vix_series.loc[vix_series.index <= vix_dt]
    if len(nearby) == 0:
        continue
    vix_val = float(nearby.iloc[-1])

    if vix_val < 20:
        regime = "bull"
    elif vix_val < 25:
        regime = "neutral"
    elif vix_val < 30:
        regime = "bear"
    else:
        regime = "extreme_bear"

    top5 = sorted(result.signals, key=lambda x: x.score, reverse=True)[:5]
    for sig in top5:
        if sig.ticker not in price_pivot.columns or sdate not in price_pivot.index:
            continue
        p0 = price_pivot[sig.ticker].loc[sdate]
        if pd.isna(p0) or p0 <= 0:
            continue
        future = price_pivot[sig.ticker].loc[price_pivot.index > sdate].dropna()
        if len(future) >= 30:
            pnl_14 = (future.iloc[13] - p0) / p0
            pnl_30 = (future.iloc[29] - p0) / p0
            regime_data[regime].append({
                "date": sdate, "vix": vix_val, "ticker": sig.ticker,
                "score": sig.score, "pnl_14d": pnl_14, "pnl_30d": pnl_30,
            })
        elif len(future) >= 14:
            pnl_14 = (future.iloc[13] - p0) / p0
            regime_data[regime].append({
                "date": sdate, "vix": vix_val, "ticker": sig.ticker,
                "score": sig.score, "pnl_14d": pnl_14, "pnl_30d": np.nan,
            })

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        total = {k: len(v) for k, v in regime_data.items()}
        print(f"  [{i+1}/{len(score_dates)}] {elapsed:.0f}s | {total} | errors={errors}")

elapsed = time.time() - t0
print(f"\n完成! 耗时{elapsed:.0f}s, 评分{len(score_dates)-errors}个日期, 错误{errors}个")

# ═══ 结果 ═══
print(f"\n{'='*70}")
print("I2: VIX Regime 10年验证结果")
print("="*70)
print(f"{'Regime':>15s} | {'VIX':>8s} | {'案例':>6s} | {'14天均':>8s} | {'30天均':>8s} | {'30天胜率':>8s} | {'30天Sharpe':>10s}")
print("-" * 80)

summary = {}
for regime, label, vr in [
    ("bull", "🟢 Bull", "<20"),
    ("neutral", "🟡 Neutral", "20-25"),
    ("bear", "🟠 Bear", "25-30"),
    ("extreme_bear", "🔴 Extreme", ">30"),
]:
    d = regime_data[regime]
    if not d:
        print(f"  {label:15s} | {vr:>8s} |      0 |      N/A |      N/A |      N/A |        N/A")
        summary[regime] = {"n": 0}
        continue
    df = pd.DataFrame(d)
    a14 = df["pnl_14d"].mean()
    a30 = df["pnl_30d"].dropna().mean() if df["pnl_30d"].notna().any() else np.nan
    w30 = (df["pnl_30d"].dropna() > 0).mean() if df["pnl_30d"].notna().any() else np.nan
    s30 = a30 / df["pnl_30d"].dropna().std() if df["pnl_30d"].notna().any() and df["pnl_30d"].dropna().std() > 0 else np.nan
    vix_avg = df["vix"].mean()
    print(f"  {label:15s} | {vr:>8s} | {len(d):6d} | {a14:+8.2%} | {a30:+8.2%} | {w30:+8.1%} | {s30:+10.3f}")
    summary[regime] = {"n": len(d), "avg_14d": a14, "avg_30d": a30, "win_30d": w30, "sharpe_30d": s30, "vix_avg": vix_avg}

# 统计检验
from scipy import stats
print(f"\n--- 统计检验 ---")
for r1, r2 in [("bull", "bear"), ("bull", "extreme_bear"), ("neutral", "bear"), ("bull", "neutral")]:
    d1 = pd.DataFrame(regime_data[r1])["pnl_30d"].dropna() if regime_data[r1] else pd.Series()
    d2 = pd.DataFrame(regime_data[r2])["pnl_30d"].dropna() if regime_data[r2] else pd.Series()
    if len(d1) > 10 and len(d2) > 10:
        t, p = stats.ttest_ind(d1, d2)
        sig = "✅显著" if p < 0.05 else "⚠️不显著"
        print(f"  {sig} {r1} vs {r2}: t={t:+.2f}, p={p:.4f}, n={len(d1)} vs {len(d2)}")
    else:
        print(f"  ❌ {r1} vs {r2}: 样本不足 ({len(d1)} vs {len(d2)})")

# 保存
output = {
    "timestamp": datetime.now().isoformat(),
    "score_dates": len(score_dates),
    "elapsed_seconds": elapsed,
    "summary": summary,
    "raw_counts": {k: len(v) for k, v in regime_data.items()},
}
out_file = PROJECT_ROOT / "data" / "falcon" / "i2_vix_regime_10y_validation.json"
with open(out_file, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\n结果已保存: {out_file}")
