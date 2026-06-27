#!/usr/bin/env python3
"""
🦅 Falcon 独立评分脚本
=====================
FV0.3.1 独立评分 → 与 alpaca_trade.py / futu_trade.py 对接。
不依赖 V10/V12 的模型文件，用 Falcon 自己的 FMP PIT 因子 rank。

用法:
    python3 scripts/falcon/falcon_score.py                  # 评分最新交易日
    python3 scripts/falcon/falcon_score.py --date 2024-12-31  # 评分指定日期
    python3 scripts/falcon/falcon_score.py --top-n 10       # 取 Top-10

输出: data/falcon/falcon_scored_YYYYMMDD.json
格式: 与 V10/V12 scored JSON 兼容，alpaca_trade.py 可直接读取。
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# 路径
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
OUTPUT_DIR = DATA_DIR  # scored JSON 输出到 data/falcon/

sys.path.insert(0, str(FALCON_DIR))
from falcon_v03_engine import (
    get_pit, precompute_pit_ranks, RATIO_FIELDS, METRIC_FIELDS,
    GROWTH_FIELDS, ANALYST_FIELDS, TECH_FIELDS,
)


# ═══════════════════════════════════════════════════
# SPX 最优配置 (来自 OOS 验证)
# ═══════════════════════════════════════════════════
SPX_WEIGHTS = {
    "fund_ratio": 0.7,
    "analyst": 0.2,
    "fund_metric": 0.1,
    "tech": 0.0,  # 技术因子无增量 (V0.2.1 已验证)
}
# R2K 最优配置 (来自 OOS 验证: Pure_Fund, Fixed_10d, SL=-15%)
R2K_WEIGHTS = {
    "fund_ratio": 0.8,
    "analyst": 0.15,
    "fund_metric": 0.05,
    "tech": 0.0,
}
TOP_N = 5
HOLD_DAYS = 30


def load_spx_data():
    """加载 SPX 全量数据 + FMP 历史。"""
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


def load_r2k_data():
    """加载 R2K (Russell 2000) 数据。格式与 SPX 相同。"""
    # R2K 没有 features_v02.parquet, 需要从 russell_prices.json 构建
    prices_file = DATA_DIR / "russell_prices.json"
    if not prices_file.exists():
        print("❌ russell_prices.json 不存在, 请先运行 fetch_russell_data.py")
        return None, None

    prices = json.loads(prices_file.read_text())
    # Convert to DataFrame: {ticker: [{date, open, high, low, close, volume}, ...]}
    rows = []
    for ticker, bars in prices.items():
        for bar in bars:
            bar["ticker"] = ticker
            rows.append(bar)
    master = pd.DataFrame(rows)
    master["date"] = master["date"].astype(str).str[:10]
    print(f"  ✅ {len(prices)} 只, {len(master)} 行")

    data = {}
    for name, fname in [
        ("fmp_ratios_historical", "fmp_ratios_russell.json"),
        ("analyst_historical", "fmp_analyst_russell.json"),
        ("fmp_key_metrics", "fmp_metrics_russell.json"),
        ("fmp_financial_growth", "fmp_growth_russell.json"),
    ]:
        f = DATA_DIR / fname
        data[name] = json.loads(f.read_text()) if f.exists() else {}

    return master, data


def compute_today_rank(master, data, target_date=None, weights=None):
    """计算指定日期的截面 rank（单日，非全量预计算）。"""
    dates = sorted(master["date"].unique())
    if target_date:
        # 找到 <= target_date 的最近交易日
        available = [d for d in dates if d <= target_date]
        if not available:
            print(f"❌ 无 {target_date} 之前的交易数据")
            return None, None
        date = available[-1]
    else:
        date = dates[-1]

    day = master[master["date"] == date].copy()
    if len(day) < 10:
        print(f"❌ {date} 只有 {len(day)} 只股票，不足")
        return None, None

    day.index = day["ticker"].values
    row = day[["ticker"]].copy()

    # Tech rank
    tech_r = []
    for f in TECH_FIELDS:
        if f in day.columns and day[f].notna().sum() > 5:
            row[f"t_{f}"] = day[f].rank(pct=True)
            tech_r.append(f"t_{f}")
    row["tech"] = row[tech_r].mean(axis=1) if tech_r else 0.5

    # FMP Ratios
    for f in RATIO_FIELDS:
        vals = {}
        for t in day["ticker"].values:
            pit = get_pit(data.get("fmp_ratios_historical", {}).get(t, []), date)
            v = pit.get(f)
            if v is not None:
                vals[t] = v
        if len(vals) > 10:
            row[f"r_{f}"] = pd.Series(vals).rank(pct=True)

    # Key Metrics
    for f in METRIC_FIELDS:
        vals = {}
        for t in day["ticker"].values:
            pit = get_pit(data.get("fmp_key_metrics", {}).get(t, []), date)
            v = pit.get(f)
            if v is not None:
                vals[t] = v
        if len(vals) > 10:
            row[f"m_{f}"] = pd.Series(vals).rank(pct=True)

    # Growth
    for f in GROWTH_FIELDS:
        vals = {}
        for t in day["ticker"].values:
            pit = get_pit(data.get("fmp_financial_growth", {}).get(t, []), date)
            v = pit.get(f)
            if v is not None:
                vals[t] = v
        if len(vals) > 10:
            row[f"g_{f}"] = pd.Series(vals).rank(pct=True)

    # Analyst
    for f in ANALYST_FIELDS:
        vals = {}
        for t in day["ticker"].values:
            pit = get_pit(data.get("analyst_historical", {}).get(t, []), date)
            v = pit.get(f)
            if v is not None:
                vals[t] = v
        if len(vals) > 5:
            row[f"a_{f}"] = pd.Series(vals).rank(pct=True)

    # 分组得分
    r_cols = [c for c in row.columns if c.startswith("r_")]
    m_cols = [c for c in row.columns if c.startswith("m_")]
    g_cols = [c for c in row.columns if c.startswith("g_")]
    a_cols = [c for c in row.columns if c.startswith("a_")]

    row["fund_ratio"] = row[r_cols].mean(axis=1) if r_cols else 0.5
    row["fund_metric"] = row[m_cols].mean(axis=1) if m_cols else 0.5
    row["fund_growth"] = row[g_cols].mean(axis=1) if g_cols else 0.5
    row["analyst"] = row[a_cols].mean(axis=1) if a_cols else 0.5
    row["tech"] = row.get("tech", 0.5)

    # 加权综合分
    w = weights or SPX_WEIGHTS
    row["falcon_score"] = sum(
        w.get(f, 0) * row[f]
        for f in ["fund_ratio", "fund_metric", "fund_growth", "analyst", "tech"]
        if f in row.columns
    )

    # 排名百分位
    row["rank_pct"] = row["falcon_score"].rank(pct=True)

    # 价格
    row["close"] = day["close"].values

    return row, date


def score_to_signal(score: float, pct: float) -> str:
    """综合评分 + 百分位 → 信号等级。
    绝对门槛: 分数不够高时, 即使排名靠前也不给绿灯。
    这避免了"熊市里烂公司也拿🟢🟢"的问题。
    """
    # 绝对门槛 (基于历史OOS验证的分布)
    ABSOLUTE_HIGH = 0.65   # 综合分>0.65才算"好公司"
    ABSOLUTE_MID = 0.50    # 综合分>0.50算"及格"
    if score >= ABSOLUTE_HIGH and pct >= 0.95:
        return "🟢🟢"
    elif score >= ABSOLUTE_HIGH and pct >= 0.80:
        return "🟢"
    elif score >= ABSOLUTE_MID:
        return "🟡"
    else:
        return "🔴"


def main():
    parser = argparse.ArgumentParser(description="Falcon 独立评分")
    parser.add_argument("--date", default=None, help="评分日期 (YYYY-MM-DD), 默认最新")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"取 Top-N (默认 {TOP_N})")
    parser.add_argument("--universe", default="spx", choices=["spx", "r2k", "all"],
                        help="评分范围: spx=标普500, r2k=Russell2000, all=合并")
    parser.add_argument("--skip-freshness", action="store_true", help="跳过数据新鲜度检查")
    args = parser.parse_args()

    t0 = time.time()
    print("🦅 Falcon 独立评分")
    print("=" * 60)

    # ── 数据新鲜度检查 ──
    if not args.skip_freshness:
        try:
            sys.path.insert(0, str(FALCON_DIR))
            from check_data_fresh import check_price_freshness
            is_fresh, msg, gap, _ = check_price_freshness()
            print(f"  📅 {msg}")
            if not is_fresh:
                print(f"  ❌ 数据过期{gap}天，拒绝评分！请运行: python3 scripts/falcon/update_price_data.py")
                sys.exit(1)
            if gap > 0:
                print(f"  ⚠️ 数据差{gap}天，评分结果可能不够新")
        except Exception as e:
            print(f"  ⚠️ 新鲜度检查失败: {e}")

    all_rows = []

    # SPX
    if args.universe in ("spx", "all"):
        print("📊 加载 SPX 数据...")
        master, data = load_spx_data()
        print(f"  ✅ {master['ticker'].nunique()} 只, {len(master)} 行")
        print(f"📊 计算 SPX PIT rank...")
        row, date = compute_today_rank(master, data, args.date, weights=SPX_WEIGHTS)
        if row is not None:
            row["universe"] = "SPX"
            all_rows.append(row)

    # R2K
    if args.universe in ("r2k", "all"):
        print("📊 加载 R2K 数据...")
        r2k_master, r2k_data = load_r2k_data()
        if r2k_master is not None:
            print(f"📊 计算 R2K PIT rank...")
            r2k_row, r2k_date = compute_today_rank(r2k_master, r2k_data, args.date, weights=R2K_WEIGHTS)
            if r2k_row is not None:
                r2k_row["universe"] = "R2K"
                all_rows.append(r2k_row)
                date = max(date, r2k_date)

    if not all_rows:
        print("❌ 评分失败")
        sys.exit(1)

    # 合并并统一排名
    combined = pd.concat(all_rows, ignore_index=True)
    # 每个universe内部排名
    for uni in combined["universe"].unique():
        mask = combined["universe"] == uni
        combined.loc[mask, "rank_in_universe"] = combined.loc[mask, "falcon_score"].rank(ascending=False)
        combined.loc[mask, "pct_in_universe"] = combined.loc[mask, "falcon_score"].rank(pct=True)

    # 全局排名
    combined["rank_pct"] = combined["falcon_score"].rank(pct=True)
    combined = combined.sort_values("falcon_score", ascending=False)
    picks = combined.head(args.top_n)

    # 构造输出
    output_picks = []
    for _, r in picks.iterrows():
        rp = r["rank_pct"]
        output_picks.append({
            "sym": r["ticker"],
            "score": round(float(r["falcon_score"]), 4),
            "close": round(float(r["close"]), 2),
            "rank_pct": round(float(rp), 4),
            "signal": score_to_signal(float(r["falcon_score"]), rp),
            "universe": r["universe"],
            "fund_ratio": round(float(r.get("fund_ratio", 0.5)), 4),
            "analyst": round(float(r.get("analyst", 0.5)), 4),
            "fund_metric": round(float(r.get("fund_metric", 0.5)), 4),
        })

    # Universe breakdown
    uni_counts = combined["universe"].value_counts().to_dict()

    result = {
        "model": "falcon_v031",
        "date": date,
        "universe_size": int(combined.shape[0]),
        "scored_count": int(combined.shape[0]),
        "universes": uni_counts,
        "weights": {"spx": SPX_WEIGHTS, "r2k": R2K_WEIGHTS},
        "strategy": f"fixed_{HOLD_DAYS}d",
        "top_n": args.top_n,
        "picks": output_picks,
    }

    # 保存
    # 保存 — 文件名用 model 名前缀，与 alpaca_trade.py 的搜索模式对齐
    out_file = OUTPUT_DIR / f"falcon_v031_scored_{str(date).replace('-', '')}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    # 打印
    print(f"\n{'='*60}")
    print(f"📊 Falcon 评分结果 — {date}")
    print(f"{'='*60}")
    print(f"{'排名':>4} {'代码':<8} {'来源':<5} {'分数':>8} {'排名%':>8} {'信号':<6} {'价格':>10} {'Fund':>6} {'Ana':>6} {'Met':>6}")
    print("-" * 80)
    for i, p in enumerate(output_picks, 1):
        print(f"{i:>4} {p['sym']:<8} {p.get('universe','?'):<5} {p['score']:>8.4f} {p['rank_pct']*100:>7.1f}% "
              f"{p['signal']:<6} ${p['close']:>8.2f} "
              f"{p['fund_ratio']:>5.2f} {p['analyst']:>5.2f} {p['fund_metric']:>5.2f}")

    # 统计
    green2 = sum(1 for p in output_picks if "🟢🟢" in p["signal"])
    green1 = sum(1 for p in output_picks if p["signal"].count("🟢") == 1)
    yellow = sum(1 for p in output_picks if "🟡" in p["signal"])

    print(f"\n  🟢🟢: {green2} | 🟢: {green1} | 🟡: {yellow}")
    print(f"  📁 输出: {out_file}")
    print(f"  ⏱️ {time.time()-t0:.1f}秒")


if __name__ == "__main__":
    main()
