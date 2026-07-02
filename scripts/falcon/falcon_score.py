#!/usr/bin/env python3
"""
🦅 Falcon 独立评分脚本 (V0.4.4)
=====================
V0.4.4: 使用features_v04_1.parquet预计算因子，Walk-Forward验证通过。

模型层面: fund_ratio(45%) + growth_composite(20%) + qoq(20%) + cashflow(15%)
执行层面: VIX大盘感知 + 动态仓位管理

用法:
    python3 scripts/falcon/falcon_score.py                  # 评分最新交易日
    python3 scripts/falcon/falcon_score.py --date 2024-12-31  # 评分指定日期
    python3 scripts/falcon/falcon_score.py --top-n 10       # 取 Top-10

输出: data/falcon/falcon_v044_scored_YYYYMMDD.json
"""

import sys
import os
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
OUTPUT_DIR = DATA_DIR

# ═══════════════════════════════════════════════════
# V0.4.4 因子配置 (Walk-Forward验证通过, RI=68.4%)
# ═══════════════════════════════════════════════════

# 因子组定义
FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_assetGrowth',
        'g_bookValueperShareGrowth', 'g_debtGrowth',
    ],
    'analyst': [
        'a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps',
    ],
    'income': [
        'i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
        'i_revenue_growth_yoy', 'i_gross_margin_delta',
    ],
    'qoq': [
        'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
        'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
    ],
    'cashflow': [
        'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
    ],
}

# 需要翻转的因子 (越高越差)
FLIP_FACTORS = {
    # 估值（越低越好）
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    # 杠杆（越低越好）
    'r_debtToEquityRatio', 'r_financialLeverageRatio',
    # 周转（数据验证：IC=-0.008，越高收益越差）
    'r_inventoryTurnover',
    # 股息（数据验证：IC=-0.04，越高收益越差 → 成长股溢价）
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    # 资本开支（越低越好）
    'c_capex_intensity',
    # 负债/应收/存货增长（越低越好）
    'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    # 分析师修正（数据验证：IC=-0.07/-0.06，越高收益越差 → 滞后指标）
    'a_eps_revision', 'a_revenue_revision',
    # 不翻：a_eps_dispersion（IC=+0.019，越高收益越好 → 分歧=机会）
    # 不翻：g_assetGrowth（IC=+0.023，越高收益越好）
}

# V0.4.4 主权重
V044_WEIGHTS = {
    "fund_ratio": 0.45,
    "growth_composite": 0.20,
    "qoq": 0.20,
    "cashflow": 0.15,
}

# growth_composite 子权重
GC_WEIGHTS = {
    "fund_growth": 0.60,
    "analyst": 0.25,
    "income": 0.15,
}

TOP_N = 10
HOLD_DAYS = 30
VIX_THRESHOLD = 25

# ═══════════════════════════════════════════════════
# 大盘感知 (执行层面)
# ═══════════════════════════════════════════════════

REGIME_THRESHOLDS = {
    "extreme_bear": {"vix_min": 30, "position_pct": 0.25},
    "bear":         {"vix_min": 25, "position_pct": 0.50},
    "neutral":      {"vix_min": 20, "position_pct": 0.75},
    "bull":         {"vix_max": 20, "position_pct": 1.00},
}


def detect_market_regime(vix_value: float, trend_pct: float = 0.0) -> dict:
    """根据VIX和趋势判断市场状态。"""
    if vix_value > 30:
        regime = "extreme_bear"
    elif vix_value > 25:
        regime = "bear"
    elif vix_value >= 20:
        regime = "neutral"
    else:
        regime = "bull"
    
    position_pct = REGIME_THRESHOLDS[regime]["position_pct"]
    return {
        "regime": regime,
        "vix": round(vix_value, 1),
        "trend_pct": round(trend_pct, 4),
        "position_pct": position_pct,
    }


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════

def load_features() -> pd.DataFrame:
    """加载V0.4.4特征文件。"""
    path = DATA_DIR / "features_v04_1.parquet"
    if not path.exists():
        print(f"❌ 特征文件不存在: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["date"] = df["date"].astype(str)
    return df


def compute_group_score(day: pd.DataFrame, group_cols: list, flip_set: set) -> pd.Series:
    """计算因子组得分 (截面rank百分位)。"""
    available = [c for c in group_cols if c in day.columns and day[c].notna().sum() > 5]
    if not available:
        return pd.Series(0.5, index=day.index)
    
    ranks = pd.DataFrame(index=day.index)
    for col in available:
        r = day[col].rank(pct=True)
        if col in flip_set:
            r = 1 - r
        ranks[col] = r
    
    return ranks.mean(axis=1)


def compute_score(features: pd.DataFrame, target_date: str = None) -> tuple:
    """计算指定日期的V0.4.4评分。"""
    dates = sorted(features["date"].unique())
    if target_date:
        available = [d for d in dates if d <= target_date]
        if not available:
            print(f"❌ 无 {target_date} 之前的数据")
            return None, None
        date = available[-1]
    else:
        date = dates[-1]
    
    day = features[features["date"] == date].copy()
    if len(day) < 10:
        print(f"❌ {date} 只有 {len(day)} 只股票")
        return None, None
    
    day.index = day["ticker"].values
    
    # 计算各因子组得分
    scores = {}
    
    # fund_ratio
    scores["fund_ratio"] = compute_group_score(day, FACTOR_GROUPS["fund_ratio"], FLIP_FACTORS)
    
    # growth_composite = 0.60*fund_growth + 0.25*analyst + 0.15*income
    fg = compute_group_score(day, FACTOR_GROUPS["fund_growth"], FLIP_FACTORS)
    an = compute_group_score(day, FACTOR_GROUPS["analyst"], FLIP_FACTORS)
    inc = compute_group_score(day, FACTOR_GROUPS["income"], FLIP_FACTORS)
    scores["growth_composite"] = (
        GC_WEIGHTS["fund_growth"] * fg +
        GC_WEIGHTS["analyst"] * an +
        GC_WEIGHTS["income"] * inc
    )
    
    # qoq
    scores["qoq"] = compute_group_score(day, FACTOR_GROUPS["qoq"], FLIP_FACTORS)
    
    # cashflow
    scores["cashflow"] = compute_group_score(day, FACTOR_GROUPS["cashflow"], FLIP_FACTORS)
    
    # 加权综合分
    falcon_score = sum(
        V044_WEIGHTS[f] * scores[f]
        for f in V044_WEIGHTS
    )
    
    # 构建结果DataFrame
    result = day[["ticker", "close"]].copy()
    result["date"] = date
    for f in V044_WEIGHTS:
        result[f] = scores[f]
    result["falcon_score"] = falcon_score
    result["rank_pct"] = falcon_score.rank(pct=True)
    
    # 子因子详情
    result["fund_growth"] = fg
    result["analyst_score"] = an
    result["income_score"] = inc
    
    return result, date


def score_to_signal(score: float, pct: float) -> str:
    """评分 → 信号等级。"""
    if score >= 0.55 and pct >= 0.95:
        return "🟢🟢"
    elif score >= 0.55 and pct >= 0.80:
        return "🟢"
    elif score >= 0.50:
        return "🟡"
    else:
        return "🔴"


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Falcon V0.4.4 评分")
    parser.add_argument("--date", default=None, help="评分日期 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"取 Top-N (默认 {TOP_N})")
    parser.add_argument("--universe", default="spx", choices=["spx", "all"],
                        help="评分范围")
    parser.add_argument("--skip-freshness", action="store_true", help="跳过数据新鲜度检查")
    args = parser.parse_args()

    t0 = time.time()
    print("🦅 Falcon V0.4.4 评分")
    print("=" * 60)

    # ── VIX 检查 (执行层面) ──
    vix_info = {"vix": 20.0, "regime": "bull", "position_pct": 1.0}
    try:
        vix_path = PROJECT_ROOT / "data" / "us" / "vix_10y.parquet"
        if vix_path.exists():
            vix_raw = pd.read_parquet(vix_path)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_close = vix_raw[("Close", "^VIX")]
            elif "Close" in vix_raw.columns:
                vix_close = vix_raw["Close"]
            elif "close" in vix_raw.columns:
                vix_close = vix_raw["close"]
            else:
                numeric_cols = vix_raw.select_dtypes(include='number').columns
                vix_close = vix_raw[numeric_cols[0]] if len(numeric_cols) > 0 else vix_raw.iloc[:, 0]
            
            latest_vix = float(vix_close.iloc[-1])
            vix_info = detect_market_regime(latest_vix)
            print(f"  📊 VIX: {latest_vix:.1f} | 市场状态: {vix_info['regime']} | 仓位: {vix_info['position_pct']*100:.0f}%")
            
            if latest_vix > VIX_THRESHOLD:
                print(f"  ⚠️ VIX > {VIX_THRESHOLD}，市场恐慌，交易信号标记为SKIP")
                os.environ["FALCON_VIX_SKIP"] = "1"
            else:
                os.environ["FALCON_VIX_SKIP"] = "0"
    except Exception as e:
        print(f"  ⚠️ VIX检查失败: {e}")

    # ── 数据新鲜度检查 ──
    if not args.skip_freshness:
        try:
            sys.path.insert(0, str(FALCON_DIR))
            from check_data_fresh import check_price_freshness
            is_fresh, msg, gap, _ = check_price_freshness()
            print(f"  📅 {msg}")
            if not is_fresh:
                print(f"  ❌ 数据过期{gap}天，拒绝评分！")
                sys.exit(1)
        except Exception as e:
            print(f"  ⚠️ 新鲜度检查失败: {e}")

    # ── 加载特征并评分 ──
    print("\n📊 加载 V0.4.4 特征...")
    features = load_features()
    print(f"  ✅ {features['ticker'].nunique()} 只, {len(features)} 行")

    print(f"📊 计算评分...")
    result, date = compute_score(features, args.date)
    if result is None:
        print("❌ 评分失败")
        sys.exit(1)

    result["universe"] = "SPX"
    result = result.sort_values("falcon_score", ascending=False)
    picks = result.head(args.top_n)

    # 构造输出
    output_picks = []
    for _, r in picks.iterrows():
        rp = float(r["rank_pct"])
        output_picks.append({
            "sym": r["ticker"],
            "score": round(float(r["falcon_score"]), 4),
            "close": round(float(r["close"]), 2),
            "rank_pct": round(rp, 4),
            "signal": score_to_signal(float(r["falcon_score"]), rp),
            "universe": "SPX",
            "fund_ratio": round(float(r.get("fund_ratio", 0.5)), 4),
            "growth_composite": round(float(r.get("growth_composite", 0.5)), 4),
            "qoq": round(float(r.get("qoq", 0.5)), 4),
            "cashflow": round(float(r.get("cashflow", 0.5)), 4),
            "fund_growth": round(float(r.get("fund_growth", 0.5)), 4),
            "analyst": round(float(r.get("analyst_score", 0.5)), 4),
            "income": round(float(r.get("income_score", 0.5)), 4),
        })

    # 保存
    out_file = OUTPUT_DIR / f"falcon_v044_scored_{str(date).replace('-', '')}.json"
    result_json = {
        "model": "falcon_v044",
        "version": "V0.4.4",
        "date": date,
        "universe_size": int(result.shape[0]),
        "scored_count": int(result.shape[0]),
        "weights": V044_WEIGHTS,
        "growth_composite_weights": GC_WEIGHTS,
        "market_regime": vix_info,
        "strategy": f"fixed_{HOLD_DAYS}d",
        "vix_threshold": VIX_THRESHOLD,
        "vix_skip": os.environ.get("FALCON_VIX_SKIP", "0") == "1",
        "top_n": args.top_n,
        "picks": output_picks,
    }
    with open(out_file, "w") as f:
        json.dump(result_json, f, indent=2)

    # 打印
    print(f"\n{'='*60}")
    print(f"📊 Falcon V0.4.4 评分结果 — {date}")
    print(f"{'='*60}")
    print(f"{'排名':>4} {'代码':<8} {'分数':>8} {'排名%':>8} {'信号':<6} {'价格':>10} {'比率':>6} {'成长':>6} {'QoQ':>6} {'现金流':>6}")
    print("-" * 90)
    for i, p in enumerate(output_picks, 1):
        print(f"{i:>4} {p['sym']:<8} {p['score']:>8.4f} {p['rank_pct']*100:>7.1f}% "
              f"{p['signal']:<6} ${p['close']:>8.2f} "
              f"{p['fund_ratio']:>5.2f} {p['growth_composite']:>5.2f} {p['qoq']:>5.2f} {p['cashflow']:>5.2f}")

    green2 = sum(1 for p in output_picks if "🟢🟢" in p["signal"])
    green1 = sum(1 for p in output_picks if p["signal"].count("🟢") == 1)
    yellow = sum(1 for p in output_picks if "🟡" in p["signal"])

    print(f"\n  🟢🟢: {green2} | 🟢: {green1} | 🟡: {yellow}")
    print(f"  📊 市场: {vix_info['regime']} (VIX={vix_info['vix']}) → 仓位{vix_info['position_pct']*100:.0f}%")
    print(f"  📁 输出: {out_file}")
    print(f"  ⏱️ {time.time()-t0:.1f}秒")


if __name__ == "__main__":
    main()
