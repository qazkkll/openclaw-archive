#!/usr/bin/env python3
"""
🦅 Falcon V0.4.6 评分脚本
=========================
V0.4.6 (2026-07-03):
  - V0.4.4因子结构(53因子) + IC^0.5加权(lookback=126天)
  - WF Sharpe: 2.140, CAGR: 51.9%, MaxDD: -15.2%
  - 5年每年全赢等权baseline，无过拟合
  
  顶层权重不变: fund_ratio(45%) + gc(20%) + qoq(20%) + cashflow(15%)
  底层因子: 从等权 → 滚动IC^0.5加权
  IC权重每日由 compute_rolling_ic.py 更新

用法:
    python3 scripts/falcon/falcon_score.py                  # 评分最新交易日
    python3 scripts/falcon/falcon_score.py --date 2024-12-31  # 评分指定日期
    python3 scripts/falcon/falcon_score.py --top-n 10       # 取 Top-10

输出: data/falcon/falcon_v046_scored_YYYYMMDD.json
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

# ═══════════════════════════════════════════════════
# 路径配置
# ═══════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
SCORED_DIR = DATA_DIR
IC_WEIGHTS_PATH = DATA_DIR / "factor_ic_weights.json"

# ═══════════════════════════════════════════════════
# V0.4.6 因子组定义 (53因子，与V0.4.4相同)
# ═══════════════════════════════════════════════════

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
    'analyst': ['a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps'],
    'income': ['i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
               'i_revenue_growth_yoy', 'i_gross_margin_delta'],
    'qoq': ['r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
            'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq'],
    'cashflow': ['c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield'],
}

# V0.4.6 FLIP_FACTORS (翻转方向 = 数值越高收益越差)
FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity', 'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    'a_eps_revision', 'a_revenue_revision',
}

# V0.4.6 主权重 (与V0.4.4相同)
V046_WEIGHTS = {
    "fund_ratio": 0.45,
    "growth_composite": 0.20,
    "qoq": 0.20,
    "cashflow": 0.15,
}

# V0.4.6 growth_composite 子权重 (与V0.4.4相同)
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
# IC权重加载
# ═══════════════════════════════════════════════════

def load_ic_weights() -> dict:
    """加载IC权重文件。缺失或过期则直接报错，不允许回退等权。"""
    if not IC_WEIGHTS_PATH.exists():
        print(f"❌ IC权重文件不存在: {IC_WEIGHTS_PATH}")
        print(f"❌ 必须先运行 compute_rolling_ic.py，拒绝以等权评分运行")
        sys.exit(1)
    
    with open(IC_WEIGHTS_PATH) as f:
        data = json.load(f)
    
    computed_at = data.get('computed_at', '')
    if not computed_at:
        print(f"❌ IC权重文件缺少computed_at字段")
        sys.exit(1)
    
    from datetime import timedelta
    try:
        comp_date = datetime.strptime(computed_at, '%Y-%m-%d')
        age_days = (datetime.now() - comp_date).days
        if age_days > 3:
            print(f"❌ IC权重已过期 ({age_days}天前计算): {computed_at}")
            print(f"❌ 必须重新运行 compute_rolling_ic.py")
            sys.exit(1)
    except ValueError:
        print(f"❌ IC权重文件computed_at格式错误: {computed_at}")
        sys.exit(1)
    
    return data


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════

def load_features() -> pd.DataFrame:
    path = DATA_DIR / "features_v04_1.parquet"
    if not path.exists():
        print(f"❌ 特征文件不存在: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    df["date"] = df["date"].astype(str)
    return df


# ═══════════════════════════════════════════════════
# IC加权评分
# ═══════════════════════════════════════════════════

def compute_group_score_ic(day: pd.DataFrame, group_cols: list, flip_set: set,
                           ic_weights: dict = None) -> pd.Series:
    """计算因子组得分 (截面rank百分位 + IC加权)。
    
    如果ic_weights为None，回退到等权。
    """
    available = [c for c in group_cols if c in day.columns and day[c].notna().sum() > 5]
    if not available:
        return pd.Series(0.5, index=day.index)
    
    ranks = pd.DataFrame(index=day.index)
    for col in available:
        r = day[col].rank(pct=True)
        if col in flip_set:
            r = 1 - r
        ranks[col] = r
    
    if ic_weights is not None:
        # IC加权
        weights = {}
        for col in available:
            w = ic_weights.get(col, 0)
            weights[col] = max(0, w)  # 负IC的因子权重设为0
        
        total = sum(weights.values())
        if total > 0:
            # 归一化
            weights = {k: v / total for k, v in weights.items()}
            score = pd.Series(0.0, index=day.index)
            for col in available:
                score += weights[col] * ranks[col]
            return score
        # 如果所有权重都是0，回退等权
        pass
    
    # 等权fallback
    return ranks.mean(axis=1)


def compute_score(features: pd.DataFrame, target_date: str,
                  ic_data: dict) -> tuple:
    """计算指定日期的V0.4.6评分。"""
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
    
    # 提取IC权重（必须存在）
    ic_weights = ic_data['weights']
    
    # 计算各因子组得分（IC加权）
    scores = {}
    
    fr_ic = ic_weights.get('fund_ratio', {}) if ic_weights else None
    scores["fund_ratio"] = compute_group_score_ic(
        day, FACTOR_GROUPS["fund_ratio"], FLIP_FACTORS, fr_ic)
    
    fg_ic = ic_weights.get('fund_growth', {}) if ic_weights else None
    fg = compute_group_score_ic(day, FACTOR_GROUPS["fund_growth"], FLIP_FACTORS, fg_ic)
    
    an_ic = ic_weights.get('analyst', {}) if ic_weights else None
    an = compute_group_score_ic(day, FACTOR_GROUPS["analyst"], FLIP_FACTORS, an_ic)
    
    inc_ic = ic_weights.get('income', {}) if ic_weights else None
    inc = compute_group_score_ic(day, FACTOR_GROUPS["income"], FLIP_FACTORS, inc_ic)
    
    scores["growth_composite"] = (
        GC_WEIGHTS["fund_growth"] * fg +
        GC_WEIGHTS["analyst"] * an +
        GC_WEIGHTS["income"] * inc
    )
    
    qoq_ic = ic_weights.get('qoq', {}) if ic_weights else None
    scores["qoq"] = compute_group_score_ic(day, FACTOR_GROUPS["qoq"], FLIP_FACTORS, qoq_ic)
    
    cf_ic = ic_weights.get('cashflow', {}) if ic_weights else None
    scores["cashflow"] = compute_group_score_ic(day, FACTOR_GROUPS["cashflow"], FLIP_FACTORS, cf_ic)
    
    # 加权综合分
    falcon_score = sum(
        V046_WEIGHTS[f] * scores[f]
        for f in V046_WEIGHTS
    )
    
    # 构建结果DataFrame
    result = day[["ticker", "close"]].copy()
    result["date"] = date
    for f in V046_WEIGHTS:
        result[f] = scores[f]
    result["falcon_score"] = falcon_score
    result["rank_pct"] = falcon_score.rank(pct=True)
    
    # 子因子详情
    result["fund_growth"] = fg
    result["analyst_score"] = an
    result["income_score"] = inc
    
    return result, date


def score_to_signal(score: float, pct: float) -> str:
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
    parser = argparse.ArgumentParser(description="Falcon V0.4.6 评分")
    parser.add_argument("--date", default=None, help="评分日期 (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=TOP_N, help=f"取 Top-N (默认 {TOP_N})")
    parser.add_argument("--universe", default="spx", choices=["spx", "all"],
                        help="评分范围")
    parser.add_argument("--skip-freshness", action="store_true", help="跳过数据新鲜度检查")
    args = parser.parse_args()

    t0 = time.time()
    print("🦅 Falcon V0.4.6 评分 (IC加权)")
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
            freshness_path = DATA_DIR / "freshness_check_log.json"
            if freshness_path.exists():
                with open(freshness_path) as f:
                    freshness = json.load(f)
                last_check = freshness.get("last_check", "unknown")
                print(f"  📅 数据新鲜度: {last_check}")
        except Exception:
            pass

    # ── 加载IC权重（必须存在，缺失/过期直接退出） ──
    print("\n📊 加载IC权重...")
    ic_data = load_ic_weights()
    computed_at = ic_data.get('computed_at', 'unknown')
    n_positive = ic_data.get('factors_with_positive_ic', 0)
    n_total = ic_data.get('total_factors', 0)
    print(f"  ✅ IC权重: {computed_at} ({n_positive}/{n_total}因子正IC)")

    # ── 加载特征 ──
    print("\n📊 加载特征数据...")
    features = load_features()
    
    # ── 计算评分 ──
    print(f"📊 计算 {args.date or '最新'} 评分...")
    result, date = compute_score(features, args.date, ic_data)
    if result is None:
        print("❌ 评分失败")
        sys.exit(1)
    
    # ── B3修复: 用实时价格覆盖parquet旧价格 ──
    # 因子计算需要parquet中的历史数据(技术指标), 但close必须是最新价
    print("\n📊 获取实时价格覆盖旧价格...")
    try:
        import yfinance as yf
        all_tickers = result["ticker"].tolist()
        rt_data = yf.download(all_tickers, period="2d", progress=False, threads=True)
        if not rt_data.empty:
            updated = 0
            if isinstance(rt_data.columns, pd.MultiIndex):
                for ticker in all_tickers:
                    try:
                        prices = rt_data["Close"][ticker].dropna()
                        if len(prices) > 0:
                            latest = float(prices.iloc[-1])
                            if latest > 0:
                                result.loc[result["ticker"] == ticker, "close"] = latest
                                updated += 1
                    except (KeyError, IndexError):
                        pass
            else:
                if len(all_tickers) == 1:
                    prices = rt_data["Close"].dropna()
                    if len(prices) > 0:
                        result.loc[result["ticker"] == all_tickers[0], "close"] = float(prices.iloc[-1])
                        updated = 1
            print(f"  ✅ 实时价格覆盖: {updated}/{len(all_tickers)}只")
        else:
            print("  ⚠️ yfinance返回空数据, 使用parquet价格")
    except Exception as e:
        print(f"  ⚠️ 实时价格获取失败({e}), 使用parquet价格")

    # ── 生成信号 ──
    result["signal"] = result.apply(
        lambda r: score_to_signal(r["falcon_score"], r["rank_pct"]), axis=1
    )
    
    # ── 过滤universe ──
    if args.universe == "spx":
        spx_path = PROJECT_ROOT / "data" / "us" / "sp500_components.json"
        if spx_path.exists():
            with open(spx_path) as f:
                spx_data = json.load(f)
            spx_tickers = set(spx_data.get("tickers", []))
            result = result[result["ticker"].isin(spx_tickers)]
            print(f"  📋 Universe: S&P 500 ({len(result)} 只)")
    
    # ── 排序输出 ──
    result = result.sort_values("falcon_score", ascending=False)
    top_n = result.head(args.top_n)
    
    print(f"\n🏆 Top-{args.top_n} 评分 ({date})")
    print("-" * 60)
    for _, row in top_n.iterrows():
        print(f"{row['signal']} {row['ticker']:6s} "
              f"Score={row['falcon_score']:.3f} "
              f"Pct={row['rank_pct']:.1%} "
              f"Close=${row['close']:.2f}")
    
    # ── 保存结果 ──
    output_file = SCORED_DIR / f"falcon_v046_scored_{date.replace('-', '')}.json"
    
    # IC权重摘要
    ic_summary = {}
    if ic_data and 'weights' in ic_data:
        for group, weights in ic_data['weights'].items():
            sorted_w = sorted(weights.items(), key=lambda x: -x[1])
            ic_summary[group] = {f: w for f, w in sorted_w[:5]}
    
    output_data = {
        "version": "V0.4.6",
        "date": date,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "weights": V046_WEIGHTS,
            "gc_weights": GC_WEIGHTS,
            "factors": {k: len(v) for k, v in FACTOR_GROUPS.items()},
            "total_factors": sum(len(v) for v in FACTOR_GROUPS.values()),
            "ic_weighted": ic_data is not None,
            "ic_lookback": ic_data.get('lookback', 'N/A') if ic_data else 'N/A',
            "ic_power": ic_data.get('power', 'N/A') if ic_data else 'N/A',
            "ic_computed_at": ic_data.get('computed_at', 'N/A') if ic_data else 'N/A',
        },
        "vix": vix_info,
        "top_n": [
            {
                "ticker": row["ticker"],
                "score": round(row["falcon_score"], 4),
                "rank_pct": round(row["rank_pct"], 4),
                "signal": row["signal"],
                "close": round(row["close"], 2),
                "fund_ratio": round(row["fund_ratio"], 4),
                "growth_composite": round(row["growth_composite"], 4),
                "qoq": round(row["qoq"], 4),
                "cashflow": round(row["cashflow"], 4),
            }
            for _, row in top_n.iterrows()
        ],
        "ic_weight_summary": ic_summary,
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    elapsed = time.time() - t0
    print(f"\n✅ 评分完成，耗时 {elapsed:.1f}s")
    print(f"📁 输出: {output_file}")
    
    return output_data


if __name__ == "__main__":
    main()
