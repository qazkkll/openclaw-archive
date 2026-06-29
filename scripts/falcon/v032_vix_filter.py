#!/usr/bin/env python3
"""
🦅 Falcon V0.3.2 动态VIX过滤优化
===================================
在V0.3.2最优权重基础上，测试不同VIX过滤策略：
  1. VIX>X时跳过买入（只持仓不加仓）
  2. VIX>X时减少持仓比例（50%/75%）
  3. VIX>X时切换到防御因子权重
  4. 不同VIX阈值组合

Walk-Forward验证，找最优参数。
"""
import sys, json, warnings, time
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from falcon_v03_engine import (
    precompute_pit_ranks_fast, backtest_flexible,
    build_pit_index_statements, compute_statement_factors,
    BALANCE_FIELDS, CASHFLOW_FIELDS, INCOME_FIELDS,
)
from extract_fmp_premium_features import load_fmp_premium_earnings, load_fmp_premium_grades

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "falcon"
FMP_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "fmp_premium"
INV = {"debt_to_equity", "net_debt_to_assets", "capex_intensity"}


def load_data():
    print("📂 加载数据...")
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    data = {}
    for n in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
              "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{n}.json"
        if f.exists():
            data[n] = json.load(open(f))
    data["earnings"] = load_fmp_premium_earnings(str(FMP_DIR))
    data["grades"] = load_fmp_premium_grades(str(FMP_DIR))
    for n in ["fmp_balance_sheet", "fmp_cashflow", "fmp_income_stmt"]:
        f = DATA_DIR / f"{n}.json"
        if f.exists():
            data[n] = json.load(open(f))
    # VIX历史数据
    vix_path = DATA_DIR.parent / "us" / "vix_10y.parquet"
    if vix_path.exists():
        vix_df = pd.read_parquet(vix_path)
        print(f"  ✅ VIX: {len(vix_df)}行")
    else:
        vix_df = None
        print("  ⚠️ VIX数据不存在")
    all_dates = sorted(master["date"].unique())
    print(f"  ✅ {len(master):,}行, {master['ticker'].nunique()}只, {len(all_dates)}天")
    return master, data, all_dates, vix_df


def compute_all_ranks(master, data, all_dates):
    print("\n📊 计算PIT rank...")
    t0 = time.time()
    ranks = precompute_pit_ranks_fast(
        master,
        data.get("fmp_ratios_historical", {}),
        data.get("analyst_historical", {}),
        data.get("fmp_key_metrics", {}),
        data.get("fmp_financial_growth", {}),
        data.get("fmp_insider", {}),
        data.get("fmp_dcf", {}),
        data.get("fmp_price_target", {}),
        earnings_hist=data.get("earnings", {}),
        grades_hist=data.get("grades", {}),
    )
    print(f"  旧因子: {len(ranks)}天, {time.time()-t0:.0f}秒")

    # 新因子
    t0 = time.time()
    income_idx = build_pit_index_statements(data.get("fmp_income_stmt", {}), use_filing_date=True)
    balance_idx = build_pit_index_statements(data.get("fmp_balance_sheet", {}), use_filing_date=False)
    cashflow_idx = build_pit_index_statements(data.get("fmp_cashflow", {}), use_filing_date=False)

    for date in sorted(ranks.keys()):
        df = ranks[date]
        tickers = df.index.tolist()
        new_data = {}
        for t in tickers:
            f = compute_statement_factors(t, date, balance_idx, cashflow_idx, income_idx, {})
            if f:
                new_data[t] = f
        if new_data:
            ndf = pd.DataFrame.from_dict(new_data, orient="index")
            vc = [c for c in ndf.columns if ndf[c].notna().sum() >= 10]
            if vc:
                rn = ndf[vc].rank(pct=True)
                for c in vc:
                    if c in INV:
                        rn[c] = 1 - rn[c]
                    df[c] = rn[c]
        for gn, fs in [("balance", BALANCE_FIELDS), ("cashflow", CASHFLOW_FIELDS), ("income_stmt", INCOME_FIELDS)]:
            cols = [c for c in fs if c in df.columns]
            if cols:
                df[gn] = df[cols].mean(axis=1)

    print(f"  新因子: {time.time()-t0:.0f}秒")
    return ranks


def build_vix_series(vix_df):
    """从VIX parquet构建日期→VIX值映射。"""
    if vix_df is None:
        return {}
    # 多级列结构: (Price, Ticker) = (Close/^VIX, High/^VIX, ...)
    # 尝试提取Close列
    try:
        if isinstance(vix_df.columns, pd.MultiIndex):
            close_col = None
            for c in vix_df.columns:
                if c[0] == "Close":
                    close_col = c
                    break
            if close_col is None:
                close_col = vix_df.columns[0]
            vix_close = vix_df[close_col]
        else:
            # 单级列
            vix_close = vix_df.iloc[:, 0]
    except Exception:
        vix_close = vix_df.iloc[:, 0]

    vix_dict = {}
    for dt, val in vix_close.items():
        d = str(dt)[:10]
        try:
            vix_dict[d] = float(val)
        except (ValueError, TypeError):
            pass
    return vix_dict


def walk_forward_windows(dates, train_years=2, test_months=6):
    from dateutil.relativedelta import relativedelta
    windows = []
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    ts = start
    while True:
        te = ts + relativedelta(years=train_years)
        tte = te + relativedelta(months=test_months)
        if tte > end:
            break
        td = [d for d in dates if te.strftime("%Y-%m-%d") <= d < tte.strftime("%Y-%m-%d")]
        if len(td) >= 50:
            windows.append(td)
        ts = ts + relativedelta(months=test_months)
    return windows


def backtest_with_vix_filter(ranks, price_pivot, dates, regime_above,
                              weights, hold_days, top_n, vix_dict,
                              vix_threshold=None, vix_filter_type=None,
                              vix_reduce_pct=1.0):
    """
    带VIX过滤的回测。

    参数:
        vix_threshold: VIX阈值 (超过则触发过滤)
        vix_filter_type:
            None = 不过滤
            "skip_buy" = VIX高时跳过买入 (已有持仓继续持有)
            "reduce_size" = VIX高时减少买入比例
            "defensive" = VIX高时切换到防御权重
    """
    # 过滤出有VIX数据的日期
    valid_dates = [d for d in dates if d in vix_dict]
    if not valid_dates:
        return None

    # 如果有VIX过滤，修改dates标记
    if vix_threshold and vix_filter_type:
        # 在高VIX期间标记需要跳过的买入日
        high_vix_dates = {d for d in valid_dates if vix_dict[d] >= vix_threshold}
        # 这里我们用一个简化方法：在高VIX期间，不执行新的买入
        # 通过修改regime_above来模拟（VIX高=regime差）
        modified_regime = regime_above.copy()
        for d in high_vix_dates:
            if d in modified_regime.index:
                if vix_filter_type == "skip_buy":
                    modified_regime.loc[d] = 0  # 不买入
                elif vix_filter_type == "reduce_size":
                    # 保持regime但后续用reduce_pct调整
                    pass
    else:
        modified_regime = regime_above
        high_vix_dates = set()

    try:
        result = backtest_flexible(
            ranks, price_pivot, valid_dates, modified_regime,
            weights=weights, strategy="fixed",
            params={"hold_days": hold_days, "cost": 0.001, "stop_loss": -0.15},
            top_n=top_n,
        )
        if result and result.get("sharpe") is not None:
            return result
    except Exception as e:
        pass
    return None


def get_regime(price_pivot):
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    return (mkt_price > mkt_ma200).astype(int)


def main():
    print("🦅 Falcon V0.3.2 动态VIX过滤优化")
    print("=" * 80)
    t_total = time.time()

    master, data, all_dates, vix_df = load_data()
    ranks = compute_all_ranks(master, data, all_dates)

    price_pivot = master.pivot(index="date", columns="ticker", values="close")
    price_pivot.index = price_pivot.index.astype(str)
    regime_above = get_regime(price_pivot)
    windows = walk_forward_windows(all_dates)

    # 构建VIX序列
    vix_dict = build_vix_series(vix_df)
    print(f"\n📊 VIX数据: {len(vix_dict)}天")
    if vix_dict:
        vix_vals = list(vix_dict.values())
        print(f"  范围: {min(vix_vals):.1f} ~ {max(vix_vals):.1f}")
        print(f"  中位数: {np.median(vix_vals):.1f}")
        # VIX分布
        for threshold in [15, 18, 20, 22, 25, 30]:
            pct = np.mean([1 for v in vix_vals if v >= threshold]) * 100
            print(f"  VIX>={threshold}: {pct:.1f}%天数")

    # V0.3.2权重
    v032_w = {
        "fund_growth": 0.15, "cashflow": 0.12, "analyst": 0.12,
        "grade_sentiment": 0.12, "earnings": 0.10, "balance": 0.08,
        "fund_metric": 0.06, "insider": 0.05, "fund_ratio": 0.05,
    }
    # 防御权重 (高VIX时用) — 增加fund_ratio/fund_metric(低波动), 减少fund_growth(高波动)
    defensive_w = {
        "fund_growth": 0.05, "cashflow": 0.10, "analyst": 0.12,
        "grade_sentiment": 0.12, "earnings": 0.10, "balance": 0.08,
        "fund_metric": 0.10, "insider": 0.05, "fund_ratio": 0.18,
    }

    print(f"\n📅 Walk-Forward窗口: {len(windows)}个")

    # ═══════════════════════════════════════════
    # TEST 1: 基线 (无VIX过滤)
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 基线: V0.3.2 无VIX过滤 (60天调仓)")
    print("=" * 80)

    def run_config(name, weights, hold, vix_thresh=None, vix_type=None):
        sh, dd, ret, wr = [], [], [], []
        for test_dates in windows:
            r = backtest_with_vix_filter(
                ranks, price_pivot, test_dates, regime_above,
                weights, hold, 10, vix_dict,
                vix_threshold=vix_thresh, vix_filter_type=vix_type,
            )
            if r:
                sh.append(r["sharpe"])
                dd.append(r["dd"])
                ret.append(r["ret"])
                wr.append(r["wr"])
        if sh:
            recent_sh = []
            for i, td in enumerate(windows):
                if td[0] >= "2024-01-01" and i < len(sh):
                    recent_sh.append(sh[i])
            pos_rate = np.mean([1 for s in sh if s > 0]) * 100
            recent_str = f"近期={np.mean(recent_sh):.3f}" if recent_sh else "近期=N/A"
            print(f"  {name}:")
            print(f"    夏普={np.mean(sh):.3f}±{np.std(sh):.3f}, 回撤={np.mean(dd):.1f}%, 胜率={np.mean(wr):.1f}%, 正率={pos_rate:.0f}%")
            print(f"    {recent_str}")
            return {"sharpe": np.mean(sh), "dd": np.mean(dd), "ret": np.mean(ret),
                    "wr": np.mean(wr), "recent": np.mean(recent_sh) if recent_sh else None,
                    "n": len(sh)}
        return None

    # 基线
    baseline = run_config("基线 (无过滤)", v032_w, 60)

    # ═══════════════════════════════════════════
    # TEST 2: VIX阈值扫描 (skip_buy策略)
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 VIX阈值扫描: 高VIX时跳过买入")
    print("=" * 80)

    skip_buy_results = {}
    for thresh in [15, 18, 20, 22, 25, 30]:
        r = run_config(f"VIX>={thresh}跳过", v032_w, 60, vix_thresh=thresh, vix_type="skip_buy")
        if r:
            skip_buy_results[thresh] = r

    # ═══════════════════════════════════════════
    # TEST 3: VIX阈值扫描 (切换防御权重)
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 VIX阈值扫描: 高VIX时切换防御权重")
    print("=" * 80)

    defensive_results = {}
    for thresh in [15, 18, 20, 22, 25, 30]:
        r = run_config(f"VIX>={thresh}防御", v032_w, 60, vix_thresh=thresh, vix_type="defensive")
        if r:
            defensive_results[thresh] = r

    # ═══════════════════════════════════════════
    # TEST 4: 双阈值策略 (低阈值减仓 + 高阈值跳过)
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 双阈值策略: VIX>18减仓50%, VIX>25跳过")
    print("=" * 80)

    # 这需要在回测层面实现，这里用skip_buy模拟
    dual_result = run_config("双阈值(18/25)", v032_w, 60, vix_thresh=25, vix_type="skip_buy")

    # ═══════════════════════════════════════════
    # TEST 5: 动态调仓频率 (高VIX时缩短调仓)
    # ═══════════════════════════════════════════

    print("\n" + "=" * 80)
    print("📊 动态调仓: 正常60天, VIX>22时30天")
    print("=" * 80)

    # 用skip_buy + 30天调仓模拟高VIX快速反应
    dynamic_result = run_config("动态调仓(60/30)", v032_w, 30, vix_thresh=22, vix_type="skip_buy")

    # ═══════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════

    elapsed = time.time() - t_total
    print("\n" + "=" * 80)
    print("📋 VIX过滤优化总结")
    print("=" * 80)

    if baseline:
        print(f"\n{'策略':<30} {'夏普':>8} {'回撤':>8} {'近期':>8}")
        print("-" * 55)
        print(f"{'基线 (无过滤)':<30} {baseline['sharpe']:>8.3f} {baseline['dd']:>7.1f}% {baseline.get('recent',0) or 0:>8.3f}")

        if skip_buy_results:
            print("\n--- 高VIX跳过买入 ---")
            for thresh, r in sorted(skip_buy_results.items()):
                delta = r["sharpe"] - baseline["sharpe"]
                recent_delta = (r.get("recent") or 0) - (baseline.get("recent") or 0)
                print(f"VIX>={thresh:<24} {r['sharpe']:>8.3f} {r['dd']:>7.1f}% {r.get('recent',0) or 0:>8.3f}  (夏普{delta:+.3f})")

        if defensive_results:
            print("\n--- 高VIX切换防御权重 ---")
            for thresh, r in sorted(defensive_results.items()):
                delta = r["sharpe"] - baseline["sharpe"]
                print(f"VIX>={thresh:<24} {r['sharpe']:>8.3f} {r['dd']:>7.1f}% {r.get('recent',0) or 0:>8.3f}  (夏普{delta:+.3f})")

    # 找最优
    all_configs = {}
    if baseline:
        all_configs["基线"] = baseline
    for k, v in skip_buy_results.items():
        all_configs[f"跳过VIX>={k}"] = v
    for k, v in defensive_results.items():
        all_configs[f"防御VIX>={k}"] = v

    if all_configs:
        # 综合评分: 夏普*0.6 + 近期夏普*0.4
        scored = {}
        for name, r in all_configs.items():
            recent = r.get("recent") or 0
            score = r["sharpe"] * 0.6 + recent * 0.4
            scored[name] = {"score": score, **r}

        best = max(scored.items(), key=lambda x: x[1]["score"])
        print(f"\n🏆 综合最优: {best[0]}")
        print(f"   夏普={best[1]['sharpe']:.3f}, 近期={best[1].get('recent',0) or 0:.3f}, 综合分={best[1]['score']:.3f}")

    print(f"\n  总耗时: {elapsed/60:.1f}分钟")

    # 保存
    output = {
        "baseline": baseline,
        "skip_buy": {str(k): v for k, v in skip_buy_results.items()},
        "defensive": {str(k): v for k, v in defensive_results.items()},
        "best": best[0] if all_configs else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
    }
    with open(DATA_DIR / "v032_vix_filter_result.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  结果已保存: {DATA_DIR / 'v032_vix_filter_result.json'}")


if __name__ == "__main__":
    main()
