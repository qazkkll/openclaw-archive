#!/usr/bin/env python3
"""
🦅 Falcon V0.3 — 主回测: 5种调仓策略 × 全量FMP因子 × Futu成本
"""
import pandas as pd, numpy as np, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from falcon_v03_engine import precompute_pit_ranks, backtest_flexible

DATA_DIR = Path("/home/hermes/.hermes/openclaw-archive/data/falcon")


def load_all_data():
    """加载所有数据源。"""
    master = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    master["date"] = master["date"].astype(str)
    
    data = {}
    for name in ["fmp_ratios_historical", "analyst_historical", "fmp_key_metrics",
                  "fmp_financial_growth", "fmp_insider", "fmp_dcf", "fmp_price_target"]:
        f = DATA_DIR / f"{name}.json"
        if f.exists():
            with open(f) as fh:
                data[name] = json.load(fh)
            print(f"  ✅ {name}: {len(data[name])} 只")
        else:
            data[name] = {}
            print(f"  ❌ {name}: 不存在")
    
    return master, data


def main():
    t0 = time.time()
    print("=" * 100)
    print("🦅 Falcon V0.3 — 全量FMP因子 + 灵活调仓 + Futu成本")
    print("=" * 100)
    
    master, data = load_all_data()
    print(f"\n📊 {len(master)} 行, {master['ticker'].nunique()} 只")
    
    # 预计算PIT rank
    ranks_dict = precompute_pit_ranks(
        master, 
        data["fmp_ratios_historical"], 
        data["analyst_historical"],
        data["fmp_key_metrics"],
        data["fmp_financial_growth"],
        data["fmp_insider"],
        data["fmp_dcf"],
        data["fmp_price_target"]
    )
    
    # 价格矩阵 + regime
    price_pivot = master.pivot_table(index="date", columns="ticker", values="close").sort_index()
    mkt_ret = price_pivot.pct_change(fill_method=None).mean(axis=1)
    mkt_price = (1 + mkt_ret).cumprod()
    mkt_ma200 = mkt_price.rolling(200, min_periods=100).mean()
    regime_above = (mkt_price > mkt_ma200).astype(int)
    
    bull_dates = sorted([d for d in ranks_dict if "2023" in d or "2024" in d])
    bear_dates = sorted([d for d in ranks_dict if "2022" in d])
    
    # ═══════════════════════════════════════════════════
    # 定义权重组合
    # ═══════════════════════════════════════════════════
    weight_configs = {
        "V0.2(Fund+Ana)": {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1},
        "Pure_Fund": {"tech": 0.0, "fund_ratio": 0.5, "fund_metric": 0.3, "fund_growth": 0.2},
        "Fund+Insider": {"fund_ratio": 0.5, "fund_metric": 0.2, "insider": 0.2, "analyst": 0.1},
        "Full_FMP": {"fund_ratio": 0.3, "fund_metric": 0.2, "fund_growth": 0.15, "analyst": 0.1, "insider": 0.1, "valuation": 0.1, "tech": 0.05},
        "Insider_Heavy": {"insider": 0.4, "fund_ratio": 0.3, "analyst": 0.2, "valuation": 0.1},
        "Valuation": {"valuation": 0.4, "fund_ratio": 0.3, "analyst": 0.2, "tech": 0.1},
    }
    
    # 定义调仓策略
    strategy_configs = {
        "Fixed_30d": {"strategy": "fixed", "params": {"hold_days": 30}},
        "Fixed_60d": {"strategy": "fixed", "params": {"hold_days": 60}},
        "Signal_5d": {"strategy": "signal", "params": {"check_every": 5, "rank_threshold": 0.5}},
        "Signal_10d": {"strategy": "signal", "params": {"check_every": 10, "rank_threshold": 0.4}},
        "Hybrid_20d": {"strategy": "hybrid", "params": {"check_every": 20, "rank_threshold": 0.3, "hold_min": 10}},
        "Adaptive": {"strategy": "adaptive", "params": {"base_hold": 30, "vol_factor": 2.0}},
    }
    
    # ═══════════════════════════════════════════════════
    # 全组合测试
    # ═══════════════════════════════════════════════════
    results = []
    total = len(weight_configs) * len(strategy_configs) * 3  # 3 stop-loss levels
    done = 0
    
    print(f"\n🔍 测试 {len(weight_configs)} 权重 × {len(strategy_configs)} 策略 × 3 止损 = {total} 组合")
    print(f"{'='*120}")
    
    for w_name, weights in weight_configs.items():
        for s_name, s_config in strategy_configs.items():
            for sl in [-0.10, -0.15, -0.25]:
                for bear_alloc in [0.30, 0.50]:
                    params = dict(s_config["params"])
                    params["stop_loss"] = sl
                    params["bear_alloc"] = bear_alloc
                    
                    bull = backtest_flexible(ranks_dict, price_pivot, bull_dates, regime_above,
                                            weights, s_config["strategy"], params)
                    bear = backtest_flexible(ranks_dict, price_pivot, bear_dates, regime_above,
                                            weights, s_config["strategy"], params)
                    
                    done += 1
                    if bull and bear:
                        passed = bear["dd"] <= 28 and bear["wr"] >= 42
                        results.append({
                            "weight": w_name, "strategy": s_name,
                            "stop_loss": sl, "bear_alloc": bear_alloc,
                            "bull_sr": bull["sharpe"], "bull_dd": bull["dd"], "bull_ret": bull["ret"],
                            "bear_sr": bear["sharpe"], "bear_dd": bear["dd"], "bear_wr": bear["wr"],
                            "bear_trades": bear["trades"],
                            "bull_rebal": bull.get("rebalances", 0),
                            "bear_rebal": bear.get("rebalances", 0),
                            "passed": passed,
                        })
                    
                    if done % 20 == 0:
                        n_pass = sum(1 for r in results if r["passed"])
                        print(f"  {done}/{total}: {n_pass} 通过")
    
    rdf = pd.DataFrame(results)
    passed = rdf[rdf["passed"] == True].sort_values("bull_sr", ascending=False)
    
    print(f"\n{'='*120}")
    print(f"📊 结果: {len(passed)}/{len(rdf)} 通过 Falcon协议 (DD≤28% & WR≥42%)")
    print(f"{'='*120}")
    
    if len(passed) > 0:
        print(f"\n🏆 Top-15 (按牛市Sharpe排序):")
        print(f"{'权重':20} {'策略':12} {'SL':5} {'Bear':5} | {'牛SR':7} {'DD':6} {'Ret':7} | {'熊SR':7} {'DD':6} {'WR':5} {'Tr':5} {'Rebal':5}")
        for _, r in passed.head(15).iterrows():
            print(f"{r['weight']:20} {r['strategy']:12} {r['stop_loss']:.0%} {r['bear_alloc']:.0%} | "
                  f"{r['bull_sr']:7.3f} {r['bull_dd']:5.1f}% {r['bull_ret']:6.0f}% | "
                  f"{r['bear_sr']:7.3f} {r['bear_dd']:5.1f}% {r['bear_wr']:4.0f}% {r['bear_trades']:5} {r['bear_rebal']:5}")
        
        # 最优配置详情
        best = passed.iloc[0]
        print(f"\n🎯 最优配置:")
        print(f"  权重: {best['weight']}")
        print(f"  策略: {best['strategy']}")
        print(f"  止损: {best['stop_loss']:.0%}")
        print(f"  熊市仓位: {best['bear_alloc']:.0%}")
        print(f"  牛市: Sharpe={best['bull_sr']:.3f}, DD={best['bull_dd']:.0f}%, Ret={best['bull_ret']:.0f}%")
        print(f"  熊市: Sharpe={best['bear_sr']:.3f}, DD={best['bear_dd']:.0f}%, WR={best['bear_wr']:.0f}%")
    else:
        # 最接近通过
        rdf["dd_gap"] = (rdf["bear_dd"] - 28).clip(lower=0)
        rdf["wr_gap"] = (42 - rdf["bear_wr"]).clip(lower=0)
        rdf["total_gap"] = rdf["dd_gap"] + rdf["wr_gap"]
        closest = rdf.sort_values("total_gap").head(10)
        print(f"\n⚠️ 无组合通过。最接近:")
        for _, r in closest.iterrows():
            dd_ok = "✅" if r["bear_dd"] <= 28 else f"❌+{r['dd_gap']:.0f}"
            wr_ok = "✅" if r["bear_wr"] >= 42 else f"❌-{r['wr_gap']:.0f}"
            print(f"  {r['weight']:20} {r['strategy']:12} SL={r['stop_loss']:.0%} → "
                  f"Bull_SR={r['bull_sr']:.3f} DD={dd_ok} WR={wr_ok}")
    
    # ── 因子增量测试 ──
    print(f"\n{'='*120}")
    print("📊 因子增量贡献测试 (固定策略=Signal_5d, SL=-15%, Bear=50%)")
    print(f"{'='*120}")
    
    base_strategy = "signal"
    base_params = {"check_every": 5, "rank_threshold": 0.5, "stop_loss": -0.15, "bear_alloc": 0.50}
    
    factor_tests = {
        "Tech only": {"tech": 1.0},
        "Fund_ratio only": {"fund_ratio": 1.0},
        "Fund_metric only": {"fund_metric": 1.0},
        "Fund_growth only": {"fund_growth": 1.0},
        "Analyst only": {"analyst": 1.0},
        "Insider only": {"insider": 1.0},
        "Valuation only": {"valuation": 1.0},
        "V0.2 baseline": {"tech": 0.0, "fund_ratio": 0.7, "analyst": 0.2, "fund_metric": 0.1},
        "+metric": {"fund_ratio": 0.5, "fund_metric": 0.3, "analyst": 0.2},
        "+growth": {"fund_ratio": 0.4, "fund_metric": 0.2, "fund_growth": 0.2, "analyst": 0.2},
        "+insider": {"fund_ratio": 0.4, "fund_metric": 0.2, "insider": 0.2, "analyst": 0.2},
        "+valuation": {"fund_ratio": 0.4, "fund_metric": 0.2, "valuation": 0.2, "analyst": 0.2},
        "ALL factors": {"tech": 0.05, "fund_ratio": 0.3, "fund_metric": 0.2, "fund_growth": 0.1,
                        "analyst": 0.1, "insider": 0.15, "valuation": 0.1},
    }
    
    for f_name, weights in factor_tests.items():
        bull = backtest_flexible(ranks_dict, price_pivot, bull_dates, regime_above,
                                weights, base_strategy, base_params)
        bear = backtest_flexible(ranks_dict, price_pivot, bear_dates, regime_above,
                                weights, base_strategy, base_params)
        if bull and bear:
            passed_str = "✅" if bear["dd"] <= 28 and bear["wr"] >= 42 else "❌"
            print(f"  {f_name:20} → Bull_SR={bull['sharpe']:7.3f} DD={bull['dd']:5.1f}% | "
                  f"Bear_SR={bear['sharpe']:7.3f} DD={bear['dd']:5.1f}% WR={bear['wr']:4.0f}% {passed_str}")
    
    print(f"\n⏱️ {time.time()-t0:.0f}秒")
    
    # 保存
    if len(rdf) > 0:
        rdf.to_csv(DATA_DIR / "falcon_v03_results.csv", index=False)
        print(f"💾 保存: data/falcon/falcon_v03_results.csv")


if __name__ == "__main__":
    main()
