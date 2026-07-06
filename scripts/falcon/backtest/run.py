#!/usr/bin/env python3
"""
Falcon统一回测框架 — CLI入口

用法:
    # Walk-Forward回测(默认V0.4.6参数, 10年, Futu真实成本)
    python3 -m scripts.falcon.backtest.run
    
    # 单次回测
    python3 -m scripts.falcon.backtest.run --mode single
    
    # 参数覆盖
    python3 -m scripts.falcon.backtest.run --hold-days 45 --top-n 5 --cost flat
    
    # 指定日期范围
    python3 -m scripts.falcon.backtest.run --start 2016-01-01 --end 2026-01-01
    
    # 对比模式: 同时跑Futu成本和平坦成本
    python3 -m scripts.falcon.backtest.run --compare-costs
    
    # 启用行业限制
    python3 -m scripts.falcon.backtest.run --sector-limit 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 确保项目根目录在path中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: str = None, overrides: dict = None) -> dict:
    """加载配置文件, 应用命令行覆盖。"""
    if config_path is None:
        config_path = Path(__file__).parent / "backtest_config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if overrides:
        # 深度合并覆盖参数
        for key, value in overrides.items():
            if '.' in key:
                parts = key.split('.')
                d = config
                for p in parts[:-1]:
                    d = d.setdefault(p, {})
                d[parts[-1]] = value
            else:
                config[key] = value
    
    return config


def load_data(config: dict):
    """加载特征和价格数据。"""
    data_config = config.get('data', {})
    features_path = PROJECT_ROOT / data_config.get('features_path', 'data/falcon/features_v04_1.parquet')
    prices_path = PROJECT_ROOT / data_config.get('prices_path', 'data/falcon/us_prices_daily.parquet')
    
    print(f"📂 Loading data...")
    t0 = time.time()
    
    features = pd.read_parquet(features_path)
    features['date'] = features['date'].astype(str)
    
    prices_df = pd.read_parquet(prices_path)
    prices_df['date'] = prices_df['date'].astype(str)
    prices = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    
    print(f"  ✅ Features: {features.shape}, Prices: {prices.shape} ({time.time()-t0:.1f}s)")
    
    return features, prices


def load_ic_weights(config: dict):
    """加载IC权重(静态模式)。"""
    data_config = config.get('data', {})
    ic_path = PROJECT_ROOT / data_config.get('ic_weights_path', 'data/falcon/factor_ic_weights.json')
    
    if not ic_path.exists():
        print(f"⚠️ IC权重文件不存在: {ic_path}")
        return None
    
    with open(ic_path) as f:
        ic_data = json.load(f)
    
    print(f"  ✅ IC weights: computed_at={ic_data.get('computed_at')}, "
          f"lookback={ic_data.get('lookback')}, power={ic_data.get('power')}")
    
    return ic_data


def run_single_backtest(config: dict, features: pd.DataFrame, prices: pd.DataFrame,
                        ic_data=None):
    """运行单次回测。"""
    from .engine import BacktestEngine
    
    engine = BacktestEngine(config)
    
    ic_weights = ic_data.get('weights') if isinstance(ic_data, dict) else None
    
    result = engine.run(features, prices, ic_data=ic_weights)
    
    return result


def run_walk_forward(config: dict, features: pd.DataFrame, prices: pd.DataFrame,
                     ic_data=None):
    """运行Walk-Forward回测。"""
    from .walk_forward import WalkForwardValidator
    
    wf = WalkForwardValidator(config)
    
    ic_mode = config.get('scoring', {}).get('ic', {}).get('source', 'rolling')
    
    result = wf.run(features, prices,
                    ic_mode=ic_mode,
                    ic_weights_data=ic_data)
    
    return result


def print_result(result, label: str = ""):
    """打印回测结果。"""
    prefix = f"[{label}] " if label else ""
    print(f"\n{'='*70}")
    print(f"📊 {prefix}回测结果")
    print(f"{'='*70}")
    print(result.summary())
    
    if result.window_details:
        print(f"\n  Walk-Forward窗口明细:")
        for w in result.window_details:
            if 'sharpe' in w:
                print(f"    {w['period']}: Sharpe={w['sharpe']:.3f} "
                      f"CAGR={w['cagr']:.1%} MaxDD={w['max_dd']:.1%} "
                      f"Trades={w['n_trades']} Cost={w.get('total_cost_pct', 0):.2%}")
            elif 'error' in w:
                print(f"    {w.get('period', '?')}: ❌ {w['error']}")
    
    if result.yearly:
        print(f"\n  分年统计:")
        for year, y in sorted(result.yearly.items()):
            print(f"    {year}: Trades={y['trades']} "
                  f"WR={y['win_rate']:.0%} "
                  f"PnL=${y['total_pnl']:+,.0f} "
                  f"Cost=${y['costs']:,.0f}")
    
    if result.warnings:
        print(f"\n  ⚠️ 警告:")
        for w in result.warnings:
            print(f"    - {w}")


def main():
    parser = argparse.ArgumentParser(description="Falcon统一回测框架")
    parser.add_argument("--mode", choices=["single", "wf"], default="wf",
                       help="回测模式: single=单次, wf=Walk-Forward(默认)")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径(默认backtest_config.yaml)")
    parser.add_argument("--hold-days", type=int, default=None, help="持有天数")
    parser.add_argument("--top-n", type=int, default=None, help="选股数量")
    parser.add_argument("--stop-loss", type=float, default=None, help="止损线")
    parser.add_argument("--cost", choices=["futu_real", "flat"], default=None,
                       help="成本模型")
    parser.add_argument("--years", type=int, default=None, help="回测年数")
    parser.add_argument("--start", type=str, default=None, help="起始日期")
    parser.add_argument("--end", type=str, default=None, help="结束日期")
    parser.add_argument("--sector-limit", type=int, default=None,
                       help="行业限制(每行业最多N只)")
    parser.add_argument("--vix-filter", action="store_true", help="启用VIX过滤")
    parser.add_argument("--compare-costs", action="store_true",
                       help="对比Futu真实成本vs平坦成本")
    parser.add_argument("--save", type=str, default=None, help="保存结果路径")
    
    args = parser.parse_args()
    
    # 构建覆盖参数
    overrides = {}
    if args.hold_days:
        overrides['trading.hold_days'] = args.hold_days
    if args.top_n:
        overrides['trading.top_n'] = args.top_n
    if args.stop_loss:
        overrides['trading.stop_loss'] = args.stop_loss
    if args.cost:
        overrides['trading.cost_model'] = args.cost
    if args.years:
        overrides['backtest.years'] = args.years
    if args.start:
        overrides['backtest.start_date'] = args.start
    if args.end:
        overrides['backtest.end_date'] = args.end
    if args.sector_limit:
        overrides['sector_limit.enabled'] = True
        overrides['sector_limit.max_per_sector'] = args.sector_limit
    if args.vix_filter:
        overrides['vix_filter.enabled'] = True
    
    # 加载配置和数据
    config = load_config(args.config, overrides)
    features, prices = load_data(config)
    ic_data = load_ic_weights(config)
    
    # 运行
    t0 = time.time()
    
    if args.compare_costs:
        # 对比模式: 跑两组
        print("\n🔬 对比模式: Futu真实成本 vs 平坦成本")
        
        config_futu = load_config(args.config, {**overrides, 'trading.cost_model': 'futu_real'})
        config_flat = load_config(args.config, {**overrides, 'trading.cost_model': 'flat'})
        
        if args.mode == "wf":
            result_futu = run_walk_forward(config_futu, features, prices, ic_data)
            result_flat = run_walk_forward(config_flat, features, prices, ic_data)
        else:
            result_futu = run_single_backtest(config_futu, features, prices, ic_data)
            result_flat = run_single_backtest(config_flat, features, prices, ic_data)
        
        print_result(result_futu, "Futu真实成本")
        print_result(result_flat, "平坦成本(0.2%/侧)")
        
        # 对比
        print(f"\n{'='*70}")
        print(f"📊 成本对比")
        print(f"{'='*70}")
        print(f"{'指标':<20} {'Futu真实':>12} {'平坦0.2%':>12} {'差异':>12}")
        print(f"{'-'*56}")
        print(f"{'Sharpe':<20} {result_futu.sharpe:>12.3f} {result_flat.sharpe:>12.3f} "
              f"{result_futu.sharpe - result_flat.sharpe:>+12.3f}")
        print(f"{'CAGR':<20} {result_futu.cagr:>12.1%} {result_flat.cagr:>12.1%} "
              f"{result_futu.cagr - result_flat.cagr:>+12.1%}")
        print(f"{'MaxDD':<20} {result_futu.max_dd:>12.1%} {result_flat.max_dd:>12.1%} "
              f"{result_futu.max_dd - result_flat.max_dd:>+12.1%}")
        print(f"{'总成本':<20} {result_futu.total_cost_pct:>12.2%} {result_flat.total_cost_pct:>12.2%} "
              f"{result_futu.total_cost_pct - result_flat.total_cost_pct:>+12.2%}")
    else:
        # 单模式
        if args.mode == "wf":
            result = run_walk_forward(config, features, prices, ic_data)
        else:
            result = run_single_backtest(config, features, prices, ic_data)
        
        print_result(result)
    
    print(f"\n⏱️ 总耗时: {time.time()-t0:.0f}s")
    
    # 保存
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'sharpe': result.sharpe,
            'cagr': result.cagr,
            'max_dd': result.max_dd,
            'win_rate': result.win_rate,
            'n_trades': result.n_trades,
            'total_cost_pct': result.total_cost_pct,
            'warnings': result.warnings,
            'config': config,
        }
        
        if result.window_details:
            output['windows'] = result.window_details
        if result.yearly:
            output['yearly'] = result.yearly
        
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"💾 保存到: {save_path}")


if __name__ == "__main__":
    main()
