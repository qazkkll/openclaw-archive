#!/usr/bin/env python3
"""
RL环境评估器 — 对比多种交易策略（支持A股+美股）
===============================================
用历史数据跑回测，对比不同策略的表现。

用法：
  python run_eval.py                    # 默认美股（10只代表性股票）
  python run_eval.py --market us        # 美股多股票回测
  python run_eval.py --market cn        # A股（平安银行）
  python run_eval.py --market us --stocks AAPL,MSFT,NVDA  # 指定股票
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trading_env import TradingEnv


# ============================================================
# 策略定义
# ============================================================

def strategy_buy_and_hold(env, obs):
    """买入持有：第一天买满，之后一直hold"""
    if env.position_shares == 0:
        return TradingEnv.ACTION_BUY_100
    return TradingEnv.ACTION_HOLD


def strategy_random(env, obs):
    """随机交易：验证环境正确性"""
    return env.action_space.sample()


def strategy_rsi_reversion(env, obs):
    """RSI均值回归：RSI<30买，RSI>70卖"""
    rsi = obs[0] * 50 + 50  # 反归一化
    position_pct = obs[9]   # 当前仓位比例
    
    if rsi < 30 and position_pct < 0.5:
        return TradingEnv.ACTION_BUY_50
    elif rsi < 40 and position_pct < 0.25:
        return TradingEnv.ACTION_BUY_25
    elif rsi > 70 and position_pct > 0.5:
        return TradingEnv.ACTION_SELL_50
    elif rsi > 80 and position_pct > 0.25:
        return TradingEnv.ACTION_SELL_25
    return TradingEnv.ACTION_HOLD


def strategy_model_score(env, obs):
    """跟随模型信号：分数高买入，低卖出"""
    rsi = obs[0] * 50 + 50
    macd_hist = obs[1]
    ret5 = obs[4]
    position_pct = obs[9]
    
    # 综合信号
    signal = 0
    if macd_hist > 0:
        signal += 1
    if rsi < 45:
        signal += 1
    if ret5 > 0:
        signal += 1
    if rsi > 55:
        signal -= 1
    if macd_hist < 0:
        signal -= 1
    if ret5 < -0.02:
        signal -= 1
    
    if signal >= 2 and position_pct < 0.5:
        return TradingEnv.ACTION_BUY_50
    elif signal >= 1 and position_pct < 0.25:
        return TradingEnv.ACTION_BUY_25
    elif signal <= -2 and position_pct > 0.5:
        return TradingEnv.ACTION_SELL_50
    elif signal <= -1 and position_pct > 0.25:
        return TradingEnv.ACTION_SELL_25
    return TradingEnv.ACTION_HOLD


def strategy_momentum(env, obs):
    """追涨杀跌：涨了追，跌了跑"""
    ret5 = obs[4]
    ret10 = obs[5]
    position_pct = obs[9]
    
    if ret5 > 0.02 and ret10 > 0.03 and position_pct < 0.5:
        return TradingEnv.ACTION_BUY_50
    elif ret5 > 0.01 and position_pct < 0.25:
        return TradingEnv.ACTION_BUY_25
    elif ret5 < -0.02 and position_pct > 0.5:
        return TradingEnv.ACTION_SELL_50
    elif ret5 < -0.01 and position_pct > 0.25:
        return TradingEnv.ACTION_SELL_25
    return TradingEnv.ACTION_HOLD


def strategy_anti_momentum(env, obs):
    """逆向操作：别人恐惧我贪婪"""
    rsi = obs[0] * 50 + 50
    ret5 = obs[4]
    position_pct = obs[9]
    
    # RSI超卖 + 近期大跌 = 买入
    if rsi < 35 and ret5 < -0.03 and position_pct < 0.5:
        return TradingEnv.ACTION_BUY_75  # 越跌越买
    elif rsi < 45 and ret5 < -0.02 and position_pct < 0.25:
        return TradingEnv.ACTION_BUY_25
    # RSI超买 + 近期大涨 = 卖出
    elif rsi > 70 and ret5 > 0.05 and position_pct > 0.5:
        return TradingEnv.ACTION_SELL_75
    elif rsi > 65 and ret5 > 0.03 and position_pct > 0.25:
        return TradingEnv.ACTION_SELL_25
    return TradingEnv.ACTION_HOLD


def strategy_turtle(env, obs):
    """海龟交易：突破20日高点买入，跌破10日低点卖出"""
    ret20 = obs[6]  # 20日收益
    rsi = obs[0] * 50 + 50
    position_pct = obs[9]
    days_held = obs[11] * 20  # 反归一化
    
    # 趋势跟踪
    if ret20 > 0.05 and rsi < 70 and position_pct < 0.5:
        return TradingEnv.ACTION_BUY_50
    elif ret20 > 0.03 and position_pct < 0.25:
        return TradingEnv.ACTION_BUY_25
    # 止损
    elif position_pct > 0 and days_held > 5 and ret20 < -0.05:
        return TradingEnv.ACTION_SELL_100
    elif position_pct > 0.5 and ret20 < -0.03:
        return TradingEnv.ACTION_SELL_50
    return TradingEnv.ACTION_HOLD


def strategy_xgboost_score(env, obs):
    """真实XGBoost评分策略：用蓝盾+绿箭模型评分决策"""
    # obs[12] = bs_score_norm (蓝盾), obs[13] = ga_score_norm (绿箭)
    # obs[0] = RSI, obs[9] = position_pct
    rsi = obs[0] * 50 + 50
    position_pct = obs[9]
    
    # 检查是否有评分特征
    if len(obs) < 14:
        return strategy_model_score(env, obs)  # 降级到模拟策略
    
    bs_score = (obs[12] + 1) / 2  # 反归一化到[0,1]
    ga_score = (obs[13] + 1) / 2
    
    # 综合信号：蓝盾高分+RSI不超买 → 买入
    # 绿箭高分+RSI超卖 → 买入
    buy_signal = 0
    sell_signal = 0
    
    # 蓝盾信号（>$10股票）
    if bs_score > 0.55:  # 高于均值
        buy_signal += 2
    elif bs_score > 0.52:
        buy_signal += 1
    
    # 绿箭信号（<$10股票）
    if ga_score > 0.50:
        buy_signal += 1
    
    # RSI确认
    if rsi < 40:
        buy_signal += 1
    elif rsi > 60:
        sell_signal += 1
    
    # MACD确认
    macd_hist = obs[1]
    if macd_hist > 0:
        buy_signal += 1
    else:
        sell_signal += 1
    
    # 决策
    if buy_signal >= 3 and position_pct < 0.5:
        return TradingEnv.ACTION_BUY_50
    elif buy_signal >= 2 and position_pct < 0.25:
        return TradingEnv.ACTION_BUY_25
    elif sell_signal >= 2 and position_pct > 0.5:
        return TradingEnv.ACTION_SELL_50
    elif sell_signal >= 1 and position_pct > 0.75:
        return TradingEnv.ACTION_SELL_25
    return TradingEnv.ACTION_HOLD


# ============================================================
# 策略注册
# ============================================================

STRATEGIES = {
    "Buy & Hold": strategy_buy_and_hold,
    "Random": strategy_random,
    "RSI Reversion": strategy_rsi_reversion,
    "Model Score": strategy_model_score,
    "XGBoost Score": strategy_xgboost_score,
    "Momentum": strategy_momentum,
    "Anti-Momentum": strategy_anti_momentum,
    "Turtle": strategy_turtle,
}


# ============================================================
# 数据加载
# ============================================================

def load_a_share_data(sym: str = "000001") -> pd.DataFrame:
    """加载A股数据"""
    path = os.path.expanduser("~/.hermes/openclaw-archive/data/a_hist_10y.parquet")
    df = pd.read_parquet(path)
    df = df.rename(columns={"Code": "sym", "Date": "date", "O": "open", "H": "high", "L": "low", "C": "close", "V": "volume"})
    df["date"] = df["date"].astype(int)
    stock = df[df["sym"] == sym].sort_values("date").reset_index(drop=True)
    
    if len(stock) < 100:
        print(f"⚠️ {sym} 数据不足({len(stock)}行)，使用全部A股平均")
        avg = df.groupby("date").agg({"close": "mean", "high": "mean", "low": "mean", "open": "mean", "volume": "mean"}).reset_index()
        avg["sym"] = "MARKET_AVG"
        return avg
    
    return stock


def load_us_data(symbols: list = None) -> dict:
    """加载美股数据（含预计算模型评分），返回 {sym: DataFrame}"""
    path = os.path.expanduser("~/.hermes/openclaw-archive/data/us/us_hist_yf_10y.parquet")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 加载预计算模型评分
    scores_path = os.path.expanduser("~/.hermes/openclaw-archive/data/rl/model_scores_us.parquet")
    if os.path.exists(scores_path):
        scores_df = pd.read_parquet(scores_path)
        scores_df["date"] = pd.to_datetime(scores_df["date"])
        df = pd.merge(df, scores_df, on=["sym", "date"], how="left")
        for col in ["bs_score", "ga_score"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        print(f"  ✅ 已加载模型评分")
    else:
        print(f"  ⚠️ 模型评分文件不存在，跳过")
    
    # 默认选10只代表性股票（mega-cap + 不同sector）
    if symbols is None:
        symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "JNJ", "XOM", "UNH"]
    
    result = {}
    for sym in symbols:
        stock = df[df["sym"] == sym].sort_values("date").reset_index(drop=True)
        if len(stock) >= 500:  # 至少500天数据
            result[sym] = stock
        else:
            print(f"⚠️ {sym} 数据不足({len(stock)}行)，跳过")
    
    return result


# ============================================================
# 回测引擎
# ============================================================

def run_backtest(
    data: pd.DataFrame,
    strategy_fn,
    strategy_name: str,
    start_idx: int = 252,
    initial_cash: float = 100000.0,
    reward_type: str = "return",
    commission: float = 0.001,
    slippage: float = 0.002,
) -> dict:
    """运行单次回测"""
    env = TradingEnv(
        data=data,
        initial_cash=initial_cash,
        reward_type=reward_type,
        commission=commission,
        slippage=slippage,
    )
    
    obs, info = env.reset(start_idx=start_idx)
    total_reward = 0
    
    while True:
        action = strategy_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    summary = env.get_summary()
    summary["strategy"] = strategy_name
    summary["total_reward"] = round(total_reward, 4)
    
    return summary


def run_walk_forward(
    data: pd.DataFrame,
    strategy_fn,
    strategy_name: str,
    n_windows: int = 5,
    train_days: int = 504,
    test_days: int = 126,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.002,
) -> dict:
    """Walk-Forward回测（滚动窗口）"""
    results = []
    
    for i in range(n_windows):
        start = i * test_days + 252
        if start + test_days >= len(data) - 1:
            break
        
        env = TradingEnv(
            data=data,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
        )
        obs, info = env.reset(start_idx=start)
        
        while True:
            action = strategy_fn(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        results.append(env.get_summary())
    
    if not results:
        return {"strategy": strategy_name, "error": "no valid windows"}
    
    # 汇总
    avg_return = np.mean([r["total_return_pct"] for r in results])
    avg_sharpe = np.mean([r["sharpe_ratio"] for r in results])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in results])
    avg_trades = np.mean([r["total_trades"] for r in results])
    avg_alpha = np.mean([r["alpha_pct"] for r in results])
    
    return {
        "strategy": strategy_name,
        "windows": len(results),
        "avg_return_pct": round(avg_return, 2),
        "avg_sharpe": round(avg_sharpe, 3),
        "avg_max_dd_pct": round(avg_dd, 2),
        "avg_trades": round(avg_trades, 1),
        "avg_alpha_pct": round(avg_alpha, 2),
        "win_windows": sum(1 for r in results if r["total_return_pct"] > 0),
        "window_results": results,
    }


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RL Trading Strategy Evaluator")
    parser.add_argument("--market", choices=["us", "cn"], default="us", help="Market: us or cn")
    parser.add_argument("--stocks", type=str, default=None, help="Comma-separated stock symbols (e.g. AAPL,MSFT)")
    parser.add_argument("--skip-random", action="store_true", help="Skip Random strategy")
    parser.add_argument("--single-only", action="store_true", help="Skip Walk-Forward, single window only")
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"TradingRL — 离线策略评估 ({args.market.upper()} Market)")
    print("=" * 70)
    
    # 加载数据
    print(f"\n📊 加载{args.market.upper()}数据...")
    
    symbols = args.stocks.split(",") if args.stocks else None
    
    if args.market == "us":
        stocks_data = load_us_data(symbols)
        # 美股交易成本：极低（零佣金时代）
        commission = 0.0001   # 0.01%
        slippage = 0.0003     # 0.03%（大盘股流动性好）
        initial_cash = 100000.0  # $100K
    else:
        stock = load_a_share_data(symbols[0] if symbols else "000001")
        stocks_data = {stock["sym"].iloc[0]: stock}
        # A股交易成本
        commission = 0.001    # 0.1%
        slippage = 0.002      # 0.2%
        initial_cash = 100000.0  # ¥10万
    
    print(f"  股票数: {len(stocks_data)}")
    for sym, df in stocks_data.items():
        print(f"  {sym}: {len(df)} 行 | {df['date'].min().date() if hasattr(df['date'].min(), 'date') else df['date'].min()} ~ {df['date'].max().date() if hasattr(df['date'].max(), 'date') else df['date'].max()}")
    print(f"  交易成本: commission={commission*100:.2f}%, slippage={slippage*100:.2f}%")
    
    # 策略列表
    strategies = {k: v for k, v in STRATEGIES.items() if not (args.skip_random and k == "Random")}
    
    # ============================================================
    # 多股票汇总结果
    # ============================================================
    all_stock_results = {}  # {sym: {strategy_name: result}}
    
    for sym, data in stocks_data.items():
        print(f"\n{'='*70}")
        print(f"📈 {sym} — 单窗口回测 (从2023年开始)")
        print(f"{'='*70}")
        
        # 找2023年起始位置
        if args.market == "us":
            mask = data["date"] >= "2023-01-01"
        else:
            mask = data["date"] >= 20230101
        start_idx = data[mask].index[0] if mask.any() else 252
        
        stock_results = {}
        for name, fn in strategies.items():
            t0 = time.time()
            result = run_backtest(
                data, fn, name, start_idx=start_idx,
                initial_cash=initial_cash,
                commission=commission, slippage=slippage,
            )
            elapsed = time.time() - t0
            stock_results[name] = result
            
            # 格式化输出
            print(f"  {name:20s} | 收益: {result['total_return_pct']:>8.2f}% | Sharpe: {result['sharpe_ratio']:>7.3f} | DD: {result['max_drawdown_pct']:>7.2f}% | Alpha: {result['alpha_pct']:>7.2f}% | 交易: {result['total_trades']:>4d} | {elapsed:.2f}s")
        
        all_stock_results[sym] = stock_results
    
    # ============================================================
    # Walk-Forward（可选）
    # ============================================================
    if not args.single_only:
        print(f"\n{'='*70}")
        print(f"📈 Walk-Forward回测 (5个窗口，每窗口6个月)")
        print(f"{'='*70}")
        
        wf_all = {}  # {sym: {strategy_name: wf_result}}
        
        for sym, data in stocks_data.items():
            print(f"\n  --- {sym} ---")
            wf_all[sym] = {}
            
            for name, fn in strategies.items():
                if name == "Random":
                    continue
                
                t0 = time.time()
                result = run_walk_forward(
                    data, fn, name, n_windows=5,
                    initial_cash=initial_cash,
                    commission=commission, slippage=slippage,
                )
                elapsed = time.time() - t0
                wf_all[sym][name] = result
                
                if "error" not in result:
                    print(f"    {name:20s} | 收益: {result['avg_return_pct']:>7.2f}% | Sharpe: {result['avg_sharpe']:>6.3f} | DD: {result['avg_max_dd_pct']:>6.2f}% | Alpha: {result['avg_alpha_pct']:>6.2f}% | 胜率: {result['win_windows']}/{result['windows']} | {elapsed:.2f}s")
                else:
                    print(f"    {name:20s} | ❌ {result['error']}")
    
    # ============================================================
    # 跨股票汇总排名
    # ============================================================
    print(f"\n{'='*70}")
    print(f"🏆 跨股票策略汇总 (按平均Walk-Forward Sharpe)")
    print(f"{'='*70}")
    
    if not args.single_only:
        # 汇总WF结果
        strategy_agg = {}
        for name in strategies:
            if name == "Random":
                continue
            sharpes = []
            returns = []
            alphas = []
            dds = []
            win_rates = []
            for sym in wf_all:
                if name in wf_all[sym] and "error" not in wf_all[sym][name]:
                    r = wf_all[sym][name]
                    sharpes.append(r["avg_sharpe"])
                    returns.append(r["avg_return_pct"])
                    alphas.append(r["avg_alpha_pct"])
                    dds.append(r["avg_max_dd_pct"])
                    win_rates.append(r["win_windows"] / r["windows"])
            
            if sharpes:
                strategy_agg[name] = {
                    "avg_sharpe": round(np.mean(sharpes), 3),
                    "avg_return_pct": round(np.mean(returns), 2),
                    "avg_alpha_pct": round(np.mean(alphas), 2),
                    "avg_max_dd_pct": round(np.mean(dds), 2),
                    "avg_win_rate": round(np.mean(win_rates) * 100, 1),
                    "stocks_tested": len(sharpes),
                }
        
        # 排序
        ranked = sorted(strategy_agg.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True)
        
        print(f"\n  {'排名':<4} {'策略':<20} {'Sharpe':>8} {'收益':>8} {'Alpha':>8} {'回撤':>8} {'胜率':>6} {'股票数':>6}")
        print(f"  {'-'*72}")
        for i, (name, agg) in enumerate(ranked):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"  {medal} {i+1:<3} {name:<20} {agg['avg_sharpe']:>8.3f} {agg['avg_return_pct']:>7.2f}% {agg['avg_alpha_pct']:>7.2f}% {agg['avg_max_dd_pct']:>7.2f}% {agg['avg_win_rate']:>5.1f}% {agg['stocks_tested']:>5d}")
    else:
        # 单窗口汇总
        strategy_agg = {}
        for name in strategies:
            sharpes = []
            returns = []
            alphas = []
            for sym in all_stock_results:
                if name in all_stock_results[sym]:
                    r = all_stock_results[sym][name]
                    sharpes.append(r["sharpe_ratio"])
                    returns.append(r["total_return_pct"])
                    alphas.append(r["alpha_pct"])
            
            if sharpes:
                strategy_agg[name] = {
                    "avg_sharpe": round(np.mean(sharpes), 3),
                    "avg_return_pct": round(np.mean(returns), 2),
                    "avg_alpha_pct": round(np.mean(alphas), 2),
                }
        
        ranked = sorted(strategy_agg.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True)
        print(f"\n  {'排名':<4} {'策略':<20} {'Sharpe':>8} {'收益':>8} {'Alpha':>8}")
        print(f"  {'-'*50}")
        for i, (name, agg) in enumerate(ranked):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"  {medal} {i+1:<3} {name:<20} {agg['avg_sharpe']:>8.3f} {agg['avg_return_pct']:>7.2f}% {agg['avg_alpha_pct']:>7.2f}%")
    
    # ============================================================
    # 每只股票最佳策略
    # ============================================================
    print(f"\n{'='*70}")
    print(f"📊 每只股票最佳策略")
    print(f"{'='*70}")
    
    for sym in all_stock_results:
        best_name = None
        best_sharpe = -999
        for name, r in all_stock_results[sym].items():
            if name == "Random":
                continue
            if r["sharpe_ratio"] > best_sharpe:
                best_sharpe = r["sharpe_ratio"]
                best_name = name
        if best_name:
            r = all_stock_results[sym][best_name]
            bh = all_stock_results[sym]["Buy & Hold"]
            print(f"  {sym:6s} → {best_name:20s} | Sharpe: {r['sharpe_ratio']:>7.3f} | 收益: {r['total_return_pct']:>7.2f}% | B&H: {bh['total_return_pct']:>7.2f}%")
    
    # ============================================================
    # 保存结果
    # ============================================================
    output_dir = os.path.expanduser("~/.hermes/openclaw-archive/data/rl")
    os.makedirs(output_dir, exist_ok=True)
    
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "market": args.market,
        "stocks": list(stocks_data.keys()),
        "commission": commission,
        "slippage": slippage,
        "single_window": {
            sym: [{k: v for k, v in r.items() if k != "trades"} for r in results.values()]
            for sym, results in all_stock_results.items()
        },
    }
    
    if not args.single_only:
        save_data["walk_forward"] = {
            sym: {
                name: {k: v for k, v in r.items() if k != "window_results"}
                for name, r in wf_results.items()
            }
            for sym, wf_results in wf_all.items()
        }
        save_data["strategy_summary"] = strategy_agg
    
    output_path = os.path.join(output_dir, f"eval_results_{args.market}.json")
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n✅ 结果已保存: {output_path}")
    print(f"\n{'='*70}")
    print("评估完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
