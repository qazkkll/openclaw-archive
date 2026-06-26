#!/usr/bin/env python3
"""
Paper Trading Simulator — 分钟级模拟交易
用法: python3 paper_trade.py [--days 5] [--model blueshield_v10|arrow_v12]
"""
import json
import sys
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# ─── 配置 ───────────────────────────────────────────────────
MODEL_CONFIGS = {
    "blueshield_v10": {
        "score_file_pattern": "data/us/blueshield_v10_scored_*.json",
        "universe": "blue_shield",
        "hold_days": 10,
        "top_n": 15,
        "exit_strategy": "trailing_stop",
        "trailing_pct": 0.15,
        "min_price": 10.0,
    },
    "arrow_v12": {
        "score_file_pattern": "data/us/arrow_v12_scored_*.json",
        "universe": "green_arrow",
        "hold_days": 20,
        "top_n": 5,
        "exit_strategy": "fixed_hold",
        "trailing_pct": None,
        "min_price": 1.0,
        "max_price": 10.0,
    },
}


def load_latest_scores(model_name: str) -> list[dict]:
    """加载最新评分文件"""
    pattern = MODEL_CONFIGS[model_name]["score_file_pattern"]
    import glob
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        print(f"⚠️ 未找到评分文件: {pattern}")
        return []
    latest = files[-1]
    with open(latest) as f:
        data = json.load(f)
    # 支持两种格式: {picks: [...]} 或 {results: [...]} 或直接 [...]
    if isinstance(data, dict):
        return data.get("picks", data.get("results", []))
    return data


def fetch_minute_data(symbol: str, period: str = "5d") -> pd.DataFrame:
    """拉分钟级K线"""
    import yfinance as yf
    try:
        df = yf.download(symbol, period=period, interval="1m", progress=False)
        if df.empty:
            return pd.DataFrame()
        # 标准化列名
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        return pd.DataFrame()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_vwap(df: pd.DataFrame) -> pd.Series:
    """计算日内VWAP"""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tp_vol = (typical * df["Volume"]).cumsum()
    return cum_tp_vol / cum_vol.replace(0, np.nan)


class Position:
    def __init__(self, symbol: str, entry_price: float, entry_time: datetime,
                 shares: int, strategy: str, trailing_pct: float = None):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.shares = shares
        self.strategy = strategy
        self.trailing_pct = trailing_pct
        self.highest_price = entry_price
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None

    def update(self, current_price: float, current_time: datetime, hold_days: int) -> bool:
        """更新持仓，返回是否应该卖出"""
        self.highest_price = max(self.highest_price, current_price)

        # 检查持有天数
        days_held = (current_time - self.entry_time).days
        if days_held >= hold_days:
            self.exit_price = current_price
            self.exit_time = current_time
            self.exit_reason = "hold_expired"
            return True

        # Trailing stop
        if self.strategy == "trailing_stop" and self.trailing_pct:
            drawdown = (self.highest_price - current_price) / self.highest_price
            if drawdown >= self.trailing_pct:
                self.exit_price = current_price
                self.exit_time = current_time
                self.exit_reason = "trailing_stop"
                return True

        # 硬止损 -15%
        loss_pct = (current_price - self.entry_price) / self.entry_price
        if loss_pct <= -0.15:
            self.exit_price = current_price
            self.exit_time = current_time
            self.exit_reason = "hard_stop"
            return True

        return False

    @property
    def pnl_pct(self) -> float:
        if self.exit_price:
            return (self.exit_price - self.entry_price) / self.entry_price
        return 0.0

    @property
    def pnl_amount(self) -> float:
        if self.exit_price:
            return (self.exit_price - self.entry_price) * self.shares
        return 0.0


class PaperTrader:
    def __init__(self, model_name: str, initial_capital: float = 100000,
                 period: str = "5d", interval: str = "1m"):
        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.period = period
        self.interval = interval

        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.trade_log: list[dict] = []
        self.daily_snapshots: list[dict] = []

    def run(self):
        """运行模拟"""
        print(f"\n{'='*60}")
        print(f"📊 Paper Trading — {self.model_name}")
        print(f"{'='*60}")
        print(f"初始资金: ${self.initial_capital:,.0f}")
        print(f"策略: {self.config['exit_strategy']}")
        print(f"持有天数: {self.config['hold_days']}")
        print(f"Top-N: {self.config['top_n']}")
        print()

        # 1. 加载评分
        scores = load_latest_scores(self.model_name)
        if not scores:
            print("❌ 无评分数据")
            return

        # 按score排序取Top-N
        if isinstance(scores[0], dict) and "score" in scores[0]:
            scores.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_stocks = scores[:self.config["top_n"]]
        symbols = [s.get("sym", s.get("symbol", s.get("ticker", ""))) for s in top_stocks]
        symbols = [s for s in symbols if s]
        print(f"模型推荐 Top-{self.config['top_n']}: {symbols[:5]}...")

        # 2. 拉分钟数据
        print(f"\n拉取分钟数据 ({self.period})...")
        all_data = {}
        for sym in symbols[:self.config["top_n"]]:
            df = fetch_minute_data(sym, self.period)
            if not df.empty and len(df) > 100:
                df["RSI"] = calc_rsi(df["Close"])
                df["VWAP"] = calc_vwap(df)
                all_data[sym] = df
                print(f"  ✅ {sym}: {len(df)} bars")
            else:
                print(f"  ⚠️ {sym}: 数据不足")

        if not all_data:
            print("❌ 无可用数据")
            return

        # 3. 模拟交易
        print(f"\n开始模拟交易...")
        # 对齐所有股票的时间戳
        all_times = set()
        for df in all_data.values():
            all_times.update(df.index.tolist())
        all_times = sorted(all_times)

        position_size = self.initial_capital / self.config["top_n"]

        for t in all_times:
            # 检查现有持仓
            to_close = []
            for pos in self.positions:
                if pos.symbol in all_data and t in all_data[pos.symbol].index:
                    row = all_data[pos.symbol].loc[t]
                    current_price = float(row["Close"])
                    if pos.update(current_price, t, self.config["hold_days"]):
                        to_close.append(pos)
                        self.capital += pos.exit_price * pos.shares
                        self.trade_log.append({
                            "time": str(t),
                            "symbol": pos.symbol,
                            "action": "SELL",
                            "price": pos.exit_price,
                            "shares": pos.shares,
                            "pnl_pct": f"{pos.pnl_pct:.2%}",
                            "reason": pos.exit_reason,
                        })

            for pos in to_close:
                self.closed_positions.append(pos)
                self.positions.remove(pos)

            # 开新仓（只在每天开盘后30分钟内开仓）
            if hasattr(t, 'time') and t.time() < pd.Timestamp("10:00").time():
                for sym in all_data:
                    if sym in [p.symbol for p in self.positions]:
                        continue
                    if len(self.positions) >= self.config["top_n"]:
                        break

                    df = all_data[sym]
                    if t not in df.index:
                        continue

                    row = df.loc[t]
                    price = float(row["Close"])
                    rsi = float(row["RSI"]) if not pd.isna(row["RSI"]) else 50
                    vol = float(row["Volume"])

                    # 入场条件: RSI < 70 + 成交量 > 均值
                    vol_mean = df["Volume"].rolling(60).mean()
                    vol_avg = float(vol_mean.loc[t]) if t in vol_mean.index and not pd.isna(vol_mean.loc[t]) else vol

                    if rsi < 70 and vol > vol_avg * 0.5:
                        shares = int(position_size / price)
                        if shares > 0:
                            pos = Position(
                                symbol=sym,
                                entry_price=price,
                                entry_time=t,
                                shares=shares,
                                strategy=self.config["exit_strategy"],
                                trailing_pct=self.config.get("trailing_pct"),
                            )
                            self.positions.append(pos)
                            self.capital -= price * shares
                            self.trade_log.append({
                                "time": str(t),
                                "symbol": sym,
                                "action": "BUY",
                                "price": price,
                                "shares": shares,
                                "rsi": f"{rsi:.1f}",
                            })

        # 4. 强制平仓剩余持仓
        for pos in self.positions:
            sym = pos.symbol
            if sym in all_data:
                last_price = float(all_data[sym]["Close"].iloc[-1])
                pos.exit_price = last_price
                pos.exit_time = all_data[sym].index[-1]
                pos.exit_reason = "simulation_end"
                self.capital += last_price * pos.shares
                self.closed_positions.append(pos)

        self.positions = []

        # 5. 输出结果
        self._print_results()

    def _print_results(self):
        """打印结果"""
        total_trades = len(self.closed_positions)
        if total_trades == 0:
            print("\n❌ 没有执行任何交易")
            return

        wins = [p for p in self.closed_positions if p.pnl_pct > 0]
        losses = [p for p in self.closed_positions if p.pnl_pct <= 0]

        total_pnl = self.capital - self.initial_capital
        total_return = total_pnl / self.initial_capital

        avg_win = np.mean([p.pnl_pct for p in wins]) if wins else 0
        avg_loss = np.mean([p.pnl_pct for p in losses]) if losses else 0

        # 计算夏普
        returns = [p.pnl_pct for p in self.closed_positions]
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0

        # 按退出原因统计
        reasons = {}
        for p in self.closed_positions:
            r = p.exit_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1

        # 按股票统计
        by_stock = {}
        for p in self.closed_positions:
            if p.symbol not in by_stock:
                by_stock[p.symbol] = {"trades": 0, "pnl": 0}
            by_stock[p.symbol]["trades"] += 1
            by_stock[p.symbol]["pnl"] += p.pnl_pct

        print(f"\n{'='*60}")
        print(f"📊 Paper Trading 结果 — {self.model_name}")
        print(f"{'='*60}")
        print(f"初始资金:    ${self.initial_capital:>12,.0f}")
        print(f"最终资金:    ${self.capital:>12,.0f}")
        print(f"总盈亏:      ${total_pnl:>12,.0f} ({total_return:+.2%})")
        print(f"总交易:      {total_trades:>12}")
        print(f"胜率:        {len(wins)/total_trades:>12.1%}")
        print(f"平均盈利:    {avg_win:>12.2%}")
        print(f"平均亏损:    {avg_loss:>12.2%}")
        print(f"盈亏比:      {abs(avg_win/avg_loss) if avg_loss != 0 else float('inf'):>12.2f}")
        print(f"夏普(模拟):  {sharpe:>12.2f}")
        print()

        print("退出原因:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

        print("\n按股票:")
        for sym, data in sorted(by_stock.items(), key=lambda x: -x[1]["pnl"]):
            print(f"  {sym}: {data['trades']}笔, 累计{data['pnl']:+.2%}")

        # 保存结果
        output = {
            "model": self.model_name,
            "period": self.period,
            "initial_capital": self.initial_capital,
            "final_capital": round(self.capital, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return": round(total_return, 4),
            "total_trades": total_trades,
            "win_rate": round(len(wins) / total_trades, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "sharpe": round(sharpe, 4),
            "exit_reasons": reasons,
            "trades": self.trade_log,
        }
        out_path = ROOT / "data" / f"paper_trade_{self.model_name}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\n结果已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Simulator")
    parser.add_argument("--model", default="blueshield_v10",
                        choices=["blueshield_v10", "arrow_v12"])
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--period", default="5d",
                        help="yfinance period: 1d/5d/1mo")
    args = parser.parse_args()

    trader = PaperTrader(
        model_name=args.model,
        initial_capital=args.capital,
        period=args.period,
    )
    trader.run()


if __name__ == "__main__":
    main()
