"""
TradingRL — 交易强化学习环境
============================
基于gymnasium的离线交易模拟环境。

设计思路：
  - 不是从零学选股，而是学"什么时候买/卖/持有"
  - XGBoost已经会选股，RL学的是仓位管理和时机
  - 用历史数据当模拟器，零风险离线评估

State（观察空间）：
  - market_features: RSI, MACD, ATR%, vol_ratio, ret5, ret10, ret20, breadth
  - portfolio_state: cash_pct, position_pct, unrealized_pnl_pct, days_held

Action（动作空间）：
  - 0: hold（持有不动）
  - 1: buy_25（买入25%仓位）
  - 2: buy_50（买入50%仓位）
  - 3: buy_75（买入75%仓位）
  - 4: buy_100（满仓）
  - 5: sell_25（卖出25%持仓）
  - 6: sell_50（卖出50%持仓）
  - 7: sell_75（卖出75%持仓）
  - 8: sell_100（清仓）

Reward（奖励）：
  - 基于单步收益率
  - 可选Sharpe-adjusted reward
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import json
from typing import Optional, Tuple, Dict, Any


class TradingEnv(gym.Env):
    """
    单股票交易环境。
    
    用法：
        env = TradingEnv(data_df, model_scores)
        obs, info = env.reset(start_date=20230101)
        for _ in range(max_steps):
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human"]}
    
    # 动作定义
    ACTION_HOLD = 0
    ACTION_BUY_25 = 1
    ACTION_BUY_50 = 2
    ACTION_BUY_75 = 3
    ACTION_BUY_100 = 4
    ACTION_SELL_25 = 5
    ACTION_SELL_50 = 6
    ACTION_SELL_75 = 7
    ACTION_SELL_100 = 8
    
    ACTION_NAMES = {
        0: "hold", 1: "buy_25", 2: "buy_50", 3: "buy_75", 4: "buy_100",
        5: "sell_25", 6: "sell_50", 7: "sell_75", 8: "sell_100"
    }
    
    def __init__(
        self,
        data: pd.DataFrame,
        model_scores: Optional[pd.Series] = None,
        initial_cash: float = 100000.0,
        commission: float = 0.001,       # 手续费0.1%
        slippage: float = 0.002,         # 滑点0.2%
        reward_type: str = "return",     # "return" or "sharpe"
        window_size: int = 20,           # 观察窗口大小
        max_position_pct: float = 1.0,   # 最大持仓比例
    ):
        super().__init__()
        
        self.data = data.sort_values("date").reset_index(drop=True)
        self.model_scores = model_scores
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.reward_type = reward_type
        self.window_size = window_size
        self.max_position_pct = max_position_pct
        
        # 预计算技术特征
        self._compute_features()
        
        # 动作空间
        self.action_space = spaces.Discrete(9)
        
        # 观察空间: [market(8) + portfolio(4) + scores(2)] = 14
        self._has_scores = "bs_score_norm" in self.data.columns and self.data["bs_score_norm"].sum() != 0
        obs_dim = 14 if self._has_scores else 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 状态变量
        self.current_step = 0
        self.cash = initial_cash
        self.position_shares = 0.0
        self.position_cost = 0.0  # 持仓成本
        self.entry_step = -1
        self.backtest_start_step = window_size  # 回测起始步（用于计算基准）
        self.trade_log = []
        self.portfolio_history = []
        
    def _compute_features(self):
        """预计算技术指标（避免每步重复计算）"""
        df = self.data.copy()
        
        # RSI 14
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta).clip(lower=0).rolling(14, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)
        df["rsi_14"] = df["rsi_14"].fillna(50)
        
        # MACD
        ema12 = df["close"].ewm(span=12, min_periods=1).mean()
        ema26 = df["close"].ewm(span=26, min_periods=1).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, min_periods=1).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        
        # ATR%
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        df["atr_pct"] = tr.rolling(14, min_periods=1).mean() / df["close"]
        
        # 成交量比率
        df["vol_ratio"] = df["volume"].rolling(5, min_periods=1).mean() / df["volume"].rolling(20, min_periods=1).mean()
        
        # 收益率
        df["ret5"] = df["close"].pct_change(5)
        df["ret10"] = df["close"].pct_change(10)
        df["ret20"] = df["close"].pct_change(20)
        
        # MA偏差
        df["ma20"] = df["close"].rolling(20, min_periods=1).mean()
        df["ma20_bias"] = (df["close"] - df["ma20"]) / df["ma20"]
        
        # 日收益率（用于reward计算）
        df["daily_return"] = df["close"].pct_change().fillna(0)
        
        # 归一化model score
        if self.model_scores is not None:
            df["model_score"] = np.array(self.model_scores).flatten()[:len(df)]
        else:
            df["model_score"] = 0.0
        
        # 预计算评分特征（从预计算parquet传入时可用）
        if "bs_score" in df.columns:
            df["bs_score_norm"] = df["bs_score"] * 2 - 1  # 映射到[-1,1]
        else:
            df["bs_score_norm"] = 0.0
        if "ga_score" in df.columns:
            df["ga_score_norm"] = df["ga_score"] * 2 - 1  # 映射到[-1,1]
        else:
            df["ga_score_norm"] = 0.0
        
        self.data = df.fillna(0)
        
        # 特征列
        self.feature_cols = [
            "rsi_14", "macd_hist", "atr_pct", "vol_ratio",
            "ret5", "ret10", "ret20", "ma20_bias"
        ]
        self.portfolio_cols = [
            "cash_pct", "position_pct", "unrealized_pnl_pct", "days_held"
        ]
        # 模型评分特征（如果数据中有）
        self.score_cols = ["bs_score_norm", "ga_score_norm"]
    
    def _get_state(self) -> np.ndarray:
        """返回当前观察向量"""
        row = self.data.iloc[self.current_step]
        
        # 市场特征（归一化到[-1,1]附近）
        market = np.array([
            (row["rsi_14"] - 50) / 50,           # RSI: [-1, 1]
            np.clip(row["macd_hist"] / (row["close"] * 0.01 + 1e-8), -1, 1),  # MACD
            np.clip(row["atr_pct"] * 10, -1, 1),  # ATR%
            np.clip(row["vol_ratio"] - 1, -1, 1),  # Vol ratio
            np.clip(row["ret5"] * 10, -1, 1),     # ret5
            np.clip(row["ret10"] * 5, -1, 1),     # ret10
            np.clip(row["ret20"] * 3, -1, 1),     # ret20
            np.clip(row["ma20_bias"] * 5, -1, 1), # MA20 bias
        ], dtype=np.float32)
        
        # 组合状态
        portfolio_value = self.cash + self.position_shares * row["close"]
        cash_pct = self.cash / portfolio_value if portfolio_value > 0 else 1.0
        position_pct = self.position_shares * row["close"] / portfolio_value if portfolio_value > 0 else 0.0
        unrealized_pnl = (row["close"] / self.position_cost - 1) if self.position_cost > 0 else 0.0
        days_held = (self.current_step - self.entry_step) / 20.0 if self.entry_step >= 0 else 0.0  # 归一化到月
        
        portfolio = np.array([
            np.clip(cash_pct, 0, 1),
            np.clip(position_pct, 0, 1),
            np.clip(unrealized_pnl, -1, 1),
            np.clip(days_held, 0, 1),
        ], dtype=np.float32)
        
        # 模型评分特征
        if self._has_scores:
            scores = np.array([
                np.clip(row["bs_score_norm"], -1, 1),
                np.clip(row["ga_score_norm"], -1, 1),
            ], dtype=np.float32)
            return np.concatenate([market, portfolio, scores])
        return np.concatenate([market, portfolio])
    
    def _execute_trade(self, action: int):
        """执行交易（含手续费+滑点）"""
        row = self.data.iloc[self.current_step]
        price = row["close"]
        portfolio_value = self.cash + self.position_shares * price
        
        target_value = 0.0
        if action == self.ACTION_HOLD:
            return
        
        # 买入操作
        elif action == self.ACTION_BUY_25:
            target_value = portfolio_value * 0.25
        elif action == self.ACTION_BUY_50:
            target_value = portfolio_value * 0.50
        elif action == self.ACTION_BUY_75:
            target_value = portfolio_value * 0.75
        elif action == self.ACTION_BUY_100:
            target_value = portfolio_value * self.max_position_pct
        
        # 卖出操作
        elif action == self.ACTION_SELL_25:
            sell_value = self.position_shares * price * 0.25
            self._sell(sell_value, price)
            return
        elif action == self.ACTION_SELL_50:
            sell_value = self.position_shares * price * 0.50
            self._sell(sell_value, price)
            return
        elif action == self.ACTION_SELL_75:
            sell_value = self.position_shares * price * 0.75
            self._sell(sell_value, price)
            return
        elif action == self.ACTION_SELL_100:
            sell_value = self.position_shares * price
            self._sell(sell_value, price)
            return
        
        # 执行买入
        if action >= self.ACTION_BUY_25 and action <= self.ACTION_BUY_100:
            buy_value = target_value - self.position_shares * price  # 增量买入
            if buy_value <= 0:
                return  # 已经有足够的持仓
            
            buy_value = min(buy_value, self.cash)  # 不能超过现金
            if buy_value <= 0:
                return
            
            # 加入滑点和手续费
            actual_price = price * (1 + self.slippage)
            cost = buy_value * (1 + self.commission)
            shares_bought = buy_value / actual_price
            
            if self.position_shares == 0:
                self.position_cost = actual_price
                self.entry_step = self.current_step
            else:
                # 加权平均成本
                total_cost = self.position_cost * self.position_shares + actual_price * shares_bought
                self.position_cost = total_cost / (self.position_shares + shares_bought)
            
            self.position_shares += shares_bought
            self.cash -= cost
            
            # 日期兼容：datetime64 → str, int/str → 原值
            date_val = row["date"]
            if hasattr(date_val, 'strftime'):
                date_val = date_val.strftime('%Y-%m-%d')
            else:
                date_val = str(date_val)
            self.trade_log.append({
                "step": self.current_step,
                "date": date_val,
                "action": self.ACTION_NAMES[action],
                "price": actual_price,
                "shares": shares_bought,
                "value": buy_value,
            })
    
    def _sell(self, sell_value: float, price: float):
        """执行卖出"""
        if sell_value <= 0 or self.position_shares <= 0:
            return
        
        actual_price = price * (1 - self.slippage)
        shares_sold = min(sell_value / actual_price, self.position_shares)
        proceeds = shares_sold * actual_price * (1 - self.commission)
        
        self.position_shares -= shares_sold
        self.cash += proceeds
        
        row = self.data.iloc[self.current_step]
        # 日期兼容：datetime64 → str
        date_val = row["date"]
        if hasattr(date_val, 'strftime'):
            date_val = date_val.strftime('%Y-%m-%d')
        else:
            date_val = str(date_val)
        self.trade_log.append({
            "step": self.current_step,
            "date": date_val,
            "action": f"sell_{int(shares_sold/self.position_shares*100) if self.position_shares > 0 else 100}" if self.position_shares > 0 else "sell_100",
            "price": actual_price,
            "shares": shares_sold,
            "value": proceeds,
        })
        
        if self.position_shares <= 1e-10:
            self.position_shares = 0.0
            self.position_cost = 0.0
            self.entry_step = -1
    
    def reset(self, start_idx: Optional[int] = None, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        if start_idx is not None:
            self.current_step = max(self.window_size, start_idx)
        else:
            self.current_step = self.window_size
        
        self.cash = self.initial_cash
        self.position_shares = 0.0
        self.position_cost = 0.0
        self.entry_step = -1
        self.trade_log = []
        self.portfolio_history = [self.initial_cash]
        
        # 日期兼容
        _date_raw = self.data.iloc[self.current_step]["date"]
        _date_str = _date_raw.strftime('%Y-%m-%d') if hasattr(_date_raw, 'strftime') else str(_date_raw)
        self.backtest_start_step = self.current_step  # 记录实际起始步
        return self._get_state(), {"step": self.current_step, "date": _date_str}
    
    def step(self, action: int):
        """执行一步"""
        prev_value = self.cash + self.position_shares * self.data.iloc[self.current_step]["close"]
        
        # 执行交易
        self._execute_trade(action)
        
        # 移动到下一步
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # 计算新的组合价值
        if not terminated:
            new_value = self.cash + self.position_shares * self.data.iloc[self.current_step]["close"]
        else:
            new_value = prev_value  # 最后一步用之前的价值
        
        self.portfolio_history.append(new_value)
        
        # 计算奖励
        if self.reward_type == "return":
            reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        elif self.reward_type == "sharpe":
            # 用最近20步的收益率计算滚动Sharpe
            returns = np.diff(self.portfolio_history[-21:]) / np.array(self.portfolio_history[-21:-1])
            if len(returns) > 1 and np.std(returns) > 0:
                reward = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                reward = 0
        else:
            reward = (new_value - prev_value) / prev_value if prev_value > 0 else 0
        
        # 日期兼容
        _date_raw = self.data.iloc[min(self.current_step, len(self.data)-1)]["date"]
        _date_str = _date_raw.strftime('%Y-%m-%d') if hasattr(_date_raw, 'strftime') else str(_date_raw)
        info = {
            "step": self.current_step,
            "date": _date_str,
            "portfolio_value": new_value,
            "cash": self.cash,
            "position_shares": self.position_shares,
            "position_pct": self.position_shares * self.data.iloc[min(self.current_step, len(self.data)-1)]["close"] / new_value if new_value > 0 else 0,
            "trade_count": len(self.trade_log),
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def get_summary(self) -> Dict[str, Any]:
        """获取回测摘要"""
        if not self.portfolio_history:
            return {}
        
        returns = np.diff(self.portfolio_history) / np.array(self.portfolio_history[:-1])
        returns = returns[~np.isnan(returns)]
        
        total_return = (self.portfolio_history[-1] / self.portfolio_history[0] - 1) * 100
        ann_return = ((self.portfolio_history[-1] / self.portfolio_history[0]) ** (252 / max(len(returns), 1)) - 1) * 100
        
        # 最大回撤
        peak = np.maximum.accumulate(self.portfolio_history)
        drawdown = (np.array(self.portfolio_history) - peak) / peak
        max_dd = drawdown.min() * 100
        
        # Sharpe
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # 买入持有基准
        first_price = self.data.iloc[self.backtest_start_step]["close"]
        last_price = self.data.iloc[min(self.current_step, len(self.data)-1)]["close"]
        bh_return = (last_price / first_price - 1) * 100
        
        return {
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(ann_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 3),
            "total_trades": len(self.trade_log),
            "final_value": round(self.portfolio_history[-1], 2),
            "buy_hold_return_pct": round(bh_return, 2),
            "alpha_pct": round(total_return - bh_return, 2),
            "steps": len(self.portfolio_history) - 1,
        }
