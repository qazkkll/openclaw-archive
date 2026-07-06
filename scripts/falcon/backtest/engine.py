"""
Falcon统一回测框架 — 回测引擎

与旧backtest_engine.py的区别:
  1. 成本模型: 使用Futu真实成本(按股数/金额阶梯计算), 不是固定百分比
  2. 评分逻辑: 直接调用ScoringEngine, 与falcon_score.py完全一致
  3. 参数化: 所有参数从config读取, 可替换
  4. 10年默认: 覆盖2020新冠+2018熊市
  5. 行业限制: 可选的行业分散约束
  6. VIX过滤: 可选的regime仓位调整

用法:
    from backtest.engine import BacktestEngine
    engine = BacktestEngine(config)
    result = engine.run()
    wf_result = engine.walk_forward()
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .cost_model import create_cost_model, FutuCostModel, FlatCostModel
from .scoring import ScoringEngine


@dataclass
class TradeRecord:
    """单笔交易记录。"""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl_pct: float
    pnl_dollar: float
    cost_dollar: float
    reason: str           # "rebalance" | "stop_loss"
    hold_days: int


@dataclass
class BacktestResult:
    """回测结果。"""
    sharpe: float
    max_dd: float
    cagr: float
    win_rate: float
    total_return: float
    n_trades: int
    n_rebalances: int
    avg_hold_days: float
    total_cost_pct: float     # 总成本占初始资金%
    avg_cost_per_trade: float # 每笔交易平均成本%
    daily_equity: np.ndarray
    dates: list
    trades: List[TradeRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    window_details: Optional[List[dict]] = None
    yearly: Optional[Dict[str, dict]] = None
    config_snapshot: Optional[dict] = None
    
    def summary(self) -> str:
        lines = [
            f"Sharpe={self.sharpe:.3f}  MaxDD={self.max_dd:.1%}  "
            f"CAGR={self.cagr:.1%}  WR={self.win_rate:.0%}  "
            f"Trades={self.n_trades}  Rebalances={self.n_rebalances}",
            f"AvgHold={self.avg_hold_days:.0f}d  "
            f"TotalCost={self.total_cost_pct:.2%}  "
            f"AvgCost/Trade={self.avg_cost_per_trade:.3%}",
        ]
        if self.warnings:
            lines.append(f"⚠️ {'; '.join(self.warnings)}")
        return "\n".join(lines)


class BacktestEngine:
    """Falcon统一回测引擎。
    
    所有参数从config字典读取。
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # 交易参数
        trading = config.get('trading', {})
        self.hold_days = trading.get('hold_days', 30)
        self.top_n = trading.get('top_n', 10)
        self.stop_loss = trading.get('stop_loss', -0.15)
        
        # 成本模型
        self.cost_model = create_cost_model(config)
        
        # 评分引擎
        self.scoring = ScoringEngine(config)
        
        # 行业限制
        sector_config = config.get('sector_limit', {})
        self.sector_limit_enabled = sector_config.get('enabled', False)
        self.max_per_sector = sector_config.get('max_per_sector', 3)
        self.sector_map = {}  # {ticker: sector}, 在run()中加载
        
        # VIX过滤
        vix_config = config.get('vix_filter', {})
        self.vix_filter_enabled = vix_config.get('enabled', False)
        self.vix_thresholds = vix_config.get('thresholds', {})
        
        # Walk-Forward参数
        wf_config = config.get('walk_forward', {})
        self.wf_train_months = wf_config.get('train_months', 12)
        self.wf_test_months = wf_config.get('test_months', 6)
        self.wf_step_months = wf_config.get('step_months', 6)
        self.min_coverage = wf_config.get('min_coverage', 0.80)
        
        # 回测范围
        bt_config = config.get('backtest', {})
        self.bt_years = bt_config.get('years', 10)
        self.bt_start = bt_config.get('start_date')
        self.bt_end = bt_config.get('end_date')
    
    def _load_sector_map(self) -> dict:
        """加载行业分类映射。"""
        import json
        path = self.config.get('data', {}).get('sector_map_path', '')
        if not path:
            return {}
        try:
            from pathlib import Path
            full_path = Path(__file__).resolve().parent.parent.parent.parent / path
            if full_path.exists():
                with open(full_path) as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def _apply_sector_limit(self, scores: pd.Series) -> pd.Series:
        """应用行业分散限制。
        
        保留每个行业得分最高的max_per_sector只股票。
        """
        if not self.sector_limit_enabled or not self.sector_map:
            return scores
        
        # 给每只股票标注行业
        sectors = pd.Series(
            {t: self.sector_map.get(t, 'Unknown') for t in scores.index},
            name='sector'
        )
        
        # 按得分降序排列, 每行业保留前N
        ranked = scores.sort_values(ascending=False)
        result = pd.Series(dtype=float)
        sector_counts = {}
        
        for ticker, score in ranked.items():
            sector = sectors.get(ticker, 'Unknown')
            count = sector_counts.get(sector, 0)
            if count < self.max_per_sector:
                result[ticker] = score
                sector_counts[sector] = count + 1
        
        return result
    
    def _get_position_pct(self, vix_value: float) -> float:
        """根据VIX确定仓位比例。"""
        if not self.vix_filter_enabled:
            return 1.0
        
        thresholds = self.vix_thresholds
        if vix_value > thresholds.get('extreme_bear', 30):
            return 0.25
        elif vix_value > thresholds.get('bear', 25):
            return 0.50
        elif vix_value >= thresholds.get('neutral', 20):
            return 0.75
        else:
            return 1.0
    
    def _compute_cost_pct(self, price: float, shares: int, side: str) -> float:
        """计算交易成本百分比。"""
        if isinstance(self.cost_model, FutuCostModel):
            if side == 'buy':
                return self.cost_model.buy_pct(price, shares)
            else:
                return self.cost_model.sell_pct(price, shares)
        else:
            return self.cost_model.buy_pct(price, shares)
    
    def run(self, features: pd.DataFrame, prices: pd.DataFrame,
            ic_data: object = None,
            start_date: str = None, end_date: str = None) -> BacktestResult:
        """运行单次回测。
        
        Args:
            features: features_v04_1.parquet数据
            prices: 价格pivot DataFrame (index=date, columns=ticker)
            ic_data: IC权重数据(静态或滚动)
            start_date: 起始日期(覆盖config)
            end_date: 结束日期(覆盖config)
        
        Returns:
            BacktestResult
        """
        # 日期范围
        if start_date is None:
            start_date = self.bt_start
        if end_date is None:
            end_date = self.bt_end
        
        if start_date is None:
            # 默认: 最近N年
            all_dates = sorted(features['date'].unique())
            years_ago = (pd.Timestamp.now() - pd.DateOffset(years=self.bt_years)).strftime('%Y-%m-%d')
            start_date = str(years_ago)
        
        if end_date is None:
            end_date = '9999-12-31'
        else:
            end_date = str(end_date)
        
        # 加载行业映射
        self.sector_map = self._load_sector_map()
        
        # 获取有效日期
        feature_dates = sorted(features['date'].unique())
        price_dates = set(prices.index.astype(str))
        valid_dates = [d for d in feature_dates
                      if start_date <= d <= end_date and d in price_dates]
        
        if not valid_dates:
            raise ValueError(f"日期范围 {start_date}~{end_date} 无有效数据")
        
        # 模拟
        return self._simulate(features, prices, valid_dates, ic_data)
    
    def _simulate(self, features: pd.DataFrame, prices: pd.DataFrame,
                  valid_dates: list, ic_data: object) -> BacktestResult:
        """日频模拟。"""
        initial_capital = 100000.0
        cash = initial_capital
        portfolio = {}  # ticker -> {'entry_idx': i, 'entry_price': p, 'shares': s, 'cost': c}
        equity_list = []
        trade_list = []
        rebalance_count = 0
        total_cost_dollar = 0.0
        price_dates = sorted(prices.index.astype(str))
        
        for i, date in enumerate(valid_dates):
            if date not in prices.index:
                if equity_list:
                    equity_list.append(equity_list[-1])
                continue
            
            pr = prices.loc[date]
            
            # ── 止损(每天检查) ──
            to_close = []
            for ticker, pos in portfolio.items():
                if ticker in pr.index and pd.notna(pr[ticker]) and pos['entry_price'] > 0:
                    current_price = pr[ticker]
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                    
                    if pnl_pct <= self.stop_loss:
                        # 止损卖出
                        sell_cost_pct = self._compute_cost_pct(
                            current_price, pos['shares'], 'sell')
                        proceeds = pos['shares'] * current_price * (1 - sell_cost_pct)
                        cost_dollar = pos['shares'] * current_price * sell_cost_pct
                        
                        cash += proceeds
                        total_cost_dollar += cost_dollar
                        
                        trade_list.append(TradeRecord(
                            ticker=ticker,
                            entry_date=pos['entry_date'],
                            exit_date=date,
                            entry_price=pos['entry_price'],
                            exit_price=current_price,
                            shares=pos['shares'],
                            pnl_pct=pnl_pct,
                            pnl_dollar=proceeds - pos['shares'] * pos['entry_price'],
                            cost_dollar=cost_dollar,
                            reason='stop_loss',
                            hold_days=i - pos['entry_idx'],
                        ))
                        to_close.append(ticker)
            
            for ticker in to_close:
                del portfolio[ticker]
            
            # ── 调仓检查 ──
            sell_tickers = []
            for ticker, pos in portfolio.items():
                if (i - pos['entry_idx']) >= self.hold_days:
                    sell_tickers.append(ticker)
            
            if sell_tickers or len(portfolio) == 0:
                # 卖出到期持仓
                for ticker in sell_tickers:
                    if ticker in portfolio:
                        pos = portfolio.pop(ticker)
                        if ticker in pr.index and pd.notna(pr[ticker]):
                            current_price = pr[ticker]
                            pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                            sell_cost_pct = self._compute_cost_pct(
                                current_price, pos['shares'], 'sell')
                            proceeds = pos['shares'] * current_price * (1 - sell_cost_pct)
                            cost_dollar = pos['shares'] * current_price * sell_cost_pct
                            
                            cash += proceeds
                            total_cost_dollar += cost_dollar
                            
                            trade_list.append(TradeRecord(
                                ticker=ticker,
                                entry_date=pos['entry_date'],
                                exit_date=date,
                                entry_price=pos['entry_price'],
                                exit_price=current_price,
                                shares=pos['shares'],
                                pnl_pct=pnl_pct,
                                pnl_dollar=proceeds - pos['shares'] * pos['entry_price'],
                                cost_dollar=cost_dollar,
                                reason='rebalance',
                                hold_days=i - pos['entry_idx'],
                            ))
                
                # 买入新持仓
                if len(portfolio) == 0 and cash > initial_capital * 0.01:
                    # 评分
                    day_features = features[features['date'] == date].copy()
                    if len(day_features) < 10:
                        continue
                    day_features.index = day_features['ticker'].tolist()
                    
                    # 评分: 提取当天的IC权重
                    # rolling IC格式: {date: {factor: value}}
                    # static IC格式: {group: {factor: value}}
                    day_ic = ic_data
                    if isinstance(ic_data, dict) and date in ic_data:
                        # rolling IC: 用当天的IC权重
                        day_ic = ic_data[date]
                    scores = self.scoring.compute_composite_score(day_features, day_ic)
                    if scores.empty:
                        continue
                    
                    # 行业限制
                    scores = self._apply_sector_limit(scores)
                    
                    # VIX仓位调整
                    position_pct = 1.0
                    if self.vix_filter_enabled:
                        # 从features数据中取当日VIX(比从prices取更可靠)
                        vix_col = 'vix_close'
                        if vix_col in day_features.columns:
                            vix_raw = day_features[vix_col].tolist()
                            vix_clean = [v for v in vix_raw if v is not None and not (isinstance(v, float) and v != v)]
                            if len(vix_clean) > 0:
                                position_pct = self._get_position_pct(float(vix_clean[0]))
                        elif 'VIX' in pr.index and pd.notna(pr.get('VIX', None)):
                            position_pct = self._get_position_pct(float(pr['VIX']))
                    
                    # 选股
                    picks = scores.head(self.top_n).index.tolist()
                    picks = [t for t in picks
                            if t in pr.index and pd.notna(pr[t]) and pr[t] > 0]
                    
                    if not picks:
                        continue
                    
                    # 分配资金
                    deploy_cash = cash * position_pct
                    per_stock = deploy_cash / len(picks)
                    
                    for ticker in picks:
                        price = pr[ticker]
                        shares = max(1, int(per_stock / price))
                        
                        buy_cost_pct = self._compute_cost_pct(price, shares, 'buy')
                        cost_dollar = price * shares * buy_cost_pct
                        
                        cash -= (price * shares + cost_dollar)
                        total_cost_dollar += cost_dollar
                        
                        portfolio[ticker] = {
                            'entry_idx': i,
                            'entry_price': price,
                            'shares': shares,
                            'entry_date': date,
                        }
                    
                    rebalance_count += 1
            
            # ── 日频净值 ──
            pv = cash
            for ticker, pos in portfolio.items():
                if ticker in pr.index and pd.notna(pr[ticker]):
                    pv += pos['shares'] * pr[ticker]
                else:
                    pv += pos['shares'] * pos['entry_price']
            equity_list.append(pv)
        
        return self._compute_result(
            equity_list, trade_list, rebalance_count,
            total_cost_dollar, initial_capital,
            valid_dates[:len(equity_list)]
        )
    
    def _compute_result(self, equity_list, trade_list, rebalance_count,
                       total_cost_dollar, initial_capital, dates) -> BacktestResult:
        """从日频净值推导所有指标。"""
        eq = np.array(equity_list, dtype=np.float64)
        
        if len(eq) < 2:
            return BacktestResult(
                sharpe=0, max_dd=0, cagr=0, win_rate=0, total_return=0,
                n_trades=0, n_rebalances=0, avg_hold_days=0,
                total_cost_pct=0, avg_cost_per_trade=0,
                daily_equity=eq, dates=dates[:len(eq)],
                warnings=["数据不足2天"],
            )
        
        # 日收益率
        returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        std = np.std(returns)
        
        # Sharpe
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0
        
        # MaxDD
        peak = np.maximum.accumulate(eq)
        dd_series = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(np.min(dd_series))
        
        # CAGR
        n_days = len(eq)
        total_return = float(eq[-1] / eq[0] - 1) if eq[0] > 0 else 0
        cagr = float((eq[-1] / eq[0]) ** (252 / max(n_days, 1)) - 1) if eq[0] > 0 else 0
        
        # 胜率
        wins = sum(1 for t in trade_list if t.pnl_pct > 0)
        win_rate = wins / len(trade_list) if trade_list else 0
        
        # 平均持有天数
        avg_hold = np.mean([t.hold_days for t in trade_list]) if trade_list else 0
        
        # 成本统计
        total_cost_pct = total_cost_dollar / initial_capital if initial_capital > 0 else 0
        avg_cost = total_cost_pct / len(trade_list) if trade_list else 0
        
        # 分年统计
        yearly = {}
        for t in trade_list:
            year = t.entry_date[:4]
            if year not in yearly:
                yearly[year] = {'trades': 0, 'wins': 0, 'total_pnl': 0.0, 'costs': 0.0}
            yearly[year]['trades'] += 1
            if t.pnl_pct > 0:
                yearly[year]['wins'] += 1
            yearly[year]['total_pnl'] += t.pnl_dollar
            yearly[year]['costs'] += t.cost_dollar
        
        for year, y in yearly.items():
            y['win_rate'] = y['wins'] / y['trades'] if y['trades'] > 0 else 0
        
        # 校验
        warns = self._validate(sharpe, max_dd, n_days, win_rate, len(trade_list))
        
        return BacktestResult(
            sharpe=round(sharpe, 3),
            max_dd=round(max_dd, 4),
            cagr=round(cagr, 4),
            win_rate=round(win_rate, 3),
            total_return=round(total_return, 4),
            n_trades=len(trade_list),
            n_rebalances=rebalance_count,
            avg_hold_days=round(float(avg_hold), 1),
            total_cost_pct=round(total_cost_pct, 4),
            avg_cost_per_trade=round(avg_cost, 5),
            daily_equity=eq,
            dates=dates[:len(eq)],
            trades=trade_list,
            warnings=warns,
            yearly=yearly,
            config_snapshot={
                'hold_days': self.hold_days,
                'top_n': self.top_n,
                'stop_loss': self.stop_loss,
                'cost_model': self.config.get('trading', {}).get('cost_model', 'unknown'),
                'sector_limit': self.sector_limit_enabled,
                'vix_filter': self.vix_filter_enabled,
            },
        )
    
    def _validate(self, sharpe, max_dd, n_days, win_rate, n_trades) -> list:
        """结果校验。"""
        warns = []
        if abs(max_dd) < 0.03 and n_days > 500:
            warns.append(f"MaxDD={max_dd:.1%}太小, 10年美股不可能<3%")
        if sharpe > 3:
            warns.append(f"Sharpe={sharpe:.2f}>3, 高度过拟合风险")
        if sharpe < 0:
            warns.append(f"Sharpe={sharpe:.3f}<0, 策略亏钱")
        if n_trades < 20:
            warns.append(f"交易数={n_trades}<20, 统计不可靠")
        return warns
