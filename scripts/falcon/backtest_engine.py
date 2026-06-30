"""
Falcon统一回测框架 — backtest_engine.py

设计原则:
  1. 数据门禁: 覆盖率<80%直接拒绝，垃圾数据进不来
  2. 日频净值: MaxDD只从日频曲线算，不可能出-0.2%
  3. 自动baseline: 每次回测自动跑等权baseline，不依赖人记得
  4. 结果校验: 异常自动warn，不依赖人检查

用法:
  from backtest_engine import BacktestEngine
  
  engine = BacktestEngine()
  result, baseline = engine.run(ranks, prices, weights, hold_days=30, top_n=10)
  
  # Walk-Forward
  windows = engine.walk_forward(ranks, prices, weights, train_years=2, test_months=6)

  所有回测脚本必须调用此模块。禁止自行实现回测逻辑。
"""
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """回测结果。所有指标从日频净值推导，不会出错。"""
    sharpe: float
    max_dd: float          # 负数, 如 -0.22 表示 -22%
    cagr: float
    win_rate: float
    total_return: float
    n_trades: int
    n_rebalances: int
    daily_equity: Any  # np.ndarray
    dates: Any  # List[str]
    trades: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    window_details: Optional[List[dict]] = None  # Walk-Forward时填充

    def summary(self) -> str:
        lines = [
            f"Sharpe={self.sharpe:.3f}  MaxDD={self.max_dd:.1%}  "
            f"CAGR={self.cagr:.1%}  WR={self.win_rate:.0%}  "
            f"Trades={self.n_trades}  Rebalances={self.n_rebalances}"
        ]
        if self.warnings:
            lines.append(f"  ⚠️ {'; '.join(self.warnings)}")
        return "\n".join(lines)


@dataclass
class DataQualityReport:
    """数据质量报告。"""
    passed: bool
    coverage_by_year: Dict[int, float]
    coverage_by_factor: Dict[str, float]
    n_tickers: int
    n_dates: int
    date_range: Tuple[str, str]
    issues: List[str]


# ═══════════════════════════════════════════════════════════════════
#  核心引擎
# ═══════════════════════════════════════════════════════════════════

class BacktestEngine:
    """Falcon统一回测引擎。
    
    所有回测必须通过此类。禁止自行实现回测逻辑。
    
    Attributes:
        cost: 单边交易成本 (默认0.001 = 0.1%)
        stop_loss: 止损线 (默认-0.15 = -15%)
        min_coverage: 数据覆盖率最低要求 (默认0.8)
    """
    
    def __init__(self, cost: float = 0.001, stop_loss: float = -0.15, 
                 min_coverage: float = 0.8):
        self.cost = cost
        self.stop_loss = stop_loss
        self.min_coverage = min_coverage
    
    # ─────────────────────────────────────────────────────────────
    #  主入口
    # ─────────────────────────────────────────────────────────────
    
    def run(self, 
            ranks: Dict[str, pd.DataFrame], 
            prices: pd.DataFrame,
            weights: Dict[str, float],
            hold_days: int = 30,
            top_n: int = 10,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            run_baseline: bool = True) -> Tuple["BacktestResult", Optional["BacktestResult"]]:
        """运行单次回测。
        
        Args:
            ranks: {date_str: DataFrame(ticker→factor)} 由precompute_pit_ranks_fast生成
            prices: DataFrame(date×ticker) 价格矩阵, index=date, columns=ticker
            weights: {factor_name: weight} 因子权重
            hold_days: 调仓周期(天)
            top_n: 选股数量
            start_date: 起始日期(可选)
            end_date: 结束日期(可选)
            run_baseline: 是否自动跑baseline(默认True)
            
        Returns:
            (result, baseline) — baseline在run_baseline=False时为None
            
        Raises:
            DataQualityError: 数据质量不达标
        """
        # ① 数据门禁
        common_dates = self._get_dates(ranks, prices, start_date, end_date)
        active_factors = [f for f, w in weights.items() if w > 0 and f in self._get_all_factors(ranks)]
        self._validate_data(ranks, prices, common_dates, active_factors)
        
        # ② 策略回测
        # 用prices的完整日期序列做模拟(不是ranks的稀疏日期)
        all_price_dates = sorted(prices.index.astype(str))
        if start_date:
            all_price_dates = [d for d in all_price_dates if d >= start_date]
        if end_date:
            all_price_dates = [d for d in all_price_dates if d <= end_date]
        result = self._simulate(ranks, prices, all_price_dates, weights, hold_days, top_n)
        
        # ③ 自动baseline
        baseline = None
        if run_baseline:
            baseline_weights = {f: 1.0/len(active_factors) for f in active_factors} if active_factors else weights
            baseline = self._simulate(ranks, prices, all_price_dates, baseline_weights, hold_days, top_n)
        
        # ④ 结果校验
        self._validate_result(result, baseline)
        
        return result, baseline
    
    def walk_forward(self,
                     ranks: Dict[str, pd.DataFrame],
                     prices: pd.DataFrame, 
                     weights: Dict[str, float],
                     hold_days: int = 30,
                     top_n: int = 10,
                     train_years: int = 2,
                     test_months: int = 6) -> BacktestResult:
        """Walk-Forward验证。
        
        使用expanding window: train从起点扩展到train_end, test是之后test_months。
        每个窗口调用self.run()，不重复实现回测逻辑。
        
        Returns:
            聚合后的BacktestResult, 含window_details
        """
        dates = sorted(ranks.keys())
        if not dates:
            raise ValueError("No dates in ranks")
        
        start = pd.Timestamp(dates[0])
        end = pd.Timestamp(dates[-1])
        
        train_start = start
        windows = []
        window_idx = 0
        
        while True:
            train_end = train_start + pd.DateOffset(years=train_years)
            test_end = train_end + pd.DateOffset(months=test_months)
            
            try:
                if str(test_end) > str(end):
                    break
            except Exception:
                break
            
            test_start_str = str(train_end)[:10]
            test_end_str = str(test_end)[:10]
            
            # 跑单窗口(不做baseline，节省时间)
            try:
                result, _ = self.run(
                    ranks, prices, weights, hold_days, top_n,
                    start_date=test_start_str, end_date=test_end_str,
                    run_baseline=False
                )
                windows.append({
                    "index": window_idx,
                    "period": f"{test_start_str} → {test_end_str}",
                    "sharpe": result.sharpe,
                    "max_dd": result.max_dd,
                    "cagr": result.cagr,
                    "win_rate": result.win_rate,
                    "n_trades": result.n_trades,
                    "n_days": len(result.daily_equity),
                })
            except DataQualityError as e:
                windows.append({
                    "index": window_idx,
                    "period": f"{test_start_str} → {test_end_str}",
                    "error": str(e),
                })
            
            window_idx += 1
            train_start += pd.DateOffset(months=test_months)
        
        if not windows:
            raise ValueError("Walk-Forward produced no windows")
        
        # 聚合
        valid_windows = [w for w in windows if "sharpe" in w]
        if not valid_windows:
            raise ValueError("All Walk-Forward windows failed data quality check")
        
        all_sharpes = [w["sharpe"] for w in valid_windows]
        all_dds = [w["max_dd"] for w in valid_windows]
        all_cagrs = [w["cagr"] for w in valid_windows]
        all_wrs = [w["win_rate"] for w in valid_windows]
        all_trades = [w["n_trades"] for w in valid_windows]
        
        # 全局指标(近似: 用窗口均值)
        agg_sharpe = float(np.mean(all_sharpes))
        agg_dd = float(np.min(all_dds))  # 最差窗口的MaxDD
        agg_cagr = float(np.mean(all_cagrs))
        agg_wr = float(np.mean(all_wrs))
        
        # 构造伪daily_equity(窗口均值, 仅用于summary)
        agg_equity = np.cumprod(1 + np.array([np.mean(all_cagrs)/252] * sum(w["n_days"] for w in valid_windows)))
        
        result = BacktestResult(
            sharpe=round(agg_sharpe, 3),
            max_dd=round(agg_dd, 4),
            cagr=round(agg_cagr, 4),
            win_rate=round(agg_wr, 3),
            total_return=round(float(agg_equity[-1] / agg_equity[0] - 1), 4),
            n_trades=sum(all_trades),
            n_rebalances=len(valid_windows),
            daily_equity=agg_equity,
            dates=[w["period"] for w in valid_windows],
            window_details=windows,
        )
        
        # Walk-Forward级别的校验
        self._validate_walk_forward(result, windows)
        
        return result
    
    # ─────────────────────────────────────────────────────────────
    #  数据门禁
    # ─────────────────────────────────────────────────────────────
    
    def check_data_quality(self, 
                           ranks: Dict[str, pd.DataFrame],
                           prices: pd.DataFrame,
                           factors: List[str] = None) -> DataQualityReport:
        """检查数据质量。可在回测前独立调用。"""
        dates = sorted(ranks.keys())
        issues = []
        
        # 日期范围
        date_range = (dates[0], dates[-1]) if dates else ("", "")
        
        # ticker数量
        sample_date = dates[len(dates)//2] if dates else None
        n_tickers = len(ranks[sample_date]) if sample_date and sample_date in ranks else 0
        
        # 因子覆盖率(按年)
        coverage_by_year = {}
        if factors:
            for year in sorted(set(d[:4] for d in dates)):
                year_dates = [d for d in dates if d.startswith(year)]
                year_coverages = []
                for d in year_dates[::5]:  # 每5天采样一次
                    if d in ranks:
                        df = ranks[d]
                        for f in factors:
                            if f in df.columns:
                                year_coverages.append(df[f].notna().mean())
                if year_coverages:
                    coverage_by_year[int(year)] = float(np.mean(year_coverages))
        
        # 因子覆盖率(全局)
        coverage_by_factor = {}
        if factors:
            sample_dates = dates[::max(1, len(dates)//50)]  # 采样50个日期
            for f in factors:
                fc = []
                for d in sample_dates:
                    if d in ranks and f in ranks[d].columns:
                        fc.append(ranks[d][f].notna().mean())
                if fc:
                    coverage_by_factor[f] = float(np.mean(fc))
        
        # 检查问题
        low_cov_years = {y: c for y, c in coverage_by_year.items() if c < self.min_coverage}
        if low_cov_years:
            issues.append(f"以下年份因子覆盖率<{self.min_coverage:.0%}: {low_cov_years}")
        
        low_cov_factors = {f: c for f, c in coverage_by_factor.items() if c < self.min_coverage}
        if low_cov_factors:
            issues.append(f"以下因子覆盖率<{self.min_coverage:.0%}: {low_cov_factors}")
        
        if n_tickers < 50:
            issues.append(f"股票数过少: {n_tickers} (<50)")
        
        return DataQualityReport(
            passed=len(issues) == 0,
            coverage_by_year=coverage_by_year,
            coverage_by_factor=coverage_by_factor,
            n_tickers=n_tickers,
            n_dates=len(dates),
            date_range=date_range,
            issues=issues,
        )
    
    def _validate_data(self, ranks, prices, dates, factors):
        """数据门禁: 不通过直接拒绝。
        
        修复(2026-07-01): 只检查当前窗口日期范围内的因子覆盖率，
        而非全量ranks。避免早期数据覆盖率低导致所有窗口失败。
        """
        # 只检查窗口内的日期覆盖率
        filtered_ranks = {d: ranks[d] for d in dates if d in ranks}
        if not filtered_ranks:
            raise DataQualityError("窗口内无有效日期")
        report = self.check_data_quality(filtered_ranks, prices, factors)
        if not report.passed:
            raise DataQualityError(
                f"数据质量不达标:\n" + "\n".join(f"  - {i}" for i in report.issues)
            )
    
    # ─────────────────────────────────────────────────────────────
    #  回测核心(日频模拟)
    # ─────────────────────────────────────────────────────────────
    
    def _simulate(self, ranks, prices, all_dates_in_prices, weights, hold_days, top_n) -> BacktestResult:
        """日频模拟: 每天更新净值，MaxDD从日频曲线推导。"""
        cash = 1.0  # 归一化
        portfolio = {}  # ticker -> (entry_idx, entry_price, shares_as_fraction)
        equity_list = []
        trade_list = []
        rebalance_count = 0

        # 用prices的完整日期序列(不是ranks的稀疏日期)
        # ranks只在需要选股时查找
        for i, date in enumerate(all_dates_in_prices):
            if date not in prices.index:
                if equity_list:
                    equity_list.append(equity_list[-1])
                continue

            pr = prices.loc[date]
            
            # ── 止损(每天检查) ──
            to_close = []
            for t, (ei, ep, frac) in portfolio.items():
                if t in pr.index and pd.notna(pr[t]) and ep > 0:
                    pnl = (pr[t] - ep) / ep
                    if pnl <= self.stop_loss:
                        # 止损卖出
                        cash += frac * pr[t] / ep * (1 - self.cost)  # 卖出回款扣成本
                        trade_list.append({"pnl": pnl, "reason": "stop_loss", "date": date})
                        to_close.append(t)
            for t in to_close:
                del portfolio[t]
            
            # ── 调仓检查 ──
            should_rebalance = False
            sell_tickers = []
            
            for t, (ei, ep, frac) in list(portfolio.items()):
                if (i - ei) >= hold_days:
                    sell_tickers.append(t)
            
            if sell_tickers or len(portfolio) == 0:
                should_rebalance = True
            
            # ── 卖出到期持仓 ──
            for t in sell_tickers:
                if t in portfolio:
                    ei, ep, frac = portfolio.pop(t)
                    if t in pr.index and pd.notna(pr[t]) and ep > 0:
                        pnl = (pr[t] - ep) / ep
                        cash += frac * pr[t] / ep * (1 - self.cost)  # 卖出回款扣成本
                        trade_list.append({"pnl": pnl, "reason": "rebalance", "date": date})
            
            # ── 买入新持仓 ──
            if should_rebalance and len(portfolio) == 0 and cash > 0.01:
                scores = self._get_scores(ranks, date, weights)
                if scores is not None:
                    picks = scores.head(top_n).index.tolist()
                    picks = [t for t in picks if t in pr.index and pd.notna(pr[t]) and pr[t] > 0]
                    if picks:
                        per = cash / len(picks)
                        buy_cost = sum(per * self.cost for _ in picks)
                        for t in picks:
                            portfolio[t] = (i, pr[t], per)
                        cash = 0.0  # 全部买入
                        cash -= buy_cost  # 扣买入交易成本
                        rebalance_count += 1
            
            # ── 日频净值 ──
            pv = cash
            for t, (_, ep, frac) in portfolio.items():
                if t in pr.index and pd.notna(pr[t]) and ep > 0:
                    pv += frac * pr[t] / ep
            equity_list.append(pv)
        
        # 计算指标
        return self._compute_result(equity_list, trade_list, rebalance_count, all_dates_in_prices[:len(equity_list)])
    
    def _get_scores(self, ranks, date, weights) -> Optional[pd.Series]:
        """计算因子组合分数。"""
        if date not in ranks:
            return None
        r = ranks[date]
        available = [f for f in weights if f in r.columns and weights[f] > 0]
        if not available:
            return None
        combined = pd.Series(0.0, index=r.index)
        for f in available:
            combined = combined + weights[f] * r[f]
        return combined.dropna().sort_values(ascending=False)
    
    def _compute_result(self, equity_list, trade_list, rebalance_count, dates) -> BacktestResult:
        """从日频净值推导所有指标。"""
        eq = np.array(equity_list, dtype=np.float64)
        
        if len(eq) < 2:
            return BacktestResult(
                sharpe=0, max_dd=0, cagr=0, win_rate=0, total_return=0,
                n_trades=len(trade_list), n_rebalances=rebalance_count,
                daily_equity=eq, dates=dates[:len(eq)], trades=trade_list,
                warnings=["数据不足2天"]
            )
        
        # 日收益率
        returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        std = np.std(returns)
        
        # Sharpe (年化)
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0
        
        # MaxDD (从日频净值, 不可能错)
        peak = np.maximum.accumulate(eq)
        dd_series = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(np.min(dd_series))
        
        # CAGR
        n_days = len(eq)
        total_return = float(eq[-1] / eq[0] - 1) if eq[0] > 0 else 0
        cagr = float((eq[-1] / eq[0]) ** (252 / max(n_days, 1)) - 1) if eq[0] > 0 else 0
        
        # 胜率
        wins = sum(1 for t in trade_list if t.get("pnl", 0) > 0)
        win_rate = wins / len(trade_list) if trade_list else 0
        
        return BacktestResult(
            sharpe=round(sharpe, 3),
            max_dd=round(max_dd, 4),
            cagr=round(cagr, 4),
            win_rate=round(win_rate, 3),
            total_return=round(total_return, 4),
            n_trades=len(trade_list),
            n_rebalances=rebalance_count,
            daily_equity=eq,
            dates=dates[:len(eq)],
            trades=trade_list,
        )
    
    # ─────────────────────────────────────────────────────────────
    #  结果校验
    # ─────────────────────────────────────────────────────────────
    
    def _validate_result(self, result: BacktestResult, baseline: Optional[BacktestResult]):
        """结果校验: 异常自动标记。"""
        warns = []
        
        # MaxDD合理性
        if abs(result.max_dd) < 0.05:
            warns.append(f"MaxDD={result.max_dd:.1%} 太小, 10年美股不可能<5%")
        
        # Sharpe合理性
        if result.sharpe > 3:
            warns.append(f"Sharpe={result.sharpe:.2f} > 3, 高度过拟合风险")
        if result.sharpe < 0:
            warns.append(f"Sharpe={result.sharpe:.3f} < 0, 策略亏钱")
        
        # Baseline对比
        if baseline and baseline.sharpe > 0:
            if result.sharpe < baseline.sharpe * 0.8:
                warns.append(f"Sharpe {result.sharpe:.3f} < baseline {baseline.sharpe:.3f}×80%")
        
        result.warnings = warns
        
        if warns:
            for w in warns:
                warnings.warn(f"⚠️ {w}")
    
    def _validate_walk_forward(self, result: BacktestResult, windows: List[dict]):
        """Walk-Forward级别校验。"""
        valid = [w for w in windows if "sharpe" in w]
        if not valid:
            result.warnings.append("所有Walk-Forward窗口失败")
            return
        
        extreme = [w for w in valid if abs(w["sharpe"]) > 10]
        if extreme:
            result.warnings.append(f"{len(extreme)}个窗口|Sharpe|>10: {[round(w['sharpe'],1) for w in extreme]}")
        
        if len(extreme) > len(valid) * 0.25:
            result.warnings.append(f"异常窗口>{len(extreme)}/{len(valid)}=25%, 整体不可信")
        
        recent = valid[-3:]
        if len(recent) >= 3 and all(w["sharpe"] < 0 for w in recent):
            result.warnings.append(f"近3窗口全负: {[w['sharpe'] for w in recent]}")
    
    # ─────────────────────────────────────────────────────────────
    #  辅助
    # ─────────────────────────────────────────────────────────────
    
    def _get_dates(self, ranks, prices, start_date=None, end_date=None):
        """获取有效日期列表(取ranks和prices的交集)。"""
        rank_dates = set(ranks.keys())
        price_dates = set(prices.index.astype(str))
        common = sorted(rank_dates & price_dates)
        
        if start_date:
            common = [d for d in common if d >= start_date]
        if end_date:
            common = [d for d in common if d <= end_date]
        
        return common
    
    def _get_all_factors(self, ranks):
        """从ranks中获取所有可用因子名。"""
        sample_date = next(iter(ranks))
        return list(ranks[sample_date].columns)


# ═══════════════════════════════════════════════════════════════════
#  异常类
# ═══════════════════════════════════════════════════════════════════

class DataQualityError(Exception):
    """数据质量不达标。回测不应继续。"""
    pass


# ═══════════════════════════════════════════════════════════════════
#  快捷函数(便于批量使用)
# ═══════════════════════════════════════════════════════════════════

def run_backtest(ranks, prices, weights, hold_days=30, top_n=10,
                 cost=0.001, stop_loss=-0.15) -> Tuple[BacktestResult, Optional[BacktestResult]]:
    """快捷函数: 一行代码跑回测+baseline。"""
    engine = BacktestEngine(cost=cost, stop_loss=stop_loss)
    return engine.run(ranks, prices, weights, hold_days, top_n)


def run_walk_forward(ranks, prices, weights, hold_days=30, top_n=10,
                     train_years=2, test_months=6) -> BacktestResult:
    """快捷函数: 一行代码跑Walk-Forward。"""
    engine = BacktestEngine()
    return engine.walk_forward(ranks, prices, weights, hold_days, top_n, 
                               train_years, test_months)
