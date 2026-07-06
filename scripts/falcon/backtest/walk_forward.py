"""
Falcon统一回测框架 — Walk-Forward验证

Walk-Forward回测:
  1. 将数据分为多个train/test窗口
  2. 每个窗口独立回测
  3. 聚合所有窗口的结果
  4. 支持expanding和rolling两种模式

IC模式:
  - static: 使用全局factor_ic_weights.json (⚠️ 有look-ahead bias, 仅供对比)
  - rolling: 每个窗口用自己的历史数据计算滚动IC (无look-ahead, 推荐)
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-Forward验证器。
    
    用法:
        wf = WalkForwardValidator(config)
        result = wf.run(features, prices, ic_data_or_mode)
    """
    
    def __init__(self, config: dict):
        self.config = config
        wf_config = config.get('walk_forward', {})
        self.train_months = wf_config.get('train_months', 12)
        self.test_months = wf_config.get('test_months', 6)
        self.step_months = wf_config.get('step_months', 6)
        self.oos_mode = wf_config.get('oos_mode', 'expanding')
        self.oos_train_years = wf_config.get('oos_train_years', 3)
        
        self.engine = BacktestEngine(config)
    
    def _precompute_daily_ic(self, features: pd.DataFrame,
                             prices: pd.DataFrame,
                             all_dates: list) -> Dict[str, Dict[str, float]]:
        """一次性预计算所有日期的每日IC (避免重复计算)。

        Returns:
            {date: {factor: ic_value}}  — 每个因子与前瞻收益的rank相关系数
        """
        scoring = self.engine.scoring
        hold_days = self.engine.hold_days

        # 按日期分组计算截面rank
        logger.info("Precomputing cross-sectional ranks for %d dates...", len(all_dates))
        ranks_by_date = {}
        for date in all_dates:
            day_df = features[features['date'] == date].copy()
            if len(day_df) < 10:
                continue
            day_df.index = day_df['ticker'].tolist()
            ranks = scoring.rank_cross_section(day_df)
            if not ranks.empty:
                ranks_by_date[date] = ranks

        logger.info("Ranks computed for %d dates, computing daily IC...",
                    len(ranks_by_date))

        # 一次性计算所有日期的daily IC
        daily_ic = scoring.compute_daily_ic(
            ranks_by_date, prices, forward_days=hold_days
        )

        logger.info("Daily IC computed for %d dates", len(daily_ic))
        return daily_ic

    def _compute_window_rolling_ic(
        self, daily_ic: Dict[str, Dict[str, float]],
        all_dates: list, cutoff_date: str
    ) -> Dict[str, Dict[str, float]]:
        """为某个窗口计算rolling IC权重 (仅使用cutoff_date之前的数据)。

        Args:
            daily_ic: 预计算的全量每日IC
            all_dates: 所有日期列表
            cutoff_date: 截止日期 (只用≤此日期的daily IC)

        Returns:
            {date: {factor: rolling_ic_value}} — 该窗口可用的rolling IC
        """
        scoring = self.engine.scoring

        # 截取cutoff_date之前的daily IC (不含cutoff_date当天,
        # 因为rolling IC应基于历史而非未来)
        # 额外减去hold_days天: 因为daily IC用了hold_days天前瞻收益,
        # cutoff_date前hold_days天的IC会用到cutoff_date之后的价格
        from datetime import timedelta
        safe_cutoff = (pd.Timestamp(cutoff_date) - timedelta(days=self.engine.hold_days * 2)).strftime('%Y-%m-%d')
        filtered_daily_ic = {
            d: ic for d, ic in daily_ic.items()
            if pd.Timestamp(d) < pd.Timestamp(safe_cutoff)
        }

        if not filtered_daily_ic:
            return {}

        # 该窗口期间的日期 (用于forward-fill)
        window_dates = [
            d for d in all_dates
            if pd.Timestamp(d) >= pd.Timestamp(cutoff_date)
        ]
        if not window_dates:
            return {}

        # 计算rolling IC并forward-fill到窗口期间
        rolling_ic = scoring.compute_rolling_ic(
            filtered_daily_ic, window_dates,
            lookback=scoring.ic_lookback, step=5
        )
        return rolling_ic

    def run(self, features: pd.DataFrame, prices: pd.DataFrame,
            ic_mode: str = "static",
            ic_weights_data=None) -> BacktestResult:
        """运行Walk-Forward回测。

        Args:
            features: 特征数据
            prices: 价格矩阵
            ic_mode: "static"或"rolling"
                - static: 使用全局IC权重 (⚠️ look-ahead bias, 仅供对比)
                - rolling: 每个窗口独立计算滚动IC (无bias, 推荐)
            ic_weights_data: 静态IC权重数据(ic_mode="static"时使用)

        Returns:
            聚合的BacktestResult
        """
        # ── look-ahead bias 警告 ──
        if ic_mode == "static":
            logger.warning(
                "⚠️  ic_mode='static' uses global IC weights computed from the "
                "entire dataset. This introduces LOOK-AHEAD BIAS — windows from "
                "2017 use IC weights that include information from 2026. "
                "Use ic_mode='rolling' for unbiased walk-forward validation."
            )

        bt_config = self.config.get('backtest', {})
        years = bt_config.get('years', 10)
        start = bt_config.get('start_date')
        end = bt_config.get('end_date')
        
        if start is None:
            start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
        if end is None:
            end = '9999-12-31'
        
        # 获取有效日期
        feature_dates = sorted(features['date'].unique())
        price_dates = set(prices.index.astype(str))
        all_dates = [d for d in feature_dates
                    if str(start) <= d <= str(end) and d in price_dates]
        
        if len(all_dates) < 252:
            raise ValueError(f"有效日期不足: {len(all_dates)} < 252")
        
        # 构建WF窗口
        first = pd.Timestamp(all_dates[0])
        last = pd.Timestamp(all_dates[-1])
        
        windows = []
        if self.oos_mode == "expanding":
            # Expanding window: train从起点开始, 每次扩展
            train_start = first
            while True:
                train_end = train_start + pd.DateOffset(months=self.train_months)
                test_end = train_end + pd.DateOffset(months=self.test_months)
                
                if test_end > last:
                    break
                
                windows.append({
                    'train_start': train_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'test_start': train_end.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                })
                train_start += pd.DateOffset(months=self.step_months)
        
        elif self.oos_mode == "rolling":
            # Rolling window: train固定长度
            current = first
            while True:
                train_start = current
                train_end = current + pd.DateOffset(years=self.oos_train_years)
                test_end = train_end + pd.DateOffset(months=self.test_months)
                
                if test_end > last:
                    break
                
                windows.append({
                    'train_start': train_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'test_start': train_end.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                })
                current += pd.DateOffset(months=self.step_months)
        
        if not windows:
            raise ValueError("Walk-Forward无法构建有效窗口")
        
        # ── rolling IC: 一次性预计算全量daily IC ──
        # 后续每个窗口只需截取subset, 避免重复计算
        cached_daily_ic = None
        if ic_mode == "rolling":
            cached_daily_ic = self._precompute_daily_ic(
                features, prices, all_dates
            )
        
        # 对每个窗口回测
        window_results = []
        all_equity_segments = []
        
        for wi, w in enumerate(windows):
            test_dates = [d for d in all_dates
                         if w['test_start'] <= d <= w['test_end']]
            
            if len(test_dates) < self.engine.hold_days + 1:
                window_results.append({
                    'window': wi,
                    'period': f"{w['test_start']} → {w['test_end']}",
                    'error': 'insufficient dates',
                })
                continue
            
            # ── 决定IC数据 ──
            if ic_mode == "static":
                # ⚠️ look-ahead: 使用全局静态IC
                window_ic = ic_weights_data
            else:
                # rolling模式: 从预计算的daily IC中截取并计算rolling IC
                window_ic = self._compute_window_rolling_ic(
                    cached_daily_ic, all_dates, w['test_start']
                )
            
            try:
                result = self.engine.run(
                    features, prices,
                    ic_data=window_ic,
                    start_date=w['test_start'],
                    end_date=w['test_end'],
                )
                
                window_results.append({
                    'window': wi,
                    'period': f"{w['test_start']} → {w['test_end']}",
                    'sharpe': result.sharpe,
                    'max_dd': result.max_dd,
                    'cagr': result.cagr,
                    'win_rate': result.win_rate,
                    'n_trades': result.n_trades,
                    'n_days': len(result.daily_equity),
                    'total_cost_pct': result.total_cost_pct,
                    'avg_cost_per_trade': result.avg_cost_per_trade,
                })
                
                all_equity_segments.append(result.daily_equity)
                
            except Exception as e:
                window_results.append({
                    'window': wi,
                    'period': f"{w['test_start']} → {w['test_end']}",
                    'error': str(e),
                })
        
        # 聚合
        return self._aggregate(window_results, all_equity_segments)
    
    def _aggregate(self, window_results: list,
                   equity_segments: list) -> BacktestResult:
        """聚合所有窗口结果。"""
        valid = [w for w in window_results if 'sharpe' in w]
        
        if not valid:
            raise ValueError("所有Walk-Forward窗口失败")
        
        sharpes = [w['sharpe'] for w in valid]
        dds = [w['max_dd'] for w in valid]
        cagrs = [w['cagr'] for w in valid]
        wrs = [w['win_rate'] for w in valid]
        trades = [w['n_trades'] for w in valid]
        costs = [w['total_cost_pct'] for w in valid]
        
        # 连接净值曲线
        if equity_segments:
            # 归一化每个段, 然后拼接
            full_equity = []
            multiplier = 1.0
            for seg in equity_segments:
                if len(seg) == 0:
                    continue
                seg_normalized = seg / seg[0] * multiplier
                full_equity.extend(seg_normalized.tolist())
                multiplier = float(seg_normalized[-1])
            equity = np.array(full_equity)
        else:
            equity = np.cumprod(1 + np.array([np.mean(cagrs) / 252] *
                              sum(w['n_days'] for w in valid)))
        
        # 从连续净值计算指标
        returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        std = np.std(returns)
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0
        
        peak = np.maximum.accumulate(equity)
        dd_series = (equity - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(np.min(dd_series))
        
        n_days = len(equity)
        total_ret = float(equity[-1] / equity[0] - 1) if equity[0] > 0 else 0
        cagr = float((equity[-1] / equity[0]) ** (252 / max(n_days, 1)) - 1) if equity[0] > 0 else 0
        
        # 校验
        warns = []
        if sharpe > 3:
            warns.append(f"WF Sharpe={sharpe:.2f}>3, 高度过拟合风险")
        extreme = [w for w in valid if abs(w['sharpe']) > 10]
        if extreme:
            warns.append(f"{len(extreme)}个窗口|Sharpe|>10")
        if len(extreme) > len(valid) * 0.25:
            warns.append(f"异常窗口>{len(extreme)}/{len(valid)}=25%, 整体不可信")
        recent = valid[-3:]
        if len(recent) >= 3 and all(w['sharpe'] < 0 for w in recent):
            warns.append(f"近3窗口全负: {[w['sharpe'] for w in recent]}")
        
        # 窗口一致性
        pos_windows = sum(1 for w in valid if w['sharpe'] > 0)
        consistency = pos_windows / len(valid) if valid else 0
        
        return BacktestResult(
            sharpe=round(sharpe, 3),
            max_dd=round(max_dd, 4),
            cagr=round(cagr, 4),
            win_rate=round(float(np.mean(wrs)), 3),
            total_return=round(total_ret, 4),
            n_trades=sum(trades),
            n_rebalances=len(valid),
            avg_hold_days=self.engine.hold_days,
            total_cost_pct=round(float(np.mean(costs)), 4),
            avg_cost_per_trade=round(float(np.mean(costs)) / max(1, sum(trades)) * len(valid), 5),
            daily_equity=equity,
            dates=[w['period'] for w in valid],
            warnings=warns,
            window_details=window_results,
        )
