"""
Falcon统一回测框架 — 评分模块

评分逻辑与falcon_score.py完全一致:
  1. 截面percentile ranking (每只股票在当日所有股票中的百分位)
  2. 翻转因子 (PE/PS等数值越高越差, rank取反)
  3. IC^power加权 (组内因子按IC权重加权)
  4. growth_composite = 0.60*fg + 0.25*analyst + 0.15*income
  5. final = 0.45*fr + 0.20*gc + 0.20*qoq + 0.15*cf

两种IC模式:
  - static: 读factor_ic_weights.json (生产用)
  - rolling: 从历史数据计算滚动IC (WF回测用, 更真实)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class ScoringEngine:
    """Falcon评分引擎。
    
    确保回测与生产评分逻辑完全一致。
    所有参数从config读取。
    """
    
    def __init__(self, config: dict):
        scoring = config.get('scoring', {})
        
        # 因子组定义
        self.factor_groups = scoring.get('factor_groups', {})
        
        # 翻转因子集合
        flip_list = scoring.get('flip_factors', [])
        self.flip_factors = set(flip_list)
        
        # 顶层权重
        self.top_weights = scoring.get('top_weights', {
            'fund_ratio': 0.45, 'growth_composite': 0.20,
            'qoq': 0.20, 'cashflow': 0.15,
        })
        
        # growth_composite子权重
        self.gc_weights = scoring.get('gc_weights', {
            'fund_growth': 0.60, 'analyst': 0.25, 'income': 0.15,
        })
        
        # IC参数
        ic_config = scoring.get('ic', {})
        self.ic_lookback = ic_config.get('lookback', 126)
        self.ic_power = ic_config.get('power', 0.5)
        self.ic_source = ic_config.get('source', 'rolling')
        
        # 预计算所有因子列表
        self.all_factors = []
        for factors in self.factor_groups.values():
            self.all_factors.extend(factors)
    
    def rank_cross_section(self, day_df: pd.DataFrame) -> pd.DataFrame:
        """截面percentile ranking (与falcon_score.py一致)。
        
        Args:
            day_df: 某天的数据, index=ticker, columns含因子列
        
        Returns:
            DataFrame(ticker × factor), 值为0~1的百分位排名
        """
        ranks = pd.DataFrame(index=day_df.index)
        
        for col in self.all_factors:
            if col not in day_df.columns:
                continue
            vals = day_df[col].values.astype(float)
            valid_count = np.sum(~np.isnan(vals))
            if valid_count < 10:
                continue
            
            r = pd.Series(vals, index=day_df.index).rank(pct=True)
            
            # 翻转: 数值越高越差的因子, rank取反
            if col in self.flip_factors:
                r = 1.0 - r
            
            ranks[col] = r
        
        return ranks
    
    def compute_group_score(self, ranks: pd.DataFrame, group_name: str,
                           ic_weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """计算单个因子组得分。
        
        与falcon_score.py的compute_group_score_ic完全一致:
        1. 取该组可用因子
        2. 用IC权重加权(负IC→权重0)
        3. 归一化后加权求和
        
        Args:
            ranks: 截面rank DataFrame
            group_name: 因子组名(如'fund_ratio')
            ic_weights: {factor_name: ic_value} 或 None(等权)
        
        Returns:
            Series(ticker → group_score)
        """
        group_cols = self.factor_groups.get(group_name, [])
        available = [c for c in group_cols if c in ranks.columns]
        
        if not available:
            return pd.Series(0.5, index=ranks.index, dtype=float)
        
        if ic_weights is not None:
            # IC加权: 负IC权重设为0
            weights = {}
            for col in available:
                w = ic_weights.get(col, 0)
                weights[col] = max(0, w) ** self.ic_power
            
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
                score = pd.Series(0.0, index=ranks.index)
                for col in available:
                    score += weights[col] * ranks[col]
                return score
        
        # 等权fallback
        return ranks[available].mean(axis=1)
    
    def compute_composite_score(self, day_df: pd.DataFrame,
                               ic_weights: object = None) -> pd.Series:
        """计算V0.4.6综合得分。
        
        与falcon_score.py的compute_score完全一致。
        
        Args:
            day_df: 某天数据, index=ticker
            ic_weights: IC权重数据:
                - 静态模式: {group_name: {factor: weight}} (from JSON)
                - 滚动模式: {factor: ic_value} (from rolling IC)
                - None: 等权
        
        Returns:
            Series(ticker → final_score), 按降序排列
        """
        # Step 1: 截面rank
        ranks = self.rank_cross_section(day_df)
        
        if ranks.empty or len(ranks.columns) == 0:
            return pd.Series(dtype=float)
        
        # Step 2: 各组得分(IC加权)
        group_scores = {}
        for group_name in self.factor_groups:
            # 提取该组的IC权重
            group_ic = None
            if ic_weights is not None:
                if isinstance(ic_weights, dict):
                    # 判断格式
                    first_val = next(iter(ic_weights.values()), None)
                    if isinstance(first_val, dict):
                        # 静态IC: {group: {factor: weight}}
                        group_ic = ic_weights.get(group_name, {})
                    else:
                        # 滚动IC: {factor: ic_value}
                        group_ic = ic_weights
            
            group_scores[group_name] = self.compute_group_score(
                ranks, group_name, group_ic
            )
        
        # Step 3: growth_composite
        gc = (self.gc_weights['fund_growth'] * group_scores.get('fund_growth', 0) +
              self.gc_weights['analyst'] * group_scores.get('analyst', 0) +
              self.gc_weights['income'] * group_scores.get('income', 0))
        
        # Step 4: 最终得分
        final = (self.top_weights['fund_ratio'] * group_scores.get('fund_ratio', 0) +
                 self.top_weights['growth_composite'] * gc +
                 self.top_weights['qoq'] * group_scores.get('qoq', 0) +
                 self.top_weights['cashflow'] * group_scores.get('cashflow', 0))
        
        return final.dropna().sort_values(ascending=False)
    
    def compute_daily_ic(self, ranks: Dict[str, pd.DataFrame],
                        prices: pd.DataFrame,
                        forward_days: int = 30) -> Dict[str, Dict[str, float]]:
        """计算每日IC(每个因子与前瞻收益的rank相关系数)。
        
        用于滚动IC计算。
        
        Args:
            ranks: {date_str: rank_DataFrame}
            prices: price pivot DataFrame
            forward_days: 前瞻天数
        
        Returns:
            {date: {factor: ic_value}}
        """
        from scipy.stats import spearmanr
        
        all_dates = sorted(ranks.keys())
        price_dates = sorted(prices.index.astype(str))
        daily_ic = {}
        
        for date in all_dates:
            # 找前瞻日期
            future_dates = [d for d in price_dates if d > date]
            if len(future_dates) < forward_days:
                continue
            fwd_date = future_dates[min(forward_days - 1, len(future_dates) - 1)]
            
            if fwd_date not in prices.index or date not in prices.index:
                continue
            
            fwd_ret = (prices.loc[fwd_date] / prices.loc[date] - 1).dropna()
            rd = ranks[date]
            common = rd.index.intersection(fwd_ret.index)
            
            if len(common) < 30:
                continue
            
            fv = fwd_ret[common].values
            ic_dict = {}
            
            for col in rd.columns:
                r = rd.loc[common, col].values
                valid = ~(np.isnan(r) | np.isnan(fv))
                if valid.sum() < 30:
                    continue
                ic, _ = spearmanr(np.asarray(r[valid]), np.asarray(fv[valid]))
                if not np.isnan(ic):
                    ic_dict[col] = ic
            
            if ic_dict:
                daily_ic[date] = ic_dict
        
        return daily_ic
    
    def compute_rolling_ic(self, daily_ic: Dict[str, Dict[str, float]],
                          all_dates: list,
                          lookback: int = None,
                          step: int = 5) -> Dict[str, Dict[str, float]]:
        """计算滚动IC权重。
        
        与run_wf_v046_production.py的rolling_ic一致。
        
        Args:
            daily_ic: 每日IC
            all_dates: 所有需要IC的日期
            lookback: IC回看天数(默认从config)
            step: 采样步长
        
        Returns:
            {date: {factor: rolling_ic_value}}, forward-filled到所有日期
        """
        if lookback is None:
            lookback = self.ic_lookback
        
        ic_dates = sorted(daily_ic.keys())
        ic_history = {}
        
        for i in range(0, len(ic_dates), step):
            date = ic_dates[i]
            ws = max(0, i - lookback // step)
            wd = ic_dates[ws:i + 1]
            
            fi = {}
            for col in self.all_factors:
                vals = [daily_ic[d].get(col, np.nan) for d in wd
                       if col in daily_ic.get(d, {})]
                vals = [v for v in vals if not np.isnan(v)]
                if len(vals) >= 10:
                    fi[col] = float(np.mean(vals))
            
            if fi:
                ic_history[date] = fi
        
        # Forward-fill到所有日期
        filled = {}
        all_ic_dates = sorted(ic_history.keys())
        for date in all_dates:
            cands = [d for d in all_ic_dates if d <= date]
            if cands:
                filled[date] = ic_history[cands[-1]]
        
        return filled
