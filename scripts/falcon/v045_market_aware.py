#!/usr/bin/env python3
"""
🦅 Falcon V0.4.5: Market-Aware Dynamic Position Sizing
======================================================
在V0.4.4配置基础上，添加大盘感知能力和动态仓位管理。

大盘感知:
  - VIX指数 (从data/us/vix_10y.parquet读取)
  - 20日滚动波动率
  - 200日均线趋势
  - 市场状态定义 (牛市/熊市/震荡/极端熊市)

动态仓位:
  - 固定仓位 (100%): baseline
  - VIX-only: 仅根据VIX水平调整
  - Trend-only: 仅根据200日均线趋势调整
  - VIX+Trend: 综合VIX和趋势调整 (推荐)

Walk-Forward 参数:
  train_years=0.5 (6个月), test_months=6, hold_days=30
  top_n=10, cost=0.001, stop_loss=-0.15

输出: data/falcon/v045_market_aware_results.json
"""
import sys, json, time, warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

WORKSPACE = Path("/home/hermes/.hermes/openclaw-archive")
sys.path.insert(0, str(WORKSPACE / "scripts" / "falcon"))

from backtest_engine import BacktestEngine, BacktestResult, DataQualityError

# ═══════════════════════════════════════════════════
#  路径
# ═══════════════════════════════════════════════════
DATA_DIR = WORKSPACE / "data" / "falcon"
FEATURES_PATH = DATA_DIR / "features_v04_1.parquet"
PRICES_PATH = DATA_DIR / "us_prices_daily.parquet"
VIX_PATH = WORKSPACE / "data" / "us" / "vix_10y.parquet"
OUTPUT_PATH = DATA_DIR / "v045_market_aware_results.json"

# ═══════════════════════════════════════════════════
#  因子组定义 (V0.4.4原始配置)
# ═══════════════════════════════════════════════════

EXCLUDE_COLS = {
    'ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap',
    'ma5', 'ma20', 'ma60', 'ma_bias20', 'ma_align', 'ma_cross_5_20', 'ma_cross_20_60',
    'price_position', 'ret1', 'ret5', 'ret10', 'ret20', 'ret30', 'ret60', 'ret90',
    'momentum_6m', 'momentum_1m', 'mom_divergence', 'trend_accel',
    'vol20', 'vol5', 'vol_ratio', 'vol_change', 'vol_regime',
    'rsi14', 'rsi_change', 'rsi_zone',
    'macd', 'macd_signal', 'macd_hist', 'macd_roc',
    'bb_std', 'bb_width', 'bb_pos',
    'ret_quality', 'range_ratio', 'avg_body', 'vwap_drift', 'dd_60', 'ud_vol_ratio', 'beta',
    'priceToEarningsRatio', 'priceToBookRatio', 'priceToSalesRatio',
    'priceToFreeCashFlowRatio', 'enterpriseValueMultiple',
    'grossProfitMargin', 'netProfitMargin', 'operatingProfitMargin', 'ebitdaMargin',
    'assetTurnover', 'inventoryTurnover', 'receivablesTurnover',
    'debtToEquityRatio', 'currentRatio', 'quickRatio', 'financialLeverageRatio',
    'freeCashFlowOperatingCashFlowRatio', 'operatingCashFlowRatio',
    'dividendYieldPercentage', 'dividendPayoutRatio',
    'eps_revision', 'revenue_revision', 'num_analysts_eps', 'num_analysts_rev',
    'eps_dispersion', 'fmp_covered', 'analyst_covered',
    'grossProfitMargin_qoq', 'netProfitMargin_qoq', 'operatingProfitMargin_qoq', 'ebitdaMargin_qoq',
}

FACTOR_GROUPS = {
    'fund_ratio': [
        'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
        'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
        'r_grossProfitMargin', 'r_netProfitMargin', 'r_operatingProfitMargin', 'r_ebitdaMargin',
        'r_assetTurnover', 'r_inventoryTurnover', 'r_receivablesTurnover',
        'r_debtToEquityRatio', 'r_currentRatio', 'r_quickRatio', 'r_financialLeverageRatio',
        'r_freeCashFlowOperatingCashFlowRatio', 'r_operatingCashFlowRatio',
        'r_dividendYieldPercentage', 'r_dividendPayoutRatio',
    ],
    'fund_growth': [
        'g_revenueGrowth', 'g_grossProfitGrowth', 'g_ebitgrowth',
        'g_operatingIncomeGrowth', 'g_netIncomeGrowth', 'g_epsdilutedGrowth',
        'g_freeCashFlowGrowth', 'g_tenYRevenueGrowthPerShare',
        'g_fiveYRevenueGrowthPerShare', 'g_threeYRevenueGrowthPerShare',
        'g_receivablesGrowth', 'g_inventoryGrowth', 'g_assetGrowth',
        'g_bookValueperShareGrowth', 'g_debtGrowth',
    ],
    'analyst': [
        'a_eps_revision', 'a_revenue_revision', 'a_eps_dispersion', 'a_num_analysts_eps',
    ],
    'income': [
        'i_gross_margin', 'i_operating_margin', 'i_net_margin', 'i_ebitda_margin',
        'i_revenue_growth_yoy', 'i_gross_margin_delta',
    ],
    'qoq': [
        'r_grossProfitMargin_qoq', 'r_netProfitMargin_qoq',
        'r_operatingProfitMargin_qoq', 'r_ebitdaMargin_qoq',
    ],
    'cashflow': [
        'c_fcf_margin', 'c_capex_intensity', 'c_fcf_to_income', 'c_buyback_yield',
    ],
}

# V0.4.4权重
V044_WEIGHTS = {
    "fund_ratio": 0.45,
    "gc_baseline": 0.20,  # growth_composite
    "qoq": 0.20,
    "cashflow": 0.15,
}

# 需要翻转的因子 (越高越差)
FLIP_FACTORS = {
    'r_priceToEarningsRatio', 'r_priceToBookRatio', 'r_priceToSalesRatio',
    'r_priceToFreeCashFlowRatio', 'r_enterpriseValueMultiple',
    'r_debtToEquityRatio', 'r_financialLeverageRatio', 'r_inventoryTurnover',
    'c_capex_intensity',
    'g_debtGrowth', 'g_receivablesGrowth', 'g_inventoryGrowth',
    'a_eps_dispersion',
}


# ═══════════════════════════════════════════════════
#  大盘感知: VIX + 市场状态
# ═══════════════════════════════════════════════════

class MarketRegime:
    """市场状态管理器。
    
    基于VIX水平和价格趋势定义4种市场状态:
    - BULL: 牛市 (VIX<20, 趋势向上) → 100%仓位
    - NEUTRAL: 震荡 (VIX 20-25, 趋势平) → 75%仓位
    - BEAR: 熊市 (VIX>25, 趋势向下) → 50%仓位
    - EXTREME_BEAR: 极端熊市 (VIX>30) → 25%仓位
    """
    
    # 市场状态定义
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    EXTREME_BEAR = "extreme_bear"
    
    def __init__(self, vix_df: pd.DataFrame, price_pivot: pd.DataFrame):
        """
        Args:
            vix_df: VIX数据 (date, close)
            price_pivot: SPX价格矩阵 (date × ticker)
        """
        # 预处理VIX数据
        self._vix = self._prepare_vix(vix_df)
        # 预处理市场趋势 (用等权平均价格作为大盘代理)
        self._market_price = self._prepare_market_price(price_pivot)
        # 预计算所有日期的市场状态
        self._regime_cache = self._compute_regimes()
    
    def _prepare_vix(self, vix_df: pd.DataFrame) -> pd.Series:
        """准备VIX时间序列。"""
        vix = vix_df[['date', 'close']].copy()
        vix['date'] = vix['date'].astype(str)
        vix = vix.set_index('date')['close'].sort_index()
        vix.index = vix.index.astype(str)
        return vix
    
    def _prepare_market_price(self, price_pivot: pd.DataFrame) -> pd.Series:
        """准备市场平均价格时间序列 (等权平均作为大盘代理)。"""
        # 用所有股票的等权平均价格作为大盘代理
        market = price_pivot.mean(axis=1).sort_index()
        market.index = market.index.astype(str)
        return market
    
    def _compute_regimes(self) -> dict:
        """预计算所有日期的市场状态。"""
        regimes = {}
        
        # VIX 20日滚动波动率 (VIX本身就是波动率指标，这里用VIX的20日移动平均来平滑)
        vix_ma20 = self._vix.rolling(window=20, min_periods=1).mean()
        
        # 市场趋势: 200日均线
        ma200 = self._market_price.rolling(window=200, min_periods=50).mean()
        
        # 获取所有可用日期
        all_dates = sorted(set(self._vix.index) & set(self._market_price.index))
        
        for date in all_dates:
            # VIX当前值 (用20日移动平均平滑)
            if date in vix_ma20.index:
                vix_val = float(vix_ma20[date])
            elif date in self._vix.index:
                vix_val = float(self._vix[date])
            else:
                vix_val = 20.0  # 默认中性
            
            # 趋势判断: 当前价格 vs 200日均线
            if date in ma200.index and pd.notna(ma200[date]):
                market_price = float(self._market_price[date]) if date in self._market_price.index else 0
                ma200_val = float(ma200[date])
                if market_price > 0 and ma200_val > 0:
                    trend_pct = (market_price - ma200_val) / ma200_val
                else:
                    trend_pct = 0.0
            else:
                trend_pct = 0.0
            
            # 定义市场状态
            if vix_val > 30:
                regime = self.EXTREME_BEAR
            elif vix_val > 25 and trend_pct < -0.05:
                regime = self.BEAR
            elif vix_val > 25:
                regime = self.BEAR
            elif vix_val >= 20 and abs(trend_pct) < 0.05:
                regime = self.NEUTRAL
            elif vix_val < 20 and trend_pct > 0.02:
                regime = self.BULL
            elif vix_val < 20:
                regime = self.BULL
            else:
                regime = self.NEUTRAL
            
            regimes[date] = {
                'regime': regime,
                'vix': vix_val,
                'trend_pct': trend_pct,
            }
        
        return regimes
    
    def get_regime(self, date_str: str) -> str:
        """获取指定日期的市场状态。"""
        if date_str in self._regime_cache:
            return self._regime_cache[date_str]['regime']
        # 找最近的可用日期
        candidates = [d for d in self._regime_cache.keys() if d <= date_str]
        if candidates:
            return self._regime_cache[candidates[-1]]['regime']
        return self.NEUTRAL  # 默认中性
    
    def get_vix(self, date_str: str) -> float:
        """获取指定日期的VIX值。"""
        if date_str in self._regime_cache:
            return self._regime_cache[date_str]['vix']
        candidates = [d for d in self._regime_cache.keys() if d <= date_str]
        if candidates:
            return self._regime_cache[candidates[-1]]['vix']
        return 20.0
    
    def get_trend(self, date_str: str) -> float:
        """获取指定日期的趋势百分比。"""
        if date_str in self._regime_cache:
            return self._regime_cache[date_str]['trend_pct']
        candidates = [d for d in self._regime_cache.keys() if d <= date_str]
        if candidates:
            return self._regime_cache[candidates[-1]]['trend_pct']
        return 0.0
    
    def get_regime_summary(self) -> dict:
        """获取市场状态分布统计。"""
        from collections import Counter
        regimes = [v['regime'] for v in self._regime_cache.values()]
        counter = Counter(regimes)
        total = len(regimes)
        return {
            'total_days': total,
            'bull_days': counter.get(self.BULL, 0),
            'bull_pct': round(counter.get(self.BULL, 0) / total, 3) if total > 0 else 0,
            'neutral_days': counter.get(self.NEUTRAL, 0),
            'neutral_pct': round(counter.get(self.NEUTRAL, 0) / total, 3) if total > 0 else 0,
            'bear_days': counter.get(self.BEAR, 0),
            'bear_pct': round(counter.get(self.BEAR, 0) / total, 3) if total > 0 else 0,
            'extreme_bear_days': counter.get(self.EXTREME_BEAR, 0),
            'extreme_bear_pct': round(counter.get(self.EXTREME_BEAR, 0) / total, 3) if total > 0 else 0,
        }


# ═══════════════════════════════════════════════════
#  动态仓位策略
# ═══════════════════════════════════════════════════

class PositionSizingStrategy:
    """仓位策略基类。"""
    
    def get_position_fraction(self, date_str: str, market_regime: MarketRegime) -> float:
        """返回仓位比例 (0.0 ~ 1.0)。"""
        raise NotImplementedError


class FixedPosition(PositionSizingStrategy):
    """固定仓位 (100%) — baseline。"""
    
    def get_position_fraction(self, date_str, market_regime):
        return 1.0
    
    def __repr__(self):
        return "FixedPosition(100%)"


class VIXOnlyPosition(PositionSizingStrategy):
    """仅根据VIX水平调整仓位。
    
    规则:
    - VIX < 18: 100% (极度乐观)
    - VIX 18-22: 90% (正常)
    - VIX 22-28: 70% (谨慎)
    - VIX 28-35: 50% (防御)
    - VIX > 35: 30% (极度恐慌)
    """
    
    def __init__(self):
        # 平滑过渡阈值
        self.thresholds = [
            (18, 1.0),
            (22, 0.9),
            (28, 0.7),
            (35, 0.5),
            (999, 0.3),
        ]
    
    def get_position_fraction(self, date_str, market_regime):
        vix = market_regime.get_vix(date_str)
        
        for i in range(len(self.thresholds) - 1):
            vix_low, frac_low = self.thresholds[i]
            vix_high, frac_high = self.thresholds[i + 1]
            
            if vix < vix_low:
                return frac_low
            elif vix < vix_high:
                # 线性插值
                t = (vix - vix_low) / (vix_high - vix_low)
                return frac_low + t * (frac_high - frac_low)
        
        return self.thresholds[-1][1]
    
    def __repr__(self):
        return "VIXOnlyPosition"


class TrendOnlyPosition(PositionSizingStrategy):
    """仅根据200日均线趋势调整仓位。
    
    规则 (趋势百分比 = (价格-MA200)/MA200):
    - 趋势 > +10%: 100% (强势上涨)
    - 趋势 +5% ~ +10%: 90% (上涨)
    - 趋势 -5% ~ +5%: 80% (震荡)
    - 趋势 -10% ~ -5%: 60% (下跌)
    - 趋势 < -10%: 40% (强势下跌)
    """
    
    def __init__(self):
        self.thresholds = [
            (-0.10, 0.4),
            (-0.05, 0.6),
            (0.05, 0.8),
            (0.10, 0.9),
            (999, 1.0),
        ]
    
    def get_position_fraction(self, date_str, market_regime):
        trend = market_regime.get_trend(date_str)
        
        for i in range(len(self.thresholds) - 1):
            t_low, frac_low = self.thresholds[i]
            t_high, frac_high = self.thresholds[i + 1]
            
            if trend < t_low:
                return frac_low
            elif trend < t_high:
                # 线性插值
                t = (trend - t_low) / (t_high - t_low)
                return frac_low + t * (frac_high - frac_low)
        
        return self.thresholds[-1][1]
    
    def __repr__(self):
        return "TrendOnlyPosition"


class VIXTrendPosition(PositionSizingStrategy):
    """综合VIX和趋势调整仓位 (推荐)。
    
    核心逻辑: VIX和趋势分别给出独立的仓位建议，取两者的加权平均。
    权重: VIX=60%, 趋势=40%
    
    这种方式比单一指标更稳健:
    - 牛市确认 (VIX低+趋势向上): 高仓位
    - 矛盾信号 (VIX低但趋势向下): 中等仓位
    - 熊市确认 (VIX高+趋势向下): 低仓位
    """
    
    def __init__(self):
        self.vix_strategy = VIXOnlyPosition()
        self.trend_strategy = TrendOnlyPosition()
        self.vix_weight = 0.6
        self.trend_weight = 0.4
    
    def get_position_fraction(self, date_str, market_regime):
        vix_frac = self.vix_strategy.get_position_fraction(date_str, market_regime)
        trend_frac = self.trend_strategy.get_position_fraction(date_str, market_regime)
        
        combined = vix_frac * self.vix_weight + trend_frac * self.trend_weight
        
        # 确保在合理范围内
        return max(0.25, min(1.0, combined))
    
    def __repr__(self):
        return "VIXTrendPosition(60%VIX+40%Trend)"


# ═══════════════════════════════════════════════════
#  MarketAwareEngine: 支持动态仓位的回测引擎
# ═══════════════════════════════════════════════════

class MarketAwareEngine(BacktestEngine):
    """支持动态仓位管理的回测引擎。
    
    继承BacktestEngine，重写_simulate方法支持:
    - 动态仓位 (根据市场状态调整)
    - 固定仓位 (baseline)
    """
    
    def __init__(self, cost=0.001, stop_loss=-0.15,
                 position_strategy=None, market_regime=None):
        """
        Args:
            cost: 单边交易成本
            stop_loss: 止损线
            position_strategy: PositionSizingStrategy实例
            market_regime: MarketRegime实例
        """
        super().__init__(cost=cost, stop_loss=stop_loss)
        self.position_strategy = position_strategy or FixedPosition()
        self.market_regime = market_regime
    
    def _simulate(self, ranks, prices, all_dates_in_prices, weights, hold_days, top_n):
        """重写模拟方法，支持动态仓位。"""
        cash = 1.0
        portfolio = {}  # ticker -> (entry_idx, entry_price, cash_allocated)
        equity_list = []
        trade_list = []
        rebalance_count = 0
        
        # 跟踪仓位信息
        position_fractions_used = []
        
        for i, date in enumerate(all_dates_in_prices):
            if date not in prices.index:
                if equity_list:
                    equity_list.append(equity_list[-1])
                continue
            
            pr = prices.loc[date]
            
            # ── 止损(每天检查) ──
            to_close = []
            for t, (ei, ep, alloc) in portfolio.items():
                if t in pr.index and pd.notna(pr[t]) and ep > 0:
                    pnl = (pr[t] - ep) / ep
                    if pnl <= self.stop_loss:
                        cash += alloc * pr[t] / ep * (1 - self.cost)
                        trade_list.append({"pnl": pnl, "reason": "stop_loss", "date": date})
                        to_close.append(t)
            for t in to_close:
                del portfolio[t]
            
            # ── 调仓检查 ──
            should_rebalance = False
            sell_tickers = []
            
            for t, (ei, ep, alloc) in list(portfolio.items()):
                if (i - ei) >= hold_days:
                    sell_tickers.append(t)
            
            if sell_tickers or len(portfolio) == 0:
                should_rebalance = True
            
            # ── 卖出到期持仓 ──
            for t in sell_tickers:
                if t in portfolio:
                    ei, ep, alloc = portfolio.pop(t)
                    if t in pr.index and pd.notna(pr[t]) and ep > 0:
                        pnl = (pr[t] - ep) / ep
                        cash += alloc * pr[t] / ep * (1 - self.cost)
                        trade_list.append({"pnl": pnl, "reason": "rebalance", "date": date})
            
            # ── 买入新持仓 ──
            if should_rebalance and len(portfolio) == 0 and cash > 0.01:
                # 获取动态仓位比例
                pos_fraction = self.position_strategy.get_position_fraction(
                    date, self.market_regime
                ) if self.market_regime else 1.0
                position_fractions_used.append((date, pos_fraction))
                
                # 只用cash的一部分来买入 (动态仓位)
                invest_cash = cash * pos_fraction
                
                if invest_cash > 0.01:
                    scores = self._get_scores(ranks, date, weights)
                    if scores is not None:
                        picks = scores.head(top_n).index.tolist()
                        picks = [t for t in picks if t in pr.index and pd.notna(pr[t]) and pr[t] > 0]
                        if picks:
                            per = invest_cash / len(picks)
                            buy_cost = 0.0
                            for t in picks:
                                portfolio[t] = (i, pr[t], per)
                                buy_cost += per * self.cost
                            
                            cash -= invest_cash  # 扣除投资部分
                            cash -= buy_cost  # 扣除交易成本
                            rebalance_count += 1
            
            # ── 日频净值 ──
            pv = cash
            for t, (_, ep, alloc) in portfolio.items():
                if t in pr.index and pd.notna(pr[t]) and ep > 0:
                    pv += alloc * pr[t] / ep
            equity_list.append(pv)
        
        result = self._compute_result(equity_list, trade_list, rebalance_count, all_dates_in_prices[:len(equity_list)])
        
        # 附加仓位统计
        if position_fractions_used:
            fracs = [f for _, f in position_fractions_used]
            result.warnings.append(
                f"仓位: avg={np.mean(fracs):.1%}, min={np.min(fracs):.1%}, max={np.max(fracs):.1%}"
            )
        
        return result


# ═══════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════

def load_data():
    """加载特征、价格和VIX数据。"""
    print("📂 加载数据...")
    t0 = time.time()

    df = pd.read_parquet(FEATURES_PATH)
    df['date'] = df['date'].astype(str)
    print(f"  ✅ Features: {df.shape[0]}行 × {df.shape[1]}列, {df['ticker'].nunique()}只")

    prices_df = pd.read_parquet(PRICES_PATH)
    prices_df['date'] = prices_df['date'].astype(str)
    price_pivot = prices_df.pivot_table(index='date', columns='ticker', values='close').sort_index()
    print(f"  ✅ Prices: {price_pivot.shape[0]}天 × {price_pivot.shape[1]}只")

    vix_df = pd.read_parquet(VIX_PATH)
    print(f"  ✅ VIX: {vix_df.shape[0]}行, 范围 {vix_df['date'].min()} ~ {vix_df['date'].max()}")

    print(f"  ⏱️ 加载耗时: {time.time()-t0:.1f}秒")
    return df, price_pivot, vix_df


# ═══════════════════════════════════════════════════
#  截面百分位排名
# ═══════════════════════════════════════════════════

def compute_cross_sectional_ranks(df, factor_cols):
    """计算截面百分位排名。"""
    print("📊 计算截面百分位排名...")
    t0 = time.time()

    from scipy.stats import rankdata

    dates = sorted(df['date'].unique())
    ranks = {}

    for date in dates:
        day_df = df[df['date'] == date].copy()
        if len(day_df) < 10:
            continue

        tickers = day_df['ticker'].values
        rank_df = pd.DataFrame(index=tickers)

        for col in factor_cols:
            if col not in day_df.columns:
                continue
            vals = day_df[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue

            ranks_raw = np.full_like(vals, np.nan)
            if valid.sum() > 0:
                ranks_raw[valid] = rankdata(vals[valid], method='average') / valid.sum()

            if col in FLIP_FACTORS:
                mask = ~np.isnan(ranks_raw)
                ranks_raw[mask] = 1.0 - ranks_raw[mask]

            rank_df[col] = ranks_raw

        ranks[date] = rank_df

    elapsed = time.time() - t0
    print(f"  ✅ {len(ranks)}天排名计算完成 ({elapsed:.0f}秒)")
    return ranks


def compute_group_ranks(ranks, factor_groups):
    """将因子组的排名合并为组级排名(等权平均)。"""
    print("📊 计算因子组排名...")
    for date in list(ranks.keys()):
        df = ranks[date]
        for group_name, factors in factor_groups.items():
            available = [f for f in factors if f in df.columns]
            if available:
                df[group_name] = df[available].mean(axis=1)
        ranks[date] = df
    print(f"  ✅ 因子组排名已添加: {list(factor_groups.keys())}")
    return ranks


# ═══════════════════════════════════════════════════
#  Growth Composite
# ═══════════════════════════════════════════════════

def add_growth_composite(ranks):
    """添加growth_composite (gc_baseline)。
    
    gc_baseline = 0.60 × fund_growth + 0.25 × analyst + 0.15 × income
    """
    gc_func = lambda d: (
        d.get('fund_growth', 0) * 0.60 +
        d.get('analyst', 0) * 0.25 +
        d.get('income', 0) * 0.15
    )

    for date in ranks:
        df = ranks[date]
        try:
            df['gc_baseline'] = gc_func(df.to_dict('series'))
        except Exception:
            df['gc_baseline'] = np.nan
        ranks[date] = df

    print("  ✅ gc_baseline (growth_composite) 已添加")
    return ranks


# ═══════════════════════════════════════════════════
#  Rank Inversion 检查 (Top5% vs Bottom20%)
# ═══════════════════════════════════════════════════

def compute_combined_scores(ranks, date, weights):
    """计算因子组合分数。"""
    if date in ranks:
        r = ranks[date]
    else:
        rank_dates = sorted(ranks.keys())
        candidates = [d for d in rank_dates if d <= date]
        if not candidates:
            return None
        r = ranks[candidates[-1]]

    available = [f for f in weights if f in r.columns and weights[f] > 0]
    if not available:
        return None
    combined = pd.Series(0.0, index=r.index)
    for f in available:
        combined = combined + weights[f] * r[f]
    return combined.dropna().sort_values(ascending=False)


def check_real_rank_inversion(ranks, prices, weights, windows):
    """正确的Rank Inversion检查: Top5% vs Bottom20% 前瞻收益。"""
    print("\n🔍 正确的Rank Inversion检查 (Top5% vs Bottom20% 前瞻收益)...")
    
    results_per_window = []
    valid_count = 0
    pass_count = 0
    
    for w in windows:
        if "error" in w:
            results_per_window.append({
                "window": w["index"],
                "period": w["period"],
                "status": "SKIPPED",
                "reason": "Window failed backtest"
            })
            continue
        
        period = w["period"]
        try:
            parts = period.split(" → ")
            test_start_str = parts[0].strip()
            test_end_str = parts[1].strip()
        except (IndexError, ValueError):
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "ERROR",
                "reason": "Cannot parse period"
            })
            continue
        
        scores = compute_combined_scores(ranks, test_start_str, weights)
        if scores is None or len(scores) < 20:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": f"Insufficient scores at {test_start_str}"
            })
            continue
        
        price_dates = sorted(prices.index.astype(str))
        start_candidates = [d for d in price_dates if d >= test_start_str]
        end_candidates = [d for d in price_dates if d >= test_end_str]
        
        if not start_candidates or not end_candidates:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": "No price data for period"
            })
            continue
        
        actual_start = start_candidates[0]
        actual_end = end_candidates[0]
        
        if actual_start not in prices.index or actual_end not in prices.index:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": "Price dates not in index"
            })
            continue
        
        start_prices = prices.loc[actual_start]
        end_prices = prices.loc[actual_end]
        
        common_tickers = scores.index.intersection(start_prices.index).intersection(end_prices.index)
        valid_start = start_prices[common_tickers]
        valid_end = end_prices[common_tickers]
        
        mask = (valid_start > 0) & valid_end.notna() & valid_start.notna()
        valid_tickers = common_tickers[mask]
        
        if len(valid_tickers) < 20:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": f"Insufficient valid prices ({len(valid_tickers)})"
            })
            continue
        
        fwd_returns = (valid_end[valid_tickers] / valid_start[valid_tickers]) - 1
        
        n_top5 = max(1, int(len(scores) * 0.05))
        top5_tickers = scores.nlargest(n_top5).index
        top5_tickers = [t for t in top5_tickers if t in fwd_returns.index]
        
        n_bot20 = max(1, int(len(scores) * 0.20))
        bot20_tickers = scores.nsmallest(n_bot20).index
        bot20_tickers = [t for t in bot20_tickers if t in fwd_returns.index]
        
        if len(top5_tickers) == 0 or len(bot20_tickers) == 0:
            results_per_window.append({
                "window": w["index"],
                "period": period,
                "status": "SKIPPED",
                "reason": "No tickers in top/bottom groups"
            })
            continue
        
        top5_ret = float(fwd_returns[top5_tickers].mean())
        bot20_ret = float(fwd_returns[bot20_tickers].mean())
        ri_passed = top5_ret > bot20_ret
        
        valid_count += 1
        if ri_passed:
            pass_count += 1
        
        results_per_window.append({
            "window": w["index"],
            "period": period,
            "status": "PASS" if ri_passed else "FAIL",
            "top5_pct_count": len(top5_tickers),
            "top5_avg_return": round(top5_ret, 4),
            "bottom20_pct_count": len(bot20_tickers),
            "bottom20_avg_return": round(bot20_ret, 4),
            "spread": round(top5_ret - bot20_ret, 4),
            "sharpe": w["sharpe"],
        })
        
        mark = "✅" if ri_passed else "❌"
        print(f"    {mark} W{w['index']}: {period} | "
              f"Top5%={top5_ret:+.2%} Bottom20%={bot20_ret:+.2%} "
              f"Spread={top5_ret-bot20_ret:+.2%}")
    
    overall_passed = pass_count > valid_count * 0.5 if valid_count > 0 else False
    
    all_top5_rets = [r["top5_avg_return"] for r in results_per_window if "top5_avg_return" in r]
    all_bot20_rets = [r["bottom20_avg_return"] for r in results_per_window if "bottom20_avg_return" in r]
    all_spreads = [r["spread"] for r in results_per_window if "spread" in r]
    
    summary = {
        "passed": overall_passed,
        "method": "Top5% vs Bottom20% forward returns",
        "valid_windows": valid_count,
        "pass_windows": pass_count,
        "pass_rate": round(pass_count / valid_count, 3) if valid_count > 0 else 0,
        "avg_top5_return": round(float(np.mean(all_top5_rets)), 4) if all_top5_rets else None,
        "avg_bottom20_return": round(float(np.mean(all_bot20_rets)), 4) if all_bot20_rets else None,
        "avg_spread": round(float(np.mean(all_spreads)), 4) if all_spreads else None,
        "per_window": results_per_window,
    }
    
    print(f"\n  📊 Rank Inversion Summary:")
    print(f"     Method: Top5% vs Bottom20% forward returns")
    print(f"     Valid windows: {valid_count}")
    print(f"     Pass: {pass_count}/{valid_count} ({summary['pass_rate']:.0%})")
    if all_top5_rets:
        print(f"     Avg Top5% return: {summary['avg_top5_return']:+.2%}")
        print(f"     Avg Bottom20% return: {summary['avg_bottom20_return']:+.2%}")
        print(f"     Avg spread: {summary['avg_spread']:+.2%}")
    print(f"     Overall: {'✅ PASS' if overall_passed else '❌ FAIL'}")
    
    return summary


# ═══════════════════════════════════════════════════
#  Walk-Forward (使用MarketAwareEngine)
# ═══════════════════════════════════════════════════

def run_wf(ranks, prices, factor_weights, market_regime, position_strategy,
           train_years=0.5, test_months=6, hold_days=30, top_n=10,
           cost=0.001, stop_loss=-0.15):
    """运行Walk-Forward, 使用MarketAwareEngine。"""
    engine = MarketAwareEngine(
        cost=cost, stop_loss=stop_loss,
        position_strategy=position_strategy,
        market_regime=market_regime,
    )
    dates = sorted(ranks.keys())
    if not dates:
        return None, []

    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])
    train_start = start
    windows = []
    idx = 0

    train_months = int(train_years * 12)
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if str(test_end) > str(end):
            break
        tss = str(train_end)[:10]
        tes = str(test_end)[:10]
        try:
            result, baseline = engine.run(
                ranks, prices, factor_weights, hold_days, top_n,
                start_date=tss, end_date=tes, run_baseline=True
            )
            windows.append({
                "index": idx, "period": f"{tss} → {tes}",
                "sharpe": result.sharpe, "max_dd": result.max_dd,
                "cagr": result.cagr, "win_rate": result.win_rate,
                "n_trades": result.n_trades, "n_days": len(result.daily_equity),
                "baseline_sharpe": baseline.sharpe if baseline else None,
                "warnings": result.warnings,
            })
        except (DataQualityError, Exception) as e:
            windows.append({"index": idx, "period": f"{tss} → {tes}", "error": str(e)[:200]})
        idx += 1
        train_start += pd.DateOffset(months=test_months)

    if not windows:
        return None, []

    valid = [w for w in windows if "sharpe" in w]
    if not valid:
        return {"error": "All windows failed", "windows": windows}, windows

    sharpes = [w["sharpe"] for w in valid]
    dds = [w["max_dd"] for w in valid]
    cagrs = [w["cagr"] for w in valid]
    wrs = [w["win_rate"] for w in valid]

    ri = check_real_rank_inversion(ranks, prices, factor_weights, windows)

    result = {
        "sharpe": round(float(np.mean(sharpes)), 3),
        "max_dd": round(float(np.min(dds)), 4),
        "cagr": round(float(np.mean(cagrs)), 4),
        "win_rate": round(float(np.mean(wrs)), 3),
        "n_trades": sum(w["n_trades"] for w in valid),
        "n_windows": len(valid),
        "total_windows": len(windows),
        "failed_windows": len(windows) - len(valid),
        "rank_inversion": ri,
        "warnings": [],
        "status": "PASS" if ri["passed"] else "WARN",
    }
    return result, windows


# ═══════════════════════════════════════════════════
#  JSON序列化辅助
# ═══════════════════════════════════════════════════

def serialize(obj):
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ═══════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 80)
    print("🦅 Falcon V0.4.5: Market-Aware Dynamic Position Sizing")
    print("=" * 80)

    # ─── 1. 加载数据 ───
    df, price_pivot, vix_df = load_data()

    # ─── 2. 确定可用PIT因子 ───
    all_pit_cols = []
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in df.columns]
        all_pit_cols.extend(available)
        print(f"  {group}: {len(available)}/{len(cols)} factors")
    print(f"  Total PIT factors: {len(all_pit_cols)}")

    # ─── 3. 计算截面百分位排名 ───
    ranks = compute_cross_sectional_ranks(df, all_pit_cols)

    # ─── 4. 计算因子组排名 ───
    ranks = compute_group_ranks(ranks, FACTOR_GROUPS)

    # ─── 5. 添加growth_composite ───
    ranks = add_growth_composite(ranks)

    # ─── 6. 构建市场状态 ───
    print("\n🌐 构建市场状态...")
    t1 = time.time()
    market_regime = MarketRegime(vix_df, price_pivot)
    regime_summary = market_regime.get_regime_summary()
    print(f"  ✅ 市场状态构建完成 ({time.time()-t1:.1f}秒)")
    print(f"  📊 市场状态分布:")
    print(f"     牛市: {regime_summary['bull_days']}天 ({regime_summary['bull_pct']:.1%})")
    print(f"     震荡: {regime_summary['neutral_days']}天 ({regime_summary['neutral_pct']:.1%})")
    print(f"     熊市: {regime_summary['bear_days']}天 ({regime_summary['bear_pct']:.1%})")
    print(f"     极端熊市: {regime_summary['extreme_bear_days']}天 ({regime_summary['extreme_bear_pct']:.1%})")

    # ═══════════════════════════════════════════════
    #  测试4种仓位策略
    # ═══════════════════════════════════════════════
    strategies = [
        ("fixed_100", "固定仓位100% (baseline)", FixedPosition()),
        ("vix_only", "VIX-only动态仓位", VIXOnlyPosition()),
        ("trend_only", "趋势-only动态仓位", TrendOnlyPosition()),
        ("vix_trend", "VIX+趋势综合动态仓位", VIXTrendPosition()),
    ]

    results_all = {}

    for name, desc, strategy in strategies:
        print(f"\n{'='*60}")
        print(f"📊 Strategy: {desc}")
        print(f"{'='*60}")
        print(f"  Strategy: {strategy}")

        res, wins = run_wf(
            ranks, price_pivot, V044_WEIGHTS, market_regime, strategy,
            train_years=0.5, test_months=6, hold_days=30, top_n=10,
            cost=0.001, stop_loss=-0.15,
        )

        if res:
            ri_str = "PASS" if res["rank_inversion"]["passed"] else "FAIL"
            print(f"\n  📊 Result:")
            print(f"     Sharpe: {res['sharpe']:.3f}")
            print(f"     MaxDD: {res['max_dd']:.1%}")
            print(f"     CAGR: {res['cagr']:.1%}")
            print(f"     Win Rate: {res['win_rate']:.0%}")
            print(f"     Windows: {res['n_windows']}/{res['total_windows']}")
            print(f"     Rank Inversion: {ri_str} ({res['rank_inversion']['pass_windows']}/{res['rank_inversion']['valid_windows']})")

            # 窗口详情
            valid_wins = [w for w in wins if "sharpe" in w]
            for w in valid_wins:
                ri_mark = "✅" if w["sharpe"] > 0 else "❌"
                baseline_str = f"  Base={w['baseline_sharpe']:.3f}" if w.get('baseline_sharpe') else ""
                print(f"    {ri_mark} W{w['index']}: {w['period']}  "
                      f"Sharpe={w['sharpe']:.3f}  MaxDD={w['max_dd']:.1%}  "
                      f"WR={w['win_rate']:.0%}  Trades={w['n_trades']}{baseline_str}")

            results_all[name] = {
                "description": desc,
                "strategy_class": strategy.__class__.__name__,
                "result": serialize(res),
                "window_details": serialize(wins),
            }
        else:
            print(f"  ❌ No result (all windows failed)")
            results_all[name] = {
                "description": desc,
                "strategy_class": strategy.__class__.__name__,
                "result": None,
                "window_details": [],
            }

    # ═══════════════════════════════════════════════
    #  对比分析
    # ═══════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"📋 Strategy Comparison")
    print(f"{'='*80}")

    baseline_sharpe = results_all["fixed_100"]["result"]["sharpe"] if results_all["fixed_100"]["result"] else 0

    print(f"  {'Strategy':<30} {'Sharpe':>8} {'MaxDD':>8} {'CAGR':>8} {'WR':>6} {'RI':>6} {'vs Base':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8}")

    for name, data in results_all.items():
        res = data["result"]
        if res:
            ri_mark = "✅" if res["rank_inversion"]["passed"] else "❌"
            improvement = ((res["sharpe"] - baseline_sharpe) / baseline_sharpe * 100) if baseline_sharpe > 0 else 0
            print(f"  {data['description']:<30} {res['sharpe']:>8.3f} {res['max_dd']:>7.1%} "
                  f"{res['cagr']:>7.1%} {res['win_rate']:>5.0%} {ri_mark:>6} {improvement:>+7.1f}%")
        else:
            print(f"  {data['description']:<30} {'N/A':>8}")

    # 找最佳策略 (RI通过 + Sharpe最高)
    best_name = None
    best_sharpe = -999
    for name, data in results_all.items():
        res = data["result"]
        if res and res["rank_inversion"]["passed"] and res["sharpe"] > best_sharpe:
            best_sharpe = res["sharpe"]
            best_name = name
    
    if best_name:
        print(f"\n  🏆 Best strategy: {results_all[best_name]['description']} (Sharpe={best_sharpe:.3f})")
    else:
        # Fallback: highest Sharpe regardless of RI
        for name, data in results_all.items():
            res = data["result"]
            if res and res["sharpe"] > best_sharpe:
                best_sharpe = res["sharpe"]
                best_name = name
        if best_name:
            print(f"\n  🏆 Best strategy (no RI pass): {results_all[best_name]['description']} (Sharpe={best_sharpe:.3f})")

    # ═══════════════════════════════════════════════
    #  保存结果
    # ═══════════════════════════════════════════════
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "task": "V0.4.5 Market-Aware Dynamic Position Sizing",
            "config": {
                "train_years": 0.5,
                "test_months": 6,
                "hold_days": 30,
                "top_n": 10,
                "cost": 0.001,
                "stop_loss": -0.15,
                "factor_weights": V044_WEIGHTS,
            },
            "features": "features_v04_1.parquet",
            "vix_source": "data/us/vix_10y.parquet",
            "pit_factors": len(all_pit_cols),
            "factor_groups": {k: len(v) for k, v in FACTOR_GROUPS.items()},
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "market_regime_summary": regime_summary,
        "strategies": results_all,
        "best_strategy": best_name,
        "comparison_with_v044": {
            "v044_sharpe": 2.122,
            "v044_config": {
                "fund_ratio": 0.45,
                "growth_composite": 0.20,
                "qoq": 0.20,
                "cashflow": 0.15,
                "position": "fixed_100%",
            },
            "best_v045_sharpe": best_sharpe if best_name else 0,
            "best_v045_strategy": best_name,
            "improvement_pct": round(
                (best_sharpe - 2.122) / 2.122 * 100, 1
            ) if best_name and best_sharpe > 0 else 0,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved: {OUTPUT_PATH}")
    print(f"⏱️ Total time: {(time.time()-t0)/60:.1f} minutes")

    # Final summary
    print(f"\n{'='*80}")
    print(f"📋 Final Summary")
    print(f"{'='*80}")
    print(f"  Factor weights: {V044_WEIGHTS}")
    print(f"  Market regime: {regime_summary}")
    if best_name:
        best_res = results_all[best_name]["result"]
        if best_res:
            print(f"  Best strategy: {results_all[best_name]['description']}")
            print(f"  Best WF Sharpe: {best_res['sharpe']:.3f}")
            print(f"  Best WF MaxDD: {best_res['max_dd']:.1%}")
            print(f"  Rank Inversion: {'PASS' if best_res['rank_inversion']['passed'] else 'FAIL'}")
    print(f"  V0.4.4 baseline Sharpe: 2.122")
    if best_name and best_sharpe > 0:
        improvement = (best_sharpe - 2.122) / 2.122 * 100
        print(f"  Improvement vs V0.4.4: {improvement:+.1f}%")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
