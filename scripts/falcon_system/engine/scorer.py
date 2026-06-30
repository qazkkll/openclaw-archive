"""
🦅 Falcon 评分引擎
==================
统一评分接口：数据输入 → 因子计算 → 排名 → 信号输出。
所有因子经过IC/ICIR验证，权重数据驱动。
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ..core.config import CONFIG, DATA_DIR
from ..core.data_manager import DataManager


# ════════════════════════════════════════════════════════════════
# 因子定义
# ════════════════════════════════════════════════════════════════

# FMP字段映射
RATIO_FIELDS = [
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "daysOfInventoryOutstanding", "operatingCycle", "daysOfPayablesOutstanding",
    "cashConversionCycle", "grossProfitMargin", "operatingProfitMargin",
    "pretaxProfitMargin", "netProfitMargin", "effectiveTaxRate",
    "returnOnAssets", "returnOnEquity", "returnOnCapitalEmployed",
    "netIncomePerEBT", "ebtPerEbit", "ebitPerRevenue",
    "debtRatio", "debtEquityRatio", "longTermDebtToCapitalization",
    "totalDebtToCapitalization", "interestCoverage", "cashFlowToDebtRatio",
    "companyEquityMultiplier", "receivablesTurnover", "payablesTurnover",
    "inventoryTurnover", "fixedAssetTurnover", "assetTurnover",
    "operatingCashFlowPerShare", "freeCashFlowPerShare",
    "cashPerShare", "payoutRatio", "salesGeneralAndAdministrativeToRevenue",
    "researchAndDdevelopementToRevenue", "intangiblesToTotalAssets",
    "capexToOperatingCashFlow", "capexToRevenue", "capexToDepreciation",
    "stockBasedCompensationToRevenue", "grahamNumber", "grahamNetNet",
    "workingCapital", "tangibleBookValue", "netCurrentAssetValue",
    "investedCapital", "averageReceivables", "averagePayables",
    "averageInventory", "ebitda", "ebit", "consensusRating",
]

METRIC_FIELDS = [
    "enterpriseValue", "marketCap", "peRatio", "priceToSalesRatio",
    "pbRatio", "pfcfRatio", "pocfratio", "evToSales", "enterpriseValueOverEBITDA",
    "evToFreeCashFlow", "evToOperatingCashFlow", "priceToBook",
    "priceToTangibleBook", "priceEarningsRatio", "priceToFreeCashFlowsRatio",
    "priceToOperatingCashFlowsRatio", "priceCashFlowRatio",
    "priceSalesRatio", "priceFairValue", "dividendYield",
    "currentRatio", "quickRatio", "cashRatio", "daysOfSalesOutstanding",
    "daysOfInventoryOutstanding", "operatingCycle", "daysOfPayablesOutstanding",
    "cashConversionCycle", "grossProfitMargin", "operatingProfitMargin",
    "pretaxProfitMargin", "netProfitMargin", "effectiveTaxRate",
    "returnOnAssets", "returnOnEquity", "returnOnCapitalEmployed",
    "netIncomePerEBT", "ebtPerEbit", "ebitPerRevenue",
    "debtRatio", "debtEquityRatio", "longTermDebtToCapitalization",
    "totalDebtToCapitalization", "interestCoverage", "cashFlowToDebtRatio",
    "companyEquityMultiplier", "receivablesTurnover", "payablesTurnover",
    "inventoryTurnover", "fixedAssetTurnover", "assetTurnover",
    "operatingCashFlowPerShare", "freeCashFlowPerShare",
    "cashPerShare", "payoutRatio", "salesGeneralAndAdministrativeToRevenue",
    "researchAndDdevelopementToRevenue", "intangiblesToTotalAssets",
    "capexToOperatingCashFlow", "capexToRevenue", "capexToDepreciation",
    "stockBasedCompensationToRevenue", "grahamNumber", "grahamNetNet",
    "workingCapital", "tangibleBookValue", "netCurrentAssetValue",
    "investedCapital", "averageReceivables", "averagePayables",
    "averageInventory", "ebitda", "ebit", "consensusRating",
]

GROWTH_FIELDS = [
    "revenueGrowth", "grossProfitGrowth", "ebitgrowth",
    "operatingIncomeGrowth", "netIncomeGrowth", "epsgrowth",
    "epsdilutedGrowth", "weightedAverageSharesGrowth",
    "weightedAverageSharesDilutedGrowth", "dividendsperShareGrowth",
    "freeCashFlowGrowth", "tenYRevenueGrowthPerShare",
    "fiveYRevenueGrowthPerShare", "threeYRevenueGrowthPerShare",
    "tenYOperatingCFGrowthPerShare", "fiveYOperatingCFGrowthPerShare",
    "threeYOperatingCFGrowthPerShare", "tenYNetIncomeGrowthPerShare",
    "fiveYNetIncomeGrowthPerShare", "threeYNetIncomeGrowthPerShare",
    "tenYShareholdersEquityGrowthPerShare", "fiveYShareholdersEquityGrowthPerShare",
    "threeYShareholdersEquityGrowthPerShare", "tenYDividendGrowthPerShare",
    "fiveYDividendGrowthPerShare", "threeYDividendGrowthPerShare",
    "receivablesGrowth", "inventoryGrowth", "assetGrowth",
    "bookValueperShareGrowth", "debtGrowth", "rdexpenseGrowth",
    "sgaexpensesGrowth",
]

ANALYST_FIELDS = [
    "eps_revision_7d", "eps_revision_30d", "eps_revision_90d",
    "revenue_revision_7d", "revenue_revision_30d", "revenue_revision_90d",
]

TECH_FIELDS = ["rsi_14", "macd_signal", "bb_position", "volume_ratio"]

EARNINGS_FIELDS = [
    "earnings_surprise", "earnings_surprise_2q",
    "earnings_beat_count_4q", "earnings_price_reaction",
]

GRADE_FIELDS = [
    "grade_upgrade_ratio_90d", "grade_downgrade_ratio_90d",
    "grade_momentum_90d", "grade_target_raised_90d",
]

# 三大报表因子
BALANCE_FIELDS = [
    "current_ratio", "debt_to_equity", "net_debt_to_assets",
    "working_capital_ratio", "tangible_book_ratio",
]

CASHFLOW_FIELDS = [
    "fcf_yield", "capex_intensity", "ocf_to_revenue",
    "dividend_coverage", "buyback_yield",
]

INCOME_FIELDS = [
    "gross_margin", "operating_margin", "net_margin",
    "rd_intensity", "margin_trend_4q",
]


# ════════════════════════════════════════════════════════════════
# 信号类型
# ════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """交易信号"""
    ticker: str
    date: str
    score: float
    rank_pct: float
    signal_type: str  # 🟢🟢 / 🟢 / 🟡 / 🔴
    close: float
    universe: str = "SPX"
    
    # 因子详情
    factors: Dict[str, float] = field(default_factory=dict)
    
    # 目标价位(由Pricer计算)
    target_buy: Optional[float] = None
    stop_loss: Optional[float] = None
    target_sell: Optional[float] = None
    atr: Optional[float] = None
    
    # 仓位建议(由PositionSizer计算)
    suggested_qty: Optional[int] = None
    suggested_value: Optional[float] = None
    position_pct: Optional[float] = None


@dataclass
class ScoringResult:
    """评分结果"""
    date: str
    model_version: str
    signals: List[Signal]
    vix_value: Optional[float] = None
    vix_skip: bool = False
    universe_size: int = 0
    scoring_time_seconds: float = 0.0


# ════════════════════════════════════════════════════════════════
# PIT数据查询
# ════════════════════════════════════════════════════════════════

def get_pit(records: List[Dict], date: str) -> Dict:
    """Point-in-Time查询：返回date之前最近的记录"""
    if not records:
        return {}
    prior = [r for r in records if r.get("date", "") <= date]
    if not prior:
        return {}
    return max(prior, key=lambda r: r["date"])


def get_pit_insider(records: List[Dict], date: str) -> Dict:
    """PIT查询insider交易数据"""
    if not records:
        return {}
    prior = [r for r in records if r.get("transactionDate", "") <= date]
    if not prior:
        return {}
    # 最近90天的insider交易
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        start = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
        recent = [r for r in prior if r.get("transactionDate", "") >= start]
        if not recent:
            return {}
        n_buy = sum(1 for r in recent if r.get("type") in ["Purchase", "buy"])
        n_sell = sum(1 for r in recent if r.get("type") in ["Sale", "sell"])
        return {"insider_buy_count": n_buy, "insider_sell_count": n_sell}
    except:
        return {}


# ════════════════════════════════════════════════════════════════
# 三大报表因子计算
# ════════════════════════════════════════════════════════════════

def build_pit_index(data: Dict, use_filing_date: bool = False) -> Dict:
    """构建PIT索引"""
    index = {}
    for ticker, records in data.items():
        if not isinstance(records, list):
            continue
        index[ticker] = {}
        for r in records:
            if use_filing_date:
                date = r.get("filingDate", r.get("date", ""))
            else:
                date = r.get("date", "")
            if date:
                index[ticker][date] = r
    return index


def compute_statement_factors(ticker: str, date: str, 
                               balance_idx: Dict, cashflow_idx: Dict, 
                               income_idx: Dict, _) -> Dict[str, Optional[float]]:
    """计算三大报表因子"""
    factors = {}
    
    # 资产负债表因子
    balance_records = balance_idx.get(ticker, {})
    balance = get_pit(list(balance_records.values()), date)
    if balance:
        total_assets = balance.get("totalAssets", 0) or 0
        total_liabilities = balance.get("totalLiabilities", 0) or 0
        total_equity = balance.get("totalStockholdersEquity", 0) or 0
        current_assets = balance.get("totalCurrentAssets", 0) or 0
        current_liabilities = balance.get("totalCurrentLiabilities", 0) or 0
        total_debt = balance.get("totalDebt", 0) or 0
        cash = balance.get("cashAndCashEquivalents", 0) or 0
        intangible = balance.get("intangibleAssets", 0) or 0
        goodwill = balance.get("goodwill", 0) or 0
        
        if current_liabilities > 0:
            factors["current_ratio"] = current_assets / current_liabilities
        if total_equity > 0:
            factors["debt_to_equity"] = total_debt / total_equity
        if total_assets > 0:
            factors["net_debt_to_assets"] = (total_debt - cash) / total_assets
            factors["working_capital_ratio"] = (current_assets - current_liabilities) / total_assets
            tangible_book = total_equity - intangible - goodwill
            factors["tangible_book_ratio"] = tangible_book / total_assets
    
    # 利润表PIT查询(提前到cashflow之前,避免UnboundLocalError)
    income_records = income_idx.get(ticker, {})
    income = get_pit(list(income_records.values()), date)
    
    # 现金流量表因子
    cashflow_records = cashflow_idx.get(ticker, {})
    cashflow = get_pit(list(cashflow_records.values()), date)
    if cashflow:
        ocf = cashflow.get("operatingCashFlow", 0) or 0
        capex = abs(cashflow.get("capitalExpenditure", 0) or 0)
        fcf = cashflow.get("freeCashFlow", 0) or 0
        dividends = abs(cashflow.get("dividendsPaid", 0) or 0)
        buyback = abs(cashflow.get("commonStockRepurchased", 0) or 0)
        
        # FCF Yield需要市值，用EV代替
        ev = balance.get("enterpriseValue", 0) if balance else 0
        if ev and ev > 0:
            factors["fcf_yield"] = fcf / ev
        
        revenue = income.get("revenue", 0) if income_idx.get(ticker) else 0
        if revenue and revenue > 0:
            factors["capex_intensity"] = capex / revenue
            factors["ocf_to_revenue"] = ocf / revenue
        
        if dividends > 0 and fcf > 0:
            factors["dividend_coverage"] = fcf / dividends
        
        market_cap = balance.get("marketCap", 0) if balance else 0
        if market_cap and market_cap > 0:
            factors["buyback_yield"] = buyback / market_cap
    
    # 利润表因子(已提前到cashflow块之前)
    if income:
        revenue = income.get("revenue", 0) or 0
        gross_profit = income.get("grossProfit", 0) or 0
        operating_income = income.get("operatingIncome", 0) or 0
        net_income = income.get("netIncome", 0) or 0
        rd_expense = income.get("researchAndDevelopmentExpenses", 0) or 0
        
        if revenue > 0:
            factors["gross_margin"] = gross_profit / revenue
            factors["operating_margin"] = operating_income / revenue
            factors["net_margin"] = net_income / revenue
            factors["rd_intensity"] = rd_expense / revenue
        
        # 4Q利润率趋势
        all_income = sorted(income_records.values(), key=lambda x: x.get("date", ""))
        recent_4q = [r for r in all_income if r.get("date", "") <= date][-4:]
        if len(recent_4q) >= 2:
            margins = []
            for r in recent_4q:
                rev = r.get("revenue", 0) or 0
                ni = r.get("netIncome", 0) or 0
                if rev > 0:
                    margins.append(ni / rev)
            if len(margins) >= 2:
                factors["margin_trend_4q"] = margins[-1] - margins[0]
    
    return factors


# ════════════════════════════════════════════════════════════════
# 评分引擎
# ════════════════════════════════════════════════════════════════

class ScoringEngine:
    """Falcon评分引擎"""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.config = CONFIG.model
    
    def score(self, target_date: Optional[str] = None, 
              universe: str = "spx") -> ScoringResult:
        """执行评分"""
        import time
        t0 = time.time()
        
        # 检查VIX
        vix_value, vix_date = self.dm.get_latest_vix()
        vix_skip = vix_value is not None and vix_value > self.config.vix_threshold
        
        # 加载数据
        master = self.dm.load_master_prices()
        fundamentals = self.dm.load_fundamentals()
        
        # 确定评分日期
        dates = sorted(master["date"].unique())
        if target_date:
            available = [d for d in dates if d <= target_date]
            if not available:
                raise ValueError(f"无{target_date}之前的交易数据")
            date = available[-1]
        else:
            date = dates[-1]
        
        # 获取当日数据
        day = master[master["date"] == date].copy()
        if len(day) < 10:
            raise ValueError(f"{date}只有{len(day)}只股票，不足")
        
        day.index = day["ticker"].values
        
        # 计算所有因子
        signals = []
        all_factors = {}
        for _, row in day.iterrows():
            ticker = row["ticker"]
            close = row["close"]
            
            # 计算各因子组得分
            factors = self._compute_factors(ticker, date, row, fundamentals)
            all_factors[ticker] = factors
            
            signals.append(Signal(
                ticker=ticker,
                date=date,
                score=0.0,
                rank_pct=0.0,
                signal_type="",
                close=close,
                factors=factors,
            ))
        
        # 因子归一化：对每个因子做截面percentile ranking
        # 这样raw values (如enterpriseValue数十亿) 会被转换为0-1范围
        factor_names = set()
        for f in all_factors.values():
            factor_names.update(f.keys())
        
        normalized_factors = {t: {} for t in all_factors}
        for fname in factor_names:
            raw_values = {t: all_factors[t].get(fname, 0.5) for t in all_factors}
            # 过滤掉无效值
            valid_values = {t: v for t, v in raw_values.items() if v is not None and not np.isnan(v)}
            if len(valid_values) > 10:
                ranked = pd.Series(valid_values).rank(pct=True)
                for t in valid_values:
                    normalized_factors[t][fname] = float(ranked[t])
            else:
                for t in valid_values:
                    normalized_factors[t][fname] = 0.5
        
        # 计算加权综合分
        for signal in signals:
            factors = normalized_factors.get(signal.ticker, {})
            score = sum(
                self.config.weights.get(f, 0) * factors.get(f, 0.5)
                for f in self.config.weights
                if self.config.weights.get(f, 0) > 0
            )
            signal.score = score
            signal.factors = factors
        
        # 统一排名
        scores = [s.score for s in signals]
        score_series = pd.Series(scores)
        rank_pcts = score_series.rank(pct=True).values
        
        for i, signal in enumerate(signals):
            signal.rank_pct = rank_pcts[i]
            signal.signal_type = self._score_to_signal(signal.score, signal.rank_pct)
        
        # 按分数排序
        signals.sort(key=lambda s: s.score, reverse=True)
        
        elapsed = time.time() - t0
        
        return ScoringResult(
            date=date,
            model_version=self.config.version,
            signals=signals,
            vix_value=vix_value,
            vix_skip=vix_skip,
            universe_size=len(signals),
            scoring_time_seconds=elapsed,
        )
    
    def _compute_factors(self, ticker: str, date: str, 
                         price_row: pd.Series, fundamentals: Dict) -> Dict[str, float]:
        """计算单只股票的所有因子"""
        factors = {}
        
        # Tech因子
        tech_cols = [c for c in price_row.index if c.startswith("t_")]
        if tech_cols:
            factors["tech"] = float(price_row[tech_cols].mean())
        else:
            factors["tech"] = 0.5
        
        # FMP Ratios
        ratio_vals = {}
        for f in RATIO_FIELDS:
            pit = get_pit(fundamentals.get("fmp_ratios_historical", {}).get(ticker, []), date)
            v = pit.get(f)
            if v is not None:
                ratio_vals[f] = v
        factors["fund_ratio"] = np.mean(list(ratio_vals.values())) if ratio_vals else 0.5
        
        # Key Metrics
        metric_vals = {}
        for f in METRIC_FIELDS:
            pit = get_pit(fundamentals.get("fmp_key_metrics", {}).get(ticker, []), date)
            v = pit.get(f)
            if v is not None:
                metric_vals[f] = v
        factors["fund_metric"] = np.mean(list(metric_vals.values())) if metric_vals else 0.5
        
        # Growth
        growth_vals = {}
        for f in GROWTH_FIELDS:
            pit = get_pit(fundamentals.get("fmp_financial_growth", {}).get(ticker, []), date)
            v = pit.get(f)
            if v is not None:
                growth_vals[f] = v
        factors["fund_growth"] = np.mean(list(growth_vals.values())) if growth_vals else 0.5
        
        # Analyst
        analyst_vals = {}
        for f in ANALYST_FIELDS:
            pit = get_pit(fundamentals.get("analyst_historical", {}).get(ticker, []), date)
            v = pit.get(f)
            if v is not None:
                analyst_vals[f] = v
        factors["analyst"] = np.mean(list(analyst_vals.values())) if analyst_vals else 0.5
        
        # Earnings
        earnings_data = fundamentals.get("earnings", {})
        earnings_vals = {}
        if earnings_data:
            records = earnings_data.get(ticker, [])
            past = [r for r in records if r.get("date", "") <= date]
            if past:
                last = past[-1]
                for f in EARNINGS_FIELDS:
                    v = last.get(f)
                    if v is not None:
                        earnings_vals[f] = v
        factors["earnings"] = np.mean(list(earnings_vals.values())) if earnings_vals else 0.5
        
        # Grade Sentiment
        grades_data = fundamentals.get("grades", {})
        grade_vals = {}
        if grades_data:
            records = grades_data.get(ticker, [])
            try:
                dt = datetime.strptime(date, "%Y-%m-%d")
                start = (dt - timedelta(days=90)).strftime("%Y-%m-%d")
                recent = [r for r in records if start <= r.get("date", "") <= date]
                if recent:
                    up = sum(1 for r in recent if r.get("upgrade"))
                    down = sum(1 for r in recent if r.get("downgrade"))
                    total = len(recent)
                    if total > 0:
                        grade_vals["grade_momentum_90d"] = (up - down) / total
            except:
                pass
        factors["grade_sentiment"] = np.mean(list(grade_vals.values())) if grade_vals else 0.5
        
        # Insider
        insider_data = fundamentals.get("fmp_insider", {})
        pit = get_pit_insider(insider_data.get(ticker, []), date)
        if pit:
            n_buy = pit.get("insider_buy_count", 0)
            n_sell = pit.get("insider_sell_count", 0)
            total = n_buy + n_sell
            factors["insider"] = n_buy / total if total > 0 else 0.5
        else:
            factors["insider"] = 0.5
        
        # 三大报表因子
        balance_idx = build_pit_index(fundamentals.get("fmp_balance_sheet", {}))
        cashflow_idx = build_pit_index(fundamentals.get("fmp_cashflow", {}))
        income_idx = build_pit_index(fundamentals.get("fmp_income_stmt", {}), use_filing_date=True)
        
        stmt_factors = compute_statement_factors(
            ticker, date, balance_idx, cashflow_idx, income_idx, {}
        )
        
        # Balance
        balance_vals = [stmt_factors[f] for f in BALANCE_FIELDS if f in stmt_factors]
        factors["balance"] = np.mean(balance_vals) if balance_vals else 0.5
        
        # Cashflow
        cashflow_vals = [stmt_factors[f] for f in CASHFLOW_FIELDS if f in stmt_factors]
        factors["cashflow"] = np.mean(cashflow_vals) if cashflow_vals else 0.5
        
        # Income
        income_vals = [stmt_factors[f] for f in INCOME_FIELDS if f in stmt_factors]
        factors["income_stmt"] = np.mean(income_vals) if income_vals else 0.5
        
        # 反向因子处理
        for inv_f in CONFIG.model.invert_factors:
            if inv_f in factors:
                factors[inv_f] = 1 - factors[inv_f]
        
        return factors
    
    def _score_to_signal(self, score: float, pct: float) -> str:
        """分数转信号类型"""
        if score >= 0.55 and pct >= 0.95:
            return "🟢🟢"
        elif score >= 0.55 and pct >= 0.80:
            return "🟢"
        elif score >= 0.50:
            return "🟡"
        else:
            return "🔴"


# ════════════════════════════════════════════════════════════════
# 目标价位计算
# ════════════════════════════════════════════════════════════════

class Pricer:
    """目标价位计算器"""
    
    def __init__(self, data_manager: DataManager):
        self.dm = data_manager
        self.config = CONFIG.trading
    
    def calculate_targets(self, signals: List[Signal]) -> List[Signal]:
        """为信号计算目标价位"""
        master = self.dm.load_master_prices()
        
        for signal in signals:
            ticker = signal.ticker
            current_price = signal.close
            
            # 计算ATR
            atr = self._calculate_atr(ticker, master)
            if atr is None:
                atr = current_price * 0.02  # 默认2%
            
            # 找支撑位
            support = self._find_support(ticker, master)
            
            # 目标买入价: 当前价 - ATR * multiplier
            target_buy = current_price - atr * self.config.atr_multiplier
            
            # 如果支撑位更高，用支撑位
            if support and support > target_buy:
                target_buy = support
            
            # 不能太远(最多回调5%)
            max_drop = current_price * (1 - self.config.max_drop_pct)
            if target_buy < max_drop:
                target_buy = max_drop
            
            # 止损价: 基于ATR
            stop_loss = current_price - atr * 3
            if stop_loss < current_price * (1 + CONFIG.model.stop_loss):
                stop_loss = current_price * (1 + CONFIG.model.stop_loss)
            
            # 目标卖出价: 风险收益比至少2:1
            risk = current_price - stop_loss
            target_sell = current_price + risk * 2
            
            signal.target_buy = round(target_buy, 2)
            signal.stop_loss = round(stop_loss, 2)
            signal.target_sell = round(target_sell, 2)
            signal.atr = round(atr, 2)
        
        return signals
    
    def _calculate_atr(self, ticker: str, master: pd.DataFrame, period: int = 14) -> Optional[float]:
        """计算ATR"""
        try:
            ticker_data = master[master["ticker"] == ticker].tail(period + 1)
            if len(ticker_data) < period:
                return None
            
            if "high" in ticker_data.columns and "low" in ticker_data.columns:
                high = ticker_data["high"].values
                low = ticker_data["low"].values
                close = ticker_data["close"].values
                
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1])
                    )
                )
                return float(np.mean(tr[-period:]))
            else:
                # 没有high/low数据，用close的波动率代替
                close = ticker_data["close"].values
                returns = np.abs(np.diff(close) / close[:-1])
                return float(np.mean(returns[-period:]) * close[-1])
        except:
            return None
    
    def _find_support(self, ticker: str, master: pd.DataFrame, lookback: int = 20) -> Optional[float]:
        """找支撑位"""
        try:
            ticker_data = master[master["ticker"] == ticker].tail(lookback)
            if len(ticker_data) < 5:
                return None
            
            if "low" in ticker_data.columns:
                return float(ticker_data["low"].min())
            else:
                return float(ticker_data["close"].min())
        except:
            return None


# ════════════════════════════════════════════════════════════════
# 仓位计算器
# ════════════════════════════════════════════════════════════════

class PositionSizer:
    """仓位计算器"""
    
    def __init__(self):
        self.config = CONFIG.trading
    
    def calculate_positions(self, signals: List[Signal], 
                           account_equity: float,
                           existing_positions: Dict) -> List[Signal]:
        """计算仓位大小"""
        # 可用资金
        available_equity = account_equity * self.config.max_total_exposure
        existing_value = sum(
            float(p.get("qty", 0)) * float(p.get("current_price", 0))
            for p in existing_positions.values()
        )
        available_cash = available_equity - existing_value
        
        if available_cash <= 0:
            return signals
        
        # 按分数加权分配
        total_score = sum(s.score for s in signals if s.target_buy)
        if total_score <= 0:
            return signals
        
        for signal in signals:
            if not signal.target_buy:
                continue
            
            # 分数加权
            weight = signal.score / total_score
            alloc = available_cash * weight
            
            # 限制单只最大仓位
            max_alloc = account_equity * self.config.max_position_pct
            alloc = min(alloc, max_alloc)
            
            # 计算股数
            qty = int(alloc / signal.target_buy)
            if qty <= 0:
                continue
            
            actual_value = qty * signal.target_buy
            if actual_value < self.config.min_order_value:
                continue
            
            signal.suggested_qty = qty
            signal.suggested_value = round(actual_value, 2)
            signal.position_pct = round(actual_value / account_equity * 100, 2)
        
        return signals


# ════════════════════════════════════════════════════════════════
# 便捷函数
# ════════════════════════════════════════════════════════════════

def run_scoring(target_date: Optional[str] = None) -> ScoringResult:
    """运行完整评分流程"""
    from ..core.data_manager import data_manager
    
    engine = ScoringEngine(data_manager)
    pricer = Pricer(data_manager)
    sizer = PositionSizer()
    
    # 1. 评分
    result = engine.score(target_date)
    
    # 2. 筛选🟢🟢信号
    green2 = [s for s in result.signals if s.signal_type == "🟢🟢"]
    
    # 3. 计算目标价位
    green2 = pricer.calculate_targets(green2)
    
    # 4. 计算仓位(需要账户信息)
    # 这里先不计算，由broker模块负责
    
    return result
