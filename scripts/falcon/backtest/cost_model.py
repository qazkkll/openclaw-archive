"""
Falcon统一回测框架 — 成本模型

真实Futu OpenD成本结构(2026-07实测):
  买入: commission + platform_fee (每股, 有最低限额)
  卖出: commission + platform_fee + SEC_fee + FINRA_TAF

用法:
    cost = FutuCostModel(config)
    buy_cost = cost.buy_cost(price, shares)
    sell_cost = cost.sell_cost(price, shares)
    round_trip_pct = cost.round_trip_pct(price, shares)
"""
from dataclasses import dataclass


@dataclass
class TradeCost:
    """单笔交易成本明细。"""
    commission: float      # 佣金
    platform_fee: float    # 平台费
    sec_fee: float         # SEC费(仅卖出)
    finra_taf: float       # FINRA TAF(仅卖出)
    total: float           # 总成本
    
    @property
    def total_pct(self) -> float:
        """成本占交易金额比例(需要外部传入交易金额)。"""
        return self.total


class FutuCostModel:
    """Futu OpenD真实成本模型。
    
    参数全部从backtest_config.yaml读取, 不硬编码。
    """
    
    def __init__(self, config: dict):
        futu = config.get('trading', {}).get('futu', {})
        self.commission_per_share = futu.get('commission_per_share', 0.0049)
        self.platform_fee_per_share = futu.get('platform_fee_per_share', 0.005)
        self.min_commission = futu.get('min_commission', 0.99)
        self.min_platform_fee = futu.get('min_platform_fee', 1.00)
        self.sec_fee_rate = futu.get('sec_fee_rate', 0.0000278)
        self.finra_taf_rate = futu.get('finra_taf_rate', 0.000166)
        self.min_trade_value = futu.get('min_trade_value', 1.0)
    
    def buy_cost(self, price: float, shares: int) -> TradeCost:
        """计算买入成本。
        
        Args:
            price: 股价
            shares: 买入股数
        
        Returns:
            TradeCost明细
        """
        commission = max(self.min_commission, self.commission_per_share * shares)
        platform_fee = max(self.min_platform_fee, self.platform_fee_per_share * shares)
        
        return TradeCost(
            commission=commission,
            platform_fee=platform_fee,
            sec_fee=0.0,
            finra_taf=0.0,
            total=commission + platform_fee,
        )
    
    def sell_cost(self, price: float, shares: int) -> TradeCost:
        """计算卖出成本。
        
        Args:
            price: 股价
            shares: 卖出股数
        
        Returns:
            TradeCost明细
        """
        trade_value = price * shares
        commission = max(self.min_commission, self.commission_per_share * shares)
        platform_fee = max(self.min_platform_fee, self.platform_fee_per_share * shares)
        sec_fee = trade_value * self.sec_fee_rate
        finra_taf = shares * self.finra_taf_rate  # FINRA TAF按股数收费
        
        return TradeCost(
            commission=commission,
            platform_fee=platform_fee,
            sec_fee=sec_fee,
            finra_taf=finra_taf,
            total=commission + platform_fee + sec_fee + finra_taf,
        )
    
    def round_trip_pct(self, price: float, shares: int) -> float:
        """计算一个完整买卖周期的成本百分比。
        
        这是回测中最常用的指标: 买入再卖出, 总成本占交易金额的百分比。
        """
        trade_value = price * shares
        if trade_value < self.min_trade_value:
            return 0.0
        
        buy = self.buy_cost(price, shares)
        sell = self.sell_cost(price, shares)
        return (buy.total + sell.total) / trade_value
    
    def buy_pct(self, price: float, shares: int) -> float:
        """买入成本百分比。"""
        trade_value = price * shares
        if trade_value < self.min_trade_value:
            return 0.0
        return self.buy_cost(price, shares).total / trade_value
    
    def sell_pct(self, price: float, shares: int) -> float:
        """卖出成本百分比。"""
        trade_value = price * shares
        if trade_value < self.min_trade_value:
            return 0.0
        return self.sell_cost(price, shares).total / trade_value


class FlatCostModel:
    """平坦成本模型(旧脚本兼容)。
    
    固定百分比, 不考虑股数/金额。用于对比旧回测结果。
    """
    
    def __init__(self, config: dict):
        trading = config.get('trading', {})
        self.cost_per_side = trading.get('flat_cost_per_side', 0.002)
    
    def buy_pct(self, price: float = 0, shares: int = 0) -> float:
        return self.cost_per_side
    
    def sell_pct(self, price: float = 0, shares: int = 0) -> float:
        return self.cost_per_side
    
    def round_trip_pct(self, price: float = 0, shares: int = 0) -> float:
        return self.cost_per_side * 2


def create_cost_model(config: dict):
    """工厂函数: 根据配置创建成本模型。
    
    config中的trading.cost_model字段决定使用哪个:
      - "futu_real": FutuCostModel (真实成本)
      - "flat": FlatCostModel (固定百分比)
    """
    model_type = config.get('trading', {}).get('cost_model', 'futu_real')
    
    if model_type == 'futu_real':
        return FutuCostModel(config)
    elif model_type == 'flat':
        return FlatCostModel(config)
    else:
        raise ValueError(f"未知成本模型: {model_type}. 可选: futu_real, flat")
