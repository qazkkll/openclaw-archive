"""
🦅 Falcon 交易系统配置 (V0.4.4)
======================
所有配置集中管理，支持环境变量覆盖。
模型层面: 选股因子权重
执行层面: 大盘感知 + 动态仓位管理
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── 路径 ──
FALCON_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = FALCON_ROOT.parent  # openclaw-archive
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
TRADE_DIR = DATA_DIR / "trades"
LOG_DIR = DATA_DIR / "logs"
ALERT_DIR = DATA_DIR / "alerts"

for d in [DATA_DIR, TRADE_DIR, LOG_DIR, ALERT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """模型层面配置 (选股逻辑)"""
    version: str = "V0.4.4"
    top_n: int = 10
    hold_days: int = 30
    stop_loss: float = -0.15
    vix_threshold: float = 25.0
    buy_score_threshold: float = 0.55
    broker_type: str = "alpaca"  # alpaca / futu

    # V0.4.4 因子权重 (Walk-Forward验证通过, RI=68.4%)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "fund_ratio": 0.45,        # 财务比率 (20个价格比率等权平均)
        "growth_composite": 0.20,  # 成长组合 (0.60*fund_growth + 0.25*analyst + 0.15*income)
        "qoq": 0.20,              # 季度环比变化
        "cashflow": 0.15,          # 现金流因子
    })

    # growth_composite 子权重
    growth_composite_weights: Dict[str, float] = field(default_factory=lambda: {
        "fund_growth": 0.60,  # 15个增长指标
        "analyst": 0.25,      # 4个分析师指标
        "income": 0.15,       # 6个收入指标
    })

    # 反向因子 (低值排高)
    invert_factors: set = field(default_factory=lambda: {
        "debt_to_equity", "net_debt_to_assets", "capex_intensity"
    })


@dataclass
class ExecutionConfig:
    """执行层面配置 (仓位管理 + 大盘感知)"""
    # 大盘感知
    market_awareness_enabled: bool = True
    vix_source: str = "data/us/vix_10y.parquet"

    # 市场状态 → 仓位映射
    regime_position_map: Dict[str, float] = field(default_factory=lambda: {
        "bull": 1.00,          # 牛市: 100%仓位
        "neutral": 0.75,       # 震荡: 75%仓位
        "bear": 0.50,          # 熊市: 50%仓位
        "extreme_bear": 0.25,  # 极端熊市: 25%仓位
    })

    # 动态仓位策略
    position_strategy: str = "trend_dynamic"  # fixed / vix_only / trend_only / trend_dynamic


@dataclass
class TradingConfig:
    """交易配置"""
    max_position_pct: float = 0.10
    max_total_exposure: float = 0.80
    min_order_value: float = 500
    order_timeout_minutes: int = 120
    max_slippage_pct: float = 0.02
    pnl_warn_threshold: float = -0.10
    price_move_alert: float = 0.03
    volume_spike_ratio: float = 3.0
    atr_multiplier: float = 1.5
    max_drop_pct: float = 0.05


@dataclass
class DataConfig:
    """数据配置"""
    price_sources: List[str] = field(default_factory=lambda: ["yfinance", "fmp"])
    fundamental_sources: List[str] = field(default_factory=lambda: ["fmp_premium", "fmp"])
    price_max_age_hours: int = 24
    fundamental_max_age_days: int = 90


@dataclass
class MonitorConfig:
    """持仓监控配置"""
    dedup_window_seconds: int = 600  # 异动去重窗口(秒)
    l1_threshold: float = 0.05      # L1价格波动阈值(5%)
    l2_threshold: float = 0.03      # L2价格波动阈值(3%)
    volume_spike_ratio: float = 3.0 # 成交量异动倍数


class FalconConfig:
    """统一配置包装器，提供 .model / .trading / .monitor / .data / .execution 访问"""
    def __init__(self):
        self.model = ModelConfig()
        self.trading = TradingConfig()
        self.monitor = MonitorConfig()
        self.data = DataConfig()
        self.execution = ExecutionConfig()

    def __getattr__(self, name):
        """兼容直接访问 model 级属性 (如 CONFIG.version, CONFIG.weights)"""
        return getattr(self.model, name)


# 全局实例
CONFIG = FalconConfig()
EXEC_CONFIG = CONFIG.execution
TRADE_CONFIG = CONFIG.trading
DATA_CONFIG = CONFIG.data
