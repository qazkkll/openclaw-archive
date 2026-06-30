"""
🦅 Falcon 交易系统配置
======================
所有配置集中管理，支持环境变量覆盖。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── 路径 ──
# falcon_system/core/config.py → 向上4级到 openclaw-archive
FALCON_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = FALCON_ROOT.parent  # openclaw-archive
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
TRADE_DIR = DATA_DIR / "trades"
LOG_DIR = DATA_DIR / "logs"
ALERT_DIR = DATA_DIR / "alerts"

# 确保目录存在
for d in [DATA_DIR, TRADE_DIR, LOG_DIR, ALERT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """模型配置"""
    version: str = "V0.3.2"
    top_n: int = 10
    hold_days: int = 60
    stop_loss: float = -0.15
    vix_threshold: float = 25.0
    buy_score_threshold: float = 0.55
    
    # V0.3.2权重 (已验证, 归一化到1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "fund_growth": 0.1875,
        "cashflow": 0.15,
        "analyst": 0.15,
        "grade_sentiment": 0.15,
        "earnings": 0.125,
        "balance": 0.10,
        "fund_metric": 0.075,
        "insider": 0.0625,
        "fund_ratio": 0.0,
        "income_stmt": 0.0,
        "tech": 0.0,
        "valuation": 0.0,
    })
    
    # 反向因子 (低值排高)
    invert_factors: set = field(default_factory=lambda: {
        "debt_to_equity", "net_debt_to_assets", "capex_intensity"
    })


@dataclass
class TradingConfig:
    """交易配置"""
    # 仓位管理
    max_position_pct: float = 0.10      # 单只最大仓位
    max_total_exposure: float = 0.80    # 最大总仓位
    min_order_value: float = 500        # 最小下单金额
    
    # 目标价位
    atr_period: int = 14
    atr_multiplier: float = 1.5
    support_lookback: int = 20
    max_drop_pct: float = 0.05          # 最大回调容忍
    
    # 订单管理
    order_timeout_minutes: int = 120
    max_slippage_pct: float = 0.02
    min_fill_pct: float = 0.80
    
    # 风险管理
    pnl_warn_threshold: float = -0.10   # 持仓预警线
    price_move_alert: float = 0.03      # 价格异动阈值
    volume_spike_ratio: float = 3.0     # 成交量异动倍数


@dataclass
class DataConfig:
    """数据配置"""
    # 数据源优先级
    price_sources: List[str] = field(default_factory=lambda: ["yfinance", "fmp"])
    fundamental_sources: List[str] = field(default_factory=lambda: ["fmp_premium", "fmp"])
    
    # 新鲜度要求
    price_max_age_hours: int = 24       # 价格数据最大年龄(小时)
    fundamental_max_age_days: int = 90  # 基本面数据最大年龄(天)
    
    # 更新频率
    price_update_cron: str = "0 20 * * 1-5"      # 每天20:00 HKT (收盘后)
    fundamental_update_cron: str = "0 10 * * 6"   # 每周六10:00 HKT


@dataclass
class MonitorConfig:
    """监控配置"""
    # 异动分级
    l1_threshold: float = 0.03          # L1: ±3%
    l2_threshold: float = 0.05          # L2: ±5%
    l3_threshold: float = 0.10          # L3: ±10% (止损级)
    
    # 去重
    dedup_window_seconds: int = 14400   # 4小时
    volume_spike_ratio: float = 3.0     # 成交量异动倍数
    
    # 盘中监控
    poll_interval_seconds: int = 300    # 5分钟
    enhanced_interval_seconds: int = 60 # 加频1分钟


@dataclass
class FalconConfig:
    """Falcon系统总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    
    # Broker配置
    broker_type: str = "alpaca"  # alpaca / futu
    
    # 时区
    timezone: str = "US/Eastern"
    
    @classmethod
    def load(cls) -> "FalconConfig":
        """从环境变量和配置文件加载"""
        config = cls()
        
        # 环境变量覆盖
        if os.environ.get("FALCON_BROKER"):
            config.broker_type = os.environ["FALCON_BROKER"]
        if os.environ.get("FALCON_VIX_THRESHOLD"):
            config.model.vix_threshold = float(os.environ["FALCON_VIX_THRESHOLD"])
        if os.environ.get("FALCON_STOP_LOSS"):
            config.model.stop_loss = float(os.environ["FALCON_STOP_LOSS"])
        
        # 从YAML加载(如果存在)
        try:
            import yaml
            yaml_path = PROJECT_ROOT / "config" / "falcon.yaml"
            if yaml_path.exists():
                with open(yaml_path) as f:
                    cfg = yaml.safe_load(f)
                
                # 模型配置
                model_cfg = cfg.get("model", {})
                config.model.top_n = model_cfg.get("top_n", config.model.top_n)
                config.model.buy_score_threshold = model_cfg.get("buy_score_threshold", config.model.buy_score_threshold)
                
                # 交易配置
                trading_cfg = cfg.get("trading", {})
                config.trading.max_position_pct = trading_cfg.get("max_position_pct", config.trading.max_position_pct)
                config.trading.max_total_exposure = trading_cfg.get("max_total_exposure", config.trading.max_total_exposure)
                
                # 监控配置
                monitor_cfg = cfg.get("monitor", {})
                config.monitor.l1_threshold = monitor_cfg.get("l1_threshold", config.monitor.l1_threshold)
                config.monitor.l2_threshold = monitor_cfg.get("l2_threshold", config.monitor.l2_threshold)
                config.monitor.l3_threshold = monitor_cfg.get("l3_threshold", config.monitor.l3_threshold)
        except Exception:
            pass
        
        return config


# 全局配置实例
CONFIG = FalconConfig.load()
