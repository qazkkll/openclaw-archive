#!/usr/bin/env python3
"""
🦅 Falcon Broker Adapter — 统一持仓/交易接口
=============================================
设计原则:
  - 所有持仓查询走这个接口，不直接调Alpaca/OpenD
  - 当前实现: Alpaca Paper Trading
  - 后续扩展: Futu OpenD (实现同一接口即可切换)

用法:
    from broker_adapter import get_broker
    broker = get_broker()           # 自动选Alpaca
    positions = broker.get_positions()
    account = broker.get_account()
"""

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

# ── 加载 .env ──
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class Position:
    """统一持仓数据结构。"""
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float
    unrealized_pl: float          # 绝对盈亏($)
    unrealized_plpc: float        # 百分比盈亏(0.05 = +5%)
    market_value: float
    side: str = "long"
    entry_date: Optional[str] = None  # 本地补充(从positions.json)

    @property
    def days_held(self) -> int:
        if not self.entry_date:
            return -1
        try:
            dt = datetime.fromisoformat(self.entry_date.replace("Z", "+00:00"))
            return (datetime.now(dt.tzinfo) - dt).days
        except Exception:
            return -1


@dataclass
class Account:
    """统一账户数据结构。"""
    cash: float
    equity: float
    buying_power: float
    position_count: int


class BrokerBase(ABC):
    """Broker抽象基类。新broker实现这些方法即可。"""

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取当前所有持仓。"""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取单只持仓。无则返回None。"""
        ...

    @abstractmethod
    def get_account(self) -> Account:
        """获取账户状态。"""
        ...

    def get_position_symbols(self) -> set:
        """便捷方法：返回持仓symbol集合。"""
        return {p.symbol for p in self.get_positions()}


class AlpacaBroker(BrokerBase):
    """Alpaca Paper Trading 实现。"""

    def __init__(self):
        from alpaca.trading.client import TradingClient
        api_key = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        if not api_key or not secret_key:
            raise RuntimeError("缺少Alpaca API凭据(APCA_API_KEY_ID/APCA_API_SECRET_KEY)")
        self._client = TradingClient(api_key, secret_key, paper=True)

    def get_positions(self) -> List[Position]:
        try:
            raw_positions = self._client.get_all_positions()
            return [
                Position(
                    symbol=p.symbol,
                    qty=int(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                    market_value=float(p.market_value),
                    side=str(getattr(p, "side", "long")),
                )
                for p in raw_positions
            ]
        except Exception as e:
            print(f"⚠️ Alpaca get_positions失败: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Position]:
        try:
            p = self._client.get_open_position(symbol)
            return Position(
                symbol=p.symbol,
                qty=int(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pl=float(p.unrealized_pl),
                unrealized_plpc=float(p.unrealized_plpc),
                market_value=float(p.market_value),
                side=str(getattr(p, "side", "long")),
            )
        except Exception:
            return None

    def get_account(self) -> Account:
        acct = self._client.get_account()
        return Account(
            cash=float(acct.cash),
            equity=float(acct.equity),
            buying_power=float(acct.buying_power),
            position_count=len(self._client.get_all_positions()),
        )


class FutuBroker(BrokerBase):
    """Futu OpenD 实现 (占位, 后续实现)。"""

    def __init__(self):
        raise NotImplementedError("Futu OpenD适配器待实现。需要futu-api SDK + OpenD连接。")

    def get_positions(self):
        raise NotImplementedError

    def get_position(self, symbol):
        raise NotImplementedError

    def get_account(self):
        raise NotImplementedError


# ── 工厂函数 ──
_broker_instance: Optional[BrokerBase] = None

def get_broker(broker_type: str = "alpaca") -> BrokerBase:
    """获取broker单例。重复调用返回同一实例。"""
    global _broker_instance
    if _broker_instance is not None:
        return _broker_instance

    if broker_type == "alpaca":
        _broker_instance = AlpacaBroker()
    elif broker_type == "futu":
        _broker_instance = FutuBroker()
    else:
        raise ValueError(f"未知broker类型: {broker_type}. 支持: alpaca, futu")

    return _broker_instance


def reset_broker():
    """重置单例(测试用)。"""
    global _broker_instance
    _broker_instance = None
