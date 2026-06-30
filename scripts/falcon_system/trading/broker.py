"""
🦅 Falcon Broker模块
====================
统一交易接口：Alpaca/Futu可替换。
包含：订单执行、持仓管理、独立验证。
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..core.config import CONFIG, TRADE_DIR, DATA_DIR, PROJECT_ROOT


# ════════════════════════════════════════════════════════════════
# Broker接口 (可替换)
# ════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """持仓信息"""
    symbol: str
    qty: int
    avg_entry_price: float
    current_price: float
    unrealized_plpc: float
    market_value: float


@dataclass
class OrderResult:
    """订单结果"""
    order_id: str
    symbol: str
    qty: int
    side: str  # BUY / SELL
    order_type: str  # MARKET / LIMIT
    limit_price: Optional[float]
    status: str  # submitted / filled / canceled / error
    filled_qty: int = 0
    filled_avg_price: Optional[float] = None
    error: Optional[str] = None
    submitted_at: Optional[str] = None


class BrokerInterface(ABC):
    """Broker接口"""
    
    @abstractmethod
    def get_account(self) -> Dict[str, float]:
        """获取账户信息"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Position]:
        """获取所有持仓"""
        pass
    
    @abstractmethod
    def place_limit_order(self, symbol: str, qty: int, 
                          limit_price: float, side: str) -> OrderResult:
        """下限价单"""
        pass
    
    @abstractmethod
    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        """下市价单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    def cancel_all_orders(self) -> bool:
        """取消所有挂单"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """查询订单状态"""
        pass
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格(可选实现)"""
        return None


# ════════════════════════════════════════════════════════════════
# Alpaca实现
# ════════════════════════════════════════════════════════════════

class AlpacaBroker(BrokerInterface):
    """Alpaca Paper Trading"""
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        
        from alpaca.trading.client import TradingClient
        api_key = os.environ.get("APCA_API_KEY_ID")
        secret_key = os.environ.get("APCA_API_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("缺少Alpaca API凭据")
        
        self.client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True
        )
        self.data_client = None
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
        except:
            pass
    
    def get_account(self) -> Dict[str, float]:
        """获取账户信息"""
        account = self.client.get_account()
        return {
            "equity": round(float(account.equity), 2),
            "cash": round(float(account.cash), 2),
            "buying_power": round(float(account.buying_power), 2),
        }
    
    def get_positions(self) -> List[Position]:
        """获取所有持仓"""
        positions = self.client.get_all_positions()
        return [
            Position(
                symbol=p.symbol,
                qty=int(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_plpc=float(p.unrealized_plpc),
                market_value=float(p.market_value),
            )
            for p in positions
        ]
    
    def place_limit_order(self, symbol: str, qty: int, 
                          limit_price: float, side: str) -> OrderResult:
        """下限价单"""
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        try:
            order = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                limit_price=round(limit_price, 2),
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            submitted = self.client.submit_order(order_data=order)
            
            return OrderResult(
                order_id=str(submitted.id),
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="LIMIT",
                limit_price=limit_price,
                status="submitted",
                submitted_at=datetime.now().isoformat(),
            )
        except Exception as e:
            return OrderResult(
                order_id="",
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="LIMIT",
                limit_price=limit_price,
                status="error",
                error=str(e),
            )
    
    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        """下市价单"""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        try:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            submitted = self.client.submit_order(order_data=order)
            
            return OrderResult(
                order_id=str(submitted.id),
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="MARKET",
                limit_price=None,
                status="submitted",
                submitted_at=datetime.now().isoformat(),
            )
        except Exception as e:
            return OrderResult(
                order_id="",
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="MARKET",
                limit_price=None,
                status="error",
                error=str(e),
            )
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        try:
            self.client.cancel_order_by_id(order_id)
            return True
        except:
            return False
    
    def cancel_all_orders(self) -> bool:
        """取消所有挂单"""
        try:
            self.client.cancel_orders()
            return True
        except:
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """查询订单状态"""
        try:
            order = self.client.get_order_by_id(order_id)
            return {
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        if not self.data_client:
            return None
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            quotes = self.data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]
            return float(quote.ask_price)
        except:
            return None


# ════════════════════════════════════════════════════════════════
# Futu实现 (占位)
# ════════════════════════════════════════════════════════════════

class FutuBroker(BrokerInterface):
    """Futu OpenD (占位，待实现)"""
    
    def get_account(self) -> Dict[str, float]:
        raise NotImplementedError("Futu broker待实现")
    
    def get_positions(self) -> List[Position]:
        raise NotImplementedError("Futu broker待实现")
    
    def place_limit_order(self, symbol: str, qty: int, 
                          limit_price: float, side: str) -> OrderResult:
        raise NotImplementedError("Futu broker待实现")
    
    def place_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        raise NotImplementedError("Futu broker待实现")
    
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Futu broker待实现")
    
    def cancel_all_orders(self) -> bool:
        raise NotImplementedError("Futu broker待实现")
    
    def get_order_status(self, order_id: str) -> Dict:
        raise NotImplementedError("Futu broker待实现")


# ════════════════════════════════════════════════════════════════
# Broker工厂
# ════════════════════════════════════════════════════════════════

def get_broker() -> BrokerInterface:
    """获取Broker实例"""
    broker_type = CONFIG.broker_type
    
    if broker_type == "alpaca":
        return AlpacaBroker()
    elif broker_type == "futu":
        return FutuBroker()
    else:
        raise ValueError(f"未知broker类型: {broker_type}")


# ════════════════════════════════════════════════════════════════
# 持仓管理器
# ════════════════════════════════════════════════════════════════

POSITIONS_FILE = TRADE_DIR / "positions.json"
JOURNAL_FILE = TRADE_DIR / "trade_journal.jsonl"


class PositionManager:
    """持仓管理器"""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
    
    def load_positions(self) -> Dict:
        """加载本地持仓记录"""
        if POSITIONS_FILE.exists():
            try:
                with open(POSITIONS_FILE) as f:
                    return json.load(f)
            except:
                pass
        return {"positions": {}}
    
    def save_positions(self, data: Dict):
        """保存持仓记录"""
        with open(POSITIONS_FILE, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def sync_from_broker(self):
        """从Broker同步持仓"""
        broker_positions = self.broker.get_positions()
        local_data = self.load_positions()
        
        synced = {"positions": {}, "synced_at": datetime.now().isoformat()}
        
        for bp in broker_positions:
            existing = local_data["positions"].get(bp.symbol, {})
            synced["positions"][bp.symbol] = {
                "qty": bp.qty,
                "avg_entry_price": bp.avg_entry_price,
                "current_price": bp.current_price,
                "unrealized_plpc": bp.unrealized_plpc,
                "market_value": bp.market_value,
                # 保留本地元数据
                "entry_date": existing.get("entry_date", ""),
                "score": existing.get("score", 0),
                "reason": existing.get("reason", ""),
                "target_buy": existing.get("target_buy"),
                "stop_loss": existing.get("stop_loss"),
                "target_sell": existing.get("target_sell"),
            }
        
        self.save_positions(synced)
        return synced
    
    def record_buy(self, symbol: str, qty: int, price: float, 
                   score: float, reason: str, 
                   stop_loss: Optional[float] = None,
                   target_sell: Optional[float] = None):
        """记录买入"""
        data = self.load_positions()
        data["positions"][symbol] = {
            "entry_date": datetime.now().isoformat(),
            "entry_price": price,
            "qty": qty,
            "score": score,
            "reason": reason,
            "stop_loss": stop_loss,
            "target_sell": target_sell,
            "model": CONFIG.model.version,
        }
        self.save_positions(data)
        
        # 写入交易日志
        self._append_journal({
            "type": "BUY",
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "score": score,
            "reason": reason,
            "stop_loss": stop_loss,
            "target_sell": target_sell,
            "timestamp": datetime.now().isoformat(),
            "model": CONFIG.model.version,
        })
    
    def record_sell(self, symbol: str, qty: int, price: float, reason: str):
        """记录卖出"""
        data = self.load_positions()
        if symbol in data["positions"]:
            entry_price = data["positions"][symbol].get("entry_price", price)
            pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0
            
            # 写入交易日志
            self._append_journal({
                "type": "SELL",
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "entry_price": entry_price,
                "pnl_pct": round(pnl_pct, 4),
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "model": CONFIG.model.version,
            })
            
            del data["positions"][symbol]
            self.save_positions(data)
    
    def _append_journal(self, entry: Dict):
        """追加交易日志"""
        with open(JOURNAL_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    
    def get_portfolio_summary(self) -> Dict:
        """获取持仓汇总"""
        broker_positions = self.broker.get_positions()
        account = self.broker.get_account()
        
        total_value = sum(p.market_value for p in broker_positions)
        total_pnl = sum(p.unrealized_plpc * p.market_value for p in broker_positions)
        
        return {
            "account": account,
            "position_count": len(broker_positions),
            "total_market_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "pnl_pct": round(p.unrealized_plpc * 100, 2),
                    "market_value": round(p.market_value, 2),
                }
                for p in broker_positions
            ],
        }
