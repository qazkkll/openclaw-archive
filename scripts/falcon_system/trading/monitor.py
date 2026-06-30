"""
🦅 Falcon 持仓监控
==================
盘中异动检测、止损/止盈提醒、自动调整建议。
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.config import CONFIG, ALERT_DIR, DATA_DIR
from .broker import BrokerInterface, Position


# ════════════════════════════════════════════════════════════════
# 异动分级
# ════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    """异动告警"""
    level: str          # L1 / L2 / L3
    alert_type: str     # price_move / stop_loss / volume_spike / target_hit
    symbol: str
    message: str
    current_price: float
    reference_price: float
    change_pct: float
    timestamp: str
    action_required: Optional[str] = None  # 建议操作


class AlertClassifier:
    """异动分级器"""
    
    def __init__(self):
        self.config = CONFIG.monitor
        self.dedup_cache = {}  # {symbol+type: expire_ts}
    
    def classify(self, position: Position, 
                 prev_close: Optional[float] = None,
                 volume_ratio: Optional[float] = None) -> Optional[Alert]:
        """判断是否有异动"""
        symbol = position.symbol
        current_price = position.current_price
        entry_price = position.avg_entry_price
        pnl_pct = position.unrealized_plpc
        
        # 止损检查 (L3)
        if pnl_pct <= CONFIG.model.stop_loss:
            if self._should_alert(symbol, "stop_loss"):
                return Alert(
                    level="L3",
                    alert_type="stop_loss",
                    symbol=symbol,
                    message=f"🛑 {symbol} 触发止损线! 亏损{pnl_pct*100:.1f}%",
                    current_price=current_price,
                    reference_price=entry_price,
                    change_pct=pnl_pct,
                    timestamp=datetime.now().isoformat(),
                    action_required="立即止损",
                )
        
        # 持仓预警 (L2)
        if pnl_pct <= CONFIG.trading.pnl_warn_threshold:
            if self._should_alert(symbol, "pnl_warn"):
                return Alert(
                    level="L2",
                    alert_type="pnl_warn",
                    symbol=symbol,
                    message=f"⚠️ {symbol} 亏损{pnl_pct*100:.1f}%，接近止损线",
                    current_price=current_price,
                    reference_price=entry_price,
                    change_pct=pnl_pct,
                    timestamp=datetime.now().isoformat(),
                    action_required="关注",
                )
        
        # 价格大幅波动 (L1/L2)
        if prev_close:
            daily_change = (current_price - prev_close) / prev_close
            
            if abs(daily_change) >= self.config.l2_threshold:
                if self._should_alert(symbol, "price_move_l2"):
                    direction = "↑" if daily_change > 0 else "↓"
                    return Alert(
                        level="L2",
                        alert_type="price_move",
                        symbol=symbol,
                        message=f"📊 {symbol} {direction}{abs(daily_change)*100:.1f}%",
                        current_price=current_price,
                        reference_price=prev_close,
                        change_pct=daily_change,
                        timestamp=datetime.now().isoformat(),
                    )
            
            if abs(daily_change) >= self.config.l1_threshold:
                if self._should_alert(symbol, "price_move_l1"):
                    direction = "↑" if daily_change > 0 else "↓"
                    return Alert(
                        level="L1",
                        alert_type="price_move",
                        symbol=symbol,
                        message=f"📈 {symbol} {direction}{abs(daily_change)*100:.1f}%",
                        current_price=current_price,
                        reference_price=prev_close,
                        change_pct=daily_change,
                        timestamp=datetime.now().isoformat(),
                    )
        
        # 成交量异动 (L1)
        if volume_ratio and volume_ratio >= self.config.volume_spike_ratio:
            if self._should_alert(symbol, "volume_spike"):
                return Alert(
                    level="L1",
                    alert_type="volume_spike",
                    symbol=symbol,
                    message=f"📊 {symbol} 成交量{volume_ratio:.1f}倍",
                    current_price=current_price,
                    reference_price=current_price,
                    change_pct=0,
                    timestamp=datetime.now().isoformat(),
                )
        
        # 目标价到达 (L1)
        local_data = self._load_local_positions()
        pos_info = local_data.get(symbol, {})
        target_sell = pos_info.get("target_sell")
        if target_sell and current_price >= target_sell:
            if self._should_alert(symbol, "target_hit"):
                return Alert(
                    level="L1",
                    alert_type="target_hit",
                    symbol=symbol,
                    message=f"🎯 {symbol} 到达目标价${target_sell:.2f}",
                    current_price=current_price,
                    reference_price=target_sell,
                    change_pct=(current_price - entry_price) / entry_price,
                    timestamp=datetime.now().isoformat(),
                    action_required="考虑止盈",
                )
        
        return None
    
    def _should_alert(self, symbol: str, alert_type: str) -> bool:
        """去重检查"""
        key = f"{symbol}:{alert_type}"
        now = time.time()
        
        if key in self.dedup_cache:
            if now < self.dedup_cache[key]:
                return False
        
        self.dedup_cache[key] = now + CONFIG.monitor.dedup_window_seconds
        return True
    
    def _load_local_positions(self) -> Dict:
        """加载本地持仓记录"""
        pos_file = DATA_DIR / "trades" / "positions.json"
        if pos_file.exists():
            try:
                with open(pos_file) as f:
                    return json.load(f).get("positions", {})
            except:
                pass
        return {}


# ════════════════════════════════════════════════════════════════
# 持仓监控器
# ════════════════════════════════════════════════════════════════

class PositionMonitor:
    """持仓监控器"""
    
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.classifier = AlertClassifier()
        self.alerts_file = ALERT_DIR / "pending.json"
        self.state_file = DATA_DIR / "monitor_state.json"
    
    def check_all(self) -> List[Alert]:
        """检查所有持仓"""
        positions = self.broker.get_positions()
        alerts = []
        
        for pos in positions:
            alert = self.classifier.classify(pos)
            if alert:
                alerts.append(alert)
        
        # 保存告警
        if alerts:
            self._save_alerts(alerts)
        
        return alerts
    
    def _save_alerts(self, alerts: List[Alert]):
        """保存告警到文件"""
        existing = []
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file) as f:
                    existing = json.load(f)
            except:
                pass
        
        for alert in alerts:
            existing.append({
                "level": alert.level,
                "type": alert.alert_type,
                "symbol": alert.symbol,
                "message": alert.message,
                "current_price": alert.current_price,
                "reference_price": alert.reference_price,
                "change_pct": alert.change_pct,
                "timestamp": alert.timestamp,
                "action_required": alert.action_required,
            })
        
        with open(self.alerts_file, "w") as f:
            json.dump(existing, f, indent=2)
    
    def get_state(self) -> Dict:
        """获取监控状态"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except:
                pass
        return {"last_check": None, "alerts_count": 0}
    
    def save_state(self, state: Dict):
        """保存监控状态"""
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)


# ════════════════════════════════════════════════════════════════
# 便捷函数
# ════════════════════════════════════════════════════════════════

def run_monitor_check() -> Tuple[List[Alert], str]:
    """运行一次监控检查"""
    from .broker import get_broker
    
    broker = get_broker()
    monitor = PositionMonitor(broker)
    
    alerts = monitor.check_all()
    
    # 更新状态
    monitor.save_state({
        "last_check": datetime.now().isoformat(),
        "alerts_count": len(alerts),
        "position_count": len(broker.get_positions()),
    })
    
    # 生成报告
    if not alerts:
        return alerts, "✅ 无异动"
    
    lines = [f"🦅 **Falcon 异动检查** — {datetime.now().strftime('%H:%M')}"]
    lines.append("")
    
    for alert in alerts:
        emoji = {"L1": "📈", "L2": "⚠️", "L3": "🛑"}.get(alert.level, "❓")
        lines.append(f"{emoji} {alert.message}")
        if alert.action_required:
            lines.append(f"   建议: {alert.action_required}")
    
    return alerts, "\n".join(lines)
