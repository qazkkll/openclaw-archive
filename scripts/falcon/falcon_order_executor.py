#!/usr/bin/env python3
"""
🦅 Falcon 订单执行器
====================
读取盘前计划(premarket_plan.json)，执行限价单买入。
监控订单状态，处理部分成交/超时取消。

用法:
    python3 falcon_order_executor.py              # 正常执行
    python3 falcon_order_executor.py --dry-run    # 模拟运行
    python3 falcon_order_executor.py --cancel-all # 取消所有挂单
"""

import json, os, sys, argparse, time
from datetime import datetime, timedelta
from pathlib import Path

# ── 路径 ──
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
TRADE_DIR = DATA_DIR / "trades"
PLAN_FILE = TRADE_DIR / "premarket_plan.json"
POSITIONS_FILE = TRADE_DIR / "positions.json"
JOURNAL_FILE = TRADE_DIR / "trade_journal.jsonl"
EXEC_STATE_FILE = TRADE_DIR / "executor_state.json"

TRADE_DIR.mkdir(parents=True, exist_ok=True)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── 配置 ──
ORDER_TIMEOUT_MINUTES = 120  # 限价单超时时间(分钟)
MAX_SLIPPAGE_PCT = 0.02      # 最大滑点容忍(2%)
MIN_FILL_PCT = 0.80          # 最小成交比例(80%)

# ── 时区 ──
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
except Exception:
    import pytz
    ET = pytz.timezone("US/Eastern")


def get_alpaca_client():
    """创建Alpaca Trading Client。"""
    from alpaca.trading.client import TradingClient
    api_key = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        print("❌ 缺少Alpaca API凭据")
        sys.exit(1)
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


def load_plan():
    """加载盘前计划。"""
    if not PLAN_FILE.exists():
        return None
    with open(PLAN_FILE) as f:
        return json.load(f)


def load_executor_state():
    """加载执行器状态。"""
    if EXEC_STATE_FILE.exists():
        with open(EXEC_STATE_FILE) as f:
            return json.load(f)
    return {"orders": {}, "last_check": None}


def save_executor_state(state):
    """保存执行器状态。"""
    with open(EXEC_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_journal(entry):
    """追加交易日志。"""
    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def get_current_price(client, symbol):
    """获取当前价格。"""
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.historical import StockHistoricalDataClient
        
        data_client = StockHistoricalDataClient(
            os.environ.get("APCA_API_KEY_ID"),
            os.environ.get("APCA_API_SECRET_KEY")
        )
        request = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        quotes = data_client.get_stock_latest_quote(request)
        quote = quotes[symbol]
        return float(quote.ask_price), float(quote.bid_price)
    except Exception as e:
        print(f"  ⚠️ 获取{symbol}价格失败: {e}")
        return None, None


def place_limit_order(client, symbol, qty, limit_price, dry_run=False):
    """下限价单。"""
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    
    if dry_run:
        return {
            "order_id": "dry-run",
            "symbol": symbol,
            "qty": qty,
            "limit_price": limit_price,
            "status": "dry_run",
        }
    
    try:
        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            limit_price=round(limit_price, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        submitted = client.submit_order(order_data=order)
        return {
            "order_id": str(submitted.id),
            "symbol": symbol,
            "qty": qty,
            "limit_price": limit_price,
            "status": "submitted",
            "submitted_at": datetime.now(ET).isoformat(),
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "qty": qty,
            "limit_price": limit_price,
            "status": "error",
            "error": str(e),
        }


def check_order_status(client, order_id):
    """检查订单状态。"""
    try:
        order = client.get_order_by_id(order_id)
        return {
            "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
            "filled_qty": int(order.filled_qty) if order.filled_qty else 0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def cancel_order(client, order_id):
    """取消订单。"""
    try:
        client.cancel_order_by_id(order_id)
        return True
    except Exception as e:
        print(f"  ⚠️ 取消订单失败: {e}")
        return False


def execute_plan(client, plan, dry_run=False):
    """执行盘前计划。"""
    now_et = datetime.now(ET)
    state = load_executor_state()
    
    buys = plan.get("buys", [])
    if not buys:
        print("ℹ️ 无买入计划")
        return {"buys": [], "errors": []}
    
    results = {"buys": [], "errors": []}
    
    for item in buys:
        symbol = item["symbol"]
        target_price = item["target_buy"]
        qty = item["qty"]
        stop_loss = item["stop_loss"]
        target_sell = item["target_sell"]
        
        # 检查是否已经下单
        if symbol in state.get("orders", {}):
            existing = state["orders"][symbol]
            if existing.get("status") in ["submitted", "partial_fill"]:
                # 检查订单状态
                order_status = check_order_status(client, existing["order_id"])
                if order_status["status"] == "filled":
                    # 成交
                    fill_price = order_status.get("filled_avg_price", target_price)
                    entry_record = {
                        "symbol": symbol,
                        "qty": order_status.get("filled_qty", qty),
                        "side": "BUY",
                        "price": fill_price,
                        "target_buy": target_price,
                        "stop_loss": stop_loss,
                        "target_sell": target_sell,
                        "score": item.get("score", 0),
                        "reason": f"盘前计划执行 | 目标${target_price:.2f} | 成交${fill_price:.2f}",
                        "order_id": existing["order_id"],
                        "timestamp": now_et.isoformat(),
                        "model": "falcon_v032",
                    }
                    results["buys"].append(entry_record)
                    
                    # 更新持仓记录
                    update_positions(symbol, entry_record)
                    
                    # 记录日志
                    append_journal(entry_record)
                    
                    # 从state中移除
                    del state["orders"][symbol]
                    print(f"  ✅ {symbol} 成交: {order_status.get('filled_qty',qty)}股 @ ${fill_price:.2f}")
                    continue
                elif order_status["status"] in ["canceled", "expired"]:
                    # 已取消/过期
                    del state["orders"][symbol]
                    print(f"  ⏰ {symbol} 订单已{order_status['status']}")
                elif order_status["status"] == "partial_fill":
                    # 部分成交，继续等待
                    filled = order_status.get("filled_qty", 0)
                    print(f"  ⏳ {symbol} 部分成交: {filled}/{qty}股")
                    continue
                else:
                    # 还在等待
                    continue
        
        # 获取当前价格
        ask_price, bid_price = get_current_price(client, symbol)
        if ask_price is None:
            # 回退到计划中的价格
            ask_price = item.get("current_price", target_price)
            bid_price = ask_price * 0.999
        
        # 决定限价: 用target_buy和当前ask的较低者
        # 但如果当前价远高于目标价，可能需要调整
        if ask_price <= target_price:
            # 当前价已经低于目标价，直接用ask价下单
            limit_price = ask_price
        else:
            # 当前价高于目标价，用目标价下单(等待回调)
            limit_price = target_price
        
        # 检查滑点容忍
        if limit_price > target_price * (1 + MAX_SLIPPAGE_PCT):
            print(f"  ⚠️ {symbol} 当前价${ask_price:.2f} 远超目标${target_price:.2f}，跳过")
            continue
        
        print(f"  📝 {symbol}: 下限价单 {qty}股 @ ${limit_price:.2f} (目标${target_price:.2f})")
        
        order_result = place_limit_order(client, symbol, qty, limit_price, dry_run)
        
        if order_result["status"] == "error":
            results["errors"].append(f"{symbol}: {order_result.get('error', '未知错误')}")
            print(f"  ❌ {symbol} 下单失败: {order_result.get('error')}")
        else:
            state.setdefault("orders", {})[symbol] = order_result
            print(f"  ✅ {symbol} 订单已提交: {order_result.get('order_id', 'N/A')}")
    
    state["last_check"] = now_et.isoformat()
    save_executor_state(state)
    
    return results


def update_positions(symbol, entry_record):
    """更新持仓记录。"""
    pos_data = {"positions": {}}
    if POSITIONS_FILE.exists():
        try:
            with open(POSITIONS_FILE) as f:
                pos_data = json.load(f)
        except Exception:
            pass
    
    pos_data["positions"][symbol] = {
        "entry_date": entry_record["timestamp"],
        "entry_price": entry_record["price"],
        "qty": entry_record["qty"],
        "score": entry_record.get("score", 0),
        "reason": entry_record.get("reason", ""),
        "stop_loss": entry_record.get("stop_loss"),
        "target_sell": entry_record.get("target_sell"),
    }
    
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos_data, f, indent=2, default=str)


def cancel_all_orders(client):
    """取消所有挂单。"""
    try:
        orders = client.get_orders()
        for order in orders:
            if order.status.value in ["new", "accepted", "pending_new", "partially_filled"]:
                client.cancel_order_by_id(order.id)
                print(f"  ❌ 取消: {order.symbol} {order.qty}股 @ ${order.limit_price}")
        print("✅ 所有挂单已取消")
    except Exception as e:
        print(f"❌ 取消挂单失败: {e}")


def format_report(results, plan):
    """格式化Telegram报告。"""
    lines = []
    now_et = datetime.now(ET)
    lines.append(f"🦅 **Falcon 订单执行报告**")
    lines.append(f"📅 {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    lines.append("")
    
    if results.get("buys"):
        lines.append(f"🟢 **成交 ({len(results['buys'])}只)**")
        for b in results["buys"]:
            lines.append(f"  ✅ **{b['symbol']}** {b['qty']}股 @ ${b['price']:.2f}")
            lines.append(f"     止损: ${b['stop_loss']:.2f} | 目标: ${b['target_sell']:.2f}")
        lines.append("")
    
    if results.get("errors"):
        lines.append(f"⚠️ **异常**")
        for e in results["errors"]:
            lines.append(f"  • {e}")
        lines.append("")
    
    # 显示挂单状态
    state = load_executor_state()
    pending = {k: v for k, v in state.get("orders", {}).items() if v.get("status") in ["submitted", "partial_fill"]}
    if pending:
        lines.append(f"⏳ **挂单中 ({len(pending)}只)**")
        for sym, o in pending.items():
            lines.append(f"  {sym} {o['qty']}股 @ ${o['limit_price']:.2f}")
        lines.append("")
    
    if not results.get("buys") and not results.get("errors") and not pending:
        lines.append("ℹ️ 今日无成交")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Falcon 订单执行器")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行")
    parser.add_argument("--cancel-all", action="store_true", help="取消所有挂单")
    args = parser.parse_args()
    
    # 加载计划
    plan = load_plan()
    if not plan:
        print("❌ 无盘前计划，请先运行 falcon_premarket_plan.py")
        return
    
    # 检查计划日期
    plan_date = plan.get("date", "")
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if plan_date != today:
        print(f"⚠️ 计划日期({plan_date})不是今天({today})，请重新生成计划")
        return
    
    client = get_alpaca_client()
    
    if args.cancel_all:
        cancel_all_orders(client)
        return
    
    print(f"🦅 Falcon 订单执行器 — {datetime.now(ET).strftime('%H:%M ET')}")
    print("=" * 60)
    
    # 执行计划
    results = execute_plan(client, plan, dry_run=args.dry_run)
    
    # 输出报告
    report = format_report(results, plan)
    print("\n" + report)
    
    # 保存报告
    report_file = TRADE_DIR / f"exec_report_{today.replace('-','')}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
