#!/usr/bin/env python3
"""
Alpaca Paper Trading — 模型信号自动执行
===========================================
读取评分系统输出的🟢🟢信号，通过Alpaca Paper Trading模拟下单。
验证可行后，切换到Futu OpenD实盘。

用法:
    python3 scripts/falcons/alpaca_trade.py status      # 查看账户+持仓
    python3 scripts/falcons/alpaca_trade.py signals     # 查看今日信号
    python3 scripts/falcons/alpaca_trade.py execute     # 执行信号（买🟢🟢）
    python3 scripts/falcons/alpaca_trade.py rebalance   # 轮换持仓（卖出到期/不在信号的）
    python3 scripts/falcons/alpaca_trade.py full        # 执行+轮换一步到位
    python3 scripts/falcons/alpaca_trade.py history     # 交易历史

配置从 config/central_config.json 读取，不硬编码。
"""

import sys
import os
import json
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# 加载 .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "central_config.json"
DATA_DIR = PROJECT_ROOT / "data" / "us"
STATE_DIR = PROJECT_ROOT / "output" / "state"
TRADE_LOG = STATE_DIR / "alpaca_trades.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_alpaca_client():
    """创建Alpaca Trading Client。凭据从环境变量或.env读取。"""
    from alpaca.trading.client import TradingClient

    # 尝试从环境变量读取
    api_key = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")

    # 如果环境变量没有，尝试从 .env 文件读取
    if not api_key or not secret_key:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k == "APCA_API_KEY_ID":
                    api_key = v
                elif k == "APCA_API_SECRET_KEY":
                    secret_key = v

    if not api_key or not secret_key:
        print("❌ 缺少Alpaca API凭据。设置方法：")
        print("   export APCA_API_KEY_ID=xxx")
        print("   export APCA_API_SECRET_KEY=xxx")
        print("   或在 .env 文件中添加")
        sys.exit(1)

    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


def load_latest_signals(model_name="falcon_v044"):
    """加载最新评分结果。支持 falcon/arrow/blueshield 三种模型。"""
    search_dirs = []
    
    # Falcon 信号在 data/falcon/ 目录
    if "falcon" in model_name:
        search_dirs.append(str(PROJECT_ROOT / "data" / "falcon"))
    
    # V10/V12 信号在 data/us/ 目录
    search_dirs.append(DATA_DIR)
    
    for d in search_dirs:
        pattern = str(Path(d) / f"{model_name}_scored_*.json")
        files = sorted(glob.glob(pattern))
        if files:
            latest_file = files[-1]
            with open(latest_file) as f:
                data = json.load(f)
            return latest_file, data.get("picks", [])
    
    # 回退: 尝试所有模型
    for fallback in ["falcon_v044", "arrow_v12", "blueshield_v10"]:
        if fallback == model_name:
            continue
        for d in search_dirs:
            pattern = str(Path(d) / f"{fallback}_scored_*.json")
            files = sorted(glob.glob(pattern))
            if files:
                latest_file = files[-1]
                with open(latest_file) as f:
                    data = json.load(f)
                return latest_file, data.get("picks", [])
    
    return None, []


def load_trade_log():
    """加载交易日志。"""
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            return json.load(f)
    return {"trades": [], "positions_opened": {}}


def save_trade_log(log):
    with open(TRADE_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


# ── 命令: status ──
def cmd_status(client):
    """查看账户状态和持仓。"""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    account = client.get_account()
    positions = client.get_all_positions()

    print("=" * 60)
    print("📊 Alpaca Paper Trading 账户状态")
    print("=" * 60)
    print(f"  状态:     {account.status}")
    print(f"  现金:     ${float(account.cash):,.2f}")
    print(f"  购买力:   ${float(account.buying_power):,.2f}")
    print(f"  总权益:   ${float(account.equity):,.2f}")
    print(f"  持仓数:   {len(positions)}")

    if positions:
        print(f"\n{'='*60}")
        print("📦 当前持仓")
        print(f"{'='*60}")
        print(f"{'代码':<8} {'数量':>6} {'成本':>10} {'现价':>10} {'盈亏%':>8} {'市值':>12}")
        print("-" * 60)
        for pos in positions:
            pnl_pct = float(pos.unrealized_plpc) * 100
            print(f"{pos.symbol:<8} {pos.qty:>6} ${float(pos.avg_entry_price):>8.2f} "
                  f"${float(pos.current_price):>8.2f} {pnl_pct:>+7.1f}% ${float(pos.market_value):>10,.2f}")

    # 未成交订单
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=20))
    if orders:
        print(f"\n{'='*60}")
        print("⏳ 未成交订单")
        print(f"{'='*60}")
        for o in orders:
            print(f"  {o.symbol} {o.side} {o.qty} @ {o.limit_price or 'market'} — {o.status}")


# ── 命令: signals ──
def cmd_signals(model_name="falcon_v044"):
    """查看今日信号。"""
    cfg = load_config()
    thresholds = cfg.get("signal_thresholds", {})

    for mn in [model_name, "arrow_v12", "blueshield_v10"]:
        filename, picks = load_latest_signals(mn)
        if not picks:
            print(f"⚠️ {mn}: 无信号文件")
            continue

        print(f"\n{'='*60}")
        print(f"📡 {mn} 信号 (来源: {filename})")
        print(f"{'='*60}")
        print(f"{'代码':<8} {'分数':>8} {'排名%':>8} {'信号':<15} {'价格':>10}")
        print("-" * 55)
        for p in picks[:20]:
            signal = p.get("signal", "⚪")
            print(f"{p['sym']:<8} {p['score']:>8.4f} {p.get('rank_pct', 0)*100:>7.1f}% "
                  f"{signal:<15} ${p.get('close', 0):>8.2f}")

        # 统计
        green2 = sum(1 for p in picks if "🟢🟢" in p.get("signal", ""))
        green1 = sum(1 for p in picks if p.get("signal", "").count("🟢") == 1)
        yellow = sum(1 for p in picks if "🟡" in p.get("signal", ""))
        print(f"\n  🟢🟢: {green2} | 🟢: {green1} | 🟡: {yellow}")


# ── 命令: execute ──
def cmd_execute(client, model_name="falcon_v044", dry_run=False):
    """执行信号：买入🟢🟢信号的股票。"""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    cfg = load_config()
    model_cfg = cfg["models"].get(model_name, {})
    top_n = model_cfg.get("top_n", 5)
    thresholds = cfg.get("signal_thresholds", {})
    buy_threshold = thresholds.get("buy_abs", 0.5)  # 🟢🟢阈值

    filename, picks = load_latest_signals(model_name)
    if not picks:
        print("⚠️ 无信号文件，跳过")
        return []

    # 只买🟢🟢信号
    buy_candidates = [p for p in picks if p.get("score", 0) >= buy_threshold]
    buy_candidates = buy_candidates[:top_n]  # 限制数量

    if not buy_candidates:
        print("ℹ️ 无🟢🟢信号，不买入")
        return []

    # 检查已有持仓，不重复买
    existing = {p.symbol for p in client.get_all_positions()}
    buy_candidates = [p for p in buy_candidates if p["sym"] not in existing]

    if not buy_candidates:
        print("ℹ️ 信号股票已全部持有，跳过")
        return []

    account = client.get_account()
    cash = float(account.cash)
    per_stock = cash / min(len(buy_candidates), top_n) * 0.95  # 留5%现金

    orders = []
    print(f"\n{'='*60}")
    action = "🔍 DRY RUN " if dry_run else "🚀 "
    print(f"{action}买入 {model_name} 🟢🟢信号")
    print(f"{'='*60}")
    print(f"  可用现金: ${cash:,.2f} | 每只: ${per_stock:,.2f} | 上限: {top_n}只")

    for p in buy_candidates:
        sym = p["sym"]
        price = p.get("close", 0)
        if price <= 0:
            print(f"  ⚠️ {sym}: 价格异常({price})，跳过")
            continue

        qty = int(per_stock / price)
        if qty <= 0:
            print(f"  ⚠️ {sym}: 价格${price:.2f}太贵，买不起1股")
            continue

        if dry_run:
            print(f"  📝 {sym}: {qty}股 × ${price:.2f} = ${qty*price:,.2f} (模拟)")
        else:
            try:
                order = MarketOrderRequest(
                    symbol=sym,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                submitted = client.submit_order(order_data=order)
                print(f"  ✅ {sym}: {qty}股 × ${price:.2f} — 订单 {submitted.id}")
                orders.append({
                    "symbol": sym,
                    "qty": qty,
                    "price": price,
                    "order_id": str(submitted.id),
                    "side": "BUY",
                    "timestamp": datetime.now().isoformat(),
                    "model": model_name,
                    "score": p.get("score", 0),
                    "signal": p.get("signal", "")
                })
            except Exception as e:
                print(f"  ❌ {sym}: {e}")

    # 记录交易
    if orders:
        log = load_trade_log()
        log["trades"].extend(orders)
        save_trade_log(log)

    return orders


# ── 命令: rebalance ──
def cmd_rebalance(client, hold_days=10, stop_loss_pct=-15.0, dry_run=False, keep_tickers=None):
    """轮换持仓：卖出到期或止损的股票。keep_tickers中的股票不卖。"""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    positions = client.get_all_positions()
    if not positions:
        print("ℹ️ 无持仓，跳过轮换")
        return []

    log = load_trade_log()
    opened = log.get("positions_opened", {})

    sells = []
    today = datetime.now().date()

    print(f"\n{'='*60}")
    action = "🔍 DRY RUN " if dry_run else "🔄 "
    print(f"{action}轮换持仓 (持有>{hold_days}天或亏损>{stop_loss_pct}%)")
    print(f"{'='*60}")

    if keep_tickers:
        print(f"  📌 保留(仍在🟢🟢): {', '.join(keep_tickers)}")

    for pos in positions:
        sym = pos.symbol
        pnl_pct = float(pos.unrealized_plpc) * 100
        qty = int(pos.qty)

        # 保留仍在🟢🟢列表中的持仓
        if keep_tickers and sym in keep_tickers:
            print(f"  ⏳ {sym}: P&L {pnl_pct:+.1f}%, 仍在🟢🟢 — 继续持有")
            continue

        # 计算持有天数
        entry_date_str = opened.get(sym, {}).get("date")
        if entry_date_str:
            entry_date = datetime.fromisoformat(entry_date_str).date()
            days_held = (today - entry_date).days
        else:
            days_held = 999  # 未知持仓日期，保守处理

        should_sell = False
        reason = ""

        if pnl_pct <= stop_loss_pct:
            should_sell = True
            reason = f"止损 ({pnl_pct:+.1f}%)"
        elif days_held >= hold_days:
            should_sell = True
            reason = f"到期 ({days_held}天)"

        if should_sell:
            if dry_run:
                print(f"  📝 {sym}: {qty}股, P&L {pnl_pct:+.1f}%, 持有{days_held}天 — {reason}")
            else:
                try:
                    order = MarketOrderRequest(
                        symbol=sym,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    submitted = client.submit_order(order_data=order)
                    print(f"  ✅ {sym}: 卖出{qty}股, P&L {pnl_pct:+.1f}%, {reason} — 订单 {submitted.id}")
                    sells.append({
                        "symbol": sym,
                        "qty": qty,
                        "pnl_pct": pnl_pct,
                        "days_held": days_held,
                        "reason": reason,
                        "order_id": str(submitted.id),
                        "side": "SELL",
                        "timestamp": datetime.now().isoformat()
                    })
                    # 从opened中移除
                    if sym in opened:
                        del opened[sym]
                except Exception as e:
                    print(f"  ❌ {sym}: {e}")
        else:
            print(f"  ⏳ {sym}: P&L {pnl_pct:+.1f}%, 持有{days_held}天 — 继续持有")

    if sells:
        log["trades"].extend(sells)
        log["positions_opened"] = opened
        save_trade_log(log)

    return sells


# ── 命令: full ──
def cmd_full(client, model_name="falcon_v044", hold_days=30, stop_loss_pct=-15.0, dry_run=False):
    """完整流程：先轮换卖出，再买入新信号。保留仍在🟢🟢的持仓。"""
    print("🔄 完整交易流程")
    print(f"   模型: {model_name} | 持有: {hold_days}天 | 止损: {stop_loss_pct}%")
    print(f"   {'DRY RUN模式' if dry_run else '⚡ 实盘模式'}")

    # 获取当前🟢🟢信号列表
    filename, picks = load_latest_signals(model_name)
    green2_tickers = [p.get("sym", p.get("ticker", p.get("symbol", ""))) for p in picks if "🟢🟢" in p.get("signal", "")]
    print(f"   🟢🟢信号: {', '.join(green2_tickers[:10])}")

    # Step 1: 轮换(保留仍在🟢🟢的持仓)
    sells = cmd_rebalance(client, hold_days, stop_loss_pct, dry_run, keep_tickers=green2_tickers)

    # Step 2: 执行新信号
    buys = cmd_execute(client, model_name, dry_run)

    # Step 3: 更新持仓开启日期
    if buys and not dry_run:
        log = load_trade_log()
        for b in buys:
            log["positions_opened"][b["symbol"]] = {
                "date": datetime.now().isoformat(),
                "price": b["price"],
                "qty": b["qty"],
                "model": b["model"]
            }
        save_trade_log(log)

    print(f"\n{'='*60}")
    print(f"📊 本轮: 卖出 {len(sells)} 只 | 买入 {len(buys)} 只")
    print(f"{'='*60}")


# ── 命令: history ──
def cmd_history():
    """查看交易历史。"""
    log = load_trade_log()
    trades = log.get("trades", [])

    if not trades:
        print("ℹ️ 暂无交易记录")
        return

    print(f"{'='*60}")
    print(f"📜 交易历史 ({len(trades)}笔)")
    print(f"{'='*60}")
    print(f"{'时间':<20} {'代码':<8} {'方向':<5} {'数量':>6} {'价格':>10} {'盈亏%':>8}")
    print("-" * 60)

    for t in trades[-30:]:  # 最近30笔
        ts = t.get("timestamp", "")[:19]
        pnl = f"{t.get('pnl_pct', 0):+.1f}%" if "pnl_pct" in t else "-"
        price = f"${t.get('price', 0):.2f}" if t.get("price") else "-"
        print(f"{ts:<20} {t.get('symbol', '?'):<8} {t.get('side', '?'):<5} "
              f"{t.get('qty', 0):>6} {price:>10} {pnl:>8}")


# ── 主入口 ──
def main():
    parser = argparse.ArgumentParser(description="Alpaca Paper Trading")
    parser.add_argument("command", choices=["status", "signals", "execute", "rebalance", "full", "history"],
                        help="执行命令")
    parser.add_argument("--model", default="falcon_v044", help="模型名 (falcon_v044 / arrow_v12 / blueshield_v10)")
    parser.add_argument("--hold-days", type=int, default=10, help="持有天数")
    parser.add_argument("--stop-loss", type=float, default=-15.0, help="止损百分比")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际下单")

    args = parser.parse_args()

    if args.command == "signals":
        cmd_signals(args.model)
        return

    if args.command == "history":
        cmd_history()
        return

    # 需要连接Alpaca的命令
    client = get_alpaca_client()

    if args.command == "status":
        cmd_status(client)
    elif args.command == "execute":
        cmd_execute(client, args.model, args.dry_run)
    elif args.command == "rebalance":
        cmd_rebalance(client, args.hold_days, args.stop_loss, args.dry_run)
    elif args.command == "full":
        cmd_full(client, args.model, args.hold_days, args.stop_loss, args.dry_run)


if __name__ == "__main__":
    main()
