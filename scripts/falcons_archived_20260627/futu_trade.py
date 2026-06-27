#!/usr/bin/env python3
"""
Futu OpenD 实盘交易引擎 — Falcons项目
=======================================
通过Futu OpenD执行美股实盘交易。
与alpaca_trade.py共用相同的信号读取和交易逻辑。

用法:
    python3 scripts/falcons/futu_trade.py status     # 查看账户+持仓
    python3 scripts/falcons/futu_trade.py signals    # 查看今日信号
    python3 scripts/falcons/futu_trade.py execute    # 执行信号
    python3 scripts/falcons/futu_trade.py rebalance  # 轮换持仓
    python3 scripts/falcons/futu_trade.py full       # 完整流程

前置条件:
    1. Futu OpenD 已启动 (127.0.0.1:11111)
    2. futu-opend.service 运行中
    3. 交易密码已解锁
"""

import sys
import os
import json
import glob
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "central_config.json"
DATA_DIR = PROJECT_ROOT / "data" / "us"
STATE_DIR = PROJECT_ROOT / "output" / "state"
TRADE_LOG = STATE_DIR / "futu_trades.json"

STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_futu_client():
    """连接Futu OpenD。"""
    from futu import OpenSecTradeContext, OpenQuoteContext, TrdEnv, TrdMarket, SecurityFirm

    # 检查OpenD是否运行
    import subprocess
    result = subprocess.run(
        ["systemctl", "is-active", "futu-opend.service"],
        capture_output=True, text=True
    )
    if result.stdout.strip() != "active":
        print("❌ Futu OpenD未启动。")
        print("   运行: sudo systemctl start futu-opend.service")
        sys.exit(1)

    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    trade_ctx = OpenSecTradeContext(
        filter_trdmarket=TrdMarket.US,
        host='127.0.0.1', port=11111,
        security_firm=SecurityFirm.FUTUSECURITIES
    )

    return quote_ctx, trade_ctx


def load_latest_signals(model_name="falcon_v031"):
    """加载最新评分结果。支持 falcon/arrow/blueshield 三种模型。"""
    search_dirs = []
    if "falcon" in model_name:
        search_dirs.append(str(PROJECT_ROOT / "data" / "falcon"))
    search_dirs.append(DATA_DIR)
    
    for d in search_dirs:
        pattern = str(Path(d) / f"{model_name}_scored_*.json")
        files = sorted(glob.glob(pattern))
        if files:
            latest_file = files[-1]
            with open(latest_file) as f:
                data = json.load(f)
            return latest_file, data.get("picks", [])
    
    for fallback in ["falcon_v031", "arrow_v12", "blueshield_v10"]:
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
    if TRADE_LOG.exists():
        with open(TRADE_LOG) as f:
            return json.load(f)
    return {"trades": [], "positions_opened": {}}


def save_trade_log(log):
    with open(TRADE_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)


# ── 命令: status ──
def cmd_status(quote_ctx, trade_ctx):
    """查看账户状态和持仓。"""
    from futu import TrdEnv, TrdMarket

    # 账户信息
    ret, data = trade_ctx.accinfo_query(trd_env=TrdEnv.REAL)
    if ret != 0:
        print(f"❌ 获取账户信息失败: {data}")
        return

    print("=" * 60)
    print("📊 Futu OpenD 实盘账户")
    print("=" * 60)
    print(f"  总资产:   ${data['total_assets'].iloc[0]:,.2f}")
    print(f"  现金:     ${data['cash'].iloc[0]:,.2f}")
    print(f"  购买力:   ${data['buying_power'].iloc[0]:,.2f}")
    print(f"  持仓市值: ${data['market_val'].iloc[0]:,.2f}")

    # 持仓
    ret, positions = trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
    if ret != 0:
        print(f"❌ 获取持仓失败: {positions}")
        return

    if not positions.empty:
        print(f"\n{'='*60}")
        print("📦 当前持仓")
        print(f"{'='*60}")
        print(f"{'代码':<8} {'数量':>6} {'成本':>10} {'现价':>10} {'盈亏%':>8} {'市值':>12}")
        print("-" * 60)
        for _, pos in positions.iterrows():
            pnl_pct = (pos['nominal_price'] / pos['cost_price'] - 1) * 100 if pos['cost_price'] > 0 else 0
            print(f"{pos['code']:<8} {pos['qty']:>6} ${pos['cost_price']:>8.2f} "
                  f"${pos['nominal_price']:>8.2f} {pnl_pct:>+7.1f}% ${pos['market_val']:>10,.2f}")

    # 未成交订单
    ret, orders = trade_ctx.order_list_query(trd_env=TrdEnv.REAL)
    if ret == 0 and not orders.empty:
        pending = orders[orders['order_status'].isin(['SUBMITTED', 'WAITING', 'FILLED_PART'])]
        if not pending.empty:
            print(f"\n{'='*60}")
            print("⏳ 未成交订单")
            print(f"{'='*60}")
            for _, o in pending.iterrows():
                print(f"  {o['code']} {o['trd_side']} {o['qty']} @ {o.get('price', 'market')} — {o['order_status']}")


# ── 命令: signals ──
def cmd_signals(model_name="falcon_v031"):
    """查看今日信号。"""
    cfg = load_config()

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

        green2 = sum(1 for p in picks if "🟢🟢" in p.get("signal", ""))
        green1 = sum(1 for p in picks if p.get("signal", "").count("🟢") == 1)
        yellow = sum(1 for p in picks if "🟡" in p.get("signal", ""))
        print(f"\n  🟢🟢: {green2} | 🟢: {green1} | 🟡: {yellow}")


# ── 命令: execute ──
def cmd_execute(quote_ctx, trade_ctx, model_name="arrow_v12", dry_run=False):
    """执行信号：买入🟢🟢信号的股票。"""
    from futu import TrdEnv, TrdMarket, OrderType, TrdSide

    cfg = load_config()
    model_cfg = cfg["models"].get(model_name, {})
    top_n = model_cfg.get("top_n", 5)
    thresholds = cfg.get("signal_thresholds", {})
    buy_threshold = thresholds.get("buy_abs", 0.5)

    filename, picks = load_latest_signals(model_name)
    if not picks:
        print("⚠️ 无信号文件，跳过")
        return []

    buy_candidates = [p for p in picks if p.get("score", 0) >= buy_threshold]
    buy_candidates = buy_candidates[:top_n]

    if not buy_candidates:
        print("ℹ️ 无🟢🟢信号，不买入")
        return []

    # 检查已有持仓
    ret, positions = trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
    existing = set()
    if ret == 0 and not positions.empty:
        existing = set(positions['code'].str.split('.').str[-1].tolist())

    buy_candidates = [p for p in buy_candidates if p["sym"] not in existing]

    if not buy_candidates:
        print("ℹ️ 信号股票已全部持有，跳过")
        return []

    # 获取账户现金
    ret, acc_info = trade_ctx.accinfo_query(trd_env=TrdEnv.REAL)
    cash = acc_info['cash'].iloc[0] if ret == 0 else 0
    per_stock = cash / min(len(buy_candidates), top_n) * 0.95

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

        futu_code = f"US.{sym}"

        if dry_run:
            print(f"  📝 {futu_code}: {qty}股 × ${price:.2f} = ${qty*price:,.2f} (模拟)")
        else:
            try:
                ret, data = trade_ctx.place_order(
                    price=price,
                    qty=qty,
                    code=futu_code,
                    trd_side=TrdSide.BUY,
                    order_type=OrderType.MARKET,
                    trd_env=TrdEnv.REAL
                )
                if ret == 0:
                    order_id = data['order_id'].iloc[0]
                    print(f"  ✅ {futu_code}: {qty}股 × ${price:.2f} — 订单 {order_id}")
                    orders.append({
                        "symbol": sym,
                        "qty": qty,
                        "price": price,
                        "order_id": str(order_id),
                        "side": "BUY",
                        "timestamp": datetime.now().isoformat(),
                        "model": model_name,
                        "score": p.get("score", 0),
                        "signal": p.get("signal", "")
                    })
                else:
                    print(f"  ❌ {futu_code}: {data}")
            except Exception as e:
                print(f"  ❌ {futu_code}: {e}")

    if orders:
        log = load_trade_log()
        log["trades"].extend(orders)
        save_trade_log(log)

    return orders


# ── 命令: rebalance ──
def cmd_rebalance(quote_ctx, trade_ctx, hold_days=10, stop_loss_pct=-15.0, dry_run=False):
    """轮换持仓：卖出到期或止损的股票。"""
    from futu import TrdEnv, TrdMarket, OrderType, TrdSide

    ret, positions = trade_ctx.position_list_query(trd_env=TrdEnv.REAL)
    if ret != 0 or positions.empty:
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

    for _, pos in positions.iterrows():
        sym = pos['code'].split('.')[-1]
        pnl_pct = (pos['nominal_price'] / pos['cost_price'] - 1) * 100 if pos['cost_price'] > 0 else 0
        qty = int(pos['qty'])

        entry_date_str = opened.get(sym, {}).get("date")
        if entry_date_str:
            entry_date = datetime.fromisoformat(entry_date_str).date()
            days_held = (today - entry_date).days
        else:
            days_held = 999

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
                print(f"  📝 {pos['code']}: {qty}股, P&L {pnl_pct:+.1f}%, 持有{days_held}天 — {reason}")
            else:
                try:
                    ret, data = trade_ctx.place_order(
                        price=pos['nominal_price'],
                        qty=qty,
                        code=pos['code'],
                        trd_side=TrdSide.SELL,
                        order_type=OrderType.MARKET,
                        trd_env=TrdEnv.REAL
                    )
                    if ret == 0:
                        order_id = data['order_id'].iloc[0]
                        print(f"  ✅ {pos['code']}: 卖出{qty}股, P&L {pnl_pct:+.1f}%, {reason} — 订单 {order_id}")
                        sells.append({
                            "symbol": sym,
                            "qty": qty,
                            "pnl_pct": pnl_pct,
                            "days_held": days_held,
                            "reason": reason,
                            "order_id": str(order_id),
                            "side": "SELL",
                            "timestamp": datetime.now().isoformat()
                        })
                        if sym in opened:
                            del opened[sym]
                    else:
                        print(f"  ❌ {pos['code']}: {data}")
                except Exception as e:
                    print(f"  ❌ {pos['code']}: {e}")
        else:
            print(f"  ⏳ {pos['code']}: P&L {pnl_pct:+.1f}%, 持有{days_held}天 — 继续持有")

    if sells:
        log["trades"].extend(sells)
        log["positions_opened"] = opened
        save_trade_log(log)

    return sells


# ── 命令: full ──
def cmd_full(quote_ctx, trade_ctx, model_name="arrow_v12", hold_days=10, stop_loss_pct=-15.0, dry_run=False):
    """完整流程：先轮换卖出，再买入新信号。"""
    print("🔄 完整交易流程")
    print(f"   模型: {model_name} | 持有: {hold_days}天 | 止损: {stop_loss_pct}%")
    print(f"   {'DRY RUN模式' if dry_run else '⚡ 实盘模式'}")

    sells = cmd_rebalance(quote_ctx, trade_ctx, hold_days, stop_loss_pct, dry_run)
    buys = cmd_execute(quote_ctx, trade_ctx, model_name, dry_run)

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

    for t in trades[-30:]:
        ts = t.get("timestamp", "")[:19]
        pnl = f"{t.get('pnl_pct', 0):+.1f}%" if "pnl_pct" in t else "-"
        price = f"${t.get('price', 0):.2f}" if t.get("price") else "-"
        print(f"{ts:<20} {t.get('symbol', '?'):<8} {t.get('side', '?'):<5} "
              f"{t.get('qty', 0):>6} {price:>10} {pnl:>8}")


# ── 主入口 ──
def main():
    parser = argparse.ArgumentParser(description="Futu OpenD 实盘交易 - Falcons项目")
    parser.add_argument("command", choices=["status", "signals", "execute", "rebalance", "full", "history"],
                        help="执行命令")
    parser.add_argument("--model", default="falcon_v031", help="模型名 (falcon_v031 / arrow_v12 / blueshield_v10)")
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

    quote_ctx, trade_ctx = get_futu_client()

    try:
        if args.command == "status":
            cmd_status(quote_ctx, trade_ctx)
        elif args.command == "execute":
            cmd_execute(quote_ctx, trade_ctx, args.model, args.dry_run)
        elif args.command == "rebalance":
            cmd_rebalance(quote_ctx, trade_ctx, args.hold_days, args.stop_loss, args.dry_run)
        elif args.command == "full":
            cmd_full(quote_ctx, trade_ctx, args.model, args.hold_days, args.stop_loss, args.dry_run)
    finally:
        quote_ctx.close()
        trade_ctx.close()


if __name__ == "__main__":
    main()
