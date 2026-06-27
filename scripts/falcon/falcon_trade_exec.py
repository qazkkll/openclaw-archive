#!/usr/bin/env python3
"""
🦅 Falcon 模拟盘交易执行器
============================
每日收盘后自动执行：
1. 读取今日评分信号
2. 检查持仓：止损/到期 → 卖出
3. 新信号：🟢🟢且不在持仓 → 买入
4. 记录交易日志（含买入理由）
5. 输出Telegram报告

用法:
    python3 falcon_trade_exec.py              # 正常执行
    python3 falcon_trade_exec.py --dry-run    # 模拟运行（不下单）
    python3 falcon_trade_exec.py --report     # 只输出持仓报告
"""

import json, os, sys, glob, argparse
from datetime import datetime, timedelta
from pathlib import Path

# ── 路径 ──
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
TRADE_DIR = DATA_DIR / "trades"
JOURNAL_FILE = TRADE_DIR / "trade_journal.jsonl"
POSITIONS_FILE = TRADE_DIR / "positions.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "falcon.yaml"

TRADE_DIR.mkdir(parents=True, exist_ok=True)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Broker Adapter (统一持仓接口) ──
sys.path.insert(0, str(FALCON_DIR))
from broker_adapter import get_broker

# ── 配置 ──
HOLD_DAYS = 30
STOP_LOSS = -0.15
TOP_N = 5
BUY_SCORE_THRESHOLD = 0.65  # 与 falcon_score.py 的 ABSOLUTE_HIGH 对齐


def load_falcon_config():
    """从 falcon.yaml 读取配置。"""
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        global HOLD_DAYS, STOP_LOSS, TOP_N
        HOLD_DAYS = cfg.get("trading", {}).get("hold_days", HOLD_DAYS)
        STOP_LOSS = cfg.get("trading", {}).get("stop_loss", STOP_LOSS)
        TOP_N = cfg.get("model", {}).get("top_n", TOP_N)
    except Exception:
        pass


def load_latest_signals():
    """加载最新评分结果。"""
    pattern = str(DATA_DIR / "falcon_v031_scored_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, []
    latest = files[-1]
    with open(latest) as f:
        data = json.load(f)
    return latest, data.get("picks", [])


def load_positions():
    """加载当前持仓记录。"""
    if POSITIONS_FILE.exists():
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    return {"positions": {}}


def save_positions(pos_data):
    with open(POSITIONS_FILE, "w") as f:
        json.dump(pos_data, f, indent=2, default=str)


def append_journal(entry):
    """追加交易日志（JSONL格式，每行一条记录）。"""
    with open(JOURNAL_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def get_alpaca_client():
    """创建Alpaca Trading Client。"""
    from alpaca.trading.client import TradingClient
    api_key = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        print("❌ 缺少Alpaca API凭据")
        sys.exit(1)
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=True)


def generate_buy_reason(pick, all_picks):
    """生成买入理由（人类可读）。"""
    score = pick.get("score", 0)
    fund = pick.get("fund_ratio", 0)
    analyst = pick.get("analyst", 0)
    metric = pick.get("fund_metric", 0)
    universe = pick.get("universe", "SPX")

    reasons = []
    # 主要理由
    if fund >= 0.7:
        reasons.append(f"财务健康度高({fund:.0%})")
    if analyst >= 0.7:
        reasons.append(f"分析师看好({analyst:.0%})")
    if metric >= 0.7:
        reasons.append(f"盈利能力强({metric:.0%})")

    if not reasons:
        reasons.append(f"综合评分{score:.4f}，排名前{int((1-pick.get('rank_pct',0))*100)+1}%")

    return f"[{universe}] " + "，".join(reasons)


def execute_trades(client, dry_run=False):
    """执行完整交易流程。返回交易报告。"""
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    report = {
        "timestamp": datetime.now().isoformat(),
        "sells": [],
        "buys": [],
        "holds": [],
        "errors": [],
        "account": {},
    }

    # ── 1. 获取账户状态 ──
    account = client.get_account()
    positions = client.get_all_positions()
    report["account"] = {
        "cash": round(float(account.cash), 2),
        "equity": round(float(account.equity), 2),
        "buying_power": round(float(account.buying_power), 2),
        "position_count": len(positions),
    }

    # ── 2. 加载信号 ──
    signal_file, picks = load_latest_signals()
    if not picks:
        report["errors"].append("无信号文件")
        return report

    # ── 3. 止损/到期检查 ──
    pos_data = load_positions()
    today = datetime.now().date()

    for pos in positions:
        sym = pos.symbol
        pnl_pct = float(pos.unrealized_plpc) * 100
        qty = int(pos.qty)

        # 查找持仓记录
        pos_info = pos_data["positions"].get(sym, {})
        entry_date_str = pos_info.get("entry_date")
        if entry_date_str:
            entry_date = datetime.fromisoformat(entry_date_str).date()
            days_held = (today - entry_date).days
        else:
            days_held = 999  # 未知持仓日期

        should_sell = False
        reason = ""

        if pnl_pct <= STOP_LOSS * 100:
            should_sell = True
            reason = f"触发止损线({STOP_LOSS*100:.0f}%)，当前亏损{pnl_pct:+.1f}%"
        elif days_held >= HOLD_DAYS:
            should_sell = True
            reason = f"持有{days_held}天到期(规则:{HOLD_DAYS}天)，盈亏{pnl_pct:+.1f}%"

        if should_sell:
            if dry_run:
                sell_record = {
                    "symbol": sym, "qty": qty, "side": "SELL",
                    "pnl_pct": round(pnl_pct, 2), "days_held": days_held,
                    "reason": reason, "dry_run": True,
                }
                report["sells"].append(sell_record)
            else:
                try:
                    order = MarketOrderRequest(
                        symbol=sym, qty=qty,
                        side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                    )
                    submitted = client.submit_order(order_data=order)
                    sell_record = {
                        "symbol": sym, "qty": qty, "side": "SELL",
                        "pnl_pct": round(pnl_pct, 2), "days_held": days_held,
                        "reason": reason, "order_id": str(submitted.id),
                        "entry_price": pos_info.get("entry_price"),
                        "exit_price": round(float(pos.current_price), 2),
                    }
                    report["sells"].append(sell_record)
                    # 记录到日志
                    append_journal({**sell_record, "timestamp": datetime.now().isoformat(), "model": "falcon_v031"})
                    # 从持仓中移除
                    if sym in pos_data["positions"]:
                        del pos_data["positions"][sym]
                except Exception as e:
                    report["errors"].append(f"卖出{sym}失败: {e}")
        else:
            report["holds"].append({
                "symbol": sym, "qty": qty,
                "pnl_pct": round(pnl_pct, 2), "days_held": days_held,
            })

    # ── 4. 买入新信号 ──
    existing_syms = {p.symbol for p in positions}
    # 卖出的也要排除（可能还没结算）
    sold_syms = {s["symbol"] for s in report["sells"]}
    existing_syms = existing_syms | sold_syms

    buy_candidates = [
        p for p in picks
        if p.get("score", 0) >= BUY_SCORE_THRESHOLD
        and p.get("signal", "") == "🟢🟢"
        and p["sym"] not in existing_syms
    ][:TOP_N]

    if buy_candidates:
        # 计算可用资金
        cash = float(account.cash)
        # 如果有卖出，加上卖出的预估金额
        available = cash * 0.95  # 留5%缓冲
        per_stock = available / max(len(buy_candidates), 1)

        for pick in buy_candidates:
            sym = pick["sym"]
            price = pick.get("close", 0)
            if price <= 0:
                report["errors"].append(f"{sym}价格异常({price})")
                continue

            qty = int(per_stock / price)
            if qty <= 0:
                report["errors"].append(f"{sym}价格${price:.2f}太贵，买不起")
                continue

            reason = generate_buy_reason(pick, picks)

            if dry_run:
                buy_record = {
                    "symbol": sym, "qty": qty, "side": "BUY",
                    "price": price, "score": pick.get("score", 0),
                    "reason": reason, "dry_run": True,
                    "fund_ratio": pick.get("fund_ratio", 0),
                    "analyst": pick.get("analyst", 0),
                    "fund_metric": pick.get("fund_metric", 0),
                }
                report["buys"].append(buy_record)
            else:
                try:
                    order = MarketOrderRequest(
                        symbol=sym, qty=qty,
                        side=OrderSide.BUY, time_in_force=TimeInForce.DAY
                    )
                    submitted = client.submit_order(order_data=order)
                    buy_record = {
                        "symbol": sym, "qty": qty, "side": "BUY",
                        "price": price, "score": pick.get("score", 0),
                        "reason": reason, "order_id": str(submitted.id),
                        "fund_ratio": pick.get("fund_ratio", 0),
                        "analyst": pick.get("analyst", 0),
                        "fund_metric": pick.get("fund_metric", 0),
                    }
                    report["buys"].append(buy_record)
                    # 记录到日志
                    append_journal({**buy_record, "timestamp": datetime.now().isoformat(), "model": "falcon_v031"})
                    # 更新持仓记录
                    pos_data["positions"][sym] = {
                        "entry_date": datetime.now().isoformat(),
                        "entry_price": price,
                        "qty": qty,
                        "score": pick.get("score", 0),
                        "reason": reason,
                    }
                except Exception as e:
                    report["errors"].append(f"买入{sym}失败: {e}")

    save_positions(pos_data)
    return report


def format_telegram_report(report, signal_file):
    """格式化Telegram报告。"""
    lines = []
    ts = report["timestamp"][:16]
    acct = report["account"]

    lines.append(f"🦅 **Falcon 模拟盘日报**")
    lines.append(f"📅 {ts}")
    lines.append(f"💰 账户: ${acct['equity']:,.0f} (现金${acct['cash']:,.0f})")
    lines.append("")

    # 卖出
    if report["sells"]:
        lines.append(f"🔴 **卖出 ({len(report['sells'])}只)**")
        for s in report["sells"]:
            emoji = "🛑" if "止损" in s["reason"] else "⏰"
            lines.append(f"  {emoji} **{s['symbol']}** {s['qty']}股 | {s['reason']}")
            if s.get("pnl_pct"):
                lines.append(f"     盈亏: {s['pnl_pct']:+.1f}%")
        lines.append("")

    # 买入
    if report["buys"]:
        lines.append(f"🟢 **买入 ({len(report['buys'])}只)**")
        for b in report["buys"]:
            lines.append(f"  🎯 **{b['symbol']}** {b['qty']}股 × ${b['price']:.2f}")
            lines.append(f"     理由: {b['reason']}")
            lines.append(f"     评分: {b['score']:.4f} | 财务{b.get('fund_ratio',0):.0%} 分析师{b.get('analyst',0):.0%}")
        lines.append("")

    # 继续持有
    if report["holds"]:
        lines.append(f"⏳ **继续持有 ({len(report['holds'])}只)**")
        for h in report["holds"]:
            lines.append(f"  {h['symbol']} {h['qty']}股 | 盈亏{h['pnl_pct']:+.1f}% | 已持{h['days_held']}天")
        lines.append("")

    # 无操作
    if not report["sells"] and not report["buys"]:
        lines.append("ℹ️ 今日无交易（无新信号或持仓未到期）")
        lines.append("")

    # 错误
    if report["errors"]:
        lines.append(f"⚠️ **异常**")
        for e in report["errors"]:
            lines.append(f"  • {e}")
        lines.append("")

    lines.append(f"📁 信号: {Path(signal_file).name if signal_file else '无'}")
    lines.append(f"⚙️ 规则: 持有{HOLD_DAYS}天 | 止损{STOP_LOSS*100:.0f}% | Top{TOP_N}")

    return "\n".join(lines)


def format_position_report(client):
    """生成持仓报告（不含交易）。"""
    positions = client.get_all_positions()
    account = client.get_account()

    lines = []
    lines.append(f"🦅 **Falcon 持仓快照**")
    lines.append(f"💰 总资产: ${float(account.equity):,.0f} | 现金: ${float(account.cash):,.0f}")
    lines.append("")

    if positions:
        lines.append(f"📦 **持仓 ({len(positions)}只)**")
        for pos in positions:
            pnl = float(pos.unrealized_plpc) * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"  {emoji} **{pos.symbol}** {pos.qty}股 | 成本${float(pos.avg_entry_price):.2f} → 现价${float(pos.current_price):.2f} | {pnl:+.1f}%")
    else:
        lines.append("📦 无持仓")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Falcon 模拟盘交易执行")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行，不实际下单")
    parser.add_argument("--report", action="store_true", help="只输出持仓报告")
    args = parser.parse_args()

    load_falcon_config()
    client = get_alpaca_client()

    if args.report:
        print(format_position_report(client))
        return

    # 执行交易
    signal_file, _ = load_latest_signals()
    report = execute_trades(client, dry_run=args.dry_run)

    # 输出Telegram报告
    tg_report = format_telegram_report(report, signal_file)
    print(tg_report)

    # 保存报告到文件
    report_file = TRADE_DIR / f"report_{datetime.now().strftime('%Y%m%d')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # 同步Alpaca持仓到本地备份(Observer的回退数据源)
    # 注意: MERGE模式, 保留entry_date/score/reason等本地元数据
    try:
        broker = get_broker()
        broker_positions = broker.get_positions()
        # 读取现有pos_data(含entry_date等元数据)
        existing_pos = {}
        if POSITIONS_FILE.exists():
            try:
                with open(POSITIONS_FILE) as f:
                    existing_pos = json.load(f).get("positions", {})
            except Exception:
                pass
        pos_backup = {"positions": {}, "synced_at": datetime.now().isoformat()}
        for p in broker_positions:
            existing = existing_pos.get(p.symbol, {})
            pos_backup["positions"][p.symbol] = {
                "entry_price": p.avg_entry_price,
                "qty": p.qty,
                "current_price": p.current_price,
                "unrealized_plpc": p.unrealized_plpc,
                # 保留本地元数据(不被broker覆盖)
                "entry_date": existing.get("entry_date", ""),
                "score": existing.get("score", 0),
                "reason": existing.get("reason", ""),
            }
        with open(POSITIONS_FILE, "w") as f:
            json.dump(pos_backup, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ 持仓备份同步失败: {e}")


if __name__ == "__main__":
    main()
