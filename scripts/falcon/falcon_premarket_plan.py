#!/usr/bin/env python3
"""
🦅 Falcon 盘前计划生成器
========================
每日盘前(9:00 ET)运行:
1. 更新价格数据
2. 运行评分
3. Gatekeeper检查
4. 计算目标买入价位 (基于ATR+支撑阻力)
5. 计算止损价位
6. 计算仓位大小 (Kelly/固定比例)
7. 输出买入计划JSON + Telegram报告

用法:
    python3 falcon_premarket_plan.py              # 正常运行
    python3 falcon_premarket_plan.py --dry-run    # 模拟运行
    python3 falcon_premarket_plan.py --force      # 强制运行(忽略时间检查)
"""

import json, os, sys, subprocess, argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# ── 路径 ──
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
TRADE_DIR = DATA_DIR / "trades"
PLAN_FILE = TRADE_DIR / "premarket_plan.json"
POSITIONS_FILE = TRADE_DIR / "positions.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "falcon.yaml"

TRADE_DIR.mkdir(parents=True, exist_ok=True)

# 加载 .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── 配置 ──
HOLD_DAYS = 30
STOP_LOSS = -0.15
TOP_N = 10
BUY_SCORE_THRESHOLD = 0.55
VIX_THRESHOLD = 25
# 目标价位参数
ATR_PERIOD = 14          # ATR计算周期
ATR_MULTIPLIER = 1.5     # 目标价 = 现价 - ATR * multiplier (回调买入)
SUPPORT_LOOKBACK = 20    # 支撑位回看天数
# 仓位管理
MAX_POSITION_PCT = 0.10  # 单只最大仓位(占总资产)
MAX_TOTAL_EXPOSURE = 0.80 # 最大总仓位(留20%现金)
MIN_ORDER_VALUE = 500    # 最小下单金额

# ── 时区 ──
try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("US/Eastern")
except Exception:
    import pytz
    ET = pytz.timezone("US/Eastern")


def load_falcon_config():
    """从 falcon.yaml 读取配置。"""
    global HOLD_DAYS, STOP_LOSS, TOP_N, BUY_SCORE_THRESHOLD
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        trading = cfg.get("trading", {})
        model = cfg.get("model", {})
        HOLD_DAYS = trading.get("hold_days", HOLD_DAYS)
        STOP_LOSS = trading.get("stop_loss", STOP_LOSS)
        TOP_N = model.get("top_n", TOP_N)
        BUY_SCORE_THRESHOLD = model.get("buy_score_threshold", BUY_SCORE_THRESHOLD)
    except Exception:
        pass


def update_price_data():
    """更新价格数据(调用update_price_data.py)。"""
    print("📊 更新价格数据...")
    try:
        result = subprocess.run(
            [sys.executable, str(FALCON_DIR / "update_price_data.py")],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("  ✅ 价格数据更新成功")
            return True
        else:
            print(f"  ⚠️ 价格数据更新失败: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"  ⚠️ 价格数据更新异常: {e}")
        return False


def run_scoring():
    """运行评分脚本。"""
    print("📊 运行评分...")
    try:
        result = subprocess.run(
            [sys.executable, str(FALCON_DIR / "falcon_score.py"), "--skip-freshness"],
            capture_output=True, text=True, timeout=180
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode == 0:
            # 找到最新的评分文件
            import glob
            pattern = str(DATA_DIR / "falcon_v032_scored_*.json")
            files = sorted(glob.glob(pattern))
            if files:
                print(f"  ✅ 评分完成: {Path(files[-1]).name}")
                return files[-1]
        print(f"  ❌ 评分失败: {result.stderr[:200]}")
        return None
    except Exception as e:
        print(f"  ❌ 评分异常: {e}")
        return None


def run_gatekeeper():
    """运行Gatekeeper门禁检查。"""
    print("🛡️ 运行Gatekeeper...")
    try:
        sys.path.insert(0, str(FALCON_DIR))
        from falcon_gatekeeper import run_gatekeeper
        result = run_gatekeeper()
        verdict = result.get("verdict", "SKIP")
        passed = result.get("passed", 0)
        total = result.get("total", 5)
        print(f"  {'✅' if verdict == 'EXECUTE' else '⚠️' if verdict == 'REDUCE' else '❌'} "
              f"Gatekeeper: {verdict} ({passed}/{total})")
        return result
    except Exception as e:
        print(f"  ❌ Gatekeeper异常: {e}")
        return {"verdict": "SKIP", "error": str(e)}


def calculate_atr(ticker, prices_df, period=14):
    """计算ATR (Average True Range)。"""
    try:
        ticker_data = prices_df[prices_df["ticker"] == ticker].tail(period + 1)
        if len(ticker_data) < period:
            return None
        
        high = ticker_data["high"].values if "high" in ticker_data.columns else ticker_data["close"].values * 1.01
        low = ticker_data["low"].values if "low" in ticker_data.columns else ticker_data["close"].values * 0.99
        close = ticker_data["close"].values
        
        # True Range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        atr = np.mean(tr[-period:])
        return atr
    except Exception:
        return None


def find_support_level(ticker, prices_df, lookback=20):
    """找支撑位(最近lookback天的最低价)。"""
    try:
        ticker_data = prices_df[prices_df["ticker"] == ticker].tail(lookback)
        if len(ticker_data) < 5:
            return None
        return float(ticker_data["low"].min()) if "low" in ticker_data.columns else float(ticker_data["close"].min())
    except Exception:
        return None


def calculate_target_prices(picks, prices_df):
    """为每个pick计算目标买入价、止损价、目标卖出价。"""
    targets = []
    
    for pick in picks:
        sym = pick["sym"]
        current_price = pick.get("close", 0)
        score = pick.get("score", 0)
        
        if current_price <= 0:
            continue
        
        # ATR
        atr = calculate_atr(sym, prices_df, ATR_PERIOD)
        if atr is None:
            atr = current_price * 0.02  # 默认2%
        
        # 支撑位
        support = find_support_level(sym, prices_df, SUPPORT_LOOKBACK)
        
        # 目标买入价: 当前价 - ATR * multiplier (回调买入)
        # 但如果支撑位更高，用支撑位
        target_buy = current_price - atr * ATR_MULTIPLIER
        if support and support > target_buy:
            target_buy = support
        
        # 不能太远(最多回调5%)
        max_drop = current_price * 0.95
        if target_buy < max_drop:
            target_buy = max_drop
        
        # 止损价: 基于ATR
        stop_loss_price = current_price - atr * 3  # 3倍ATR止损
        if stop_loss_price < current_price * (1 + STOP_LOSS):
            stop_loss_price = current_price * (1 + STOP_LOSS)
        
        # 目标卖出价: 基于风险收益比 (至少2:1)
        risk = current_price - stop_loss_price
        target_sell = current_price + risk * 2
        
        targets.append({
            "symbol": sym,
            "current_price": round(current_price, 2),
            "target_buy": round(target_buy, 2),
            "stop_loss": round(stop_loss_price, 2),
            "target_sell": round(target_sell, 2),
            "atr": round(atr, 2),
            "support": round(support, 2) if support else None,
            "score": score,
            "rank_pct": pick.get("rank_pct", 0),
            "signal": pick.get("signal", ""),
            "universe": pick.get("universe", "SPX"),
            "factors": {
                "fund_growth": pick.get("fund_growth", 0),
                "cashflow": pick.get("cashflow", 0),
                "analyst": pick.get("analyst", 0),
                "grade_sentiment": pick.get("grade_sentiment", 0),
                "earnings": pick.get("earnings", 0),
                "balance": pick.get("balance", 0),
            }
        })
    
    return targets


def calculate_position_sizes(targets, account_equity, existing_positions):
    """计算每个目标的仓位大小。"""
    available_equity = account_equity * MAX_TOTAL_EXPOSURE
    existing_value = sum(
        float(p.get("qty", 0)) * float(p.get("current_price", 0))
        for p in existing_positions.values()
    )
    available_cash = available_equity - existing_value
    if available_cash <= 0:
        return []
    
    # 按分数加权分配
    total_score = sum(t["score"] for t in targets)
    if total_score <= 0:
        return []
    
    sized = []
    for t in targets:
        # 分数加权
        weight = t["score"] / total_score
        alloc = available_cash * weight
        
        # 限制单只最大仓位
        max_alloc = account_equity * MAX_POSITION_PCT
        alloc = min(alloc, max_alloc)
        
        # 计算股数
        qty = int(alloc / t["target_buy"]) if t["target_buy"] > 0 else 0
        if qty <= 0:
            continue
        
        actual_value = qty * t["target_buy"]
        if actual_value < MIN_ORDER_VALUE:
            continue
        
        sized.append({
            **t,
            "qty": qty,
            "alloc_value": round(actual_value, 2),
            "alloc_pct": round(actual_value / account_equity * 100, 2),
        })
    
    return sized


def load_account_info():
    """加载Alpaca账户信息。"""
    try:
        from alpaca.trading.client import TradingClient
        api_key = os.environ.get("APCA_API_KEY_ID")
        secret_key = os.environ.get("APCA_API_SECRET_KEY")
        if not api_key or not secret_key:
            return None, {}
        client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()
        pos_dict = {}
        for p in positions:
            pos_dict[p.symbol] = {
                "qty": int(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "unrealized_plpc": float(p.unrealized_plpc),
            }
        return {
            "equity": round(float(account.equity), 2),
            "cash": round(float(account.cash), 2),
            "buying_power": round(float(account.buying_power), 2),
        }, pos_dict
    except Exception as e:
        print(f"  ⚠️ Alpaca连接失败: {e}")
        return None, {}


def generate_report(plan, gatekeeper_result):
    """生成Telegram报告。"""
    lines = []
    lines.append(f"🦅 **Falcon 盘前计划**")
    lines.append(f"📅 {plan['date']} | 模型 V0.3.2")
    lines.append("")
    
    # 账户
    acct = plan.get("account", {})
    if acct:
        lines.append(f"💰 总资产: ${acct['equity']:,.0f} | 现金: ${acct['cash']:,.0f}")
        lines.append("")
    
    # Gatekeeper
    gk = gatekeeper_result
    gk_emoji = {"EXECUTE": "✅", "REDUCE": "⚠️", "SKIP": "❌"}.get(gk.get("verdict", ""), "❓")
    lines.append(f"🛡️ **Gatekeeper**: {gk_emoji} {gk.get('verdict','?')} ({gk.get('passed',0)}/{gk.get('total',5)})")
    for c in gk.get("checks", []):
        ce = "✅" if c.get("pass") else "❌"
        lines.append(f"  {ce} {c['name']}: {c['detail']}")
    lines.append("")
    
    # 买入计划
    buys = plan.get("buys", [])
    if buys:
        lines.append(f"🎯 **买入计划 ({len(buys)}只)**")
        for b in buys:
            lines.append(f"  **{b['symbol']}** | 分数{b['score']:.4f}")
            lines.append(f"    现价: ${b['current_price']:.2f}")
            lines.append(f"    目标买入: ${b['target_buy']:.2f} (回调{((b['current_price']-b['target_buy'])/b['current_price']*100):.1f}%)")
            lines.append(f"    止损: ${b['stop_loss']:.2f} ({((b['stop_loss']-b['current_price'])/b['current_price']*100):.1f}%)")
            lines.append(f"    目标卖出: ${b['target_sell']:.2f} ({((b['target_sell']-b['current_price'])/b['current_price']*100):.1f}%)")
            lines.append(f"    仓位: {b['qty']}股 (${b['alloc_value']:,.0f}, {b['alloc_pct']:.1f}%)")
            lines.append(f"    因子: 增长{b['factors']['fund_growth']:.0%} 现金流{b['factors']['cashflow']:.0%} 分析师{b['factors']['analyst']:.0%}")
            lines.append("")
    else:
        lines.append("ℹ️ 今日无买入计划（无🟢🟢信号或Gatekeeper拒绝）")
        lines.append("")
    
    # 持仓
    positions = plan.get("existing_positions", {})
    if positions:
        lines.append(f"📦 **当前持仓 ({len(positions)}只)**")
        for sym, p in positions.items():
            pnl = float(p.get("unrealized_plpc", 0)) * 100
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(f"  {emoji} {sym} | {p.get('qty',0)}股 | 成本${float(p.get('avg_entry_price',0)):.2f} | {pnl:+.1f}%")
        lines.append("")
    
    # 配置
    lines.append(f"⚙️ V0.3.2: 持有{HOLD_DAYS}天 | 止损{STOP_LOSS*100:.0f}% | Top{TOP_N} | VIX>{VIX_THRESHOLD}停买")
    lines.append(f"📁 计划文件: {PLAN_FILE.name}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Falcon 盘前计划")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行")
    parser.add_argument("--force", action="store_true", help="强制运行(忽略时间检查)")
    parser.add_argument("--skip-update", action="store_true", help="跳过价格更新")
    args = parser.parse_args()
    
    load_falcon_config()
    
    # 时间检查
    now_et = datetime.now(ET)
    if not args.force:
        hour = now_et.hour + now_et.minute / 60
        if hour < 6 or hour > 10:
            print(f"⏰ 当前ET时间 {now_et.strftime('%H:%M')}，不在盘前窗口(6:00-10:00 ET)")
            print(f"   使用 --force 强制运行")
            return
    
    print(f"🦅 Falcon 盘前计划 — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 60)
    
    # 1. 更新价格数据
    if not args.skip_update:
        update_price_data()
    
    # 2. 运行评分
    score_file = run_scoring()
    if not score_file:
        print("❌ 评分失败，无法生成计划")
        return
    
    # 加载评分结果
    with open(score_file) as f:
        score_data = json.load(f)
    picks = score_data.get("picks", [])
    
    # 3. Gatekeeper检查
    gatekeeper_result = run_gatekeeper()
    gk_verdict = gatekeeper_result.get("verdict", "SKIP")
    
    # 4. 加载账户和持仓
    account, existing_positions = load_account_info()
    if not account:
        account = {"equity": 100000, "cash": 100000, "buying_power": 100000}
        print("  ⚠️ 使用默认账户($100,000)")
    
    # 5. 计算目标价位
    print("\n📊 计算目标价位...")
    
    # 加载价格数据用于ATR计算
    import pandas as pd
    prices_df = pd.read_parquet(DATA_DIR / "features_v02.parquet")
    prices_df["date"] = prices_df["date"].astype(str)
    
    # 筛选🟢🟢信号
    green2_picks = [p for p in picks if p.get("signal") == "🟢🟢" and p.get("score", 0) >= BUY_SCORE_THRESHOLD]
    
    if gk_verdict == "SKIP":
        green2_picks = []
        print("  ❌ Gatekeeper SKIP，清空买入列表")
    elif gk_verdict == "REDUCE":
        green2_picks = green2_picks[:max(1, len(green2_picks) // 2)]
        print(f"  ⚠️ Gatekeeper REDUCE，减半至{len(green2_picks)}只")
    
    # 计算目标价位
    targets = calculate_target_prices(green2_picks, prices_df)
    
    # 计算仓位大小
    sized = calculate_position_sizes(targets, account["equity"], existing_positions)
    
    # 6. 生成计划
    plan = {
        "date": now_et.strftime("%Y-%m-%d"),
        "model": "falcon_v032",
        "timestamp": now_et.isoformat(),
        "account": account,
        "gatekeeper": gatekeeper_result,
        "buys": sized,
        "existing_positions": existing_positions,
        "config": {
            "hold_days": HOLD_DAYS,
            "stop_loss": STOP_LOSS,
            "top_n": TOP_N,
            "vix_threshold": VIX_THRESHOLD,
            "atr_multiplier": ATR_MULTIPLIER,
        },
        "score_file": str(score_file),
    }
    
    # 保存计划
    with open(PLAN_FILE, "w") as f:
        json.dump(plan, f, indent=2, default=str)
    
    # 7. 生成报告
    report = generate_report(plan, gatekeeper_result)
    print("\n" + report)
    
    # 8. 输出到stdout (供cron使用)
    if sized:
        print(f"\n✅ 计划已生成: {len(sized)}只买入计划")
    else:
        print(f"\nℹ️ 今日无买入计划")


if __name__ == "__main__":
    main()
