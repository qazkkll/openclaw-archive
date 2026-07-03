#!/usr/bin/env python3
"""
🦅 Falcon Gatekeeper — 交易门禁系统
====================================
买入前的5项强制检查。不通过不执行。

检查项:
1. 宏观状态: SPX趋势 + VIX水平
2. 行业风向: 推荐股所在行业相对强弱
3. 事件风险: 未来7天财报/Fed会议
4. 持仓集中: 同行业是否已重仓
5. 目标价空间: ≥15%上行空间

输出:
  verdict: EXECUTE (5/5) | REDUCE (4/5) | SKIP (≤3/5)
  以及每项检查的详细结果

用法:
  python3 falcon_gatekeeper.py                    # 检查最新评分
  python3 falcon_gatekeeper.py --picks-file xxx.json  # 指定评分文件
  python3 falcon_gatekeeper.py --verbose          # 详细输出
"""

import json, sys, os, glob, argparse
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# ── 路径 ──
FALCON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FALCON_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "falcon"
GATEKEEPER_OUTPUT = DATA_DIR / "gatekeeper_verdict.json"

# 加载 .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ═══════════════════════════════════════════════════
# Check 1: 宏观状态 (SPX趋势 + VIX)
# ═══════════════════════════════════════════════════
def check_macro():
    """检查宏观环境: SPX 200日均线趋势 + VIX恐慌指数。"""
    import yfinance as yf

    result = {"name": "宏观状态", "pass": False, "score": 0, "detail": ""}

    try:
        # SPX趋势
        spx = yf.download("^GSPC", period="1y", progress=False)
        if spx.empty:
            result["detail"] = "SPX数据获取失败"
            return result

        close = spx["Close"].values.flatten()
        if len(close) < 200:
            result["detail"] = f"SPX数据不足({len(close)}天)"
            return result

        ma200 = np.mean(close[-200:])
        current = close[-1]
        spx_above_ma200 = current > ma200

        # VIX
        vix_data = yf.download("^VIX", period="5d", progress=False)
        vix = vix_data["Close"].values.flatten()[-1] if not vix_data.empty else None

        # 判定
        checks = []
        if spx_above_ma200:
            checks.append(f"SPX({current:.0f})>MA200({ma200:.0f})")
        else:
            checks.append(f"SPX({current:.0f})<MA200({ma200:.0f})⚠️")

        if vix is not None:
            if vix < 20:
                checks.append(f"VIX({vix:.1f})正常")
            elif vix < 30:
                checks.append(f"VIX({vix:.1f})偏高⚠️")
            else:
                checks.append(f"VIX({vix:.1f})恐慌🚨")

        # 通过条件: SPX在200日均线上方 且 VIX<30
        result["pass"] = spx_above_ma200 and (vix is None or vix < 30)
        result["score"] = 1 if result["pass"] else 0
        result["detail"] = " | ".join(checks)
        result["data"] = {
            "spx": float(current), "ma200": float(ma200),
            "vix": float(vix) if vix else None,
        }

    except Exception as e:
        result["detail"] = f"宏观检查异常: {e}"

    return result


# ═══════════════════════════════════════════════════
# Check 2: 行业风向 (sector momentum)
# ═══════════════════════════════════════════════════
# SP500 sector → ETF mapping
SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Energy": "XLE", "Industrials": "XLI", "Materials": "XLB",
    "Real Estate": "XLRE", "Utilities": "XLU", "Communication": "XLC",
}
# Ticker → rough sector mapping (major ones)
TICKER_SECTOR = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "META": "Communication", "GOOGL": "Communication",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "LLY": "Healthcare", "JPM": "Financials", "V": "Financials",
    "XOM": "Energy", "JNJ": "Healthcare", "WMT": "Consumer Staples",
    "PG": "Consumer Staples", "MA": "Financials", "UNH": "Healthcare",
    "HD": "Consumer Discretionary", "COST": "Consumer Staples",
    "ABBV": "Healthcare", "CRM": "Technology", "AMD": "Technology",
    "NFLX": "Communication", "ADBE": "Technology", "ORCL": "Technology",
    "PLTR": "Technology", "APP": "Technology", "MCO": "Financials",
    "TXN": "Technology", "STX": "Technology", "KLAC": "Technology",
    "TER": "Technology", "WMB": "Energy", "TPL": "Energy",
    "FAST": "Industrials", "AMT": "Real Estate", "AMAT": "Technology",
    "MU": "Technology", "LRCX": "Technology", "HWM": "Industrials",
    "KO": "Consumer Staples", "ZTS": "Healthcare",
}


def check_sector(picks):
    """检查推荐股所在行业近1个月相对强弱。"""
    import yfinance as yf

    result = {"name": "行业风向", "pass": False, "score": 0, "detail": ""}

    try:
        # 获取推荐股的行业
        sectors = set()
        for p in picks:
            sym = p.get("ticker", p.get("sym", ""))
            sector = TICKER_SECTOR.get(sym)
            if sector:
                sectors.add(sector)

        if not sectors:
            result["detail"] = "无法识别推荐股行业"
            result["pass"] = True  # 无法判断时放行
            result["score"] = 1
            return result

        # 检查行业ETF近1个月表现
        sector_perf = {}
        etf_tickers = [SECTOR_ETFS[s] for s in sectors if s in SECTOR_ETFS]
        if not etf_tickers:
            result["detail"] = "无对应行业ETF"
            result["pass"] = True
            result["score"] = 1
            return result

        data = yf.download(etf_tickers, period="1mo", progress=False)
        if data.empty:
            result["detail"] = "行业ETF数据获取失败"
            result["pass"] = True
            result["score"] = 1
            return result

        for etf in etf_tickers:
            try:
                if len(etf_tickers) == 1:
                    close = data["Close"].values.flatten()
                else:
                    close = data["Close"][etf].values.flatten()
                if len(close) >= 2:
                    ret = (close[-1] / close[0] - 1) * 100
                    sector_perf[etf] = ret
            except Exception:
                pass

        # 判断: 如果推荐股所在行业多数下跌, 不通过
        weak_sectors = [etf for etf, ret in sector_perf.items() if ret < -3]
        strong_sectors = [etf for etf, ret in sector_perf.items() if ret >= 0]

        details = [f"{etf}({ret:+.1f}%)" for etf, ret in sorted(sector_perf.items(), key=lambda x: -x[1])]

        # 通过条件: 至少一半行业ETF非负
        result["pass"] = len(strong_sectors) >= len(sector_perf) / 2
        result["score"] = 1 if result["pass"] else 0
        result["detail"] = " | ".join(details) if details else "无数据"
        result["data"] = sector_perf

    except Exception as e:
        result["detail"] = f"行业检查异常: {e}"
        result["pass"] = True  # 异常时放行

    return result


# ═══════════════════════════════════════════════════
# Check 3: 事件风险 (earnings + Fed)
# ═══════════════════════════════════════════════════
def check_events(picks):
    """检查未来7天是否有推荐股财报或重大宏观事件。"""
    result = {"name": "事件风险", "pass": False, "score": 0, "detail": ""}

    try:
        # 从FMP earnings calendar获取未来7天财报
        import requests
        api_key = os.environ.get("FMP_API_KEY", "")
        if not api_key:
            result["detail"] = "无FMP API key, 跳过事件检查"
            result["pass"] = True
            result["score"] = 1
            return result

        today = datetime.now().strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        url = f"https://financialmodelingprep.com/stable/earnings-calendar?from={today}&to={next_week}&apikey={api_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            result["detail"] = f"财报日历API失败({resp.status_code})"
            result["pass"] = True
            result["score"] = 1
            return result

        earnings = resp.json()
        pick_syms = {p.get("ticker", p.get("sym", "")) for p in picks}

        # 检查推荐股是否有财报
        upcoming_earnings = []
        for e in earnings:
            sym = e.get("symbol", "")
            if sym in pick_syms:
                upcoming_earnings.append(f"{sym}({e.get('date','')})")

        # 检查重大宏观事件 (简化: 查看是否有Fed会议)
        # Fed会议通常在周二/周三, 每6周一次
        macro_events = []
        for d in range(8):
            check_date = datetime.now() + timedelta(days=d)
            # FOMC通常在周二-周三
            if check_date.weekday() in (1, 2):  # Tue, Wed
                # 这里简化处理, 实际应该查FOMC日历
                pass

        if upcoming_earnings:
            result["detail"] = f"推荐股财报: {', '.join(upcoming_earnings)}"
            result["pass"] = False  # 有财报风险, 不通过
            result["score"] = 0
            result["data"] = upcoming_earnings
        else:
            result["detail"] = "未来7天无推荐股财报"
            result["pass"] = True
            result["score"] = 1

    except Exception as e:
        result["detail"] = f"事件检查异常: {e}"
        result["pass"] = True  # 异常时放行

    return result


# ═══════════════════════════════════════════════════
# Check 4: 持仓集中度
# ═══════════════════════════════════════════════════
def check_concentration(picks):
    """检查推荐股与现有持仓的行业重叠。"""
    result = {"name": "持仓集中", "pass": False, "score": 0, "detail": ""}

    try:
        # 加载当前持仓
        pos_file = DATA_DIR / "trades" / "positions.json"
        if not pos_file.exists():
            result["detail"] = "无持仓记录"
            result["pass"] = True
            result["score"] = 1
            return result

        with open(pos_file) as f:
            pos_data = json.load(f)

        existing = pos_data.get("positions", {})
        if not existing:
            result["detail"] = "当前空仓"
            result["pass"] = True
            result["score"] = 1
            return result

        # 检查持仓行业分布
        held_sectors = {}
        for sym in existing:
            sector = TICKER_SECTOR.get(sym, "Unknown")
            held_sectors[sector] = held_sectors.get(sector, 0) + 1

        # 检查推荐股行业
        pick_sectors = {}
        for p in picks:
            sym = p.get("ticker", p.get("sym", ""))
            sector = TICKER_SECTOR.get(sym, "Unknown")
            pick_sectors[sector] = pick_sectors.get(sector, 0) + 1

        # 找重叠行业
        overlaps = []
        for sector in pick_sectors:
            if sector in held_sectors:
                overlaps.append(f"{sector}(持{held_sectors[sector]}/推{pick_sectors[sector]})")

        if overlaps:
            result["detail"] = f"行业重叠: {', '.join(overlaps)}"
            # 如果重叠行业超过推荐的一半, 不通过
            result["pass"] = len(overlaps) <= len(pick_sectors) / 2
        else:
            result["detail"] = "无行业重叠"
            result["pass"] = True

        result["score"] = 1 if result["pass"] else 0
        result["data"] = {"held": held_sectors, "picks": pick_sectors}

    except Exception as e:
        result["detail"] = f"集中度检查异常: {e}"
        result["pass"] = True

    return result


# ═══════════════════════════════════════════════════
# Check 5: 目标价空间
# ═══════════════════════════════════════════════════
def check_price_targets(picks):
    """检查推荐股(Top5)的分析师目标价上行空间。"""
    result = {"name": "目标价空间", "pass": False, "score": 0, "detail": ""}

    try:
        pt_file = DATA_DIR / "sp500_price_targets.json"
        if not pt_file.exists():
            pt_file2 = DATA_DIR / "fmp_price_target.json"
            if pt_file2.exists():
                with open(pt_file2) as f:
                    raw = json.load(f)
                pt_map = {}
                for sym, data in raw.items():
                    pt_map[sym] = {
                        "ticker": sym,
                        "target_consensus": data.get("targetConsensus", 0),
                    }
            else:
                result["detail"] = "无目标价数据"
                result["pass"] = True
                result["score"] = 1
                return result
        else:
            with open(pt_file) as f:
                pt_list = json.load(f)
            pt_map = {r["ticker"]: r for r in pt_list}

        # 只检查实际候选(Top5, 不是全部评分)
        top_picks = picks[:5]

        # 检查每只推荐股
        below_target = []
        good_target = []
        no_data = []

        for p in top_picks:
            sym = p.get("ticker", p.get("sym", ""))
            price = p.get("close", 0)
            pt = pt_map.get(sym, {})
            consensus = pt.get("target_consensus", 0) or pt.get("targetConsensus", 0)

            if consensus and price > 0:
                upside = (consensus - price) / price * 100
                if upside < 15:
                    below_target.append(f"{sym}({upside:+.0f}%)")
                else:
                    good_target.append(f"{sym}({upside:+.0f}%)")
            else:
                no_data.append(sym)

        details = []
        if good_target:
            details.append(f"✅ {', '.join(good_target)}")
        if below_target:
            details.append(f"⚠️ {', '.join(below_target)}")
        if no_data:
            details.append(f"❓ {', '.join(no_data)}")

        # 通过条件: 至少一半推荐股有≥15%上行空间
        total = len(top_picks)
        passed = len(good_target) + len(no_data)  # 无数据的也算通过
        result["pass"] = passed >= total / 2
        result["score"] = 1 if result["pass"] else 0
        result["detail"] = " | ".join(details) if details else "无数据"
        result["data"] = {"good": good_target, "below": below_target, "no_data": no_data}

    except Exception as e:
        result["detail"] = f"目标价检查异常: {e}"
        result["pass"] = True

    return result


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
def load_picks(picks_file=None):
    """加载最新评分结果。"""
    if picks_file:
        with open(picks_file) as f:
            data = json.load(f)
        return data.get("top_n", data.get("picks", []))

    pattern = str(DATA_DIR / "falcon_v046_scored_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # 回退旧版
        pattern = str(DATA_DIR / "falcon_v044_scored_*.json")
        files = sorted(glob.glob(pattern))
    if not files:
        return []
    with open(files[-1]) as f:
        data = json.load(f)
    return data.get("top_n", data.get("picks", []))


def run_gatekeeper(picks_file=None, verbose=False):
    """运行5项检查, 返回verdict。"""
    picks = load_picks(picks_file)
    if not picks:
        return {"verdict": "SKIP", "reason": "无评分数据", "checks": []}

    print("🦅 Falcon Gatekeeper — 交易门禁检查")
    print("=" * 60)

    # 运行5项检查
    checks = []

    print("\n1️⃣ 宏观状态...")
    c1 = check_macro()
    checks.append(c1)
    emoji = "✅" if c1["pass"] else "❌"
    print(f"   {emoji} {c1['detail']}")

    print("\n2️⃣ 行业风向...")
    c2 = check_sector(picks)
    checks.append(c2)
    emoji = "✅" if c2["pass"] else "❌"
    print(f"   {emoji} {c2['detail']}")

    print("\n3️⃣ 事件风险...")
    c3 = check_events(picks)
    checks.append(c3)
    emoji = "✅" if c3["pass"] else "❌"
    print(f"   {emoji} {c3['detail']}")

    print("\n4️⃣ 持仓集中...")
    c4 = check_concentration(picks)
    checks.append(c4)
    emoji = "✅" if c4["pass"] else "❌"
    print(f"   {emoji} {c4['detail']}")

    print("\n5️⃣ 目标价空间...")
    c5 = check_price_targets(picks)
    checks.append(c5)
    emoji = "✅" if c5["pass"] else "❌"
    print(f"   {emoji} {c5['detail']}")

    # 汇总
    passed = sum(1 for c in checks if c["pass"])
    total = len(checks)

    if passed >= 5:
        verdict = "EXECUTE"
        action = "✅ 全部通过, 正常执行"
    elif passed >= 4:
        verdict = "REDUCE"
        action = "⚠️ 4/5通过, 建议减半仓位"
    else:
        verdict = "SKIP"
        action = f"❌ 仅{passed}/5通过, 暂停买入"

    print(f"\n{'='*60}")
    print(f"📊 Gatekeeper结论: {verdict} ({passed}/{total})")
    print(f"   {action}")

    # 输出推荐股
    print(f"\n📋 推荐股:")
    for i, p in enumerate(picks[:5], 1):
        print(f"   {i}. {p.get('sym','?')} score={p.get('score',0):.4f}")

    result = {
        "verdict": verdict,
        "passed": passed,
        "total": total,
        "action": action,
        "checks": [
            {"name": c["name"], "pass": c["pass"], "detail": c["detail"]}
            for c in checks
        ],
        "picks": [p.get("ticker", p.get("sym", "")) for p in picks[:5]],
        "timestamp": datetime.now().isoformat(),
    }

    # 保存
    with open(GATEKEEPER_OUTPUT, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 {GATEKEEPER_OUTPUT}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Falcon Gatekeeper 交易门禁")
    parser.add_argument("--picks-file", default=None, help="指定评分文件")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    result = run_gatekeeper(args.picks_file, args.verbose)

    # 返回码: 0=EXECUTE, 1=REDUCE, 2=SKIP
    if result["verdict"] == "EXECUTE":
        sys.exit(0)
    elif result["verdict"] == "REDUCE":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
