#!/usr/bin/env python3
"""
Dashboard自动刷新脚本 — 每次运行重新生成dashboard.html
查询Alpaca+Futu实时持仓，嵌入HTML
"""
import os, sys, json, subprocess
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT / "data" / "falcon"
DASHBOARD = PROJECT / "dashboard.html"
TEMPLATE = PROJECT / "dashboard_template.html"

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")


def get_alpaca_positions():
    """Query Alpaca Paper Trading API"""
    import urllib.request
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    if not api_key or not secret:
        return [], 0, 0

    try:
        req = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/positions",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            positions = json.loads(r.read())

        req2 = urllib.request.Request(
            "https://paper-api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": secret}
        )
        with urllib.request.urlopen(req2, timeout=10) as r:
            acct = json.loads(r.read())

        result = []
        for p in positions:
            result.append({
                "sym": p.get("symbol", "?"),
                "qty": int(float(p.get("qty", 0))),
                "entry": round(float(p.get("avg_entry_price", 0)), 2),
                "price": round(float(p.get("current_price", 0)), 2),
                "pnl": round(float(p.get("unrealized_pl", 0)), 2),
                "pnl_pct": round(float(p.get("unrealized_plpc", 0)) * 100, 2),
                "market_val": round(float(p.get("market_value", 0)), 2),
            })

        cash = round(float(acct.get("cash", 0)), 2)
        equity = round(float(acct.get("equity", 0)), 2)
        return result, cash, equity
    except Exception as e:
        print(f"Alpaca error: {e}")
        return [], 0, 0


def get_futu_positions():
    """Query Futu OpenD"""
    try:
        from futu import OpenSecTradeContext, TrdMarket, SecurityFirm
        ctx = OpenSecTradeContext(
            filter_trdmarket=TrdMarket.HK,
            host="127.0.0.1", port=11111,
            security_firm=SecurityFirm.FUTUSECURITIES
        )
        ret, data = ctx.position_list_query()
        ctx.close()
        if ret != 0:
            return []

        result = []
        for _, row in data.iterrows():
            code = str(row.get("code", "?"))
            sym = code.split(".")[-1] if "." in code else code
            result.append({
                "sym": sym,
                "qty": int(float(row.get("qty", 0))),
                "cost": round(float(row.get("cost_price", 0)), 2),
                "price": round(float(row.get("nominal_price", 0)), 2),
                "val": round(float(row.get("market_val", 0)), 2),
                "pnl": round(float(row.get("pl_val", 0)), 2),
                "pnl_pct": round(float(row.get("pl_ratio", 0)), 2),
            })
        return result
    except Exception as e:
        print(f"Futu error: {e}")
        return []


def get_scores():
    """Load latest scored data"""
    try:
        scored_files = sorted(DATA_DIR.glob("falcon_v044_scored_*.json"))
        if not scored_files:
            return {}, [], {}
        with open(scored_files[-1]) as f:
            data = json.load(f)
        # Build sym -> score map from picks
        scores = {}
        for p in data.get("picks", []):
            scores[p["sym"]] = {
                "score": p.get("score", 0),
                "rank": p.get("rank_pct", 0),
                "fr": round(p.get("fund_ratio", 0) * 100, 1),
                "gc": round(p.get("growth_composite", 0) * 100, 1),
                "qoq": round(p.get("qoq", 0) * 100, 1),
                "cf": round(p.get("cashflow", 0) * 100, 1),
            }
        picks = data.get("picks", [])
        regime = data.get("market_regime", {})
        return scores, picks, regime
    except Exception as e:
        print(f"Score error: {e}")
        return {}, [], {}


def generate_html(alpaca, alpaca_cash, alpaca_equity, futu, scores, picks, regime):
    """Generate dashboard HTML with live data embedded"""
    # Compute totals
    alpaca_pnl = sum(p["pnl"] for p in alpaca)
    futu_pnl = sum(p["pnl"] for p in futu)
    total_pnl = alpaca_pnl + futu_pnl
    total_positions = len(alpaca) + len(futu)

    # Market regime
    vix = regime.get("vix", 16.5)
    regime_name = regime.get("regime", "bull")
    regime_map = {"bull": ("牛市", "green"), "neutral": ("震荡", "yellow"), "bear": ("熊市", "red"), "extreme_bear": ("极端", "red")}
    regime_label, regime_color = regime_map.get(regime_name, ("未知", "yellow"))

    # Picks JSON for JS
    picks_js = json.dumps(picks[:10])
    alpaca_js = json.dumps(alpaca)
    futu_js = json.dumps(futu)
    scores_js = json.dumps(scores)

    # Read template
    if TEMPLATE.exists():
        html = TEMPLATE.read_text()
    else:
        # Use current dashboard as template
        html = DASHBOARD.read_text()

    # Inject live data via script tag replacement
    # We'll replace the JS data sections
    now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # Write data file for the dashboard to load
    live_data = {
        "timestamp": now,
        "alpaca": alpaca,
        "alpaca_cash": alpaca_cash,
        "alpaca_equity": alpaca_equity,
        "futu": futu,
        "scores": scores,
        "picks": picks[:10],
        "regime": regime,
        "totals": {
            "positions": total_positions,
            "alpaca_pnl": round(alpaca_pnl, 2),
            "futu_pnl": round(futu_pnl, 2),
            "total_pnl": round(total_pnl, 2),
        }
    }

    data_file = DATA_DIR / "dashboard_live_data.json"
    with open(data_file, "w") as f:
        json.dump(live_data, f, indent=2, ensure_ascii=False)

    return live_data


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Refreshing dashboard data...")

    # 1. Get positions
    alpaca, cash, equity = get_alpaca_positions()
    futu = get_futu_positions()
    print(f"  Alpaca: {len(alpaca)} positions, cash=${cash:,.0f}")
    print(f"  Futu: {len(futu)} positions")

    # 2. Get scores
    scores, picks, regime = get_scores()
    print(f"  Scores: {len(scores)} picks, regime={regime.get('regime','?')}")

    # 3. Generate
    data = generate_html(alpaca, cash, equity, futu, scores, picks, regime)
    print(f"  Total P&L: ${data['totals']['total_pnl']:+,.0f}")
    print(f"  Data saved to: {DATA_DIR}/dashboard_live_data.json")
    print("  Done.")


if __name__ == "__main__":
    main()
