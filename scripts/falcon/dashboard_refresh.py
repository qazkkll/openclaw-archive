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


def get_data_freshness():
    """Check freshness of all data sources — 因子级检查
    
    逻辑：
    - 日更数据(parquet): 最新日期<=1天=🟢, >1天=🔴
    - 季度数据(FMP JSON): 今天查过=🟢(查了没新数据也是绿), 没查过=🔴
    - 评分数据: 评分日期<=1天=🟢, >1天=🔴
    """
    import pandas as pd
    # 加载检查日志
    check_log = {}
    log_file = PROJECT / "data" / "falcon" / "freshness_check_log.json"
    if log_file.exists():
        try:
            with open(log_file) as f:
                check_log = json.load(f)
        except:
            pass
    
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    results = []

    def check_parquet(label, path, date_col="date"):
        """日更数据：检查parquet最新日期"""
        try:
            if not Path(path).exists():
                results.append({"name": label, "status": "🔴", "latest": "N/A", "age_days": -1, "source": "文件缺失"})
                return
            df = pd.read_parquet(path, columns=[date_col])
            max_d = pd.to_datetime(df[date_col]).max()
            age = (now - max_d.to_pydatetime().replace(tzinfo=None)).days
            icon = "🟢" if age <= 1 else "🔴"
            results.append({"name": label, "status": icon, "latest": max_d.strftime("%Y-%m-%d"), "age_days": age, "source": "日更"})
        except Exception as e:
            results.append({"name": label, "status": "🔴", "latest": "error", "age_days": -1, "source": str(e)[:50]})

    def check_quarterly(label, log_key, path):
        """季度数据：检查今天是否查过(查了没新数据=🟢, 没查=🔴)"""
        try:
            # 先看检查日志
            last_check = check_log.get(log_key)
            if last_check:
                check_date = last_check[:10]  # ISO datetime前10位是日期
                if check_date == today_str:
                    # 今天查过，即使数据没变也是🟢
                    mtime = datetime.fromtimestamp(Path(path).stat().st_mtime) if Path(path).exists() else None
                    data_date = mtime.strftime("%Y-%m-%d") if mtime else "N/A"
                    results.append({"name": label, "status": "🟢", "latest": f"已检查({data_date})", "age_days": 0, "source": "季度(今日已查)"})
                    return
            # 没查过=🔴
            exists = Path(path).exists()
            if exists:
                mtime = datetime.fromtimestamp(Path(path).stat().st_mtime)
                data_date = mtime.strftime("%Y-%m-%d")
            else:
                data_date = "N/A"
            results.append({"name": label, "status": "🔴", "latest": f"未检查({data_date})", "age_days": -1, "source": "季度(未检查)"})
        except Exception as e:
            results.append({"name": label, "status": "🔴", "latest": "error", "age_days": -1, "source": str(e)[:50]})

    # === 日更数据 ===
    check_parquet("价格 OHLCV / Price", DATA_DIR / "us_prices_daily.parquet")
    check_parquet("VIX波动率 / VIX", PROJECT / "data" / "us" / "vix_10y.parquet")
    check_parquet("SPX指数 / S&P 500", PROJECT / "data" / "us" / "spx_daily.parquet")
    check_parquet("板块ETF / Sector ETFs", PROJECT / "data" / "us" / "sector_etf_daily.parquet")
    check_parquet("特征矩阵 / Features", DATA_DIR / "features_v04_1.parquet")

    # === SPX基本面(季度, 检查日志判定) ===
    for label, log_key, fname in [
        ("财务比率 / Ratios", "fmp_ratios", "fmp_ratios_historical.json"),
        ("关键指标 / Key Metrics", "fmp_key_metrics", "fmp_key_metrics.json"),
        ("增长率 / Growth", "fmp_financial_growth", "fmp_financial_growth.json"),
        ("收入报表 / Income Stmt", "fmp_income_stmt", "fmp_income_stmt.json"),
        ("资产负债 / Balance Sheet", "fmp_balance_sheet", "fmp_balance_sheet.json"),
        ("现金流 / Cashflow", "fmp_cashflow", "fmp_cashflow.json"),
        ("分析师预估 / Analyst Est", "fmp_analyst", "analyst_historical.json"),
        ("目标价 / Price Target", "fmp_price_target", "sp500_price_targets.json"),
        ("分析师评级 / Grades", "fmp_grades", "fmp_grades.json"),
    ]:
        check_quarterly(label, log_key, DATA_DIR / fname)

    # === Russell 2000基本面(季度) ===
    for label, log_key, fname in [
        ("R2K财务比率 / R2K Ratios", "fmp_ratios_russell", "fmp_ratios_russell.json"),
        ("R2K关键指标 / R2K Metrics", "fmp_metrics_russell", "fmp_metrics_russell.json"),
        ("R2K分析师 / R2K Analyst", "fmp_analyst_russell", "fmp_analyst_russell.json"),
        ("R2K增长率 / R2K Growth", "fmp_growth_russell", "fmp_growth_russell.json"),
        ("R2K价格 / R2K Prices", "russell_prices", "russell_prices.json"),
    ]:
        check_quarterly(label, log_key, DATA_DIR / fname)

    # === 其他数据 ===
    check_quarterly("新闻缓存 / News", "fmp_news", DATA_DIR / "fmp_news_cache.json")
    check_quarterly("盈利日历 / Earnings Cal", "fmp_earnings", DATA_DIR / "earnings_calendar.json")

    # === 复合因子 ===

    # === 评分输出 ===
    scored_files = sorted(DATA_DIR.glob("falcon_v044_scored_*.json"))
    if scored_files:
        try:
            with open(scored_files[-1]) as f:
                d = json.load(f)
            score_date = d.get("date", "unknown")
            try:
                sd = datetime.strptime(score_date, "%Y-%m-%d")
                age = (now - sd).days
                icon = "🟢" if age <= 1 else "🔴"
            except:
                age = -1
                icon = "🔴"
            results.append({"name": "Falcon评分", "status": icon, "latest": score_date, "age_days": age, "source": "日更"})
        except:
            results.append({"name": "Falcon评分", "status": "🔴", "latest": "error", "age_days": -1, "source": "读取失败"})
    else:
        results.append({"name": "Falcon评分", "status": "🔴", "latest": "N/A", "age_days": -1, "source": "无评分文件"})

    # Summary
    green = sum(1 for r in results if r["status"] == "🟢")
    red = sum(1 for r in results if r["status"] == "🔴")
    total = len(results)

    return {
        "items": results,
        "summary": {"green": green, "red": red, "total": total},
        "overall": "🟢" if red == 0 else "🔴",
    }


def get_factor_details(top_syms):
    """Extract 62 sub-factor percentile ranks for top stocks from features parquet.
    
    Returns: {sym: {factor_name: {"raw": float, "pctrank": float}}}
    """
    import pandas as pd
    try:
        features_path = DATA_DIR / "features_v04_1.parquet"
        if not features_path.exists():
            return {}
        
        # Only load needed columns + date + ticker
        features = pd.read_parquet(features_path)
        features["date"] = features["date"].astype(str)
        
        # Get latest date
        latest_date = features["date"].max()
        latest = features[features["date"] == latest_date].copy()
        
        # Identify factor columns
        r_cols = [c for c in latest.columns if c.startswith("r_") and "_qoq" not in c]
        m_cols = [c for c in latest.columns if c.startswith("m_")]
        g_cols = [c for c in latest.columns if c.startswith("g_")]
        a_cols = [c for c in latest.columns if c.startswith("a_")]
        qoq_cols = [c for c in latest.columns if c.startswith("r_") and "_qoq" in c]
        all_factor_cols = r_cols + m_cols + g_cols + a_cols + qoq_cols
        
        # Compute cross-sectional percentile ranks for all factors
        ranked = latest[["ticker"]].copy()
        for c in all_factor_cols:
            if c in latest.columns and latest[c].notna().sum() > 5:
                ranked[c + "_pctrank"] = latest[c].rank(pct=True)
        
        # Build result for top symbols
        result = {}
        for sym in top_syms:
            row = latest[latest["ticker"] == sym]
            rank_row = ranked[ranked["ticker"] == sym]
            if row.empty:
                continue
            
            sym_data = {}
            for c in all_factor_cols:
                if c in row.columns:
                    raw_val = row.iloc[0].get(c)
                    pct_val = rank_row.iloc[0].get(c + "_pctrank") if c + "_pctrank" in rank_row.columns else None
                    if pd.notna(raw_val):
                        sym_data[c] = {
                            "raw": round(float(raw_val), 4),
                            "pctrank": round(float(pct_val), 4) if pd.notna(pct_val) else None,
                        }
            result[sym] = sym_data
        
        return result
    except Exception as e:
        print(f"Factor detail error: {e}")
        return {}


def get_scores():
    """Load latest scored data + sub-factor details for top 10"""
    try:
        scored_files = sorted(DATA_DIR.glob("falcon_v044_scored_*.json"))
        if not scored_files:
            return {}, [], {}, {}
        with open(scored_files[-1]) as f:
            data = json.load(f)
        # Build sym -> score map from picks
        scores = {}
        top_syms = []
        for p in data.get("picks", []):
            scores[p["sym"]] = {
                "score": p.get("score", 0),
                "rank": p.get("rank_pct", 0),
                "fr": round(p.get("fund_ratio", 0) * 100, 1),
                "gc": round(p.get("growth_composite", 0) * 100, 1),
                "qoq": round(p.get("qoq", 0) * 100, 1),
                "cf": round(p.get("cashflow", 0) * 100, 1),
            }
            top_syms.append(p["sym"])
        picks = data.get("picks", [])
        regime = data.get("market_regime", {})
        
        # Get sub-factor details for top 10
        factor_details = get_factor_details(top_syms[:10])
        print(f"  Factor details: {len(factor_details)} tickers, {len(list(factor_details.values())[0]) if factor_details else 0} factors each")
        
        return scores, picks, regime, factor_details
    except Exception as e:
        print(f"Score error: {e}")
        return {}, [], {}, {}


def generate_html(alpaca, alpaca_cash, alpaca_equity, futu, scores, picks, regime, freshness=None, factor_details=None):
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
        },
        "freshness": freshness or {"items": [], "summary": {"green": 0, "red": 0, "total": 0}, "overall": "⚪"},
        "factor_details": factor_details or {},
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
    scores, picks, regime, factor_details = get_scores()
    print(f"  Scores: {len(scores)} picks, regime={regime.get('regime','?')}")

    # 3. Get data freshness
    freshness = get_data_freshness()
    s = freshness["summary"]
    print(f"  Freshness: {freshness['overall']} ({s['green']}🟢 {s['red']}🔴 / {s['total']} total)")

    # 4. Generate
    data = generate_html(alpaca, cash, equity, futu, scores, picks, regime, freshness, factor_details)
    print(f"  Total P&L: ${data['totals']['total_pnl']:+,.0f}")
    print(f"  Data saved to: {DATA_DIR}/dashboard_live_data.json")
    print("  Done.")


if __name__ == "__main__":
    main()
