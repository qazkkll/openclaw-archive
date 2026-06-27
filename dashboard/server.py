#!/usr/bin/env python3
"""
Falcon Trading Dashboard — HTTP Server
Serves the dashboard on port 8080 with auto-refreshing JSON data.

Usage:
    python3 dashboard/server.py              # default port 8080
    python3 dashboard/server.py --port 9090  # custom port
"""

import json
import os
import sys
import glob
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "falcon"
DASHBOARD_DIR = BASE_DIR / "dashboard"
REVIEWS_DIR = DATA_DIR / "reviews"
TRADES_DIR = DATA_DIR / "trades"
CONFIG_PATH = BASE_DIR / "config" / "falcon.yaml"

sys.path.insert(0, str(BASE_DIR))


def load_config():
    """Load falcon.yaml config."""
    try:
        import yaml
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        return {"error": str(e)}


def load_json(path):
    """Safely load a JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e), "path": str(path)}


def load_latest_scored():
    """Load the most recent scored signal file."""
    # 优先读合并版(SPX+R2K), 回退到旧格式
    files = sorted(DATA_DIR.glob("falcon_scored_*.json"), reverse=True)
    if not files:
        files = sorted(DATA_DIR.glob("falcon_v031_scored_*.json"), reverse=True)
    if files:
        return load_json(files[0])
    return {"error": "No scored signal files found"}


def load_data_status():
    """Check status of all data sources."""
    files_to_check = {
        "features_v02.parquet": {"desc": "特征矩阵 476只股票×9个技术指标", "min_size_mb": 100},
        "fmp_ratios_historical.json": {"desc": "财务比率历史 PE/PB/利润率等20项", "min_size_mb": 1},
        "analyst_historical.json": {"desc": "分析师预期历史 盈利修正方向", "min_size_mb": 1},
        "fmp_key_metrics.json": {"desc": "关键财务指标 ROE/ROA等23项", "min_size_mb": 1},
        "fmp_financial_growth.json": {"desc": "财务增长数据 收入/利润增长率", "min_size_mb": 1},
        "data_quality_report.json": {"desc": "数据质量检查报告", "min_size_mb": 0},
        "oos_validation.json": {"desc": "样本外验证 模型在新数据上的表现", "min_size_mb": 0},
        "timing_backtest_result.json": {"desc": "择时回测 什么时候买更划算", "min_size_mb": 0},
    }
    status = {}
    for fname, info in files_to_check.items():
        fpath = DATA_DIR / fname
        if fpath.exists():
            sz = fpath.stat().st_size
            mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            status[fname] = {
                "exists": True,
                "size_mb": round(sz / (1024 * 1024), 2),
                "modified": mtime.strftime("%Y-%m-%d %H:%M"),
                "age_days": age_days,
                "desc": info["desc"],
                "ok": sz > info["min_size_mb"] * 1024 * 1024,
            }
        else:
            status[fname] = {
                "exists": False,
                "desc": info["desc"],
                "ok": False,
            }

    # FinBERT sentiment directory
    finbert_dir = BASE_DIR / "data" / "finbert_sentiment"
    if finbert_dir.exists():
        parquets = list(finbert_dir.rglob("*.parquet"))
        status["finbert_sentiment/"] = {
            "exists": True,
            "ticker_count": len(parquets),
            "desc": "新闻情绪数据 AI读新闻打分",
            "ok": len(parquets) > 0,
        }
    else:
        status["finbert_sentiment/"] = {
            "exists": False,
            "desc": "新闻情绪数据 AI读新闻打分",
            "ok": False,
        }

    return status


def load_trades():
    """Load recent trade logs."""
    trades = []
    if TRADES_DIR.exists():
        for f in sorted(TRADES_DIR.glob("*.json"))[-20:]:
            trades.append(load_json(f))
    return trades


def load_reviews():
    """Load recent weekly reviews."""
    reviews = []
    if REVIEWS_DIR.exists():
        for f in sorted(REVIEWS_DIR.glob("*.json"))[-5:]:
            reviews.append(load_json(f))
    return reviews


def load_market_overview():
    """Load market overview from learning/daily or macro_snapshot."""
    import re
    overview = {
        "sp500": {"name": "标普500 S&P 500", "value": None, "change": None, "note": "美国最大的500家公司"},
        "vix": {"name": "恐慌指数 VIX", "value": None, "change": None, "note": "<15平静 15-25紧张 >25恐慌"},
        "dxy": {"name": "美元指数 DXY", "value": None, "note": "美元强弱"},
        "tnx": {"name": "十年国债 10Y Yield", "value": None, "note": "越高越不利于成长股"},
        "oil": {"name": "原油 WTI", "value": None, "note": "影响能源股和通胀"},
        "gold": {"name": "黄金 Gold", "value": None, "note": "避险资产"},
    }
    # Try macro_snapshot.json
    macro_file = DATA_DIR / "macro_snapshot.json"
    if macro_file.exists():
        try:
            macro = json.loads(macro_file.read_text())
            for key in overview:
                if key in macro:
                    overview[key].update(macro[key])
        except Exception:
            pass
    # Try latest learning file
    if overview["sp500"]["value"] is None:
        try:
            learn_dir = Path.home() / ".hermes" / "learning" / "daily"
            if learn_dir.exists():
                files = sorted(learn_dir.glob("*.md"), reverse=True)
                if files:
                    text = files[0].read_text()
                    sp = re.search(r"S&P[\\s]*(?:500|SPX)[\\s:]*([\\d,]+\\.?\\d*)", text)
                    if sp:
                        overview["sp500"]["value"] = float(sp.group(1).replace(",", ""))
                    vix = re.search(r"VIX[\s:]*([\d.]+)", text)
                    if vix:
                        overview["vix"]["value"] = float(vix.group(1))
        except Exception:
            pass
    return overview


def load_oos_validation():
    """Load out-of-sample validation data."""
    path = DATA_DIR / "oos_validation.json"
    if path.exists():
        return load_json(path)
    return {}


def load_backtest():
    """Load backtest results."""
    path = DATA_DIR / "timing_backtest_result.json"
    if path.exists():
        return load_json(path)
    return {}


def load_observer_state():
    """Load live observer state (pre/market/post monitoring)."""
    path = DATA_DIR / "observer_state.json"
    if path.exists():
        return load_json(path)
    return {"status": "not_running"}


def load_pending_alerts():
    """Load pending alerts."""
    path = DATA_DIR / "alerts" / "pending.json"
    if path.exists():
        return load_json(path)
    return []


def build_dashboard_data():
    """Build complete dashboard data payload."""
    now = datetime.now()
    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "name": "Falcon",
            "version": "0.4.0",
            "model": "falcon_v031",
            "universe": "SPX+R2K",
            "universe_size": 1053,
        },
        "config": load_config(),
        "data_sources": load_data_status(),
        "latest_signals": load_latest_scored(),
        "oos_validation": load_oos_validation(),
        "backtest": load_backtest(),
        "trades": load_trades(),
        "reviews": load_reviews(),
        "market_overview": load_market_overview(),
        "observer": load_observer_state(),
        "alerts": load_pending_alerts(),
    }


class ReusableHTTPServer(HTTPServer):
    """HTTPServer with SO_REUSEADDR for quick restarts."""
    allow_reuse_address = True


class DashboardHandler(SimpleHTTPRequestHandler):
    """Custom handler that serves dashboard HTML and JSON API."""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/data":
            data = build_dashboard_data()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(data, indent=2, ensure_ascii=False).encode())
            return

        if parsed.path == "/" or parsed.path == "/index.html":
            self.path = "/index.html"
            return super().do_GET()

        return super().do_GET()

    def log_message(self, format, *args):
        """Suppress default logging for cleanliness."""
        pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Falcon Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    # Serve from dashboard directory for static files
    os.chdir(DASHBOARD_DIR)

    server = ReusableHTTPServer((args.host, args.port), DashboardHandler)
    print(f"🦅 Falcon Dashboard serving at http://localhost:{args.port}")
    print(f"   Data dir: {DATA_DIR}")
    print(f"   Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🦅 Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
