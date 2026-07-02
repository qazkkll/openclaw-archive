#!/usr/bin/env python3
"""
🦅 Falcon 数据Pipeline — 三道门控
===================================
Gate 1: 独立因子(A) → 原始数据拉取+新鲜度检查
Gate 2: 复合因子(A+B) → 特征构建+覆盖率检查
Gate 3: 评分 → falcon_score.py

每道门全部🟢才开下一道。每关打印结果。
"""

import json, os, sys, subprocess, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ── 路径 ──
PROJECT = Path.home() / ".hermes" / "openclaw-archive"
DATA = PROJECT / "data"
FALCON = DATA / "falcon"
US = DATA / "us"
CN = DATA / "cn"
PYTHON = sys.executable

# ── 时间 ──
TODAY = datetime.now()
TODAY_STR = TODAY.strftime("%Y-%m-%d")

# ── 颜色 ──
GREEN = "🟢"
YELLOW = "🟡"
RED = "🔴"
GRAY = "⚪"


def run(cmd: str, timeout: int = 300, label: str = "") -> Tuple[int, str]:
    """运行命令，返回(exit_code, stdout)"""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT))
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT ({timeout}s)"
    except Exception as e:
        return -2, str(e)


def file_age_days(path: Path) -> Optional[int]:
    """文件内容最新日期距今天数"""
    if not path.exists():
        return None
    try:
        suffix = path.suffix
        if suffix == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path, columns=None)
            # 找日期列
            for col in ['date', 'Date', 'trade_date']:
                if col in df.columns:
                    max_date = pd.to_datetime(df[col]).max()
                    return (TODAY - max_date.to_pydatetime().replace(tzinfo=None)).days
            # 没有日期列，用文件修改时间
            return (TODAY - datetime.fromtimestamp(path.stat().st_mtime)).days
        elif suffix == ".json":
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            return (TODAY - mtime).days
    except Exception:
        pass
    return (TODAY - datetime.fromtimestamp(path.stat().st_mtime)).days


def check_json_tickers(path: Path, expected: int = 476) -> Dict:
    """检查JSON数据文件的ticker数和最新日期"""
    if not path.exists():
        return {"exists": False, "tickers": 0, "latest": None, "status": "missing"}
    try:
        with open(path) as f:
            data = json.load(f)
        if not data:
            return {"exists": True, "tickers": 0, "latest": None, "status": "empty"}
        tickers = len(data)
        # 取第一个ticker的最新日期
        first = list(data.values())[0]
        if isinstance(first, list) and first:
            dates = [r.get('date', r.get('fillingDate', '')) for r in first if isinstance(r, dict)]
            dates = [d for d in dates if d]
            latest = max(dates) if dates else None
        else:
            latest = None
        return {"exists": True, "tickers": tickers, "latest": latest, "status": "ok"}
    except Exception as e:
        return {"exists": True, "tickers": 0, "latest": None, "status": f"error: {e}"}


def quarter_freshness(latest_date_str: Optional[str]) -> Tuple[str, int, str]:
    """判断季度数据是否新鲜。
    返回 (status_icon, age_days, message)
    
    三种状态:
    - 🟢 最新季度在当前季度内
    - 🟡 最新季度是上一个季度（正常，季报还没出）
    - 🔴 最新季度更旧（数据可更新但没更新）
    - ❌ 无数据
    """
    if not latest_date_str:
        return RED, 999, "无数据"
    
    try:
        latest = datetime.strptime(latest_date_str[:10], "%Y-%m-%d")
    except ValueError:
        return RED, 999, f"日期格式异常: {latest_date_str}"
    
    age_days = (TODAY - latest).days
    
    # 判断季度
    def get_quarter(dt):
        return (dt.year, (dt.month - 1) // 3 + 1)
    
    latest_q = get_quarter(latest)
    current_q = get_quarter(TODAY)
    
    # 季度差
    q_diff = (current_q[0] - latest_q[0]) * 4 + (current_q[1] - latest_q[1])
    
    if q_diff <= 0:
        return GREEN, age_days, f"当前季度Q{latest_q[1]} {latest_q[0]}"
    elif q_diff == 1:
        return YELLOW, age_days, f"上季度Q{latest_q[1]} {latest_q[0]}（本季度财报未出）"
    elif q_diff == 2:
        return YELLOW, age_days, f"Q{latest_q[1]} {latest_q[0]}（可能等Q{current_q[1]}财报）"
    else:
        return RED, age_days, f"Q{latest_q[1]} {latest_q[0]}（落后{q_diff}个季度，应有更新）"


# ═══════════════════════════════════════════════
# Gate 1: 独立因子(A) — 原始数据拉取
# ═══════════════════════════════════════════════
def gate1_independent_factors(pull: bool = False) -> Tuple[bool, Dict]:
    """
    Gate 1: 检查所有独立数据源。pull=True时先拉取再检查。
    返回 (all_green, results_dict)
    """
    print("=" * 60)
    print("🚪 GATE 1: 独立因子(A) — 数据新鲜度检查")
    print("=" * 60)
    
    results = {}
    
    if pull:
        # ── Step 1.1: 美股日更数据 ──
        print("\n📥 Step 1.1: 拉取美股日更数据...")
        code, out = run(f"{PYTHON} scripts/falcon/us_data_update_all.py --all", timeout=600, label="美股更新")
        us_ok = code == 0
        print(f"  {'✅' if us_ok else '❌'} us_data_update_all.py (exit={code})")
        if not us_ok:
            lines = out.strip().split('\n')[-10:]
            for l in lines:
                print(f"    {l}")
        
        # ── Step 1.2: A股日更数据 ──
        print("\n📥 Step 1.2: 拉取A股日更数据...")
        code, out = run(f"{PYTHON} scripts/cn/cn_data_update_all.py --all", timeout=600, label="A股更新")
        cn_ok = code == 0
        print(f"  {'✅' if cn_ok else '❌'} cn_data_update_all.py (exit={code})")
        if not cn_ok:
            lines = out.strip().split('\n')[-10:]
            for l in lines:
                print(f"    {l}")
    else:
        print("\n⏭️ 跳过数据拉取（由cron负责），直接检查新鲜度...")
    
    # ── Step 1.3: 检查美股独立因子 ──
    print("\n📊 Step 1.3: 检查美股独立因子新鲜度...")
    
    us_daily_checks = [
        ("价格(OHLCV)", FALCON / "us_prices_daily.parquet", 1),
        ("VIX指数", US / "vix_10y.parquet", 1),
        ("SPX指数", US / "spx_daily.parquet", 1),
        ("板块ETF", US / "sector_etf_daily.parquet", 1),
    ]
    
    us_daily_all_green = True
    for name, path, max_age in us_daily_checks:
        age = file_age_days(path)
        if age is None:
            icon, msg = RED, "文件不存在"
            us_daily_all_green = False
        elif age <= max_age:
            icon, msg = GREEN, f"{age}天前"
        elif age <= 3:
            icon, msg = YELLOW, f"{age}天前（周末可接受）"
            us_daily_all_green = False
        else:
            icon, msg = RED, f"{age}天前（过期!）"
            us_daily_all_green = False
        results[f"us_{name}"] = {"icon": icon, "age": age, "msg": msg}
        print(f"  {icon} {name}: {msg}")
    
    # ── Step 1.4: 检查FMP季度数据 ──
    print("\n📊 Step 1.4: 检查FMP季度数据...")
    
    fmp_quarterly = [
        ("财务比率", FALCON / "fmp_ratios_historical.json"),
        ("关键指标", FALCON / "fmp_key_metrics.json"),
        ("增长率", FALCON / "fmp_financial_growth.json"),
        ("资产负债表", FALCON / "fmp_balance_sheet.json"),
        ("现金流量表", FALCON / "fmp_cashflow.json"),
        ("利润表", FALCON / "fmp_income_stmt.json"),
        ("分析师预估", FALCON / "analyst_historical.json"),
    ]
    
    fmp_all_green = True
    for name, path in fmp_quarterly:
        info = check_json_tickers(path)
        if not info["exists"]:
            icon, age, msg = RED, 999, "❌ 文件不存在（从未拉取）"
            fmp_all_green = False
        elif info["status"] == "empty":
            icon, age, msg = RED, 999, "❌ 文件为空"
            fmp_all_green = False
        elif info["status"] != "ok":
            icon, age, msg = RED, 999, f"❌ {info['status']}"
            fmp_all_green = False
        else:
            icon, age, msg = quarter_freshness(info["latest"])
            tickers = info["tickers"]
            msg = f"{tickers}只, {msg}"
            if icon == RED:
                fmp_all_green = False
        results[f"fmp_{name}"] = {"icon": icon, "age": age, "msg": msg}
        print(f"  {icon} {name}: {msg}")
    
    # ── Step 1.5: 检查FMP日更数据 ──
    print("\n📊 Step 1.5: 检查FMP日更数据...")
    
    fmp_daily = [
        ("新闻缓存", FALCON / "fmp_news_cache.json", 1),
        ("盈利日历", FALCON / "earnings_calendar.json", 1),
        ("评级历史", FALCON / "fmp_grades.json", 7),
        ("分析师目标价", FALCON / "sp500_price_targets.json", 1),
    ]
    
    fmp_daily_all_green = True
    for name, path, max_age in fmp_daily:
        age = file_age_days(path)
        if age is None:
            icon, msg = RED, "文件不存在"
            fmp_daily_all_green = False
        elif age <= max_age:
            icon, msg = GREEN, f"{age}天前"
        elif age <= max_age * 3:
            icon, msg = YELLOW, f"{age}天前"
            # 不block
        else:
            icon, msg = RED, f"{age}天前（过期!）"
            fmp_daily_all_green = False
        results[f"fmp_daily_{name}"] = {"icon": icon, "age": age, "msg": msg}
        print(f"  {icon} {name}: {msg}")
    
    # ── Step 1.6: 检查A股独立因子 ──
    print("\n📊 Step 1.6: 检查A股独立因子...")
    
    cn_checks = [
        ("K线(a_hist_10y)", CN / "a_hist_10y.parquet", 1, "Date"),
        ("daily_basic", CN / "daily_basic.parquet", 1, "trade_date"),
        ("资金流(moneyflow)", CN / "moneyflow_core.parquet", 1, "trade_date"),
        ("北向资金", CN / "north_money.parquet", 2, "trade_date"),
    ]
    
    cn_all_green = True
    for name, path, max_age, date_col in cn_checks:
        if not path.exists():
            icon, msg = RED, "文件不存在"
            cn_all_green = False
        else:
            try:
                import pandas as pd
                df = pd.read_parquet(path, columns=[date_col])
                max_date = pd.to_datetime(df[date_col]).max()
                age = (TODAY - max_date.to_pydatetime().replace(tzinfo=None)).days
                if age <= max_age:
                    icon, msg = GREEN, f"{age}天前 ({max_date.strftime('%Y-%m-%d')})"
                elif age <= max_age + 2:
                    icon, msg = YELLOW, f"{age}天前 ({max_date.strftime('%Y-%m-%d')})"
                else:
                    icon, msg = RED, f"{age}天前 ({max_date.strftime('%Y-%m-%d')}) 过期!"
                    cn_all_green = False
            except Exception as e:
                icon, msg = RED, f"读取失败: {e}"
                cn_all_green = False
        results[f"cn_{name}"] = {"icon": icon, "msg": msg}
        print(f"  {icon} {name}: {msg}")
    
    # ── Gate 1 汇总 ──
    print("\n" + "─" * 60)
    
    all_green = us_daily_all_green and cn_all_green
    # FMP季度数据允许🟡（上季度财报未出）
    fmp_block = fmp_all_green  # 只有🔴才block
    
    gate1_pass = all_green and fmp_block
    
    status = f"{GREEN} PASS" if gate1_pass else f"{RED} FAIL"
    print(f"\n🚪 Gate 1 结果: {status}")
    print(f"  美股日更: {'✅' if us_daily_all_green else '❌'}")
    print(f"  FMP季度: {'✅' if fmp_all_green else '⚠️ 有红灯'}")
    print(f"  FMP日更: {'✅' if fmp_daily_all_green else '⚠️'}")
    print(f"  A股日更: {'✅' if cn_all_green else '❌'}")
    
    if not gate1_pass:
        print(f"\n  ⛔ 独立因子有红灯，不进入Gate 2")
    
    return gate1_pass, results


# ═══════════════════════════════════════════════
# Gate 2: 复合因子(A+B) — 特征构建
# ═══════════════════════════════════════════════
def gate2_composite_factors() -> Tuple[bool, Dict]:
    """
    Gate 2: 构建+检查复合因子。
    返回 (all_green, results_dict)
    """
    print("\n" + "=" * 60)
    print("🚪 GATE 2: 复合因子(A+B) — 特征构建")
    print("=" * 60)
    
    results = {}
    
    # ── Step 2.1: 构建features ──
    print("\n📥 Step 2.1: 构建 features_v04_1.parquet...")
    code, out = run(f"{PYTHON} scripts/falcon/build_features_v041.py", timeout=600, label="特征构建")
    build_ok = code == 0
    print(f"  {'✅' if build_ok else '❌'} build_features_v041.py (exit={code})")
    if not build_ok:
        lines = out.strip().split('\n')[-15:]
        for l in lines:
            print(f"    {l}")
        return False, results
    
    # ── Step 2.2: 检查features新鲜度 ──
    print("\n📊 Step 2.2: 检查 features_v04_1.parquet...")
    
    features_path = FALCON / "features_v04_1.parquet"
    if not features_path.exists():
        print(f"  {RED} features_v04_1.parquet 不存在!")
        return False, results
    
    import pandas as pd
    df = pd.read_parquet(features_path)
    max_date = pd.to_datetime(df['date']).max()
    age = (TODAY - max_date.to_pydatetime().replace(tzinfo=None)).days
    total_rows = len(df)
    total_cols = len(df.columns)
    
    icon = GREEN if age <= 1 else (YELLOW if age <= 3 else RED)
    print(f"  {icon} features_v04_1: {max_date.strftime('%Y-%m-%d')}, {age}天前, {total_rows:,}行×{total_cols}列")
    results["features_freshness"] = {"icon": icon, "age": age}
    
    # ── Step 2.3: 因子级覆盖率检查 ──
    print("\n📊 Step 2.3: 因子级覆盖率检查...")
    
    latest = df[df['date'] == df['date'].max()]
    
    factor_groups = {
        "fund_ratio (45%)": {
            "prefix": "r_",
            "weight": 0.45,
            "description": "财务比率截面排名",
        },
        "growth_composite (20%)": {
            "prefix": "g_",
            "weight": 0.20,
            "description": "成长组合(增长率+分析师+收入)",
        },
        "qoq (20%)": {
            "keywords": ["qoq"],
            "weight": 0.20,
            "description": "季度环比变化",
        },
        "cashflow (15%)": {
            "prefix": "c_",
            "weight": 0.15,
            "description": "现金流指标",
        },
    }
    
    all_groups_green = True
    
    for group_name, info in factor_groups.items():
        # 找到该组的因子列
        if "prefix" in info:
            cols = [c for c in df.columns if c.startswith(info["prefix"])]
        elif "keywords" in info:
            cols = [c for c in df.columns if any(kw in c.lower() for kw in info["keywords"])]
        else:
            cols = []
        
        if not cols:
            print(f"  {RED} {group_name}: 无因子列!")
            results[group_name] = {"icon": RED, "coverage": 0}
            all_groups_green = False
            continue
        
        # 整体覆盖率
        coverage = latest[cols].notna().mean().mean() * 100
        nan_total = latest[cols].isna().sum().sum()
        total_cells = len(cols) * len(latest)
        
        group_icon = GREEN if coverage >= 95 else (YELLOW if coverage >= 80 else RED)
        print(f"\n  {group_icon} {group_name}: 覆盖率={coverage:.1f}% ({total_cells - nan_total}/{total_cells})")
        print(f"      {info['description']}")
        
        if coverage < 80:
            all_groups_green = False
        
        results[group_name] = {"icon": group_icon, "coverage": coverage, "factors": len(cols)}
        
        # 逐因子检查
        for col in cols:
            sub_cov = latest[col].notna().mean() * 100
            sub_nan = latest[col].isna().sum()
            sub_icon = GREEN if sub_cov >= 95 else (YELLOW if sub_cov >= 80 else RED)
            
            # 标记问题因子
            note = ""
            if sub_cov == 0:
                note = " ⚠️ 全部NaN!"
                all_groups_green = False
            elif sub_cov < 50:
                note = " ⚠️ 覆盖率<50%"
            
            print(f"      {sub_icon} {col}: {sub_cov:.0f}% (NaN={sub_nan}){note}")
            results[f"factor_{col}"] = {"icon": sub_icon, "coverage": sub_cov}
    
    # ── Gate 2 汇总 ──
    print("\n" + "─" * 60)
    
    gate2_pass = build_ok and (icon == GREEN or icon == YELLOW) and all_groups_green
    status = f"{GREEN} PASS" if gate2_pass else f"{RED} FAIL"
    print(f"\n🚪 Gate 2 结果: {status}")
    print(f"  构建: {'✅' if build_ok else '❌'}")
    print(f"  新鲜度: {icon} ({age}天)")
    print(f"  因子覆盖率: {'✅ 全部≥80%' if all_groups_green else '❌ 有红灯'}")
    
    if not gate2_pass:
        print(f"\n  ⛔ 复合因子有问题，不进入Gate 3")
    
    return gate2_pass, results


# ═══════════════════════════════════════════════
# Gate 3: 评分
# ═══════════════════════════════════════════════
def gate3_scoring() -> Tuple[bool, Dict]:
    """
    Gate 3: 运行评分 + 检查输出。
    返回 (pass, results_dict)
    """
    print("\n" + "=" * 60)
    print("🚪 GATE 3: 评分 — falcon_score.py")
    print("=" * 60)
    
    results = {}
    
    # ── Step 3.1: 运行评分 ──
    print("\n📥 Step 3.1: 运行Falcon V0.4.4评分...")
    code, out = run(f"{PYTHON} scripts/falcon/falcon_score.py", timeout=120, label="Falcon评分")
    score_ok = code == 0
    print(f"  {'✅' if score_ok else '❌'} falcon_score.py (exit={code})")
    
    if not score_ok:
        lines = out.strip().split('\n')[-15:]
        for l in lines:
            print(f"    {l}")
        return False, results
    
    # 显示评分输出的关键行
    lines = out.strip().split('\n')
    for l in lines:
        if any(kw in l for kw in ['Top', 'Market', 'Regime', 'Score', 'Signal', '🟢', '🔴', 'VIX', 'saved']):
            print(f"    {l}")
    
    # ── Step 3.2: 验证输出文件 ──
    print("\n📊 Step 3.2: 验证评分输出...")
    
    # 找最新评分文件
    import glob
    scored_files = sorted(glob.glob(str(FALCON / "falcon_v044_scored_*.json")))
    if not scored_files:
        print(f"  {RED} 无评分输出文件!")
        return False, results
    
    latest_scored = Path(scored_files[-1])
    scored_age = file_age_days(latest_scored)
    
    try:
        with open(latest_scored) as f:
            scored = json.load(f)
        
        picks = scored.get("top", scored.get("picks", []))
        regime = scored.get("regime", scored.get("market_regime", {}))
        date = scored.get("date", "?")
        model = scored.get("model", "?")
        
        icon = GREEN if scored_age is not None and scored_age <= 1 else YELLOW
        print(f"  {icon} 评分文件: {latest_scored.name}")
        print(f"      日期: {date}")
        print(f"      模型: {model}")
        print(f"      Top数: {len(picks)}")
        print(f"      市场状态: {regime}")
        
        # 显示Top5
        print(f"\n  📊 Top 5 评分:")
        for i, p in enumerate(picks[:5]):
            sym = p.get("sym", p.get("ticker", "?"))
            score = p.get("score", 0)
            signal = p.get("signal", "?")
            name = p.get("name", "")
            print(f"      {i+1}. {sym} {name}: score={score:.4f} {signal}")
        
        results["scored"] = {"icon": icon, "date": date, "picks": len(picks), "model": model}
        
    except Exception as e:
        print(f"  {RED} 读取评分文件失败: {e}")
        return False, results
    
    # ── Step 3.3: 运行A股评分 ──
    print("\n📥 Step 3.3: 运行A股红杉评分...")
    code, out = run(f"{PYTHON} scripts/cn/gen_xgb_signal.py", timeout=120, label="A股评分")
    cn_score_ok = code == 0
    print(f"  {'✅' if cn_score_ok else '❌'} gen_xgb_signal.py (exit={code})")
    
    if cn_score_ok:
        lines = out.strip().split('\n')
        for l in lines:
            if any(kw in l for kw in ['Market', 'Top', '🟢', '🔴', 'saved', 'Signal', 'Bear', 'Bull']):
                print(f"      {l}")
    
    # ── Gate 3 汇总 ──
    print("\n" + "─" * 60)
    gate3_pass = score_ok
    status = f"{GREEN} PASS" if gate3_pass else f"{RED} FAIL"
    print(f"\n🚪 Gate 3 结果: {status}")
    print(f"  美股评分: {'✅' if score_ok else '❌'}")
    print(f"  A股评分: {'✅' if cn_score_ok else '❌'}")
    
    return gate3_pass, results


# ═══════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pull", action="store_true", help="先拉取数据再检查（默认只检查）")
    parser.add_argument("--skip-gate1", action="store_true", help="跳过Gate1检查")
    parser.add_argument("--skip-gate2", action="store_true", help="跳过Gate2构建")
    parser.add_argument("--score-only", action="store_true", help="只跑评分")
    args = parser.parse_args()
    
    start = time.time()
    
    print("🦅 Falcon 数据Pipeline — 三道门控")
    print(f"📅 {TODAY_STR} {datetime.now().strftime('%H:%M:%S')}")
    mode = "拉取+检查+构建+评分" if args.pull else ("仅评分" if args.score_only else "检查+构建+评分")
    print(f"模式: {mode}")
    print("=" * 60)
    
    # Gate 1
    if args.skip_gate1 or args.score_only:
        print("\n⏭️ 跳过Gate 1")
        gate1_pass = True
        g1_results = {}
    else:
        gate1_pass, g1_results = gate1_independent_factors(pull=args.pull)
    
    if not gate1_pass:
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"❌ Pipeline终止于Gate 1 ({elapsed:.0f}秒)")
        print(f"{'='*60}")
        return
    
    # Gate 2
    gate2_pass, g2_results = gate2_composite_factors()
    
    if not gate2_pass:
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"❌ Pipeline终止于Gate 2 ({elapsed:.0f}秒)")
        print(f"{'='*60}")
        return
    
    # Gate 3
    gate3_pass, g3_results = gate3_scoring()
    
    # 最终汇总
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    if gate3_pass:
        print(f"✅ Pipeline完成! ({elapsed:.0f}秒)")
        print(f"  Gate 1 (独立因子): ✅")
        print(f"  Gate 2 (复合因子): ✅")
        print(f"  Gate 3 (评分): ✅")
    else:
        print(f"❌ Pipeline在Gate 3失败 ({elapsed:.0f}秒)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
