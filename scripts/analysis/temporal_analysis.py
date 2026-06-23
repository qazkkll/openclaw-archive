#!/usr/bin/env python3
"""
Layer 2 时序分析器 — 推荐时间维度的模式挖掘

功能：
  1. 加载并评分所有pending推荐（使用auto_scorer相同逻辑）
  2. 分析时间模式：周几推荐最准、月初/中/末、持仓天数收益分布
  3. 多时间维度收益对比（1天/3天/5天/10天）
  4. 输出markdown表格到stdout和文件

用法：
  python3 temporal_analysis.py              # 完整分析
  python3 temporal_analysis.py --dry-run    # 只看不写文件
  python3 temporal_analysis.py --force      # 强制重新评分（覆盖已评分的）

依赖：yfinance, json, os, sys, datetime, collections
"""
import json, os, sys, time, argparse
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ============================================================
# 配置
# ============================================================
ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
OUTPUT_DIR = os.path.join(ROOT, 'data/backtest-rounds')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'temporal-analysis.md')

MODEL_HORIZON_DAYS = 5
MODEL_THRESHOLD = 0.02  # 2%
TIME_HORIZONS = [1, 3, 5, 10]  # 分析多个时间维度

SPECIAL_TICKERS = {
    'SPY': 'SPY', 'QQQ': 'QQQ', 'VIX': '^VIX',
    'IWM': 'IWM', 'DIA': 'DIA', 'Semiconductor': 'SOXX',
}

# ============================================================
# 工具函数
# ============================================================
def to_yahoo_ticker(target: str) -> str:
    """将target转换为Yahoo Finance ticker"""
    target = target.strip()
    if target in SPECIAL_TICKERS:
        return SPECIAL_TICKERS[target]
    if ' ' not in target and not target.isdigit():
        return target
    parts = target.split()
    code = parts[0]
    if code.isdigit() and len(code) == 6:
        return f"{code}.SS" if code.startswith('6') else f"{code}.SZ"
    return code


def parse_date(date_str: str) -> datetime | None:
    """兼容 %Y%m%d 和 %Y-%m-%d 两种日期格式"""
    if not date_str:
        return None
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def is_valid_number(val) -> bool:
    """检查值是否为有效数字（非None/NaN/Inf）"""
    if val is None:
        return False
    import math
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return False
    return True


def get_price_on_date(ticker: str, date_str: str) -> float | None:
    """获取指定日期的收盘价（向后兼容）"""
    try:
        dt = parse_date(date_str)
        if dt is None:
            return None
        start = (dt - timedelta(days=3)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=5)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            return None
        target_date = dt.date()
        for d in sorted(hist.index, reverse=True):
            if d.date() <= target_date:
                return float(hist.loc[d, 'Close'])
        return float(hist.iloc[0]['Close'])
    except Exception as e:
        print(f"  ⚠️ 获取{ticker}价格失败: {e}")
        return None


def get_price_after_days(ticker: str, date_str: str, days: int) -> tuple[float | None, str | None]:
    """获取date_str之后N个交易日的收盘价"""
    try:
        dt = parse_date(date_str)
        if dt is None:
            return None, None
        start = dt.strftime('%Y-%m-%d')
        end = (dt + timedelta(days=days + 15)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        if hist.empty:
            return None, None
        target_date = dt.date()
        future_dates = [d for d in sorted(hist.index) if d.date() > target_date]
        if len(future_dates) >= days:
            d = future_dates[days - 1]
            return float(hist.loc[d, 'Close']), d.strftime('%Y-%m-%d')
        return None, None
    except Exception as e:
        print(f"  ⚠️ 获取{ticker}后{days}天价格失败: {e}")
        return None, None


def score_result(direction: str, ret: float) -> tuple[float, str]:
    """根据方向和收益率评分，返回 (score, outcome_label)"""
    if direction == 'bullish':
        if ret > MODEL_THRESHOLD:
            return 1.0, 'correct'
        elif ret > 0:
            return 0.5, 'partial'
        elif ret > -MODEL_THRESHOLD:
            return 0.25, 'partial_wrong'
        else:
            return 0.0, 'wrong'
    elif direction == 'bearish':
        if ret < -MODEL_THRESHOLD:
            return 1.0, 'correct'
        elif ret < 0:
            return 0.5, 'partial'
        elif ret < MODEL_THRESHOLD:
            return 0.25, 'partial_wrong'
        else:
            return 0.0, 'wrong'
    else:  # neutral
        if abs(ret) < MODEL_THRESHOLD:
            return 1.0, 'correct'
        elif abs(ret) < 0.05:
            return 0.5, 'partial'
        else:
            return 0.0, 'wrong'


def score_recommendation(rec: dict, dry_run: bool = False) -> dict | None:
    """
    对单条推荐进行多时间维度评分。
    返回包含各时间维度收益率的dict，或None（无法评分）。
    """
    target = rec.get('target', '')
    date = rec.get('date', '')
    direction = rec.get('direction', '')

    if not date or not target or target in ('大盘', 'S&P 500'):
        return None

    ticker = to_yahoo_ticker(target)
    base_price = get_price_on_date(ticker, date)
    if not is_valid_number(base_price):
        print(f"  ❌ {target}({ticker}): 无法获取{date}价格")
        return None

    result = {
        'id': rec.get('id', ''),
        'target': target,
        'ticker': ticker,
        'direction': direction,
        'date': date,
        'date_dt': parse_date(date),
        'source': rec.get('source', ''),
        'market': rec.get('market', ''),
        'confidence': rec.get('confidence', 0),
        'base_price': base_price,
        'horizons': {},
    }

    for h in TIME_HORIZONS:
        price_h, date_h = get_price_after_days(ticker, date, h)
        if is_valid_number(price_h) and is_valid_number(base_price):
            ret_h = (float(price_h) - float(base_price)) / float(base_price)
            if is_valid_number(ret_h):
                score_h, outcome_h = score_result(direction, ret_h)
                result['horizons'][h] = {
                    'price': float(price_h),
                    'date': date_h,
                    'return': ret_h,
                    'score': score_h,
                    'outcome': outcome_h,
                }
            else:
                result['horizons'][h] = None
        else:
            result['horizons'][h] = None

    # 主评分用5天
    h5 = result['horizons'].get(MODEL_HORIZON_DAYS)
    if h5 is not None:
        emoji = '✅' if h5['score'] >= 0.5 else '⚠️' if h5['score'] > 0 else '❌'
        print(f"  {emoji} {target}({ticker}): {direction} | ${base_price:.2f}→${h5['price']:.2f} (5d) | ret={h5['return']:+.2%} | score={h5['score']}")
    else:
        print(f"  ⏭️ {target}({ticker}): 无法获取5天后价格")

    time.sleep(0.3)
    return result


# ============================================================
# 分析引擎
# ============================================================
def analyze_by_dow(results: list[dict]) -> str:
    """按星期几分组分析"""
    dow_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
    groups = defaultdict(list)
    for r in results:
        dt = r.get('date_dt')
        if dt is None:
            continue
        dow = dt.weekday()
        h5 = r['horizons'].get(MODEL_HORIZON_DAYS)
        if h5 is not None and h5.get('score') is not None:
            groups[dow].append(h5)

    lines = ["## 📅 按星期几分组（5天周期）\n"]
    lines.append("| 星期 | 推荐数 | 平均收益率 | 命中率 | 平均得分 |")
    lines.append("|------|--------|-----------|--------|---------|")
    for dow in range(7):
        items = groups.get(dow, [])
        if not items:
            lines.append(f"| {dow_names[dow]} | 0 | - | - | - |")
            continue
        n = len(items)
        avg_ret = sum(h['return'] for h in items) / n
        correct = sum(1 for h in items if h['score'] >= 0.5)
        avg_score = sum(h['score'] for h in items) / n
        lines.append(f"| {dow_names[dow]} | {n} | {avg_ret:+.2%} | {correct}/{n} = {correct/n:.0%} | {avg_score:.2f} |")
    return '\n'.join(lines)


def analyze_by_month_phase(results: list[dict]) -> str:
    """按月初/月中/月末分组"""
    groups = {'月初(1-10)': [], '月中(11-20)': [], '月末(21-31)': []}
    for r in results:
        dt = r.get('date_dt')
        if dt is None:
            continue
        day = dt.day
        if day <= 10:
            phase = '月初(1-10)'
        elif day <= 20:
            phase = '月中(11-20)'
        else:
            phase = '月末(21-31)'
        h5 = r['horizons'].get(MODEL_HORIZON_DAYS)
        if h5 is not None and h5.get('score') is not None:
            groups[phase].append(h5)

    lines = ["## 📆 按月份阶段分组（5天周期）\n"]
    lines.append("| 阶段 | 推荐数 | 平均收益率 | 命中率 | 平均得分 |")
    lines.append("|------|--------|-----------|--------|---------|")
    for phase in ['月初(1-10)', '月中(11-20)', '月末(21-31)']:
        items = groups[phase]
        if not items:
            lines.append(f"| {phase} | 0 | - | - | - |")
            continue
        n = len(items)
        avg_ret = sum(h['return'] for h in items) / n
        correct = sum(1 for h in items if h['score'] >= 0.5)
        avg_score = sum(h['score'] for h in items) / n
        lines.append(f"| {phase} | {n} | {avg_ret:+.2%} | {correct}/{n} = {correct/n:.0%} | {avg_score:.2f} |")
    return '\n'.join(lines)


def analyze_time_horizons(results: list[dict]) -> str:
    """多时间维度收益对比"""
    lines = ["## ⏱️ 多时间维度收益对比\n"]
    lines.append("| 股票 | 方向 | 基准价 | 1天收益 | 3天收益 | 5天收益 | 10天收益 |")
    lines.append("|------|------|--------|---------|---------|---------|----------|")
    for r in results:
        row = f"| {r['target']} | {r['direction']} | ${r['base_price']:.2f} "
        for h in TIME_HORIZONS:
            data = r['horizons'].get(h)
            if data is not None:
                row += f"| {data['return']:+.2%} "
            else:
                row += "| - "
        row += "|"
        lines.append(row)

    # 汇总统计
    lines.append("")
    lines.append("### 各时间维度汇总\n")
    lines.append("| 时间维度 | 有效数 | 平均收益率 | 命中率 | 平均得分 |")
    lines.append("|----------|--------|-----------|--------|---------|")
    for h in TIME_HORIZONS:
        items = [r['horizons'][h] for r in results if r['horizons'].get(h) is not None and r['horizons'][h].get('score') is not None]
        if not items:
            lines.append(f"| {h}天 | 0 | - | - | - |")
            continue
        n = len(items)
        avg_ret = sum(x['return'] for x in items) / n
        correct = sum(1 for x in items if x['score'] >= 0.5)
        avg_score = sum(x['score'] for x in items) / n
        lines.append(f"| {h}天 | {n} | {avg_ret:+.2%} | {correct}/{n} = {correct/n:.0%} | {avg_score:.2f} |")
    return '\n'.join(lines)


def analyze_holding_distribution(results: list[dict]) -> str:
    """实际持仓天数收益分布"""
    # 统计从推荐日到最新可用价格的实际收益率
    lines = ["## 📊 实际收益率分布（5天周期）\n"]

    # 按收益率区间统计
    bins = [
        ('<-5%', -99, -0.05),
        ('-5%~-2%', -0.05, -0.02),
        ('-2%~0%', -0.02, 0),
        ('0%~2%', 0, 0.02),
        ('2%~5%', 0.02, 0.05),
        ('>5%', 0.05, 99),
    ]
    bin_counts = defaultdict(int)
    all_rets = []

    for r in results:
        h5 = r['horizons'].get(MODEL_HORIZON_DAYS)
        if h5 is None or h5.get('score') is None:
            continue
        ret = h5['return']
        all_rets.append(ret)
        for label, lo, hi in bins:
            if lo <= ret < hi:
                bin_counts[label] += 1
                break

    total = len(all_rets)
    lines.append("| 收益率区间 | 数量 | 占比 | 条形图 |")
    lines.append("|-----------|------|------|--------|")
    for label, lo, hi in bins:
        cnt = bin_counts[label]
        pct = cnt / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 2) + '░' * (25 - int(pct / 2))
        lines.append(f"| {label} | {cnt} | {pct:.1f}% | {bar} |")

    if all_rets:
        avg = sum(all_rets) / len(all_rets)
        median = sorted(all_rets)[len(all_rets) // 2]
        max_r = max(all_rets)
        min_r = min(all_rets)
        positive = sum(1 for x in all_rets if x > 0)
        lines.append("")
        lines.append(f"**统计摘要:** 平均={avg:+.2%}, 中位数={median:+.2%}, 最大={max_r:+.2%}, 最小={min_r:+.2%}, 正收益占比={positive}/{total}={positive/total:.0%}")
    return '\n'.join(lines)


def analyze_by_source(results: list[dict]) -> str:
    """按推荐来源分组"""
    groups = defaultdict(list)
    for r in results:
        src = r.get('source', 'unknown')
        h5 = r['horizons'].get(MODEL_HORIZON_DAYS)
        if h5 is not None and h5.get('score') is not None:
            groups[src].append(h5)

    lines = ["## 🔍 按推荐来源分组（5天周期）\n"]
    lines.append("| 来源 | 推荐数 | 平均收益率 | 命中率 | 平均得分 |")
    lines.append("|------|--------|-----------|--------|---------|")
    for src in sorted(groups.keys()):
        items = groups[src]
        n = len(items)
        avg_ret = sum(h['return'] for h in items) / n
        correct = sum(1 for h in items if h['score'] >= 0.5)
        avg_score = sum(h['score'] for h in items) / n
        lines.append(f"| {src} | {n} | {avg_ret:+.2%} | {correct}/{n} = {correct/n:.0%} | {avg_score:.2f} |")
    return '\n'.join(lines)


def analyze_by_market(results: list[dict]) -> str:
    """按市场分组"""
    groups = defaultdict(list)
    for r in results:
        mkt = r.get('market', 'unknown')
        h5 = r['horizons'].get(MODEL_HORIZON_DAYS)
        if h5 is not None and h5.get('score') is not None:
            groups[mkt].append(h5)

    lines = ["## 🌍 按市场分组（5天周期）\n"]
    lines.append("| 市场 | 推荐数 | 平均收益率 | 命中率 | 平均得分 |")
    lines.append("|------|--------|-----------|--------|---------|")
    for mkt in sorted(groups.keys()):
        items = groups[mkt]
        n = len(items)
        avg_ret = sum(h['return'] for h in items) / n
        correct = sum(1 for h in items if h['score'] >= 0.5)
        avg_score = sum(h['score'] for h in items) / n
        lines.append(f"| {mkt} | {n} | {avg_ret:+.2%} | {correct}/{n} = {correct/n:.0%} | {avg_score:.2f} |")
    return '\n'.join(lines)


# ============================================================
# 评分更新逻辑（写回recommendations.json）
# ============================================================
def update_recommendations(recs: list[dict], results: list[dict], dry_run: bool) -> int:
    """将评分结果写回原始推荐数据"""
    result_map = {r['id']: r for r in results}
    updated = 0
    for rec in recs:
        rid = rec.get('id', '')
        if rid not in result_map:
            continue
        r = result_map[rid]
        h5 = r['horizons'].get(MODEL_HORIZON_DAYS)
        if h5 is None:
            continue

        if not dry_run:
            rec['status'] = 'scored'
            rec['score'] = h5['score']
            rec['outcome'] = h5['outcome']
            rec['outcome_date'] = datetime.now().strftime('%Y-%m-%d')
            rec['actual_return'] = f"{h5['return']:+.2%}"
            rec['scoring_detail'] = {
                'base_price': r['base_price'],
                'next_price': h5['price'],
                'next_date': h5['date'],
                'horizon_days': MODEL_HORIZON_DAYS,
                'threshold': MODEL_THRESHOLD,
            }
            # 附加多时间维度数据
            multi_horizon = {}
            for h in TIME_HORIZONS:
                hd = r['horizons'].get(h)
                if hd is not None:
                    multi_horizon[f'ret_{h}d'] = hd['return']
                    multi_horizon[f'price_{h}d'] = hd['price']
                    multi_horizon[f'date_{h}d'] = hd['date']
            rec['scoring_detail']['multi_horizon'] = multi_horizon

        updated += 1
    return updated


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Layer 2 时序分析器')
    parser.add_argument('--dry-run', action='store_true', help='只分析不写文件')
    parser.add_argument('--force', action='store_true', help='强制重新评分')
    args = parser.parse_args()

    # 加载数据
    print("📂 加载推荐数据...")
    with open(TRACK_FILE) as f:
        data = json.load(f)
    recs = data.get('recommendations', [])

    # 筛选需要评分的
    today_str = datetime.now().strftime('%Y%m%d')
    today_str2 = datetime.now().strftime('%Y-%m-%d')

    pending = []
    already_scored = 0
    for r in recs:
        if r.get('status') == 'scored' and not args.force:
            already_scored += 1
            continue
        if r.get('status') == 'scored' and args.force:
            pass  # 重新评分
        # 跳过没有日期的、大盘/指数类
        target = r.get('target', '')
        if target in ('大盘', 'S&P 500', 'VIX', 'Semiconductor'):
            continue
        dt = parse_date(r.get('date', ''))
        if dt is None:
            continue
        # 跳过今天及未来的
        rec_date_str = dt.strftime('%Y%m%d')
        if rec_date_str >= today_str:
            continue
        pending.append(r)

    print(f"📊 待评分: {len(pending)}条 (已评分: {already_scored}条)\n")

    if not pending and not already_scored:
        print("❌ 没有可分析的数据")
        return

    # 评分pending的
    new_results = []
    for rec in pending:
        result = score_recommendation(rec, args.dry_run)
        if result is not None:
            new_results.append(result)

    # 加载已评分的做完整分析
    all_results = list(new_results)
    for r in recs:
        if r.get('status') == 'scored' and r.get('id') not in {nr['id'] for nr in new_results}:
            # 已评分的也加入分析，但需要重新获取多维度数据
            detail = r.get('scoring_detail', {})
            base_price = None
            next_price = None
            next_date = None

            if isinstance(detail, dict) and 'next_price' in detail:
                # Dict format from temporal_analysis.py's own scoring
                base_price = detail.get('base_price')
                next_price = detail.get('next_price')
                next_date = detail.get('next_date', '')
            elif isinstance(detail, str):
                # String format like "entry=99.32, exit=95.48, ret=-3.87%"
                import re as _re
                entry_m = _re.search(r'entry[=:]\s*([\d.]+)', detail)
                exit_m = _re.search(r'exit[=:]\s*([\d.]+)', detail)
                if entry_m and exit_m:
                    base_price = float(entry_m.group(1))
                    next_price = float(exit_m.group(1))

            # Fallback: try actual_return field if we have score but no detail prices
            if base_price is None:
                actual_ret = r.get('actual_return')
                score = r.get('score')
                if actual_ret is not None and score is not None:
                    # We can still include this rec with a synthetic return
                    ret_val = float(actual_ret) / 100.0 if isinstance(actual_ret, (int, float)) else 0
                    dt = parse_date(r.get('date', ''))
                    all_results.append({
                        'id': r.get('id', ''),
                        'target': r.get('target', ''),
                        'ticker': to_yahoo_ticker(r.get('target', '')),
                        'direction': r.get('direction', ''),
                        'date': r.get('date', ''),
                        'date_dt': dt,
                        'source': r.get('source', ''),
                        'market': r.get('market', ''),
                        'confidence': r.get('confidence', 0),
                        'base_price': 1.0,  # placeholder
                        'horizons': {
                            MODEL_HORIZON_DAYS: {
                                'price': 1.0 + ret_val,
                                'date': r.get('outcome_date', ''),
                                'return': ret_val,
                                'score': float(score),
                                'outcome': r.get('outcome', 'unknown'),
                            }
                        },
                    })
                continue

            if base_price is not None and next_price is not None and base_price > 0:
                dt = parse_date(r.get('date', ''))
                all_results.append({
                    'id': r.get('id', ''),
                    'target': r.get('target', ''),
                    'ticker': to_yahoo_ticker(r.get('target', '')),
                    'direction': r.get('direction', ''),
                    'date': r.get('date', ''),
                    'date_dt': dt,
                    'source': r.get('source', ''),
                    'market': r.get('market', ''),
                    'confidence': r.get('confidence', 0),
                    'base_price': base_price,
                    'horizons': {
                        MODEL_HORIZON_DAYS: {
                            'price': next_price,
                            'date': next_date or '',
                            'return': (next_price - base_price) / base_price,
                            'score': r.get('score', 0),
                            'outcome': r.get('outcome', ''),
                        }
                    },
                })

    if not all_results:
        print("❌ 无有效评分结果")
        return

    print(f"\n📈 总计 {len(all_results)} 条有效推荐\n")

    # 写回文件
    if not args.dry_run:
        updated = update_recommendations(recs, new_results, dry_run=False)
        with open(TRACK_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已更新 {updated} 条评分到 {TRACK_FILE}\n")

    # === 生成分析报告 ===
    report_lines = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_lines.append("# 📊 时序分析报告")
    report_lines.append(f"\n> 生成时间: {now_str}")
    report_lines.append(f"> 总推荐数: {len(all_results)}")
    report_lines.append(f"> 分析周期: {MODEL_HORIZON_DAYS}天 (阈值: {MODEL_THRESHOLD:.0%})")
    report_lines.append(f"> 时间维度: {', '.join(f'{h}天' for h in TIME_HORIZONS)}")
    report_lines.append("")

    # 1. 星期几分析
    report_lines.append(analyze_by_dow(all_results))
    report_lines.append("")

    # 2. 月初/中/末
    report_lines.append(analyze_by_month_phase(all_results))
    report_lines.append("")

    # 3. 多时间维度对比
    report_lines.append(analyze_time_horizons(all_results))
    report_lines.append("")

    # 4. 收益率分布
    report_lines.append(analyze_holding_distribution(all_results))
    report_lines.append("")

    # 5. 来源分析
    report_lines.append(analyze_by_source(all_results))
    report_lines.append("")

    # 6. 市场分析
    report_lines.append(analyze_by_market(all_results))
    report_lines.append("")

    # 7. Top/Bottom 个股
    scored_items = [(r, r['horizons'][MODEL_HORIZON_DAYS]) for r in all_results if r['horizons'].get(MODEL_HORIZON_DAYS)]
    if scored_items:
        report_lines.append("## 🏆 Top 5 推荐（按5天收益）\n")
        report_lines.append("| 排名 | 股票 | 方向 | 收益率 | 评分 | 来源 |")
        report_lines.append("|------|------|------|--------|------|------|")
        top5 = sorted(scored_items, key=lambda x: x[1]['return'], reverse=True)[:5]
        for i, (r, h) in enumerate(top5, 1):
            report_lines.append(f"| {i} | {r['target']} | {r['direction']} | {h['return']:+.2%} | {h['score']} | {r['source']} |")

        report_lines.append("")
        report_lines.append("## 💀 Bottom 5 推荐（按5天收益）\n")
        report_lines.append("| 排名 | 股票 | 方向 | 收益率 | 评分 | 来源 |")
        report_lines.append("|------|------|------|--------|------|------|")
        bottom5 = sorted(scored_items, key=lambda x: x[1]['return'])[:5]
        for i, (r, h) in enumerate(bottom5, 1):
            report_lines.append(f"| {i} | {r['target']} | {r['direction']} | {h['return']:+.2%} | {h['score']} | {r['source']} |")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*由 temporal_analysis.py 自动生成 | {now_str}*")

    # 输出到stdout
    full_report = '\n'.join(report_lines)
    print(full_report)

    # 保存到文件
    if not args.dry_run:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            f.write(full_report)
        print(f"\n💾 报告已保存到 {OUTPUT_FILE}")
    else:
        print(f"\n🔍 Dry run — 未保存到 {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
