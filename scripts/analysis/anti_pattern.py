#!/usr/bin/env python3
"""
Layer 2 反模式分析器 — 推荐质量的反面模式挖掘

功能：
  1. 加载并评分所有pending推荐（使用auto_scorer相同逻辑）
  2. 反模式分析：
     - 连续错误模式：找出连续2+条错误推荐的source/market组合
     - Agent vs 模型：对比agent_applied vs auto_inferred rules_source的命中率
     - 错误类型分类：方向错误(bullish但跌) vs 幅度错误(涨但没到2%)
     - reasoning长度 vs 命中率（是不是越长越不准？）
     - 过度自信模式：conf>0.7但reasoning没有数字的推荐命中率
  3. 输出markdown表格到stdout和文件

用法：
  python3 anti_pattern.py              # 完整分析
  python3 anti_pattern.py --dry-run    # 只看不写文件
  python3 anti_pattern.py --force      # 强制重新评分（覆盖已评分的）

依赖：yfinance, json, os, sys, datetime, collections, re
"""
import json, os, sys, time, argparse, re, math
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'anti-pattern-analysis.md')

MODEL_HORIZON_DAYS = 5
MODEL_THRESHOLD = 0.02  # 2%

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


def classify_error_type(direction: str, ret: float) -> str:
    """分类错误类型：方向错误 vs 幅度错误 vs 正确"""
    if direction == 'bullish':
        if ret < -MODEL_THRESHOLD:
            return '方向错误(bullish但跌)'
        elif 0 < ret <= MODEL_THRESHOLD:
            return '幅度错误(涨但没到2%)'
        else:
            return '正确'
    elif direction == 'bearish':
        if ret > MODEL_THRESHOLD:
            return '方向错误(bearish但涨)'
        elif -MODEL_THRESHOLD <= ret < 0:
            return '幅度错误(跌但没到2%)'
        else:
            return '正确'
    else:  # neutral
        if abs(ret) < MODEL_THRESHOLD:
            return '正确'
        else:
            return '幅度错误(波动过大)'


def has_number_in_reasoning(reasoning: str) -> bool:
    """检查reasoning中是否包含数字"""
    if not reasoning:
        return False
    return bool(re.search(r'\d', reasoning))


def score_recommendation(rec: dict) -> dict | None:
    """对单条推荐进行评分，返回包含5天收益率的dict，或None（无法评分）"""
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

    price_5d, date_5d = get_price_after_days(ticker, date, MODEL_HORIZON_DAYS)
    if not is_valid_number(price_5d) or not is_valid_number(base_price):
        print(f"  ⏭️ {target}({ticker}): 无法获取5天后价格")
        return None

    ret = (float(price_5d) - float(base_price)) / float(base_price)
    if not is_valid_number(ret):
        print(f"  ⏭️ {target}({ticker}): 收益率无效")
        return None

    score, outcome = score_result(direction, ret)
    error_type = classify_error_type(direction, ret)

    emoji = '✅' if score >= 0.5 else '⚠️' if score > 0 else '❌'
    print(f"  {emoji} {target}({ticker}): {direction} | ${base_price:.2f}→${price_5d:.2f} | ret={ret:+.2%} | {error_type} | score={score}")

    return {
        'id': rec.get('id', ''),
        'target': target,
        'ticker': ticker,
        'direction': direction,
        'date': date,
        'date_dt': parse_date(date),
        'source': rec.get('source', ''),
        'market': rec.get('market', ''),
        'confidence': rec.get('confidence', 0),
        'reasoning': rec.get('reasoning', ''),
        'rules_source': rec.get('rules_source', 'auto_inferred'),
        'base_price': base_price,
        'price_5d': price_5d,
        'date_5d': date_5d,
        'return': ret,
        'score': score,
        'outcome': outcome,
        'error_type': error_type,
    }


# ============================================================
# 反模式分析引擎
# ============================================================
def analyze_consecutive_errors(results: list[dict]) -> str:
    """连续错误模式：找出连续2+条错误推荐的source/market组合"""
    # 按(source, market)分组，按日期排序
    groups = defaultdict(list)
    for r in results:
        key = (r.get('source', ''), r.get('market', ''))
        groups[key].append(r)

    # 对每组按日期排序
    for key in groups:
        groups[key].sort(key=lambda x: x.get('date', ''))

    lines = ["## 🔴 连续错误模式\n"]
    lines.append("找出连续2+条错误推荐的(source, market)组合\n")
    lines.append("| 来源 | 市场 | 连续错误数 | 错误区间 | 推荐列表 |")
    lines.append("|------|------|-----------|----------|----------|")

    found_any = False
    for (src, mkt), items in sorted(groups.items()):
        # 找连续错误序列
        streak = 0
        streak_start = 0
        streaks = []
        for i, item in enumerate(items):
            if item['score'] == 0.0:  # 完全错误
                if streak == 0:
                    streak_start = i
                streak += 1
            else:
                if streak >= 2:
                    streaks.append((streak_start, i - 1, streak))
                streak = 0
        if streak >= 2:
            streaks.append((streak_start, len(items) - 1, streak))

        for start, end, count in streaks:
            found_any = True
            date_range = f"{items[start]['date']} ~ {items[end]['date']}"
            targets = ', '.join(f"{it['target']}({it['return']:+.1%})" for it in items[start:end+1])
            lines.append(f"| {src} | {mkt} | {count}条连续错误 | {date_range} | {targets} |")

    if not found_any:
        lines.append("| - | - | 无连续错误模式 | - | - |")
    return '\n'.join(lines)


def analyze_agent_vs_model(results: list[dict]) -> str:
    """Agent vs 模型：对比agent_applied vs auto_inferred rules_source的命中率"""
    groups = defaultdict(list)
    for r in results:
        src_type = r.get('rules_source', 'unknown')
        groups[src_type].append(r)

    lines = ["## 🤖 Agent vs 自动推理 命中率对比\n"]
    lines.append("| rules_source | 推荐数 | 正确数 | 部分正确 | 完全错误 | 命中率 | 平均收益率 | 平均得分 |")
    lines.append("|-------------|--------|--------|----------|----------|--------|-----------|---------|")

    for src_type in ['auto_inferred', 'agent_applied']:
        items = groups.get(src_type, [])
        if not items:
            lines.append(f"| {src_type} | 0 | - | - | - | - | - | - |")
            continue
        n = len(items)
        correct = sum(1 for r in items if r['score'] >= 0.5)
        partial = sum(1 for r in items if 0 < r['score'] < 0.5)
        wrong = sum(1 for r in items if r['score'] == 0.0)
        avg_ret = sum(r['return'] for r in items) / n
        avg_score = sum(r['score'] for r in items) / n
        hit_rate = correct / n if n > 0 else 0
        lines.append(f"| {src_type} | {n} | {correct} | {partial} | {wrong} | {correct}/{n} = {hit_rate:.0%} | {avg_ret:+.2%} | {avg_score:.2f} |")

    # 额外分析：agent_applied的高conf是否更准
    agent_items = groups.get('agent_applied', [])
    if agent_items:
        lines.append("")
        lines.append("### Agent推荐的置信度分布\n")
        lines.append("| 置信度区间 | 推荐数 | 命中率 | 平均得分 |")
        lines.append("|-----------|--------|--------|---------|")
        conf_bins = [
            ('0.5-0.6', 0.5, 0.6),
            ('0.6-0.7', 0.6, 0.7),
            ('0.7-0.8', 0.7, 0.8),
            ('0.8-1.0', 0.8, 1.0),
        ]
        for label, lo, hi in conf_bins:
            subset = [r for r in agent_items if lo <= r['confidence'] < hi]
            if not subset:
                lines.append(f"| {label} | 0 | - | - |")
                continue
            n = len(subset)
            correct = sum(1 for r in subset if r['score'] >= 0.5)
            avg_score = sum(r['score'] for r in subset) / n
            hit_rate = correct / n if n > 0 else 0
            lines.append(f"| {label} | {n} | {correct}/{n} = {hit_rate:.0%} | {avg_score:.2f} |")

    return '\n'.join(lines)


def analyze_error_types(results: list[dict]) -> str:
    """错误类型分类：方向错误 vs 幅度错误"""
    groups = defaultdict(list)
    for r in results:
        groups[r['error_type']].append(r)

    lines = ["## 🎯 错误类型分类\n"]
    lines.append("| 错误类型 | 数量 | 占比 | 平均收益率 | 平均置信度 | 示例 |")
    lines.append("|---------|------|------|-----------|-----------|------|")

    total = len(results)
    for error_type in sorted(groups.keys()):
        items = groups[error_type]
        n = len(items)
        pct = n / total * 100 if total > 0 else 0
        avg_ret = sum(r['return'] for r in items) / n
        avg_conf = sum(r['confidence'] for r in items) / n
        example = items[0]['target'] if items else '-'
        lines.append(f"| {error_type} | {n} | {pct:.1f}% | {avg_ret:+.2%} | {avg_conf:.2f} | {example} |")

    # 按source细分
    lines.append("")
    lines.append("### 按来源的错误类型分布\n")
    lines.append("| 来源 | 总数 | 方向错误 | 幅度错误 | 正确 | 错误率 |")
    lines.append("|------|------|---------|---------|------|--------|")
    src_groups = defaultdict(lambda: defaultdict(int))
    for r in results:
        src = r.get('source', 'unknown')
        src_groups[src][r['error_type']] += 1
    for src in sorted(src_groups.keys()):
        counts = src_groups[src]
        total_src = sum(counts.values())
        direction_err = counts.get('方向错误(bullish但跌)', 0) + counts.get('方向错误(bearish但涨)', 0)
        amplitude_err = counts.get('幅度错误(涨但没到2%)', 0) + counts.get('幅度错误(跌但没到2%)', 0) + counts.get('幅度错误(波动过大)', 0)
        correct_cnt = counts.get('正确', 0)
        error_rate = (total_src - correct_cnt) / total_src if total_src > 0 else 0
        lines.append(f"| {src} | {total_src} | {direction_err} | {amplitude_err} | {correct_cnt} | {error_rate:.0%} |")

    return '\n'.join(lines)


def analyze_reasoning_length(results: list[dict]) -> str:
    """reasoning长度 vs 命中率（是不是越长越不准？）"""
    lines = ["## 📝 Reasoning长度 vs 命中率\n"]
    lines.append("分析：reasoning越长是否越不准？\n")

    # 按长度分组
    bins = [
        ('短(<30字)', 0, 30),
        ('中(30-60字)', 30, 60),
        ('长(60-100字)', 60, 100),
        ('超长(>100字)', 100, 9999),
    ]
    lines.append("| 长度区间 | 推荐数 | 命中率 | 平均得分 | 平均收益率 | 平均置信度 |")
    lines.append("|---------|--------|--------|---------|-----------|-----------|")

    for label, lo, hi in bins:
        subset = [r for r in results if lo <= len(r.get('reasoning', '')) < hi]
        if not subset:
            lines.append(f"| {label} | 0 | - | - | - | - |")
            continue
        n = len(subset)
        correct = sum(1 for r in subset if r['score'] >= 0.5)
        avg_score = sum(r['score'] for r in subset) / n
        avg_ret = sum(r['return'] for r in subset) / n
        avg_conf = sum(r['confidence'] for r in subset) / n
        hit_rate = correct / n if n > 0 else 0
        lines.append(f"| {label} | {n} | {correct}/{n} = {hit_rate:.0%} | {avg_score:.2f} | {avg_ret:+.2%} | {avg_conf:.2f} |")

    # 相关性方向
    bin_data = []
    for label, lo, hi in bins:
        subset = [r for r in results if lo <= len(r.get('reasoning', '')) < hi]
        if subset:
            n = len(subset)
            correct = sum(1 for r in subset if r['score'] >=.5)
            bin_data.append((label, n, correct / n if n > 0 else 0))

    if len(bin_data) >= 2:
        hit_rates = [b[2] for b in bin_data]
        if hit_rates[0] > hit_rates[-1]:
            lines.append("\n**结论: reasoning越短命中率越高，简洁的推荐更可靠**")
        elif hit_rates[0] < hit_rates[-1]:
            lines.append("\n**结论: reasoning越长命中率越高，详细分析更可靠**")
        else:
            lines.append("\n**结论: reasoning长度与命中率无明显相关**")

    return '\n'.join(lines)


def analyze_overconfident(results: list[dict]) -> str:
    """过度自信模式：conf>0.7但reasoning没有数字的推荐命中率"""
    overconfident = [
        r for r in results
        if r['confidence'] > 0.7 and not has_number_in_reasoning(r.get('reasoning', ''))
    ]
    normal = [
        r for r in results
        if r['confidence'] > 0.7 and has_number_in_reasoning(r.get('reasoning', ''))
    ]

    lines = ["## 😤 过度自信模式分析\n"]
    lines.append("条件: confidence > 0.7\n")
    lines.append("| 类型 | 条件 | 推荐数 | 命中率 | 平均得分 | 平均收益率 |")
    lines.append("|------|------|--------|--------|---------|-----------|")

    # 过度自信（高conf无数字）
    if overconfident:
        n = len(overconfident)
        correct = sum(1 for r in overconfident if r['score'] >= 0.5)
        avg_score = sum(r['score'] for r in overconfident) / n
        avg_ret = sum(r['return'] for r in overconfident) / n
        hit_rate = correct / n if n > 0 else 0
        lines.append(f"| ⚠️ 过度自信 | conf>0.7 & 无数字 | {n} | {correct}/{n} = {hit_rate:.0%} | {avg_score:.2f} | {avg_ret:+.2%} |")
    else:
        lines.append(f"| ⚠️ 过度自信 | conf>0.7 & 无数字 | 0 | - | - | - |")

    # 正常高conf（有数字）
    if normal:
        n = len(normal)
        correct = sum(1 for r in normal if r['score'] >= 0.5)
        avg_score = sum(r['score'] for r in normal) / n
        avg_ret = sum(r['return'] for r in normal) / n
        hit_rate = correct / n if n > 0 else 0
        lines.append(f"| ✅ 有据可依 | conf>0.7 & 有数字 | {n} | {correct}/{n} = {hit_rate:.0%} | {avg_score:.2f} | {avg_ret:+.2%} |")
    else:
        lines.append(f"| ✅ 有据可依 | conf>0.7 & 有数字 | 0 | - | - | - |")

    # 低conf对比
    low_conf = [r for r in results if r['confidence'] <= 0.5]
    if low_conf:
        n = len(low_conf)
        correct = sum(1 for r in low_conf if r['score'] >= 0.5)
        avg_score = sum(r['score'] for r in low_conf) / n
        avg_ret = sum(r['return'] for r in low_conf) / n
        hit_rate = correct / n if n > 0 else 0
        lines.append(f"| 📊 基准 | conf≤0.5 | {n} | {correct}/{n} = {hit_rate:.0%} | {avg_score:.2f} | {avg_ret:+.2%} |")

    # 详细分析
    if overconfident:
        lines.append("")
        lines.append("### 过度自信推荐详情\n")
        lines.append("| 股票 | 方向 | 置信度 | 收益率 | 得分 | reasoning |")
        lines.append("|------|------|--------|--------|------|-----------|")
        for r in overconfident[:10]:  # 最多显示10条
            reasoning_short = r.get('reasoning', '')[:40] + ('...' if len(r.get('reasoning', '')) > 40 else '')
            lines.append(f"| {r['target']} | {r['direction']} | {r['confidence']:.2f} | {r['return']:+.2%} | {r['score']} | {reasoning_short} |")

    # 结论
    if overconfident and normal:
        oc_hit = sum(1 for r in overconfident if r['score'] >= 0.5) / len(overconfident)
        n_hit = sum(1 for r in normal if r['score'] >= 0.5) / len(normal)
        if oc_hit < n_hit:
            diff = n_hit - oc_hit
            lines.append(f"\n**结论: 过度自信的推荐命中率比有数据支撑的低 {diff:.0%}，高置信度但没有数字的推荐更危险**")
        else:
            lines.append("\n**结论: 过度自信的推荐命中率与有数据支撑的持平**")
    elif overconfident:
        oc_hit = sum(1 for r in overconfident if r['score'] >= 0.5) / len(overconfident)
        lines.append(f"\n**结论: 过度自信推荐命中率 {oc_hit:.0%}（仅{len(overconfident)}条样本）**")

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

        if not dry_run:
            rec['status'] = 'scored'
            rec['score'] = r['score']
            rec['outcome'] = r['outcome']
            rec['outcome_date'] = datetime.now().strftime('%Y-%m-%d')
            rec['actual_return'] = f"{r['return']:+.2%}"
            rec['scoring_detail'] = {
                'base_price': r['base_price'],
                'next_price': r['price_5d'],
                'next_date': r['date_5d'],
                'horizon_days': MODEL_HORIZON_DAYS,
                'threshold': MODEL_THRESHOLD,
                'error_type': r['error_type'],
            }
        updated += 1
    return updated


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Layer 2 反模式分析器')
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

    pending = []
    already_scored = 0
    for r in recs:
        if r.get('status') == 'scored' and not args.force:
            already_scored += 1
            continue
        if r.get('status') == 'scored' and args.force:
            pass  # 重新评分
        target = r.get('target', '')
        if target in ('大盘', 'S&P 500'):
            continue
        dt = parse_date(r.get('date', ''))
        if dt is None:
            continue
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
        result = score_recommendation(rec)
        if result is not None:
            new_results.append(result)
        time.sleep(0.3)

    # 也加载已评分的做完整分析
    all_results = list(new_results)
    for r in recs:
        if r.get('status') == 'scored' and r.get('id') not in {nr['id'] for nr in new_results}:
            dt = parse_date(r.get('date', ''))
            detail = r.get('scoring_detail', {})
            # Handle scoring_detail as dict (new format) or string (old format)
            if isinstance(detail, dict) and 'next_price' in detail:
                base_price = detail.get('base_price', 0)
                next_price = detail.get('next_price', 0)
                if base_price and next_price and base_price != 0:
                    ret = (next_price - base_price) / base_price
                    score_val, outcome_val = score_result(r.get('direction', ''), ret)
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
                        'reasoning': r.get('reasoning', ''),
                        'rules_source': r.get('rules_source', 'auto_inferred'),
                        'base_price': base_price,
                        'price_5d': next_price,
                        'date_5d': detail.get('next_date', ''),
                        'return': ret,
                        'score': score_val,
                        'outcome': outcome_val,
                        'error_type': classify_error_type(r.get('direction', ''), ret),
                    })
            elif isinstance(detail, str):
                # Parse old string format: "entry=99.32, exit=95.48, ret=-3.87%"
                m_entry = re.search(r'entry=([\d.]+)', detail)
                m_exit = re.search(r'exit=([\d.]+)', detail)
                m_ret = re.search(r'ret=([+-]?[\d.]+)', detail)
                if m_ret:
                    ret_pct = float(m_ret.group(1))
                    ret = ret_pct / 100  # convert percentage to decimal
                    base_price = float(m_entry.group(1)) if m_entry else 0
                    next_price = float(m_exit.group(1)) if m_exit else 0
                    score_val, outcome_val = score_result(r.get('direction', ''), ret)
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
                        'reasoning': r.get('reasoning', ''),
                        'rules_source': r.get('rules_source', 'auto_inferred'),
                        'base_price': base_price,
                        'price_5d': next_price,
                        'date_5d': '',
                        'return': ret,
                        'score': score_val,
                        'outcome': outcome_val,
                        'error_type': classify_error_type(r.get('direction', ''), ret),
                    })
            elif r.get('actual_return') is not None:
                # Fallback: use actual_return field
                ret = float(r['actual_return']) / 100
                score_val, outcome_val = score_result(r.get('direction', ''), ret)
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
                    'reasoning': r.get('reasoning', ''),
                    'rules_source': r.get('rules_source', 'auto_inferred'),
                    'base_price': 0,
                    'price_5d': 0,
                    'date_5d': '',
                    'return': ret,
                    'score': score_val,
                    'outcome': outcome_val,
                    'error_type': classify_error_type(r.get('direction', ''), ret),
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

    # === 生成反模式分析报告 ===
    report_lines = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_lines.append("# 🔍 反模式分析报告")
    report_lines.append(f"\n> 生成时间: {now_str}")
    report_lines.append(f"> 总推荐数: {len(all_results)}")
    report_lines.append(f"> 分析周期: {MODEL_HORIZON_DAYS}天 (阈值: {MODEL_THRESHOLD:.0%})")
    report_lines.append("")

    # 1. 连续错误模式
    report_lines.append(analyze_consecutive_errors(all_results))
    report_lines.append("")

    # 2. Agent vs 模型
    report_lines.append(analyze_agent_vs_model(all_results))
    report_lines.append("")

    # 3. 错误类型分类
    report_lines.append(analyze_error_types(all_results))
    report_lines.append("")

    # 4. Reasoning长度 vs 命中率
    report_lines.append(analyze_reasoning_length(all_results))
    report_lines.append("")

    # 5. 过度自信模式
    report_lines.append(analyze_overconfident(all_results))
    report_lines.append("")

    # 6. 总体统计摘要
    report_lines.append("## 📊 总体统计摘要\n")
    total = len(all_results)
    correct = sum(1 for r in all_results if r['score'] >= 0.5)
    wrong = sum(1 for r in all_results if r['score'] == 0.0)
    avg_ret = sum(r['return'] for r in all_results) / total
    avg_score = sum(r['score'] for r in all_results) / total
    report_lines.append(f"- **总推荐数**: {total}")
    report_lines.append(f"- **命中率**: {correct}/{total} = {correct/total:.0%}")
    report_lines.append(f"- **完全错误**: {wrong}/{total} = {wrong/total:.0%}")
    report_lines.append(f"- **平均收益率**: {avg_ret:+.2%}")
    report_lines.append(f"- **平均得分**: {avg_score:.2f}")
    report_lines.append("")

    report_lines.append("---")
    report_lines.append(f"*由 anti_pattern.py 自动生成 | {now_str}*")

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
