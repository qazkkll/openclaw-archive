#!/usr/bin/env python3
"""
市场Regime分析器 — Layer 2
分析不同市场环境(牛/熊/震荡)下推荐的准确度

功能：
  1. 各regime下的命中率
  2. Regime切换时推荐准确度
  3. VIX区间 vs 推荐命中率
  4. 宏观事件(停火/加息/降息)对推荐准确度的影响

用法：
  python3 regime_analysis.py                # 分析已评分的推荐
  python3 regime_analysis.py --dry-run      # 只分析不写文件
  python3 regime_analysis.py --include-pending  # 用yfinance实时评分pending推荐再分析
"""
import json, os, sys, re, time
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
OUTPUT_FILE = os.path.join(ROOT, 'data/backtest-rounds/regime-analysis.md')

# === Yahoo Finance ticker mapping ===
SPECIAL_TICKERS = {
    'SPY': 'SPY', 'QQQ': 'QQQ', 'VIX': '^VIX',
    'IWM': 'IWM', 'DIA': 'DIA', 'Semiconductor': 'SOXX',
    'S&P 500': 'SPY',
}

MODEL_HORIZON_DAYS = 5
MODEL_THRESHOLD = 0.02


def to_yahoo_ticker(target: str) -> str:
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


def parse_date(date_str: str) -> str | None:
    """解析两种日期格式: YYYYMMDD 和 YYYY-MM-DD"""
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def get_price_on_date(ticker: str, date_str: str) -> float | None:
    """获取指定日期的收盘价"""
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
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
    except Exception:
        return None


def get_price_after_days(ticker: str, date_str: str, days: int = 5) -> tuple[float | None, str | None]:
    """获取date_str之后N个交易日的收盘价"""
    try:
        dt = datetime.strptime(date_str, '%Y%m%d')
        start = dt.strftime('%Y-%m-%d')
        end = (dt + timedelta(days=days + 10)).strftime('%Y-%m-%d')
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
    except Exception:
        return None, None


def score_recommendation(rec: dict) -> dict | None:
    """用yfinance实时评分一条recommendation"""
    target = rec.get('target', '')
    date = rec.get('date', '')
    direction = rec.get('direction', '')
    if not date or not target or target == '大盘':
        return None

    ticker = to_yahoo_ticker(target)
    base_price = get_price_on_date(ticker, date)
    if base_price is None:
        return None

    price_5d, date_5d = get_price_after_days(ticker, date, MODEL_HORIZON_DAYS)
    if price_5d is None:
        return None

    ret_5d = (price_5d - base_price) / base_price

    if direction == 'bullish':
        if ret_5d > MODEL_THRESHOLD:
            score = 1.0
        elif ret_5d > 0:
            score = 0.5
        elif ret_5d > -MODEL_THRESHOLD:
            score = 0.25
        else:
            score = 0.0
    elif direction == 'bearish':
        if ret_5d < -MODEL_THRESHOLD:
            score = 1.0
        elif ret_5d < 0:
            score = 0.5
        elif ret_5d < MODEL_THRESHOLD:
            score = 0.25
        else:
            score = 0.0
    else:  # neutral
        if abs(ret_5d) < MODEL_THRESHOLD:
            score = 1.0
        elif abs(ret_5d) < 0.05:
            score = 0.5
        else:
            score = 0.0

    return {
        'score': score,
        'actual_return_pct': ret_5d,
        'actual_return': f"{ret_5d:+.2%}",
        'outcome': 'correct' if score >= 0.5 else 'wrong',
        'base_price': base_price,
        'next_price': price_5d,
    }


# ============================================================
# Regime 解析
# ============================================================

# regime关键词映射
REGIME_KEYWORDS = {
    'bear':    ['熊', 'bear', '熊市', '下跌趋势', '空头'],
    'bull':    ['牛', 'bull', '牛市', '上涨趋势', '多头', '强势', '拉涨', '创新高'],
    'sideways': ['震荡', '横盘', 'sideways', '分化', '分化严重', '盘整'],
    'dovish':  ['鸽', 'dovish', '鸽派', '降息', '宽松', '放水'],
    'hawkish': ['鹰', 'hawk', 'hawkish', '鹰派', '加息', '紧缩'],
}

# 宏观事件关键词
MACRO_EVENTS = {
    'ceasefire': ['停火', 'ceasefire', 'peace deal', '和平协议'],
    'rate_hike': ['加息', 'rate hike', 'rate increase', '紧缩'],
    'rate_cut':  ['降息', 'rate cut', 'rate decrease', '宽松', 'dovish pivot'],
    'fed_hawk':  ['Fed鹰', 'Fed hawk', 'hawkish', '鹰派转向', 'Warsh'],
    'oil_crash': ['油价暴跌', 'oil crash', 'oil -', '油价-'],
    'pce_data':  ['PCE', 'pce'],
}

# VIX区间
VIX_RANGES = [
    ('<15 (低恐慌)', 0, 15),
    ('15-20 (正常)', 15, 20),
    ('20-25 (偏高)', 20, 25),
    ('25-30 (高)', 25, 30),
    ('>30 (极端)', 30, 999),
]


def parse_regime(macro_context: str) -> str:
    """从macro_context中解析市场regime"""
    if not macro_context:
        return 'unknown'

    text = macro_context.lower()

    # 优先检查显式regime声明
    if 'regime' in text:
        if 'bear' in text or '熊' in text:
            return 'bear'
        if 'bull' in text or '牛' in text:
            return 'bull'
        if 'sideways' in text or '横盘' in text or '震荡' in text:
            return 'sideways'

    # 关键词匹配（按优先级）
    scores = {'bear': 0, 'bull': 0, 'sideways': 0, 'dovish': 0, 'hawkish': 0}
    for regime, keywords in REGIME_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                scores[regime] += 1

    # 取最高分的regime
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return 'unknown'


def parse_vix_level(macro_context: str) -> float | None:
    """从macro_context中提取VIX水平"""
    if not macro_context:
        return None
    # 匹配 "VIX 16.4" 或 "VIX:16.4" 或 "vix=16.4"
    patterns = [
        r'VIX\s*[=:]\s*(\d+\.?\d*)',
        r'vix\s+(\d+\.?\d*)',
        r'VIX\s+(\d+\.?\d*)',
    ]
    for pat in patterns:
        m = re.search(pat, macro_context, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def parse_macro_events(macro_context: str) -> list[str]:
    """从macro_context中识别宏观事件"""
    if not macro_context:
        return []
    events = []
    text = macro_context.lower()
    for event, keywords in MACRO_EVENTS.items():
        for kw in keywords:
            if kw.lower() in text:
                events.append(event)
                break
    return events


def detect_regime_change(recs: list[dict]) -> dict:
    """检测regime切换点"""
    changes = {}
    sorted_recs = sorted(recs, key=lambda r: r.get('date', ''))
    prev_regime = None
    for rec in sorted_recs:
        regime = parse_regime(rec.get('macro_context', ''))
        if prev_regime and regime != prev_regime and regime != 'unknown':
            date = rec.get('date', 'unknown')
            changes[date] = {'from': prev_regime, 'to': regime}
        if regime != 'unknown':
            prev_regime = regime
    return changes


# ============================================================
# 分析逻辑
# ============================================================

def analyze_by_regime(scored_recs: list[dict]) -> dict:
    """按regime分组分析命中率"""
    groups = defaultdict(list)
    for rec in scored_recs:
        regime = parse_regime(rec.get('macro_context', ''))
        groups[regime].append(rec)

    results = {}
    for regime, recs in sorted(groups.items()):
        total = len(recs)
        correct = sum(1 for r in recs if r.get('score', 0) >= 0.5)
        avg_score = sum(r.get('score', 0) for r in recs) / total if total else 0
        avg_return = sum(r.get('_ret', 0) for r in recs) / total if total else 0
        results[regime] = {
            'total': total,
            'correct': correct,
            'hit_rate': correct / total * 100 if total else 0,
            'avg_score': avg_score,
            'avg_return': avg_return,
        }
    return results


def analyze_vix_vs_hit(scored_recs: list[dict]) -> dict:
    """VIX区间 vs 推荐命中率"""
    groups = defaultdict(list)
    for rec in scored_recs:
        vix = parse_vix_level(rec.get('macro_context', ''))
        if vix is None:
            continue
        for label, lo, hi in VIX_RANGES:
            if lo <= vix < hi:
                groups[label].append(rec)
                break

    results = {}
    for label, recs in groups.items():
        total = len(recs)
        correct = sum(1 for r in recs if r.get('score', 0) >= 0.5)
        avg_score = sum(r.get('score', 0) for r in recs) / total if total else 0
        results[label] = {
            'total': total,
            'correct': correct,
            'hit_rate': correct / total * 100 if total else 0,
            'avg_score': avg_score,
        }
    return results


def analyze_macro_events(scored_recs: list[dict]) -> dict:
    """宏观事件对推荐准确度的影响"""
    groups = defaultdict(list)
    for rec in scored_recs:
        events = parse_macro_events(rec.get('macro_context', ''))
        if not events:
            groups['无宏观事件'].append(rec)
        for event in events:
            groups[event].append(rec)

    results = {}
    for event, recs in groups.items():
        total = len(recs)
        correct = sum(1 for r in recs if r.get('score', 0) >= 0.5)
        avg_score = sum(r.get('score', 0) for r in recs) / total if total else 0
        avg_return = sum(r.get('_ret', 0) for r in recs) / total if total else 0
        results[event] = {
            'total': total,
            'correct': correct,
            'hit_rate': correct / total * 100 if total else 0,
            'avg_score': avg_score,
            'avg_return': avg_return,
        }
    return results


def analyze_regime_changes(scored_recs: list[dict]) -> dict:
    """Regime切换时的推荐准确度"""
    sorted_recs = sorted(scored_recs, key=lambda r: r.get('date', ''))
    change_dates = detect_regime_change(scored_recs)

    # 切换前后3天的推荐
    results = {'before_switch': [], 'after_switch': [], 'during_switch': []}
    for rec in sorted_recs:
        date = rec.get('date', '')
        if date in change_dates:
            results['during_switch'].append(rec)
        else:
            # 检查是否在切换日期附近
            near_switch = False
            for cd in change_dates:
                if not date or not cd:
                    continue
                try:
                    d1 = datetime.strptime(date, '%Y%m%d')
                    d2 = datetime.strptime(cd, '%Y%m%d')
                    if abs((d1 - d2).days) <= 3:
                        near_switch = True
                        break
                except ValueError:
                    pass
            if near_switch:
                regime = parse_regime(rec.get('macro_context', ''))
                if regime != 'unknown':
                    results['after_switch'].append(rec)
                else:
                    results['before_switch'].append(rec)

    summary = {}
    for phase, recs in results.items():
        if not recs:
            continue
        total = len(recs)
        correct = sum(1 for r in recs if r.get('score', 0) >= 0.5)
        avg_score = sum(r.get('score', 0) for r in recs) / total if total else 0
        summary[phase] = {
            'total': total,
            'correct': correct,
            'hit_rate': correct / total * 100 if total else 0,
            'avg_score': avg_score,
        }
    return summary


# ============================================================
# Markdown 输出
# ============================================================

def build_markdown(scored_recs, regime_results, vix_results, event_results,
                    change_results, meta) -> str:
    """构建markdown报告"""
    lines = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')

    total = len(scored_recs)
    correct = sum(1 for r in scored_recs if r.get('score', 0) >= 0.5)
    overall_rate = correct / total * 100 if total else 0

    lines.append(f"# 市场Regime分析报告")
    lines.append(f"")
    lines.append(f"> 生成时间: {now_str} | 数据量: {total}条 | 总命中率: {correct}/{total} ({overall_rate:.1f}%)")
    if meta.get('dry_run'):
        lines.append(f"> ⚠️ Dry-run模式 — 数据未写入文件")
    lines.append(f"")

    # === 1. 各Regime命中率 ===
    lines.append(f"## 1. 各Regime下的命中率")
    lines.append(f"")
    lines.append(f"| Regime | 数量 | 命中 | 命中率 | 平均分 | 平均收益率 |")
    lines.append(f"|--------|------|------|--------|--------|------------|")
    regime_order = ['bull', 'bear', 'sideways', 'dovish', 'hawkish', 'unknown']
    regime_labels = {
        'bull': '🐂 牛市',
        'bear': '🐻 熊市',
        'sideways': '📊 震荡',
        'dovish': '🕊️ 鸽派',
        'hawkish': '🦅 鹰派',
        'unknown': '❓ 未知',
    }
    for regime in regime_order:
        if regime in regime_results:
            r = regime_results[regime]
            label = regime_labels.get(regime, regime)
            lines.append(
                f"| {label} | {r['total']} | {r['correct']} | "
                f"{r['hit_rate']:.1f}% | {r['avg_score']:.2f} | "
                f"{r['avg_return']:+.2%} |"
            )
    lines.append(f"")

    # === 2. Regime切换时准确度 ===
    lines.append(f"## 2. Regime切换时推荐准确度")
    lines.append(f"")
    lines.append(f"| 阶段 | 数量 | 命中 | 命中率 | 平均分 |")
    lines.append(f"|------|------|------|--------|--------|")
    phase_labels = {
        'before_switch': '切换前(旧regime)',
        'during_switch': '切换当天',
        'after_switch': '切换后(新regime)',
    }
    for phase in ['before_switch', 'during_switch', 'after_switch']:
        if phase in change_results:
            r = change_results[phase]
            label = phase_labels.get(phase, phase)
            lines.append(
                f"| {label} | {r['total']} | {r['correct']} | "
                f"{r['hit_rate']:.1f}% | {r['avg_score']:.2f} |"
            )
    if not change_results:
        lines.append(f"| (无regime切换数据) | - | - | - | - |")
    lines.append(f"")

    # === 3. VIX区间 vs 命中率 ===
    lines.append(f"## 3. VIX区间 vs 推荐命中率")
    lines.append(f"")
    if vix_results:
        lines.append(f"| VIX区间 | 数量 | 命中 | 命中率 | 平均分 |")
        lines.append(f"|---------|------|------|--------|--------|")
        for label in [r[0] for r in VIX_RANGES]:
            if label in vix_results:
                r = vix_results[label]
                lines.append(
                    f"| {label} | {r['total']} | {r['correct']} | "
                    f"{r['hit_rate']:.1f}% | {r['avg_score']:.2f} |"
                )
    else:
        lines.append(f"> ℹ️ 当前数据中未提取到VIX数值，需要更多包含VIX的macro_context数据")
    lines.append(f"")

    # === 4. 宏观事件影响 ===
    lines.append(f"## 4. 宏观事件对推荐准确度的影响")
    lines.append(f"")
    lines.append(f"| 宏观事件 | 数量 | 命中 | 命中率 | 平均分 | 平均收益率 |")
    lines.append(f"|----------|------|------|--------|--------|------------|")
    event_labels = {
        'ceasefire': '🕊️ 停火',
        'rate_hike': '📈 加息',
        'rate_cut': '📉 降息',
        'fed_hawk': '🦅 Fed鹰派',
        'oil_crash': '⛽ 油价暴跌',
        'pce_data': '📊 PCE数据',
        '无宏观事件': '➖ 无',
    }
    for event, r in sorted(event_results.items(), key=lambda x: -x[1]['total']):
        label = event_labels.get(event, event)
        lines.append(
            f"| {label} | {r['total']} | {r['correct']} | "
            f"{r['hit_rate']:.1f}% | {r['avg_score']:.2f} | "
            f"{r['avg_return']:+.2%} |"
        )
    lines.append(f"")

    # === 5. 详细推荐列表 ===
    lines.append(f"## 5. 各推荐Regime分类明细")
    lines.append(f"")
    lines.append(f"| ID | 日期 | 标的 | 方向 | Regime | VIX | 得分 | 收益率 |")
    lines.append(f"|-----|------|------|------|--------|-----|------|--------|")
    for rec in sorted(scored_recs, key=lambda r: r.get('date', '')):
        rid = rec.get('id', '?')
        date = rec.get('date', '?')
        target = rec.get('target', '?')
        direction = rec.get('direction', '?')
        regime = parse_regime(rec.get('macro_context', ''))
        regime_label = regime_labels.get(regime, regime)
        vix = parse_vix_level(rec.get('macro_context', ''))
        vix_str = f"{vix:.1f}" if vix else '-'
        score = rec.get('score', 0)
        ret = rec.get('_ret', 0)
        score_emoji = '✅' if score >= 0.5 else '⚠️' if score > 0 else '❌'
        lines.append(
            f"| {rid} | {date} | {target} | {direction} | "
            f"{regime_label} | {vix_str} | {score_emoji}{score:.2f} | {ret:+.2%} |"
        )
    lines.append(f"")

    # === 6. 结论与建议 ===
    lines.append(f"## 6. 结论与建议")
    lines.append(f"")
    eligible = [(k, v) for k, v in regime_results.items() if v['total'] >= 2]
    best_regime = max(eligible, key=lambda x: x[1]['hit_rate'], default=None) if eligible else None
    worst_regime = min(eligible, key=lambda x: x[1]['hit_rate'], default=None) if eligible else None
    if best_regime is not None:
        br_k, br_v = best_regime
        lines.append(f"- **最佳表现regime**: {regime_labels.get(br_k, br_k)} "
                      f"({br_v['hit_rate']:.1f}%, {br_v['total']}条)")
    if worst_regime is not None:
        wr_k, wr_v = worst_regime
        lines.append(f"- **最差表现regime**: {regime_labels.get(wr_k, wr_k)} "
                      f"({wr_v['hit_rate']:.1f}%, {wr_v['total']}条)")
    if change_results:
        sw = change_results.get('during_switch', {})
        if sw:
            lines.append(f"- **切换当天命中率**: {sw.get('hit_rate', 0):.1f}% "
                          f"(建议切换期间降低仓位)")
    if event_results:
        for evt, r in event_results.items():
            if r['total'] >= 2 and r['hit_rate'] < 40:
                label = event_labels.get(evt, evt)
                lines.append(f"- ⚠️ **{label}期间命中率偏低**({r['hit_rate']:.1f}%)，需加强宏观判断")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"*由 regime_analysis.py 自动生成*")
    lines.append(f"")

    return '\n'.join(lines)


# ============================================================
# 主程序
# ============================================================

def main():
    dry_run = '--dry-run' in sys.argv
    include_pending = '--include-pending' in sys.argv

    print("=" * 60)
    print("🔍 市场Regime分析器 — Layer 2")
    print("=" * 60)

    # 加载数据
    with open(TRACK_FILE) as f:
        data = json.load(f)

    recs = data.get('recommendations', [])

    # 分离已评分和待评分
    scored = [r for r in recs if r.get('status') == 'scored' and r.get('score') is not None]
    pending = [r for r in recs if r.get('status') == 'pending']

    print(f"\n📊 数据概览:")
    print(f"   总推荐: {len(recs)}")
    print(f"   已评分: {len(scored)}")
    print(f"   待评分: {len(pending)}")

    # 如果 --include-pending，实时评分pending推荐
    newly_scored = []
    if include_pending and pending:
        print(f"\n⏳ 正在用yfinance实时评分 {len(pending)} 条pending推荐...")
        for i, rec in enumerate(pending):
            target = rec.get('target', '')
            if target == '大盘':
                continue
            if not rec.get('date'):
                continue
            result = score_recommendation(rec)
            if result:
                rec_with_ret = {**rec, '_ret': result['actual_return_pct'], 'score': result['score']}
                newly_scored.append(rec_with_ret)
                emoji = '✅' if result['score'] >= 0.5 else '❌'
                print(f"  {emoji} [{i+1}/{len(pending)}] {target}: "
                      f"{result['actual_return']} (score={result['score']})")
            else:
                print(f"  ⏭️ [{i+1}/{len(pending)}] {target}: 无法获取价格数据")
            time.sleep(0.5)
        print(f"   成功评分: {len(newly_scored)}/{len(pending)}")

    # 合并评分数据（添加_ret字段）
    all_scored = []
    for r in scored:
        actual_ret = r.get('actual_return', '')
        try:
            if isinstance(actual_ret, (int, float)):
                ret_val = float(actual_ret) / 100.0
            elif isinstance(actual_ret, str):
                ret_val = float(actual_ret.replace('%', '')) / 100.0
            else:
                ret_val = 0
        except (ValueError, AttributeError):
            ret_val = 0
        all_scored.append({**r, '_ret': ret_val})

    # 加上新评分的
    all_scored.extend(newly_scored)

    if not all_scored:
        print("\n⚠️ 没有已评分的推荐可供分析。")
        print("   提示: 使用 --include-pending 实时评分pending推荐")
        return

    print(f"\n📈 分析 {len(all_scored)} 条已评分推荐...\n")

    # 执行分析
    regime_results = analyze_by_regime(all_scored)
    vix_results = analyze_vix_vs_hit(all_scored)
    event_results = analyze_macro_events(all_scored)
    change_results = analyze_regime_changes(all_scored)

    meta = {'dry_run': dry_run, 'total': len(all_scored)}

    # 生成Markdown
    md = build_markdown(all_scored, regime_results, vix_results,
                         event_results, change_results, meta)

    # 输出到stdout
    print(md)

    # 保存到文件
    if not dry_run:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"\n✅ 报告已保存至: {OUTPUT_FILE}")
    else:
        print(f"\n🔍 Dry-run完成，未写入文件")
        print(f"   输出文件将为: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
