#!/usr/bin/env python3
"""
Layer 2 置信度校准分析器 — 评估confidence字段的预测质量

功能：
  1. 加载recommendations.json，用yfinance对pending推荐评分（5天前瞻）
  2. 按置信度分桶（0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-1.0）
     分析每桶的命中率、平均收益率
  3. 检验校准性：高置信度是否=高命中率？不一致则标记反转
  4. 规则模式分析：R001-R006在高/低置信度中的分布，agent_applied vs auto_inferred差异
  5. 输出中文markdown表格到stdout和文件

用法：
  python3 calibration.py              # 完整分析（评分+分析）
  python3 calibration.py --dry-run    # 只看不写文件
  python3 calibration.py --skip-score # 跳过评分，使用已有数据

依赖：yfinance, json, os, sys, datetime, collections
"""
import json, os, sys, time, argparse, re
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'calibration-analysis.md')

MODEL_HORIZON_DAYS = 5
MODEL_THRESHOLD = 0.02  # 2%

# 置信度分桶
CONFIDENCE_BUCKETS = [
    (0.0, 0.5, '低 (<0.5)'),
    (0.5, 0.6, '中低 (0.5-0.6)'),
    (0.6, 0.7, '中等 (0.6-0.7)'),
    (0.7, 0.8, '中高 (0.7-0.8)'),
    (0.8, 1.01, '高 (0.8-1.0)'),
]

SPECIAL_TICKERS = {
    'SPY': 'SPY', 'QQQ': 'QQQ', 'VIX': '^VIX',
    'IWM': 'IWM', 'DIA': 'DIA', 'Semiconductor': 'SOXX',
}


# ============================================================
# 工具函数
# ============================================================
def is_valid_number(val) -> bool:
    """检查值是否为有效数字（非None/NaN/Inf）"""
    if val is None:
        return False
    import math
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return False
    return True


def parse_date(date_str: str):
    """兼容 %Y%m%d 和 %Y-%m-%d 两种日期格式"""
    if not date_str:
        return None
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


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


def get_price_on_date(ticker: str, date_str: str) -> float | None:
    """获取指定日期的收盘价"""
    try:
        dt = parse_date(date_str)
        if dt is None:
            return None
        start = (dt - timedelta(days=3)).strftime('%Y-%m-%d')
        end = (dt + timedelta(days=5)).strftime('%Y-%m-%d')
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=end)
        if hist.empty or not is_valid_number(float(hist.iloc[0]['Close'])):
            return None
        target_date = dt.date()
        for d in sorted(hist.index, reverse=True):
            if d.date() <= target_date:
                price = float(hist.loc[d, 'Close'])
                if is_valid_number(price):
                    return price
        first_price = float(hist.iloc[0]['Close'])
        return first_price if is_valid_number(first_price) else None
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
            price = float(hist.loc[d, 'Close'])
            if is_valid_number(price):
                return price, d.strftime('%Y-%m-%d')
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


def is_hit(score: float) -> bool:
    """score >= 0.5 算命中"""
    return score >= 0.5


# ============================================================
# 主逻辑
# ============================================================
def load_recommendations():
    """加载推荐数据"""
    with open(TRACK_FILE) as f:
        data = json.load(f)
    return data.get('recommendations', [])


def score_pending(recs, skip=False):
    """对pending推荐评分（复用auto_scorer逻辑）"""
    if skip:
        # 使用已有评分数据
        scored = []
        for r in recs:
            if r.get('score') is not None:
                scored.append(r)
        print(f"📋 使用已有评分数据: {len(scored)} 条")
        return scored

    today_str = datetime.now().strftime('%Y%m%d')
    pending = []
    for r in recs:
        if r.get('score') is not None:
            continue  # 已评分，跳过
        if r.get('target') in ('大盘', 'S&P 500'):
            continue  # 跳过宏观推荐
        rec_date = r.get('date', '')
        if rec_date >= today_str:
            continue  # 未到期
        if not rec_date:
            continue
        pending.append(r)

    if not pending:
        print("✅ 没有需要评分的建议")
        return [r for r in recs if r.get('score') is not None]

    print(f"📊 待评分: {len(pending)} 条 (模型周期: {MODEL_HORIZON_DAYS}天, 阈值: {MODEL_THRESHOLD:.0%})\n")

    scored = list(recs)  # 复制
    scored_count = 0

    for i, rec in enumerate(pending):
        target = rec.get('target', '')
        date = rec.get('date', '')
        direction = rec.get('direction', '')
        ticker = to_yahoo_ticker(target)

        print(f"  [{i+1}/{len(pending)}] {target}({ticker}) {direction} conf={rec.get('confidence', 0):.0%} ...", end=' ')

        base_price = get_price_on_date(ticker, date)
        if base_price is None:
            print("❌ 无法获取基准价格")
            time.sleep(0.5)
            continue

        price_5d, date_5d = get_price_after_days(ticker, date, MODEL_HORIZON_DAYS)
        if price_5d is None:
            print("❌ 无法获取5天后价格")
            time.sleep(0.5)
            continue

        ret_5d = (price_5d - base_price) / base_price
        score, outcome = score_result(direction, ret_5d)

        emoji = '✅' if score >= 0.5 else '⚠️' if score > 0 else '❌'
        print(f"{emoji} ret={ret_5d:+.2%} score={score} → {outcome}")

        # 写回数据
        for r in scored:
            if r.get('id') == rec.get('id'):
                r['score'] = score
                r['outcome'] = outcome
                r['outcome_date'] = datetime.now().strftime('%Y-%m-%d')
                r['actual_return'] = f"{ret_5d:+.2%}"
                r['scoring_detail'] = {
                    'base_price': base_price,
                    'next_price': price_5d,
                    'next_date': date_5d,
                    'horizon_days': MODEL_HORIZON_DAYS,
                    'threshold': MODEL_THRESHOLD,
                    'ret_5d': ret_5d,
                }
                break

        scored_count += 1
        time.sleep(0.5)

    # 写回文件
    if scored_count > 0:
        data = {'recommendations': scored, 'meta': {'version': 2}}
        with open(TRACK_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已评分 {scored_count} 条，已写入 {TRACK_FILE}")
    else:
        print("\n⚠️ 无新评分")

    return scored


def calibration_analysis(scored_recs):
    """置信度校准分析"""
    results = []

    for rec in scored_recs:
        if rec.get('score') is None:
            continue
        # Handle scoring_detail as string or dict
        sd = rec.get('scoring_detail', {})
        if isinstance(sd, dict):
            ret_5d = sd.get('ret_5d', 0)
        elif isinstance(sd, str):
            m = re.search(r'ret=([+-]?[\d.]+)', sd)
            ret_5d = (float(m.group(1)) / 100) if m else 0
        else:
            ret_5d = 0
        # Fallback to actual_return if ret_5d is 0
        if ret_5d == 0:
            ar = rec.get('actual_return')
            if ar is not None:
                ret_5d = float(ar) / 100
        results.append({
            'id': rec.get('id', ''),
            'target': rec.get('target', ''),
            'confidence': rec.get('confidence', 0),
            'score': rec['score'],
            'outcome': rec.get('outcome', ''),
            'direction': rec.get('direction', ''),
            'ret_5d': ret_5d,
            'rules_applied': rec.get('rules_applied', []),
            'rules_source': rec.get('rules_source', 'unknown'),
        })

    if not results:
        return None

    # 按置信度分桶
    buckets = defaultdict(list)
    for r in results:
        conf = r['confidence']
        for low, high, label in CONFIDENCE_BUCKETS:
            if low <= conf < high:
                buckets[label].append(r)
                break

    # 计算每桶统计
    bucket_stats = []
    for low, high, label in CONFIDENCE_BUCKETS:
        items = buckets.get(label, [])
        if not items:
            bucket_stats.append({
                'label': label,
                'count': 0,
                'hit_rate': 0,
                'avg_return': 0,
                'avg_confidence': 0,
            })
            continue

        count = len(items)
        hits = sum(1 for r in items if is_hit(r['score']))
        avg_ret = sum(r['ret_5d'] for r in items) / count
        avg_conf = sum(r['confidence'] for r in items) / count
        hit_rate = hits / count

        bucket_stats.append({
            'label': label,
            'count': count,
            'hits': hits,
            'hit_rate': hit_rate,
            'avg_return': avg_ret,
            'avg_confidence': avg_conf,
        })

    return {
        'total': len(results),
        'overall_hit_rate': sum(1 for r in results if is_hit(r['score'])) / len(results),
        'overall_avg_return': sum(r['ret_5d'] for r in results) / len(results),
        'buckets': bucket_stats,
        'results': results,
    }


def check_calibration(bucket_stats):
    """检查校准性：高置信度是否=高命中率"""
    valid = [b for b in bucket_stats if b['count'] > 0]
    if len(valid) < 2:
        return True, "数据不足，无法校验"

    # 检查命中率是否随置信度单调递增
    hit_rates = [b['hit_rate'] for b in valid]
    is_calibrated = True
    inversions = []

    for i in range(len(valid) - 1):
        if hit_rates[i] > hit_rates[i + 1] + 0.05:  # 容忍5%噪声
            is_calibrated = False
            inversions.append((valid[i]['label'], valid[i + 1]['label']))

    if is_calibrated:
        return True, "校准良好: 高置信度=高命中率"
    else:
        inv_str = ' → '.join([f"{a}({b:.0%})" for a, b in zip([v['label'] for v in valid], hit_rates)])
        return False, f"校准异常: 命中率未随置信度单调递增\n    详情: {inv_str}"


def rules_analysis(results):
    """规则模式分析"""
    if not results:
        return {}

    # 1. 各规则在不同置信度区间的出现频率
    rule_confidence = defaultdict(lambda: {'high': 0, 'mid': 0, 'low': 0, 'total': 0})
    for r in results:
        conf = r['confidence']
        level = 'high' if conf >= 0.7 else 'mid' if conf >= 0.5 else 'low'
        for rule in r['rules_applied']:
            rule_confidence[rule][level] += 1
            rule_confidence[rule]['total'] += 1

    # 2. 各规则的命中率
    rule_outcomes = defaultdict(lambda: {'hits': 0, 'total': 0})
    for r in results:
        for rule in r['rules_applied']:
            rule_outcomes[rule]['total'] += 1
            if is_hit(r['score']):
                rule_outcomes[rule]['hits'] += 1

    # 3. agent_applied vs auto_inferred
    source_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'returns': []})
    for r in results:
        src = r['rules_source']
        source_stats[src]['total'] += 1
        source_stats[src]['returns'].append(r['ret_5d'])
        if is_hit(r['score']):
            source_stats[src]['hits'] += 1

    return {
        'rule_confidence': dict(rule_confidence),
        'rule_outcomes': dict(rule_outcomes),
        'source_stats': {k: {**v, 'avg_return': sum(v['returns']) / len(v['returns']) if v['returns'] else 0}
                         for k, v in source_stats.items()},
    }


def generate_markdown(cal, cal_ok, cal_msg, rules_data):
    """生成markdown报告"""
    lines = []
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines.append(f"# 🎯 置信度校准分析报告")
    lines.append(f"")
    lines.append(f"生成时间: {now}")
    lines.append(f"")

    if cal is None:
        lines.append("⚠️ 没有已评分的推荐数据，无法进行分析。")
        lines.append("")
        lines.append("请先运行 `python3 auto_scorer.py` 对推荐进行评分。")
        return '\n'.join(lines)

    # 总览
    lines.append(f"## 📊 总览")
    lines.append(f"")
    lines.append(f"- 总评分数: **{cal['total']}** 条")
    lines.append(f"- 整体命中率: **{cal['overall_hit_rate']:.1%}** (score≥0.5)")
    lines.append(f"- 平均5天收益率: **{cal['overall_avg_return']:+.2%}**")
    lines.append(f"")

    # 校准性检验
    lines.append(f"## 🔍 校准性检验")
    lines.append(f"")
    if cal_ok:
        lines.append(f"✅ **{cal_msg}**")
    else:
        lines.append(f"⚠️ **{cal_msg}**")
    lines.append(f"")

    # 分桶统计表
    lines.append(f"## 📋 置信度分桶统计")
    lines.append(f"")
    lines.append(f"| 置信度区间 | 数量 | 命中数 | 命中率 | 平均收益率 | 平均置信度 |")
    lines.append(f"|:---:|:---:|:---:|:---:|:---:|:---:|")

    for b in cal['buckets']:
        if b['count'] == 0:
            lines.append(f"| {b['label']} | - | - | - | - | - |")
        else:
            lines.append(
                f"| {b['label']} | {b['count']} | {b.get('hits', 0)} | "
                f"{b['hit_rate']:.1%} | {b['avg_return']:+.2%} | {b['avg_confidence']:.2f} |"
            )
    lines.append(f"")

    # 规则分析
    if rules_data:
        lines.append(f"## 📐 规则模式分析")
        lines.append(f"")

        # 各规则命中率
        lines.append(f"### 各规则命中率")
        lines.append(f"")
        lines.append(f"| 规则 | 命中数 | 总数 | 命中率 |")
        lines.append(f"|:---:|:---:|:---:|:---:|")

        for rule in sorted(rules_data.get('rule_outcomes', {}).keys()):
            stats = rules_data['rule_outcomes'][rule]
            hit_rate = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
            lines.append(f"| {rule} | {stats['hits']} | {stats['total']} | {hit_rate:.1%} |")
        lines.append(f"")

        # 规则在不同置信度区间的分布
        lines.append(f"### 规则在不同置信度区间的分布")
        lines.append(f"")
        lines.append(f"| 规则 | 低置信度(<0.5) | 中置信度(0.5-0.7) | 高置信度(≥0.7) | 总计 |")
        lines.append(f"|:---:|:---:|:---:|:---:|:---:|")

        for rule in sorted(rules_data.get('rule_confidence', {}).keys()):
            rc = rules_data['rule_confidence'][rule]
            lines.append(f"| {rule} | {rc['low']} | {rc['mid']} | {rc['high']} | {rc['total']} |")
        lines.append(f"")

        # agent_applied vs auto_inferred
        lines.append(f"### 来源对比: agent_applied vs auto_inferred")
        lines.append(f"")
        lines.append(f"| 来源 | 命中率 | 平均收益率 | 样本数 |")
        lines.append(f"|:---:|:---:|:---:|:---:|")

        for src in sorted(rules_data.get('source_stats', {}).keys()):
            s = rules_data['source_stats'][src]
            hit_rate = s['hits'] / s['total'] if s['total'] > 0 else 0
            label = 'Agent自主应用' if src == 'agent_applied' else '系统自动推断'
            lines.append(f"| {label} ({src}) | {hit_rate:.1%} | {s['avg_return']:+.2%} | {s['total']} |")
        lines.append(f"")

    # 推荐详情
    lines.append(f"## 📝 推荐评分明细")
    lines.append(f"")
    lines.append(f"| ID | 标的 | 方向 | 置信度 | 评分 | 结果 | 5天收益率 | 规则来源 |")
    lines.append(f"|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in cal['results']:
        emoji = '✅' if is_hit(r['score']) else '❌'
        rules_src = r.get('rules_source', 'unknown')[:6]
        lines.append(
            f"| {r['id'][:8]} | {r['target'][:12]} | {r['direction']} | "
            f"{r['confidence']:.2f} | {r['score']:.2f} | {emoji} | "
            f"{r['ret_5d']:+.2%} | {rules_src} |"
        )
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"*分析逻辑: 5天前瞻收益率>2%为命中(bullish), <-2%为命中(bearish)*")
    lines.append(f"*校准标准: 高置信度桶命中率应≥低置信度桶命中率*")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Layer 2 置信度校准分析')
    parser.add_argument('--dry-run', action='store_true', help='只分析不写文件')
    parser.add_argument('--skip-score', action='store_true', help='跳过评分，使用已有数据')
    args = parser.parse_args()

    dry_run = args.dry_run
    skip_score = args.skip_score

    print("=" * 70)
    print("🎯 Layer 2 置信度校准分析器")
    print("=" * 70)
    print()

    # 1. 加载数据
    print("📁 加载推荐数据...")
    recs = load_recommendations()
    total = len(recs)
    pending = sum(1 for r in recs if r.get('status') == 'pending')
    scored = sum(1 for r in recs if r.get('score') is not None)
    print(f"  总计: {total} 条, pending: {pending} 条, 已评分: {scored} 条")
    print()

    # 2. 评分
    if skip_score:
        print("⏭️ 跳过评分，使用已有数据\n")
    else:
        print("📊 开始评分...\n")
    scored_recs = score_pending(recs, skip=skip_score)

    # 3. 置信度校准分析
    print("\n🔍 执行置信度校准分析...\n")
    cal = calibration_analysis(scored_recs)

    if cal is None:
        print("⚠️ 没有已评分的推荐数据，无法分析")
        print("  请先运行 python3 auto_scorer.py 进行评分")
        # 仍然输出文件（带提示）
        md = generate_markdown(None, False, "无数据", None)
        if not dry_run:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(OUTPUT_FILE, 'w') as f:
                f.write(md)
            print(f"\n📄 已保存: {OUTPUT_FILE}")
        print(md)
        return

    # 4. 校准性检验
    cal_ok, cal_msg = check_calibration(cal['buckets'])
    print(f"{'✅' if cal_ok else '⚠️'} 校准性: {cal_msg}\n")

    # 5. 规则分析
    rules_data = rules_analysis(cal['results'])

    # 6. 生成报告
    md = generate_markdown(cal, cal_ok, cal_msg, rules_data)

    # 输出到stdout
    print("\n" + "=" * 70)
    print(md)

    # 保存文件
    if not dry_run:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            f.write(md)
        print(f"\n📄 已保存: {OUTPUT_FILE}")
    else:
        print(f"\n🔍 Dry run完成，未写入文件")


if __name__ == '__main__':
    main()
