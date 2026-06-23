#!/usr/bin/env python3
"""
特征分析器 v1 — 分析建议特征vs实际收益的关系
Layer 2 of the investment learning system.

用法：
  python3 feature_analysis.py              # 分析所有已评分+评分pending
  python3 feature_analysis.py --dry-run    # 跳过yfinance调用（用已有scored数据）
  python3 feature_analysis.py --scored-only # 只分析已评分的（不调yfinance）
"""

import json, os, sys, re, time
from datetime import datetime, timedelta
from collections import defaultdict

try:
    import yfinance as yf
except ImportError:
    print("⚠️  yfinance未安装，将使用--scored-only模式")
    yf = None

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
OUTPUT_DIR = os.path.join(ROOT, 'data/backtest-rounds')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'feature-analysis.md')

# === 模型评分参数（与auto_scorer保持一致）===
MODEL_HORIZON_DAYS = 5
MODEL_THRESHOLD = 0.02  # 2%

# === Yahoo Finance ticker mapping ===
SPECIAL_TICKERS = {
    'SPY': 'SPY', 'QQQ': 'QQQ', 'VIX': '^VIX',
    'IWM': 'IWM', 'DIA': 'DIA',
    'Semiconductor': 'SOXX',
}

def to_yahoo_ticker(target: str) -> str:
    """将目标转换为Yahoo Finance ticker"""
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

def parse_date(date_str: str):
    """解析日期，支持 %Y%m%d 和 %Y-%m-%d"""
    if not date_str:
        return None
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

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

def get_price_after_days(ticker: str, date_str: str, days: int = 5) -> tuple[float | None, str | None]:
    """获取date_str之后N个交易日的收盘价"""
    try:
        dt = parse_date(date_str)
        if dt is None:
            return None, None
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
            return float(hist.loc[d, 'Close']), d.strftime('%Y%m%d')
        return None, None
    except Exception as e:
        print(f"  ⚠️ 获取{ticker}后{days}天价格失败: {e}")
        return None, None

def score_one(rec: dict) -> dict | None:
    """对单条recommendation评分（5天周期）"""
    target = rec.get('target', '')
    date = rec.get('date', '')
    direction = rec.get('direction', '')

    if not date or not target or target == '大盘':
        return None

    ticker = to_yahoo_ticker(target)

    base_price = get_price_on_date(ticker, date)
    if base_price is None:
        print(f"  ❌ {target}({ticker}): 无法获取{date}价格")
        return None

    price_5d, date_5d = get_price_after_days(ticker, date, MODEL_HORIZON_DAYS)
    if price_5d is None:
        print(f"  ❌ {target}({ticker}): 无法获取{date}后{MODEL_HORIZON_DAYS}天价格")
        return None

    ret_5d = (price_5d - base_price) / base_price

    # 评分逻辑（与auto_scorer v3一致）
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

    emoji = '✅' if score >= 0.5 else '⚠️' if score > 0 else '❌'
    print(f"  {emoji} {target}({ticker}): {direction} | ${base_price:.2f}→${price_5d:.2f} | ret={ret_5d:+.2%} | score={score}")

    return {
        'score': score,
        'actual_return': ret_5d,
        'outcome': 'correct' if score >= 0.5 else 'wrong',
        'base_price': base_price,
        'next_price': price_5d,
        'next_date': date_5d,
    }

# === 关键词提取 ===
KEYWORDS = {
    'RSI': r'RSI',
    '超卖': r'超卖',
    '超买': r'超买',
    '反弹': r'反弹',
    '趋势': r'趋势',
    '回调': r'回调',
    '动量': r'动量',
    '放量': r'放量',
    '缩量': r'缩量',
    '均线': r'均线|MA\d',
    'MACD': r'MACD',
    'KDJ': r'KDJ',
    '背离': r'背离',
    '突破': r'突破',
    '龙头': r'龙头',
    '板块': r'板块',
    '估值': r'估值',
    '基本面': r'基本面',
    '政策': r'政策',
    '利好': r'利好|利好',
    '利空': r'利空',
    'GG信号': r'GG|G信号',
    '防御': r'防御',
    '科技': r'科技|半导体|AI|芯片',
    '消费': r'消费',
    '医药': r'医药|CRO|疫苗',
    '物流': r'物流',
    '品牌': r'品牌',
    '低估值': r'低估值|低PE',
    '分化': r'分化',
    '熊市': r'熊市|bear',
    '牛市': r'牛市|bull',
}

def extract_keywords(text: str) -> list[str]:
    """从reasoning中提取关键词"""
    if not text:
        return []
    found = []
    for kw, pattern in KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            found.append(kw)
    return found if found else ['其他']

# === 特征分析 ===
def analyze_dimension(items: list[dict], dim_key: str, dim_name: str) -> str:
    """
    分析单个维度 vs 命中率
    items: [{value, score, ret_5d, hit}, ...]
    """
    groups = defaultdict(lambda: {'total': 0, 'hit': 0, 'ret_sum': 0.0, 'scores': []})
    for item in items:
        g = groups[item['value']]
        g['total'] += 1
        g['hit'] += item['hit']
        g['ret_sum'] += item['ret_5d']
        g['scores'].append(item['score'])

    lines = [f"### {dim_name}\n"]
    lines.append("| 分类 | 样本数 | 命中数 | 命中率 | 平均收益率 | 平均得分 |")
    lines.append("|------|--------|--------|--------|------------|----------|")

    for val in sorted(groups.keys(), key=lambda x: -groups[x]['total']):
        g = groups[val]
        hit_rate = g['hit'] / g['total'] * 100 if g['total'] > 0 else 0
        avg_ret = g['ret_sum'] / g['total'] * 100 if g['total'] > 0 else 0
        avg_score = sum(g['scores']) / len(g['scores']) if g['scores'] else 0
        lines.append(f"| {val} | {g['total']} | {g['hit']} | {hit_rate:.1f}% | {avg_ret:+.2f}% | {avg_score:.2f} |")

    lines.append("")
    return "\n".join(lines)

def get_confidence_bin(confidence: float) -> str:
    """将confidence分桶"""
    if confidence < 0.5:
        return "0-0.5 (低)"
    elif confidence < 0.7:
        return "0.5-0.7 (中)"
    else:
        return "0.7-1.0 (高)"

def main():
    dry_run = '--dry-run' in sys.argv
    scored_only = '--scored-only' in sys.argv

    # 加载数据
    if not os.path.exists(TRACK_FILE):
        print(f"❌ 找不到 {TRACK_FILE}")
        sys.exit(1)

    with open(TRACK_FILE) as f:
        data = json.load(f)

    recs = data.get('recommendations', [])
    print(f"📊 加载 {len(recs)} 条建议")

    # 分离已评分和待评分
    scored_recs = [r for r in recs if r.get('status') == 'scored' and r.get('score') is not None]
    pending_recs = [r for r in recs if r.get('status') == 'pending' and r.get('target') != '大盘']

    print(f"  已评分: {len(scored_recs)} 条")
    print(f"  待评分: {len(pending_recs)} 条")

    # 对pending进行评分（如果允许）
    if pending_recs and not dry_run and not scored_only and yf is not None:
        print(f"\n🔄 评分 {len(pending_recs)} 条pending...\n")
        for rec in pending_recs:
            result = score_one(rec)
            if result is not None:
                rec['status'] = 'scored'
                rec['score'] = result['score']
                rec['outcome'] = result['outcome']
                rec['actual_return'] = f"{result['actual_return']:+.2%}"
                rec['scoring_detail'] = {
                    'base_price': result['base_price'],
                    'next_price': result['next_price'],
                    'next_date': result['next_date'],
                    'horizon_days': MODEL_HORIZON_DAYS,
                    'threshold': MODEL_THRESHOLD,
                }
                scored_recs.append(rec)
                time.sleep(0.5)

        # 写回数据
        with open(TRACK_FILE, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 已评分并写入 {len(pending_recs)} 条")
    elif pending_recs and (dry_run or scored_only):
        print(f"\n⏭️  跳过yfinance调用（dry-run/scored-only模式）")
    elif pending_recs and yf is None:
        print(f"\n⚠️  yfinance不可用，跳过评分")

    # 汇总所有已评分数据
    all_scored = [r for r in recs if r.get('status') == 'scored' and r.get('score') is not None]

    if not all_scored:
        print("\n⚠️  没有已评分的数据，无法进行特征分析")
        print("提示：先运行 auto_scorer.py 或去掉 --scored-only 参数")
        sys.exit(0)

    # 构建分析数据集
    analysis_items = []
    for rec in all_scored:
        score = rec.get('score', 0)
        actual_return_raw = rec.get('actual_return', '0%')
        # 解析收益率
        try:
            if isinstance(actual_return_raw, (int, float)):
                ret = float(actual_return_raw) / 100
            else:
                ret = float(str(actual_return_raw).replace('%', '').replace('+', '')) / 100
        except (ValueError, TypeError):
            ret = 0.0

        # 也尝试从scoring_detail获取
        if ret == 0.0 and rec.get('scoring_detail'):
            try:
                base = rec['scoring_detail'].get('base_price', 0)
                nxt = rec['scoring_detail'].get('next_price', 0)
                if base and nxt:
                    ret = (nxt - base) / base
            except:
                pass

        confidence = rec.get('confidence', 0.5) or 0.5
        confidence_bin = get_confidence_bin(confidence)
        source = rec.get('source', 'unknown')
        market = rec.get('market', 'unknown')
        direction = rec.get('direction', 'unknown')
        reasoning = rec.get('reasoning', '')
        keywords = extract_keywords(reasoning)

        analysis_items.append({
            'id': rec.get('id', ''),
            'target': rec.get('target', ''),
            'confidence': confidence,
            'confidence_bin': confidence_bin,
            'source': source,
            'market': market,
            'direction': direction,
            'score': score,
            'ret_5d': ret,
            'hit': 1 if score >= 0.5 else 0,
            'reasoning': reasoning,
            'keywords': keywords,
        })

    total = len(analysis_items)
    total_hits = sum(1 for a in analysis_items if a['hit'])
    total_ret = sum(a['ret_5d'] for a in analysis_items)
    avg_ret = total_ret / total * 100 if total else 0
    hit_rate = total_hits / total * 100 if total else 0

    # === 生成Markdown报告 ===
    md_lines = []
    md_lines.append("# 特征分析报告 (Feature Analysis)")
    md_lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md_lines.append(f"> 数据范围: {len(all_scored)} 条已评分建议")
    md_lines.append(f"> 评分周期: {MODEL_HORIZON_DAYS}天, 阈值: {MODEL_THRESHOLD:.0%}")
    md_lines.append("")

    md_lines.append("## 总体统计\n")
    md_lines.append(f"- **总样本数**: {total}")
    md_lines.append(f"- **命中数**: {total_hits}/{total} = **{hit_rate:.1f}%**")
    md_lines.append(f"- **平均收益率**: {avg_ret:+.2f}%")
    md_lines.append("")

    # 1. 置信度 vs 收益率
    conf_items = [{'value': a['confidence_bin'], 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']} for a in analysis_items]
    md_lines.append(analyze_dimension(conf_items, 'confidence_bin', '1. 置信度区间 vs 命中率'))

    # 2. 来源 vs 命中率
    source_items = [{'value': a['source'], 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']} for a in analysis_items]
    md_lines.append(analyze_dimension(source_items, 'source', '2. 来源 (source) vs 命中率'))

    # 3. 市场 vs 命中率
    market_items = [{'value': a['market'], 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']} for a in analysis_items]
    md_lines.append(analyze_dimension(market_items, 'market', '3. 市场 (market) vs 命中率'))

    # 4. 方向 vs 命中率
    direction_items = [{'value': a['direction'], 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']} for a in analysis_items]
    md_lines.append(analyze_dimension(direction_items, 'direction', '4. 方向 (direction) vs 命中率'))

    # 5. 关键词 vs 命中率
    kw_items = []
    for a in analysis_items:
        for kw in a['keywords']:
            kw_items.append({'value': kw, 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']})
    md_lines.append(analyze_dimension(kw_items, 'keyword', '5. Reasoning关键词 vs 命中率'))

    # 6. 方向×市场交叉分析
    md_lines.append("### 6. 方向×市场 交叉分析\n")
    cross_items = [{'value': f"{a['direction']}×{a['market']}", 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']} for a in analysis_items]
    md_lines.append(analyze_dimension(cross_items, 'cross', '6. 方向×市场 交叉分析'))

    # 7. 来源×方向交叉分析
    md_lines.append("### 7. 来源×方向 交叉分析\n")
    cross_items2 = [{'value': f"{a['source']}×{a['direction']}", 'score': a['score'], 'ret_5d': a['ret_5d'], 'hit': a['hit']} for a in analysis_items]
    md_lines.append(analyze_dimension(cross_items2, 'cross2', '7. 来源×方向 交叉分析'))

    # 8. 详细数据表
    md_lines.append("## 8. 详细数据表\n")
    md_lines.append("| ID | 日期 | 标的 | 方向 | 置信度 | 来源 | 市场 | 得分 | 收益率 | 命中 |")
    md_lines.append("|----|------|------|------|--------|------|------|------|--------|------|")

    for a in sorted(analysis_items, key=lambda x: x['score'], reverse=False):
        # 查找原始记录获取日期
        orig_date = ''
        for r in all_scored:
            if r.get('id') == a['id']:
                orig_date = r.get('date', '')
                break
        hit_mark = '✅' if a['hit'] else '❌'
        md_lines.append(f"| {a['id'][:12]} | {orig_date} | {a['target'][:12]} | {a['direction']} | {a['confidence']:.2f} | {a['source']} | {a['market']} | {a['score']:.1f} | {a['ret_5d']:+.2%} | {hit_mark} |")

    md_lines.append("")

    # 生成报告
    report = "\n".join(md_lines)

    # 打印到stdout
    print("\n" + "=" * 60)
    print(report)

    # 保存到文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(report)
    print(f"\n📄 报告已保存到: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
