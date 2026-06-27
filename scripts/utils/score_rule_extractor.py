#!/usr/bin/env python3
"""
评分→规则自动提取器 — 从历史评分数据中提取可执行规则
用法：
  python3 score_rule_extractor.py              # 分析并输出建议规则
  python3 score_rule_extractor.py --apply      # 自动写入lessons.json
  python3 score_rule_extractor.py --days 30    # 分析最近N天
  python3 score_rule_extractor.py --summary    # 输出摘要供cron注入
"""
import json, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
LESSONS_FILE = os.path.join(ROOT, 'data/lessons.json')

def load_recs():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE) as f:
            return json.load(f).get('recommendations', [])
    return []

def load_lessons():
    if os.path.exists(LESSONS_FILE):
        with open(LESSONS_FILE) as f:
            return json.load(f)
    return {'lessons': []}

def save_lessons(data):
    with open(LESSONS_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def analyze(days=30):
    """分析评分数据，返回结构化发现"""
    recs = load_recs()
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    scored = [r for r in recs if r.get('date', '') >= cutoff and r.get('status') == 'scored']
    
    if len(scored) < 3:
        return {'error': f'已评分数据不足（{len(scored)}条），需要至少3条'}
    
    findings = []
    
    # 1. 按来源分析
    by_source = defaultdict(list)
    for r in scored:
        by_source[r.get('source', 'unknown')].append(r)
    
    for src, recs_list in by_source.items():
        correct = sum(1 for r in recs_list if (r.get('score') or 0) >= 0.5)
        total = len(recs_list)
        rate = correct / total if total > 0 else 0
        
        if rate >= 0.7 and total >= 3:
            findings.append({
                'type': 'strength',
                'source': src,
                'message': f'{src}来源命中率{rate:.0%}（{correct}/{total}），可信赖',
                'evidence': [(r.get('target', ''), r.get('actual_return', '')) for r in recs_list]
            })
        elif rate <= 0.3 and total >= 3:
            findings.append({
                'type': 'weakness',
                'source': src,
                'message': f'{src}来源命中率仅{rate:.0%}（{correct}/{total}），需限制',
                'evidence': [(r.get('target', ''), r.get('actual_return', '')) for r in recs_list]
            })
    
    # 2. 置信度校准
    conf_brackets = [
        ('high', 0.7, 1.0),
        ('mid', 0.5, 0.7),
        ('low', 0.0, 0.5),
    ]
    conf_data = {}
    for label, lo, hi in conf_brackets:
        bracket = [r for r in scored if lo <= r.get('confidence', 0) < hi]
        if bracket:
            correct = sum(1 for r in bracket if r.get('score') or 0 >= 0.5)
            conf_data[label] = {'total': len(bracket), 'correct': correct, 'rate': correct/len(bracket)}
    
    # 检查置信度反转（高置信度反而低命中率）
    if 'high' in conf_data and 'low' in conf_data:
        if conf_data['high']['rate'] < conf_data['low']['rate']:
            findings.append({
                'type': 'calibration_warning',
                'message': f"置信度反转：高置信度命中率{conf_data['high']['rate']:.0%} < 低置信度{conf_data['low']['rate']:.0%}",
                'data': conf_data
            })
    
    # 3. 方向准确性
    by_dir = defaultdict(list)
    for r in scored:
        by_dir[r.get('direction', 'unknown')].append(r)
    
    for direction, dir_recs in by_dir.items():
        correct = sum(1 for r in dir_recs if r.get('score') or 0 >= 0.5)
        total = len(dir_recs)
        if total >= 3:
            rate = correct / total
            if rate <= 0.3:
                findings.append({
                    'type': 'weakness',
                    'source': f'direction:{direction}',
                    'message': f'{direction}方向命中率仅{rate:.0%}（{correct}/{total}）',
                    'evidence': [(r.get('target', ''), r.get('actual_return', '')) for r in dir_recs]
                })
    
    # 4. rules_applied使用率
    with_rules = [r for r in scored if r.get('rules_applied')]
    if len(scored) >= 5:
        usage_rate = len(with_rules) / len(scored)
        if usage_rate < 0.5:
            findings.append({
                'type': 'process_issue',
                'message': f'rules_applied使用率仅{usage_rate:.0%}（{len(with_rules)}/{len(scored)}），规则执行不足',
            })
    
    # 5. 失败模式（连续错误）
    sorted_recs = sorted(scored, key=lambda r: r.get('date', ''))
    streak = 0
    max_streak = 0
    for r in sorted_recs:
        if r.get('score') or 0 < 0.5:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    
    if max_streak >= 3:
        findings.append({
            'type': 'pattern',
            'message': f'最大连续错误{max_streak}条，需检查系统性偏差',
        })
    
    # 整体统计
    total_scored = len(scored)
    total_correct = sum(1 for r in scored if r.get('score') or 0 >= 0.5)
    overall_rate = total_correct / total_scored if total_scored > 0 else 0
    
    return {
        'period_days': days,
        'total_scored': total_scored,
        'total_correct': total_correct,
        'overall_rate': overall_rate,
        'conf_data': conf_data,
        'findings': findings,
        'by_source': {src: {'total': len(rs), 'correct': sum(1 for r in rs if r.get('score') or 0 >= 0.5)} for src, rs in by_source.items()},
    }

def generate_rules(analysis):
    """从分析结果中生成建议规则"""
    rules = []
    
    for f in analysis.get('findings', []):
        if f['type'] == 'weakness' and 'close_review' in f.get('source', ''):
            rules.append({
                'id': 'auto_R007',
                'rule': 'close_review来源不产生新推荐',
                'reason': f"数据支撑：{f['message']}",
                'status': 'suggested'
            })
        
        if f['type'] == 'calibration_warning':
            rules.append({
                'id': 'auto_R008',
                'rule': '高置信度(>0.7)推荐需额外量化证据，否则降权',
                'reason': f"数据支撑：{f['message']}",
                'status': 'suggested'
            })
        
        if f['type'] == 'process_issue' and 'rules_applied' in f.get('message', ''):
            rules.append({
                'id': 'auto_R009',
                'rule': '推荐必须填写rules_applied字段',
                'reason': f"数据支撑：{f['message']}",
                'status': 'suggested'
            })
    
    return rules

def apply_rules(rules):
    """将建议规则写入lessons.json"""
    lessons_data = load_lessons()
    existing_ids = {l.get('id') for l in lessons_data.get('lessons', [])}
    
    added = 0
    for rule in rules:
        if rule['id'] not in existing_ids:
            lessons_data['lessons'].append({
                'id': rule['id'],
                'created': datetime.now().strftime('%Y-%m-%d'),
                'category': 'auto_extracted',
                'rule': rule['rule'],
                'reason': rule['reason'],
                'status': rule['status'],
                'evidence_count': 0,
            })
            added += 1
    
    if added > 0:
        save_lessons(lessons_data)
        print(f"✅ 已写入{added}条新规则到lessons.json")
    else:
        print("没有新规则需要写入")

def output_summary(analysis):
    """输出简洁摘要，供cron prompt注入"""
    if 'error' in analysis:
        print(f"⚠️ {analysis['error']}")
        return
    
    print(f"📊 评分统计（最近{analysis['period_days']}天）")
    print(f"总评: {analysis['total_correct']}/{analysis['total_scored']} = {analysis['overall_rate']:.0%}")
    
    # 按来源
    for src, data in analysis.get('by_source', {}).items():
        rate = data['correct'] / data['total'] if data['total'] > 0 else 0
        emoji = '✅' if rate >= 0.6 else '⚠️' if rate >= 0.4 else '❌'
        print(f"  {emoji} {src}: {data['correct']}/{data['total']}={rate:.0%}")
    
    # 关键发现
    for f in analysis.get('findings', []):
        if f['type'] in ('weakness', 'calibration_warning', 'process_issue'):
            print(f"  ⚠️ {f['message']}")

def main():
    days = 30
    apply = '--apply' in sys.argv
    summary = '--summary' in sys.argv
    
    if '--days' in sys.argv:
        idx = sys.argv.index('--days')
        if idx + 1 < len(sys.argv):
            days = int(sys.argv[idx + 1])
    
    analysis = analyze(days)
    
    if summary:
        output_summary(analysis)
        return
    
    # 输出分析
    if 'error' in analysis:
        print(f"⚠️ {analysis['error']}")
        return
    
    print(f"📊 评分→规则分析（最近{analysis['period_days']}天，{analysis['total_scored']}条已评分）")
    print(f"整体命中率: {analysis['total_correct']}/{analysis['total_scored']} = {analysis['overall_rate']:.0%}\n")
    
    for f in analysis['findings']:
        emoji = {'strength': '✅', 'weakness': '⚠️', 'calibration_warning': '🔄', 'process_issue': '🔧', 'pattern': '📊'}.get(f['type'], '📌')
        print(f"{emoji} {f['message']}")
    
    # 生成规则
    rules = generate_rules(analysis)
    if rules:
        print(f"\n📝 建议新增规则:")
        for r in rules:
            print(f"  {r['id']}: {r['rule']}")
        
        if apply:
            apply_rules(rules)
    else:
        print("\n无新增规则建议")

if __name__ == '__main__':
    main()
