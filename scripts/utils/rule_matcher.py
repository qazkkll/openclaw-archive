#!/usr/bin/env python3
"""
规则匹配器 — 从reasoning中自动匹配适用规则
用法：
  python3 rule_matcher.py '{"reasoning":"GG score 0.08 RSI=45",...}'
  python3 rule_matcher.py --reasoning "GG score 0.08 RSI=45"
  python3 rule_matcher.py --check-lessons  # 检查lessons.json规则是否被使用
"""
import json, os, sys, re

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
LESSONS_FILE = os.path.join(ROOT, 'data/lessons.json')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')

# 规则 → 关键词映射（比post_validate更精细）
RULE_MATCHERS = {
    'R001': {
        'desc': '红杉信号必须读现成文件',
        'keywords': ['latest_xgb', '红杉', 'XGB', 'xgb', 'top数组', '信号文件', 'cn_alpha'],
        'weight': 1.0,
    },
    'R002': {
        'desc': '不能重排模型输出的排序',
        'keywords': ['rank', '模型排序', 'score排序', '按score', 'rank字段'],
        'weight': 1.0,
    },
    'R003': {
        'desc': '推荐必须同时记录到tracker',
        'keywords': ['tracker', 'recommendation_tracker', '记录'],
        'weight': 0.5,  # 总是适用
    },
    'R004': {
        'desc': '推荐必须有模型信号支撑',
        'keywords': ['score', 'probability', 'prob', 'rank', '蓝盾', '绿箭', 'GG', 'G级', 'Y级', '模型信号', 'blueshield', 'arrow', 'signal', 'top pick'],
        'weight': 1.0,
    },
    'R005': {
        'desc': 'close_review只复盘不产生新推荐',
        'keywords': ['close_review', '收盘复盘', '复盘'],
        'weight': 0.8,
        'negative': True,  # 如果source是close_review，这条是约束而非适用
    },
    'R006': {
        'desc': '置信度必须有量化依据',
        'keywords': ['RSI', 'score', 'probability', 'prob', 'rank', '%', '倍', '涨', '跌', '市值', 'PE', '市盈率'],
        'weight': 0.8,
        'condition': 'confidence > 0.6',  # 只在高置信度时适用
    },
}

def match_rules(rec):
    """匹配推荐适用的规则"""
    reasoning = rec.get('reasoning', '').lower()
    source = rec.get('source', '')
    confidence = rec.get('confidence', 0)
    
    matched = []
    
    for rule_id, rule in RULE_MATCHERS.items():
        # 关键词匹配
        keyword_hit = False
        for kw in rule['keywords']:
            if kw.lower() in reasoning:
                keyword_hit = True
                break
        
        # 条件检查
        condition_met = True
        if rule.get('condition') == 'confidence > 0.6':
            condition_met = confidence > 0.6
        
        # 负面规则（如R005：close_review不应产生推荐）
        if rule.get('negative'):
            if source == 'close_review':
                matched.append({
                    'rule': rule_id,
                    'desc': rule['desc'],
                    'match_type': 'negative_constraint',
                    'reason': 'close_review来源触发约束规则'
                })
            continue
        
        if keyword_hit and condition_met:
            matched.append({
                'rule': rule_id,
                'desc': rule['desc'],
                'match_type': 'keyword',
                'reason': f'reasoning中匹配到关键词'
            })
        elif rule_id in ('R003',) and not keyword_hit:
            # R003总是适用（推荐必须记录）
            matched.append({
                'rule': rule_id,
                'desc': rule['desc'],
                'match_type': 'always',
                'reason': '推荐必须记录（无条件适用）'
            })
    
    return matched

def check_lessons_usage():
    """检查lessons.json中的规则是否被推荐引用"""
    with open(LESSONS_FILE) as f:
        lessons = json.load(f)
    
    with open(TRACK_FILE) as f:
        data = json.load(f)
    
    recs = data['recommendations']
    
    print('=== lessons.json规则使用情况 ===')
    for l in lessons['lessons']:
        rule_id = l['id']
        # 检查是否在任何推荐的rules_applied中
        used_in = [r for r in recs if rule_id in str(r.get('rules_applied', []))]
        status = '✅' if used_in else '❌'
        print(f'  {status} {rule_id}: {l["rule"][:50]}... | 被引用{len(used_in)}次')
    
    # 统计
    total_lessons = len(lessons['lessons'])
    used = sum(1 for l in lessons['lessons'] 
               if any(l['id'] in str(r.get('rules_applied',[])) for r in recs))
    print(f'\n引用率: {used}/{total_lessons} = {used/total_lessons*100:.0f}%' if total_lessons else '无规则')

def main():
    if '--check-lessons' in sys.argv:
        check_lessons_usage()
        return
    
    if '--reasoning' in sys.argv:
        idx = sys.argv.index('--reasoning')
        reasoning = sys.argv[idx+1]
        rec = {'reasoning': reasoning, 'source': 'manual', 'confidence': 0.6}
    else:
        if not sys.stdin.isatty():
            rec = json.load(sys.stdin)
        else:
            print('用法: python3 rule_matcher.py --reasoning "GG score 0.08"')
            print('  或: python3 rule_matcher.py --check-lessons')
            return
    
    matched = match_rules(rec)
    
    if matched:
        print('匹配到的规则:')
        for m in matched:
            print(f'  {m["rule"]}: {m["desc"]} ({m["match_type"]})')
    else:
        print('未匹配到任何规则')
    
    # 输出规则ID列表
    rule_ids = [m['rule'] for m in matched if m['match_type'] != 'negative_constraint']
    print(f'\nrules_applied: {rule_ids}')

if __name__ == '__main__':
    main()
