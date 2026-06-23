#!/usr/bin/env python3
"""
推荐后处理校验器 — 强制执行rules_applied填写
用法：
  echo '{"source":"morning","reasoning":"GG score 0.08 RSI=45",...}' | python3 post_validate.py
  python3 post_validate.py --file rec.json
  python3 post_validate.py --check-all  # 检查tracker中所有空rules_applied
"""
import json, os, sys, re

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
LESSONS_FILE = os.path.join(ROOT, 'data/lessons.json')

# 规则匹配关键词（reasoning中出现这些词 → 对应规则）
RULE_KEYWORDS = {
    'R001': ['latest_xgb', '红杉', 'XGB', 'xgb', 'top数组', '信号文件'],
    'R002': ['rank', '模型排序', 'score排序', '按score'],
    'R003': ['tracker', 'recommendation_tracker', '记录'],
    'R004': ['score', 'probability', 'prob', 'rank', '蓝盾', '绿箭', 'GG', '模型信号', 'Blueshield', 'Arrow'],
    'R005': ['close_review', '收盘复盘', '复盘'],
    'R006': ['RSI', 'score', 'probability', 'prob', 'rank', '%', '倍', '涨', '跌'],
}

# 来源 → 预期规则
SOURCE_DEFAULTS = {
    'morning': ['R001', 'R004'],
    'pre_market': ['R004'],
    'close_review': [],  # 不应产生推荐
    'manual': ['R004'],
}

def infer_rules(rec):
    """从reasoning中推断适用规则"""
    reasoning = rec.get('reasoning', '')
    source = rec.get('source', '')
    rules = set()
    
    # 关键词匹配
    for rule_id, keywords in RULE_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in reasoning.lower():
                rules.add(rule_id)
                break
    
    # 来源默认规则
    for rule_id in SOURCE_DEFAULTS.get(source, []):
        rules.add(rule_id)
    
    # R004: 如果有confidence>0.6，必须有量化数字
    conf = rec.get('confidence', 0)
    if conf > 0.6:
        has_number = bool(re.search(r'\d+\.?\d*', reasoning))
        if has_number:
            rules.add('R006')
    
    return sorted(rules)

def validate(rec, auto_fix=True):
    """校验单条推荐，返回(通过, 违规列表, 修复后rec)"""
    violations = []
    fixed = dict(rec)
    
    # 检查1: rules_applied是否填写
    current_rules = rec.get('rules_applied', [])
    if not current_rules:
        inferred = infer_rules(rec)
        if inferred:
            if auto_fix:
                fixed['rules_applied'] = inferred
                return True, ['auto_filled'], fixed
            else:
                violations.append(f'rules_applied为空，推断为{inferred}')
        else:
            violations.append('rules_applied为空且无法推断')
    
    # 检查2: close_review不应产生推荐
    if rec.get('source') == 'close_review':
        target = rec.get('target', '')
        action = rec.get('action', '')
        if action in ('buy', 'sell') and target != '大盘':
            violations.append('close_review不应产生买入/卖出推荐(R005)')
    
    # 检查3: confidence>0.6需有量化依据
    conf = rec.get('confidence', 0)
    reasoning = rec.get('reasoning', '')
    if conf > 0.6:
        has_number = bool(re.search(r'\d+\.?\d*', reasoning))
        if not has_number:
            violations.append(f'conf={conf}>0.6但reasoning无量化数字(R006)')
            if auto_fix:
                fixed['confidence'] = 0.5
                return True, ['confidence_demoted'], fixed
    
    # 检查4: 推荐必须有模型信号支撑（非macro判断）
    if rec.get('type') != 'macro':
        has_signal = any(kw in reasoning.lower() for kw in ['score', 'prob', 'rank', 'gg', '蓝盾', '绿箭', 'signal'])
        if not has_signal and rec.get('action') in ('buy', 'sell'):
            violations.append('推荐无模型信号支撑(R004)')
    
    passed = len(violations) == 0
    return passed, violations, fixed

def check_all():
    """检查tracker中所有推荐的rules_applied"""
    with open(TRACK_FILE) as f:
        data = json.load(f)
    
    recs = data['recommendations']
    total = len(recs)
    empty = [r for r in recs if not r.get('rules_applied')]
    filled = [r for r in recs if r.get('rules_applied')]
    
    print(f'=== rules_applied检查 ({total}条推荐) ===')
    print(f'已填写: {len(filled)}/{total} = {len(filled)/total*100:.0f}%')
    print(f'空: {len(empty)}/{total}')
    
    if empty:
        print(f'\n空rules_applied的推荐:')
        for r in empty:
            inferred = infer_rules(r)
            print(f'  {r["id"]} | {r.get("source","?"):12s} | date={r.get("date","?")} | target={r.get("target","?")[:15]} | 推断规则: {inferred}')
    
    return empty, filled

def main():
    if '--check-all' in sys.argv:
        check_all()
        return
    
    # 从stdin或file读取推荐
    rec = None
    if '--file' in sys.argv:
        idx = sys.argv.index('--file')
        with open(sys.argv[idx+1]) as f:
            rec = json.load(f)
    else:
        if not sys.stdin.isatty():
            rec = json.load(sys.stdin)
    
    if not rec:
        print('用法: echo \'{"source":"morning",...}\' | python3 post_validate.py')
        print('  或: python3 post_validate.py --check-all')
        return
    
    passed, violations, fixed = validate(rec)
    
    if passed:
        print('✅ 校验通过')
        if 'auto_filled' in violations:
            print(f'  rules_applied已自动回填: {fixed["rules_applied"]}')
        if 'confidence_demoted' in violations:
            print(f'  confidence已降级到0.5')
    else:
        print('❌ 校验失败:')
        for v in violations:
            print(f'  - {v}')
    
    print(json.dumps(fixed, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
