#!/usr/bin/env python3
"""
推荐完整性审计 — 检查Agent是否真的调用了tracker
用法：
  python3 recommendation_audit.py --cron morning    # 检查今天的晨报推荐
  python3 recommendation_audit.py --cron pre_market # 检查今天的盘前推荐
  python3 recommendation_audit.py --full            # 全面审计
  python3 recommendation_audit.py --date 20260623   # 检查指定日期
"""
import json, os, sys
from datetime import datetime

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
OUTPUT_DIR = os.path.join(ROOT, 'output')

def load_recs():
    with open(TRACK_FILE) as f:
        return json.load(f)['recommendations']

def get_cron_output_dir():
    """获取cron output目录"""
    return os.path.expanduser('~/.hermes/cron/output')

def extract_recommendations_from_output(text):
    """从cron输出文本中提取推荐（启发式匹配）"""
    recs = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        # 匹配推荐格式：| Ticker | 方向 | 置信度 |
        if '|' in line and any(kw in line.lower() for kw in ['bullish', 'bearish', '🟢', '🔴', '买', '卖']):
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                # 尝试提取ticker
                for part in parts:
                    if any(c.isalpha() for c in part) and len(part) <= 10 and not any(kw in part.lower() for kw in ['方向', '置信', '理由', '建议']):
                        recs.append({'target': part, 'raw_line': line})
                        break
    
    return recs

def audit_date(date_str=None):
    """审计指定日期的推荐完整性"""
    if not date_str:
        date_str = datetime.now().strftime('%Y%m%d')
    
    # 1. 从tracker获取该日期的推荐
    all_recs = load_recs()
    tracker_recs = [r for r in all_recs if r.get('date') == date_str]
    
    # 2. 检查cron output中是否有该日期的输出
    cron_dir = get_cron_output_dir()
    cron_outputs = {}
    
    if os.path.exists(cron_dir):
        for job_id in os.listdir(cron_dir):
            job_dir = os.path.join(cron_dir, job_id)
            if os.path.isdir(job_dir):
                for f in os.listdir(job_dir):
                    if date_str.replace('-', '') in f or date_str in f:
                        filepath = os.path.join(job_dir, f)
                        try:
                            with open(filepath) as fh:
                                content = fh.read()
                            cron_outputs[job_id] = content[:5000]  # 只取前5000字
                        except:
                            pass
    
    # 3. 审计
    print(f'=== 推荐完整性审计 | {date_str} ===')
    print(f'Tracker中该日推荐: {len(tracker_recs)}条')
    print(f'Cron output文件: {len(cron_outputs)}个')
    
    # 4. 检查每个推荐
    issues = []
    
    for rec in tracker_recs:
        rec_id = rec.get('id', '?')
        target = rec.get('target', '?')
        source = rec.get('source', '?')
        rules = rec.get('rules_applied', [])
        conf = rec.get('confidence', 0)
        reasoning = rec.get('reasoning', '')
        
        # 检查rules_applied
        if not rules:
            issues.append(f'{rec_id} {target}: rules_applied为空')
        
        # 检查reasoning是否有模型信号
        has_signal = any(kw in reasoning.lower() for kw in ['score', 'prob', 'rank', 'gg', '蓝盾', '绿箭', 'signal'])
        if not has_signal and rec.get('action') in ('buy', 'sell'):
            issues.append(f'{rec_id} {target}: reasoning无模型信号')
        
        # 检查confidence是否有量化依据
        if conf > 0.6:
            import re
            has_number = bool(re.search(r'\d+\.?\d*', reasoning))
            if not has_number:
                issues.append(f'{rec_id} {target}: conf={conf}但无量化依据')
        
        # 检查type标记
        macro_targets = ['VIX', '大盘', 'SPY', 'QQQ', 'Semiconductor', 'DIA', 'IWM']
        if target in macro_targets and rec.get('type') != 'macro':
            issues.append(f'{rec_id} {target}: macro标的未标记type')
    
    # 5. 输出
    if issues:
        print(f'\n⚠️ 发现{len(issues)}个问题:')
        for issue in issues:
            print(f'  - {issue}')
    else:
        print(f'\n✅ 所有{len(tracker_recs)}条推荐校验通过')
    
    # 6. 评分状态
    scored = [r for r in tracker_recs if r.get('status') == 'scored']
    pending = [r for r in tracker_recs if r.get('status') == 'pending']
    print(f'\n评分状态: {len(scored)}已评 / {len(pending)}待评')
    
    return issues

def full_audit():
    """全面审计"""
    all_recs = load_recs()
    
    print('=' * 60)
    print('📋 推荐完整性全面审计')
    print('=' * 60)
    
    # 总体统计
    total = len(all_recs)
    scored = [r for r in all_recs if r.get('status') == 'scored']
    filled = sum(1 for r in all_recs if r.get('rules_applied'))
    macro = sum(1 for r in all_recs if r.get('type') == 'macro')
    
    # rules_source统计
    source_counts = {}
    for r in all_recs:
        s = r.get('rules_source', 'unknown')
        source_counts[s] = source_counts.get(s, 0) + 1
    
    agent_rate = source_counts.get('agent_applied', 0) / total * 100 if total else 0
    
    print(f'\n总推荐: {total}')
    print(f'rules_applied填写率: {filled}/{total} = {filled/total*100:.0f}%')
    print(f'rules_source分布:')
    print(f'  agent_applied (Agent自觉): {source_counts.get("agent_applied",0)}/{total} = {agent_rate:.0f}%')
    print(f'  auto_inferred (系统回填): {source_counts.get("auto_inferred",0)}/{total}')
    print(f'  auto_default (默认值): {source_counts.get("auto_default",0)}/{total}')
    print(f'⚠️ 真实执行率: {agent_rate:.0f}% (目标>80%)')
    print(f'macro标记: {macro}条')
    print(f'已评分: {len(scored)}条')
    
    # 按来源统计
    from collections import defaultdict
    by_source = defaultdict(lambda: {'total':0, 'filled_rules':0, 'scored':0, 'correct':0})
    for r in all_recs:
        src = r.get('source', '?')
        by_source[src]['total'] += 1
        if r.get('rules_applied'):
            by_source[src]['filled_rules'] += 1
        if r.get('status') == 'scored':
            by_source[src]['scored'] += 1
            if r.get('score', 0) >= 0.5:
                by_source[src]['correct'] += 1
    
    print(f'\n按来源:')
    for src, d in sorted(by_source.items()):
        rate = d['correct']/d['scored']*100 if d['scored'] else 0
        fill = d['filled_rules']/d['total']*100 if d['total'] else 0
        print(f'  {src:15s} | {d["total"]:>3}条 | rules:{fill:.0f}% | 命中:{d["correct"]}/{d["scored"]}={rate:.0f}%')
    
    # 检查今日
    today = datetime.now().strftime('%Y%m%d')
    print(f'\n--- 今日({today})审计 ---')
    audit_date(today)

def main():
    if '--full' in sys.argv:
        full_audit()
    elif '--date' in sys.argv:
        idx = sys.argv.index('--date')
        audit_date(sys.argv[idx+1])
    else:
        audit_date()

if __name__ == '__main__':
    main()
