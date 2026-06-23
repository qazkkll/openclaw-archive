#!/usr/bin/env python3
"""
预测归因分析 — 不只看涨跌，而是分析"为什么对/为什么错"
用法：
  python3 prediction_attribution.py              # 分析所有已评分推荐
  python3 prediction_attribution.py --date 20260623  # 分析指定日期
  python3 prediction_attribution.py --report      # 输出归因报告
"""
import json, os, sys
from datetime import datetime
from collections import defaultdict

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
ATTRIBUTION_FILE = os.path.join(ROOT, 'data/attribution.json')

def load_recs():
    with open(TRACK_FILE) as f:
        return json.load(f)['recommendations']

def load_attribution():
    if os.path.exists(ATTRIBUTION_FILE):
        with open(ATTRIBUTION_FILE) as f:
            return json.load(f)
    return {'attributions': [], 'meta': {'created': datetime.now().isoformat()}}

def save_attribution(data):
    with open(ATTRIBUTION_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def classify_failure_mode(rec):
    """分类失败原因"""
    reasoning = rec.get('reasoning', '').lower()
    target = rec.get('target', '')
    actual_return = rec.get('actual_return', '')
    source = rec.get('source', '')
    rules = rec.get('rules_applied', [])
    conf = rec.get('confidence', 0)
    
    # 如果成功，不是失败
    if rec.get('score', 0) >= 0.5:
        return None
    
    modes = []
    
    # 1. Agent主观加戏（有模型信号但Agent加了主观判断）
    if any(kw in reasoning for kw in ['最健康', '最看好', '首选', '龙头', '白马']):
        if 'score' in reasoning or 'GG' in reasoning:
            modes.append('agent_override')  # Agent在模型信号上加了主观判断
    
    # 2. 高置信度无支撑
    if conf > 0.7 and not any(kw in reasoning for kw in ['RSI', 'score', 'prob']):
        modes.append('conf_no_evidence')
    
    # 3. close_review来源
    if source == 'close_review':
        modes.append('close_review_recommend')
    
    # 4. 宏观误判（direction错但方向是宏观判断）
    if rec.get('type') == 'macro':
        modes.append('macro_wrong')
    
    # 5. 无模型信号
    if not rules or not any(kw in reasoning for kw in ['score', 'prob', 'rank', 'GG', '蓝盾', '绿箭']):
        modes.append('no_model_signal')
    
    # 6. 默认：模型信号错误
    if not modes:
        modes.append('model_signal_wrong')
    
    return modes

def classify_success_mode(rec):
    """分类成功原因"""
    if rec.get('score', 0) < 0.5:
        return None
    
    modes = []
    reasoning = rec.get('reasoning', '').lower()
    source = rec.get('source', '')
    rules = rec.get('rules_applied', [])
    
    # 1. 纯模型信号（Agent只是格式化输出）
    if source == 'morning' and rules:
        modes.append('model_signal_only')
    
    # 2. 宏观配合
    if any(kw in reasoning for kw in ['macro', '宏观', 'vix', '大盘']):
        modes.append('macro_aligned')
    
    # 3. RSI/技术面配合
    if 'RSI' in reasoning:
        modes.append('technical_aligned')
    
    if not modes:
        modes.append('unknown_success')
    
    return modes

def attribute_one(rec):
    """对单条推荐做归因"""
    is_correct = rec.get('score', 0) >= 0.5
    actual_return = rec.get('actual_return', '0%')
    
    # 解析实际收益率
    try:
        ret_str = actual_return.replace('%', '').replace('+', '')
        ret_val = float(ret_str) / 100
    except:
        ret_val = 0
    
    attribution = {
        'rec_id': rec.get('id'),
        'target': rec.get('target'),
        'source': rec.get('source'),
        'date': rec.get('date'),
        'direction': rec.get('direction'),
        'confidence': rec.get('confidence'),
        'is_correct': is_correct,
        'actual_return': actual_return,
        'actual_return_pct': ret_val,
    }
    
    if is_correct:
        attribution['success_modes'] = classify_success_mode(rec)
        attribution['failure_modes'] = []
    else:
        attribution['success_modes'] = []
        attribution['failure_modes'] = classify_failure_mode(rec)
    
    # 评分与置信度差距
    attribution['calibration_gap'] = rec.get('confidence', 0) - (1.0 if is_correct else 0.0)
    
    return attribution

def run_attribution(date_filter=None):
    """运行归因分析"""
    recs = load_recs()
    scored = [r for r in recs if r.get('status') == 'scored']
    
    if date_filter:
        scored = [r for r in scored if r.get('date') == date_filter]
    
    if not scored:
        print('没有已评分的推荐可供归因')
        return []
    
    attributions = [attribute_one(r) for r in scored]
    
    # 保存
    attr_data = load_attribution()
    # 去重：同一条推荐只保留最新归因
    existing_ids = {a['rec_id'] for a in attr_data['attributions']}
    new_attrs = [a for a in attributions if a['rec_id'] not in existing_ids]
    updated_attrs = [a for a in attributions if a['rec_id'] in existing_ids]
    
    # 更新已有的
    for i, a in enumerate(attr_data['attributions']):
        for new_a in updated_attrs:
            if a['rec_id'] == new_a['rec_id']:
                attr_data['attributions'][i] = new_a
                break
    
    # 添加新的
    attr_data['attributions'].extend(new_attrs)
    attr_data['meta']['last_run'] = datetime.now().isoformat()
    save_attribution(attr_data)
    
    return attributions

def generate_report(attributions):
    """生成归因报告"""
    if not attributions:
        print('无归因数据')
        return
    
    total = len(attributions)
    correct = sum(1 for a in attributions if a['is_correct'])
    
    print(f'📊 预测归因报告 | {total}条已评分')
    print(f'命中率: {correct}/{total} = {correct/total*100:.0f}%')
    
    # 1. 按来源归因
    print(f'\n--- 来源归因 ---')
    by_source = defaultdict(lambda: {'correct':0, 'total':0, 'failure_modes':[], 'success_modes':[]})
    for a in attributions:
        src = a['source']
        by_source[src]['total'] += 1
        if a['is_correct']:
            by_source[src]['correct'] += 1
            by_source[src]['success_modes'].extend(a.get('success_modes', []))
        else:
            by_source[src]['failure_modes'].extend(a.get('failure_modes', []))
    
    for src, d in sorted(by_source.items()):
        rate = d['correct']/d['total']*100 if d['total'] else 0
        emoji = '✅' if rate >= 60 else '⚠️' if rate >= 40 else '❌'
        print(f'  {emoji} {src}: {d["correct"]}/{d["total"]} = {rate:.0f}%')
        
        # 失败模式统计
        if d['failure_modes']:
            mode_counts = defaultdict(int)
            for m in d['failure_modes']:
                mode_counts[m] += 1
            for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
                print(f'      失败模式: {mode} ×{count}')
        
        # 成功模式统计
        if d['success_modes']:
            mode_counts = defaultdict(int)
            for m in d['success_modes']:
                mode_counts[m] += 1
            for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
                print(f'      成功模式: {mode} ×{count}')
    
    # 2. 置信度校准
    print(f'\n--- 置信度校准 ---')
    buckets = [
        ('high(>0.7)', 0.7, 1.0),
        ('mid(0.5-0.7)', 0.5, 0.7),
        ('low(<0.5)', 0.0, 0.5),
    ]
    for label, lo, hi in buckets:
        bucket = [a for a in attributions if lo <= a['confidence'] < hi]
        if bucket:
            correct_b = sum(1 for a in bucket if a['is_correct'])
            avg_ret = sum(a['actual_return_pct'] for a in bucket) / len(bucket)
            print(f'  {label}: {correct_b}/{len(bucket)} = {correct_b/len(bucket)*100:.0f}% | 均收益{avg_ret:+.2%}')
    
    # 3. 失败模式汇总
    print(f'\n--- 失败模式汇总 ---')
    all_failure_modes = []
    for a in attributions:
        if not a['is_correct']:
            all_failure_modes.extend(a.get('failure_modes', []))
    
    if all_failure_modes:
        mode_counts = defaultdict(int)
        for m in all_failure_modes:
            mode_counts[m] += 1
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            desc = {
                'agent_override': 'Agent在模型信号上加主观判断',
                'conf_no_evidence': '高置信度无量化支撑',
                'close_review_recommend': 'close_review产生推荐',
                'macro_wrong': '宏观方向判断错误',
                'no_model_signal': '推荐无模型信号',
                'model_signal_wrong': '模型信号本身错误',
            }.get(mode, mode)
            print(f'  {mode}: {count}次 — {desc}')
    else:
        print('  无失败')
    
    # 4. 关键发现
    print(f'\n--- 关键发现 ---')
    
    # 发现1: Agent主观判断 vs 纯模型信号
    agent_override_fails = sum(1 for a in attributions if 'agent_override' in a.get('failure_modes', []))
    model_only_correct = sum(1 for a in attributions if 'model_signal_only' in a.get('success_modes', []))
    print(f'  Agent主观加戏失败: {agent_override_fails}次')
    print(f'  纯模型信号成功: {model_only_correct}次')
    
    if agent_override_fails > 0:
        print(f'  → Agent主观判断在破坏模型价值')
    
    # 发现2: 高置信度反转
    high_conf = [a for a in attributions if a['confidence'] > 0.7]
    low_conf = [a for a in attributions if a['confidence'] <= 0.5]
    if high_conf and low_conf:
        high_rate = sum(1 for a in high_conf if a['is_correct']) / len(high_conf)
        low_rate = sum(1 for a in low_conf if a['is_correct']) / len(low_conf)
        if high_rate < low_rate:
            print(f'  ⚠️ 置信度反转: 高置信度{high_rate:.0%} < 低置信度{low_rate:.0%}')

def main():
    date_filter = None
    report_mode = '--report' in sys.argv
    
    if '--date' in sys.argv:
        idx = sys.argv.index('--date')
        date_filter = sys.argv[idx + 1]
    
    attributions = run_attribution(date_filter)
    
    if report_mode or not date_filter:
        generate_report(attributions)

if __name__ == '__main__':
    main()
