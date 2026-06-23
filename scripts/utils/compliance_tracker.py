#!/usr/bin/env python3
"""
合规率趋势追踪 — 跟踪Agent是否在改善规则遵守
用法：
  python3 compliance_tracker.py              # 输出当前合规率
  python3 compliance_tracker.py --trend      # 输出趋势
  python3 compliance_tracker.py --snapshot   # 保存快照（供趋势分析）
"""
import json, os, sys
from datetime import datetime
from collections import defaultdict

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
SNAPSHOT_FILE = os.path.join(ROOT, 'data/compliance_snapshots.json')

def load_recs():
    with open(TRACK_FILE) as f:
        return json.load(f)['recommendations']

def load_snapshots():
    if os.path.exists(SNAPSHOT_FILE):
        with open(SNAPSHOT_FILE) as f:
            return json.load(f)
    return {'snapshots': []}

def save_snapshots(data):
    with open(SNAPSHOT_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_compliance(date_filter=None):
    """计算合规率"""
    recs = load_recs()
    if date_filter:
        recs = [r for r in recs if r.get('date') == date_filter]
    
    total = len(recs)
    if total == 0:
        return None
    
    # rules_source统计
    agent_applied = sum(1 for r in recs if r.get('rules_source') == 'agent_applied')
    auto_inferred = sum(1 for r in recs if r.get('rules_source') == 'auto_inferred')
    auto_default = sum(1 for r in recs if r.get('rules_source') == 'auto_default')
    unknown = total - agent_applied - auto_inferred - auto_default
    
    # 完整rules_applied
    has_rules = sum(1 for r in recs if r.get('rules_applied'))
    
    # close_review违规
    close_review_recs = [r for r in recs if r.get('source') == 'close_review']
    close_review_with_buy = sum(1 for r in close_review_recs if r.get('action') in ('buy', 'sell'))
    
    return {
        'total': total,
        'agent_compliance': agent_applied / total * 100,
        'system_coverage': has_rules / total * 100,
        'agent_applied': agent_applied,
        'auto_inferred': auto_inferred,
        'auto_default': auto_default,
        'unknown': unknown,
        'close_review_violations': close_review_with_buy,
    }

def snapshot():
    """保存当前合规率快照"""
    compliance = compute_compliance()
    if not compliance:
        print('无数据可快照')
        return
    
    snapshots = load_snapshots()
    
    # 按日期去重
    today = datetime.now().strftime('%Y-%m-%d')
    snapshots['snapshots'] = [s for s in snapshots['snapshots'] if s['date'] != today]
    
    compliance['date'] = today
    compliance['timestamp'] = datetime.now().isoformat()
    snapshots['snapshots'].append(compliance)
    snapshots['snapshots'].sort(key=lambda x: x['date'])
    
    save_snapshots(snapshots)
    print(f'✅ 快照已保存: {today} | Agent合规率{compliance["agent_compliance"]:.0f}% | 系统覆盖率{compliance["system_coverage"]:.0f}%')

def trend():
    """输出趋势"""
    snapshots = load_snapshots()
    if not snapshots['snapshots']:
        print('无历史快照')
        return
    
    print('=== 合规率趋势 ===')
    print(f'{"日期":12s} {"Agent合规率":>12s} {"系统覆盖率":>12s} {"总推荐":>8s} {"Agent自觉":>8s} {"系统回填":>8s}')
    print('-' * 65)
    
    for s in snapshots['snapshots']:
        emoji = '✅' if s['agent_compliance'] >= 80 else '⚠️' if s['agent_compliance'] >= 50 else '❌'
        print(f'{emoji} {s["date"]:10s} {s["agent_compliance"]:>10.0f}% {s["system_coverage"]:>10.0f}% {s["total"]:>8d} {s["agent_applied"]:>8d} {s["auto_inferred"]:>8d}')
    
    # 趋势判断
    if len(snapshots['snapshots']) >= 2:
        first = snapshots['snapshots'][0]['agent_compliance']
        last = snapshots['snapshots'][-1]['agent_compliance']
        if last > first + 10:
            print(f'\n📈 趋势改善: {first:.0f}% → {last:.0f}%')
        elif last < first - 10:
            print(f'\n📉 趋势恶化: {first:.0f}% → {last:.0f}%')
        else:
            print(f'\n➡️ 趋势稳定: {first:.0f}% → {last:.0f}%')
    
    # 目标
    current = snapshots['snapshots'][-1]['agent_compliance']
    target = 80
    gap = target - current
    if gap > 0:
        print(f'距离目标({target}%): 还差{gap:.0f}%')
    else:
        print(f'✅ 已达目标({target}%)')

def main():
    if '--snapshot' in sys.argv:
        snapshot()
    elif '--trend' in sys.argv:
        trend()
    else:
        compliance = compute_compliance()
        if compliance:
            print(f'当前合规率:')
            print(f'  Agent自觉执行率: {compliance["agent_applied"]}/{compliance["total"]} = {compliance["agent_compliance"]:.0f}%')
            print(f'  系统覆盖率: {compliance["system_coverage"]:.0f}%')
            print(f'  close_review违规: {compliance["close_review_violations"]}条')
        else:
            print('无数据')

if __name__ == '__main__':
    main()
