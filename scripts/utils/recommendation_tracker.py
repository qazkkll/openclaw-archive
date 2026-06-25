#!/usr/bin/env python3
"""
建议追踪器 v2 — 记录所有预测+建议，支持评分和规则提取
用法：
  python3 recommendation_tracker.py add '{...}'           # 添加建议
  python3 recommendation_tracker.py score DATE SCORE      # 评分
  python3 recommendation_tracker.py stats [days]          # 统计
  python3 recommendation_tracker.py pending               # 待评分列表
  python3 recommendation_tracker.py patterns [days]       # 分析模式
  python3 recommendation_tracker.py backfill              # 从旧predictions.json迁移
"""
import json, os, sys, hashlib
from datetime import datetime, timedelta

ROOT = os.path.expanduser('~/.hermes/openclaw-archive')
TRACK_FILE = os.path.join(ROOT, 'data/recommendations.json')
OLD_FILE = os.path.join(ROOT, 'data/predictions.json')

def load():
    if os.path.exists(TRACK_FILE):
        with open(TRACK_FILE) as f:
            return json.load(f)
    return {'recommendations': [], 'lessons': [], 'meta': {'created': datetime.now().isoformat(), 'version': 2}}

def save(data):
    os.makedirs(os.path.dirname(TRACK_FILE), exist_ok=True)
    with open(TRACK_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def make_id(entry):
    """生成唯一ID"""
    raw = f"{entry.get('date','')}{entry.get('target','')}{entry.get('source','')}"
    return 'R' + hashlib.md5(raw.encode()).hexdigest()[:8]

def validate_recommendation(entry):
    """校验推荐，返回(is_valid, violations, fixed_entry)"""
    import re
    fixed = dict(entry)
    violations = []
    
    # 规则匹配关键词
    RULE_KEYWORDS = {
        'R001': ['latest_xgb', '红杉', 'XGB', 'xgb', 'top数组'],
        'R004': ['score', 'probability', 'prob', 'rank', '蓝盾', '绿箭', 'GG', '模型信号'],
        'R006': ['RSI', 'score', 'probability', '%', '倍'],
    }
    
    reasoning = entry.get('reasoning', '').lower()
    source = entry.get('source', '')
    confidence = entry.get('confidence', 0)
    action = entry.get('action', '')
    target = entry.get('target', '')
    
    # 1. rules_applied必须填写
    rules_source = entry.get('rules_source', '')
    if not entry.get('rules_applied'):
        inferred = []
        for rule_id, keywords in RULE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in reasoning:
                    inferred.append(rule_id)
                    break
        if inferred:
            fixed['rules_applied'] = inferred
            fixed['rules_source'] = 'auto_inferred'
        else:
            fixed['rules_applied'] = ['R004']  # 默认
            fixed['rules_source'] = 'auto_default'
        violations.append(f'rules_applied已自动回填: {fixed["rules_applied"]}')
    elif not rules_source:
        # Agent显式填写了rules_applied
        fixed['rules_source'] = 'agent_applied'
    
    # 2. close_review不应产生买入/卖出推荐
    if source == 'close_review' and action in ('buy', 'sell') and target != '大盘':
        violations.append('close_review不应产生买入/卖出推荐(R005)')
        return False, violations, fixed
    
    # 3. 高置信度需有量化依据
    if confidence > 0.6:
        has_number = bool(re.search(r'\d+\.?\d*', reasoning))
        if not has_number:
            fixed['confidence'] = 0.5
            violations.append(f'confidence降级: {confidence}→0.5 (无量化依据)')
    
    # 4. macro类型标记
    macro_targets = ['VIX', '大盘', 'SPY', 'QQQ', 'Semiconductor', 'DIA', 'IWM', 'S&P 500']
    if target in macro_targets:
        fixed['type'] = 'macro'
    
    return True, violations, fixed

def add_recommendation(entry):
    """添加一条建议/预测（自动校验rules_applied）
    entry格式：
    {
        "source": "morning_report" | "cron" | "session" | "manual",
        "market": "cn" | "us",
        "date": "20260622",         # 预测日期
        "target": "300260 新莱应材", # 标的
        "action": "buy" | "sell" | "hold" | "avoid",
        "direction": "bullish" | "bearish" | "neutral",
        "confidence": 0.65,
        "reasoning": "GG 70分 RSI温和",
        "rules_applied": ["micro_cap_filter"],  # 当时应用的规则
        "macro_context": "bear regime"          # 宏观背景
    }
    """
    # 后处理校验
    is_valid, violations, entry = validate_recommendation(entry)
    if violations:
        for v in violations:
            print(f"  ⚠️ {v}")
    if not is_valid:
        print(f"  ❌ 拒绝写入: {entry.get('target','?')}")
        return None
    
    data = load()
    rec_id = make_id(entry)
    
    # 去重：同一天同一标的同一来源不重复
    for i, r in enumerate(data['recommendations']):
        if r.get('date') == entry.get('date') and r.get('target') == entry.get('target') and r.get('source') == entry.get('source'):
            data['recommendations'][i].update(entry)
            save(data)
            print(f"Updated {rec_id}: {entry.get('target','?')}")
            return rec_id
    
    rec = {
        'id': rec_id,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',  # pending | scored | expired
        'outcome': None,
        'outcome_date': None,
        'actual_return': None,
        'score': None,        # 1=正确, 0.5=部分, 0=错误
    }
    rec.update(entry)
    data['recommendations'].append(rec)
    
    # 保留最近180天
    cutoff = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
    data['recommendations'] = [r for r in data['recommendations'] if r.get('date','') >= cutoff]
    save(data)
    print(f"Added {rec_id}: {entry.get('target','?')} ({entry.get('direction','?')})")
    return rec_id

def score_recommendation(rec_id, score, outcome=None, actual_return=None):
    """评分一条建议
    score: 1.0=正确, 0.5=部分准确, 0=错误
    """
    data = load()
    for r in data['recommendations']:
        if r['id'] == rec_id:
            r['status'] = 'scored'
            r['score'] = score
            r['outcome'] = outcome or ('correct' if score >= 0.5 else 'wrong')
            r['outcome_date'] = datetime.now().strftime('%Y-%m-%d')
            r['actual_return'] = actual_return
            save(data)
            print(f"Scored {rec_id}: {score} ({r.get('target','?')})")
            return
    print(f"Not found: {rec_id}")

def score_by_date(date_str, scores):
    """批量评分某天的建议
    scores: {"target": score, ...} 或 {"all": score}
    """
    data = load()
    count = 0
    for r in data['recommendations']:
        if r.get('date') == date_str and r.get('status') == 'pending':
            target = r.get('target', '')
            if 'all' in scores:
                score = scores['all']
            elif target in scores:
                score = scores[target]
            else:
                continue
            r['status'] = 'scored'
            r['score'] = score
            r['outcome'] = 'correct' if score >= 0.5 else 'wrong'
            r['outcome_date'] = datetime.now().strftime('%Y-%m-%d')
            count += 1
    save(data)
    print(f"Scored {count} recommendations for {date_str}")

def list_pending():
    """列出待评分的建议"""
    data = load()
    pending = [r for r in data['recommendations'] if r.get('status') == 'pending']
    if not pending:
        print("没有待评分的建议")
        return
    print(f"=== 待评分建议 ({len(pending)}条) ===\n")
    for r in pending:
        arrow = {'buy': '🟢', 'sell': '🔴', 'hold': '⚪', 'avoid': '🚫'}.get(r.get('action',''), '?')
        print(f"  {r['id']} | {r.get('date','')} | {arrow} {r.get('target','?')}")
        print(f"    {r.get('direction','?')} conf={r.get('confidence','?')} | {r.get('source','?')}")
        print(f"    {r.get('reasoning','')[:80]}")
        print()

def stats(days=30):
    """统计评分情况"""
    data = load()
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    recent = [r for r in data['recommendations'] if r.get('date','') >= cutoff]
    
    if not recent:
        print(f"最近{days}天没有建议记录")
        return
    
    total = len(recent)
    scored = [r for r in recent if r.get('status') == 'scored']
    pending = [r for r in recent if r.get('status') == 'pending']
    
    print(f"=== 最近{days}天建议统计 ===")
    print(f"总建议: {total}")
    print(f"已评分: {len(scored)}")
    print(f"待评分: {len(pending)}")
    
    if scored:
        avg_score = sum((r.get('score') or 0) for r in scored) / len(scored)
        correct = sum(1 for r in scored if (r.get('score') or 0) >= 0.5)
        print(f"平均分: {avg_score:.2f}")
        print(f"命中率: {correct}/{len(scored)} = {correct/len(scored)*100:.1f}%")
        
        # 按来源分
        by_source = {}
        for r in scored:
            src = r.get('source', 'unknown')
            by_source.setdefault(src, []).append((r.get('score') or 0))
        print(f"\n按来源:")
        for src, scores in sorted(by_source.items()):
            avg = sum(scores) / len(scores)
            hit = sum(1 for s in scores if s >= 0.5)
            print(f"  {src}: {len(scores)}条, 命中{hit}/{len(scores)}, 均分{avg:.2f}")
        
        # 按方向分
        by_dir = {}
        for r in scored:
            d = r.get('direction', 'unknown')
            by_dir.setdefault(d, []).append((r.get('score') or 0))
        print(f"\n按方向:")
        for d, scores in sorted(by_dir.items()):
            avg = sum(scores) / len(scores)
            hit = sum(1 for s in scores if s >= 0.5)
            print(f"  {d}: {len(scores)}条, 命中{hit}/{len(scores)}, 均分{avg:.2f}")

def analyze_patterns(days=30):
    """分析失败模式，供规则提取用"""
    data = load()
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    scored = [r for r in data['recommendations'] 
              if r.get('date','') >= cutoff and r.get('status') == 'scored']
    
    if len(scored) < 5:
        print(f"已评分数据不足（{len(scored)}条），需要至少5条才能分析模式")
        return
    
    print(f"=== 最近{days}天模式分析 ({len(scored)}条已评分) ===\n")
    
    # 1. 置信度校准
    print("【置信度校准】")
    brackets = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    for lo, hi in brackets:
        bracket = [r for r in scored if lo <= r.get('confidence', 0) < hi]
        if bracket:
            avg_score = sum((r.get('score') or 0) for r in bracket) / len(bracket)
            hit = sum(1 for r in bracket if (r.get('score') or 0) >= 0.5)
            print(f"  置信度{lo:.1f}-{hi:.1f}: {len(bracket)}条, 命中率{hit/len(bracket)*100:.0f}%, 实际均分{avg_score:.2f}")
    
    # 2. 按标的特征
    print("\n【标的特征】")
    micro_cap = [r for r in scored if '$' in r.get('target','') or any(c.isdigit() and int(c) < 5 for c in [r.get('target','').split('$')[-1].split('.')[0]] if c.isdigit())]
    # 简单判断：价格<$5的
    print(f"  (需要价格数据来分类微盘/大盘)")
    
    # 3. direction vs outcome
    print("\n【方向准确性】")
    for d in ['bullish', 'bearish', 'neutral']:
        dir_recs = [r for r in scored if r.get('direction') == d]
        if dir_recs:
            correct = sum(1 for r in dir_recs if (r.get('score') or 0) >= 0.5)
            print(f"  {d}: {correct}/{len(dir_recs)} = {correct/len(dir_recs)*100:.0f}%")
    
    # 4. source vs accuracy
    print("\n【来源准确性】")
    for src in set(r.get('source','') for r in scored):
        src_recs = [r for r in scored if r.get('source') == src]
        correct = sum(1 for r in src_recs if (r.get('score') or 0) >= 0.5)
        print(f"  {src}: {correct}/{len(src_recs)} = {correct/len(src_recs)*100:.0f}%")

def backfill():
    """从旧predictions.json迁移数据"""
    if not os.path.exists(OLD_FILE):
        print("没有旧predictions.json")
        return
    
    with open(OLD_FILE) as f:
        old = json.load(f)
    
    old_preds = old.get('predictions', old) if isinstance(old, dict) else old
    data = load()
    count = 0
    
    for entry in old_preds:
        date = entry.get('date', '')
        market = entry.get('market', '')
        source = entry.get('type', 'unknown')
        macro = entry.get('macro_context', '')
        
        for pred in entry.get('predictions', []):
            rec_id = make_id({'date': date, 'target': pred.get('target',''), 'source': source})
            # 跳过已存在的
            if any(r['id'] == rec_id for r in data['recommendations']):
                continue
            
            rec = {
                'id': rec_id,
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'market': market,
                'date': date,
                'target': pred.get('target', ''),
                'action': 'buy' if pred.get('direction') == 'bullish' else ('sell' if pred.get('direction') == 'bearish' else 'hold'),
                'direction': pred.get('direction', ''),
                'confidence': pred.get('confidence', 0),
                'reasoning': pred.get('reasoning', ''),
                'rules_applied': [],
                'macro_context': macro,
                'status': 'pending',
                'outcome': None,
                'outcome_date': None,
                'actual_return': None,
                'score': None,
            }
            data['recommendations'].append(rec)
            count += 1
    
    save(data)
    print(f"Backfilled {count} recommendations from old predictions.json")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: recommendation_tracker.py [add|score|pending|stats|patterns|backfill]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    if cmd == 'add':
        entry = json.loads(sys.argv[2])
        add_recommendation(entry)
    elif cmd == 'score':
        rec_id = sys.argv[2]
        score_val = float(sys.argv[3])
        outcome = sys.argv[4] if len(sys.argv) > 4 else None
        score_recommendation(rec_id, score_val, outcome)
    elif cmd == 'pending':
        list_pending()
    elif cmd == 'stats':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        stats(days)
    elif cmd == 'patterns':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        analyze_patterns(days)
    elif cmd == 'backfill':
        backfill()
    else:
        print(f"未知命令: {cmd}")
