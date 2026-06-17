#!/usr/bin/env python3
"""
mem0_integration.py — mem0长期记忆：搜索 + 增量seed + 开机简报

用法：
  --seed           # 增量补充新经验（只灌新的，不重复）
  --query Q        # 搜索相关记忆
  --brief [N]      # 生成开机简报（默认5条关键记忆）
  --rerun          # 强刷：重新灌入全部数据（清库后使用）

    建议 cron:
  工作日 05:55 --brief    # 生成开机简报
  工作日 06:00 --seed     # 增量喂新经验
"""

import sys, json, os, argparse, hashlib, datetime
sys.stdout.reconfigure(encoding='utf-8')

from mem0 import Memory

WORKSPACE = os.environ.get('OPENCLAW_WORKSPACE', r'/home/hermes/.hermes/openclaw-archive')
MEM0_PATH = r'/home/hermes/.hermes/openclaw-archive/data\mem0_persistent'
TRACKER_PATH = os.path.join(WORKSPACE, 'data', 'mem0_seed_tracker.json')

config = {
    'vector_store': {
        'provider': 'qdrant',
        'config': {
            'path': MEM0_PATH,
            'on_disk': True,
            'embedding_model_dims': 1024
        }
    },
    'llm': {
        'provider': 'ollama',
        'config': {
            'model': 'qwen2.5:3b',
            'ollama_base_url': 'http://localhost:11434',
            'temperature': 0.1
        }
    },
    'embedder': {
        'provider': 'ollama',
        'config': {
            'model': 'mxbai-embed-large',
            'ollama_base_url': 'http://localhost:11434'
        }
    }
}

m = Memory.from_config(config)


# ── 行号追踪 ──
def load_tracker():
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'experience_log': 0, 'decision_history': 0}


def save_tracker(tracker):
    with open(TRACKER_PATH, 'w', encoding='utf-8') as f:
        json.dump(tracker, f, indent=2, ensure_ascii=False)


def content_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]


# ── 检查是否已存在相似记忆 ──
def exists_already(summary, threshold=0.92):
    """通过语义搜索检查是否已有高度相似的记忆"""
    try:
        results = m.search(summary[:100], limit=3)
        for r in results.get('results', []):
            if r.get('score', 0) >= threshold:
                return True
    except:
        pass
    return False


# ── 增量seed：经验日志 ──
def seed_experience_log(rerun=False):
    exp_path = os.path.join(WORKSPACE, 'data', 'experience_log.jsonl')
    if not os.path.exists(exp_path):
        return 0
    tracker = load_tracker()
    start_line = 0 if rerun else tracker.get('experience_log', 0)
    with open(exp_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [l.strip() for l in f if l.strip()]
    new_lines = lines[start_line:]
    if not new_lines:
        return 0
    count = 0
    for line in new_lines:
        try:
            rec = json.loads(line)
            event = rec.get('event', '').strip()
            lesson = rec.get('lesson', '').strip()
            fix = rec.get('fix', rec.get('action', '')).strip()
            date = rec.get('date', '')
            summary = f'[{date}] 经验: {event} | 教训: {lesson} | 措施: {fix}'
            if len(summary) < 30:
                continue
            if not rerun and exists_already(summary):
                continue
            m.add(summary, user_id='system', metadata={'source': 'experience_log', 'date': date, 'type': 'lesson'})
            count += 1
        except:
            pass
    tracker['experience_log'] = len(lines)
    save_tracker(tracker)
    print(f'经验日志: 新增{count}条（现存{len(lines)}条, 上次seed到第{start_line}行）')
    return count


# ── 增量seed：决策历史 ──
def seed_decision_history(rerun=False):
    dec_path = os.path.join(WORKSPACE, 'data', 'decision_history.jsonl')
    if not os.path.exists(dec_path):
        return 0
    tracker = load_tracker()
    start_line = 0 if rerun else tracker.get('decision_history', 0)
    with open(dec_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [l.strip() for l in f if l.strip()]
    new_lines = lines[start_line:]
    if not new_lines:
        return 0
    count = 0
    for line in new_lines:
        try:
            rec = json.loads(line)
            code = rec.get('code', rec.get('stock', '')).strip()
            action = rec.get('action', rec.get('recommendation', '')).strip()
            reason = rec.get('reason', rec.get('rationale', '')).strip()
            date = rec.get('date', '')
            summary = f'[{date}] 决策: {code} → {action} | 理由: {reason}'
            if len(summary) < 30:
                continue
            if not rerun and exists_already(summary):
                continue
            m.add(summary, user_id='trading', metadata={'source': 'decision', 'code': code, 'date': date, 'type': 'decision'})
            count += 1
        except:
            pass
    tracker['decision_history'] = len(lines)
    save_tracker(tracker)
    print(f'决策历史: 新增{count}条（现存{len(lines)}条, 上次seed到第{start_line}行）')
    return count


# ── 搜索 ──
def query(q, limit=5):
    """搜索相关记忆"""
    all_results = []
    for uid in ('system', 'trading'):
        try:
            r = m.search(q, filters={'user_id': uid}, limit=limit)
            all_results.extend(r.get('results', []))
        except:
            pass
    all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    return all_results[:limit]


# ── 简报 ──
def brief(n=5):
    """生成开机简报：搜索4个关键维度，汇总最相关的记忆"""
    probes = [
        'scoring data pipeline A shares US stocks technical analysis',
        'position risk management stop loss position sizing allocation',
        'bug error fix Windows encoding data stale model version',
        'trading recommendation Blue Shield Green Arrow scoring model',
        'lesson experience avoid repeated mistakes strategy improvement',
    ]
    seen = set()
    items = []
    for probe in probes:
        for uid in ('system', 'trading'):
            try:
                r = m.search(probe, filters={'user_id': uid}, limit=5)
                for item in r.get('results', []):
                    mem = item.get('memory', '')
                    score = item.get('score', 0)
                    if mem not in seen and score > 0.25:
                        seen.add(mem)
                        items.append({'memory': mem[:200], 'score': round(score, 3)})
            except:
                pass
    items.sort(key=lambda x: x['score'], reverse=True)
    return items[:n]


def main():
    parser = argparse.ArgumentParser(description='mem0记忆系统')
    parser.add_argument('--seed', action='store_true')
    parser.add_argument('--rerun', action='store_true')
    parser.add_argument('--query', type=str)
    parser.add_argument('--brief', nargs='?', const=5, type=int)
    args = parser.parse_args()

    if args.seed or args.rerun:
        total = 0
        total += seed_experience_log(rerun=args.rerun)
        total += seed_decision_history(rerun=args.rerun)
        print(f'Seed完成: 共新增{total}条记忆')

    if args.brief:
        items = brief(args.brief)
        brief_path = os.path.join(WORKSPACE, 'data', 'mem0_brief.json')
        with open(brief_path, 'w', encoding='utf-8') as f:
            json.dump({
                'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'memories': items
            }, f, indent=2, ensure_ascii=False)
        print(f'简报已写入: {len(items)}条记忆 → {brief_path}')
        for item in items:
            print(f'  [{item["score"]:.3f}] {item["memory"][:80]}...')

    if args.query:
        results = query(args.query)
        print(f'搜索结果: {args.query}')
        print('-' * 60)
        for r in results:
            mem = r.get('memory', '?')
            score = r.get('score', 0)
            print(f'  [{score:.3f}] {mem}')
        print('-' * 60)


if __name__ == '__main__':
    main()
