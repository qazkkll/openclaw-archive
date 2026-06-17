#!/usr/bin/env python3
"""
mem0_seed_all.py — 全量重建mem0向量记忆库

数据源：
1. experience_log.jsonl — 经验教训
2. decision_history.jsonl — 决策记录
3. bug_log.md — Bug记录
4. memory/*.md — 最近30天日记

用法：python scripts/mem0_seed_all.py
首次跑或清库后跑一次。后续靠 mem0_integration.py --seed 增量补充。
"""
import sys, json, os, glob, hashlib
sys.stdout.reconfigure(encoding='utf-8')

from mem0 import Memory

WORKSPACE = os.environ.get('OPENCLAW_WORKSPACE', r'/home/hermes/.hermes/openclaw-archive')
MEM0_PATH = r'/home/hermes/.hermes/openclaw-archive/data\mem0_persistent'

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
total = 0
seen_hashes = set()  # 内容hash去重

def content_hash(text):
    """生成内容hash用于去重"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]

def safe_add(summary, user_id, metadata, min_len=30):
    """安全添加：检查长度+去重"""
    global total
    if len(summary) < min_len:
        return False
    
    h = content_hash(summary)
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    
    m.add(summary, user_id=user_id, metadata=metadata)
    total += 1
    return True

# ── 1. 经验日志 ──
exp_path = os.path.join(WORKSPACE, 'data', 'experience_log.jsonl')
if os.path.exists(exp_path):
    with open(exp_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = [l.strip() for l in f if l.strip()]
    count = 0
    for line in raw:
        try:
            rec = json.loads(line)
            event = rec.get('event', '').strip()
            lesson = rec.get('lesson', '').strip()
            fix = rec.get('fix', rec.get('action', '')).strip()
            date = rec.get('date', '')
            # 组合成有意义的摘要
            summary = f'[{date}] 经验: {event} | 教训: {lesson} | 措施: {fix}'
            if safe_add(summary, 'system', {'source': 'experience_log', 'date': date, 'type': 'lesson'}):
                count += 1
        except:
            pass
    print(f'经验日志: {count}/{len(raw)} 条')

# ── 2. 决策历史 ──
dec_path = os.path.join(WORKSPACE, 'data', 'decision_history.jsonl')
if os.path.exists(dec_path):
    with open(dec_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = [l.strip() for l in f if l.strip()]
    count = 0
    for line in raw:
        try:
            rec = json.loads(line)
            code = rec.get('code', rec.get('stock', '')).strip()
            action = rec.get('action', rec.get('recommendation', '')).strip()
            reason = rec.get('reason', rec.get('rationale', '')).strip()
            date = rec.get('date', '')
            sender = rec.get('sender_name', '')
            summary = f'[{date}] 决策: {code} → {action} | 理由: {reason}'
            if safe_add(summary, 'trading', {'source': 'decision', 'code': code, 'date': date, 'type': 'decision'}):
                count += 1
        except:
            pass
    print(f'决策历史: {count}/{len(raw)} 条')

# ── 3. Bug日志 ──
bug_path = os.path.join(WORKSPACE, 'data', 'bug_log.md')
if os.path.exists(bug_path):
    with open(bug_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # 按 ### BUG-XXX 拆分
    sections = content.split('### ')
    count = 0
    for sec in sections:
        sec = sec.strip()
        if not sec or len(sec) < 50:
            continue
        lines = sec.split('\n')
        title = lines[0][:100]
        # 提取状态和描述
        body_lines = [l for l in lines[1:] if l.strip() and not l.startswith('---')]
        body = ' '.join(body_lines)[:400]
        summary = f'Bug记录: {title} | {body}'
        if safe_add(summary, 'system', {'source': 'bug_log', 'type': 'bug'}, min_len=50):
            count += 1
    print(f'Bug日志: {count} 条')

# ── 4. 日记（最近30天） ──
memory_dir = os.path.join(WORKSPACE, 'memory')
if os.path.exists(memory_dir):
    md_files = sorted(glob.glob(os.path.join(memory_dir, '20*.md')), reverse=True)[:30]
    count = 0
    for fpath in md_files:
        try:
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            # 提取关键内容（跳过空行和分隔线）
            lines = [l.strip() for l in content.split('\n') 
                     if l.strip() and not l.startswith('---') and len(l.strip()) > 10]
            body = ' '.join(lines)[:600]
            fname = os.path.basename(fpath).replace('.md', '')
            summary = f'日记 {fname}: {body}'
            if safe_add(summary, 'system', {'source': 'diary', 'date': fname, 'type': 'diary'}, min_len=80):
                count += 1
        except:
            pass
    print(f'日记: {count} 条（最近30天）')

print(f'\n✅ 共喂入 {total} 条记忆到mem0（已去重）')
