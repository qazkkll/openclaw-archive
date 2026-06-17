#!/usr/bin/env python3
"""
🍤 每2小时全量刷新Top100候选
盘中重新评分全部1,444只，更新Top100列表供盘中扫描用
"""
import sys, json, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from score_engine import v1_score_from_data
from data_source import AShareKline, AShareRealtime, code_to_board

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUALITY_POOL = os.path.join(ROOT, 'data', 'quality_pool.json')
TOP100_FILE = os.path.join(ROOT, 'data', 'morning_top100.json')

# 🔥 主观推荐配置（每次刷新时手动更新）
FIRE_PICKS = {}  # 格式: 'code': '一句话判断'

kl = AShareKline()

def refresh():
    t0 = time.time()
    
    with open(QUALITY_POOL) as f:
        pool = json.load(f)
    
    all_stocks = pool.get('stocks', [])
    print(f'🔄 全量评分刷新: {len(all_stocks)}只...', flush=True)
    
    results = []
    errors = 0
    
    for i, s in enumerate(all_stocks):
        code = s['code']
        try:
            data = kl.get_best(code)
            if not data or len(data) < 60:
                errors += 1
                continue
            close = [d['close'] for d in data]
            high = [d['high'] for d in data]
            low = [d['low'] for d in data]
            score = v1_score_from_data(close, high, low)
            if score is None:
                errors += 1
                continue
            
            board = s.get('board', code_to_board(code))
            results.append({
                'code': code,
                'name': s.get('name', code),
                'board': board,
                'score': round(float(score), 0),
                'price': close[-1],
                'change_pct': 0
            })
        except:
            errors += 1
            continue
        
        if (i + 1) % 300 == 0:
            pct = (i + 1) / len(all_stocks) * 100
            print(f'  {pct:.0f}% | 有效: {len(results)} | 跳过: {errors}', flush=True)
    
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 保存当前Top100为prev（供下次对比）
    PREV_FILE = TOP100_FILE.replace('.json', '_prev.json')
    if os.path.exists(PREV_FILE):
        import shutil
        shutil.copy2(TOP100_FILE, TOP100_FILE.replace('.json', '_prev.json'))
    
    # 保存新Top100
    top100 = [{'code': r['code'], 'name': r['name'], 'board': r['board'], 'score': r['score']}
              for r in results[:100]]
    with open(TOP100_FILE, 'w') as f:
        json.dump(top100, f, ensure_ascii=False)
    
    elapsed = time.time() - t0
    print(f'✅ Top100刷新完成 ({elapsed:.0f}s) | 有效: {len(results)}/{len(all_stocks)}', flush=True)
    
    # 读取上次Top100做比较
    prev_top = {}
    PREV_FILE = TOP100_FILE.replace('.json', '_prev.json')
    if os.path.exists(PREV_FILE):
        try:
            with open(PREV_FILE) as f:
                prev_list = json.load(f)
            prev_top = {s['code']: s['score'] for s in prev_list}
        except:
            pass
    
    # 当前Top100
    curr_top = {s['code']: s['score'] for s in results[:100]}
    
    # 新进入视野的（之前不在Top100或评分大幅提升）
    new_entries = [s for s in results[:30] if s['code'] not in prev_top or 
                   (s['code'] in prev_top and s['score'] - prev_top[s['code']] >= 5)]
    
    # 输出结果
    print(f'')
    print(f'🏆 Top30 更新:')
    print(f'')
    
    for i, s in enumerate(results[:30]):
        code = s['code']
        name = s['name']
        score = s['score']
        board = s['board']
        
        # 板块标记
        board_mark = '📊' if '主板' in board else '📈' if '创业板' in board else '📡'
        
        # 评分变化
        old_score = prev_top.get(code)
        score_chg = ''
        if old_score is not None:
            diff = score - old_score
            if diff > 0:
                score_chg = f'(+{diff:.0f}⬆)' 
            elif diff < 0:
                score_chg = f'({diff:.0f}⬇)'
            else:
                score_chg = '(→)'
        else:
            score_chg = '(新增)'
        
        # 标记 - 机械标记
        mark = ''
        if code not in prev_top:
            mark = '🆕'  # 新进入视野
        
        name_str = f'{board_mark} {name}'
        if '主板' in board:
            name_str += ' → Andy'
        else:
            name_str += ' → 妈妈'
        
        print(f'  #{i+1} {mark} {name_str} {score:.0f}分 {score_chg}', flush=True)
    
    print(f'')
    # 🔥 主观分析区（这里我手动填，不机械标记）
    # 每次刷新时在这里加入我的主观判断
    # 格式: code -> 一句话判断
    fire_picks = FIRE_PICKS or {
        # 示例: '600719': '电力板块走强+评分71，等回调到¥9.5附近可入场',
        # 示例: '603685': '晨丰科技趋势刚起，等放量确认',
    }
    
    # 输出结论
    print(f'')
    if fire_picks:
        print(f'🔥 主观推荐 — 值得买:')
        for code, verdict in fire_picks.items():
            # 找对应股票信息
            match = next((s for s in results if s['code'] == code), None)
            if match:
                board = match['board']
                mark = '📊' if '主板' in board else '📈'
                owner = 'Andy' if '主板' in board else '妈妈'
                print(f'  {mark} {match["name"]} ({code}) {match["score"]}分 → {owner}')
                print(f'    {verdict}')
        print(f'')
    
    # 机械候选供参考
    print(f'📋 评分达标候选（供参考，需主观判断）:')
    for s in results[:10]:
        if s['score'] >= 62:
            board = s['board']
            mark = '📊' if '主板' in board else '📈'
            print(f'  {mark} {s["name"]} ({s["code"]}) {s["score"]}分')
    print(f'')

    # 审计记录
    try:
        from audit_engine import audit
        valid = len(results)
        level = 'success' if valid >= 100 else 'error' if valid < 10 else 'warning'
        audit('refresh_top100', level, f'Top100刷新: {valid}只有效/{len(all_stocks)}总, {errors}跳过, {elapsed:.0f}s')
    except: pass

if __name__ == '__main__':
    refresh()

    # 审计记录
    try:
        from audit_engine import audit
        valid = len(results)
        level = 'success' if valid >= 100 else 'error' if valid < 10 else 'warning'
        audit('refresh_top100', level, f'Top100刷新: {valid}只有效/{len(all_stocks)}总, {errors}跳过, {elapsed:.0f}s')
    except: pass

if __name__ == '__main__':
    refresh()
