#!/usr/bin/env python3
"""
🍤 市场模式检查 — 判断当前处在牛市还是熊市

逻辑:
  全市场V1评分≥50的占比
    >25% 连续2次检查 → 切牛市
    <15% 连续2次检查 → 切熊市
    15-25% → 死区，维持不变

用法:
  python3 scripts/market_mode_check.py          # 检查+更新状态
  python3 scripts/market_mode_check.py --dry    # 只看不写
  python3 scripts/market_mode_check.py --status # 只看当前状态
"""
import sys, os, json, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

STATE_FILE = os.path.join(ROOT, 'data', 'market_mode.json')

# 切换阈值
BULL_THRESHOLD = 25    # >25% → 牛市候选
BEAR_THRESHOLD = 15    # <15% → 熊市候选
DEAD_ZONE_LOW = 15
DEAD_ZONE_HIGH = 25
REQUIRED_CONSECUTIVE = 2  # 连续2次确认才切换
DAYS_BETWEEN_CHECKS = 7  # 与调仓同步


def get_default_state():
    return {
        'mode': '牛市',
        'mode_since': '2026-05-20',
        'last_check_pct': None,
        'last_check_date': None,
        'consecutive_bull_checks': 0,
        'consecutive_bear_checks': 0,
        'history': [],
        'oscillation_protection': {
            'required_consecutive': REQUIRED_CONSECUTIVE,
            'dead_zone': {'low': DEAD_ZONE_LOW, 'high': DEAD_ZONE_HIGH},
            'days_per_check': DAYS_BETWEEN_CHECKS
        },
        'updated_at': None
    }


def load_state():
    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        return state
    except:
        return get_default_state()


def save_state(state):
    state['updated_at'] = datetime.datetime.now().isoformat()
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def scan_market_strength():
    """
    扫描全市场，计算V1评分≥50的占比。
    返回 {pct, sampled, above_50, total}
    """
    from score_engine import v1_score_from_data
    from data_source import AShareKline

    kl = AShareKline()
    
    # 读取质量池
    pool_path = os.path.join(ROOT, 'data', 'quality_pool.json')
    with open(pool_path) as f:
        import json as j
        pool = j.load(f)
    
    stocks = pool.get('stocks', [])
    # 取Top 200（按活跃度排列）
    sample = stocks[:200]
    
    above = 0
    total = 0
    
    for s in sample:
        code = s['code']
        try:
            data = kl.get_kline(code, 120, source='sina')
            if data and len(data) >= 60:
                close = [d['close'] for d in data]
                high = [d['high'] for d in data]
                low = [d['low'] for d in data]
                sc = v1_score_from_data(close, high, low)
                total += 1
                if sc >= 50:
                    above += 1
        except:
            pass
    
    pct = (above / total * 100) if total > 0 else 0
    return {
        'pct': round(pct, 1),
        'sampled': len(sample),
        'above_50': above,
        'total_scored': total
    }


def check_and_switch(dry_run=False):
    state = load_state()
    today = datetime.date.today().isoformat()
    
    # 如果今天已经检查过，跳过
    if state.get('last_check_date') == today:
        print(f"今日已检查({today})，跳过")
        return state
    
    # 扫描市场
    result = scan_market_strength()
    pct = result['pct']
    
    print(f"市场强度: {result['above_50']}/{result['total_scored']} = {pct}%")
    
    old_mode = state['mode']
    signal = None
    
    if pct > BULL_THRESHOLD:
        signal = 'bull'
        state['consecutive_bull_checks'] += 1
        state['consecutive_bear_checks'] = 0
        print(f"牛市信号(>{BULL_THRESHOLD}%) 第{state['consecutive_bull_checks']}次")
    elif pct < BEAR_THRESHOLD:
        signal = 'bear'
        state['consecutive_bear_checks'] += 1
        state['consecutive_bull_checks'] = 0
        print(f"熊市信号(<{BEAR_THRESHOLD}%) 第{state['consecutive_bear_checks']}次")
    else:
        print(f"死区({DEAD_ZONE_LOW}-{DEAD_ZONE_HIGH}%) 维持当前模式")
        # 在死区不累积连续计数，但也不清零（避免死区出来时重新计数）
    
    # 判断是否切换
    switched = False
    if state['consecutive_bull_checks'] >= REQUIRED_CONSECUTIVE and old_mode != '牛市':
        state['mode'] = '牛市'
        state['mode_since'] = today
        state['consecutive_bull_checks'] = 0
        switched = True
        print(f"🚀 切换至牛市模式（连续{REQUIRED_CONSECUTIVE}次确认）")
    elif state['consecutive_bear_checks'] >= REQUIRED_CONSECUTIVE and old_mode != '熊市':
        state['mode'] = '熊市'
        state['mode_since'] = today
        state['consecutive_bear_checks'] = 0
        switched = True
        print(f"🛡️ 切换至熊市模式（连续{REQUIRED_CONSECUTIVE}次确认）")
    else:
        needed = REQUIRED_CONSECUTIVE - max(state['consecutive_bull_checks'], state['consecutive_bear_checks'])
        print(f"维持{old_mode}模式（还需{needed}次确认）")
    
    # 记录历史
    state['last_check_pct'] = pct
    state['last_check_date'] = today
    
    state['history'].append({
        'date': today,
        'pct': pct,
        'sampled': result['sampled'],
        'above_50': result['above_50'],
        'total_scored': result['total_scored'],
        'signal': signal,
        'mode': state['mode'],
        'switched': switched
    })
    
    # 最多保留50条历史
    if len(state['history']) > 50:
        state['history'] = state['history'][-50:]
    
    if not dry_run:
        save_state(state)
    
    # 如果切换了，发通知
    if switched and not dry_run:
        try:
            from notify import send
            msg = (
                f"{'🚀' if state['mode']=='牛市' else '🛡️'} 市场模式切换: {old_mode} → {state['mode']}\n"
                f"市场强度: {pct}% (阈值{BEAR_THRESHOLD}%-{BULL_THRESHOLD}%)\n"
                f"({REQUIRED_CONSECUTIVE}次连续确认后触发)"
            )
            send(msg)
        except:
            pass
    
    return state


def show_status():
    state = load_state()
    print(f"当前模式: {state['mode']}")
    print(f"起始日期: {state.get('mode_since', '?')}")
    print(f"上次检查: {state.get('last_check_date', '从未')} ({state.get('last_check_pct', '?')}%)")
    print(f"连续牛市计数: {state['consecutive_bull_checks']}")
    print(f"连续熊市计数: {state['consecutive_bear_checks']}")
    print(f"历史记录: {len(state['history'])}条")
    
    if state['history']:
        print("\n最近5次:")
        for h in state['history'][-5:]:
            print(f"  {h['date']} | {h['pct']}% | {h['mode']} {'🔀' if h.get('switched') else ''}")


if __name__ == '__main__':
    if '--status' in sys.argv:
        show_status()
    elif '--dry' in sys.argv:
        check_and_switch(dry_run=True)
    else:
        check_and_switch()
