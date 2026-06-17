#!/usr/bin/env python3
"""
蓝盾3.0 — SP500全量评分 (yfinance实时数据)
速度优先: 不依赖本地parquet, 直接用yfinance拉60-250天数据
单只0.3-1秒, 500只约3-5分钟
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from us_score_engine import v5s_calc, v5s_score

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
ENTRY_THRESHOLD = 90
EXIT_THRESHOLD = 75
MAX_POSITIONS = 10
STOP_LOSS = 0.15
SP500_FILE = f'{DATA_DIR}/sp500_symbols.json'
EXTRA_POOL = ['HPK', 'MRDN']  # 不在SP500的持仓

def load_sp500_list():
    """加载SP500全量+扩展池"""
    if os.path.exists(SP500_FILE):
        with open(SP500_FILE) as f:
            sp500 = json.load(f)
    else:
        print('  SP500列表不存在, 用硬编码30只保底')
        sp500 = [
            'AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','AVGO','TSLA','META','LLY',
            'JPM','V','XOM','WMT','PG','JNJ','UNH','HD','BAC','KO','ABBV','AMD',
            'TXN','ADBE','CRM','GE','GS','DIS','T','NFLX'
        ]
    all_target = list(set(sp500 + EXTRA_POOL))
    all_target.sort()
    return all_target

def score_one(code):
    """yfinance拉数据→评分(含RSI+52周高%)"""
    try:
        ticker = yf.Ticker(code)
        hist = ticker.history(period='2y')  # 2年够算52周高
        if len(hist) < 120:
            return None, None, None, None
        
        c = hist['Close'].astype(float).tolist()
        h = hist['High'].astype(float).tolist()
        l = hist['Low'].astype(float).tolist()
        # v5s_calc要252+天才能正确算p52, 传全量
        
        ind = v5s_calc(c, h, l)
        if ind is None:
            return None, None, None, None
        s = v5s_score(ind, len(c)-1)
        # 取最新RSI和52周高位置%
        rsi_val = ind['rsi'][-1] if ind['rsi'] and ind['rsi'][-1] is not None else 0
        p52_val = ind['p52'][-1] if ind['p52'] and ind['p52'][-1] is not None else 0
        return int(round(s)), float(c[-1]), round(rsi_val, 1), round(p52_val, 1)
    except:
        return None, None, None, None

def main():
    t0 = time.time()
    today = time.strftime('%Y-%m-%d')
    print(f'蓝盾3.0 SP500全量评分 — {today}')
    print('=' * 50)
    
    pool = load_sp500_list()
    print(f'候选池: {len(pool)} 只 (SP500 {len(pool)-len(EXTRA_POOL)}只 + 扩展{len(EXTRA_POOL)}只)')
    print(f'评分中 (每只~1秒)...')
    
    scores = []
    failed = 0
    for i, code in enumerate(pool):
        s, p, r, pct52 = score_one(code)
        if s is not None:
            scores.append({'code': code, 'score': s, 'price': p, 'rsi': r, 'pct52': pct52})
        else:
            failed += 1
        
        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            remain = (len(pool) - i - 1) / rate if rate > 0 else 0
            print(f'  [{i+1}/{len(pool)}] 成功{len(scores)}/{failed}失败 ETA{remain:.0f}s')
    
    scores.sort(key=lambda x: -x['score'])
    
    print(f'\n评分完成: {len(scores)}/{len(pool)} 只成功, {failed} 只失败')
    print(f'  🟢💪 强势买入 (≥{ENTRY_THRESHOLD}): {sum(1 for s in scores if s["score"] >= ENTRY_THRESHOLD)}只')
    print(f'  🟢  持有 (≥{EXIT_THRESHOLD}): {sum(1 for s in scores if EXIT_THRESHOLD <= s["score"] < ENTRY_THRESHOLD)}只')
    print(f'  🟡  观望 (<{EXIT_THRESHOLD}): {sum(1 for s in scores if s["score"] < EXIT_THRESHOLD)}只')
    
    # 持仓自查 — 从OpenD实时拉取，不再硬编码
    # 路径: scripts/_futu_opend.py → get_codes_only() → ['NVDA', 'ON', ...]
    try:
        from _futu_opend import get_codes_only
        held = get_codes_only(silent=True)
        if not held:
            print('⚠️ OpenD无持仓数据，使用缓存/空列表')
    except Exception as e:
        print(f'⚠️ OpenD连接失败({e})，使用缓存持仓')
        held = ['NVDA', 'ON', 'GNRC', 'HPK', 'MRDN']  # fallback缓存
    print(f'\n持仓评分:')
    for code in held:
        for s in scores:
            if s['code'] == code:
                tag = '++' if s['score'] >= 80 else '+' if s['score'] >= 70 else '-' 
                rsi_str = f'RSI{s["rsi"]:.0f}' if s.get('rsi') else 'RSI?'
                p52_str = f'P52{s["pct52"]:.0f}%' if s.get('pct52') else 'P52?'
                print(f'  {code}: {tag} 评分{s["score"]}  ${s["price"]:.2f}  {rsi_str}  {p52_str}')
                break
        else:
            print(f'  {code}: 评分失败/无数据')
    
    # 买入推荐
    buys = [s for s in scores if s['score'] >= ENTRY_THRESHOLD]
    if buys:
        print(f'\n买入候选 (评分≥{ENTRY_THRESHOLD}, 最多{MAX_POSITIONS}只):')
        for i, b in enumerate(buys[:MAX_POSITIONS]):
            total_sc = sum(s['score'] for s in buys[:MAX_POSITIONS])
            wgt = b['score'] / total_sc * 100 if total_sc > 0 else 100/len(buys)
            rsi_str = f'RSI{b["rsi"]:.0f}' if b.get('rsi') else ''
            p52_str = f'距52高{b["pct52"]:.0f}%' if b.get('pct52') else ''
            print(f'  {i+1}. {b["code"]:6s} 评分{b["score"]:3d}  ${b["price"]:>7.2f}  {rsi_str}  {p52_str}  建议{wgt:.0f}%')
    
    # 保存
    output = {
        'date': today, 'scores': scores,
        'rules': {'entry_threshold': ENTRY_THRESHOLD, 'exit_threshold': EXIT_THRESHOLD,
                  'max_positions': MAX_POSITIONS, 'stop_loss': STOP_LOSS},
        'pool_size': len(pool)
    }
    out_path = f'{DATA_DIR}/ld3_scored_{today}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\n已保存: {out_path}')
    print(f'⏱️ 总耗时: {time.time()-t0:.1f}s')

if __name__ == '__main__':
    main()
