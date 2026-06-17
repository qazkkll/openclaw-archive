#!/usr/bin/env python3
"""
美股盘中评分刷新 — 每30分钟 21:30~03:30
重新对Top 100做V4.2评分，看排名变化
"""
import yfinance as yf, json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from notify import send

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
US_FILE = os.path.join(ROOT, 'data', 'us_scored.json')
PREV_FILE = os.path.join(ROOT, 'data', 'us_intraday_prev.json')

def refresh():
    # Load current scored data
    with open(US_FILE) as f:
        current = json.load(f)
    
    # Get top 50 tickers
    top100 = [s['ticker'] for s in current[:100] if s.get('ticker')]
    
    # Save previous state for comparison
    prev_scores = {s['ticker']: s['score'] for s in current[:100]}
    
    # Re-score with live data
    new_results = []
    for t in top100:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period='3mo')
            if hist.empty:
                continue
            close = hist['Close'].iloc[-1]
            ret_30d = (close / hist['Close'].iloc[-22] - 1) * 100 if len(hist) >= 22 else 0
            new_results.append({'ticker': t, 'price': round(close, 2), 'mom30': round(ret_30d, 1)})
        except:
            continue
    
    # Compare rankings
    changes = []
    for nr in new_results:
        t = nr['ticker']
        old_rank = next((i for i, s in enumerate(current) if s.get('ticker') == t), 999)
        new_rank = next((i for i, nr2 in enumerate(new_results) if nr2['ticker'] == t), 999)
        if abs(old_rank - new_rank) > 3:
            changes.append(f'{t} #{old_rank+1}→#{new_rank+1}')
    
    # 🆕🔥 标记
    fire = []
    for nr in new_results[:5]:
        if nr['mom30'] and nr['mom30'] > 50:
            # Find V4.2 score
            score = next((s.get('score', 0) for s in current if s.get('ticker') == nr['ticker']), 0)
            fire.append(f'🔥 {nr["ticker"]} {score:.0f}分 (30日+{nr["mom30"]:.0f}%)')
        old_rank_for_new = next((i for i, s in enumerate(current) if s.get('ticker') == nr['ticker']), 999)
        if old_rank_for_new >= 20:  # 新进入Top20
            fire.append(f'🆕 {nr["ticker"]} (新进, ${nr["price"]})')
    
    if changes or fire:
        output = f'🔄 美股排名更新 ({len(new_results)}只)\n'
        if fire:
            output += '\n'.join(fire[:5]) + '\n'
        if changes:
            output += '排名变化: ' + ' | '.join(changes[:5]) + '\n'
        # 基金经理简要分析
        for nr in new_results[:3]:
            score = next((s.get('score', 0) for s in current if s.get('ticker') == nr['ticker']), 0)
            output += f'🧠 {nr["ticker"]} {score:.0f}分: ${nr["price"]}, 30日+{nr["mom30"]:.0f}%\n'
        send(output)
    # 即使无明显变化也静默，不刷屏
    
    # Save for next comparison
    with open(PREV_FILE, 'w') as f:
        json.dump(new_results, f)

if __name__ == '__main__':
    refresh()
