#!/usr/bin/env python3
"""
基金经理自动筛选 — 三层过滤
Layer 1: V4.2动量排名 (从us_scored.json读)
Layer 2: 基本面过滤 (估值/成交量/盈利)
Layer 3: 输出带标记的推荐

用法: python3 fund_manager_screen.py [--top 50]
"""
import json, os, sys, yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def analyze_stock(ticker):
    """Layer 2: 基本面+技术面检查"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='3mo')
        if hist.empty:
            return None
        
        close = hist['Close'].iloc[-1]
        ma20 = hist['Close'].tail(20).mean()
        
        # RSI
        delta = hist['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1]
        
        # Volume
        avg_vol = hist['Volume'].tail(20).mean()
        last_vol = hist['Volume'].iloc[-1]
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0
        
        # Returns
        ret_3m = (close/hist['Close'].iloc[0]-1)*100
        
        # Fundamentals
        pe = info.get('trailingPE')
        target = info.get('targetMeanPrice')
        profit_margin = info.get('profitMargins', 0)
        rec = info.get('recommendationKey', 'N/A')
        
        # Layer 2: Red flags
        red_flags = []
        if target and target > 0 and close > target * 1.2:
            red_flags.append(f'高估{(close/target-1)*100:.0f}%')
        if vol_ratio < 0.5:
            red_flags.append('缩量')
        if rsi > 80:
            red_flags.append('超买')
        if (not pe or pe <= 0) and profit_margin < 0:
            red_flags.append('亏损')
        if ret_3m > 100:
            red_flags.append(f'过热({ret_3m:.0f}%)')
        
        green_flags = []
        if target and target > 0 and close < target * 0.95:
            green_flags.append('低估')
        if vol_ratio > 1.2:
            green_flags.append('放量')
        if rsi < 65 and rsi > 40:
            green_flags.append('RSI健康')
        if pe and 0 < pe < 25:
            green_flags.append('PE低')
        
        return {
            'ticker': ticker,
            'price': round(close, 2),
            'rsi': round(rsi, 0),
            'vol_ratio': round(vol_ratio, 2),
            'ret_3m': round(ret_3m, 0),
            'pe': pe,
            'target': target,
            'profit_margin': profit_margin,
            'red_flags': red_flags,
            'green_flags': green_flags,
            'score': 0  # will be filled from Layer 1
        }
    except:
        return None

def main():
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # Layer 1: Load V4 rankings
    with open(os.path.join(ROOT, 'data', 'us_scored.json')) as f:
        us = json.load(f)
    
    print(f'🏁 基金经理自动筛选 · Top {n}')
    print(f'{"="*60}')
    print()
    
    for i, s in enumerate(us[:n]):
        ticker = s.get('ticker', '')
        score = s.get('score', 0)
        
        # Layer 2
        result = analyze_stock(ticker)
        if not result:
            continue
        
        result['score'] = score
        
        # Layer 3: Final judgment
        n_red = len(result['red_flags'])
        n_green = len(result['green_flags'])
        
        if n_red >= 3:
            verdict = '❌ 不推荐'
        elif n_red >= 2:
            verdict = '⚠️ 谨慎'
        elif n_red == 1 and n_green <= 1:
            verdict = '➡️ 观望'
        elif n_green >= 2 and n_red == 0:
            verdict = '✅ 可关注'
        else:
            verdict = '➡️ 中性'
        
        flags = ' | '.join(['🔴'+f for f in result['red_flags']] + ['🟢'+f for f in result['green_flags']]) if (result['red_flags'] or result['green_flags']) else '无显著信号'
        
        print(f'#{i+1} {ticker} | {score:.0f}分 | ${result["price"]} | RSI {result["rsi"]:.0f}')
        print(f'  判断: {verdict} | {flags}')
        print()

if __name__ == '__main__':
    main()
