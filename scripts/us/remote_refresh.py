#!/usr/bin/env python3
"""
本地Windows专用：并行晨扫刷新
- 5线程并发 + 请求间隔防限流
- Tushare主数据源（fallback sina）
- 无notify/audit依赖，纯文件输出
"""
import sys, json, time, os, concurrent.futures
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from data_source import AShareKline, code_to_board
from score_engine import v1_score_from_data

kl = AShareKline()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_FILE = os.path.join(ROOT, 'data', 'quality_pool.json')
OUT_FILE = os.path.join(ROOT, 'data', 'morning_top100.json')

lock = __import__('threading').Lock()
last_req = 0

def rate_limited_score(code):
    """限流评分：每只间隔0.2秒防Tushare限流"""
    global last_req
    with lock:
        now = time.time()
        if now - last_req < 0.2:
            time.sleep(0.2 - (now - last_req))
        last_req = time.time()
    
    try:
        data = kl.get_best(code)
        if not data or len(data) < 60:
            return None
        close = [d['close'] for d in data]
        high = [d['high'] for d in data]
        low = [d['low'] for d in data]
        score = v1_score_from_data(close, high, low)
        if score is None:
            return None
        return {'code': code, 'score': round(score, 1)}
    except:
        return None

def refresh():
    global last_req
    t0 = time.time()
    with open(POOL_FILE, encoding='utf-8') as f:
        pool = json.load(f)
    all_stocks = pool.get('stocks', [])
    total = len(all_stocks)
    
    print(f"加载 {total} 只股票 | 数据源: Tushare | 5线程+限流", flush=True)
    
    codes = [(i, s['code'], s.get('name','?')) for i, s in enumerate(all_stocks)]
    all_results = []
    errors = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(rate_limited_score, item[1]): item for item in codes}
        
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            idx, code, name = item
            try:
                result = future.result()
                if result:
                    result['name'] = name
                    result['board'] = code_to_board(code)
                    all_results.append(result)
                else:
                    errors += 1
            except:
                errors += 1
    
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results[:100], f, ensure_ascii=False)
    
    elapsed = time.time() - t0
    rate = len(all_results) / total * 100
    qualified = len([r for r in all_results if r['score'] >= 62])
    
    print(f"\n完成: {elapsed:.0f}s | 有效: {len(all_results)}/{total} ({rate:.0f}%) | 错误: {errors}")
    print(f"达标(>=62): {qualified}只")
    print(f"Top 5:")
    for r in all_results[:5]:
        print(f"  {r['name']} ({r['code']}) {r['score']}分")

if __name__ == '__main__':
    refresh()
