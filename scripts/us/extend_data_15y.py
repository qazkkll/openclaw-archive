#!/usr/bin/env python3
"""
用AKShare将回测数据扩展到15年（2011-2026）
在后台运行，不影响学习
"""
import json, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YAHOO_FILE = os.path.join(ROOT, 'data', 'backtest_hist_yahoo.json')
OUTPUT_FILE = os.path.join(ROOT, 'data', 'backtest_hist_15y.json')
PROGRESS_FILE = os.path.join(ROOT, 'data', 'extend_progress.json')

import akshare as ak

def extend_stock(code, name, existing_dates, existing_data):
    """用AKShare补全缺失的历史数据"""
    try:
        # 从AKShare获取全量数据
        df = ak.stock_zh_a_hist(
            symbol=code, 
            period='daily', 
            start_date='20050101',  # 拉取全部可用数据
            end_date='20260528',
            adjust='qfq'  # 前复权，保持一致性
        )
        if df.empty:
            return None
        
        # 转换为标准格式
        result = {}
        for _, row in df.iterrows():
            d = str(row['日期']).replace('-', '')  # Format: YYYYMMDD
            result[d] = {
                'open': float(row['开盘']),
                'close': float(row['收盘']),
                'high': float(row['最高']),
                'low': float(row['最低']),
                'volume': int(row['成交量']),
            }
        
        # 补充到已有数据
        if existing_data:
            # 用Yahoo数据覆盖相同日期的值（保持一致性）
            for ed, ev in existing_data.items():
                if ed in result:
                    result[ed] = ev
        
        return result
    except Exception as e:
        return None

def main():
    t0 = time.time()
    print(f'[延展数据] 开始加载AKShare数据...', flush=True)
    
    # 加载现有Yahoo数据
    with open(YAHOO_FILE) as f:
        yahoo = json.load(f)
    
    all_codes = list(yahoo.keys())
    print(f'[延展数据] 共{len(all_codes)}只股票', flush=True)
    
    total = len(all_codes)
    results = {}
    errors = 0
    
    for i, code in enumerate(all_codes):
        # 进度报告
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'[延展数据] {i+1}/{total} | 完成: {len(results)} | 错误: {errors} | {elapsed:.0f}s', flush=True)
        
        # 转换现有Yahoo数据格式
        existing = yahoo[code]
        if isinstance(existing, dict):
            existing_dates = existing.get('dates', [])
            existing_close = existing.get('close', [])
            existing_high = existing.get('high', [])
            existing_low = existing.get('low', [])
            existing_open = existing.get('open', [])
            existing_vol = existing.get('volume', [])
            
            existing_data = {}
            for i_date, d in enumerate(existing_dates):
                if i_date < len(existing_close):
                    existing_data[d] = {
                        'close': existing_close[i_date],
                        'high': existing_high[i_date] if i_date < len(existing_high) else existing_close[i_date],
                        'low': existing_low[i_date] if i_date < len(existing_low) else existing_close[i_date],
                    }
        else:
            existing_data = None
        
        # 用AKShare延展
        extended = extend_stock(code, '', existing_dates if isinstance(existing, dict) else [], existing_data)
        if extended:
            results[code] = extended
        else:
            # 保留原数据
            if isinstance(existing, dict) and existing_data:
                results[code] = existing_data
            else:
                errors += 1
    
    # 保存
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False)
    
    elapsed = time.time() - t0
    print(f'[延展数据] ✅ 完成! {elapsed:.0f}s | {len(results)}只 | {errors}错误', flush=True)

if __name__ == '__main__':
    main()
