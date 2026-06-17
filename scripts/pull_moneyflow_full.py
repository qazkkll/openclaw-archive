#!/usr/bin/env python3
"""
全量拉取A股资金流数据 - tushare moneyflow
限速: 0.3s/请求, 每100次暂停60s
预计: 5500只 × 0.3s ≈ 30-40分钟
"""
import tushare as ts
import pandas as pd
import json
import time
import os
import sys
from datetime import datetime

TUSHARE_TOKEN = 'ca0ba9b80be379ea2f9ae466772d42bd270ac71b95ab6d116fa094db'
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

DATA_DIR = '/home/hermes/.hermes/openclaw-archive/data'
OUTPUT_FILE = os.path.join(DATA_DIR, 'moneyflow_full.json')
CHECKPOINT_DIR = os.path.join(DATA_DIR, 'moneyflow_checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

START_DATE = '20160101'
END_DATE = '20260617'

def load_checkpoint():
    """加载断点续传状态"""
    state_file = os.path.join(CHECKPOINT_DIR, 'state.json')
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {'completed': [], 'failed': [], 'last_index': 0}

def save_checkpoint(state):
    """保存断点"""
    state_file = os.path.join(CHECKPOINT_DIR, 'state.json')
    with open(state_file, 'w') as f:
        json.dump(state, f)

def save_partial(results, batch_num):
    """每500只保存一次增量"""
    partial_file = os.path.join(CHECKPOINT_DIR, f'batch_{batch_num}.json')
    with open(partial_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"  💾 保存 batch_{batch_num}.json ({len(results)}只)")

def main():
    t0 = time.time()
    
    # 1. 获取全市场股票列表
    print("="*60)
    print("📥 全量资金流数据拉取")
    print("="*60)
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    
    stock_basic = pro.query('stock_basic', exchange='', list_status='L')
    codes = stock_basic['ts_code'].tolist()
    print(f"股票总数: {len(codes)}")
    
    # 2. 加载断点
    state = load_checkpoint()
    completed = set(state['completed'])
    failed = state['failed']
    start_idx = state['last_index']
    
    if completed:
        print(f"断点续传: 已完成{len(completed)}只, 从第{start_idx}只继续")
    
    # 3. 拉取数据
    results = {}
    total = len(codes)
    errors = 0
    
    # 先加载已有的批量文件
    for f in sorted(os.listdir(CHECKPOINT_DIR)):
        if f.startswith('batch_') and f.endswith('.json'):
            with open(os.path.join(CHECKPOINT_DIR, f)) as fp:
                batch_data = json.load(fp)
                results.update(batch_data)
    if results:
        print(f"已加载 {len(results)} 只历史数据")
    
    for i in range(start_idx, total):
        code = codes[i]
        if code in completed:
            continue
        
        try:
            df = pro.query('moneyflow', ts_code=code, start_date=START_DATE, end_date=END_DATE)
            if df is not None and len(df) > 0:
                records = df.to_dict('records')
                results[code] = records
                completed.add(code)
            else:
                results[code] = []
                completed.add(code)
        except Exception as e:
            errors += 1
            failed.append({'code': code, 'error': str(e), 'index': i})
            if errors > 10:
                print(f"\n❌ 连续错误过多，暂停保存")
                break
        
        # 限速
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i - start_idx + 1) / elapsed * 60
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {len(results)}只 | {errors}错误 | {rate:.0f}只/分 | ETA {eta:.0f}分")
            save_checkpoint({'completed': list(completed), 'failed': failed, 'last_index': i + 1})
            save_partial(results, i // 500)
            time.sleep(60)  # 每100只暂停60秒
        else:
            time.sleep(0.3)
    
    # 4. 保存最终结果
    print(f"\n{'='*60}")
    print(f"📊 拉取完成")
    print(f"{'='*60}")
    print(f"股票数: {len(results)}")
    print(f"有数据: {sum(1 for v in results.values() if v)}")
    print(f"错误: {errors}")
    print(f"耗时: {(time.time()-t0)/60:.1f}分钟")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"保存: {OUTPUT_FILE}")
    print(f"大小: {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB")
    
    # 清理checkpoint
    for f in os.listdir(CHECKPOINT_DIR):
        if f.startswith('batch_') or f == 'state.json':
            os.remove(os.path.join(CHECKPOINT_DIR, f))
    print("清理checkpoint完成")

if __name__ == '__main__':
    main()
