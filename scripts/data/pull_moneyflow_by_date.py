#!/usr/bin/env python3
"""
按天拉取全市场资金流数据（tushare moneyflow）
速度：0.5秒/天 × 2500交易日 ≈ 20分钟全量
"""
import tushare as ts
import json
import time
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import DATA_DIR, CN_DATA, TUSHARE_CFG

# 配置
with open(TUSHARE_CFG) as f:
    config = json.load(f)
ts.set_token(config['token'])
pro = ts.pro_api()

OUTPUT_FILE = os.path.join(DATA_DIR, 'moneyflow_full.json')
START_DATE = '20160101'
END_DATE = datetime.now().strftime('%Y%m%d')

def get_trade_dates():
    """获取所有交易日"""
    cal = pro.query('trade_cal', exchange='SSE', start_date=START_DATE, end_date=END_DATE)
    dates = sorted(cal[cal['is_open'] == 1]['cal_date'].tolist())
    return dates

def pull_by_date(trade_date):
    """按天拉取资金流"""
    df = pro.query('moneyflow', trade_date=trade_date)
    if df is None or len(df) == 0:
        return {}
    result = {}
    for _, row in df.iterrows():
        code = row['ts_code']
        result[code] = {
            'ts_code': code,
            'trade_date': trade_date,
            'buy_sm_vol': float(row.get('buy_sm_vol', 0) or 0),
            'buy_sm_amount': float(row.get('buy_sm_amount', 0) or 0),
            'sell_sm_vol': float(row.get('sell_sm_vol', 0) or 0),
            'sell_sm_amount': float(row.get('sell_sm_amount', 0) or 0),
            'buy_md_vol': float(row.get('buy_md_vol', 0) or 0),
            'buy_md_amount': float(row.get('buy_md_amount', 0) or 0),
            'sell_md_vol': float(row.get('sell_md_vol', 0) or 0),
            'sell_md_amount': float(row.get('sell_md_amount', 0) or 0),
            'buy_lg_vol': float(row.get('buy_lg_vol', 0) or 0),
            'buy_lg_amount': float(row.get('buy_lg_amount', 0) or 0),
            'sell_lg_vol': float(row.get('sell_lg_vol', 0) or 0),
            'sell_lg_amount': float(row.get('sell_lg_amount', 0) or 0),
            'buy_elg_vol': float(row.get('buy_elg_vol', 0) or 0),
            'buy_elg_amount': float(row.get('buy_elg_amount', 0) or 0),
            'sell_elg_vol': float(row.get('sell_elg_vol', 0) or 0),
            'sell_elg_amount': float(row.get('sell_elg_amount', 0) or 0),
            'net_mf_vol': float(row.get('net_mf_vol', 0) or 0),
            'net_mf_amount': float(row.get('net_mf_amount', 0) or 0),
        }
    return result

def main():
    print("="*60)
    print("📥 按天拉取全市场资金流")
    print(f"时间范围: {START_DATE} ~ {END_DATE}")
    print("="*60)
    
    t0 = time.time()
    
    # 1. 获取交易日
    print("获取交易日历...")
    trade_dates = get_trade_dates()
    print(f"共 {len(trade_dates)} 个交易日")
    
    # 2. 加载已有数据（断点续传）
    all_data = {}  # {ts_code: [records]}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            existing = json.load(f)
        # 已有数据格式: {ts_code: [records]}
        all_data = existing
        done_dates = set()
        for code, records in all_data.items():
            for r in records:
                done_dates.add(r['trade_date'])
        print(f"已有数据: {len(all_data)} 只股票, {len(done_dates)} 个交易日")
        trade_dates = [d for d in trade_dates if d not in done_dates]
        print(f"剩余: {len(trade_dates)} 个交易日")
    else:
        print("全新拉取")
    
    # 3. 按天拉取
    total = len(trade_dates)
    errors = 0
    
    for i, td in enumerate(trade_dates):
        try:
            day_data = pull_by_date(td)
            for code, record in day_data.items():
                if code not in all_data:
                    all_data[code] = []
                all_data[code].append(record)
        except Exception as e:
            errors += 1
            if errors > 5:
                print(f"  ❌ 连续错误: {e}")
                break
        
        # 限速 + 进度
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {len(all_data)}只 | {errors}错误 | {rate:.0f}天/分 | ETA {eta:.0f}分")
            
            # 每50天保存一次
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_data, f, ensure_ascii=False)
        
        time.sleep(0.3)  # 限速
    
    # 4. 最终保存
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_data, f, ensure_ascii=False)
    
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ 完成!")
    print(f"股票数: {len(all_data)}")
    print(f"总记录: {sum(len(v) for v in all_data.values())}")
    print(f"耗时: {elapsed/60:.1f}分钟")
    print(f"大小: {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f}MB")

if __name__ == '__main__':
    main()
