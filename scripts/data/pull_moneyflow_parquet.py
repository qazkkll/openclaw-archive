#!/usr/bin/env python3
"""
补充拉取moneyflow数据（Parquet格式）
从上次中断处继续，追加到现有parquet文件
"""
import tushare as ts
import json
import pandas as pd
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _paths import DATA_DIR, TUSHARE_CFG

# 配置
with open(TUSHARE_CFG) as f:
    config = json.load(f)
ts.set_token(config['token'])
pro = ts.pro_api()

PARQUET_FILE = os.path.join(DATA_DIR, 'cn', 'moneyflow_full.parquet')
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
        return None
    return df

def main():
    print("=" * 60)
    print("📥 补充拉取moneyflow（Parquet格式）")
    print("=" * 60)
    
    # 1. 获取交易日
    print("获取交易日历...")
    trade_dates = get_trade_dates()
    print(f"共 {len(trade_dates)} 个交易日")
    
    # 2. 读取现有数据，获取已完成的日期
    done_dates = set()
    if os.path.exists(PARQUET_FILE):
        print("读取现有数据...")
        existing_df = pd.read_parquet(PARQUET_FILE)
        done_dates = set(existing_df['trade_date'].unique())
        print(f"已完成: {len(done_dates)} 个交易日")
        print(f"日期范围: {min(done_dates)} ~ {max(done_dates)}")
        
        # 过滤剩余日期
        remaining_dates = [d for d in trade_dates if d not in done_dates]
        print(f"剩余: {len(remaining_dates)} 个交易日")
    else:
        remaining_dates = trade_dates
        print("全新拉取")
    
    if not remaining_dates:
        print("✅ 数据已完整，无需补充")
        return
    
    # 3. 拉取剩余数据
    total = len(remaining_dates)
    errors = 0
    batch_data = []
    batch_size = 50  # 每50天保存一次
    
    t0 = time.time()
    
    for i, td in enumerate(remaining_dates):
        try:
            df = pull_by_date(td)
            if df is not None and len(df) > 0:
                batch_data.append(df)
        except Exception as e:
            errors += 1
            print(f"  ❌ {td}: {e}")
            if errors > 10:
                print("连续错误过多，停止")
                break
        
        # 进度 + 批量保存
        if (i + 1) % batch_size == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta = (total - i - 1) / rate if rate > 0 else 0
            
            print(f"  [{i+1}/{total}] {errors}错误 | {rate:.0f}天/分 | ETA {eta:.0f}分")
            
            # 保存批次
            if batch_data:
                batch_df = pd.concat(batch_data, ignore_index=True)
                
                if os.path.exists(PARQUET_FILE):
                    # 追加模式：读取现有 + 合并 + 保存
                    existing_df = pd.read_parquet(PARQUET_FILE)
                    combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                    combined_df.to_parquet(PARQUET_FILE, index=False, engine='pyarrow')
                else:
                    # 新建
                    batch_df.to_parquet(PARQUET_FILE, index=False, engine='pyarrow')
                
                batch_data = []  # 清空批次
        
        time.sleep(0.3)  # 限速
    
    # 4. 最终统计
    elapsed = time.time() - t0
    final_df = pd.read_parquet(PARQUET_FILE)
    final_size = os.path.getsize(PARQUET_FILE) / 1024 / 1024
    
    print(f"\n{'=' * 60}")
    print(f"✅ 完成!")
    print(f"总记录: {len(final_df)}")
    print(f"股票数: {final_df['ts_code'].nunique()}")
    print(f"交易日: {final_df['trade_date'].nunique()}")
    print(f"日期范围: {final_df['trade_date'].min()} ~ {final_df['trade_date'].max()}")
    print(f"文件大小: {final_size:.1f} MB")
    print(f"耗时: {elapsed/60:.1f} 分钟")

if __name__ == '__main__':
    main()
