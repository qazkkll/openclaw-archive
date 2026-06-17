"""
step2_ml_features.py — A股ML特征工程
直接从JSON资金流数据流式读取，按股票逐只处理
内存友好：一次只加载一只股票的数据

输出: /home/hermes/.hermes/openclaw-project/scripts/system/a_ml_feats_v1.parquet (分批写入)
"""
import json, os, sys, gc, time, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.json'
DST = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_v1.parquet'
BATCH = 200  # 每200只存盘一次

# 字段映射
MF_FIELDS = ['net_mf_amount','net_mf_ratio','buy_lg_amount','sell_lg_amount',
             'buy_elg_amount','sell_elg_amount','buy_md_amount','sell_md_amount']

print(f'开始A股ML特征工程...')
t0 = time.time()

def parse_moneyflow_json(src):
    """流式解析资金流JSON，yield (code, records)"""
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f'JSON已读入: {len(content)/1024/1024:.1f} MB')
    
    # 正则匹配每个code
    pattern = r'"(\d{6}\.SZ|00\d{4}|60\d{4})":\['
    idx = 0
    count = 0
    while True:
        m = re.search(pattern, content[idx:])
        if not m:
            break
        code = m.group(1)
        start = idx + m.end()
        # 找匹配的]
        depth = 1
        pos = start
        while depth > 0 and pos < len(content):
            if content[pos] == '[':
                depth += 1
            elif content[pos] == ']':
                depth -= 1
            pos += 1
        if depth != 0:
            break
        try:
            records = json.loads('[' + content[start:pos-1] + ']')
        except:
            idx += m.end()
            continue
        # 过滤：至少50条记录
        if records and len(records) >= 50:
            yield code, records
            count += 1
            if count % 500 == 0:
                print(f'  解析{count}只...', end='\r')
        idx = pos + 1

def compute_features(code, records):
    """对一只股票计算ML特征"""
    df = pd.DataFrame(records)
    if len(df) < 50:
        return []
    
    # 按日期排序
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    features_list = []
    # 对每个日期窗口（用未来5天收益做标签）
    for i in range(len(df) - 10):
        row = df.iloc[i]
        # 当日资金面特征
        feat = {
            'code': code,
            'trade_date': row['trade_date'],
            # 资金流核心
            'net_mf': row.get('net_mf_amount', 0) / 1e4 if row.get('net_mf_amount') else 0,
            'net_mf_ratio': row.get('net_mf_ratio', 0),
            'big_net': (row.get('buy_lg_amount',0) - row.get('sell_lg_amount',0)) / 1e4,
            'big_ratio': row.get('buy_lg_amount',0) / max(row.get('sell_lg_amount',0), 1),
            'xbig_net': (row.get('buy_elg_amount',0) - row.get('sell_elg_amount',0)) / 1e4,
            'mid_net': (row.get('buy_md_amount',0) - row.get('sell_md_amount',0)) / 1e4,
        }
        
        # 近3日资金流趋势
        if i >= 2:
            mf_3d = sum(df.iloc[i-j].get('net_mf_amount',0) or 0 for j in range(3))
            feat['net_mf_3d'] = mf_3d / 1e4
            feat['net_mf_3d_trend'] = 1 if mf_3d > 0 else 0
        else:
            feat['net_mf_3d'] = feat['net_mf']
            feat['net_mf_3d_trend'] = 1 if feat['net_mf'] > 0 else 0
        
        # 主力 vs 散户背离
        big_flow = row.get('buy_lg_amount',0) + row.get('buy_elg_amount',0) - row.get('sell_lg_amount',0) - row.get('sell_elg_amount',0)
        small_flow = row.get('buy_sm_amount',0) - row.get('sell_sm_amount',0)
        feat['big_small_divergence'] = big_flow - small_flow
        
        # Y标签：未来5天涨跌 (0=跌, 1=涨)
        if i + 5 < len(df):
            future_close = df.iloc[i+5].get('close', 0) or 0
            current_close = row.get('close', 0) or 0
            if current_close > 0 and future_close > 0:
                ret_5d = (future_close - current_close) / current_close
                feat['ret_5d'] = ret_5d
                feat['label'] = 1 if ret_5d > 0.02 else 0  # 2%作为显著线
            else:
                continue
        else:
            continue
        
        features_list.append(feat)
    
    return features_list

# 主循环
all_rows = []
stock_count = 0
batch_num = 0

for code, records in parse_moneyflow_json(SRC):
    features = compute_features(code, records)
    all_rows.extend(features)
    stock_count += 1
    
    if stock_count % BATCH == 0:
        df_batch = pd.DataFrame(all_rows)
        mode = 'w' if batch_num == 0 else 'a'
        df_batch.to_parquet(DST, engine='pyarrow', index=False, compression='snappy')
        print(f'\n[Batch {batch_num+1}] 存盘{len(all_rows)}行, {stock_count}只完成')
        all_rows = []
        batch_num += 1
        gc.collect()

# 最后一组
if all_rows:
    df_batch = pd.DataFrame(all_rows)
    df_batch.to_parquet(DST, engine='pyarrow', index=False, compression='snappy')
    print(f'\n[Final] 存盘{len(all_rows)}行')

t1 = time.time()
print(f'\n✅ 特征工程完成: {stock_count}只股票, 总耗时{t1-t0:.1f}s')
print(f'输出: {DST}')
