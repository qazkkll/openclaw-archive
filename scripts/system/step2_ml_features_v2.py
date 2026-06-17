"""
step2_ml_features_v2.py — A股ML特征工程 (流式版)
不一次性加载整个JSON，用字符查找流式解析
内存友好，逐股票处理

输出: /home/hermes/.hermes/openclaw-project/scripts/system/a_ml_feats_v1.parquet
"""
import json, os, sys, gc, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.json'
DST = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_v1.parquet'
BATCH = 200

# 字段存在性检查：某些股票可能某些字段缺失
MF_NUM_FIELDS = ['net_mf_amount','net_mf_ratio','buy_lg_amount','sell_lg_amount',
                 'buy_elg_amount','sell_elg_amount','buy_md_amount','sell_md_amount',
                 'buy_sm_amount','sell_sm_amount']

t0 = time.time()
print('流式解析资金流JSON...')

def safe_float(v):
    if v is None:
        return 0.0
    try:
        return float(v)
    except:
        return 0.0

def parse_stock_block(text, start_pos):
    """从start_pos开始解析一个股票块: "000001.SZ":[{...},{...}]"""
    # 先找code
    code_start = start_pos
    if text[code_start] != '"':
        return None, start_pos + 1
    
    # 找code结束
    q2 = text.find('"', code_start + 1)
    if q2 == -1:
        return None, code_start + 1
    code = text[code_start+1:q2]
    
    # 找 ':[' 
    arr_start = text.find(':[', q2)
    if arr_start == -1:
        return None, q2 + 1
    arr_start += 2  # 跳过 ':['
    
    # 找匹配的 ]
    depth = 1
    pos = arr_start
    while depth > 0 and pos < len(text):
        if text[pos] == '[':
            depth += 1
        elif text[pos] == ']':
            depth -= 1
        pos += 1
    
    if depth != 0:
        return None, arr_start
    
    json_str = text[arr_start:pos-1]
    try:
        records = json.loads('[' + json_str + ']')
    except:
        return None, pos
    
    return {'code': code, 'records': records, 'end': pos}

# 流式读取：每次读一块，处理完再读下一块
BUFFER_SIZE = 50 * 1024 * 1024  # 50MB chunks

all_rows = []
stock_count = 0
batch_num = 0
buffer = ''
eof = False

with open(SRC, 'r', encoding='utf-8') as f:
    # 跳过第一个 {
    f.read(1)
    
    while not eof:
        chunk = f.read(BUFFER_SIZE)
        if not chunk:
            eof = True
        buffer += chunk
        
        # 处理buffer中完整的股票块
        while True:
            # 找 "xxxx...
            q1 = buffer.find('"')
            if q1 == -1:
                break
            
            # 试试解析
            result, next_pos = parse_stock_block(buffer, q1)
            if result is None:
                # 可能要更多数据
                if not eof and len(buffer) - q1 < 50000000:
                    break  # 等更多数据
                # 跳过这个坏块
                buffer = buffer[q1+1:]
                continue
            
            code = result['code']
            records = result['records']
            
            # 过滤: 主板 + 至少50条
            if records and len(records) >= 50:
                # 生成特征
                df = pd.DataFrame(records)
                df = df.sort_values('trade_date').reset_index(drop=True)
                
                for i in range(len(df) - 10):
                    row = df.iloc[i]
                    
                    net_mf = safe_float(row.get('net_mf_amount'))
                    
                    feat = {
                        'code': code,
                        'trade_date': str(row.get('trade_date', '')),
                        'net_mf': net_mf / 1e4 if abs(net_mf) > 1 else net_mf,
                        'net_mf_ratio': safe_float(row.get('net_mf_ratio')),
                        'big_net': (safe_float(row.get('buy_lg_amount')) - safe_float(row.get('sell_lg_amount'))) / 1e4,
                        'xbig_net': (safe_float(row.get('buy_elg_amount')) - safe_float(row.get('sell_elg_amount'))) / 1e4,
                        'mid_net': (safe_float(row.get('buy_md_amount')) - safe_float(row.get('sell_md_amount'))) / 1e4,
                    }
                    
                    # 3日趋势
                    if i >= 2:
                        mf3 = sum(safe_float(df.iloc[i-j].get('net_mf_amount')) for j in range(3))
                        feat['net_mf_3d'] = mf3 / 1e4 if abs(mf3) > 1 else mf3
                        feat['net_mf_trend'] = 1 if mf3 > 0 else 0
                    else:
                        feat['net_mf_3d'] = feat['net_mf']
                        feat['net_mf_trend'] = 1 if feat['net_mf'] > 0 else 0
                    
                    # 主力vs散户背离
                    big_in = safe_float(row.get('buy_lg_amount')) + safe_float(row.get('buy_elg_amount'))
                    big_out = safe_float(row.get('sell_lg_amount')) + safe_float(row.get('sell_elg_amount'))
                    small_net = safe_float(row.get('buy_sm_amount')) - safe_float(row.get('sell_sm_amount'))
                    feat['big_small_div'] = (big_in - big_out) - small_net
                    
                    # Y标签: 未来5天>2%
                    if i + 5 < len(df):
                        curr_c = safe_float(row.get('close'))
                        fut_c = safe_float(df.iloc[i+5].get('close'))
                        if curr_c > 0 and fut_c > 0:
                            ret = (fut_c - curr_c) / curr_c
                            feat['ret_5d'] = round(ret, 6)
                            feat['label'] = 1 if ret > 0.02 else 0
                            all_rows.append(feat)
                
                stock_count += 1
                if stock_count % 200 == 0:
                    print(f'  {stock_count}只, {len(all_rows)}行', flush=True)
            
            # 移走已处理部分
            buffer = buffer[result['end']:]
            
            # 定期存盘
            if len(all_rows) >= BATCH * 500:  # 约10万行
                df_batch = pd.DataFrame(all_rows)
                mode = 'w' if batch_num == 0 else 'a'
                df_batch.to_parquet(DST, engine='pyarrow', index=False, compression='snappy')
                print(f'  [Batch {batch_num+1}] 存盘{len(df_batch)}行', flush=True)
                all_rows = []
                batch_num += 1
                gc.collect()
        
        # 内存管理: 如果buffer太大，截断
        if len(buffer) > 100 * 1024 * 1024:
            last_quote = buffer.rfind('"')
            if last_quote > 0:
                buffer = buffer[last_quote:]

# 最后一批
if all_rows:
    df_batch = pd.DataFrame(all_rows)
    df_batch.to_parquet(DST, engine='pyarrow', index=False, compression='snappy')
    print(f'  [Final] 存盘{len(df_batch)}行', flush=True)

t1 = time.time()
print(f'\n✅ 完成: {stock_count}只, 耗时{(t1-t0)/60:.1f}分钟')
print(f'输出: {DST}')
