"""
step2_simple.py — 最简单方式：按股票从JSON提取，特征计算，逐只存盘
每只股票独立计算，内存影响可控
"""
import json, os, sys, gc, time, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.json'
DST = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_v1.parquet'
TMP = '/home/hermes/.hermes/openclaw-project/scripts/system/feats_tmp'
os.makedirs(TMP, exist_ok=True)

t0 = time.time()
print('按股票逐只解析资金流数据...')

# 预先读出一个索引：每只股票的起始偏移量和长度
# 让 f.read(offset, length) 直跳，不用搜索
print('建立索引...')
idx = []
with open(SRC, 'r', encoding='utf-8') as f:
    pos = 1  # 跳过开头的 {
    buf = ''
    while True:
        chunk = f.read(1024*1024)  # 1MB chunks
        if not chunk:
            break
        buf += chunk
        # 找所有 "XXXXXX.SZ"或"XXXXXX": 的起始位置
        for m in re.finditer(r'"(\d{6}\.S?Z?)":\s*\[', buf):
            code = m.group(1)
            start = pos + m.start()
            idx.append((code, start))
            if len(idx) % 1000 == 0:
                print(f'  索引{len(idx)}只...', end='\r')
        pos += len(chunk)
        # 如果buf太大，截断
        if len(buf) > 10*1024*1024:
            buf = buf[-1024*1024:]

print(f'\n索引完成: {len(idx)}只股票')

# 现在逐只读取完整数据
stock_count = 0
feat_rows = []

with open(SRC, 'r', encoding='utf-8') as f:
    for stock_code, offset in idx:
        # 跳转到该股票位置
        f.seek(offset)
        # 读一个合理大小（大多数股票<1MB）
        chunk = f.read(512*1024)  # 512KB
        
        # 找匹配的]
        depth = 1
        end = 0
        for i, c in enumerate(chunk):
            if c == '[':
                depth += 1
            elif c == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        
        if end == 0:
            continue
        
        # 解析
        try:
            records = json.loads('[' + chunk[1:end] + ']')
        except:
            continue
        
        if len(records) < 50:
            stock_count += 1
            continue
        
        # 排序（最新在前）
        records.sort(key=lambda x: x.get('trade_date', ''), reverse=True)
        
        # 特征计算
        for i in range(len(records) - 10):
            r = records[i]
            
            def sv(k, default=0.0):
                v = r.get(k, default)
                return float(v) if v is not None else default
            
            net = sv('net_mf_amount')
            feat = {
                'stock_code': stock_code,
                'trade_date': int(r.get('trade_date', 0)),
                'net_mf': net / 1e4 if abs(net) > 1 else net,
                'net_mf_ratio': sv('net_mf_ratio'),
                'big_net': (sv('buy_lg_amount') - sv('sell_lg_amount')) / 1e4,
                'xbig_net': (sv('buy_elg_amount') - sv('sell_elg_amount')) / 1e4,
                'mid_net': (sv('buy_md_amount') - sv('sell_md_amount')) / 1e4,
            }
            
            # 3日趋势
            if i <= len(records) - 3:
                mf3 = sum(sv('net_mf_amount', 0) for r2 in records[i:i+3])
                feat['net_mf_3d'] = mf3 / 1e4 if abs(mf3) > 1 else mf3
                feat['net_mf_trend'] = 1.0 if mf3 > 0 else 0.0
            else:
                feat['net_mf_3d'] = feat['net_mf']
                feat['net_mf_trend'] = 1.0 if feat['net_mf'] > 0 else 0.0
            
            # 主力vs散户
            big_in = sv('buy_lg_amount') + sv('buy_elg_amount')
            big_out = sv('sell_lg_amount') + sv('sell_elg_amount')
            small_net = sv('buy_sm_amount') - sv('sell_sm_amount')
            feat['big_small_div'] = (big_in - big_out) - small_net
            
            # Y标签
            if i + 5 < len(records):
                fut = records[i+5]
                curr_c = sv('close')
                fut_c = float(fut.get('close', 0) or 0)
                if curr_c > 0 and fut_c > 0:
                    ret = (fut_c - curr_c) / curr_c
                    feat['ret_5d'] = round(ret, 6)
                    feat['label'] = 1.0 if ret > 0.02 else 0.0
                    feat_rows.append(feat)
            
            # 数据太多时压缩
            if len(feat_rows) >= 500000:
                df = pd.DataFrame(feat_rows)
                df.to_parquet(DST, index=False, compression='snappy')
                print(f'[中间存盘] {len(df)}行', flush=True)
                feat_rows = []
                gc.collect()
        
        stock_count += 1
        if stock_count % 200 == 0:
            print(f'  {stock_count}/{len(idx)}只, {len(feat_rows)}行特征', flush=True)

# 最终存盘
if feat_rows:
    df = pd.DataFrame(feat_rows)
    if os.path.exists(DST):
        existing = pd.read_parquet(DST)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(DST, index=False, compression='snappy')
    print(f'[最终存盘] {len(df)}行', flush=True)

t1 = time.time()
print(f'\n✅ 完成: {stock_count}只股票, 耗时{(t1-t0)/60:.1f}分钟')
print(f'输出: {DST}')

# 验证
df = pd.read_parquet(DST)
print(f'  总行数: {len(df)}')
print(f'  字段: {list(df.columns)}')
print(f'  日期: {df["trade_date"].min()} ~ {df["trade_date"].max()}')
print(f'  股票数: {df["stock_code"].nunique()}')
print(f'  正负例: {df["label"].value_counts().to_dict()}')
