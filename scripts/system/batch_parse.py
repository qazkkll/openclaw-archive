"""
batch_parse.py — 将4.7GB资金流JSON按股票分批解析为小parquet
不一次性加载：只开文件流，按需seek+read

分片输出: /home/hermes/.hermes/openclaw-project/data/moneyflow_slices/slice_*.parquet
每个分片1000只股票，互不依赖
"""
import json, os, sys, gc, time, re
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.json'
SLICE_DIR = '/home/hermes/.hermes/openclaw-project/data/moneyflow_slices'
os.makedirs(SLICE_DIR, exist_ok=True)

t0 = time.time()

# 第一步：建立股票偏移索引（只扫描一次）
# JSON结构: {"000001.SZ": [{...},{...}], "000002.SZ": [{...}], ...}
# 我们需要找到每个 "XXXX":[{...}] 的位置

print('建立股票索引...')
offsets = []
with open(SRC, 'r', encoding='utf-8') as f:
    # 定位到第一个 [
    pos = 0
    stock_buf = b''
    reading_stock = False
    first_char = f.read(1)
    
    # 读取大块来加速查找
    f.seek(0)
    chunk_size = 4*1024*1024  # 4MB chunks
    remainder = ''
    
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        # 用文本模式搜索 "XXXXXX":[
        text = remainder + chunk
        # 找所有股票代码位置
        for m in re.finditer(r'"(\d{6}\.S?Z?)":\s*\[', text):
            code = m.group(1)
            start_offset = pos + m.start()
            offsets.append((code, start_offset))
        
        pos += len(chunk)
        # 保留末尾1KB以保证跨块的模式匹配
        remainder = text[-1024:] if len(text) > 1024 else ''
        
        if len(offsets) % 500 == 0:
            print(f'  索引{len(offsets)}只...', flush=True)

print(f'索引完成: {len(offsets)}只股票')
print(f'索引耗时: {time.time()-t0:.1f}s')

# 第二步：排序（代码不一定是排序的）
offsets.sort(key=lambda x: x[1])  # 按文件位置排序

# 第三步：分批读取 + 计算特征 + 存dask兼容的parquet
BATCH = 1000  # 每批1000只
feat_cols = ['stock_code', 'trade_date', 'ts_code', 'net_mf', 'net_mf_ratio', 
             'big_net', 'xbig_net', 'mid_net', 'net_mf_3d', 
             'net_mf_trend', 'big_small_div', 'ret_5d', 'label',
             'close', 'buy_lg_amount', 'sell_lg_amount',
             'buy_elg_amount', 'sell_elg_amount']

# 预置字段，方便从records里取值
def safe(v, default=0.0):
    if v is None: return default
    try: return float(v)
    except: return default

batch = []
slice_num = 0

with open(SRC, 'r', encoding='utf-8') as f:
    for idx, (stock_code, offset) in enumerate(offsets):
        # 跳到股票数据起始位置
        f.seek(offset)
        
        # 找到匹配的]
        # 先跳过一个字符（跳过多读的"）
        data_start = f.tell()
        
        # 读取足够大的块来包含这个股票全部数据
        # 大多数股票<200KB，少数大票<2MB
        read_size = 1024*1024  # 1MB应该够
        chunk = f.read(read_size)
        
        # 找到完整数组的结束位置
        depth = 0
        end_pos = 0
        started = False
        for i, c in enumerate(chunk):
            if c == '[':
                depth += 1
                started = True
            elif c == ']':
                depth -= 1
                if started and depth == 0:
                    end_pos = i
                    break
        
        if end_pos == 0:
            print(f'  ⚠️ {stock_code}: 找不到数组结束', flush=True)
            continue
        
        # 提取完整JSON数组
        json_text = chunk[1:end_pos]  # 去掉[
        try:
            records = json.loads('[' + json_text + ']')
        except:
            print(f'  ⚠️ {stock_code}: JSON解析失败', flush=True)
            continue
        
        if len(records) < 50:
            continue
        
        # 排序（最新在前，tushare默认按日期降序）
        
        # 计算特征（每只股票的每个交易日一条）
        for i in range(len(records) - 10):
            r = records[i]
            net_mf = safe(r.get('net_mf_amount'))
            
            feat = {
                'stock_code': stock_code,
                'trade_date': int(r.get('trade_date', 0)),
                'ts_code': r.get('ts_code', stock_code),
                
                # 资金面特征
                'close': safe(r.get('close')),
                'net_mf': net_mf / 1e4 if abs(net_mf) > 1 else net_mf,
                'net_mf_ratio': safe(r.get('net_mf_ratio')),
                'big_net': (safe(r.get('buy_lg_amount')) - safe(r.get('sell_lg_amount'))) / 1e4,
                'xbig_net': (safe(r.get('buy_elg_amount')) - safe(r.get('sell_elg_amount'))) / 1e4,
                'mid_net': (safe(r.get('buy_md_amount')) - safe(r.get('sell_md_amount'))) / 1e4,
                
                # 保留原始值（可能分批后用于后期加特征）
                'buy_lg_amount': safe(r.get('buy_lg_amount')),
                'sell_lg_amount': safe(r.get('sell_lg_amount')),
                'buy_elg_amount': safe(r.get('buy_elg_amount')),
                'sell_elg_amount': safe(r.get('sell_elg_amount')),
            }
            
            # 3日资金流趋势
            if i <= len(records) - 3:
                mf3 = sum(safe(records[i+j].get('net_mf_amount')) for j in range(3))
                feat['net_mf_3d'] = mf3 / 1e4 if abs(mf3) > 1 else mf3
                feat['net_mf_trend'] = 1.0 if mf3 > 0 else 0.0
            else:
                feat['net_mf_3d'] = feat['net_mf']
                feat['net_mf_trend'] = 1.0 if feat['net_mf'] > 0 else 0.0
            
            # 主力散户背离
            big_in = safe(r.get('buy_lg_amount')) + safe(r.get('buy_elg_amount'))
            big_out = safe(r.get('sell_lg_amount')) + safe(r.get('sell_elg_amount'))
            small_net = safe(r.get('buy_sm_amount')) - safe(r.get('sell_sm_amount'))
            feat['big_small_div'] = (big_in - big_out) - small_net
            
            # Y标签：未来5天涨跌
            if i + 5 < len(records):
                fut = records[i+5]
                curr_c = safe(r.get('close'))
                fut_c = safe(fut.get('close'))
                if curr_c > 0 and fut_c > 0:
                    ret = (fut_c - curr_c) / curr_c
                    feat['ret_5d'] = round(ret, 6)
                    feat['label'] = 1.0 if ret > 0.02 else 0.0
                    batch.append(feat)
        
        # 每1000只存一个分片
        if (idx + 1) % BATCH == 0:
            df = pd.DataFrame(batch)
            fname = os.path.join(SLICE_DIR, f'slice_{slice_num:04d}.parquet')
            df.to_parquet(fname, index=False, compression='snappy')
            print(f'  💾 [{slice_num+1}] {stock_code}: {len(df)}行 -> {fname}', flush=True)
            batch = []
            slice_num += 1
            gc.collect()

# 最后一组
if batch:
    df = pd.DataFrame(batch)
    fname = os.path.join(SLICE_DIR, f'slice_{slice_num:04d}.parquet')
    df.to_parquet(fname, index=False, compression='snappy')
    print(f'  💾 [Final] {len(df)}行 -> {fname}', flush=True)

t1 = time.time()
print(f'\n✅ 全部完成: {len(offsets)}只, {(t1-t0)/60:.1f}分钟')
print(f'分片: {slice_num+1}个')

# 打印摘要
total_rows = 0
for i in range(slice_num+1):
    fp = os.path.join(SLICE_DIR, f'slice_{i:04d}.parquet')
    if os.path.exists(fp):
        sz = os.path.getsize(fp)
        # 用pyarrow读行数
        from pyarrow.parquet import read_metadata
        meta = read_metadata(fp)
        total_rows += meta.num_rows
        print(f'  slice_{i:04d}: {meta.num_rows}行, {sz/1024/1024:.1f}MB')

print(f'\n总特征行数: {total_rows}')
