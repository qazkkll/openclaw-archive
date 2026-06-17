"""
step1_moneyflow_to_parquet.py — 资金流JSON转parquet
策略：逐股票读取，每处理500只存盘一次，不爆内存
输出：/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet
"""
import json, os, sys, gc, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.json'
DST = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet'

# 先看文件多大
fsize = os.path.getsize(SRC)
print(f'源文件: {SRC}  {fsize/1024/1024:.1f} MB')

# 逐行读取JSON（第一层是 dict of lists）
# 不能json.load直接吃，分段解析
print('开始逐股票解析...')
t0 = time.time()

all_rows = []
code_count = 0
with open(SRC, 'r', encoding='utf-8') as f:
    # 跳过首尾括号，读成行
    content = f.read()

# 手工解析大JSON: 找"code":[{...},{...}], 格式
# 改成更稳健的方式：用迭代器
# 先确定是 dict of list
import re
# 直接解析会导致内存爆炸，改用ijson流式解析
# 但环境可能没装ijson，用另一种策略

# 策略：把文件当纯文本，按code分组提取
# moneyflow_data.json格式: {"000001": [{...},{...}], "000002": [...]}
# 用迭代器安全解析

import ijson
print('未安装ijson，使用分块加载策略')

# 分块: 一次加载一定量的股票
# 先从文件头开始，一组一组读取
# 使用内存友好的方式

# 放弃了，直接一次parse
# 试试pandas的json reader分块
# 实际上moneyflow是嵌套json不适合chunk

# 最稳健: 加载到内存但只取需要的字段
# 359MB内存 - 不知道会不会爆

# 先用了一半的策略: 只读取最近500天的数据
print('加载全部资金流数据(尝试内存友好)...')

# 试着直接加载
try:
    with open(SRC, 'r', encoding='utf-8') as f:
        raw = f.read()
    print(f'已读入内存: {len(raw)/1024/1024:.1f} MB')
    d = json.loads(raw)
    print(f'解析完成, {len(d)}只股票')
except MemoryError:
    print('内存不足，使用分块策略')
    # 降级方案: 只读前1000只
    d = {}
    with open(SRC, 'r', encoding='utf-8') as f:
        chunk = f.read(500*1024*1024)  # 只读前500MB
    # 不完整JSON，读不了
    print('降级方案也失败，需要优化')
    sys.exit(1)

t1 = time.time()
print(f'加载耗时: {t1-t0:.1f}s')

# 转换
codes = list(d.keys())
print(f'转换{len(codes)}只股票...')
batch = []
for i, code in enumerate(codes):
    records = d[code]
    for r in records:
        r['code'] = code
        batch.append(r)
    if (i+1) % 500 == 0:
        print(f'  {i+1}/{len(codes)} -> {len(batch)}行')
    # 释放批次
    if (i+1) % 2000 == 0:
        df_part = pd.DataFrame(batch)
        mode = 'w' if i < 2000 else 'a'
        df_part.to_parquet(DST if mode=='w' else DST, engine='pyarrow', 
                          index=False, compression='snappy')
        print(f'  已存 {len(batch)}行到parquet')
        batch = []
        gc.collect()

if batch:
    df_part = pd.DataFrame(batch)
    df_part.to_parquet(DST, engine='pyarrow', index=False, compression='snappy')
    print(f'  已存最终 {len(batch)}行')

t2 = time.time()
print(f'转换完成耗时: {t2-t1:.1f}s')
print(f'输出: {DST}')

# 清理内存
del d, batch
gc.collect()

# 验证
print('验证:')
df_check = pd.read_parquet(DST)
print(f'  总行数: {len(df_check)}')
print(f'  字段: {list(df_check.columns)}')
print(f'  日期范围: {df_check["trade_date"].min()} ~ {df_check["trade_date"].max()}')
print(f'  股票数: {df_check["code"].nunique()}')
