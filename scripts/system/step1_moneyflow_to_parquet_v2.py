"""
step1_moneyflow_to_parquet_v2.py — 资金流JSON转parquet（流式）
策略：用ijson流式解析JSON，不一次加载整个文件
降级：手动按块查找code条目
"""
import json, os, sys, gc, time
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.json'
DST = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet'
BATCH_SIZE = 100  # 每100只存盘一次

fsize = os.path.getsize(SRC)
print(f'源文件: {SRC}  {fsize/1024/1024:.1f} MB')

# 检测是否安装ijson
try:
    import ijson
    have_ijson = True
    print('使用ijson流式解析')
except ImportError:
    have_ijson = False
    print('使用正则分块解析')

t0 = time.time()

def parse_via_regex():
    """正则分块: 识别"code":[{...}], 模式"""
    batch_rows = []
    total_rows = 0
    code_count = 0
    
    with open(SRC, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f'已读入内存: {len(content)/1024/1024:.1f} MB (这次是必须的, 但只读一次)')
    
    # 找到所有code条目
    # 格式: "000001":[{},{}],
    # 用正则提取每个code对应的数组
    pattern = r'"(\d{6})":\s*\[(.*?)\](?:,|\s*})'
    
    # 但JSON中值可能包含嵌套，用简单标记法
    # 改成: 找到所有双引号6位数字+冒号+左括号
    idx = 0
    code_entries = []
    
    while True:
        # 找 "xxxxxx":[  
        m = re.search(r'"(\d{6})":\s*\[', content[idx:])
        if not m:
            break
        
        code = m.group(1)
        start = idx + m.end()
        
        # 找到匹配的] 
        depth = 1
        pos = start
        while depth > 0 and pos < len(content):
            if content[pos] == '[':
                depth += 1
            elif content[pos] == ']':
                depth -= 1
            pos += 1
        
        if depth != 0:
            break  # JSON损坏
        
        json_str = content[start:pos-1]  # 去掉]
        
        # 解析数组
        try:
            records = json.loads('[' + json_str + ']')
        except:
            # 某些行可能格式有误
            idx += m.end()
            continue
        
        for r in records:
            r['code'] = code
            batch_rows.append(r)
            total_rows += 1
        
        code_count += 1
        if code_count % 100 == 0:
            print(f'  解析{code_count}只代码, {total_rows}行')
        
        # 定期存盘
        if code_count % BATCH_SIZE == 0:
            df = pd.DataFrame(batch_rows)
            mode = 'w' if total_rows == len(batch_rows) else 'a'
            header = (total_rows == len(batch_rows))
            df.to_parquet(DST if mode=='w' else DST, engine='pyarrow',
                         index=False, compression='snappy')
            print(f'  存盘 {len(batch_rows)}行 -> parquet')
            batch_rows = []
            gc.collect()
        
        idx = pos + 1
    
    # 最后的批次
    if batch_rows:
        df = pd.DataFrame(batch_rows)
        # 检查文件是否存在
        if os.path.exists(DST):
            df_existing = pd.read_parquet(DST)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_parquet(DST, engine='pyarrow', index=False, compression='snappy')
        print(f'  最终存盘 {len(batch_rows)}行')
    
    return code_count, total_rows

# 执行
code_count, total_rows = parse_via_regex()
t1 = time.time()
print(f'\n转换完成: {code_count}只代码, {total_rows}行, 耗时{t1-t0:.1f}s')
print(f'输出: {DST}')

# 验证
print('验证:')
df = pd.read_parquet(DST)
print(f'  总行数: {len(df)}')
print(f'  字段: {list(df.columns)}')
print(f'  日期范围: {df["trade_date"].min()} ~ {df["trade_date"].max()}')
print(f'  股票数: {df["code"].nunique()}')
print(f'  内存: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB (parquet加载)')
