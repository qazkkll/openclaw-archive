#!/usr/bin/env python3
"""
统一数据读取接口 — 优先读parquet，fallback到json

用法:
    from _read_data import read_data
    df = read_data('a_hist_10y')  # 自动读 .parquet 或 .json
    df = read_data('a_hist_10y', fmt='df')   # 返回pandas DataFrame
    df = read_data('a_hist_10y', fmt='dict')  # 返回dict（原始格式）

支持的数据集：
    a_hist_10y, us_hist_clean, us_hist_10y, top_list_data,
    a1_daily, moneyflow_data, daily_basic_data

注意：us_hist 系列返回的是 {code: {c:[],h:[],l:[]}} 格式，
      moneyflow 返回 {code: [{record}]}，
      daily_basic / top_list / a1_daily 返回 {date: [record]}
"""
import os, json

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def read_data(name, fmt='df'):
    """
    读取数据集，优先parquet。

    参数:
        name: 数据集名（不含扩展名），例如 'a_hist_10y', 'us_hist_clean'
        fmt: 'df' → pandas DataFrame, 'dict' → Python dict
    
    返回:
        fmt='df': pandas DataFrame（需要 pandas 已安装）
        fmt='dict': Python dict（原始JSON结构）
    """
    # 尝试parquet
    pp = os.path.join(BASE, name + '.parquet')
    jp = os.path.join(BASE, name + '.json')
    rawjp = os.path.join(BASE, 'raw_json', name + '.json')
    
    # parquet优先
    if os.path.exists(pp):
        if fmt == 'df':
            import pandas as pd
            return pd.read_parquet(pp)
        else:
            raise ValueError("parquet 格式只能返回 DataFrame，请用 fmt='df'")
    
    # fallback到json
    fp = None
    for p in [jp, rawjp]:
        if os.path.exists(p):
            fp = p
            break
    
    if fp is None:
        raise FileNotFoundError(f"数据集 '{name}' 未找到 (checked: parquet, json, raw_json)")
    
    print(f"[read_data] loading {name} from JSON (slow)...", flush=True)
    with open(fp, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if fmt == 'dict':
        return data
    
    # fmt='df' 但只有json
    import pandas as pd
    # 尝试智能推断结构
    if isinstance(data, dict):
        first_val = next(iter(data.values()), None)
        if isinstance(first_val, list):
            # {key: [records]} → 堆叠
            rows = []
            for k, records in data.items():
                for r in records:
                    if isinstance(r, dict):
                        r['_key'] = k
                        rows.append(r)
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        elif isinstance(first_val, dict):
            # {code: {c:[],h:[],l:[]}} — 展开为每代码一行
            rows = []
            for code, v in data.items():
                row = {'code': code}
                for k2, v2 in v.items():
                    row[k2] = v2
                rows.append(row)
            return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    return pd.DataFrame(data)

def list_datasets():
    """列出所有可用数据集"""
    import glob
    parquets = glob.glob(os.path.join(BASE, '*.parquet'))
    jsons = glob.glob(os.path.join(BASE, '*.json'))
    rawjsons = glob.glob(os.path.join(BASE, 'raw_json', '*.json'))
    
    all_files = set(parquets + jsons + rawjsons)
    datasets = {}
    for fp in sorted(all_files):
        base = os.path.basename(fp)
        name = os.path.splitext(base)[0]
        ext = os.path.splitext(base)[1]
        if name not in datasets:
            datasets[name] = []
        datasets[name].append(ext)
    
    print("可用数据集:")
    for name, exts in sorted(datasets.items()):
        fmt = "parquet" if '.parquet' in exts else "json"
        print(f"  {name} ({fmt})")

if __name__ == '__main__':
    list_datasets()
