#!/usr/bin/env python3
"""
绿箭 S2：下载SP500成分股5年日K线
yfinance批量下载，按字母分批
"""
import json, time, warnings, os
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd

sp500 = json.load(open('/home/hermes/.hermes/openclaw-project/data/sp500_list.json', 'r'))
syms = sp500['syms']
print(f'共{len(syms)}只SP500成分股')

BATCH = 50
OUT_DIR = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'
os.makedirs(OUT_DIR, exist_ok=True)

full_data = {}
errors = []

for i in range(0, len(syms), BATCH):
    batch = syms[i:i+BATCH]
    try:
        print(f'[{i//BATCH+1}/{(len(syms)-1)//BATCH+1}] {batch[0]}...{batch[-1]}')
        df = yf.download(batch, period='5y', group_by='ticker', threads=True, progress=False, auto_adjust=True)
        
        for sym in batch:
            try:
                sym_k = sym
                if isinstance(df.columns, pd.MultiIndex):
                    if sym not in df.columns.get_level_values(0):
                        continue
                    sdf = df[sym].dropna().reset_index()
                else:
                    continue  # single ticker case won't happen with batch>1
                
                if len(sdf) < 100:
                    errors.append(sym)
                    continue
                
                # Convert
                records = []
                for _, r in sdf.iterrows():
                    dt = r.iloc[0]
                    if isinstance(dt, pd.Timestamp):
                        dt = dt.strftime('%Y-%m-%d')
                    records.append({
                        'Date': dt,
                        'O': float(r['Open']),
                        'H': float(r['High']),
                        'L': float(r['Low']),
                        'C': float(r['Close']),
                        'V': int(r['Volume'])
                    })
                full_data[sym_k] = records
                print(f'  OK {sym_k}: {len(records)}d', end='  ')
            except Exception as e:
                errors.append(sym)
                print(f'  ERR {sym}: {str(e)[:60]}', end='  ')
        print()
    except Exception as e:
        print(f'Batch失败 {batch[0]}: {str(e)[:80]}')
        errors.extend(batch)
    
    time.sleep(1.5)

print(f'\n下载完成: {len(full_data)}只成功, {len(errors)}只失败')
if errors:
    print(f'失败列表: {errors[:30]}...')

# 分批保存
chunk = 100
all_syms = sorted(full_data.keys())
for ci in range(0, len(all_syms), chunk):
    chunk_syms = all_syms[ci:ci+chunk]
    chunk_data = {s: full_data[s] for s in chunk_syms}
    path = f'{OUT_DIR}/sp500_chunk_{ci//chunk}.json'
    json.dump(chunk_data, open(path, 'w'))
    print(f'保存: {path} ({len(chunk_syms)}只)')

# 元数据
meta = {
    'syms': all_syms, 'count': len(all_syms),
    'errors': errors[:50],
    'total_rows': sum(len(full_data[s]) for s in all_syms),
    'date': time.strftime('%Y-%m-%d')
}
json.dump(meta, open(f'{OUT_DIR}/sp500_hist_meta.json', 'w'), indent=2)
print(f'\n完成: {meta["count"]}只, {meta["total_rows"]}行, {time.strftime("%Y-%m-%d %H:%M")}')
