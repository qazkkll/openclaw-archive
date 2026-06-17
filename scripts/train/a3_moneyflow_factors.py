"""
A3 资金流因子提取 (optimized - minimal pandas import)
从moneyflow_data.parquet提取15个资金流因子 + 5个北向因子 + 5个龙虎榜因子
"""
import sys
import os
import json
import time
import warnings
import traceback
import gc

warnings.filterwarnings('ignore')

print("Importing numpy...", flush=True)
t_import = time.time()
import numpy as np
print(f"  numpy OK ({time.time()-t_import:.1f}s)", flush=True)

print("Importing ijson...", flush=True)
t_import = time.time()
import ijson
print(f"  ijson OK ({time.time()-t_import:.1f}s)", flush=True)

# === PATHS ===
MONEYFLOW_PATH = r"/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet"
# 统一路径管理
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import NORTH_MONEY as NORTH_MONEY_PATH
TOP_LIST_PATH = r"/home/hermes/.hermes/openclaw-project/data/top_list_data.json"
OUTPUT_PARQUET = r"/home/hermes/.hermes/openclaw-project/data/a3_moneyflow_factors.parquet"
OUTPUT_REPORT = r"/home/hermes/.hermes/openclaw-project/data/a3_factor_report.json"

print("=" * 60, flush=True)
print("A3 资金流因子提取", flush=True)
print("=" * 60, flush=True)

# ============================================================
# STEP 1: Stream moneyflow data with ijson
# ============================================================
print("\n[1/6] 流式读取资金流数据...", flush=True)
t0 = time.time()

MF_COL_MAP = {
    'buy_sm_vol': 0, 'buy_sm_amount': 1, 'sell_sm_vol': 2, 'sell_sm_amount': 3,
    'buy_md_vol': 4, 'buy_md_amount': 5, 'sell_md_vol': 6, 'sell_md_amount': 7,
    'buy_lg_vol': 8, 'buy_lg_amount': 9, 'sell_lg_vol': 10, 'sell_lg_amount': 11,
    'buy_elg_vol': 12, 'buy_elg_amount': 13, 'sell_elg_vol': 14, 'sell_elg_amount': 15,
    'net_mf_amount': 16, 'net_mf_vol': 17,
}
MF_COLS = list(MF_COL_MAP.keys())
N_MF_COLS = len(MF_COLS)

# Store per-stock: {code: {'dates': [...], 'data': [[col0, col1, ...], ...]}}
stock_data = {}
stock_count = 0
record_count = 0

with open(MONEYFLOW_PATH, 'rb') as f:
    for code, records in ijson.kvitems(f, ''):
        if not records or len(records) < 10:
            continue
        
        dates = []
        data = []
        for r in records:
            dates.append(r.get('trade_date', ''))
            row = [0.0] * N_MF_COLS
            for col_name, col_idx in MF_COL_MAP.items():
                val = r.get(col_name)
                if val is not None:
                    try:
                        row[col_idx] = float(val)
                    except (ValueError, TypeError):
                        row[col_idx] = 0.0
            data.append(row)
        
        # Sort by date
        paired = sorted(zip(dates, data), key=lambda x: x[0])
        stock_data[code] = {
            'dates': [p[0] for p in paired],
            'data': [p[1] for p in paired],
        }
        
        stock_count += 1
        record_count += len(records)
        if stock_count % 500 == 0:
            elapsed = time.time() - t0
            print(f"  已读取 {stock_count} 只股票, {record_count} 条记录 ({elapsed:.0f}s)", flush=True)

elapsed = time.time() - t0
print(f"  完成: {stock_count} 只股票, {record_count} 条记录 ({elapsed:.1f}s)", flush=True)
print(f"  内存占用: {sum(len(v['data'])*len(v['data'][0])*8 for v in stock_data.values())/1024/1024:.0f} MB", flush=True)

# ============================================================
# STEP 2: Calculate 15 moneyflow factors per stock
# ============================================================
print("\n[2/6] 计算资金流因子 (15个)...", flush=True)
t2 = time.time()

# Column indices
BUY_SM_VOL = 0; BUY_SM_AMT = 1; SELL_SM_VOL = 2; SELL_SM_AMT = 3
BUY_MD_VOL = 4; BUY_MD_AMT = 5; SELL_MD_VOL = 6; SELL_MD_AMT = 7
BUY_LG_VOL = 8; BUY_LG_AMT = 9; SELL_LG_VOL = 10; SELL_LG_AMT = 11
BUY_ELG_VOL = 12; BUY_ELG_AMT = 13; SELL_ELG_VOL = 14; SELL_ELG_AMT = 15
NET_MF_AMT = 16; NET_MF_VOL = 17

# Pre-allocate lists for all factor columns
# We'll build arrays per stock then concatenate
all_ts_codes = []
all_dates = []
all_factor_arrays = {
    'mf_net_mf_5d': [],
    'mf_net_mf_10d': [],
    'mf_net_mf_trend': [],
    'mf_elg_net': [],
    'mf_lg_net': [],
    'mf_md_net': [],
    'mf_sm_net': [],
    'mf_vol_ratio': [],
    'mf_elg_activity': [],
    'mf_lg_activity': [],
    'mf_sm_sell_ratio': [],
    'mf_divergence': [],  # will be NaN (no price data merged yet)
    'mf_momentum': [],
    'mf_elg_momentum': [],
    'mf_panic_index': [],
}

processed = 0
skipped = 0

def rolling_mean(arr, window):
    """Simple rolling mean with min_periods=1"""
    result = np.empty(len(arr))
    cumsum = np.cumsum(arr)
    for i in range(len(arr)):
        w = min(i + 1, window)
        start = i - w + 1
        if start <= 0:
            result[i] = cumsum[i] / (i + 1)
        else:
            result[i] = (cumsum[i] - cumsum[start - 1]) / w
    return result

def rolling_slope(arr, window):
    """5-day linear regression slope"""
    n = len(arr)
    result = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(window - 1, n):
        y = arr[i - window + 1:i + 1]
        if np.any(np.isnan(y)):
            continue
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        result[i] = slope
    # Fill initial NaNs with first valid
    first_valid = None
    for i in range(n):
        if not np.isnan(result[i]):
            first_valid = result[i]
            break
    if first_valid is not None:
        for i in range(n):
            if np.isnan(result[i]):
                result[i] = first_valid
            else:
                break
    return result

def pct_change(arr, period):
    """Percentage change over period"""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period, n):
        if arr[i - period] != 0 and not np.isnan(arr[i - period]):
            result[i] = (arr[i] - arr[i - period]) / abs(arr[i - period])
    return result

for code, sdata in stock_data.items():
    try:
        dates = sdata['dates']
        data = sdata['data']
        n = len(data)
        
        if n < 10:
            skipped += 1
            continue
        
        # Convert to numpy arrays for speed
        arr = np.array(data, dtype=np.float64)  # shape: (n, 18)
        
        # Extract columns
        net_mf = arr[:, NET_MF_AMT]
        buy_elg = arr[:, BUY_ELG_AMT]
        sell_elg = arr[:, SELL_ELG_AMT]
        buy_lg = arr[:, BUY_LG_AMT]
        sell_lg = arr[:, SELL_LG_AMT]
        buy_md = arr[:, BUY_MD_AMT]
        sell_md = arr[:, SELL_MD_AMT]
        buy_sm = arr[:, BUY_SM_AMT]
        sell_sm = arr[:, SELL_SM_AMT]
        buy_sm_vol = arr[:, BUY_SM_VOL]
        buy_md_vol = arr[:, BUY_MD_VOL]
        buy_lg_vol = arr[:, BUY_LG_VOL]
        buy_elg_vol = arr[:, BUY_ELG_VOL]
        sell_sm_vol = arr[:, SELL_SM_VOL]
        sell_md_vol = arr[:, SELL_MD_VOL]
        sell_lg_vol = arr[:, SELL_LG_VOL]
        sell_elg_vol = arr[:, SELL_ELG_VOL]
        
        # Factor 1: net_mf_5d
        f1 = rolling_mean(net_mf, 5)
        # Factor 2: net_mf_10d
        f2 = rolling_mean(net_mf, 10)
        # Factor 3: net_mf_trend
        f3 = rolling_slope(net_mf, 5)
        # Factor 4-7: tier net flows
        f4 = buy_elg - sell_elg  # elg_net
        f5 = buy_lg - sell_lg    # lg_net
        f6 = buy_md - sell_md    # md_net
        f7 = buy_sm - sell_sm    # sm_net
        
        # Factor 8: vol_ratio
        buy_vol_total = buy_sm_vol + buy_md_vol + buy_lg_vol + buy_elg_vol
        sell_vol_total = sell_sm_vol + sell_md_vol + sell_lg_vol + sell_elg_vol
        f8 = np.where(sell_vol_total > 0, buy_vol_total / sell_vol_total, 1.0)
        
        # Factor 9-11: activity ratios
        total_amt = buy_sm + buy_md + buy_lg + buy_elg + sell_sm + sell_md + sell_lg + sell_elg
        total_amt_safe = np.where(total_amt > 0, total_amt, 1.0)
        f9 = (buy_elg + sell_elg) / total_amt_safe   # elg_activity
        f10 = (buy_lg + sell_lg) / total_amt_safe     # lg_activity
        f11 = sell_sm / total_amt_safe                 # sm_sell_ratio
        
        # Factor 12: divergence (no price data here, fill NaN)
        f12 = np.full(n, np.nan)
        
        # Factor 13-14: momentum
        f13 = pct_change(net_mf, 5)    # mf_momentum
        f14 = pct_change(f4, 5)         # elg_momentum
        
        # Factor 15: panic index
        f15 = pct_change(f11, 5)
        
        # Fill NaN in momentum/panic with 0
        f13 = np.nan_to_num(f13, nan=0.0)
        f14 = np.nan_to_num(f14, nan=0.0)
        f15 = np.nan_to_num(f15, nan=0.0)
        f3 = np.nan_to_num(f3, nan=0.0)
        
        # Append
        all_ts_codes.extend([code] * n)
        all_dates.extend(dates)
        all_factor_arrays['mf_net_mf_5d'].append(f1)
        all_factor_arrays['mf_net_mf_10d'].append(f2)
        all_factor_arrays['mf_net_mf_trend'].append(f3)
        all_factor_arrays['mf_elg_net'].append(f4)
        all_factor_arrays['mf_lg_net'].append(f5)
        all_factor_arrays['mf_md_net'].append(f6)
        all_factor_arrays['mf_sm_net'].append(f7)
        all_factor_arrays['mf_vol_ratio'].append(f8)
        all_factor_arrays['mf_elg_activity'].append(f9)
        all_factor_arrays['mf_lg_activity'].append(f10)
        all_factor_arrays['mf_sm_sell_ratio'].append(f11)
        all_factor_arrays['mf_divergence'].append(f12)
        all_factor_arrays['mf_momentum'].append(f13)
        all_factor_arrays['mf_elg_momentum'].append(f14)
        all_factor_arrays['mf_panic_index'].append(f15)
        
        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - t2
            print(f"  已处理 {processed}/{stock_count} 只股票 ({elapsed:.0f}s)", flush=True)
    
    except Exception as e:
        skipped += 1
        if skipped <= 3:
            print(f"  跳过 {code}: {e}", flush=True)
        continue

elapsed = time.time() - t2
print(f"  完成: {processed} 只股票成功, {skipped} 跳过 ({elapsed:.1f}s)", flush=True)

# Concatenate all factor arrays
print("  合并因子数组...", flush=True)
factor_cols = {}
for name, arrays in all_factor_arrays.items():
    factor_cols[name] = np.concatenate(arrays)

total_rows = len(all_ts_codes)
print(f"  总行数: {total_rows}", flush=True)

# Free memory from stock_data
del stock_data
gc.collect()

# ============================================================
# STEP 3: North money factors (5个)
# ============================================================
print("\n[3/6] 计算北向资金因子 (5个)...", flush=True)
t3 = time.time()

north_by_date = {}  # date_str -> {5 factor values}
try:
    with open(NORTH_MONEY_PATH, 'r', encoding='utf-8') as f:
        north_raw = json.load(f)
    
    if isinstance(north_raw, dict) and 'records' in north_raw:
        north_records = north_raw['records']
    elif isinstance(north_raw, list):
        north_records = north_raw
    else:
        north_records = []
    
    print(f"  北向资金记录数: {len(north_records)}", flush=True)
    
    if north_records:
        # Sort by date
        north_records.sort(key=lambda x: x.get('trade_date', ''))
        
        dates_n = []
        nm_values = []
        for r in north_records:
            d = r.get('trade_date', '')
            nm = r.get('north_money')
            if nm is None:
                hgt = r.get('hgt')
                sgt = r.get('sgt')
                nm = (float(hgt) if hgt else 0) + (float(sgt) if sgt else 0)
            else:
                try:
                    nm = float(nm)
                except:
                    nm = 0.0
            dates_n.append(d)
            nm_values.append(nm)
        
        nm_arr = np.array(nm_values, dtype=np.float64)
        n_north = len(nm_arr)
        
        # Factor 1: 5d average
        n1 = rolling_mean(nm_arr, 5)
        # Factor 2: trend
        n2 = rolling_slope(nm_arr, 5)
        n2 = np.nan_to_num(n2, nan=0.0)
        # Factor 3: consecutive days
        n3 = np.zeros(n_north)
        count = 0
        for i in range(n_north):
            if nm_arr[i] > 0:
                count = count + 1 if count > 0 else 1
            elif nm_arr[i] < 0:
                count = count - 1 if count < 0 else -1
            else:
                count = 0
            n3[i] = count
        # Factor 4: 5d sum
        n4 = np.empty(n_north)
        cumsum_n = np.cumsum(nm_arr)
        for i in range(n_north):
            w = min(i + 1, 5)
            start = i - w + 1
            if start <= 0:
                n4[i] = cumsum_n[i]
            else:
                n4[i] = cumsum_n[i] - cumsum_n[start - 1]
        # Factor 5: momentum
        n5 = pct_change(nm_arr, 5)
        n5 = np.nan_to_num(n5, nan=0.0)
        
        for i in range(n_north):
            north_by_date[dates_n[i]] = (n1[i], n2[i], n3[i], n4[i], n5[i])
        
        print(f"  北向因子: {len(north_by_date)} 个交易日", flush=True)

except Exception as e:
    print(f"  北向资金处理失败: {e}", flush=True)
    traceback.print_exc()

elapsed = time.time() - t3
print(f"  ({elapsed:.1f}s)", flush=True)

# ============================================================
# STEP 4: Top list (dragon tiger) factors (5个)
# ============================================================
print("\n[4/6] 计算龙虎榜因子 (5个)...", flush=True)
t4 = time.time()

dragon_by_stock_date = {}  # (code, date) -> {5 factors}
try:
    with open(TOP_LIST_PATH, 'r', encoding='utf-8') as f:
        top_raw = json.load(f)
    
    if isinstance(top_raw, dict) and 'records' in top_raw:
        top_records = top_raw['records']
    elif isinstance(top_raw, list):
        top_records = top_raw
    elif isinstance(top_raw, dict):
        top_records = []
        for date_key, recs in top_raw.items():
            if isinstance(recs, list):
                for r in recs:
                    r['trade_date'] = r.get('trade_date', date_key)
                    top_records.append(r)
    else:
        top_records = []
    
    print(f"  龙虎榜记录数: {len(top_records)}", flush=True)
    
    if top_records:
        # Group by (ts_code, trade_date)
        from collections import defaultdict
        grouped = defaultdict(lambda: {'count': 0, 'net_amount': 0.0, 'l_buy': 0.0, 'l_sell': 0.0})
        
        for r in top_records:
            code = r.get('ts_code', '')
            date = str(r.get('trade_date', ''))
            if not code or not date:
                continue
            key = (code, date)
            grouped[key]['count'] += 1
            try:
                grouped[key]['net_amount'] += float(r.get('net_amount', 0) or 0)
            except:
                pass
            try:
                grouped[key]['l_buy'] += float(r.get('l_buy', 0) or 0)
            except:
                pass
            try:
                grouped[key]['l_sell'] += float(r.get('l_sell', 0) or 0)
            except:
                pass
        
        print(f"  龙虎榜(股票,日期)对数: {len(grouped)}", flush=True)
        
        # Now compute rolling per stock
        stock_dragon = defaultdict(list)  # code -> [(date, count, net, l_buy, l_sell)]
        for (code, date), vals in grouped.items():
            stock_dragon[code].append((date, vals['count'], vals['net_amount'], vals['l_buy'], vals['l_sell']))
        
        for code, entries in stock_dragon.items():
            entries.sort(key=lambda x: x[0])
            n_e = len(entries)
            dates_e = [e[0] for e in entries]
            counts = np.array([e[1] for e in entries], dtype=np.float64)
            nets = np.array([e[2] for e in entries], dtype=np.float64)
            l_buys = np.array([e[3] for e in entries], dtype=np.float64)
            
            # 20d count sum
            d1 = rolling_mean(counts, 20) * np.minimum(np.arange(1, n_e + 1), 20)
            # Actually just use cumsum-based rolling sum
            d1 = np.empty(n_e)
            cs = np.cumsum(counts)
            for i in range(n_e):
                w = min(i + 1, 20)
                s = i - w + 1
                d1[i] = cs[i] if s <= 0 else cs[i] - cs[s - 1]
            
            # 5d inst_buy (l_buy sum)
            d2 = np.empty(n_e)
            cs2 = np.cumsum(l_buys)
            for i in range(n_e):
                w = min(i + 1, 5)
                s = i - w + 1
                d2[i] = cs2[i] if s <= 0 else cs2[i] - cs2[s - 1]
            
            # 5d inst_net
            d3 = np.empty(n_e)
            cs3 = np.cumsum(nets)
            for i in range(n_e):
                w = min(i + 1, 5)
                s = i - w + 1
                d3[i] = cs3[i] if s <= 0 else cs3[i] - cs3[s - 1]
            
            # 5d hot_money (std of net)
            d4 = np.full(n_e, 0.0)
            for i in range(4, n_e):
                d4[i] = np.std(nets[i - 4:i + 1])
            
            # dragon_premium = net_amount_mean (5d)
            d5 = np.empty(n_e)
            cs5 = np.cumsum(nets)
            for i in range(n_e):
                w = min(i + 1, 5)
                s = i - w + 1
                total = cs5[i] if s <= 0 else cs5[i] - cs5[s - 1]
                d5[i] = total / w
            
            for i in range(n_e):
                dragon_by_stock_date[(code, dates_e[i])] = (d1[i], d2[i], d3[i], d4[i], d5[i])
        
        print(f"  龙虎榜因子: {len(dragon_by_stock_date)} 个(股票,日期)对", flush=True)

except Exception as e:
    print(f"  龙虎榜处理失败: {e}", flush=True)
    traceback.print_exc()

elapsed = time.time() - t4
print(f"  ({elapsed:.1f}s)", flush=True)

# ============================================================
# STEP 5: Build final arrays and save parquet
# ============================================================
print("\n[5/6] 合并因子并保存parquet...", flush=True)
t5 = time.time()

# Build north and dragon arrays aligned to all_dates/all_ts_codes
north_net_5d = np.zeros(total_rows)
north_trend = np.zeros(total_rows)
north_consecutive = np.zeros(total_rows)
north_ratio = np.zeros(total_rows)
north_momentum = np.zeros(total_rows)

dragon_count_20d = np.zeros(total_rows)
dragon_inst_buy_5d = np.zeros(total_rows)
dragon_inst_net_5d = np.zeros(total_rows)
dragon_hot_money_5d = np.zeros(total_rows)
dragon_premium = np.zeros(total_rows)

if north_by_date:
    for i in range(total_rows):
        d = all_dates[i]
        if d in north_by_date:
            vals = north_by_date[d]
            north_net_5d[i] = vals[0]
            north_trend[i] = vals[1]
            north_consecutive[i] = vals[2]
            north_ratio[i] = vals[3]
            north_momentum[i] = vals[4]

if dragon_by_stock_date:
    for i in range(total_rows):
        key = (all_ts_codes[i], all_dates[i])
        if key in dragon_by_stock_date:
            vals = dragon_by_stock_date[key]
            dragon_count_20d[i] = vals[0]
            dragon_inst_buy_5d[i] = vals[1]
            dragon_inst_net_5d[i] = vals[2]
            dragon_hot_money_5d[i] = vals[3]
            dragon_premium[i] = vals[4]

print(f"  构建完成, 准备写入parquet...", flush=True)

# Write parquet using pyarrow
import pyarrow as pa
import pyarrow.parquet as pq

# Build column arrays
columns = {
    'ts_code': all_ts_codes,
    'trade_date': all_dates,
}
# Add mf factors
for name, arr in factor_cols.items():
    columns[name] = arr

# Add north factors
columns['north_net_5d'] = north_net_5d
columns['north_trend'] = north_trend
columns['north_consecutive'] = north_consecutive
columns['north_ratio'] = north_ratio
columns['north_momentum'] = north_momentum

# Add dragon factors
columns['dragon_count_20d'] = dragon_count_20d
columns['dragon_inst_buy_5d'] = dragon_inst_buy_5d
columns['dragon_inst_net_5d'] = dragon_inst_net_5d
columns['dragon_hot_money_5d'] = dragon_hot_money_5d
columns['dragon_premium'] = dragon_premium

# Create table
table = pa.table(columns)

# Write
pq.write_table(table, OUTPUT_PARQUET, compression='snappy')
file_size = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
print(f"  Parquet保存: {OUTPUT_PARQUET}", flush=True)
print(f"  大小: {file_size:.1f} MB", flush=True)

# ============================================================
# STEP 6: Factor statistics report
# ============================================================
print("\n[6/6] 生成因子统计报告...", flush=True)

mf_col_names = list(factor_cols.keys())
north_col_names = ['north_net_5d', 'north_trend', 'north_consecutive', 'north_ratio', 'north_momentum']
dragon_col_names = ['dragon_count_20d', 'dragon_inst_buy_5d', 'dragon_inst_net_5d', 'dragon_hot_money_5d', 'dragon_premium']

report = {
    'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'total_stocks': len(set(all_ts_codes)),
    'total_rows': total_rows,
    'date_range': [min(all_dates), max(all_dates)],
    'parquet_size_mb': round(file_size, 1),
    'factor_summary': {
        'moneyflow_factors': len(mf_col_names),
        'north_factors': len(north_col_names),
        'dragon_factors': len(dragon_col_names),
        'total_factors': len(mf_col_names) + len(north_col_names) + len(dragon_col_names),
    },
    'factor_columns': {},
}

all_col_names = mf_col_names + north_col_names + dragon_col_names
all_col_arrays = list(factor_cols.values()) + [north_net_5d, north_trend, north_consecutive, north_ratio, north_momentum,
    dragon_count_20d, dragon_inst_buy_5d, dragon_inst_net_5d, dragon_hot_money_5d, dragon_premium]

for name, arr in zip(all_col_names, all_col_arrays):
    valid_mask = ~np.isnan(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(len(arr), dtype=bool)
    valid = arr[valid_mask]
    report['factor_columns'][name] = {
        'valid_count': int(len(valid)),
        'nan_count': int(total_rows - len(valid)),
        'coverage_pct': round(len(valid) / total_rows * 100, 2) if total_rows > 0 else 0,
        'mean': round(float(np.mean(valid)), 4) if len(valid) > 0 else 0,
        'std': round(float(np.std(valid)), 4) if len(valid) > 0 else 0,
        'min': round(float(np.min(valid)), 4) if len(valid) > 0 else 0,
        'max': round(float(np.max(valid)), 4) if len(valid) > 0 else 0,
        'median': round(float(np.median(valid)), 4) if len(valid) > 0 else 0,
    }

with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"  报告保存: {OUTPUT_REPORT}", flush=True)

# ============================================================
# Summary
# ============================================================
total_time = time.time() - t0
print("\n" + "=" * 60, flush=True)
print("✅ 完成!", flush=True)
print(f"  总耗时: {total_time:.1f}s", flush=True)
print(f"  股票数: {report['total_stocks']}", flush=True)
print(f"  总行数: {report['total_rows']}", flush=True)
print(f"  日期范围: {report['date_range']}", flush=True)
print(f"  因子列数: {report['factor_summary']['total_factors']}", flush=True)
print(f"    - 资金流因子: {len(mf_col_names)}", flush=True)
print(f"    - 北向资金因子: {len(north_col_names)}", flush=True)
print(f"    - 龙虎榜因子: {len(dragon_col_names)}", flush=True)
print(f"  Parquet大小: {file_size:.1f} MB", flush=True)
print("=" * 60, flush=True)
