"""
step2_ml_features_v3.py — A股ML特征工程（基于parquet行组迭代版）
从已有的 moneyflow_data.parquet 按行组读取，逐股票计算特征
用pyarrow避免pandas转换内存爆炸

输出: /home/hermes/.hermes/openclaw-project/scripts/system/a_ml_feats_v1.parquet
"""
import sys, gc, time, os
sys.stdout.reconfigure(encoding='utf-8')
from pyarrow.parquet import ParquetFile
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

SRC = '/home/hermes/.hermes/openclaw-project/data/moneyflow_data.parquet'
DST = '/home/hermes/.hermes/openclaw-project/data/a_ml_feats_v1.parquet'
TMP_DIR = '/home/hermes/.hermes/openclaw-project/scripts/system/feat_batches'
os.makedirs(TMP_DIR, exist_ok=True)

t0 = time.time()
pf = ParquetFile(SRC)
total_rgs = pf.metadata.num_row_groups
total_stocks = 0
total_features = 0
batch_files = []

print(f'资金来源流parquet: {total_rgs}个行组')
print(f'总行数: {pf.metadata.num_rows}')

FEAT_COLS = ['stock_code', 'trade_date', 'net_mf', 'net_mf_ratio', 'big_net', 'xbig_net',
             'mid_net', 'net_mf_3d', 'net_mf_trend', 'big_small_div', 'ret_5d', 'label']

def compute_features_for_stock(tbl):
    """用pyarrow处理一只股票的数据"""
    codes = tbl.column(0).to_pylist()  # ts_code
    dates = tbl.column(1).to_pylist()  # trade_date
    n = len(codes)
    
    if n < 50:
        return []
    
    # 提取数值列
    field_names = [tbl.field(i).name for i in range(tbl.num_columns)]
    
    def get_col(name):
        for i, fn in enumerate(field_names):
            if fn == name:
                return np.array(tbl.column(i).to_numpy(zero_copy_only=False), dtype=np.float64)
        return np.zeros(n, dtype=np.float64)
    
    mf = get_col('net_mf_amount')
    mf_ratio = get_col('net_mf_ratio')
    buy_lg = get_col('buy_lg_amount')
    sell_lg = get_col('sell_lg_amount')
    buy_elg = get_col('buy_elg_amount')
    sell_elg = get_col('sell_elg_amount')
    buy_md = get_col('buy_md_amount')
    sell_md = get_col('sell_md_amount')
    buy_sm = get_col('buy_sm_amount')
    sell_sm = get_col('sell_sm_amount')
    close = get_col('close')
    
    features = []
    for i in range(n - 10):
        # 当前行
        f = {}
        
        # 资金流特征
        net = mf[i] if not np.isnan(mf[i]) else 0
        f['net_mf'] = net / 1e4 if abs(net) > 1 else net
        f['net_mf_ratio'] = mf_ratio[i] if not np.isnan(mf_ratio[i]) else 0
        f['big_net'] = ((buy_lg[i] if not np.isnan(buy_lg[i]) else 0) -
                       (sell_lg[i] if not np.isnan(sell_lg[i]) else 0)) / 1e4
        f['xbig_net'] = ((buy_elg[i] if not np.isnan(buy_elg[i]) else 0) -
                        (sell_elg[i] if not np.isnan(sell_elg[i]) else 0)) / 1e4
        f['mid_net'] = ((buy_md[i] if not np.isnan(buy_md[i]) else 0) -
                       (sell_md[i] if not np.isnan(sell_md[i]) else 0)) / 1e4
        
        # 3日趋势
        if i >= 2:
            mf3 = sum(mf[i-j] if not np.isnan(mf[i-j]) else 0 for j in range(3))
            f['net_mf_3d'] = mf3 / 1e4 if abs(mf3) > 1 else mf3
            f['net_mf_trend'] = 1.0 if mf3 > 0 else 0.0
        else:
            f['net_mf_3d'] = f['net_mf']
            f['net_mf_trend'] = 1.0 if f['net_mf'] > 0 else 0.0
        
        # 主力vs散户背离
        big_in = (buy_lg[i] if not np.isnan(buy_lg[i]) else 0) + (buy_elg[i] if not np.isnan(buy_elg[i]) else 0)
        big_out = (sell_lg[i] if not np.isnan(sell_lg[i]) else 0) + (sell_elg[i] if not np.isnan(sell_elg[i]) else 0)
        small_net = (buy_sm[i] if not np.isnan(buy_sm[i]) else 0) - (sell_sm[i] if not np.isnan(sell_sm[i]) else 0)
        f['big_small_div'] = (big_in - big_out) - small_net
        
        # Y标签
        if i + 5 < n:
            curr_c = close[i] if not np.isnan(close[i]) else 0
            fut_c = close[i+5] if not np.isnan(close[i+5]) else 0
            if curr_c > 0 and fut_c > 0:
                ret = (fut_c - curr_c) / curr_c
                f['ret_5d'] = round(ret, 6)
                f['label'] = 1.0 if ret > 0.02 else 0.0
                f['stock_code'] = codes[i]
                f['trade_date'] = int(dates[i])
                features.append(f)
    
    return features

# 逐行组处理
batch = []
batch_idx = 0

for rg_idx in range(total_rgs):
    try:
        tbl = pf.read_row_group(rg_idx, 
            columns=['ts_code','trade_date','net_mf_amount','net_mf_ratio',
                     'buy_lg_amount','sell_lg_amount','buy_elg_amount','sell_elg_amount',
                     'buy_md_amount','sell_md_amount','buy_sm_amount','sell_sm_amount','close'])
    except:
        print(f'  ⚠️ 行组{rg_idx}跳过')
        continue
    
    if tbl.num_rows < 50:
        total_stocks += 1
        continue
    
    stock_code = tbl.column(0).to_pylist()[0]  # 该行组对应一只股票
    feats = compute_features_for_stock(tbl)
    batch.extend(feats)
    total_stocks += 1
    total_features += len(feats)
    
    if total_stocks % 50 == 0:
        print(f'  {total_stocks}/{total_rgs}只, {total_features}行特征', flush=True)
    
    # 每300只存盘
    if total_stocks % 300 == 0 and batch:
        import pandas as pd
        df = pd.DataFrame(batch)
        fpath = os.path.join(TMP_DIR, f'batch_{batch_idx}.parquet')
        df.to_parquet(fpath, index=False, compression='snappy')
        batch_files.append(fpath)
        batch = []
        batch_idx += 1
        print(f'  💾 存盘batch_{batch_idx}, {len(df)}行', flush=True)
        del df
        gc.collect()

# 最后一批
if batch:
    import pandas as pd
    df = pd.DataFrame(batch)
    fpath = os.path.join(TMP_DIR, f'batch_{batch_idx}.parquet')
    df.to_parquet(fpath, index=False, compression='snappy')
    batch_files.append(fpath)
    del df, batch
    gc.collect()

# 合并
if batch_files:
    import pandas as pd
    print(f'\n合并{batch_files}个分片...')
    parts = []
    for f in batch_files:
        parts.append(pd.read_parquet(f))
    final = pd.concat(parts, ignore_index=True)
    final.to_parquet(DST, index=False, compression='snappy')
    print(f'✅ 最终特征文件: {DST}')
    print(f'  总特征行数: {len(final)}')
    print(f'  特征维度: {len(final.columns)}')
    print(f'  日期范围: {final["trade_date"].min()} ~ {final["trade_date"].max()}')
    print(f'  股票数: {final["stock_code"].nunique()}')
else:
    print('❌ 没有生成任何特征')
    sys.exit(1)

# 清理临时文件
for f in batch_files:
    os.remove(f)

t1 = time.time()
elapsed = (t1 - t0) / 60
print(f'总耗时: {elapsed:.1f}分钟')
print(f'处理: {total_stocks}只股票, {total_features}行特征')
