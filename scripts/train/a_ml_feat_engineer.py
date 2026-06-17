"""ML第1步：特征工程 — 生成训练数据
从资金流 + K线生成特征向量，label=次日涨跌

输出: D:\\openclaw_ml\\ml_training_data.parquet / ml_feature_cols.json
运行: python scripts/ml_feat_engineer.py
预计: 30-60分钟
"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_IN = os.path.join(WORKSPACE, 'data')
DATA_OUT = r'/home/hermes/.hermes/openclaw-archive/data'
os.makedirs(DATA_OUT, exist_ok=True)

T0 = time.time()

# ─── 1. 加载资金流 ───
print('[1/4] 加载资金流数据...')
t = time.time()
with open(os.path.join(DATA_IN, 'moneyflow_data.parquet'), 'r', encoding='utf-8') as f:
    mf_raw = json.load(f)
print(f'  {len(mf_raw)}只股票 | {time.time()-t:.0f}秒')

# ─── 2. 加载K线数据 ───
print('[2/4] 加载K线数据...')
t = time.time()
with open(os.path.join(DATA_IN, 'a_hist_10y.parquet'), 'r', encoding='utf-8') as f:
    kl_raw = json.load(f)
print(f'  {len(kl_raw)}只股票 | {time.time()-t:.0f}秒')

# ─── 3. 对齐code ───
def normalize_code(code):
    return code.replace('.SZ', '').replace('.SH', '').replace('.BJ', '')

# ─── 4. 遍历每只股票 ───
print('[3/4] 合并特征...')
t = time.time()

rows = []
processed = 0
skipped_no_kl = 0
skipped_short = 0

MF_FIELDS = [
    'net_mf_amount', 'buy_sm_amount', 'sell_sm_amount',
    'buy_md_amount', 'sell_md_amount',
    'buy_lg_amount', 'sell_lg_amount',
    'buy_elg_amount', 'sell_elg_amount',
]

for code_mf, mf_list in mf_raw.items():
    code_norm = normalize_code(code_mf)
    
    if code_norm not in kl_raw:
        skipped_no_kl += 1
        continue
    
    kl = kl_raw[code_norm]
    if len(mf_list) < 50:
        skipped_short += 1
        continue
    
    # 资金流倒序转正序
    mf_df = pd.DataFrame(mf_list).sort_values('trade_date').reset_index(drop=True)
    
    # K线
    kl_df = pd.DataFrame({
        'trade_date': kl['dates'],
        'open': kl['o'], 'high': kl['h'], 'low': kl['l'],
        'close': kl['c'], 'vol': kl['v'],
    })
    kl_df['trade_date'] = kl_df['trade_date'].astype(str)
    
    merged = pd.merge(kl_df, mf_df, on='trade_date', how='inner', suffixes=('', '_mf'))
    if len(merged) < 60:
        skipped_short += 1
        continue
    
    # 技术指标
    merged['pct_chg'] = merged['close'].pct_change() * 100
    
    for w in [5, 10, 20]:
        merged[f'ma{w}'] = merged['close'].rolling(w).mean()
        merged[f'vol_ma{w}'] = merged['vol'].rolling(w).mean()
        merged[f'mf_ma{w}'] = merged['net_mf_amount'].rolling(w).mean()
    
    merged['big_buy_ratio'] = merged['buy_lg_amount'] / (merged['buy_lg_amount'] + merged['sell_lg_amount'] + 1)
    merged['big_net_ratio'] = (merged['buy_lg_amount'] - merged['sell_lg_amount']) / (merged['buy_lg_amount'] + merged['sell_lg_amount'] + 1)
    merged['elg_buy_ratio'] = merged['buy_elg_amount'] / (merged['buy_elg_amount'] + merged['sell_elg_amount'] + 1)
    merged['elg_net_ratio'] = (merged['buy_elg_amount'] - merged['sell_elg_amount']) / (merged['buy_elg_amount'] + merged['sell_elg_amount'] + 1)
    
    delta = merged['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean() + 0.001
    merged['rsi_14'] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    merged['day_of_week'] = pd.to_datetime(merged['trade_date']).dt.dayofweek
    merged['month'] = pd.to_datetime(merged['trade_date']).dt.month
    
    # Label: 次日涨（不用shift，直接用下一个交易日的数据）
    merged['close_next'] = merged['close'].shift(-1)
    merged['label'] = (merged['close_next'] > merged['close']).astype(int)
    
    merged['code'] = code_mf
    merged = merged.dropna(subset=['label'])
    
    if len(merged) < 50:
        continue
    
    # 保留合适字段
    keep_cols = ['trade_date', 'code', 'close', 'label', 'net_mf_amount',
                 'pct_chg', 'ma5', 'ma10', 'ma20', 'vol_ma5', 'vol_ma10',
                 'mf_ma5', 'mf_ma10', 'mf_ma20',
                 'big_buy_ratio', 'big_net_ratio', 'elg_buy_ratio', 'elg_net_ratio',
                 'rsi_14', 'day_of_week', 'month', 'vol', 'close_next']
    keep_cols = [c for c in keep_cols if c in merged.columns]
    rows.append(merged[keep_cols])
    
    processed += 1
    if processed % 200 == 0:
        print(f'  处理 {processed}/{len(mf_raw)} | 行数 {sum(len(r) for r in rows)}')

print(f'  完成 | 处理{processed}只 | 无K线:{skipped_no_kl} | 太短:{skipped_short}')
t2 = time.time()
print(f'  合并耗时: {t2-t:.0f}秒')

all_data = pd.concat(rows, ignore_index=True)
print(f'  总行数: {len(all_data)}')

# 特征列名
feature_cols = [c for c in all_data.columns if c not in ['label', 'trade_date', 'code', 'close_next']]
with open(os.path.join(DATA_OUT, 'ml_feature_cols.json'), 'w') as f:
    json.dump(feature_cols, f)
print(f'  特征列: {len(feature_cols)}个')

# 保存parquet到D盘
all_data.to_parquet(os.path.join(DATA_OUT, 'ml_training_data.parquet'), index=False)
print(f'  已保存: D:\\openclaw_ml\\ml_training_data.parquet ({len(all_data)}行)')

TOTAL = time.time() - T0
print(f'✅ 总耗时: {TOTAL:.0f}秒 ({TOTAL/60:.1f}分钟)')
