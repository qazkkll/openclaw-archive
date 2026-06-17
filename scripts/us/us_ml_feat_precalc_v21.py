"""在现有us_ml_feats基础上加5天label+新特征"""
import sys, json, os, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import _paths

T0 = time.time()
print("═══ 预计算v2.1: 基于现有feats加5天label ═══")

df = pd.read_parquet(_paths.US_ML_FEATS)
print(f"  原始: {len(df):,}行, {df['sym'].nunique()}只")
print(f"  现有列: {df.columns.tolist()}")

# 必须按股票分组+保持原始顺序
# 由于没有date列，假设原始数据已按时间顺序排列

# 按sym分组，每组内算forward_5d
def add_forward_5d(g):
    g = g.copy()
    close = g['price'].values
    n = len(g)
    # 5天后收益
    fwd5 = np.full(n, np.nan)
    fwd5[:n-5] = (close[5:] - close[:n-5]) / close[:n-5] * 100
    g['label_5d_pct'] = fwd5
    
    # 5档分类
    cls = np.full(n, -1, dtype=int)
    cls[fwd5 < -5] = 0
    cls[(fwd5 >= -5) & (fwd5 < -2)] = 1
    cls[(fwd5 >= -2) & (fwd5 <= 2)] = 2
    cls[(fwd5 > 2) & (fwd5 <= 5)] = 3
    cls[fwd5 > 5] = 4
    cls[np.isnan(fwd5)] = -1
    g['label_5d_5class'] = cls
    
    # 5天波动率
    ret1 = g['ret1'].values
    vol5 = np.full(n, np.nan)
    for i in range(4, n):
        vol5[i] = np.nanstd(ret1[i-4:i+1]) * 100
    g['vol5'] = vol5
    
    # 趋势加速(5天到20天)
    g['trend_accel'] = g['ret20'].fillna(0) - g['ret5'].fillna(0)
    
    return g

print("\n[1/3] 计算5天label...")
df2 = df.groupby('sym', group_keys=False).apply(add_forward_5d)
df2 = df2.dropna(subset=['label_5d_pct'])
print(f"  有5d label: {len(df2):,}行")

for cl in range(5):
    cnt = (df2['label_5d_5class'] == cl).sum()
    print(f"    第{cl}档 ({['跌>5%','跌2-5%','平±2%','涨2-5%','涨>5%'][cl]}): {cnt:,}行 ({cnt/len(df2)*100:.1f}%)")

print(f"\n[2/3] 保存...")
save_path = _paths.ML_DIR + "/us_ml_feats_v2.parquet"
df2.to_parquet(save_path, index=False)

# 特征列表
base = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
        'ret1','ret5','ret20','ret60',
        'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
        'vol5','trend_accel','label_5d_pct','label_5d_5class']
feats_no_target = ['price','volume','ma5','ma10','ma20','ma60','rsi14','vol20','p52',
                   'ret1','ret5','ret20','ret60',
                   'macd','macd_signal','macd_hist','vol_ratio','ma_bias20',
                   'vol5','trend_accel']
with open(_paths.ML_DIR + "/us_feature_cols_v2.json", 'w') as f:
    json.dump(feats_no_target, f, indent=2)

print(f"[3/3] 统计")
print(f"  保存: {save_path}")
print(f"  特征列: {len(feats_no_target)}个")
print(f"  标签: label_5d_pct, label_5d_5class")

TOTAL = time.time() - T0
print(f"\n✅ 预计算v2.1 完成! ({TOTAL:.0f}s)")
