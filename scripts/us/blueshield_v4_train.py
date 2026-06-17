# -*- coding: utf-8 -*-
"""
蓝盾V4 — ML做仓位分配，不做买卖决策
买卖由V5.5规则体系决定
ML的任务：每天给候选票打置信度分，规则决定买不买，ML决定买多少
"""
import warnings, json, os
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import xgboost as xgb

print('加载数据...')
feat = pd.read_parquet('/home/hermes/.hermes/openclaw-project/data/us/sp500_feats.parquet')
feat = feat.sort_values(['Code','Date']).reset_index(drop=True)
feat['Date'] = pd.to_datetime(feat['Date'])

raw_dir = '/home/hermes/.hermes/openclaw-project/data/hist_sp500'
all_rows = []
for f in sorted(os.listdir(raw_dir)):
    if not f.startswith('sp500_chunk_') or not f.endswith('.json'): continue
    raw = json.load(open(os.path.join(raw_dir, f)))
    for sym, bars in raw.items():
        for b in bars: b['Code'] = sym
        all_rows.extend(bars)
raw_df = pd.DataFrame(all_rows)
raw_df['Date'] = pd.to_datetime(raw_df['Date'])
raw_df['DollarVol'] = raw_df['C'] * raw_df['V']

feat = feat.merge(raw_df[['Code','Date','C','DollarVol']], on=['Code','Date'], how='left')
feat['dvol_ma5'] = feat.groupby('Code')['DollarVol'].transform(lambda x: x.rolling(5).mean())

# 基础特征
market_ret = feat.groupby('Date')['ret_1d'].mean().reset_index()
market_ret.columns = ['Date', 'market_ret']
feat = feat.merge(market_ret, on='Date', how='left')
feat['rel_ret_5d'] = feat['ret_5d'] - feat.groupby('Date')['ret_5d'].transform('mean')
feat['rel_ret_10d'] = feat['ret_10d'] - feat.groupby('Date')['ret_10d'].transform('mean')
feat['vol_5d_norm'] = feat['vol_5d'] / (feat.groupby('Date')['vol_5d'].transform('mean') + 1e-8)
feat['rsi_50_pct'] = (feat['rsi_14'] - 50) / 50
feat['ma20_ma50_cross'] = feat['ma_20_ratio'] - feat['ma_50_ratio']
feat['dvol_ratio'] = np.where(feat['dvol_ma5'] > 0, feat['DollarVol'] / feat['dvol_ma5'], 1.0)
feat['price_above_ma50'] = (feat['ma_50_ratio'] > 1.0).astype(int)
feat['price_above_ma20'] = (feat['ma_20_ratio'] > 1.0).astype(int)

# V5.5信号
feat['v55_trend_up'] = ((feat['ret_5d'] > 0) & (feat['ma_50_ratio'] > 1.0) & (feat['dvol_ma5'] >= 5_000_000)).astype(int)
feat['v55_strong'] = ((feat['ret_10d'] > feat['ret_20d']) & (feat['ret_5d'] > 0) & (feat['ma_50_ratio'] > 1.05)).astype(int)

feat_cols = [
    'ret_1d','ret_3d','ret_5d','ret_10d','ret_20d',
    'ma_5_ratio','ma_10_ratio','ma_20_ratio','ma_50_ratio',
    'vol_5d','vol_10d','vol_20d','rsi_14','rsi_50_pct',
    'vol_ratio_5','vol_ratio_20','vol_5d_norm',
    'price_pos_20','price_pos_50','price_pos_100',
    'macd','macd_sig','macd_hist','atr_pct',
    'rel_ret_5d','rel_ret_10d','ma20_ma50_cross','dvol_ratio','dvol_ma5',
    'price_above_ma50','price_above_ma20',
    'market_ret',
]

valid = feat.dropna(subset=feat_cols + ['ret_f5']).copy()
valid = valid[valid['dvol_ma5'] >= 5_000_000].copy()

dates = sorted(valid['Date'].unique())
dates_set = set(dates)
train_dates = set(dates[:int(len(dates)*0.7)])
val_dates = set(dates[int(len(dates)*0.7):int(len(dates)*0.85)])
test_dates = set(dates[int(len(dates)*0.85):])

train = valid[valid['Date'].isin(train_dates)].copy()
val = valid[valid['Date'].isin(val_dates)].copy()
test = valid[valid['Date'].isin(test_dates)].copy()

# ========= 分位数回归：预测收益的分位，而不是具体数值 =========
# 目标：将每个票在同一天的所有票中排名
# 构建排名标签：同一日，按ret_f5排序，前10%=赢家
def rank_within_day(df):
    df = df.copy()
    df['day_rank'] = df['ret_f5'].rank(pct=True)
    df['is_top10pct'] = (df['day_rank'] >= 0.9).astype(int)
    return df

# rank within each day
for name, df_ in [('train',train),('val',val),('test',test)]:
    df_.loc[:,'day_rank'] = df_.groupby('Date')['ret_f5'].rank(pct=True)
    df_.loc[:,'is_top10pct'] = (df_['day_rank'] >= 0.9).astype(int)

print(f'\n同日前10%赢家占比: 训练{train["is_top10pct"].mean()*100:.1f}%, 验证{val["is_top10pct"].mean()*100:.1f}%, 测试{test["is_top10pct"].mean()*100:.1f}%')

# 训练：预测排名
model = xgb.XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.4,
    reg_alpha=0.3, reg_lambda=1.0,
    scale_pos_weight=9.0,  # 前10% vs 后90%
    eval_metric='auc',
    early_stopping_rounds=80,
    random_state=42, n_jobs=-1
)

model.fit(
    train[feat_cols].values, train['is_top10pct'].values,
    eval_set=[(train[feat_cols].values, train['is_top10pct'].values),
              (val[feat_cols].values, val['is_top10pct'].values)],
    verbose=200
)

# ========= 评估 =========
from sklearn.metrics import roc_auc_score

for name, df, X in [('训练', train, train[feat_cols].values),
                     ('验证', val, val[feat_cols].values),
                     ('测试', test, test[feat_cols].values)]:
    prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(df['is_top10pct'], prob)
    print(f'{name}: AUC={auc:.4f}')

# ========= 回测：ML仓位分配 =========
print('\n=== 蓝盾V4 回测（V5.5规则+ML仓位分配）===')
test_df = test.copy()
test_df['prob'] = model.predict_proba(test[feat_cols].values)[:, 1]

# 策略：每一天只买V5.5信号票，但ML评分高的多买
all_trades = []
te_dates_clean = sorted(test_df['Date'].dropna().unique())
for d in te_dates_clean:
    day = test_df[test_df['Date'] == d]
    if len(day) < 10: continue
    
    # V5.5核心条件：趋势向上
    v55 = day[(day['ret_5d'] > 0) & (day['ma_50_ratio'] > 1.0)].copy()
    if len(v55) < 3: continue
    
    # 按ML评分分三层
    v55['weight'] = pd.qcut(v55['prob'], 3, labels=[0.5, 1.0, 1.5]).astype(float)
    # 最多选12只
    v55 = v55.nlargest(min(12, len(v55)), 'prob')
    
    # 等权加权
    total_weight = v55['weight'].sum()
    v55['wgt'] = v55['weight'] / total_weight
    
    avg_ret = (v55['ret_f5'] * v55['wgt']).sum()
    all_trades.append(avg_ret)

arr = np.array(all_trades)
vm = (arr > 0).mean()
geo = np.exp(np.log(1 + arr).mean()) - 1
ann = geo * 252 / 5
std = arr.std() * np.sqrt(252/5)
shp = ann / max(std, 0.001)
cum = (1 + pd.Series(arr)).cumprod()
dd = (cum / cum.cummax() - 1).min()

print(f'ML仓位分配: 年化={ann*100:.1f}%, 夏普={shp:.2f}, 回撤={dd*100:.1f}%, 方向率={vm*100:.1f}%')

# ========= 对比：V5.5纯规则（每日Top10等权，只用V5.5条件） =========
print('\n--- V5.5纯规则对比 ---')
v55_trades = []
for d in te_dates_clean:
    day = test_df[test_df['Date'] == d]
    if len(day) < 10: continue
    v55 = day[(day['ret_5d'] > 0) & (day['ma_50_ratio'] > 1.0)]
    if len(v55) < 3: continue
    picks = v55.nlargest(10, 'ret_5d')  # V5.5按涨幅选
    v55_trades.append(picks['ret_f5'].mean())

v55_arr = np.array(v55_trades)
v55_vm = (v55_arr > 0).mean()
v55_geo = np.exp(np.log(1 + v55_arr).mean()) - 1
v55_ann = v55_geo * 252 / 5
v55_std = v55_arr.std() * np.sqrt(252/5)
v55_shp = v55_ann / max(v55_std, 0.001)

print(f'V5.5: 年化={v55_ann*100:.1f}%, 夏普={v55_shp:.2f}, 方向率={v55_vm*100:.1f}%')
print(f'V4增量: 年化+{(ann-v55_ann)*100:.1f}%, 夏普+{shp-v55_shp:.2f}')

# 保存
booster = model.get_booster()
booster.save_model('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v4.model')
meta = {
    'model': 'blueshield_v4',
    'strategy': 'v55_rules + ml_weight_allocation',
    'features': feat_cols,
    'n_features': len(feat_cols),
    'backtest': {
        'ml_weighted': {'annual_return': float(ann), 'sharpe': float(shp), 'max_drawdown': float(dd)},
        'v55_pure': {'annual_return': float(v55_ann), 'sharpe': float(v55_shp)},
    },
    'date': '2026-06-11'
}
json.dump(meta, open('/home/hermes/.hermes/openclaw-project/data/models/blueshield_v4_meta.json', 'w'), indent=2)
print(f'\n完成: blueshield_v4')
