#!/usr/bin/env python3
"""快速诊断：截面特征为什么让IC变负"""
import pandas as pd, numpy as np, xgboost as xgb, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.expanduser('~/.hermes/openclaw-archive'))

hist = pd.read_parquet('data/cn/features_v2.parquet')
hist['date'] = pd.to_datetime(hist['date'])
hist['date_int'] = hist['date'].dt.strftime('%Y%m%d').astype(int)

# V1.1模型
m11 = xgb.Booster()
m11.load_model('models/cn/cn_alpha_v1.1.json')
v11_feats = m11.feature_names

# 用一个fold测试
train = hist[(hist['date_int'] >= 20160101) & (hist['date_int'] <= 20201231)].copy()
test = hist[(hist['date_int'] >= 20210101) & (hist['date_int'] <= 20220630)].copy()

combined = pd.concat([train, test]).sort_values(['sym', 'date_int'])
combined['fwd_ret'] = combined.groupby('sym')['close'].transform(lambda x: x.shift(-10) / x - 1)

train = combined[combined['date_int'].between(20160101, 20201231)].dropna(subset=['fwd_ret'])
test = combined[combined['date_int'].between(20210101, 20220630)].dropna(subset=['fwd_ret'])

# 确保v1.1特征存在
for f in v11_feats:
    if f not in train.columns:
        train[f] = 0
        test[f] = 0

# 测试1: V1.1原始特征
X_tr = train[v11_feats].fillna(0)
X_te = test[v11_feats].fillna(0)
dtrain = xgb.DMatrix(X_tr, label=train['fwd_ret'])
dtest = xgb.DMatrix(X_te, label=test['fwd_ret'])

params = {'max_depth': 6, 'eta': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8,
          'min_child_weight': 100, 'objective': 'reg:squarederror', 'tree_method': 'hist'}
m1 = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)
pred1 = m1.predict(dtest)
ic1 = test.assign(pred=pred1).groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()

print(f"V1.1原始特征: IC={ic1:.4f}")

# 测试2: V1.1 + 截面排名
for src, dst in [('rev_20d', 'rev_20d_rank'), ('lg_net_20', 'lg_net_20_csrank'),
                  ('total_net_20', 'total_net_20_csrank'), ('vol_r', 'vol_r_csrank')]:
    if src in combined.columns:
        combined[dst] = combined.groupby('date_int')[src].rank(pct=True)

for src in ['rev_20d', 'lg_net_20', 'vol_r']:
    if src in combined.columns:
        zname = f'{src}_zscore'
        grp = combined.groupby('date_int')[src]
        combined[zname] = (combined[src] - grp.transform('mean')) / grp.transform('std').clip(lower=1e-8)

# 市场特征
market_daily = combined.groupby('date_int')['close'].mean().reset_index()
market_daily.columns = ['date_int', 'mkt_avg']
for w in [20, 60, 120]:
    market_daily[f'mkt_ma{w}'] = market_daily['mkt_avg'].rolling(w).mean()
market_daily['mkt_ma60_above_120'] = (market_daily['mkt_ma60'] > market_daily['mkt_ma120']).astype(float)
market_daily['mkt_ret_20d'] = market_daily['mkt_avg'].pct_change(20)
market_daily['mkt_momentum'] = (market_daily['mkt_ret_20d'] > 0).astype(float)

combined = combined.merge(market_daily[['date_int', 'mkt_ma60_above_120', 'mkt_momentum']], on='date_int', how='left')

new_feats = ['rev_20d_rank', 'lg_net_20_csrank', 'total_net_20_csrank', 'vol_r_csrank',
             'rev_20d_zscore', 'lg_net_20_zscore', 'vol_r_zscore',
             'mkt_ma60_above_120', 'mkt_momentum']

train2 = combined[combined['date_int'].between(20160101, 20201231)].dropna(subset=['fwd_ret'])
test2 = combined[combined['date_int'].between(20210101, 20220630)].dropna(subset=['fwd_ret'])

v12_feats = v11_feats + new_feats
for f in new_feats:
    if f not in train2.columns:
        train2[f] = 0
        test2[f] = 0

X_tr2 = train2[v12_feats].fillna(0)
X_te2 = test2[v12_feats].fillna(0)
dtrain2 = xgb.DMatrix(X_tr2, label=train2['fwd_ret'])
dtest2 = xgb.DMatrix(X_te2, label=test2['fwd_ret'])

m2 = xgb.train(params, dtrain2, num_boost_round=500, verbose_eval=False)
pred2 = m2.predict(dtest2)
ic2 = test2.assign(pred=pred2).groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()

print(f"V1.1+截面/市场: IC={ic2:.4f}")

# 测试3: 只加市场特征
v12_feats_mkt = v11_feats + ['mkt_ma60_above_120', 'mkt_momentum']
for f in ['mkt_ma60_above_120', 'mkt_momentum']:
    if f not in train2.columns:
        train2[f] = 0
        test2[f] = 0

X_tr3 = train2[v12_feats_mkt].fillna(0)
X_te3 = test2[v12_feats_mkt].fillna(0)
dtrain3 = xgb.DMatrix(X_tr3, label=train2['fwd_ret'])
dtest3 = xgb.DMatrix(X_te3, label=test2['fwd_ret'])

m3 = xgb.train(params, dtrain3, num_boost_round=500, verbose_eval=False)
pred3 = m3.predict(dtest3)
ic3 = test2.assign(pred=pred3).groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()

print(f"V1.1+仅市场: IC={ic3:.4f}")

# 测试4: V1.1 + 截面排名(不加zscore)
v12_feats_rank = v11_feats + ['rev_20d_rank', 'lg_net_20_csrank', 'total_net_20_csrank', 'vol_r_csrank']
for f in ['rev_20d_rank', 'lg_net_20_csrank', 'total_net_20_csrank', 'vol_r_csrank']:
    if f not in train2.columns:
        train2[f] = 0
        test2[f] = 0

X_tr4 = train2[v12_feats_rank].fillna(0)
X_te4 = test2[v12_feats_rank].fillna(0)
dtrain4 = xgb.DMatrix(X_tr4, label=train2['fwd_ret'])
dtest4 = xgb.DMatrix(X_te4, label=test2['fwd_ret'])

m4 = xgb.train(params, dtrain4, num_boost_round=500, verbose_eval=False)
pred4 = m4.predict(dtest4)
ic4 = test2.assign(pred=pred4).groupby('date_int').apply(lambda x: x['pred'].corr(x['fwd_ret'])).mean()

print(f"V1.1+仅排名: IC={ic4:.4f}")

print(f"\n结论:")
print(f"  V1.1原始:  {ic1:.4f}")
print(f"  +截面/市场: {ic2:.4f} (diff={ic2-ic1:+.4f})")
print(f"  +仅市场:   {ic3:.4f} (diff={ic3-ic1:+.4f})")
print(f"  +仅排名:   {ic4:.4f} (diff={ic4-ic1:+.4f})")
