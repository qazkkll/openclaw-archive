#!/usr/bin/env python3
"""
us_ml_09_train_v7_quick.py — V7快速训练版（不拉基本面）
- 基于V5的27个技术面特征
- 添加价格分段特征（<10/10-20/20-50/>50 proxy for size）
- 样本加权：价格越高权重越大（正向大盘偏好）
- 大盘子模型单独训练（price>20USD）
- XGBoost + Platt校准
"""
import sys, json, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

INPUT = '/home/hermes/.hermes/openclaw-project/scripts/system/us_ml_feats_v5.parquet'
MODEL_DIR = '/home/hermes/.hermes/openclaw-project/data/models/us'

print("[V7.quick] 1. 加载数据...", flush=True)
df = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])

# 已有特征
excl = {'ticker','date','label','fwd_5d_ret'}
v5_feats = [c for c in df.columns if c not in excl]
print(f"  V5特征: {len(v5_feats)}个", flush=True)

# 从K线数据加价格分段特征
# 用close过去5日均价做价格分段
print("  计算价格分段...", flush=True)
# 按ticker取最后一天价格做proxy
g = df.groupby('ticker').tail(1)
price_proxy = dict(zip(g['ticker'], g['ret_20d']))  # fallback: just use ret_20d

# 实际上用fwd_5d_ret不能直接用prev close。从原始parquet加载K线
hist = pd.read_parquet('/home/hermes/.hermes/openclaw-project/scripts/system/us_hist_yf_5y.parquet')
hist = hist.sort_values(['ticker', 'date'])

# 取每个ticker最后一条的close
last_prices = hist.groupby('ticker').tail(1)[['ticker','close']].dropna()
price_map = dict(zip(last_prices['ticker'], last_prices['close']))
del hist

# 添加价格分段特征到df
df['price'] = df['ticker'].map(price_map)

# 价格分段
def price_low(p):
    if p is None or np.isnan(p) or p <= 0:
        return 0
    return 1 if p < 10 else 0

def price_midlow(p):
    if p is None or np.isnan(p) or p <= 0:
        return 0
    return 1 if 10 <= p < 20 else 0

def price_mid(p):
    if p is None or np.isnan(p) or p <= 0:
        return 0
    return 1 if 20 <= p < 50 else 0

def price_high(p):
    if p is None or np.isnan(p) or p <= 0:
        return 0
    return 1 if p >= 50 else 0

df['price_lt10'] = df['price'].apply(price_low)
df['price_10to20'] = df['price'].apply(price_midlow)
df['price_20to50'] = df['price'].apply(price_mid)
df['price_ge50'] = df['price'].apply(price_high)
df['log_price'] = np.log1p(df['price'].fillna(15))  # 默认15USD

new_feats = ['price_lt10', 'price_10to20', 'price_20to50', 'price_ge50', 'log_price']
all_feats = v5_feats + new_feats

# 样本权重：价格越高权重越大（2-4倍）
w = np.ones(len(df))
mask_low = df['price_lt10'] == 1
mask_mlow = df['price_10to20'] == 1
mask_mid = df['price_20to50'] == 1
mask_high = df['price_ge50'] == 1
w[mask_low] = 1.0
w[mask_mlow] = 2.0
w[mask_mid] = 4.0
w[mask_high] = 8.0

y = (df['fwd_5d_ret'] > 0.05).astype(int).values
X = np.nan_to_num(df[all_feats].values.astype(np.float32), nan=0.0)

# 时序切分
n = len(X)
te, ce = int(n*0.75), int(n*0.85)
Xt, Xc, Xv = X[:te], X[te:ce], X[ce:]
yt, yc, yv = y[:te], y[te:ce], y[ce:]
wt, wc, wv = w[:te], w[te:ce], w[ce:]
print(f"  训练:{len(Xt):,} 校准:{len(Xc):,} 测试:{len(Xv):,}", flush=True)
print(f"  正样本: {yt.mean()*100:.1f}% / {yc.mean()*100:.1f}% / {yv.mean()*100:.1f}%", flush=True)

del X, y, w, df

print("[V7.quick] 2. 主模型训练 (XGBoost + 加权)...", flush=True)
import xgboost as xgb

dt = xgb.DMatrix(Xt, yt, weight=wt)
dc = xgb.DMatrix(Xc, yc)
dv = xgb.DMatrix(Xv, yv)

params = {
    'objective':'binary:logistic','eval_metric':'logloss',
    'tree_method':'hist','device':'cuda','max_depth':6,
    'learning_rate':0.05,'subsample':0.8,'colsample_bytree':0.8,
    'random_state':42,'scale_pos_weight':5.0
}
t0 = time.time()
model = xgb.train(params, dt, num_boost_round=400,
                  evals=[(dt,'train'),(dc,'val')],
                  early_stopping_rounds=15, verbose_eval=100)
print(f"  耗时: {time.time()-t0:.0f}s", flush=True)

# Platt校准
rc = model.predict(dc, output_margin=True)
rv = model.predict(dv, output_margin=True)
pm = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
pm.fit(rc.reshape(-1,1), yc)
yp = expit(pm.coef_[0][0]*rv + pm.intercept_[0])
auc_main = roc_auc_score(yv, yp)
bias_main = (yp.mean() - yv.mean())*100
nu_main = len(set(np.round(yp, 4)))
print(f"  主模型: AUC={auc_main:.4f} 偏{bias_main:+.2f}% 离散={nu_main}", flush=True)

# 大盘子模型 (price>20 only)
print("[V7.quick] 3. 大盘子模型 (price>=20)...", flush=True)
df2 = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
df2['price'] = df2['ticker'].map(price_map)
for col in new_feats:
    if col == 'log_price':
        df2[col] = np.log1p(df2['price'].fillna(15))
    elif col == 'price_lt10':
        df2[col] = df2['price'].apply(price_low)
    elif col == 'price_10to20':
        df2[col] = df2['price'].apply(price_midlow)
    elif col == 'price_20to50':
        df2[col] = df2['price'].apply(price_mid)
    elif col == 'price_ge50':
        df2[col] = df2['price'].apply(price_high)
high_mask = (df2['price'].fillna(0) >= 20).values

X_high = np.nan_to_num(df2[all_feats].values.astype(np.float32), nan=0.0)
y_high = (df2['fwd_5d_ret'] > 0.05).astype(int).values

X_ht = X_high[:te][high_mask[:te]]
y_ht = y_high[:te][high_mask[:te]]
X_hc = X_high[te:ce][high_mask[te:ce]]
y_hc = y_high[te:ce][high_mask[te:ce]]
X_hv = X_high[ce:][high_mask[ce:]]
y_hv = y_high[ce:][high_mask[ce:]]
print(f"  大盘数据: 训练{len(X_ht):,} 校准{len(X_hc):,} 测试{len(X_hv):,}", flush=True)

large_model = None
if len(X_ht) > 1000 and sum(y_ht) > 50:
    dht = xgb.DMatrix(X_ht, y_ht)
    dhc = xgb.DMatrix(X_hc, y_hc)
    t0 = time.time()
    large_model = xgb.train(params, dht, num_boost_round=300,
                            evals=[(dht,'train'),(dhc,'val')],
                            early_stopping_rounds=10, verbose_eval=False)
    print(f"  大盘训练: {time.time()-t0:.0f}s", flush=True)
    
    rc_l = large_model.predict(dhc, output_margin=True)
    rv_l = large_model.predict(xgb.DMatrix(X_hv), output_margin=True)
    pm_l = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    pm_l.fit(rc_l.reshape(-1,1), y_hc)
    yp_l = expit(pm_l.coef_[0][0]*rv_l + pm_l.intercept_[0])
    
    if len(set(y_hv)) > 1:
        auc_l = roc_auc_score(y_hv, yp_l)
        print(f"  大盘AUC: {auc_l:.4f}", flush=True)
    else:
        print(f"  大盘测试集无正样本", flush=True)
    
    large_calib = {'slope':float(pm_l.coef_[0][0]), 'intercept':float(pm_l.intercept_[0])}
    large_model.save_model(f'{MODEL_DIR}/greenshaft_v7_large.json')
    json.dump(large_calib, open(f'{MODEL_DIR}/greenshaft_v7_large_calib.json','w'), indent=2)
else:
    print(f"  大盘数据不足, 跳过子模型", flush=True)

# 保存主模型
model.save_model(f'{MODEL_DIR}/greenshaft_v7.json')
calib = {'method':'platt','framework':'xgboost',
         'slope':float(pm.coef_[0][0]),'intercept':float(pm.intercept_[0]),
         'auc':round(float(auc_main),4),'prob_unique':nu_main,
         'bias_pct':round(bias_main,2)}
json.dump(calib, open(f'{MODEL_DIR}/greenshaft_v7_calib.json','w'), indent=2)
print(f"  模型已保存", flush=True)

# 特征重要性
print("[V7.quick] 4. 特征重要性...", flush=True)
imp = model.get_score(importance_type='gain')
sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
print(f"  Top 15:")
for f, s in sorted_imp[:15]:
    idx = int(f[1:])
    if idx < len(all_feats):
        print(f"    {all_feats[idx]:<20} {s:.1f}")

# 生成推荐
print("[V7.quick] 5. 生成推荐...", flush=True)
df3 = pd.read_parquet(INPUT).dropna(subset=['fwd_5d_ret'])
df3['price'] = df3['ticker'].map(price_map)
for col in new_feats:
    if col == 'log_price':
        df3[col] = np.log1p(df3['price'].fillna(15))
    elif col == 'price_lt10':
        df3[col] = df3['price'].apply(price_low)
    elif col == 'price_10to20':
        df3[col] = df3['price'].apply(price_midlow)
    elif col == 'price_20to50':
        df3[col] = df3['price'].apply(price_mid)
    elif col == 'price_ge50':
        df3[col] = df3['price'].apply(price_high)

g = df3.groupby('ticker').last().reset_index()
Xl = np.nan_to_num(g[all_feats].values.astype(np.float32), nan=0.0)

model2 = xgb.Booster()
model2.load_model(f'{MODEL_DIR}/greenshaft_v7.json')
raw = model2.predict(xgb.DMatrix(Xl), output_margin=True)
probs = expit(calib['slope']*raw + calib['intercept'])

if large_model is not None:
    large_model2 = xgb.Booster()
    large_model2.load_model(f'{MODEL_DIR}/greenshaft_v7_large.json')
    large_idx = g['price'].fillna(0) >= 20
    if large_idx.any():
        raw_l = large_model2.predict(xgb.DMatrix(Xl[large_idx]), output_margin=True)
        probs[large_idx] = expit(large_calib['slope']*raw_l + large_calib['intercept'])

# Price info
price_df = g[['ticker']].copy()
price_df['price'] = g['price'].fillna(0)
price_df['prob'] = np.round(probs, 4)
price_df = price_df.sort_values('prob', ascending=False)

n35 = (price_df['prob']>0.35).sum()
n30 = (price_df['prob']>0.30).sum()
n25 = (price_df['prob']>0.25).sum()
print(f"  >35:{n35} >30:{n30} >25:{n25} 离散:{len(set(price_df['prob']))}", flush=True)
print(f"  {'#':>3} {'代码':<8} {'概率':>8} {'价格':>8} {'线':>6}", flush=True)

big_caps = price_df[price_df['price'] >= 20]
print(f"\n  大盘股(price>=20)推荐:")
for i,(_,r) in enumerate(big_caps.head(15).iterrows()):
    tag = "BUY" if r['prob']>0.35 else "WATCH" if r['prob']>0.30 else ""
    print(f"  {i+1:>3} {r['ticker']:<8} {r['prob']:.4f}  ${r['price']:<6.1f} {tag}", flush=True)

pred = {'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
        'model':'greenshaft_v7 (XGBoost + 价格分段 + 加权 + 大盘子模型)',
        'auc':calib['auc'],
        'calib_slope':calib['slope'], 'calib_intercept':calib['intercept'],
        'n_above_35':int(n35),'n_above_30':int(n30),
        'top_50':[{'ticker':r['ticker'],'prob':r['prob'],'price':r['price']} for _,r in price_df.head(50).iterrows()]}
json.dump(pred, open(f'{MODEL_DIR}/greenshaft_v7_prediction.json','w'), indent=2)
print(f"完成! 总耗时{time.time()-T0:.1f}分钟", flush=True)
