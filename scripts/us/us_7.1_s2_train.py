#!/usr/bin/env python3
"""
us_7.1_s2_train.py — 绿箭V7.1训练
基于V3特征（排除<5usd + 补充大盘股后），纯技术指标训练
输出: /home/hermes/.hermes/openclaw-project/data/models/us_xgb_v71.json + 校准器 + 报告
"""
import sys, os, json, pickle, warnings, time, math
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

T0 = time.time()
BASE = '/home/hermes/.hermes/openclaw-archive'
ML_DIR = f'{BASE}/ml'
MODEL_DIR = f'{BASE}/data/models'
VERSION = 'us_xgb_v71'

print('=' * 60)
print(f'{VERSION} — 绿箭V7.1训练（干净池 + 大盘股）')
print('=' * 60)

# === 1. 加载特征 ===
print('\n[1/5] 加载特征...')
df = pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
print(f'  行: {len(df):,}, 股票: {df.sym.nunique()}, 日期: {df.date.min()}~{df.date.max()}')

# 确认大盘股存在
for s in ['NVDA','AAPL','MSFT','GOOGL','AMZN','AVGO','META','TSLA']:
    cnt = (df['sym']==s).sum()
    print(f'  {s:>6}: {cnt}行')

# === 2. 特征选择（纯技术指标，排除标签和基本面列） ===
print('\n[2/5] 特征选择...')
EXCLUDE = ['sym','date','price','volume','vol5',
           'label_pct','label_bucket','label_5d_pct','label_5d_5class',
           'sector','industry','pe_trailing','pe_forward','div_yield','beta',
           'short_ratio','short_pct','fund_price','market_cap',
           'spy_ret5','qqq_ret5','iwm_ret5','sector_etf_ret5','sc','sc_cat']

# 技术指标 + ret系列
FEATURES = [c for c in df.columns 
            if c not in EXCLUDE 
            and not c.startswith('label_') 
            and c not in ['price_next_5','ret_5d']]

# 部分特征可能有object类型，转numeric
for f in FEATURES:
    if df[f].dtype == 'object':
        df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0)
    if f not in FEATURES:
        continue
    # 补NaN
    miss = df[f].isna().sum()
    if miss > 0:
        df[f] = df[f].fillna(0)

print(f'  特征数: {len(FEATURES)}')
print(f'  特征: {FEATURES}')

# 确认都是数值
for f in FEATURES:
    if df[f].dtype not in ['float64','float32','int64','int32']:
        print(f'  ⚠️ {f}: dtype={df[f].dtype}')

# === 3. 生成标签 ===
print('\n[3/5] 生成标签...')
df = df.sort_values(['sym','date']).reset_index(drop=True)
df['price_next_5'] = df.groupby('sym')['price'].shift(-5)
df['ret_5d'] = df['price_next_5'] / df['price'] - 1
df['label'] = (df['ret_5d'] > 0.05).astype(int)

valid = df['label'].notna()
print(f'  有效标签: {valid.sum():,}/{len(df)} ({valid.sum()/len(df)*100:.1f}%)')
print(f'  正样本率: {df[valid]["label"].mean():.2%}')

# === 4. 时间切分训练 ===
print('\n[4/5] 训练...')
# 用V7的方案：2024及以前训练, 2025验证, 2026测试
d_train = df[(df['date'] < '2025-01-01') & df['label'].notna()].dropna(subset=FEATURES)
d_val = df[(df['date'] >= '2025-01-01') & (df['date'] < '2026-01-01') & df['label'].notna()].dropna(subset=FEATURES)
d_test = df[(df['date'] >= '2026-01-01') & df['label'].notna()].dropna(subset=FEATURES)

print(f'  训练: {len(d_train):,}行, 正样本{d_train.label.mean():.2%}')
print(f'  验证: {len(d_val):,}行, 正样本{d_val.label.mean():.2%}')
print(f'  测试: {len(d_test):,}行, 正样本{d_test.label.mean():.2%}')

# 市值分层采样（有market_cap用，没有也可以不用）
# 直接从可用数据训练

X_train = d_train[FEATURES].values.astype(np.float32)
X_train = np.nan_to_num(X_train, nan=0.0)
y_train = d_train['label'].values.astype(np.float32)

X_val = d_val[FEATURES].values.astype(np.float32)
X_val = np.nan_to_num(X_val, nan=0.0)
y_val = d_val['label'].values.astype(np.float32)

X_test = d_test[FEATURES].values.astype(np.float32)
X_test = np.nan_to_num(X_test, nan=0.0)
y_test = d_test['label'].values.astype(np.float32)

# XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURES)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=FEATURES)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURES)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'seed': 42,
}

print('\n  训练中...')
model = xgb.train(
    params, dtrain, num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=30,
    verbose_eval=False,
)

print(f'  最佳迭代: {model.best_iteration}')
print(f'  训练AUC: {model.best_score:.4f}')

# === 5. 评估 ===
print('\n[5/5] 评估...')

# 验证集
pred_val_raw = model.predict(dval)
val_auc = roc_auc_score(y_val, pred_val_raw)
print(f'  验证集AUC(原始): {val_auc:.4f}')

# 测试集
pred_test_raw = model.predict(dtest)
test_auc = roc_auc_score(y_test, pred_test_raw)
print(f'  测试集AUC(原始): {test_auc:.4f}')

# Platt校准
calibrator = LogisticRegression()
calibrator.fit(pred_val_raw.reshape(-1,1), y_val)
pred_val_calib = calibrator.predict_proba(pred_val_raw.reshape(-1,1))[:,1]
pred_test_calib = calibrator.predict_proba(pred_test_raw.reshape(-1,1))[:,1]

val_auc_c = roc_auc_score(y_val, pred_val_calib)
test_auc_c = roc_auc_score(y_test, pred_test_calib)
print(f'  验证AUC(校准后): {val_auc_c:.4f}')
print(f'  测试AUC(校准后): {test_auc_c:.4f}')

# 校准检查
print('\n校准检查:')
for lb in np.arange(0, 0.6, 0.1):
    mask = (pred_test_calib >= lb) & (pred_test_calib < lb+0.1)
    if mask.sum() > 50:
        actual = y_test[mask].mean()
        print(f'  prob {lb:.1f}-{lb+0.1:.1f} (n={mask.sum():,}): pred≈{lb+0.05:.1%} actual={actual:.1%}')

# 特征重要性
imp = model.get_score(importance_type='weight')
print('\n特征重要性 (weight):')
for fn, wgt in sorted(imp.items(), key=lambda x:-x[1])[:15]:
    print(f'  {fn:20s} {wgt:8.0f}')

# 最新日预测
last_date = df[df['label'].notna()]['date'].max()
last_day = df[df['date']==last_date].dropna(subset=FEATURES).copy()
X_last = np.nan_to_num(last_day[FEATURES].values.astype(np.float32), nan=0.0)
dlast = xgb.DMatrix(X_last, feature_names=FEATURES)
pred_last_raw = model.predict(dlast)
pred_last_calib = calibrator.predict_proba(pred_last_raw.reshape(-1,1))[:,1]
last_day = last_day.copy()
last_day['v71_prob'] = pred_last_calib
last_day = last_day.sort_values('v71_prob', ascending=False)

# 持仓评分
print(f'\n持仓评分 (截至{last_date}):')
for code, sym in [('NOK','NOK'),('NVDA','NVDA'),('GNRC','GNRC'),('ON','ON'),('QCOM','QCOM')]:
    row = last_day[last_day['sym']==sym]
    if len(row) > 0:
        r = row.iloc[0]
        rank = last_day['sym'].eq(sym).values.argmax() + 1
        print(f'  {code:>6}  ${r["price"]:>7.2f}  V7.1={r["v71_prob"]:>6.1%}  rank={rank}')

# Top推荐（$5+）
print(f'\nV7.1 Top30 ($5+):')
day5 = last_day[last_day['price']>=5].head(30)
print(f'{"#":>3} {"代码":>7} {"价格":>8} {"V7.1概率":>8}')
for i,(_,r) in enumerate(day5.iterrows(),1):
    print(f'{i:>3} {r["sym"]:>7} ${r["price"]:>7.2f} {r["v71_prob"]:>7.1%}')

# 截面回测（2025+）
print('\n截面回测 (验证集2025+):')
bt_dates = sorted(d_val['date'].unique()) + sorted(d_test['date'].unique())
bt_dates = [d for d in bt_dates if str(d) >= '2025-01-02']

# 全量预测
d_all = df.dropna(subset=FEATURES).copy()
X_all = np.nan_to_num(d_all[FEATURES].values.astype(np.float32), nan=0.0)
dall = xgb.DMatrix(X_all, feature_names=FEATURES)
pred_all_raw = model.predict(dall)
pred_all_calib = calibrator.predict_proba(pred_all_raw.reshape(-1,1))[:,1]
d_all = d_all.copy()
d_all['v71_prob'] = pred_all_calib

for tn in [5, 10, 15, 20]:
    returns = []
    for d in bt_dates:
        day = d_all[d_all['date']==d].dropna(subset=['v71_prob','ret_5d'])
        if len(day) < 30:
            continue
        picks = day.nlargest(tn, 'v71_prob')
        r = picks['ret_5d'].values
        r = np.clip(r, -0.50, 1.0)
        returns.append(r.mean())
    
    arr = np.array(returns)
    avg = arr.mean()
    std = arr.std()
    sharpe = avg/std*math.sqrt(252) if std>1e-10 else 0
    win = (arr>0).mean()
    hit5 = (arr>0.05).mean()
    cum = np.prod(1+arr)
    peak = np.maximum.accumulate(np.cumprod(1+arr))
    dd = (np.cumprod(1+arr)-peak)/peak
    ann = (cum**(252/len(arr))-1)*100
    print(f'  d{tn:>2}: 累计{ann:+.1f}% 夏普{sharpe:.2f} 回撤{dd.min()*100:.1f}% 胜率{win:.1%} >5%={hit5:.1%}')

# 保存模型
print('\n保存模型...')
model.save_model(f'{MODEL_DIR}/{VERSION}.json')
pickle.dump(calibrator, open(f'{MODEL_DIR}/{VERSION}_calibrator.pkl', 'wb'))

report = {
    'version': VERSION,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'val_auc': round(val_auc_c, 4),
    'test_auc': round(test_auc_c, 4),
    'features': FEATURES,
    'n_features': len(FEATURES),
    'n_train': len(d_train),
    'n_val': len(d_val),
    'n_test': len(d_test),
    'pos_rate_train': round(d_train.label.mean(), 4),
    'pos_rate_val': round(d_val.label.mean(), 4),
    'pos_rate_test': round(d_test.label.mean(), 4),
    'data_dates': f'{df.date.min()}~{df.date.max()}',
    'n_stocks': int(df.sym.nunique()),
    'params': params,
    'feature_importance': {fn: int(wgt) for fn, wgt in sorted(imp.items(), key=lambda x:-x[1])},
}
json.dump(report, open(f'{MODEL_DIR}/{VERSION}_report.json', 'w'), indent=2)
print(f'  → {MODEL_DIR}/{VERSION}.json')
print(f'  → {MODEL_DIR}/{VERSION}_calibrator.pkl')
print(f'  → {MODEL_DIR}/{VERSION}_report.json')

print(f'\n总耗时: {time.time()-T0:.0f}s')
print('=' * 60)
