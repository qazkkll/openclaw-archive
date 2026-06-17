# -*- coding: utf-8 -*-
"""
V8.1 — 绿箭重新训练
训练期: 2016-10-18 ~ 2024-12-31 (9年+)
验证期: 2025-01-01 ~ 2026-06-10 (1.5年)
模型: XGBoost (同V7.5架构, 51特征)

对比V7.5:
  V7.5训练期: 2020-2024 (5年,仅牛市+疫情)
  V8.1训练期: 2016-2024 (9年,覆盖完整周期)
"""
import os, json, pickle, time, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

print('V8.1 绿箭重新训练'); print('='*60); t0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML=f'{BASE}/ml'; MD=f'{BASE}/data/models'

# 数据
df=pd.read_parquet(f'{ML}/us_ml_feats_v75.parquet')
df['d']=df['date'].astype(str).str[:10]
# 确保d是字符串
if isinstance(df['d'].iloc[0],float):
    # 浮点数日期：convert
    df['d']=df['d'].apply(lambda x: str(int(x)) if not pd.isna(x) else '')
print(f'数据: {len(df):,}行, {df.sym.nunique()}只')

# 特征
report=json.load(open(f'{MD}/us_v7_5_report.json'))
FEATS=report['features']
print(f'特征: {len(FEATS)}个')

# 目标: 未来5日涨>5%
df['target']=(df['fwd_5d_ret']>0.05).astype(int)

# 特征清洗
for f in FEATS:
    if f in df.columns:
        df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).clip(-1e6,1e6)
df=df.replace([np.inf,-np.inf],0)

# ===== 切分: 2016-2024 训练 / 2025-2026 验证 =====
day_min=df['date'].min()
day_max=df['date'].max()
print(f'日期范围: {day_min} ~ {day_max}')
# date是datetime类型，直接比较
train_mask=df['date']<pd.Timestamp('2025-01-01')
val_mask=df['date']>=pd.Timestamp('2025-01-01')
train_df=df[train_mask].copy()
val_df=df[val_mask].copy()

# 用$5过滤名单（保持和V7.5一致）
fl=json.load(open(f'{ML}/us_filtered_syms_v5.json'))
pool=set(fl['syms'])
train_df=train_df[train_df['sym'].isin(pool)].copy()
val_df=val_df[val_df['sym'].isin(pool)].copy()

print(f'\n训练: {len(train_df):,}行, {train_df.sym.nunique()}只')
print(f'验证: {len(val_df):,}行, {val_df.sym.nunique()}只')

# 检查目标分布
trg=train_df['target'].value_counts()
vlg=val_df['target'].value_counts()
print(f'训练目标: 涨>5%={trg.get(1,0)} ({trg.get(1,0)/trg.sum()*100:.1f}%), 否={trg.get(0,0)}')
print(f'验证目标: 涨>5%={vlg.get(1,0)} ({vlg.get(1,0)/vlg.sum()*100:.1f}%), 否={vlg.get(0,0)}')

# ===== 平衡采样（V7.5原版用权重） =====
pos_train=train_df[train_df['target']==1]
neg_train=train_df[train_df['target']==0]
print(f'\n正样本: {len(pos_train):,}, 负样本: {len(neg_train):,}')
sample_size=min(len(pos_train)*2, len(neg_train))
neg_sampled=neg_train.sample(sample_size, random_state=42)
train_balanced=pd.concat([pos_train, neg_sampled]).sample(frac=1, random_state=42)
print(f'平衡后: {len(train_balanced):,}行, 正={len(pos_train)}, 负={len(neg_sampled)}')

# 训练数据
X_train=np.nan_to_num(train_balanced[FEATS].values.astype(np.float32),nan=0)
y_train=train_balanced['target'].values.astype(int)
X_val=np.nan_to_num(val_df[FEATS].values.astype(np.float32),nan=0)
y_val=val_df['target'].values.astype(int)

# ===== 训练XGB =====
print('\n训练XGBoost...')
scale_pos_weight=len(neg_sampled)/max(1,len(pos_train))

params={
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'max_depth':6,
    'learning_rate':0.05,
    'subsample':0.8,
    'colsample_bytree':0.8,
    'scale_pos_weight':scale_pos_weight,
    'min_child_weight':3,
    'gamma':0.1,
    'lambda':1,
    'alpha':0,
    'seed':42,
}

dtrain=xgb.DMatrix(X_train,label=y_train,feature_names=FEATS)
dval=xgb.DMatrix(X_val,label=y_val,feature_names=FEATS)

model=xgb.train(
    params, dtrain,
    num_boost_round=2000,
    evals=[(dtrain,'train'),(dval,'val')],
    early_stopping_rounds=50,
    verbose_eval=50
)

# ===== 校准 =====
print('\n校准...')
from sklearn.isotonic import IsotonicRegression
raw_val=model.predict(dval)
cal=IsotonicRegression(out_of_bounds='clip')
cal.fit(raw_val, y_val.astype(float))
calib_val=cal.predict(raw_val)

# ===== 验证集评分 =====
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
auc=roc_auc_score(y_val,calib_val)
print(f'\n验证集AUC: {auc:.4f}')
print(f'Brier分数: {brier_score_loss(y_val,calib_val):.4f}')

# 分桶检验
val_df=val_df.copy()
val_df['score']=calib_val
val_df['bucket']=pd.qcut(val_df['score'],5,labels=['Q1','Q2','Q3','Q4','Q5'])
bucket_stats=val_df.groupby('bucket',observed=True).agg(
    cnt=('score','count'),mean_score=('score','mean'),mean_target=('target','mean'),
    mean_ret=('fwd_5d_ret','mean'))
print('\n分桶检验:')
print(bucket_stats.to_string())

# ===== 保存 =====
model.save_model(f'{MD}/us_v8_1.json')
pickle.dump(cal,open(f'{MD}/us_v8_1_calibrator.pkl','wb'))

report_out={
    'version':'v8.1',
    'features':FEATS,
    'auc_train':round(float(auc),4),
    'params':params,
    'best_iteration':model.best_iteration,
    'train_period':'2016-10 ~ 2024-12',
    'val_period':'2025-01 ~ 2026-06',
    'train_samples':len(train_balanced),
    'val_samples':len(val_df),
    'bucket_stats':{str(k):{'cnt':int(v['cnt']),'mean_score':round(float(v['mean_score']),4),
                          'mean_target':round(float(v['mean_target']),4),
                          'mean_ret':round(float(v['mean_ret']),4)} for k,v in bucket_stats.iterrows()},
    'feature_importance':[float(v) for v in model.get_score(importance_type='gain').values()],
    'time':time.strftime('%Y-%m-%d %H:%M')
}
json.dump(report_out,open(f'{MD}/us_v8_1_report.json','w'),indent=2)

print(f'\n完成({time.time()-t0:.0f}s)')
print(f'模型: {MD}/us_v8_1.json')
print(f'报告: {MD}/us_v8_1_report.json')
