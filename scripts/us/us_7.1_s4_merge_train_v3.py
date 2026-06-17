#!/usr/bin/env python3
"""
us_7.1_s4_merge_train_v3.py — V7.1 v3训练
合并yfinance强信号(机构/内幕/分析师/EPS/做空)到V3特征 + 训练

输出: models/us_xgb_v71_v3.* (模型+校准器+报告)
"""
import sys, os, json, pickle, warnings, time, math
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

T0=time.time()
BASE='/home/hermes/.hermes/openclaw-archive'; ML_DIR=f'{BASE}/ml'; MD=f'{BASE}/data/models'; VER='us_xgb_v71_v3'
SIGNAL_FILE=f'{ML_DIR}/us_signals_v71.json'

print('='*60)
print(f'{VER} — 合并信号+训练')

# ===== 1. 加载特征 & 信号 =====
print('\n[1/5] 加载特征+信号...')
df=pd.read_parquet(f'{ML_DIR}/us_ml_feats_v3_dated.parquet')
print(f'  特征: {len(df):,}行, {df.sym.nunique()}只')

signals=json.load(open(SIGNAL_FILE,'r'))
print(f'  信号: {len(signals)}只')
has_data=sum(1 for v in signals.values() if len(v)>1)
print(f'  有信号数据: {has_data}只')

# ===== 2. 合并信号到特征 =====
print('\n[2/5] 合并信号到特征...')

# 信号字段映射
SIGNAL_MAP={
    'inst_pct_change':'inst_pct_change','inst_pct_held':'inst_pct_held','inst_avg_change':'inst_avg_change',
    'insider_net':'insider_net','insider_txns':'insider_txns',
    'shares_out':'shares_out','shares_chg_3m':'shares_chg_3m',
    'pt_avg_chg':'pt_avg_chg','pt_net':'pt_net',
    'upgrades':'upgrades','downgrades':'downgrades',
    'eps_up_7d':'eps_up_7d','eps_down_7d':'eps_down_7d',
    'short_ratio':'signal_short_ratio','short_pct':'signal_short_pct',
    'iv_call':'iv_call','iv_put':'iv_put','iv_skew':'iv_skew',
}

# 每个sym: 信号数据构建一个Series
signal_df=pd.DataFrame.from_dict(signals,orient='index')
print(f'  信号DataFrame: {signal_df.shape}')

# 合并到特征
for src,tgt in SIGNAL_MAP.items():
    if src in signal_df.columns:
        df[tgt]=df['sym'].map(signal_df[src])
        miss=df[tgt].isna().sum()
        df[tgt]=df[tgt].fillna(0)  # 无信号=0
    else:
        print(f'  ⚠️ 信号字段{src}不存在, 跳过')

# 信号基础宽度
signal_cols=[tgt for src,tgt in SIGNAL_MAP.items() if src in signal_df.columns]
print(f'  信号列({len(signal_cols)}个): {signal_cols}')

# 覆盖率
for c in signal_cols:
    nonzero=(df[c]!=0).sum()
    print(f'  {c:20s}: {nonzero}/{len(df)} ({nonzero/len(df)*100:.1f}%)')

# ===== 3. 特征组合 =====
print('\n[3/5] 特征配置...')
TECH=['ma5','ma10','ma20','ma60','rsi14','p52','ret1','ret5','ret20','ret60',
      'macd','macd_signal','macd_hist','vol20','vol_ratio','ma_bias20','trend_accel']
FUND=['market_cap','pe_trailing','pe_forward','beta','div_yield']
FEATS=TECH+FUND+signal_cols
FEATS=[f for f in FEATS if f in df.columns]
print(f'  总特征: {len(FEATS)}')
print(f'  技术: {len(TECH)}, 基础基本面: {len(FUND)}, 信号: {len(signal_cols)}')

for f in FEATS:
    if df[f].dtype=='object': df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0)
    df[f]=pd.to_numeric(df[f],errors='coerce').fillna(0).replace([np.inf,-np.inf],0)

# ===== 4. 标签 =====
print('\n[4/5] 生成标签...')
df=df.sort_values(['sym','date']).reset_index(drop=True)
df['price_next_5']=df.groupby('sym')['price'].shift(-5)
df['ret_5d']=df['price_next_5']/df['price']-1
df['label']=(df['ret_5d']>0.05).astype(int)

v=df['label'].notna()
print(f'  有效: {v.sum():,}行, 正样本{df[v]["label"].mean():.2%}')

# 时间切分
tr=df[(df['date']<'2025-01-01')&df['label'].notna()].dropna(subset=FEATS+['label'])
va=df[(df['date']>='2025-01-01')&(df['date']<'2026-01-01')&df['label'].notna()].dropna(subset=FEATS+['label'])
te=df[(df['date']>='2026-01-01')&df['label'].notna()].dropna(subset=FEATS+['label'])

print(f'  训练: {len(tr):,}行 {tr.label.mean():.2%}正')
print(f'  验证: {len(va):,}行 {va.label.mean():.2%}正')
print(f'  测试: {len(te):,}行 {te.label.mean():.2%}正')

# ===== 5. 训练 =====
print('\n[5/5] XGBoost训练(1000轮深树)...')
spw=max(1,(1-tr.label.mean())/max(tr.label.mean(),0.01))
print(f'  scale_pos_weight={spw:.1f}')

Xtr=np.nan_to_num(tr[FEATS].values.astype(np.float32),nan=0)
ytr=tr.label.values.astype(float)
Xva=np.nan_to_num(va[FEATS].values.astype(np.float32),nan=0)
yva=va.label.values.astype(float)
Xte=np.nan_to_num(te[FEATS].values.astype(np.float32),nan=0)
yte=te.label.values.astype(float)

dt=xgb.DMatrix(Xtr,label=ytr,feature_names=FEATS)
dv=xgb.DMatrix(Xva,label=yva,feature_names=FEATS)

model=xgb.train({'objective':'binary:logistic','eval_metric':'auc',
    'max_depth':8,'eta':0.03,'subsample':0.8,'colsample_bytree':0.7,
    'min_child_weight':3,'scale_pos_weight':spw,'seed':42, 'device':'cuda'},
    dt,1000,evals=[(dt,'tr'),(dv,'va')],early_stopping_rounds=200,verbose_eval=0)

pv=model.predict(dv); pt=model.predict(xgb.DMatrix(Xte,label=yte,feature_names=FEATS))
va_auc=roc_auc_score(yva,pv); te_auc=roc_auc_score(yte,pt)
print(f'  最佳迭代: {model.best_iteration} 训练AUC: {model.best_score:.4f}')
print(f'  验证AUC: {va_auc:.4f}  测试AUC: {te_auc:.4f}')

# Isotonic校准
cal=IsotonicRegression(out_of_bounds='clip')
cal.fit(pv,yva); pvc=cal.transform(pv); ptc=cal.transform(pt)
va_c=roc_auc_score(yva,pvc); te_c=roc_auc_score(yte,ptc)
print(f'  验证AUC(iso): {va_c:.4f}  测试AUC(iso): {te_c:.4f}')

# 校准矩阵
print('\n校准检查:')
for lb in [x/10 for x in range(6)]:
    msk=(ptc>=lb)&(ptc<lb+0.1)
    if msk.sum()>50:
        actual=yte[msk].mean()
        print(f'  {lb:.0%}-{lb+0.1:.0%} (n={msk.sum():,}): pred≈{lb+0.05:.0%} actual={actual:.1%}')

# 特征重要性
imp=model.get_score('weight')
print('\n特征重要性:')
for fn,wgt in sorted(imp.items(),key=lambda x:-x[1])[:20]:
    print(f'  {fn:20s} {wgt:8,}')

# ===== 核心验证 =====
print('\n'+'='*40)
print('核心验证')
print('='*40)

da=df.dropna(subset=FEATS+['label']).copy()
Xa=np.nan_to_num(da[FEATS].values.astype(np.float32),nan=0)
da['prob']=cal.transform(model.predict(xgb.DMatrix(Xa,feature_names=FEATS)))

# 验证A: 概率阈值命中率分析
print('\n[验证A] 预测概率>阈值的实际表现:')
for th in [0.5,0.45,0.4,0.35,0.3,0.25,0.2]:
    sub=da[da['prob']>th]
    if len(sub)>20:
        r=sub['ret_5d'].clip(-0.5,0.5)
        hit=sub['label'].mean()
        loss=(sub['ret_5d']<-0.01).mean()
        loss3=(sub['ret_5d']<-0.03).mean()
        print(f'  >{th:.0%} (n={len(sub):,}): 命中={hit:.1%} 均值={r.mean():+.2%} 亏损率={loss:.1%} 亏>3%={loss3:.1%}')

# 验证B: 截面选TopN
print('\n[验证B] 截面选TopN (2025+):')
btd=sorted(da['date'].unique())
btd=[d for d in btd if str(d)>='2025-01-02']

for tn in [5,10,15,20]:
    ho=[];lo=[];ro=[];nd=0
    for d in btd:
        dy=da[da['date']==d]
        if len(dy)<30: continue
        pk=dy.nlargest(tn,'prob')
        r=pk['ret_5d'].clip(-0.5,1.0).values
        ho.append((r>0.05).mean())
        lo.append((r<0).mean())
        ro.append(r.mean())
        nd+=1
    print(f'  d{tn:>2} (n={nd:,}): 命中={np.mean(ho):.1%} 亏损率={np.mean(lo):.1%} 均值={np.mean(ro):+.2%}')

# 持仓评分
last_date=da['date'].max()
ld=da[da['date']==last_date].sort_values('prob',ascending=False)
print(f'\n持仓评分 ({last_date}):')
for code,sym in [('NVDA','NVDA'),('NOK','NOK'),('GNRC','GNRC'),('ON','ON'),('QCOM','QCOM')]:
    row=ld[ld['sym']==sym]
    if len(row)>0:
        r=row.iloc[0]; rank=ld['sym'].eq(sym).values.argmax()+1
        print(f'  {code:>6} ${r["price"]:>7.2f} V7.3={r["prob"]:>6.1%} rank={rank}')
    else:
        print(f'  {code:>6}: 不在最新日')

print('\nTop20 ($5+):')
for i,(_,r) in enumerate(ld[ld['price']>=5].head(20).iterrows(),1):
    print(f'  {i:>2} {r["sym"]:>7} ${r["price"]:>7.2f} {r["prob"]:>6.1%}')

# ===== 保存 =====
print('\n保存模型...')
model.save_model(f'{MD}/{VER}.json')
pickle.dump(cal,open(f'{MD}/{VER}_calibrator.pkl','wb'))
json.dump({
    'version':VER,'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),
    'val_auc':round(va_c,4),'test_auc':round(te_c,4),
    'features':FEATS,'n_features':len(FEATS),
    'n_train':len(tr),'n_val':len(va),'n_test':len(te),
    'pos_rate':round(tr.label.mean(),4),
    'best_iteration':int(model.best_iteration),
    'param_depth':8,'param_rounds':1000,
    'signal_sources':'yfinance institutional/insider/analyst/revisions/short',
    'calibration':'Isotonic',
},open(f'{MD}/{VER}_report.json','w'),indent=2)
print(f'  => {VER}.json + _calibrator.pkl + _report.json')
print(f'\n总耗时: {time.time()-T0:.0f}s')
print('='*60)
